"""Widget de Comprensión de vídeo: analiza un vídeo y entrega un informe.

El pipeline lo ejecuta :class:`UnderstandingWorker` (un ``QThread``) y consiste
en: medir la duración → muestrear fotogramas (uno cada N segundos, no cada N ms)
→ pasar cada fotograma por los motores de percepción disponibles (YOLO, CLIP,
OCR) → extraer el audio y transcribirlo con Whisper → resumir el conjunto con
Ollama → montar el informe.

El *plan* (muestreo, prompts y formato) vive en
:mod:`services.video_understanding`; aquí solo se ejecuta, igual que
``DocumentWorker`` ejecuta el pipeline de ``services.document_pipeline``.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from core.frame_extractor import FrameExtractor
from core.paths import PROJECT_ROOT
from engines.base_engine import CANCEL_MESSAGE, EngineError, Perception
from services import video_understanding as vu
from ui.task_widget import TaskWidget
from ui.workers import ThreadWorker, run_pool_task

logger = logging.getLogger(__name__)

#: Nombre del fichero que se propone al guardar el informe.
DEFAULT_REPORT_NAME = "informe_video.txt"

#: Segundos que un fotograma espera a que la GPU tenga hueco antes de renunciar.
#: Son trabajos de percepción previstos como cortos, pero si hay un lote en curso
#: conviene ceder en vez de competir por la VRAM.
PERCEPTION_WAIT_S = 60.0


class UnderstandingWorker(ThreadWorker):
    """Ejecuta el análisis de un vídeo fuera del hilo de la UI."""

    def __init__(
        self,
        video_path: str | Path,
        plan: vu.UnderstandingPlan,
        *,
        extractor: FrameExtractor,
        objects_engine=None,
        labels_engine=None,
        ocr_engine=None,
        transcription_engine=None,
        chat_engine=None,
        model: str | None = None,
        scheduler=None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._video = Path(video_path)
        self._plan = plan
        self._extractor = extractor
        self._objects_engine = objects_engine
        self._labels_engine = labels_engine
        self._ocr_engine = ocr_engine
        self._transcription_engine = transcription_engine
        self._chat_engine = chat_engine
        self._model = model
        self._scheduler = scheduler

    # ---- Pipeline --------------------------------------------------

    def work(self) -> str:
        plan = self._plan.normalized()
        duracion = self._extractor.duration(self._video)
        if duracion is None or duracion <= 0:
            raise EngineError(
                "No se pudo leer la duración del vídeo (¿está dañado o no es un vídeo?)."
            )

        instantes = vu.sample_timestamps(duracion, plan.every_s, plan.max_frames)
        logger.info(
            "Comprensión: %d fotogramas de '%s' (%s)",
            len(instantes),
            self._video.name,
            plan.describe(),
        )

        notices: list[str] = []
        frames = self._analyze_frames(instantes, plan, notices)
        transcript = self._transcribe(plan, notices)
        summary = self._summarize(plan, frames, transcript, notices)
        return vu.build_report(plan, frames, transcript, summary, tuple(notices))

    # ---- Fases -----------------------------------------------------

    def _perception_engines(self, plan: vu.UnderstandingPlan) -> list:
        """Motores de percepción pedidos y disponibles, en orden de coste."""
        candidatos = []
        if plan.do_objects:
            candidatos.append(self._objects_engine)
        if plan.do_labels:
            candidatos.append(self._labels_engine)
        if plan.do_ocr:
            candidatos.append(self._ocr_engine)
        return [
            motor
            for motor in candidatos
            if motor is not None and motor.is_available()
        ]

    def _analyze_frames(
        self, instantes: list[float], plan: vu.UnderstandingPlan, notices: list[str]
    ) -> list[vu.FrameReport]:
        motores = self._perception_engines(plan)
        if plan.wants_perception and not motores:
            notices.append(vu.NO_PERCEPTION_NOTICE)
            return []
        if not motores:
            return []

        frames: list[vu.FrameReport] = []
        total = len(instantes)
        with tempfile.TemporaryDirectory(prefix="imperator_video_") as tmp:
            for indice, instante in enumerate(instantes):
                self._check_cancel()
                destino = Path(tmp) / f"frame_{indice:04d}.png"
                self._extractor.extract_frame(self._video, instante, destino)
                percepcion = Perception()
                for motor in motores:
                    self._check_cancel()
                    percepcion = percepcion.merge(self._perceive(motor, destino))
                frames.append(
                    vu.FrameReport(timestamp=instante, summary=percepcion.summary())
                )
                self.progress.emit(
                    int((indice + 1) / max(1, total) * 70),
                    f"Fotograma {indice + 1}/{total}",
                )
        return frames

    def _perceive(self, motor, frame_path: Path) -> Perception:
        """Pasa un fotograma por un motor, respetando el planificador de GPU."""
        etiqueta = f"percepcion:{motor.id}"
        if self._scheduler is None:
            return motor.perceive(frame_path)
        # Los trabajos de percepción son cortos: se les da un margen para
        # esperar si la GPU está ocupada con un lote (vídeo, imagen).
        with self._scheduler.job(etiqueta, wait=PERCEPTION_WAIT_S) as permitido:
            if not permitido:
                logger.info(
                    "Comprensión: %s aplazado por GPU ocupada; fotograma sin anotar",
                    etiqueta,
                )
                return Perception()
            return motor.perceive(frame_path)

    def _transcribe(
        self, plan: vu.UnderstandingPlan, notices: list[str]
    ) -> str:
        if not plan.transcribe:
            return ""
        motor = self._transcription_engine
        if motor is None or not motor.is_available():
            notices.append(vu.NO_TRANSCRIPT_NOTICE)
            return ""
        with tempfile.TemporaryDirectory(prefix="imperator_audio_") as tmp:
            wav = Path(tmp) / "audio.wav"
            self.progress.emit(75, "Extrayendo el audio...")
            self._extractor.extract_audio(self._video, wav)
            self._check_cancel()
            self.progress.emit(78, "Transcribiendo el audio...")
            srt = motor.transcribe(str(wav), language=plan.language)
            try:
                crudo = Path(srt).read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                raise EngineError(f"No se pudo leer la transcripción: {exc}") from exc
        return vu.srt_to_text(crudo)

    def _summarize(
        self,
        plan: vu.UnderstandingPlan,
        frames: list[vu.FrameReport],
        transcript: str,
        notices: list[str],
    ) -> str:
        motor = self._chat_engine
        if motor is None or not motor.is_available():
            notices.append(vu.NO_SUMMARY_NOTICE)
            return ""

        anotaciones = (
            "\n".join(
                f"t={frame.label} {frame.summary.strip()}" if frame.summary.strip()
                else f"t={frame.label} (sin hallazgos)"
                for frame in frames
            )
            or "(sin anotaciones)"
        )
        prompt = vu.aggregate_prompt(plan, len(frames), bool(transcript.strip()))
        mensaje = {
            "role": "user",
            "content": (
                f"{prompt}\n"
                f"--- ANOTACIONES DE FOTOGRAMAS ---\n{anotaciones}\n\n"
                f"--- TRANSCRIPCIÓN DEL AUDIO ---\n{transcript or '(sin transcripción)'}"
            ),
        }
        self.progress.emit(85, "Resumiendo el vídeo con el modelo...")
        self._check_cancel()
        summary = motor.chat([mensaje], model=self._model)
        self.progress.emit(95, "Montando el informe...")
        return summary

    def _check_cancel(self) -> None:
        if self.isInterruptionRequested():
            raise EngineError(CANCEL_MESSAGE)


class UnderstandingWidget(TaskWidget):
    """Elige un vídeo, configura el análisis y muestra el informe."""

    def __init__(self, engine_manager, parent=None) -> None:
        super().__init__(parent)
        self._engine_manager = engine_manager
        self._video_path: Path | None = None
        self._last_report = ""
        self._model_task = None
        self._build_ui()
        self._populate_models()
        self._refresh_availability()

    # ---- Construcción de la UI -------------------------------------

    def _build_ui(self) -> None:
        self._video_label = QLabel("Ningún vídeo seleccionado.")
        self._choose_button = QPushButton("Elegir vídeo...")

        video_row = QHBoxLayout()
        video_row.addWidget(self._choose_button)
        video_row.addWidget(self._video_label, 1)

        self._every = QDoubleSpinBox()
        self._every.setRange(vu.MIN_EVERY_S, 120.0)
        self._every.setValue(vu.DEFAULT_EVERY_S)
        self._every.setSuffix(" s")
        self._every.setToolTip(
            "Se analiza un fotograma cada este tiempo (no cada milisegundo:\n"
            "un vídeo a 30 fps serían decenas de inferencias por segundo)."
        )

        self._max_frames = QSpinBox()
        self._max_frames.setRange(1, vu.MAX_MAX_FRAMES)
        self._max_frames.setValue(vu.DEFAULT_MAX_FRAMES)

        options_row = QHBoxLayout()
        options_row.addWidget(QLabel("Cada"))
        options_row.addWidget(self._every)
        options_row.addWidget(QLabel("Máx."))
        options_row.addWidget(self._max_frames)
        options_row.addWidget(QLabel("fotogramas"))

        self._objects = QCheckBox("Objetos (YOLO)")
        self._objects.setChecked(True)
        self._labels = QCheckBox("Etiquetas (CLIP)")
        self._ocr = QCheckBox("Texto en pantalla (OCR)")
        self._transcribe = QCheckBox("Transcribir audio (Whisper)")
        self._transcribe.setChecked(True)

        signals_row = QHBoxLayout()
        for widget in (self._objects, self._labels, self._ocr, self._transcribe):
            signals_row.addWidget(widget)
        signals_row.addStretch(1)

        self._model = QComboBox()
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Modelo:"))
        model_row.addWidget(self._model, 1)

        self._availability = QLabel("")
        self._availability.setWordWrap(True)

        self._analyze_button = QPushButton("Analizar")
        self._cancel_button = QPushButton("Cancelar")
        self._cancel_button.setEnabled(False)
        self._save_button = QPushButton("Guardar informe")
        self._save_button.setEnabled(False)

        buttons_row = QHBoxLayout()
        buttons_row.addWidget(self._analyze_button)
        buttons_row.addWidget(self._cancel_button)
        buttons_row.addWidget(self._save_button)
        buttons_row.addStretch(1)

        self._result = QPlainTextEdit()
        self._result.setReadOnly(True)
        self._result.setPlaceholderText("El informe del vídeo aparecerá aquí.")

        self._status = QLabel("Elige un vídeo y pulsa Analizar.")

        layout = QVBoxLayout(self)
        layout.addLayout(video_row)
        layout.addLayout(options_row)
        layout.addLayout(signals_row)
        layout.addLayout(model_row)
        layout.addWidget(self._availability)
        layout.addLayout(buttons_row)
        layout.addWidget(self._result, 1)
        layout.addWidget(self._status)

        self._choose_button.clicked.connect(self._choose_video)
        self._analyze_button.clicked.connect(self._analyze)
        self._cancel_button.clicked.connect(self._cancel)
        self._save_button.clicked.connect(self._save_report)

    # ---- Estado ----------------------------------------------------

    def _refresh_availability(self) -> None:
        """Explica qué motores hacen falta si el análisis va a ser parcial."""
        motores = self._resolve_engines()
        faltan = [
            nombre
            for nombre, motor in (
                ("YOLO", motores["objects"]),
                ("CLIP", motores["labels"]),
                ("OCR", motores["ocr"]),
                ("Whisper", motores["transcription"]),
                ("Ollama", motores["chat"]),
            )
            if motor is None
        ]
        if not faltan:
            self._availability.setText(
                "Motores disponibles: YOLO, CLIP, OCR, Whisper y Ollama."
            )
            return
        self._availability.setText(
            "Sin motor para: "
            + ", ".join(faltan)
            + ". Actívalos en Motores → Gestionar motores… (los que falten se "
            "omitirán avisando en el informe)."
        )

    def _resolve_engines(self) -> dict:
        manager = self._engine_manager
        return {
            "objects": manager.get_engine("yolo"),
            "labels": manager.get_engine("clip"),
            "ocr": manager.get_engine("paddle_ocr"),
            "transcription": manager.get_engine("whisper"),
            "chat": manager.get_engine("ollama"),
        }

    def set_engine_manager(self, engine_manager) -> None:
        """Relee los motores (tras cambiar la configuración de Motores)."""
        self._engine_manager = engine_manager
        self._populate_models()
        self._refresh_availability()

    # ---- Modelos ---------------------------------------------------

    def _populate_models(self) -> None:
        engine = self._engine_manager.get_engine("ollama")
        self._model.clear()
        if engine is None:
            self._model.setEnabled(False)
            return
        self._model.setEnabled(True)
        self._model.addItem("Cargando modelos…")
        self._model_task = run_pool_task(
            engine.list_models,
            on_result=self._on_models,
            on_error=self._on_models_error,
        )

    def _on_models_error(self, message: str) -> None:
        logger.debug("No se pudieron listar los modelos del chat: %s", message)
        self._on_models([])

    def _on_models(self, models: list) -> None:
        engine = self._engine_manager.get_engine("ollama")
        default = getattr(engine, "default_model", None)
        self._model.clear()
        for nombre in models or []:
            self._model.addItem(str(nombre), str(nombre))
        if default and self._model.findData(default) < 0:
            self._model.addItem(str(default), str(default))
        if self._model.count() == 0 and default:
            self._model.addItem(str(default), str(default))

    # ---- Acciones --------------------------------------------------

    def _choose_video(self) -> None:
        ruta, _ = QFileDialog.getOpenFileName(
            self,
            "Elegir vídeo",
            "",
            "Vídeos (*.mp4 *.mkv *.mov *.avi *.webm *.m4v);;Todos los archivos (*)",
        )
        if not ruta:
            return
        self._video_path = Path(ruta)
        self._video_label.setText(self._video_path.name)
        self._set_status("Vídeo listo para analizar.")

    def _build_plan(self) -> vu.UnderstandingPlan:
        return vu.UnderstandingPlan(
            every_s=self._every.value(),
            max_frames=self._max_frames.value(),
            do_objects=self._objects.isChecked(),
            do_labels=self._labels.isChecked(),
            do_ocr=self._ocr.isChecked(),
            transcribe=self._transcribe.isChecked(),
        ).normalized()

    def _validation_error(self, plan: vu.UnderstandingPlan) -> str | None:
        """Motivo por el que no se puede analizar (o ``None`` si todo está listo)."""
        if self._video_path is None:
            return "Elige primero un vídeo."
        if not plan.wants_perception and not plan.transcribe:
            return "Activa al menos una señal: objetos, etiquetas, OCR o transcripción."
        if plan.wants_perception:
            extractor_ok = bool(self._perception_engines_available(plan))
            if not extractor_ok:
                return (
                    "No hay ningún motor de percepción activo "
                    "(YOLO/CLIP/OCR). Actívalos en Motores o desmarca esas señales."
                )
        return None

    def _perception_engines_available(self, plan: vu.UnderstandingPlan) -> list:
        motores = self._resolve_engines()
        candidatos = []
        if plan.do_objects:
            candidatos.append(motores["objects"])
        if plan.do_labels:
            candidatos.append(motores["labels"])
        if plan.do_ocr:
            candidatos.append(motores["ocr"])
        return [m for m in candidatos if m is not None and m.is_available()]

    def _analyze(self) -> None:
        plan = self._build_plan()
        error = self._validation_error(plan)
        if error is not None:
            self._set_status(error)
            return

        motores = self._resolve_engines()
        self._set_status("Analizando el vídeo...")
        self._result.clear()
        self._last_report = ""
        self._set_busy(True)

        self._worker = UnderstandingWorker(
            self._video_path,
            plan,
            extractor=FrameExtractor(),
            objects_engine=motores["objects"],
            labels_engine=motores["labels"],
            ocr_engine=motores["ocr"],
            transcription_engine=motores["transcription"],
            chat_engine=motores["chat"],
            model=self._model.currentData(),
            scheduler=self._scheduler_hint(),
            parent=self,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.result.connect(self._on_result)
        self._worker.failed.connect(self._on_error)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.start()

    def _scheduler_hint(self):
        """Planificador de GPU, si la ventana principal lo expone (opcional)."""
        ventana = self.window()
        return getattr(ventana, "gpu_scheduler", None)

    def _cancel(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.requestInterruption()
            self._set_status("Cancelando...")

    def _on_progress(self, value: int, message: str) -> None:
        self._set_status(f"{message} ({value}%)")

    def _on_result(self, report: str) -> None:
        self._last_report = report or ""
        self._result.setPlainText(self._last_report)
        self._set_status("Informe listo.")
        self._on_finished()

    def _on_cancelled(self, message: str) -> None:
        self._set_status(message)
        self._on_finished()

    def _on_failed(self, message: str) -> None:
        self._set_status(f"Error: {message}")
        self._on_finished()

    def _on_finished(self) -> None:
        self._set_busy(False)
        self._worker = None
        self._save_button.setEnabled(bool(self._last_report))
        self._restore_focus()

    def _save_report(self) -> None:
        if not self._last_report:
            return
        destino, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar informe",
            str(PROJECT_ROOT / "output" / DEFAULT_REPORT_NAME),
            "Texto (*.txt)",
        )
        if not destino:
            return
        try:
            Path(destino).write_text(self._last_report, encoding="utf-8")
        except OSError as exc:
            self._set_status(f"No se pudo guardar el informe: {exc}")
            return
        self._set_status(f"Informe guardado en {Path(destino).name}.")

    def _busy_widgets(self) -> tuple:
        return (
            self._choose_button,
            self._analyze_button,
            self._save_button,
            self._every,
            self._max_frames,
            self._objects,
            self._labels,
            self._ocr,
            self._transcribe,
            self._model,
        )

    def _busy_only_widgets(self) -> tuple:
        return (self._cancel_button,)

    def _on_busy_changed(self, busy: bool) -> None:
        # Guardar solo depende del estado ocupado y de que haya informe.
        if not busy:
            self._save_button.setEnabled(bool(self._last_report))

    def _focus_widget(self):
        """No hay campo de escritura: nada que enfocar al terminar."""
        return None
