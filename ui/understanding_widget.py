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
from datetime import datetime
from pathlib import Path

from PyQt6 import sip
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from core.frame_cache import FrameCache
from core.frame_extractor import FrameExtractor
from core.paths import OUTPUT, PROJECT_ROOT
from engines.base_engine import EngineError, Perception
from services import chat_engine as ce
from services import preferences as prefs
from services import assembler_params as asp
from services import scoreboard as sb
from services import sport_params as sp
from services import video_understanding as vu
from ui import discipline
from ui.task_widget import TaskWidget
from ui.pool_callbacks import PoolCallbackReceiver
from ui.workers import ThreadWorker, run_pool_task

logger = logging.getLogger(__name__)

#: Nombre del fichero que se propone al guardar el informe.
DEFAULT_REPORT_NAME = "informe_video.txt"

#: Carpeta donde Comprensión deja su percepción estructurada (reutilizable por
#: la pestaña Cuaderno, igual que el Cuaderno deja su informe para la Crónica).
UNDERSTANDING_DIR = PROJECT_ROOT / "projects" / "comprension"

#: Segundos que un fotograma espera a que la GPU tenga hueco antes de renunciar.
#: Son trabajos de percepción previstos como cortos, pero si hay un lote en curso
#: conviene ceder en vez de competir por la VRAM.
PERCEPTION_WAIT_S = 60.0

#: Ejemplo que se muestra en la caja de contexto (una línea por dato).
CONTEXT_PLACEHOLDER = (
    "Fecha: 9/10/2026 a las 20:30\n"
    "Lugar: Polideportivo de Muskiz\n"
    "Equipos: PAN MENESA MUSKIZ camiseta azul (local) vs BAIGENE CORAZONISTAS "
    "camisetas rojas (visitante).\n"
    "Alineación PAN MENESA MUSKIZ: 2 ARANE PEREZ HERNANDEZ, 5 IRIA MORAGA GARCIA, "
    "6 JUGATX FERNANDEZ GONZALEZ...\n"
    "Alineación BAIGENE CORAZONISTAS: 3 AMANDA OTERO BENITO, 8 PAULA GARCIA "
    "VALENCIA, 9 AITANA ESTEBAN GARCIA...\n"
    "Marcador final: 42-29\n"
    "Nota: faltas del segundo tiempo del minuto 30 al 40, por error de cámara."
)


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
        clip_labels: str = "",
        context: str = "",
        instructions: str = "",
        out_dir: Path | None = None,
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
        #: Vocabulario que puntúa CLIP (lista separada por comas). CLIP no describe:
        #: elige entre estas etiquetas, así que con genéricas da ruido.
        self._clip_labels = str(clip_labels or "")
        #: Datos del partido (fecha, lugar, equipos, alineación…). Van al resumen y
        #: al comprension.json para que el Cuaderno y la Crónica los reutilicen.
        self._context = str(context or "")
        #: Instrucciones del usuario (qué destacar): el resumen las prioriza.
        self._instructions = str(instructions or "")
        #: Carpeta donde dejar la percepción estructurada (si se indica).
        self._out_dir = Path(out_dir) if out_dir is not None else None
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
        marcador_lecturas = self._read_scoreboard(duracion)
        transcript = self._transcribe(plan, notices)
        summary = self._summarize(plan, frames, transcript, notices, marcador_lecturas)
        self._save_structured(
            plan, frames, transcript, summary, duracion, notices, marcador_lecturas
        )
        return vu.build_report(
            plan, frames, transcript, summary, tuple(notices), context=self._context
        )

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
        for indice, instante in enumerate(instantes):
            self._check_cancel()
            # Caché: el mismo fotograma (vídeo+instante) no se re-extrae.
            destino = self._extractor.cached_frame(self._video, instante)
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
        # Solo CLIP usa ``prompt``: es su vocabulario de etiquetas (lista separada
        # por comas). YOLO y el OCR lo ignoran.
        prompt = self._clip_labels if getattr(motor, "id", "") == "clip" else ""
        if self._scheduler is None:
            return motor.perceive(frame_path, prompt)
        # Los trabajos de percepción son cortos: se les da un margen para
        # esperar si la GPU está ocupada con un lote (vídeo, imagen).
        with self._scheduler.job(etiqueta, wait=PERCEPTION_WAIT_S) as permitido:
            if not permitido:
                logger.info(
                    "Comprensión: %s aplazado por GPU ocupada; fotograma sin anotar",
                    etiqueta,
                )
                return Perception()
            return motor.perceive(frame_path, prompt)

    def _read_scoreboard(self, duracion: float) -> list[tuple[float, str]]:
        """Lee el marcador por OCR con muestreo denso (cada ~2 s), separado de la visión.

        Devuelve ``[]`` si no hay OCR disponible. El OCR se hace sobre un recorte de
        la franja superior, donde suele estar el marcador: es más rápido y evita
        confundir el tanteo con otros textos de la escena.
        """
        ocr = self._ocr_engine
        if ocr is None or not ocr.is_available():
            return []
        instantes = vu.scoreboard_timestamps(duracion)
        if not instantes:
            return []
        lecturas: list[tuple[float, str]] = []
        total = len(instantes)
        with tempfile.TemporaryDirectory(prefix="imperator_marcador_") as tmp:
            for indice, t in enumerate(instantes):
                self._check_cancel()
                png = Path(tmp) / f"marcador_{indice:05d}.png"
                try:
                    self._extractor.extract_cropped_frame(
                        self._video, t, png, crop=vu.SCOREBOARD_CROP
                    )
                    percepcion = ocr.perceive(png, "")
                    texto = (percepcion.text or "").strip()
                except EngineError as exc:
                    logger.debug("Comprensión: marcador t=%gs falló: %s", t, exc)
                    continue
                if texto:
                    lecturas.append((t, texto))
                self.progress.emit(
                    70 + int((indice + 1) / max(1, total) * 5),
                    f"Leyendo marcador {indice + 1}/{total}…",
                )
        return lecturas

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
        marcador_lecturas: list[tuple[float, str]] | None = None,
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
        # El marcador real (derivado del texto del OCR) se le pasa al resumen como
        # dato duro: sin él, el modelo «ve» un gol en cada anotación. Se prefiere
        # la lectura densa del marcador (cada ~2 s); si no hay, la de los fotogramas.
        lecturas = marcador_lecturas or [
            (frame.timestamp, frame.summary) for frame in frames
        ]
        marcador = sb.marcador_block(lecturas)
        prompt = vu.aggregate_prompt(
            plan,
            len(frames),
            bool(transcript.strip()),
            self._context,
            marcador,
            self._instructions,
        )
        # El prompt del resumen se acota: la transcripción de un partido largo (y
        # las anotaciones) llenaban la ventana y Ollama respondía «se agotó el
        # espacio de la respuesta». Se recorta por palabra y se avisa con » […]».
        mensaje = {
            "role": "user",
            "content": (
                f"{prompt}\n"
                "--- ANOTACIONES DE FOTOGRAMAS ---\n"
                f"{vu.clamp_text(anotaciones, vu.MAX_ANNOTATIONS_CHARS)}\n\n"
                "--- TRANSCRIPCIÓN DEL AUDIO ---\n"
                f"{vu.clamp_text(transcript, vu.MAX_TRANSCRIPT_CHARS) or '(sin transcripción)'}"
            ),
        }
        self.progress.emit(85, "Resumiendo el vídeo con el modelo...")
        self._check_cancel()
        summary = motor.chat([mensaje], model=self._model)
        self.progress.emit(95, "Montando el informe...")
        return summary

    def _save_structured(
        self,
        plan: vu.UnderstandingPlan,
        frames: list[vu.FrameReport],
        transcript: str,
        summary: str,
        duracion: float,
        notices: list[str],
        marcador_lecturas: list[tuple[float, str]] | None = None,
    ) -> None:
        """Deja la percepción estructurada para que el Cuaderno la reutilice."""
        if self._out_dir is None:
            return
        lecturas = marcador_lecturas or []
        serie = sb.score_series(lecturas)
        try:
            vu.save_understanding(
                self._out_dir / vu.UNDERSTANDING_FILENAME,
                vu.understanding_to_dict(
                    video_path=str(self._video),
                    video_name=self._video.name,
                    duration_s=duracion,
                    plan=plan,
                    frames=frames,
                    transcript=transcript,
                    summary=summary,
                    notices=tuple(notices),
                    context=self._context,
                    marcador=[
                        {"t_s": p.t_s, "izquierda": p.izquierda, "derecha": p.derecha}
                        for p in serie
                    ],
                    goles=[
                        {"t_s": g.t_s, "lado": g.lado, "marcador": g.marcador}
                        for g in sb.goals_from_series(serie)
                    ],
                    juego_inicio=sb.game_start(lecturas),
                    descanso=sb.half_time(lecturas),
                ),
            )
        except OSError as exc:
            logger.warning("Comprensión: no se pudo guardar la percepción: %s", exc)


class UnderstandingWidget(TaskWidget):
    """Elige un vídeo, configura el análisis y muestra el informe."""

    def __init__(self, engine_manager, parent=None) -> None:
        super().__init__(parent)
        self._engine_manager = engine_manager
        self._prefs = prefs.PrefsStore(getattr(engine_manager, "config_manager", None))
        self._video_path: Path | None = None
        self._last_report = ""
        self._model_task = None
        #: Token que descarta respuestas del pool de un motor anterior.
        self._generation = 0
        #: Receptor del pool atado a la vida del widget (C2).
        self._models_rx = PoolCallbackReceiver(self, self._on_models, self._on_models_error)
        #: Carpeta de la sesión: donde se deja la percepción estructurada.
        self._session_dir: Path | None = None
        self._build_ui()
        self._restore_instructions()
        self._populate_models()
        self._refresh_availability()
        # La disciplina es compartida: se restaura y se sincroniza en vivo.
        self._restore_discipline()
        discipline.BUS.changed.connect(self._on_sport_changed)

    def _on_sport_picked(self, _index: int) -> None:
        """Guarda la disciplina elegida y la propaga a las demás pestañas."""
        clave = self._cmb_sport.currentData()
        if not clave:
            return
        self._prefs.set(prefs.SPORT, clave)
        discipline.BUS.changed.emit(str(clave))

    def _on_sport_changed(self, clave: str) -> None:
        """Aplica la disciplina elegida en otra pestaña (sin reemitir)."""
        discipline.apply(self._cmb_sport, str(clave))

    def _restore_discipline(self) -> None:
        discipline.apply(self._cmb_sport, discipline.saved(self._prefs))

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

        self._cmb_language = QComboBox()
        for code, label in asp.LANGUAGES:
            self._cmb_language.addItem(label, code)
        self._cmb_language.setToolTip(
            "Idioma del audio y del resumen. «Auto» lo detecta el motor de\n"
            "transcripción (Whisper) y el resumen sale en ese idioma; con una\n"
            "grada ruidosa, Whisper puede confundirlo con inglés: fija «Español»."
        )
        language_row = QHBoxLayout()
        language_row.addWidget(QLabel("Idioma:"))
        language_row.addWidget(self._cmb_language)
        language_row.addStretch(1)

        self._cmb_sport = QComboBox()
        discipline.fill(self._cmb_sport)
        self._cmb_sport.setToolTip(
            "Disciplina del vídeo: sus reglas y vocabulario se le pasan al resumen\n"
            "(sin esto confunde deportes: p. ej. llama «voleibol» a un balonmano).\n"
            "Es la MISMA en Cuaderno, Crónica y Comprensión: cambiarla aquí la\n"
            "cambia en las demás."
        )
        self._cmb_sport.currentIndexChanged.connect(self._on_sport_picked)
        sport_row = QHBoxLayout()
        sport_row.addWidget(QLabel("Disciplina:"))
        sport_row.addWidget(self._cmb_sport)
        sport_row.addStretch(1)

        # Por defecto, TODAS las señales activadas: es la opción de más calidad
        # (cada motor que falte se omite avisando en el informe).
        self._objects = QCheckBox("Objetos (YOLO)")
        self._objects.setChecked(True)
        self._labels = QCheckBox("Etiquetas (CLIP)")
        self._labels.setChecked(True)
        self._ocr = QCheckBox("Texto en pantalla (OCR)")
        self._ocr.setChecked(True)
        self._transcribe = QCheckBox("Transcribir audio (Whisper)")
        self._transcribe.setChecked(True)

        signals_row = QHBoxLayout()
        for widget in (self._objects, self._labels, self._ocr, self._transcribe):
            signals_row.addWidget(widget)
        signals_row.addStretch(1)

        self._clip_labels = QLineEdit()
        self._clip_labels.setText(vu.clip_labels_text())
        self._clip_labels.setToolTip(
            "Vocabulario que puntúa CLIP (lista separada por comas). CLIP no describe:\n"
            "elige entre estas etiquetas. Lo recomendado: etiquetas CONCRETAS y\n"
            "separables del contenido (deporte, cine...), sin nombres ni colores.\n"
            "Genéricas («calle, coche, animal») o dos casi iguales dan ruido."
        )
        clip_row = QHBoxLayout()
        clip_row.addWidget(QLabel("Etiquetas CLIP:"))
        clip_row.addWidget(self._clip_labels, 1)
        layout_clip = clip_row

        self._context = QPlainTextEdit()
        self._context.setPlaceholderText(CONTEXT_PLACEHOLDER)
        self._context.setFixedHeight(72)
        self._context.setToolTip(
            "Datos del partido (fecha, lugar, equipos, alineación, marcador, notas),\n"
            "una línea por dato. Se toman como ciertos: el resumen nombra con ellos y\n"
            "quedan guardados en comprension.json, así el Cuaderno y la Crónica los\n"
            "reutilizan sin volver a escribirlos."
        )
        context_layout = QVBoxLayout()
        context_layout.addWidget(QLabel("Contexto (datos del partido, opcional):"))
        context_layout.addWidget(self._context)

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
        layout.addLayout(sport_row)
        layout.addLayout(language_row)
        layout.addLayout(context_layout)
        # Instrucciones para la IA: qué quiere el usuario que destaque el resumen.
        instr_row = QHBoxLayout()
        instr_row.addWidget(QLabel("Instrucciones para la IA (qué destacar):"))
        instr_row.addStretch(1)
        self._btn_ejemplo_ia = QPushButton("Poner ejemplo (balonmano)")
        self._btn_ejemplo_ia.setToolTip(
            "Rellena el ejemplo recomendado: presentar la alineación al principio\n"
            "(mientras calientan) y después centrarse en goles, paradas del portero\n"
            "y lanzamientos de 7 metros. Edítalo a tu gusto."
        )
        self._btn_ejemplo_ia.clicked.connect(
            lambda: self._txt_instructions.setPlainText(sp.INSTRUCTIONS_EXAMPLE)
        )
        instr_row.addWidget(self._btn_ejemplo_ia)
        self._txt_instructions = QPlainTextEdit()
        self._txt_instructions.setFixedHeight(72)
        self._txt_instructions.setPlaceholderText(sp.INSTRUCTIONS_EXAMPLE)
        self._txt_instructions.setToolTip(
            "Escribe qué quieres que destaque el resumen (p. ej. goles, paradas y\n"
            "7 metros). El Cuaderno y la Crónica heredan este material."
        )
        layout.addLayout(instr_row)
        layout.addWidget(self._txt_instructions)
        layout.addLayout(signals_row)
        layout.addLayout(layout_clip)
        layout.addLayout(model_row)
        layout.addWidget(self._availability)
        # Esta pestaña «echa una mano» a la Crónica: su comprension.json lo reutilizan
        # el Cuaderno y la Crónica sin volver a mirar el vídeo.
        self._help_cronica = QLabel(
            "Lo que analices aquí queda en comprension.json: el Cuaderno lo reutiliza "
            "(sin repetir YOLO/CLIP/OCR/Whisper) y la Crónica lo usa para escribir la "
            "crónica sin volver a analizar el vídeo."
        )
        self._help_cronica.setWordWrap(True)
        layout.addWidget(self._help_cronica)
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
            "chat": ce.preferred_chat_engine(manager),
        }

    def set_engine_manager(self, engine_manager) -> None:
        """Relee los motores (tras cambiar la configuración de Motores)."""
        self._engine_manager = engine_manager
        self._populate_models()
        self._refresh_availability()

    # ---- Modelos ---------------------------------------------------

    def _populate_models(self) -> None:
        engine = ce.preferred_chat_engine(self._engine_manager)
        self._model.clear()
        if engine is None:
            self._model.setEnabled(False)
            return
        self._model.setEnabled(True)
        self._model.addItem("Cargando modelos…")
        self._generation += 1
        self._models_rx.arm()
        self._model_task = run_pool_task(
            engine.list_models,
            on_result=self._models_rx.handle,
            on_error=self._models_rx.handle_error,
        )

    def _on_models_error(self, token: int, message: str) -> None:
        if sip.isdeleted(self) or token != self._generation:
            return
        logger.debug("No se pudieron listar los modelos del chat: %s", message)
        self._on_models(token, [])

    def _on_models(self, token: int, models: list) -> None:
        if sip.isdeleted(self) or token != self._generation:
            return
        engine = ce.preferred_chat_engine(self._engine_manager)
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
            self, "Elegir vídeo", "", vu.VIDEO_FILTER_ALL
        )
        if not ruta:
            return
        self._video_path = Path(ruta)
        self._video_label.setText(self._video_path.name)
        # Vídeo nuevo -> carpeta de sesión nueva (no mezclar percepciones).
        self._session_dir = None
        self._set_status("Vídeo listo para analizar.")

    def _out_dir(self) -> Path:
        """Carpeta donde se deja la percepción estructurada (una por vídeo)."""
        destino = self._session_dir
        if destino is None:
            marca = datetime.now().strftime("%Y%m%d_%H%M%S")
            stem = self._video_path.stem if self._video_path else "comprension"
            destino = UNDERSTANDING_DIR / f"{stem}_{marca}"
            self._session_dir = destino
        destino.mkdir(parents=True, exist_ok=True)
        return destino

    def _restore_instructions(self) -> None:
        """Recupera las instrucciones guardadas (o pone el ejemplo de balonmano)."""
        guardado = self._prefs.get_str(prefs.UNDERSTANDING_INSTRUCTIONS)
        self._txt_instructions.setPlainText(
            guardado if (guardado or "").strip() else sp.INSTRUCTIONS_EXAMPLE
        )

    def _audio_language(self) -> str | None:
        """Idioma del audio/resumen elegido (``None`` = automático)."""
        code = str(self._cmb_language.currentData() or "auto")
        return None if code == "auto" else code

    def _build_plan(self) -> vu.UnderstandingPlan:
        return vu.UnderstandingPlan(
            every_s=self._every.value(),
            max_frames=self._max_frames.value(),
            do_objects=self._objects.isChecked(),
            do_labels=self._labels.isChecked(),
            do_ocr=self._ocr.isChecked(),
            transcribe=self._transcribe.isChecked(),
            sport=self._cmb_sport.currentData(),
            language=self._audio_language(),
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
        instructions = self._txt_instructions.toPlainText()
        self._prefs.set(prefs.UNDERSTANDING_INSTRUCTIONS, instructions)
        self._set_status("Analizando el vídeo...")
        self._result.clear()
        self._last_report = ""
        self._set_busy(True)

        self._worker = UnderstandingWorker(
            self._video_path,
            plan,
            extractor=FrameExtractor(cache=FrameCache()),
            objects_engine=motores["objects"],
            labels_engine=motores["labels"],
            ocr_engine=motores["ocr"],
            transcription_engine=motores["transcription"],
            chat_engine=motores["chat"],
            model=self._model.currentData(),
            clip_labels=self._clip_labels.text(),
            context=self._context.toPlainText(),
            instructions=instructions,
            out_dir=self._out_dir(),
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
        self._set_status(
            "Informe listo. Percepción guardada para el Cuaderno "
            f"({vu.UNDERSTANDING_FILENAME})."
        )
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
            str(OUTPUT / DEFAULT_REPORT_NAME),
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
            self._clip_labels,
            self._cmb_sport,
            self._cmb_language,
            self._model,
            self._context,
            self._txt_instructions,
            self._btn_ejemplo_ia,
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
