"""Widget de generación de vídeo: prompt, parámetros, progreso y salida MP4."""

import logging
import math
import re
import shutil
import subprocess
import time
from dataclasses import replace
from pathlib import Path

try:
    import imageio_ffmpeg
except ImportError:  # pragma: no cover - dependencia declarada en requirements.txt
    imageio_ffmpeg = None

from PyQt6.QtCore import QSize, Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices, QIcon, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.frame_extractor import FrameExtractor
from core.video_assembler import VideoAssembler
from engines.base_engine import CANCEL_MESSAGE, Engine, EngineError
from services import preferences as prefs
from services import system_stats as stats
from services import video_params as vp
from services import video_project as vproject
from services import video_script as vs
from ui.task_widget import TaskWidget
from ui.workers import ThreadWorker, gpu_slot, run_pool_task

logger = logging.getLogger(__name__)

# Timeout del sondeo de duración del audio (ffmpeg -i), para no colgar el hilo.
_AUDIO_PROBE_TIMEOUT = 30


class VideoWorker(ThreadWorker):
    """Ejecuta ``engine.generate_video()`` fuera del hilo de la UI."""

    def __init__(
        self,
        engine: Engine,
        params: dict,
        parent=None,
        scheduler=None,
        liberar_otros=None,
    ) -> None:
        super().__init__(parent)
        self._engine = engine
        self._params = params
        self._scheduler = scheduler
        self._liberar_otros = liberar_otros

    def work(self) -> str:
        params = dict(self._params)
        params["cancel_check"] = self.isInterruptionRequested
        images = params.pop("image_paths", None)
        prompts = params.pop("prompts", None)
        # Un clip (o una secuencia entera) = una reserva de GPU.
        with gpu_slot(self._scheduler, "video", liberar_otros=self._liberar_otros):
            if images and len(images) > 1:
                # Varias imágenes: un clip por imagen y concatenación final.
                return self._generate_sequence(images, prompts, params)
            params["progress_callback"] = self._on_progress
            # Duck-typing: el motor debe exponer generate_video() (VideoEngine).
            return str(self._engine.generate_video(**params))

    def _generate_sequence(
        self, images: list[str], prompts: list[str] | None, params: dict
    ) -> str:
        """Genera un clip por imagen (con su prompt) y los concatena en un MP4."""
        total = len(images)
        clips: list[str] = []
        for i, image in enumerate(images):
            if self.isInterruptionRequested():
                raise EngineError(CANCEL_MESSAGE)
            clip_params = dict(params)
            clip_params["image_path"] = image
            clip_params["prompt"] = self._prompt_for(i, prompts)
            clip_params["progress_callback"] = (
                lambda pct, msg, i=i: self._on_clip_progress(i, total, pct, msg)
            )
            clips.append(str(self._engine.generate_video(**clip_params)))
        return str(VideoAssembler().concat_clips(clips))

    @staticmethod
    def _prompt_for(index: int, prompts: list[str] | None) -> str:
        """Prompt para la imagen ``index``; reutiliza el último si faltan líneas."""
        if not prompts:
            return ""
        return prompts[min(index, len(prompts) - 1)]

    def _on_progress(self, percent: int, message: str) -> None:
        # Emitir desde un hilo distinto es seguro en PyQt (conexión encolada).
        self.progress.emit(percent, message)

    def _on_clip_progress(
        self, clip_index: int, total: int, percent: int, message: str
    ) -> None:
        """Mapea el progreso de un clip al progreso global de la secuencia."""
        overall = int(((clip_index + percent / 100.0) / total) * 100)
        self.progress.emit(overall, f"Clip {clip_index + 1}/{total}: {message}")


class SequenceWorker(ThreadWorker):
    """Genera todas las escenas de un guion con continuidad y las concatena.

    Continuidad = misma semilla base (``video_params.scene_seed``) + prompt con
    estilo y personajes repetidos + **encadenado i2v**: el último fotograma de una
    escena entra como imagen de la siguiente (``FrameExtractor``). Todo el trabajo
    pesado va bajo una sola reserva de GPU (``gpu_slot``).
    """

    def __init__(
        self,
        engine: Engine,
        script: "vs.Script",
        project: "vproject.Project",
        *,
        base_seed: int,
        negative_prompt: str = "",
        fps: int = 16,
        chain: bool = True,
        transitions: bool = False,
        parent=None,
        scheduler=None,
        liberar_otros=None,
    ) -> None:
        super().__init__(parent)
        self._engine = engine
        self._script = script
        self._project = project
        self._base_seed = base_seed
        self._negative_prompt = negative_prompt
        self._fps = fps
        self._chain = chain
        self._transitions = transitions
        self._scheduler = scheduler
        self._liberar_otros = liberar_otros

    def work(self) -> dict:
        total = len(self._script.scenes)
        if total == 0:
            raise EngineError("El guion no tiene escenas que generar.")
        extractor = FrameExtractor(self._project.directory)
        clips: list[str] = []
        previous_frame: str | None = None
        with gpu_slot(self._scheduler, "video", liberar_otros=self._liberar_otros):
            for index in range(total):
                if self.isInterruptionRequested():
                    raise EngineError(CANCEL_MESSAGE)
                mode = "i2v" if (self._chain and previous_frame) else "t2v"
                params = vp.scene_generate_params(
                    self._script,
                    index,
                    base_seed=self._base_seed,
                    fps=self._fps,
                    mode=mode,
                    negative_prompt=self._negative_prompt,
                    previous_frame=previous_frame,
                )
                params["cancel_check"] = self.isInterruptionRequested
                params["progress_callback"] = (
                    lambda percent, message, i=index: self._on_scene_progress(
                        i, total, percent, message
                    )
                )
                clip = Path(str(self._engine.generate_video(**params)))
                destination = self._project.clip_path(index)
                destination.parent.mkdir(parents=True, exist_ok=True)
                if clip.resolve() != destination.resolve():
                    shutil.copyfile(clip, destination)
                clips.append(str(destination))
                if self._chain:
                    previous_frame = self._last_frame(extractor, destination, index)

        sequence = None
        if len(clips) > 1:
            builder = VideoAssembler(output_dir=self._project.directory)
            sequence = self._concat(builder, clips)
        elif clips:
            sequence = clips[0]
        return {"clips": clips, "sequence": sequence}

    def _concat(self, builder: VideoAssembler, clips: list[str]) -> str:
        """Concatena los clips; con transiciones activadas usa ``xfade``."""
        if self._transitions:
            try:
                return str(
                    builder.concat_clips_with_transitions(
                        clips,
                        transitions=vs.transition_sequence(self._script),
                        output_name="secuencia.mp4",
                    )
                )
            except EngineError as exc:
                # Las transiciones recodifican y son más frágiles; si fallan, se
                # deja la secuencia unida (corte seco) en vez de no entregar nada.
                logger.warning("Transiciones no aplicadas (%s); corte seco.", exc)
        return str(builder.concat_clips(clips, output_name="secuencia.mp4"))

    def _last_frame(
        self, extractor: FrameExtractor, clip: Path, index: int
    ) -> str | None:
        """Último fotograma del clip (para encadenar); ``None`` si falla."""
        try:
            duration = extractor.duration(clip) or 0.0
            output = self._project.directory / f"ultimo_{index:02d}.png"
            extractor.extract_frame(clip, max(0.0, duration - 0.1), output)
            return str(output)
        except EngineError as exc:
            logger.debug("Sin encadenado en la escena %d: %s", index + 1, exc)
            return None

    def _on_scene_progress(
        self, index: int, total: int, percent: int, message: str
    ) -> None:
        overall = int(((index + percent / 100.0) / total) * 100)
        self.progress.emit(overall, f"Escena {index + 1}/{total}: {message}")


class ScriptWorker(ThreadWorker):
    """Pide a Ollama un guion (JSON) y lo convierte en un ``Script`` validado.

    Es la forma correcta de hablar con el motor para un vídeo: se le da el rol de
    guionista y un esquema (``services.video_script``), no una conversación
    abierta. La llamada HTTP va fuera del hilo de la UI y bajo ``gpu_slot('chat')``.
    """

    def __init__(
        self,
        engine: Engine,
        idea: str,
        *,
        scenes: int = vs.DEFAULT_SCENES,
        language: str = vs.DEFAULT_LANGUAGE,
        aspect: str = vs.ASPECT_HORIZONTAL,
        style: str = "",
        model: str | None = None,
        parent=None,
        scheduler=None,
        liberar_otros=None,
    ) -> None:
        super().__init__(parent)
        self._engine = engine
        self._idea = idea
        self._scenes = scenes
        self._language = language
        self._aspect = aspect
        self._style = style
        self._model = model
        self._scheduler = scheduler
        self._liberar_otros = liberar_otros

    def work(self) -> "vs.Script":
        messages = vs.script_generation_messages(
            self._idea,
            scenes=self._scenes,
            language=self._language,
            aspect=self._aspect,
            style=self._style,
        )
        with gpu_slot(self._scheduler, "chat", liberar_otros=self._liberar_otros):
            reply = self._engine.chat(messages, model=self._model)
        script, error = vs.try_parse_script(reply)
        if script is None:
            raise EngineError(f"No se pudo interpretar el guion del modelo: {error}")
        return script


class VideoWidget(TaskWidget):
    """Formulario de generación de vídeo con barra de progreso y salida MP4."""

    _PREVIEW_SIZE = 256

    #: Emite la ruta del vídeo de secuencia cuando el usuario lo envía al
    #: Ensamblador (la ventana principal lo precarga en esa pestaña).
    clips_ready = pyqtSignal(object)

    def __init__(
        self,
        engine: Engine | None,
        config_manager=None,
        parent=None,
        chat_engine: Engine | None = None,
    ) -> None:
        super().__init__(parent)
        self._engine: Engine | None = engine
        #: Motor de chat (Ollama) para generar el guion; puede ser ``None``.
        self._chat_engine: Engine | None = chat_engine
        self._config_manager = config_manager
        self._prefs = prefs.PrefsStore(config_manager)
        self._worker: VideoWorker | None = None
        self._last_path: str | None = None
        self._photos: list[dict] = []
        self._prompt_edits: list[QLineEdit] = []
        # Guion de escenas (lo carga el chat/ventana); su proyecto y clips.
        self._script: vs.Script | None = None
        self._project: vproject.Project | None = None
        self._clips: list[str] = []
        self._sequence_path: str | None = None
        self._script_worker = None
        self._reordering = False
        self._stats_task = None
        self._ram_free_bytes: object = None
        self._ram_free_at = 0.0
        self._build_ui()
        self._populate_modes()
        self._populate_script_models()
        self._restore_timeout()
        self._refresh_system_stats()
        if self._engine is None:
            self._set_status("Wan Video no está configurado o deshabilitado.")

    def _build_ui(self) -> None:
        self._prompt = QPlainTextEdit()
        self._prompt.setPlaceholderText("Describe el vídeo...")

        self._negative_prompt = QLineEdit()
        self._negative_prompt.setPlaceholderText("(opcional)")

        self._mode = QComboBox()
        self._mode.currentIndexChanged.connect(self._on_mode_changed)

        self._width = QSpinBox()
        self._width.setRange(256, 2048)
        self._width.setValue(832)

        self._height = QSpinBox()
        self._height.setRange(256, 2048)
        self._height.setValue(480)

        self._duration = QSpinBox()
        self._duration.setRange(1, 30)
        self._duration.setValue(4)
        self._duration.setSuffix(" s")

        self._adjust_audio_button = QPushButton("Ajustar al audio...")
        self._adjust_audio_button.setToolTip(
            "Lee la duración de un archivo de audio y ajusta la duración del vídeo."
        )
        self._adjust_audio_button.clicked.connect(self._adjust_to_audio)

        self._fps = QSpinBox()
        self._fps.setRange(1, 60)
        self._fps.setValue(16)

        self._timeout = QSpinBox()
        self._timeout.setRange(300, 3600)
        self._timeout.setValue(1800)
        self._timeout.setSuffix(" s")
        self._timeout.valueChanged.connect(self._on_timeout_changed)

        self._seed = QLineEdit()
        self._seed.setPlaceholderText("12345")
        self._random_seed = QCheckBox("Aleatorio")
        self._random_seed.setChecked(True)
        self._random_seed.toggled.connect(
            lambda checked: self._seed.setEnabled(not checked)
        )
        self._seed.setEnabled(False)

        self._preset_buttons: dict[str, QPushButton] = {}
        for key, (label, _w, _h, _dur, tooltip) in vp.PRESETS.items():
            btn = QPushButton(label)
            btn.setToolTip(tooltip)
            btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            btn.clicked.connect(lambda _checked=False, k=key: self._apply_preset(k))
            self._preset_buttons[key] = btn

        # --- Controles de imagen (modo I2V): lista de fotos con prompt propio ---
        self._use_last_button = QPushButton("Usar última imagen generada")
        self._use_last_button.clicked.connect(self._use_last_image)
        self._select_button = QPushButton("Seleccionar imágenes...")
        self._select_button.clicked.connect(self._select_image)

        image_buttons = QHBoxLayout()
        image_buttons.addWidget(self._use_last_button)
        image_buttons.addWidget(self._select_button)

        self._photos_layout = QVBoxLayout()
        self._photos_layout.setContentsMargins(0, 0, 0, 0)
        self._photos_container = QWidget()
        self._photos_container.setLayout(self._photos_layout)

        photos_scroll = QScrollArea()
        photos_scroll.setWidgetResizable(True)
        photos_scroll.setWidget(self._photos_container)
        photos_scroll.setMinimumHeight(120)
        photos_scroll.setMaximumHeight(300)

        self._image_section = QWidget()
        image_layout = QVBoxLayout(self._image_section)
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_layout.addLayout(image_buttons)
        image_layout.addWidget(photos_scroll)

        # --- Guion: escenas editables y envío al Ensamblador ---
        self._scene_table = QTableWidget(0, 4)
        self._scene_table.setHorizontalHeaderLabels(
            ["Escena", "Narración", "Prompt visual", "Duración"]
        )
        self._scene_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._scene_table.setMinimumHeight(140)
        self._scene_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        # Reordenar escenas arrastrando (la coherencia i2v se avisa al generar).
        self._scene_table.setDragDropMode(QTableWidget.DragDropMode.InternalMove)
        self._scene_table.model().rowsMoved.connect(self._on_scene_rows_moved)
        self._scene_table.setIconSize(QSize(48, 27))

        self._plan_button = QPushButton("Generar escenas")
        self._plan_button.clicked.connect(self._generate_sequence)
        self._send_button = QPushButton("Enviar al Ensamblador")
        self._send_button.setEnabled(False)
        self._send_button.clicked.connect(self._send_to_assembler)

        script_buttons = QHBoxLayout()
        script_buttons.addWidget(self._plan_button)
        script_buttons.addWidget(self._send_button)
        script_buttons.addStretch(1)

        self._idea = QLineEdit()
        self._idea.setPlaceholderText("Idea del vídeo para generar el guion...")
        self._scenes_spin = QSpinBox()
        self._scenes_spin.setRange(1, 20)
        self._scenes_spin.setValue(vs.DEFAULT_SCENES)
        self._script_button = QPushButton("Generar guion")
        self._script_button.clicked.connect(self._generate_script)

        idea_row = QHBoxLayout()
        idea_row.addWidget(self._idea, 1)
        idea_row.addWidget(QLabel("Escenas:"))
        idea_row.addWidget(self._scenes_spin)
        idea_row.addWidget(self._script_button)

        self._script_model = QComboBox()
        self._script_model.setToolTip("Modelo de Ollama que escribe el guion.")
        self._refresh_models_button = QPushButton("↻")
        self._refresh_models_button.setFixedWidth(28)
        self._refresh_models_button.setToolTip("Recargar la lista de modelos.")
        self._refresh_models_button.clicked.connect(self._populate_script_models)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Modelo:"))
        model_row.addWidget(self._script_model, 1)
        model_row.addWidget(self._refresh_models_button)

        self._transitions_check = QCheckBox("Transiciones entre escenas")
        self._transitions_check.setChecked(True)
        self._transition_combo = QComboBox()
        self._transition_combo.addItem("(por escena)", None)
        for key in vs.TRANSITIONS:
            self._transition_combo.addItem(vs.TRANSITION_LABELS.get(key, key), key)
        self._transition_combo.setToolTip(
            "Tipo de transición para todos los cruces (o «por escena»)."
        )

        transitions_row = QHBoxLayout()
        transitions_row.addWidget(self._transitions_check)
        transitions_row.addWidget(self._transition_combo, 1)

        self._projects_button = QPushButton("Proyectos…")
        self._projects_button.setToolTip("Abrir o borrar proyectos guardados.")
        self._projects_button.clicked.connect(self._open_projects_dialog)
        script_buttons.addWidget(self._projects_button)

        self._script_group = QGroupBox("Guion · escenas")
        script_layout = QVBoxLayout(self._script_group)
        script_layout.addLayout(idea_row)
        script_layout.addLayout(model_row)
        script_layout.addLayout(transitions_row)
        script_layout.addWidget(self._scene_table)
        script_layout.addLayout(script_buttons)

        self._generate_button = QPushButton("Generar")
        self._cancel_button = QPushButton("Cancelar")
        self._cancel_button.setEnabled(False)
        self._save_button = QPushButton("Guardar como...")
        self._save_button.setEnabled(False)
        self._open_button = QPushButton("Abrir carpeta")
        self._open_button.setEnabled(False)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)

        self._status = QLabel("")

        horizontal_row = QHBoxLayout()
        for key in vp.HORIZONTAL_KEYS:
            horizontal_row.addWidget(self._preset_buttons[key])
        horizontal_row.addStretch(1)

        vertical_row = QHBoxLayout()
        for key in vp.VERTICAL_KEYS:
            vertical_row.addWidget(self._preset_buttons[key])
        vertical_row.addStretch(1)

        presets_group = QVBoxLayout()
        presets_group.addWidget(QLabel("Horizontal:"))
        presets_group.addLayout(horizontal_row)
        presets_group.addWidget(QLabel("Vertical (TikTok):"))
        presets_group.addLayout(vertical_row)

        duration_row = QHBoxLayout()
        duration_row.addWidget(self._duration)
        duration_row.addWidget(self._adjust_audio_button)
        duration_row.addStretch(1)

        form = QFormLayout()
        self._form = form
        form.addRow("Modo", self._mode)
        form.addRow("Prompt", self._prompt)
        form.addRow("Negativo", self._negative_prompt)
        form.addRow("Ancho", self._width)
        form.addRow("Alto", self._height)
        form.addRow("Duración", duration_row)
        form.addRow("FPS", self._fps)
        form.addRow("Presets", presets_group)
        form.addRow("Seed", self._seed)
        form.addRow("", self._random_seed)

        buttons = QHBoxLayout()
        buttons.addWidget(self._generate_button)
        buttons.addWidget(self._cancel_button)
        buttons.addWidget(self._save_button)
        buttons.addWidget(self._open_button)

        advanced_form = QFormLayout()
        advanced_form.addRow("Timeout (s):", self._timeout)
        advanced_group = QGroupBox("Avanzado")
        advanced_group.setLayout(advanced_form)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(advanced_group)
        layout.addWidget(self._image_section)
        layout.addWidget(self._script_group)
        layout.addLayout(buttons)
        layout.addWidget(self._progress)
        layout.addWidget(self._status)
        layout.addStretch(1)

        self._generate_button.clicked.connect(self._generate)
        self._cancel_button.clicked.connect(self._cancel)
        self._save_button.clicked.connect(self._save_as)
        self._open_button.clicked.connect(self._open_folder)

        # Mostrar/ocultar la sección de imagen según el modo inicial.
        self._update_image_section()

    # ---- Guion y secuencia ---------------------------------------

    def load_script(self, script: "vs.Script", project=None) -> None:
        """Carga un guion (del chat o de un proyecto) en la tabla de escenas."""
        self._script = script
        self._project = project if isinstance(project, vproject.Project) else None
        self._clips = []
        self._sequence_path = None
        if self._project is not None:
            self._clips = [str(path) for path in self._project.saved_clips()]
            sequence = self._project.directory / "secuencia.mp4"
            if sequence.is_file():
                self._sequence_path = str(sequence)
            elif self._clips:
                self._sequence_path = self._clips[-1]
        self._send_button.setEnabled(bool(self._clips))
        self._restore_config()
        self._rebuild_scenes()
        self._script_group.setVisible(True)
        self._refresh_thumbnails()
        self._set_status(
            f"Guion cargado: «{script.title}» ({len(script.scenes)} escenas)."
        )

    def _rebuild_scenes(self) -> None:
        script = self._script
        self._scene_table.setRowCount(0)
        if script is None:
            return
        self._scene_table.setRowCount(len(script.scenes))
        for row, scene in enumerate(script.scenes):
            self._set_scene_row(row, scene)

    def _set_scene_row(self, row: int, scene) -> None:
        heading = QTableWidgetItem(scene.heading or f"Escena {row + 1}")
        heading.setFlags(heading.flags() & ~Qt.ItemFlag.ItemIsEditable)
        # El índice original viaja con la fila: es lo que permite reordenar
        # arrastrando sin depender del texto (que puede repetirse).
        heading.setData(Qt.ItemDataRole.UserRole, row)
        self._scene_table.setItem(row, 0, heading)
        self._scene_table.setItem(row, 1, QTableWidgetItem(scene.narration))
        self._scene_table.setItem(row, 2, QTableWidgetItem(scene.visual_prompt))
        self._scene_table.setItem(row, 3, QTableWidgetItem(str(scene.duration_s)))

    def _cell_text(self, row: int, column: int) -> str:
        item = self._scene_table.item(row, column)
        return item.text().strip() if item is not None else ""

    def _script_from_table(self) -> "vs.Script | None":
        """Vuelca la tabla (editable) en un guion validado, o ``None`` si falla."""
        script = self._script
        if script is None:
            return None
        scenes = []
        for row, scene in enumerate(script.scenes):
            narration = self._cell_text(row, 1)
            visual = self._cell_text(row, 2)
            duration = vs.clamp_duration(self._cell_text(row, 3), scene.duration_s)
            scenes.append(
                replace(
                    scene,
                    narration=narration,
                    visual_prompt=visual or scene.action,
                    duration_s=duration,
                )
            )
        try:
            updated = replace(script, scenes=tuple(scenes))
            updated.validate()
        except vs.ScriptError as exc:
            self._set_status(f"Guion inválido: {exc}")
            return None
        self._script = updated
        return updated

    def _generate_sequence(self) -> None:
        if self._engine is None:
            self._set_status("Wan Video no está configurado.")
            return
        if self._worker is not None and self._worker.isRunning():
            return
        script = self._script_from_table()
        if script is None:
            self._set_status("Carga y revisa un guion antes de generar escenas.")
            return
        script = self._apply_transition_override(script)
        script = self._with_config(script)
        self._script = script
        try:
            project = vproject.create_project(script.title, script, overwrite=True)
        except (OSError, vs.ScriptError) as exc:
            self._set_status(f"No se pudo crear el proyecto: {exc}")
            return
        self._project = project

        set_timeout = getattr(self._engine, "set_timeout", None)
        if callable(set_timeout):
            set_timeout(self._timeout.value())
        if not self._warn_low_ram():
            return

        self._set_busy(True)
        self._progress.setValue(0)
        self._set_status("Generando escenas... 0%")
        self._worker = SequenceWorker(
            self._engine,
            script,
            project,
            base_seed=vp.new_base_seed(),
            negative_prompt=self._negative_prompt.text().strip(),
            fps=self._fps.value(),
            transitions=self._transitions_check.isChecked(),
            parent=self,
            scheduler=self._gpu_scheduler(),
            liberar_otros=self._gpu_liberar_otros,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.result.connect(self._on_sequence_ready)
        self._worker.failed.connect(self._on_error)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _on_sequence_ready(self, result: dict) -> None:
        self._clips = list(result.get("clips") or [])
        self._sequence_path = result.get("sequence")
        self._last_path = self._sequence_path
        self._progress.setValue(100)
        self._save_button.setEnabled(bool(self._last_path))
        self._open_button.setEnabled(bool(self._last_path))
        self._send_button.setEnabled(bool(self._clips))
        self._refresh_thumbnails()
        where = f" en «{self._project.name}»" if self._project is not None else ""
        self._set_status(f"Escenas listas: {len(self._clips)} clips{where}.")

    def _send_to_assembler(self) -> None:
        target = self._sequence_path or (self._clips[0] if self._clips else None)
        if not target:
            self._set_status("No hay vídeo de secuencia que enviar.")
            return
        self.clips_ready.emit(target)
        self._set_status("Secuencia enviada al Ensamblador.")

    def _generate_script(self) -> None:
        """Pide un guion a Ollama y lo carga en la tabla de escenas."""
        if self._chat_engine is None:
            self._set_status("Ollama no está configurado; no se puede generar el guion.")
            return
        idea = self._idea.text().strip()
        if not idea:
            self._set_status("Escribe una idea para el guion.")
            return
        if self._script_worker is not None and self._script_worker.isRunning():
            return
        self._script_button.setEnabled(False)
        self._script_button.setText("Generando guion...")
        self._set_status("Generando guion con el modelo...")
        self._script_worker = ScriptWorker(
            self._chat_engine,
            idea,
            scenes=self._scenes_spin.value(),
            aspect=vs.ASPECT_HORIZONTAL,
            model=self._script_model.currentData(),
            parent=self,
            scheduler=self._gpu_scheduler(),
            liberar_otros=self._gpu_liberar_otros,
        )
        self._script_worker.result.connect(self._on_script_ready)
        self._script_worker.failed.connect(self._on_script_failed)
        self._script_worker.finished.connect(self._on_script_finished)
        self._script_worker.start()

    def start_script(self, idea: str, scenes: int | None = None) -> None:
        """Genera el guion desde el chat: rellena la idea y lanza el worker.

        Es el punto de entrada que usa la ventana principal cuando el mensaje del
        chat trae un guion (``Dispatch.payload``): mismo camino que el botón.
        """
        self._idea.setText(str(idea or "").strip())
        if scenes:
            try:
                self._scenes_spin.setValue(int(scenes))
            except (TypeError, ValueError):
                pass
        self._generate_script()

    def _on_script_ready(self, script) -> None:
        self.load_script(script)

    def _on_script_failed(self, message: str) -> None:
        self._set_status(f"No se pudo generar el guion: {message}")

    def _on_script_finished(self) -> None:
        self._script_button.setEnabled(True)
        self._script_button.setText("Generar guion")
        self._script_worker = None

    def set_chat_engine(self, engine: Engine | None) -> None:
        """Actualiza el motor de chat usado para generar el guion."""
        self._chat_engine = engine

    def _populate_script_models(self) -> None:
        """Rellena el combo de modelo del guion con los del motor de chat."""
        self._script_model.clear()
        self._script_model.addItem("(modelo por defecto)", None)
        engine = self._chat_engine
        list_models = getattr(engine, "list_models", None)
        if not callable(list_models):
            return
        try:
            names = [str(name) for name in list_models()]
        except Exception as exc:  # noqa: BLE001 - un motor caído no rompe la UI
            logger.debug("No se pudieron listar los modelos de chat: %s", exc)
            names = []
        default = getattr(engine, "default_model", None)
        if default and default not in names:
            names.insert(0, default)
        for name in names:
            self._script_model.addItem(name, name)
        if default:
            index = self._script_model.findData(default)
            if index >= 0:
                self._script_model.setCurrentIndex(index)

    def _restore_config(self) -> None:
        """Restaura fps y transiciones guardados en ``Script.config``."""
        config = self._script.config if self._script is not None else {}
        fps = config.get("fps")
        if isinstance(fps, int) and self._fps.minimum() <= fps <= self._fps.maximum():
            self._fps.setValue(fps)
        if isinstance(config.get("transitions"), bool):
            self._transitions_check.setChecked(config["transitions"])
        key = config.get("transition")
        index = self._transition_combo.findData(key) if key else 0
        self._transition_combo.setCurrentIndex(index if index >= 0 else 0)

    def _with_config(self, script: "vs.Script") -> "vs.Script":
        """Guarda en el guion los ajustes de generación actuales (por proyecto)."""
        config = dict(script.config)
        config.update(
            {
                "fps": self._fps.value(),
                "transitions": self._transitions_check.isChecked(),
                "transition": self._transition_combo.currentData(),
            }
        )
        return replace(script, config=config)

    def _apply_transition_override(self, script: "vs.Script") -> "vs.Script":
        """Aplica la transición elegida a todos los cruces (si no es «por escena»)."""
        key = self._transition_combo.currentData()
        if not key:
            return script
        scenes = tuple(replace(scene, transition=key) for scene in script.scenes)
        return replace(script, scenes=scenes)

    def _on_scene_rows_moved(self, parent, start, end, dest, row) -> None:
        """Reordena el guion tras arrastrar una fila de la tabla."""
        if self._reordering or self._script is None:
            return
        script = self._script
        order: list[int] = []
        for position in range(self._scene_table.rowCount()):
            item = self._scene_table.item(position, 0)
            data = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
            try:
                order.append(int(data))
            except (TypeError, ValueError):
                return
        if sorted(order) != list(range(len(script.scenes))):
            return
        self._script = replace(script, scenes=tuple(script.scenes[i] for i in order))
        self._reordering = True
        try:
            self._rebuild_scenes()
        finally:
            self._reordering = False
        self._set_status("Escenas reordenadas (la coherencia i2v puede resentirse).")
        self._refresh_thumbnails()

    def _refresh_thumbnails(self) -> None:
        """Pone como icono de cada escena el primer fotograma de su clip (si existe)."""
        if self._script is None or self._project is None:
            return
        extractor = FrameExtractor(self._project.directory)
        for row in range(self._scene_table.rowCount()):
            item = self._scene_table.item(row, 0)
            if item is None:
                continue
            clip = self._project.clip_path(row)
            if not clip.is_file():
                item.setIcon(QIcon())
                continue
            mini = self._project.directory / f"miniatura_{row:02d}.png"
            if not mini.is_file():
                try:
                    extractor.extract_frame(clip, 0.1, mini)
                except EngineError as exc:
                    logger.debug("Sin miniatura para %s: %s", clip, exc)
                    continue
            pixmap = QPixmap(str(mini))
            if not pixmap.isNull():
                item.setIcon(
                    QIcon(
                        pixmap.scaled(
                            48,
                            27,
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation,
                        )
                    )
                )

    def _open_projects_dialog(self) -> None:
        """Abre la cola de proyectos y carga el elegido en la tabla."""
        from ui.projects_dialog import ProjectsDialog

        dialog = ProjectsDialog(self)
        if not dialog.exec() or dialog.selected_project is None:
            return
        try:
            project = vproject.load_project(dialog.selected_project)
        except vs.ScriptError as exc:
            self._set_status(f"No se pudo abrir el proyecto: {exc}")
            return
        self.load_script(project.script, project)

    def _apply_preset(self, key: str) -> None:
        label, width, height, duration, _tooltip = vp.PRESETS[key]
        self._width.setValue(width)
        self._height.setValue(height)
        self._duration.setValue(duration)
        self._set_status(f"Preset aplicado: {label}")

    def _adjust_to_audio(self) -> None:
        """Ajusta la duración del vídeo a la de un archivo de audio."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar audio",
            "",
            "Audio (*.wav *.mp3 *.m4a *.ogg *.flac)",
        )
        if not path:
            return
        seconds = self._probe_audio_duration(path)
        if seconds is None:
            self._set_status("No se pudo leer la duración del audio.")
            return
        # Redondear hacia arriba para que -shortest no corte la voz.
        target = max(self._duration.minimum(), math.ceil(seconds))
        target = min(self._duration.maximum(), target)
        self._duration.setValue(target)
        self._set_status(f"Duración ajustada a {target}s (audio: {seconds:.1f}s).")

    @staticmethod
    def _probe_audio_duration(path: str) -> float | None:
        """Devuelve la duración en segundos del audio, o ``None`` si no se pudo leer."""
        if imageio_ffmpeg is None:
            return None
        try:
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            result = subprocess.run(
                [ffmpeg, "-i", str(path), "-hide_banner"],
                capture_output=True,
                text=True,
                timeout=_AUDIO_PROBE_TIMEOUT,
            )
        except Exception as exc:  # noqa: BLE001 - binario ausente o fallo de ejecución
            logger.debug("No se pudo sondear la duración del audio: %r", exc)
            return None
        match = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", result.stderr)
        if not match:
            return None
        h, m, s = match.groups()
        return int(h) * 3600 + int(m) * 60 + float(s)

    def _generate(self) -> None:
        if self._engine is None:
            self._set_status("Wan Video no está configurado.")
            return
        if self._worker is not None and self._worker.isRunning():
            return

        mode = self._mode.currentData() or "t2v"

        photos: list[dict] = []
        prompt = ""
        if mode == "i2v":
            self._sync_prompts()
            photos = [p for p in self._photos if p["path"]]
            if not photos:
                self._set_status("Selecciona una o más imágenes para el modo Imagen a vídeo.")
                return
            if any(not p["prompt"] for p in photos):
                self._set_status("Escribe un prompt para cada foto.")
                return
        else:
            prompt = self._prompt.toPlainText().strip()
            if not prompt:
                self._set_status("Escribe un prompt.")
                return

        seed = None
        if not self._random_seed.isChecked():
            try:
                seed = vp.parse_seed(self._seed.text())
            except ValueError:
                self._set_status(
                    "Seed inválido: usa un número entero o marca 'Aleatorio'."
                )
                return
        params = vp.build_generate_params(
            mode=mode,
            negative_prompt=self._negative_prompt.text().strip(),
            width=self._width.value(),
            height=self._height.value(),
            duration=self._duration.value(),
            fps=self._fps.value(),
            prompt=prompt,
            photos=photos,
            seed=seed,
        )

        set_timeout = getattr(self._engine, "set_timeout", None)
        if callable(set_timeout):
            set_timeout(self._timeout.value())

        if not self._warn_low_ram():
            return

        self._set_busy(True)
        self._progress.setValue(0)
        self._set_status("Generando... 0%")

        self._worker = VideoWorker(
            self._engine,
            params,
            self,
            scheduler=self._gpu_scheduler(),
            liberar_otros=self._gpu_liberar_otros,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.result.connect(self._on_video_ready)
        self._worker.failed.connect(self._on_error)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _on_progress(self, percent: int, message: str) -> None:
        self._progress.setValue(percent)
        self._set_status(f"Generando... {percent}% ({message})")

    def _on_video_ready(self, path: str) -> None:
        self._last_path = path
        self._progress.setValue(100)
        self._save_button.setEnabled(True)
        self._open_button.setEnabled(True)
        self._set_status(f"Listo: {path}")

    def _cancel(self) -> None:
        if self._worker is None or not self._worker.isRunning():
            return
        self._worker.requestInterruption()
        cancel_remote = getattr(self._engine, "cancel_remote", None)
        if callable(cancel_remote):
            cancel_remote()
        self._set_status("Cancelando...")

    def _save_as(self) -> None:
        if not self._last_path:
            return
        target, _ = QFileDialog.getSaveFileName(
            self, "Guardar vídeo como...", self._last_path, "MP4 (*.mp4);;WebM (*.webm)"
        )
        if not target:
            return
        try:
            shutil.copyfile(self._last_path, target)
        except OSError as exc:
            QMessageBox.critical(self, "Error", f"No se pudo guardar el vídeo: {exc}")
            return
        self._set_status(f"Vídeo guardado en: {target}")

    def _open_folder(self) -> None:
        if not self._last_path:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(self._last_path).parent)))

    # ---- Modo (t2v/i2v) e imagen ---------------------------------

    def _populate_modes(self) -> None:
        """Rellena el selector de modo con los workflows del motor."""
        saved = prefs.VideoPrefs.from_store(self._prefs)
        self._mode.blockSignals(True)
        self._mode.clear()
        if self._engine is not None:
            for model_id in self._engine.list_models():
                self._mode.addItem(vp.MODE_LABELS.get(model_id, model_id), model_id)
        idx = self._mode.findData(saved.mode)
        if idx >= 0:
            self._mode.setCurrentIndex(idx)
        self._mode.blockSignals(False)
        # Sincronizar la visibilidad de la sección de imagen con el modo restaurado.
        self._update_image_section()
        # Restaurar la última imagen usada. Se comprueba que siga existiendo: la
        # retención de output/ (services/output_cleanup) puede haberla borrado y
        # restaurarla a ciegas dejaba en la lista una foto inexistente.
        if saved.last_image and Path(saved.last_image).is_file():
            self._add_photos([saved.last_image])

    def _on_mode_changed(self) -> None:
        self._prefs.set(prefs.VIDEO, self._mode.currentData())
        self._update_image_section()

    def _update_image_section(self) -> None:
        is_i2v = self._mode.currentData() == "i2v"
        self._image_section.setVisible(is_i2v)
        # En I2V cada foto tiene su propio prompt; ocultar el prompt global.
        self._prompt.setVisible(not is_i2v)
        label = self._form.labelForField(self._prompt)
        if label is not None:
            label.setVisible(not is_i2v)

    def _use_last_image(self) -> None:
        output_dir = Path(__file__).resolve().parent.parent / "output"
        pngs = sorted(
            (p for p in output_dir.glob("*.png") if p.is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not pngs:
            self._set_status("No hay imágenes generadas en output/.")
            return
        self._add_photos([str(pngs[0])])

    def _select_image(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Seleccionar imagen(es)", "", "Imágenes (*.png *.jpg *.jpeg)"
        )
        if not paths:
            return
        self._add_photos(list(paths))

    def _add_photos(self, paths: list[str]) -> None:
        """Añade imágenes a la lista (sin duplicar), con prompt vacío."""
        existing = {item["path"] for item in self._photos}
        for path in paths:
            path = str(path).strip()
            if path and path not in existing:
                self._photos.append({"path": path, "prompt": ""})
                existing.add(path)
        if self._photos:
            self._prefs.set(prefs.VIDEO_LAST_IMAGE, self._photos[0]["path"])
        self._rebuild_photos()

    def _sync_prompts(self) -> None:
        """Copia el texto de los campos de prompt al modelo de datos."""
        for item, edit in zip(self._photos, self._prompt_edits):
            item["prompt"] = edit.text().strip()

    def _rebuild_photos(self) -> None:
        """Recrea las filas de fotos a partir de ``self._photos``."""
        while self._photos_layout.count():
            child = self._photos_layout.takeAt(0)
            widget = child.widget()
            if widget is not None:
                widget.deleteLater()
        self._prompt_edits = []
        for index, item in enumerate(self._photos):
            self._photos_layout.addWidget(self._build_photo_row(index, item))
        self._photos_layout.addStretch(1)

    def _build_photo_row(self, index: int, item: dict) -> QWidget:
        """Construye una fila: miniatura + prompt + subir/bajar/quitar."""
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)

        thumb = QLabel()
        thumb.setFixedSize(56, 56)
        thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        thumb.setStyleSheet("border: 1px solid #888;")
        pixmap = QPixmap(item["path"])
        if pixmap.isNull():
            thumb.setText("?")
        else:
            thumb.setPixmap(
                pixmap.scaled(
                    56,
                    56,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

        prompt_edit = QLineEdit(item["prompt"])
        prompt_edit.setPlaceholderText(f"Prompt foto {index + 1}...")

        up = QPushButton("↑")
        up.setToolTip("Subir de posición")
        up.setFixedWidth(28)
        up.setEnabled(index > 0)
        up.clicked.connect(lambda _checked=False, i=index: self._move_photo(i, -1))

        down = QPushButton("↓")
        down.setToolTip("Bajar de posición")
        down.setFixedWidth(28)
        down.setEnabled(index < len(self._photos) - 1)
        down.clicked.connect(lambda _checked=False, i=index: self._move_photo(i, 1))

        remove = QPushButton("×")
        remove.setToolTip("Quitar foto")
        remove.setFixedWidth(28)
        remove.clicked.connect(lambda _checked=False, i=index: self._remove_photo(i))

        layout.addWidget(thumb)
        layout.addWidget(prompt_edit, stretch=1)
        layout.addWidget(up)
        layout.addWidget(down)
        layout.addWidget(remove)

        self._prompt_edits.append(prompt_edit)
        return row

    def _move_photo(self, index: int, delta: int) -> None:
        """Intercambia la foto con la posición adyacente (reordenar)."""
        new_index = index + delta
        if new_index < 0 or new_index >= len(self._photos):
            return
        self._sync_prompts()
        self._photos[index], self._photos[new_index] = (
            self._photos[new_index],
            self._photos[index],
        )
        self._rebuild_photos()

    def _remove_photo(self, index: int) -> None:
        """Quita la foto de la lista."""
        self._sync_prompts()
        del self._photos[index]
        self._rebuild_photos()

    # ---- Utilidades -----------------------------------------------

    def _busy_widgets(self) -> tuple:
        return (
            self._generate_button,
            self._timeout,
            self._prompt,
            self._negative_prompt,
            self._mode,
            self._width,
            self._height,
            self._duration,
            self._adjust_audio_button,
            self._fps,
            self._random_seed,
            *self._preset_buttons.values(),
            self._use_last_button,
            self._select_button,
            self._photos_container,
            self._plan_button,
            self._send_button,
            self._scene_table,
        )

    def _busy_only_widgets(self) -> tuple:
        return (self._cancel_button,)

    def _on_busy_changed(self, busy: bool) -> None:
        # El seed solo se habilita si no está marcado "Aleatorio".
        self._seed.setEnabled(not busy and not self._random_seed.isChecked())

    def _focus_widget(self):
        """El prompt es el control que debe recuperar el foco."""
        return self._prompt

    def _refresh_system_stats(self) -> None:
        """Pide la RAM libre en segundo plano (no bloquea la UI, M5).

        La consulta vive en :mod:`services.system_stats` (función pura y
        testeable); aquí solo se lanza en el pool y se guarda el resultado.
        """
        base_url = None
        if self._engine is not None:
            base_url = getattr(self._engine, "base_url", None)
        self._stats_task = run_pool_task(
            stats.read_ram_free, base_url, on_result=self._on_ram_free
        )

    def _on_ram_free(self, ram_free: object) -> None:
        self._ram_free_bytes = ram_free
        self._ram_free_at = time.monotonic()

    def _warn_low_ram(self) -> bool:
        """Avisa si queda poca RAM libre, usando datos en caché (sin bloquear).

        Si no hay datos frescos (p. ej. ComfyUI no responde) no se avisa.
        """
        fresh = self._ram_free_bytes is not None and stats.is_fresh(
            self._ram_free_at, time.monotonic()
        )
        if not fresh:
            return True
        gb = float(self._ram_free_bytes) / (1024 ** 3)
        if gb >= 10:
            return True
        answer = QMessageBox.question(
            self,
            "RAM baja",
            f"Solo quedan {gb:.1f} GB de RAM libre. El vídeo puede ir lento o fallar. "
            "¿Continuar?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _on_timeout_changed(self) -> None:
        self._prefs.set(prefs.VIDEO_TIMEOUT, self._timeout.value())

    def _restore_timeout(self) -> None:
        value = self._prefs.get_int(prefs.VIDEO_TIMEOUT, self._timeout.value())
        if self._timeout.minimum() <= value <= self._timeout.maximum():
            self._timeout.blockSignals(True)
            self._timeout.setValue(value)
            self._timeout.blockSignals(False)

    def set_engine(self, engine: Engine | None) -> None:
        self._engine = engine
        self._populate_modes()
        if engine is None:
            self._set_status("Wan Video no está configurado o deshabilitado.")

