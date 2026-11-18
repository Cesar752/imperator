"""Widget de generación de vídeo: prompt, parámetros, progreso y salida MP4."""

import logging
import math
import re
import shutil
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable

try:
    import imageio_ffmpeg
except ImportError:  # pragma: no cover - dependencia declarada en requirements.txt
    imageio_ffmpeg = None

from PyQt6 import sip
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
    QMenu,
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
from core.paths import OUTPUT
from core.memory import release_memory
from core.subprocess_utils import no_window_kwargs
from core.video_assembler import VideoAssembler
from engines.base_engine import CANCEL_MESSAGE, Engine, EngineError, chat_engine_reply
from services import image_params as ip
from services import preferences as prefs
from services import system_stats as stats
from services import video_params as vp
from services import video_project as vproject
from services import video_script as vs
from ui.pool_callbacks import PoolCallbackReceiver
from ui.task_widget import TaskWidget
from ui.theme import semantic
from ui.workers import ThreadWorker, gpu_slot, own_model_loaded, run_pool_task

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
        # ``image_paths``/``prompts`` son claves de la UI (varias fotos), no del
        # motor: se extraen aquí y se traducen a un ``image_path`` por llamada.
        # Dejarlas pasar era el «unexpected keyword argument 'image_paths'».
        images = params.pop("image_paths", None)
        prompts = params.pop("prompts", None)
        # Un clip (o una secuencia entera) = una reserva de GPU.
        with gpu_slot(self._scheduler, "video", liberar_otros=self._liberar_otros):
            if images:
                if len(images) == 1:
                    # Una sola foto: mismo camino que el i2v de una imagen.
                    params["image_path"] = images[0]
                    params["prompt"] = self._prompt_for(0, prompts)
                else:
                    # Varias imágenes: un clip por imagen y concatenación final.
                    return self._generate_sequence(images, prompts, params)
            # ``prompt`` es obligatorio en el motor: si el modo no lo traía
            # (p. ej. i2v), se envía vacío en vez de fallar por argumento ausente.
            params.setdefault("prompt", "")
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

    La generación es **reanudable**: cada escena se guarda en el proyecto al
    terminar y, al volver a lanzar, se salta el prefijo contiguo ya hecho y
    continúa desde la primera que falta (con la misma semilla común). Entre
    escena y escena se libera la memoria retenida (VRAM del motor, fotograma de
    encadenado ya consumido y RAM ociosa del proceso con
    :func:`core.memory.release_memory`), que es lo que evita que una secuencia
    larga muera tras unas cuantas escenas por acumulación de memoria.

    Con **fotos primero** (``keyframes=True`` y un ``image_engine``) cada escena
    pasa por dos pasos: primero se genera su **foto** con el mismo prompt visual
    y la misma semilla que usará el clip, y esa foto entra como primer fotograma
    del clip i2v. Así cada escena tiene foto y prompt a la vez y el clip no
    depende del fotograma anterior. Las fotos viven en el proyecto
    (``foto_NN.png``) y se **reutilizan** si ya existen, igual que los clips.
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
        resume: bool = True,
        release_between_scenes: bool = True,
        width: int | None = None,
        height: int | None = None,
        keyframes: bool = False,
        image_engine: Engine | None = None,
        image_model: str | None = None,
        image_negative: str = "",
        only: int | None = None,
        parent=None,
        scheduler=None,
        liberar_otros=None,
        on_release: Callable[[], None] | None = None,
        trim: Callable[[], bool] | None = None,
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
        self._resume = resume
        self._release_between_scenes = release_between_scenes
        self._width = width
        self._height = height
        #: «Fotos primero»: foto por escena (motor de imagen) + clip i2v desde ella.
        self._keyframes = keyframes
        self._image_engine = image_engine
        self._image_model = image_model
        self._image_negative = image_negative
        #: Si no es ``None``, se rehace **solo** esa escena (0-based); lo usa
        #: «Regenerar esta escena» tras cambiar su foto, sin tocar las demás.
        self._only = only
        self._scheduler = scheduler
        self._liberar_otros = liberar_otros
        # ``on_release``/``trim`` son puntos de inyección para tests: cuentan y
        # neutralizan la liberación sin tocar el working set del proceso.
        self._on_release = on_release
        self._trim = trim

    def work(self) -> dict:
        total = len(self._script.scenes)
        if total == 0:
            raise EngineError("El guion no tiene escenas que generar.")
        project = self._project
        # Reanudar = no rehacer el prefijo contiguo de escenas ya guardadas. Un
        # hueco invalida las posteriores (el encadenado i2v dejaría de ser
        # continuo), así que se regenera desde la primera que falta.
        if self._only is not None:
            # Una sola escena (p. ej. tras cambiar su foto): se rehace esa y solo
            # esa, sin arrastrar a las posteriores.
            start = max(0, min(int(self._only), total - 1))
            end = start + 1
        else:
            start = project.resume_index() if self._resume else 0
            end = total
        if self._only is None and start:
            logger.info(
                "Proyecto '%s': %d escena(s) ya hechas; se reanuda en la %d.",
                project.name,
                start,
                start + 1,
            )
        keyframes = self._keyframes and self._image_engine is not None
        if self._keyframes and not keyframes:
            logger.warning(
                "Fotos primero activado sin motor de imagen: se genera sin fotos."
            )
        extractor = FrameExtractor(project.directory)
        previous_frame: str | None = None
        if start > 0 and self._chain and not keyframes:
            # El encadenado continúa con el último fotograma de la escena previa.
            previous_frame = self._last_frame(
                extractor, project.clip_path(start - 1), start - 1
            )
        # Un paso por escena, o dos si hay foto (foto + clip).
        steps = (end - start) * 2 if keyframes else (end - start)
        with gpu_slot(self._scheduler, "video", liberar_otros=self._liberar_otros):
            for index in range(start, end):
                if self.isInterruptionRequested():
                    raise EngineError(CANCEL_MESSAGE)
                consumed_frame = previous_frame
                frame: str | None = None
                if keyframes:
                    # La foto de la escena es su primer fotograma: el clip no
                    # depende del último fotograma del clip anterior.
                    frame = self._ensure_keyframe(project, index, steps)
                    mode = "i2v"
                else:
                    mode = "i2v" if (self._chain and previous_frame) else "t2v"
                params = vp.scene_generate_params(
                    self._script,
                    index,
                    base_seed=self._base_seed,
                    fps=self._fps,
                    mode=mode,
                    negative_prompt=self._negative_prompt,
                    previous_frame=frame or previous_frame,
                    width=self._width,
                    height=self._height,
                )
                params["cancel_check"] = self.isInterruptionRequested
                clip_step = index * 2 + 1 if keyframes else index
                params["progress_callback"] = (
                    lambda percent, message, i=clip_step, k=index: self._on_scene_progress(
                        i, steps, percent, message, scene_label=f"Escena {k + 1}/{total}"
                    )
                )
                # Migas de pan: con el fallo registrado en el log, esta línea dice
                # exactamente en qué escena (y modo) se quedó la secuencia, que es
                # lo que faltaba para diagnosticar un corte a mitad de generación.
                logger.info(
                    "Proyecto '%s': generando escena %d/%d (modo %s).",
                    project.name,
                    index + 1,
                    total,
                    mode,
                )
                clip = Path(str(self._engine.generate_video(**params)))
                destination = project.clip_path(index)
                destination.parent.mkdir(parents=True, exist_ok=True)
                if clip.resolve() != destination.resolve():
                    shutil.copyfile(clip, destination)
                if self._chain and not keyframes:
                    previous_frame = self._last_frame(extractor, destination, index)
                # Guardado hecho: antes de la siguiente escena se suelta lo que
                # retenía la anterior (es el punto que evita el peta por RAM).
                if self._release_between_scenes and index + 1 < end:
                    self._release_memory(consumed_frame)

        # El último fotograma de encadenado ya no se necesita.
        if previous_frame:
            Path(previous_frame).unlink(missing_ok=True)

        # Todas las escenas presentes (reanudadas + nuevas), en orden.
        clips = [
            str(path)
            for index in range(total)
            if (path := project.clip_path(index)).is_file()
        ]
        sequence = None
        if len(clips) > 1:
            builder = VideoAssembler(output_dir=project.directory)
            sequence = self._concat(builder, clips)
        elif clips:
            sequence = clips[0]
        return {"clips": clips, "sequence": sequence}

    def _release_memory(self, consumed_frame: str | None) -> None:
        """Suelta lo que retenía la escena anterior antes de empezar la siguiente.

        - Los motores ComfyUI ya liberan su VRAM al terminar cada clip
          (``POST /free`` dentro de ``generate_video``); se repite aquí por si el
          motor no lo hace y para que la liberación quede explícita.
        - El fotograma de encadenado que acaba de entrar en la escena ya no se
          necesita: se borra para no dejar intermedios ni ocupar disco.
        - ``release_memory`` fuerza un ``gc.collect()`` y devuelve al SO la RAM
          ociosa del proceso (working set), que es lo que se acumulaba.
        """
        unload = getattr(self._engine, "unload_model", None)
        if callable(unload):
            try:
                unload()
            except Exception as exc:  # noqa: BLE001 - liberar no debe romper
                logger.debug("No se pudo liberar el motor entre escenas: %s", exc)
        if consumed_frame:
            Path(consumed_frame).unlink(missing_ok=True)
        release_memory(trim=self._trim)
        if self._on_release is not None:
            self._on_release()

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

    def _ensure_keyframe(self, project: "vproject.Project", index: int, steps: int) -> str:
        """Devuelve la foto de la escena ``index``, generándola si aún no existe.

        Se pide al motor de imagen con el prompt visual de la escena y su semilla
        (``scene_image_params``), de modo que la foto es reproducible y coherente
        con el clip. Si el fichero ya está en el proyecto (reanudación, o el
        usuario lo sustituyó a mano) no se vuelve a gastar GPU en él.
        """
        frame = project.frame_path(index)
        if frame.is_file():
            return str(frame)
        self.progress.emit(
            int(((index * 2) / max(1, steps)) * 100),
            f"Escena {index + 1}/{len(self._script.scenes)}: foto...",
        )
        params = vp.scene_image_params(
            self._script,
            index,
            model=self._image_model,
            width=self._width,
            height=self._height,
            negative_prompt=self._image_negative,
            seed=vp.scene_seed(self._base_seed, index),
        )
        params["cancel_check"] = self.isInterruptionRequested
        generated = Path(str(self._image_engine.generate_image(**params)))
        frame.parent.mkdir(parents=True, exist_ok=True)
        if generated.resolve() != frame.resolve():
            shutil.copyfile(generated, frame)
        return str(frame)

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
        self,
        step: int,
        steps: int,
        percent: int,
        message: str,
        *,
        scene_label: str = "",
    ) -> None:
        """Progreso global de la secuencia a partir del paso que se está haciendo.

        ``step``/``steps`` cuentan pasos (foto y clip son pasos distintos), pero
        la etiqueta siempre habla de la escena, que es lo que el usuario sigue.
        """
        overall = int(((step + percent / 100.0) / max(1, steps)) * 100)
        prefix = scene_label or f"Escena {step + 1}/{steps}"
        self.progress.emit(overall, f"{prefix}: {message}")


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
        with gpu_slot(
            self._scheduler,
            "chat",
            liberar_otros=self._liberar_otros,
            already_loaded=lambda: own_model_loaded(self._engine, self._model),
        ):
            reply = chat_engine_reply(
                self._engine,
                messages,
                model=self._model,
                on_token=self._on_token,
                cancel_check=self.isInterruptionRequested,
            )
        script, error = vs.try_parse_script(reply)
        if script is None:
            # Un guion que no parsea es casi siempre una respuesta **cortada**:
            # un modelo que razona (qwen3.x) gasta la ventana de contexto
            # "pensando" y el JSON se queda sin espacio, así que llega a medias y
            # el parser falla con «Expecting ',' delimiter». Decirlo evita perder
            # el tiempo mirando el motor cuando lo que hay que subir es num_ctx.
            note = getattr(type(self._engine), "TRUNCATED_NOTE", None)
            if note and note in reply:
                raise EngineError(
                    "El guion se cortó a mitad: se agotó el contexto de Ollama "
                    "(el modelo llenó la ventana razonando antes de escribir el "
                    "JSON). Sube 'num_ctx' del motor 'ollama' en engines.json "
                    "(p. ej. 16384) y, si tienes OLLAMA_NUM_PARALLEL=2, ponlo a 1 "
                    "para no repartir el contexto. Detalle: " + error
                )
            raise EngineError(f"No se pudo interpretar el guion del modelo: {error}")
        return script

    def _on_token(self, _delta: str, total: int) -> None:
        """Avisa del avance del guion (streaming): caracteres recibidos."""
        self.progress.emit(0, f"recibiendo… {total} caracteres")


class ScriptExtensionWorker(ThreadWorker):
    """Pide a Ollama **escenas nuevas** que continúen el guion cargado.

    Devuelve una tupla de ``Scene`` (solo las nuevas): el guion existente no se
    reescribe, así que las escenas ya generadas no se tocan.
    """

    def __init__(
        self,
        engine: Engine,
        script: "vs.Script",
        extra: int,
        *,
        model: str | None = None,
        parent=None,
        scheduler=None,
        liberar_otros=None,
    ) -> None:
        super().__init__(parent)
        self._engine = engine
        self._script = script
        self._extra = max(1, int(extra))
        self._model = model
        self._scheduler = scheduler
        self._liberar_otros = liberar_otros

    def work(self) -> tuple:
        messages = vs.extension_messages(self._script, self._extra)
        with gpu_slot(
            self._scheduler,
            "chat",
            liberar_otros=self._liberar_otros,
            already_loaded=lambda: own_model_loaded(self._engine, self._model),
        ):
            reply = chat_engine_reply(
                self._engine,
                messages,
                model=self._model,
                on_token=self._on_token,
                cancel_check=self.isInterruptionRequested,
            )
        return vs.parse_extension(reply)

    def _on_token(self, _delta: str, total: int) -> None:
        """Avisa del avance de la ampliación (streaming)."""
        self.progress.emit(0, f"recibiendo… {total} caracteres")


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
        image_engines: dict[str, Engine | None] | None = None,
    ) -> None:
        super().__init__(parent)
        self._engine: Engine | None = engine
        #: Motor de chat (Ollama) para generar el guion; puede ser ``None``.
        self._chat_engine: Engine | None = chat_engine
        #: Motores de imagen (ComfyUI local + BFL de pago) para «Fotos primero».
        self._image_engines: dict[str, Engine | None] = image_engines or {}
        self._image_model_engine: dict[str, str] = {}
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
        self._extend_worker = None
        self._reordering = False
        #: Token que descarta miniaturas de un proyecto anterior (C5.3).
        self._generation = 0
        #: Filas con miniatura en vuelo y su receptor del pool.
        self._thumb_pending: set[int] = set()
        self._thumb_rx: dict[int, PoolCallbackReceiver] = {}
        #: Guardia del recálculo de avisos: marcar una celda emite ``itemChanged``
        #: y sin esto la llamada se re-entraría hasta desbordar la pila.
        self._refreshing_warnings = False
        self._stats_task = None
        self._ram_free_bytes: object = None
        self._ram_free_at = 0.0
        self._build_ui()
        self._populate_modes()
        self._populate_script_models()
        self._populate_image_models()
        self._restore_image_config()
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
        # Menú por escena (clic derecho): reemplazar su clip o su foto, o rehacer
        # solo esa escena, sin tocar las demás.
        self._scene_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._scene_table.customContextMenuRequested.connect(self._scene_context_menu)

        # Avisos del validador ligero (determinista, sin modelo). No bloquean nada:
        # solo orientan, y se recalculan al editar cualquier celda.
        self._warnings = QLabel()
        self._warnings.setWordWrap(True)
        self._warnings.setVisible(False)
        self._warnings.setStyleSheet(f"color: {semantic('warning_fg').name()};")
        self._scene_table.itemChanged.connect(self._refresh_warnings)

        # Añadir escenas: una en blanco (a mano) o varias que continúen la historia
        # (se pide al modelo SOLO las nuevas, sin reescribir el guion).
        self._add_scene_button = QPushButton("Añadir escena")
        self._add_scene_button.setToolTip(
            "Añade una escena en blanco al final para escribirla a mano."
        )
        self._add_scene_button.clicked.connect(self._add_blank_scene)
        self._extend_spin = QSpinBox()
        self._extend_spin.setRange(1, 10)
        self._extend_spin.setValue(2)
        self._extend_button = QPushButton("Ampliar con IA")
        self._extend_button.setToolTip(
            "Pide al modelo N escenas nuevas que continúen la historia, con el mismo "
            "reparto y estilo."
        )
        self._extend_button.clicked.connect(self._extend_with_ai)
        add_scene_row = QHBoxLayout()
        add_scene_row.addWidget(self._add_scene_button)
        add_scene_row.addWidget(QLabel("Nuevas:"))
        add_scene_row.addWidget(self._extend_spin)
        add_scene_row.addWidget(self._extend_button)
        add_scene_row.addStretch(1)

        self._plan_button = QPushButton("Generar escenas")
        self._plan_button.clicked.connect(self._generate_sequence)
        self._send_button = QPushButton("Enviar al Ensamblador")
        self._send_button.setEnabled(False)
        self._send_button.clicked.connect(self._send_to_assembler)
        self._new_button = QPushButton("Nuevo")
        self._new_button.setToolTip(
            "Empezar un vídeo desde cero: quita de la pestaña el guion, sus clips y "
            "el proyecto cargado (el proyecto NO se borra del disco)."
        )
        # ``clicked`` emite un bool; la lambda evita depender del recorte de
        # argumentos de PyQt (``_start_fresh`` tiene ``confirm`` keyword-only).
        self._new_button.clicked.connect(lambda: self._start_fresh())
        self._delete_project_button = QPushButton("Borrar proyecto")
        self._delete_project_button.setToolTip(
            "Borra del disco el proyecto cargado y todos sus clips."
        )
        self._delete_project_button.clicked.connect(self._delete_current_project)

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

        # «Fotos primero»: una foto por escena antes de su clip (imagen → vídeo).
        self._keyframes_check = QCheckBox("Fotos primero (imagen → vídeo)")
        self._keyframes_check.setToolTip(
            "Genera una foto por escena con su prompt visual y la usa como primer "
            "fotograma del clip (i2v). Las fotos quedan en el proyecto y se "
            "reutilizan si ya existen."
        )
        self._keyframes_check.toggled.connect(self._on_keyframes_toggled)
        self._image_model = QComboBox()
        self._image_model.setToolTip(
            "Motor y modelo con el que se generan las fotos de las escenas "
            "(«PAGO» es la API de Black Forest Labs)."
        )
        self._image_model.currentIndexChanged.connect(self._on_image_model_changed)

        keyframes_row = QHBoxLayout()
        keyframes_row.addWidget(self._keyframes_check)
        keyframes_row.addWidget(self._image_model, 1)

        self._resume_check = QCheckBox("Reanudar proyecto")
        self._resume_check.setChecked(True)
        self._resume_check.setToolTip(
            "Si el proyecto ya tiene escenas, continúa desde la primera que falta "
            "en vez de rehacerlo todo. Cada escena se guarda al terminar y la "
            "semilla común del proyecto se conserva."
        )

        self._projects_button = QPushButton("Proyectos…")
        self._projects_button.setToolTip("Abrir o borrar proyectos guardados.")
        self._projects_button.clicked.connect(self._open_projects_dialog)
        script_buttons.addWidget(self._new_button)
        script_buttons.addWidget(self._delete_project_button)
        script_buttons.addWidget(self._projects_button)

        self._script_group = QGroupBox("Guion · escenas")
        script_layout = QVBoxLayout(self._script_group)
        script_layout.addLayout(idea_row)
        script_layout.addLayout(model_row)
        script_layout.addLayout(transitions_row)
        script_layout.addLayout(keyframes_row)
        script_layout.addWidget(self._resume_check)
        script_layout.addWidget(self._scene_table)
        script_layout.addWidget(self._warnings)
        script_layout.addLayout(add_scene_row)
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
        # Un proyecto nuevo invalida las miniaturas del anterior (C5.3).
        self._generation += 1
        self._thumb_pending.clear()
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
        hechas = len(self._clips)
        extra = f" · {hechas} escena(s) ya generadas" if hechas else ""
        self._set_status(
            f"Guion cargado: «{script.title}» ({len(script.scenes)} escenas){extra}."
        )

    def _rebuild_scenes(self) -> None:
        script = self._script
        # Sin señales mientras se rellenan las filas: ``itemChanged`` dispararía un
        # recálculo de avisos por cada celda.
        self._scene_table.blockSignals(True)
        try:
            self._scene_table.setRowCount(0)
            if script is not None:
                self._scene_table.setRowCount(len(script.scenes))
                for row, scene in enumerate(script.scenes):
                    self._set_scene_row(row, scene)
        finally:
            self._scene_table.blockSignals(False)
        self._refresh_warnings()

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

        # Reanudar conserva las escenas ya guardadas y la semilla común del
        # proyecto: así una generación interrumpida no vuelve a empezar de cero
        # ni pierde la coherencia visual de los clips anteriores.
        resume = self._resume_check.isChecked()
        existing = vproject.find_project(script.title) if resume else None
        base_seed = (
            existing.base_seed
            if existing is not None and existing.base_seed is not None
            else vp.new_base_seed()
        )
        try:
            project = vproject.create_project(
                script.title, script, overwrite=True, base_seed=base_seed
            )
        except (OSError, vs.ScriptError) as exc:
            self._set_status(f"No se pudo crear el proyecto: {exc}")
            return
        self._project = project

        pending = project.resume_index() if resume else 0
        if resume and pending >= len(script.scenes):
            self._set_status(
                f"El proyecto «{project.name}» ya tiene las {len(script.scenes)} "
                "escenas; se recompondrá la secuencia."
            )
        elif resume and pending:
            self._set_status(
                f"Reanudando «{project.name}» desde la escena {pending + 1}."
            )

        # «Fotos primero»: se hace una foto por escena antes de su clip. Sin
        # motor de imagen disponible se avisa y se sigue sin fotos (t2v/encadenado).
        keyframes = self._keyframes_check.isChecked()
        image_engine = self._current_image_engine() if keyframes else None
        if keyframes and image_engine is None:
            self._set_status(
                "«Fotos primero» activado, pero no hay motor de imagen "
                "disponible; se generará sin fotos."
            )

        set_timeout = getattr(self._engine, "set_timeout", None)
        if callable(set_timeout):
            set_timeout(self._timeout.value())
        if not self._warn_low_ram():
            return

        self._set_busy(True)
        self._progress.setValue(0)
        if not (resume and pending):
            self._set_status("Generando escenas... 0%")
        self._worker = SequenceWorker(
            self._engine,
            script,
            project,
            base_seed=base_seed,
            negative_prompt=self._negative_prompt.text().strip(),
            fps=self._fps.value(),
            transitions=self._transitions_check.isChecked(),
            resume=resume,
            width=self._width.value(),
            height=self._height.value(),
            keyframes=keyframes,
            image_engine=image_engine,
            image_model=self._image_model.currentData(),
            image_negative=self._negative_prompt.text().strip(),
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

    def _start_fresh(self, *, confirm: bool = True) -> None:
        """Deja la pestaña lista para un **vídeo nuevo**.

        Quita el guion, sus clips y el proyecto **de la pestaña**; no toca el disco
        (para eso están «Borrar proyecto» y el diálogo «Proyectos…»). Sin esto,
        al terminar un vídeo la pestaña seguía mostrando el anterior y no había
        forma clara de empezar otro desde cero.
        """
        project = self._project
        if confirm and project is not None and project.saved_clips():
            answer = QMessageBox.question(
                self,
                "Empezar de cero",
                "Se quitará de la pestaña el guion y sus clips.\n\n"
                f"El proyecto «{project.name}» seguirá en disco "
                "(usa «Borrar proyecto» para eliminarlo).\n\n¿Continuar?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self._script = None
        self._project = None
        self._clips = []
        self._sequence_path = None
        self._last_path = None
        self._scene_table.setRowCount(0)
        self._send_button.setEnabled(False)
        self._save_button.setEnabled(False)
        self._open_button.setEnabled(False)
        self._progress.setValue(0)
        self._idea.clear()
        self._set_status(
            "Vídeo nuevo: escribe una idea y pulsa «Generar guion», o pega un guion "
            "y revisa la tabla."
        )

    def _delete_current_project(self) -> None:
        """Borra del disco el proyecto cargado y deja la pestaña en blanco."""
        project = self._project
        if project is None:
            self._set_status(
                "No hay proyecto cargado. Usa «Proyectos…» para elegir uno y borrarlo."
            )
            return
        answer = QMessageBox.question(
            self,
            "Borrar proyecto",
            f"¿Borrar el proyecto «{project.name}» y todos sus clips del disco?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        name = project.name
        if vproject.delete_project(project.directory):
            self._start_fresh(confirm=False)
            self._set_status(f"Proyecto «{name}» borrado del disco.")
        else:
            self._set_status(f"No se pudo borrar el proyecto «{name}».")

    # ---- Edición por escena (clip / foto) -------------------------

    @staticmethod
    def _output_dir() -> Path:
        """Carpeta ``output/`` (para abrir los diálogos donde suelen estar los MP4)."""
        return OUTPUT

    def _refresh_from_project(self, *, sequence_valid: bool = True) -> None:
        """Recalcula clips y secuencia del proyecto tras tocar un fichero.

        Con ``sequence_valid=False`` la ``secuencia.mp4`` guardada queda obsoleta
        (se ha cambiado un clip), así que se deja de ofrecer hasta recomponerla.
        """
        project = self._project
        if project is None:
            return
        self._clips = [str(path) for path in project.saved_clips()]
        if not sequence_valid:
            self._sequence_path = None
        elif self._clips:
            sequence = project.directory / "secuencia.mp4"
            self._sequence_path = (
                str(sequence) if sequence.is_file() else self._clips[-1]
            )
        else:
            self._sequence_path = None
        self._send_button.setEnabled(
            bool(self._clips) and self._sequence_path is not None
        )

    def _scene_context_menu(self, pos) -> None:
        """Clic derecho en una escena: reemplazar su clip/foto o rehacerla."""
        if self._script is None or self._project is None:
            return
        if (self._worker is not None and self._worker.isRunning()) or (
            self._script_worker is not None and self._script_worker.isRunning()
        ):
            return
        item = self._scene_table.itemAt(pos)
        if item is None:
            return
        row = item.row()
        menu = QMenu(self)
        replace_clip = menu.addAction("Reemplazar clip (MP4)…")
        replace_photo = menu.addAction("Reemplazar foto (PNG/JPG)…")
        regenerate = menu.addAction("Regenerar esta escena")
        quitar_foto = menu.addAction("Quitar foto")
        chosen = menu.exec(self._scene_table.viewport().mapToGlobal(pos))
        if chosen is replace_clip:
            self._replace_scene_clip(row)
        elif chosen is replace_photo:
            self._replace_scene_photo(row)
        elif chosen is regenerate:
            self._regenerate_scene(row)
        elif chosen is quitar_foto:
            self._clear_scene_photo(row)

    def _replace_scene_clip(self, row: int) -> None:
        """Pone otro MP4 (p. ej. uno recién generado en ``output/``) como clip."""
        project = self._project
        if project is None:
            return
        source, _ = QFileDialog.getOpenFileName(
            self,
            f"Clip para la escena {row + 1}",
            str(self._output_dir()),
            "Vídeo (*.mp4 *.webm *.mov)",
        )
        if not source:
            return
        try:
            shutil.copyfile(source, project.clip_path(row))
        except OSError as exc:
            self._set_status(f"No se pudo reemplazar el clip: {exc}")
            return
        # La miniatura cacheada ya no corresponde al clip nuevo.
        (project.directory / f"miniatura_{row:02d}.png").unlink(missing_ok=True)
        self._refresh_from_project(sequence_valid=False)
        self._refresh_thumbnails()
        self._set_status(
            f"Escena {row + 1}: clip reemplazado por «{Path(source).name}». "
            "Pulsa «Generar escenas» para recomponer la secuencia (no rehará las "
            "escenas ya hechas)."
        )

    def _replace_scene_photo(self, row: int) -> None:
        """Cambia la **foto** de arranque de la escena (sirve para cambiar el personaje)."""
        project = self._project
        if project is None:
            return
        source, _ = QFileDialog.getOpenFileName(
            self,
            f"Foto para la escena {row + 1}",
            str(self._output_dir()),
            "Imagen (*.png *.jpg *.jpeg *.webp)",
        )
        if not source:
            return
        try:
            shutil.copyfile(source, project.frame_path(row))
        except OSError as exc:
            self._set_status(f"No se pudo reemplazar la foto: {exc}")
            return
        self._set_status(
            f"Escena {row + 1}: foto reemplazada por «{Path(source).name}». "
            "Pulsa «Regenerar esta escena» para rehacer su clip con la nueva foto."
        )

    def _clear_scene_photo(self, row: int) -> None:
        """Quita la foto de la escena (su clip se rehará sin foto si se regenera)."""
        project = self._project
        if project is None:
            return
        project.frame_path(row).unlink(missing_ok=True)
        self._set_status(f"Escena {row + 1}: foto quitada.")

    def _regenerate_scene(self, row: int) -> None:
        """Rehace **solo** el clip de esa escena (con su foto actual, si la hay)."""
        if self._engine is None:
            self._set_status("Wan Video no está configurado.")
            return
        if self._worker is not None and self._worker.isRunning():
            return
        script = self._script_from_table()
        if script is None or self._project is None:
            self._set_status("Carga un guion antes de regenerar una escena.")
            return
        script = self._with_config(self._apply_transition_override(script))
        self._script = script
        project = self._project
        # Se borra el clip (y su miniatura) para que el worker lo rehaga.
        project.clip_path(row).unlink(missing_ok=True)
        (project.directory / f"miniatura_{row:02d}.png").unlink(missing_ok=True)
        self._refresh_from_project(sequence_valid=False)

        keyframes = self._keyframes_check.isChecked()
        image_engine = self._current_image_engine() if keyframes else None
        set_timeout = getattr(self._engine, "set_timeout", None)
        if callable(set_timeout):
            set_timeout(self._timeout.value())
        if not self._warn_low_ram():
            return
        base_seed = (
            project.base_seed
            if project.base_seed is not None
            else vp.new_base_seed()
        )
        self._set_busy(True)
        self._progress.setValue(0)
        self._set_status(f"Regenerando la escena {row + 1}…")
        self._worker = SequenceWorker(
            self._engine,
            script,
            project,
            base_seed=base_seed,
            negative_prompt=self._negative_prompt.text().strip(),
            fps=self._fps.value(),
            transitions=self._transitions_check.isChecked(),
            resume=False,
            only=row,
            width=self._width.value(),
            height=self._height.value(),
            keyframes=keyframes,
            image_engine=image_engine,
            image_model=self._image_model.currentData(),
            image_negative=self._negative_prompt.text().strip(),
            parent=self,
            scheduler=self._gpu_scheduler(),
            liberar_otros=self._gpu_liberar_otros,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.result.connect(self._on_sequence_ready)
        self._worker.failed.connect(self._on_error)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _add_blank_scene(self) -> None:
        """Añade una escena vacía al final del guion (para escribirla a mano)."""
        if self._script is None:
            self._set_status("Genera o carga un guion antes de añadir escenas.")
            return
        number = len(self._script.scenes) + 1
        nueva = vs.Scene(
            heading=f"ESC {number}",
            action="(describe qué ocurre en esta escena)",
            visual_prompt="(describe qué se ve en el plano)",
            narration="",
            duration_s=vs.clamp_duration(4),
        )
        self._script = replace(self._script, scenes=(*self._script.scenes, nueva))
        self._rebuild_scenes()
        self._refresh_thumbnails()
        self._set_status(
            f"Escena {number} añadida. Edítala en la tabla y pulsa «Generar escenas» "
            "(solo generará las que falten)."
        )

    def _extend_with_ai(self) -> None:
        """Pide al modelo N escenas nuevas que continúen el guion cargado."""
        if self._chat_engine is None:
            self._set_status("Ollama no está configurado; no se puede ampliar el guion.")
            return
        if self._script is None:
            self._set_status("Genera o carga un guion antes de ampliarlo.")
            return
        if self._script_worker is not None and self._script_worker.isRunning():
            return
        if self._worker is not None and self._worker.isRunning():
            return
        if self._extend_worker is not None and self._extend_worker.isRunning():
            return
        extra = self._extend_spin.value()
        self._script_button.setEnabled(False)
        self._extend_button.setEnabled(False)
        self._extend_button.setText("Ampliando…")
        self._set_status(f"Pidiendo {extra} escena(s) nuevas al modelo…")
        self._extend_worker = ScriptExtensionWorker(
            self._chat_engine,
            self._script,
            extra,
            model=self._script_model.currentData(),
            parent=self,
            scheduler=self._gpu_scheduler(),
            liberar_otros=self._gpu_liberar_otros,
        )
        self._extend_worker.result.connect(self._on_extension_ready)
        self._extend_worker.failed.connect(self._on_extension_failed)
        self._extend_worker.progress.connect(self._on_extension_progress)
        self._extend_worker.finished.connect(self._on_extension_finished)
        self._extend_worker.start()

    def _on_extension_ready(self, scenes) -> None:
        if self._script is None:
            return
        try:
            updated = vs.extend_script(self._script, tuple(scenes))
        except vs.ScriptError as exc:
            self._set_status(f"No se pudo ampliar el guion: {exc}")
            return
        self._script = updated
        self._rebuild_scenes()
        self._refresh_thumbnails()
        self._set_status(
            f"Guion ampliado con {len(scenes)} escena(s) (ahora "
            f"{len(updated.scenes)}). Pulsa «Generar escenas» para generar solo las "
            "nuevas."
        )

    def _on_extension_progress(self, _percent: int, message: str) -> None:
        """Avance de la ampliación (streaming)."""
        self._set_status(f"Ampliando el guion… {message}")

    def _on_extension_failed(self, message: str) -> None:
        self._set_status(f"No se pudo ampliar el guion: {message}")

    def _on_extension_finished(self) -> None:
        self._extend_button.setEnabled(True)
        self._extend_button.setText("Ampliar con IA")
        self._script_button.setEnabled(self._chat_engine is not None)
        self._extend_worker = None

    def _script_for_warnings(self) -> "vs.Script | None":
        """Copia silenciosa del guion con lo que hay en la tabla (sin validar).

        Sirve para calcular avisos mientras se edita; usar ``_script_from_table``
        escribiría el estado en cada pulsación.
        """
        script = self._script
        if script is None:
            return None
        scenes = []
        for row, scene in enumerate(script.scenes):
            scenes.append(
                replace(
                    scene,
                    narration=self._cell_text(row, 1) or scene.narration,
                    visual_prompt=self._cell_text(row, 2) or scene.visual_prompt,
                    duration_s=vs.clamp_duration(
                        self._cell_text(row, 3), scene.duration_s
                    ),
                )
            )
        return replace(script, scenes=tuple(scenes))

    def _refresh_warnings(self) -> None:
        """Recalcula los avisos y los marca en la tabla, con guardia de reentrada.

        Marcar una celda (``setData``/``setBackground``/``setToolTip``) emite
        ``itemChanged``: sin el guardia, ``_refresh_warnings`` se volvería a llamar
        a sí misma de forma infinita (desbordaba la pila en los tests).
        """
        if self._refreshing_warnings:
            return
        self._refreshing_warnings = True
        try:
            self._refresh_warnings_inner()
        finally:
            self._refreshing_warnings = False

    def _refresh_warnings_inner(self) -> None:
        label = self._warnings
        script = self._script_for_warnings()
        if script is None:
            label.setVisible(False)
            return
        warnings = vs.script_warnings(script)
        for row in range(self._scene_table.rowCount()):
            item = self._scene_table.item(row, 0)
            if item is not None:
                item.setData(Qt.ItemDataRole.BackgroundRole, None)
                item.setToolTip("")
        by_scene: dict[int, list[str]] = {}
        for warning in warnings:
            if warning.scene is not None:
                by_scene.setdefault(warning.scene, []).append(warning.message)
        for scene_number, messages in by_scene.items():
            item = self._scene_table.item(scene_number - 1, 0)
            if item is not None:
                item.setBackground(semantic("warning_bg"))
                item.setToolTip("\n".join(messages))
        if not warnings:
            label.setVisible(False)
            return
        first = "; ".join(warning.message for warning in warnings[:4])
        extra = f" (+{len(warnings) - 4} más)" if len(warnings) > 4 else ""
        label.setText(f"⚠ {len(warnings)} aviso(s): {first}{extra}")
        label.setVisible(True)

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
        # Un guion nuevo es un vídeo nuevo: se limpia lo anterior para no arrastrar
        # los clips del anterior. Al llegar el guion, si su título coincide con un
        # proyecto existente se reengancha (reanudar tras un corte, ver TODO §10.6).
        self._start_fresh(confirm=False)
        self._idea.setText(idea)
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
        self._script_worker.progress.connect(self._on_script_progress)
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
        # Un guion recién generado no trae proyecto. Si ya existe uno con el mismo
        # título (escenas hechas en una sesión anterior), se engancha: así la tabla
        # muestra sus miniaturas y «Enviar al Ensamblador» queda disponible sin
        # volver a generar nada. Antes se cargaba con ``project=None`` y el widget
        # «no sabía» que las escenas ya estaban hechas.
        self.load_script(script, vproject.find_project(script.title))

    def _on_script_progress(self, _percent: int, message: str) -> None:
        """Avance del guion (streaming): se muestra mientras el modelo responde."""
        self._set_status(f"Generando guion… {message}")

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

    # ---- Fotos de escena (imagen → vídeo) -------------------------

    def set_engines(self, engines: dict[str, Engine | None] | None) -> None:
        """Actualiza los motores de imagen disponibles (ComfyUI local + BFL)."""
        self._image_engines = engines or {}
        self._populate_image_models()

    def _populate_image_models(self) -> None:
        """Rellena el combo con los modelos de todos los motores de imagen."""
        self._image_model.blockSignals(True)
        self._image_model.clear()
        self._image_model_engine = {}
        for engine_id, engine in self._image_engines.items():
            if engine is None:
                continue
            list_models = getattr(engine, "list_models", None)
            if not callable(list_models):
                continue
            try:
                models = [str(name) for name in list_models()]
            except Exception as exc:  # noqa: BLE001 - un motor caído no rompe la UI
                logger.debug("No se pudieron listar los modelos de imagen: %s", exc)
                continue
            default = getattr(engine, "default_model", None)
            if default and default not in models:
                models.insert(0, default)
            for model_id in models:
                self._image_model.addItem(
                    ip.display_model(model_id, engine_id), model_id
                )
                self._image_model_engine[model_id] = engine_id
        saved = self._prefs.get_str(prefs.VIDEO_IMAGE_MODEL)
        index = self._image_model.findData(saved) if saved else -1
        self._image_model.setCurrentIndex(index if index >= 0 else 0)
        self._image_model.blockSignals(False)

    def _current_image_engine(self) -> Engine | None:
        """Motor de imagen elegido para las fotos de escena (o ``None``)."""
        model = self._image_model.currentData()
        return self._image_engines.get(self._image_model_engine.get(model))

    def _restore_image_config(self) -> None:
        """Restaura de las preferencias la casilla de fotos y el modelo elegido."""
        self._keyframes_check.setChecked(self._prefs.get_bool(prefs.VIDEO_KEYFRAMES))
        saved = self._prefs.get_str(prefs.VIDEO_IMAGE_MODEL)
        index = self._image_model.findData(saved) if saved else -1
        if index >= 0:
            self._image_model.setCurrentIndex(index)

    def _on_keyframes_toggled(self, checked: bool) -> None:
        self._prefs.set_bool(prefs.VIDEO_KEYFRAMES, checked)

    def _on_image_model_changed(self) -> None:
        self._prefs.set(prefs.VIDEO_IMAGE_MODEL, self._image_model.currentData())

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
        """Pone como icono de cada escena el primer fotograma de su clip (C5.3).

        La extracción con ffmpeg va al pool: antes se hacía escena a escena en el
        hilo de la UI y con 20 escenas la ventana se congelaba decenas de
        segundos. Si la miniatura ya está en disco se pinta sin lanzar nada.
        """
        if self._script is None or self._project is None:
            return
        for row in range(self._scene_table.rowCount()):
            item = self._scene_table.item(row, 0)
            if item is None:
                continue
            clip = self._project.clip_path(row)
            if not clip.is_file():
                item.setIcon(QIcon())
                continue
            mini = self._project.directory / f"miniatura_{row:02d}.png"
            if mini.is_file():
                self._set_thumbnail(item, mini)
            elif row not in self._thumb_pending:
                # Placeholder hasta que el pool devuelva el fotograma.
                item.setIcon(QIcon())
                self._thumb_pending.add(row)
                self._request_thumbnail(row, clip, mini)

    def _request_thumbnail(self, row: int, clip: Path, mini: Path) -> None:
        """Lanza la extracción del fotograma de ``row`` al pool (C5.3)."""
        rx = PoolCallbackReceiver(
            self, self._on_thumbnail, self._on_thumbnail_error, context=(row, str(mini))
        )
        self._thumb_rx[row] = rx
        rx.arm()
        extractor = FrameExtractor(self._project.directory)
        run_pool_task(
            extractor.extract_frame,
            clip,
            0.1,
            mini,
            on_result=rx.handle,
            on_error=rx.handle_error,
        )

    def _set_thumbnail(self, item, mini: Path) -> None:
        """Pinta ``mini`` como icono de la celda (solo Qt, sin ffmpeg)."""
        pixmap = QPixmap(str(mini))
        if pixmap.isNull():
            return
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

    def _on_thumbnail(self, contexto, _valor) -> None:
        """Rellena la miniatura cuando el pool termina la extracción."""
        if sip.isdeleted(self):
            return
        row, mini = contexto
        self._thumb_pending.discard(row)
        item = self._scene_table.item(row, 0)
        if item is not None:
            self._set_thumbnail(item, Path(mini))

    def _on_thumbnail_error(self, contexto, mensaje: str) -> None:
        if sip.isdeleted(self):
            return
        self._thumb_pending.discard(contexto[0])
        logger.debug("Sin miniatura para la escena %s: %s", contexto[0], mensaje)

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
                **no_window_kwargs(),
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
                # «Generar» hace un clip i2v y necesita una foto. Con un guion
                # cargado se nombra el botón exacto; y si sus escenas ya están
                # todas hechas, el paso que falta es enviarlas al Ensamblador.
                escenas = len(self._script.scenes) if self._script else 0
                if escenas and len(self._clips) >= escenas:
                    self._set_status(
                        f"Las {escenas} escenas ya están generadas. Usa "
                        "«Enviar al Ensamblador» para montar el vídeo final "
                        "(y Audio para la narración)."
                    )
                elif escenas:
                    self._set_status(
                        "«Generar» hace un clip de imagen→vídeo y necesita una foto. "
                        f"Para las {escenas} escenas del guion usa «Generar escenas»."
                    )
                else:
                    self._set_status(
                        "Selecciona una o más imágenes para el modo Imagen a vídeo."
                    )
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
        output_dir = OUTPUT
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
            self._new_button,
            self._delete_project_button,
            self._add_scene_button,
            self._extend_spin,
            self._extend_button,
            self._scene_table,
            self._resume_check,
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

