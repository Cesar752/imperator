"""Widget de generación de vídeo: prompt, parámetros, progreso y salida MP4."""

import logging
import math
import re
import shutil
import subprocess
import time
from pathlib import Path

try:
    import imageio_ffmpeg
except ImportError:  # pragma: no cover - dependencia declarada en requirements.txt
    imageio_ffmpeg = None

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
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
    QVBoxLayout,
    QWidget,
)

from core.video_assembler import VideoAssembler
from engines.base_engine import CANCEL_MESSAGE, Engine, EngineError
from services import preferences as prefs
from services import system_stats as stats
from services import video_params as vp
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


class VideoWidget(TaskWidget):
    """Formulario de generación de vídeo con barra de progreso y salida MP4."""

    _PREVIEW_SIZE = 256

    def __init__(self, engine: Engine | None, config_manager=None, parent=None) -> None:
        super().__init__(parent)
        self._engine: Engine | None = engine
        self._config_manager = config_manager
        self._prefs = prefs.PrefsStore(config_manager)
        self._worker: VideoWorker | None = None
        self._last_path: str | None = None
        self._photos: list[dict] = []
        self._prompt_edits: list[QLineEdit] = []
        self._stats_task = None
        self._ram_free_bytes: object = None
        self._ram_free_at = 0.0
        self._build_ui()
        self._populate_modes()
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
        # Restaurar la última imagen usada.
        if saved.last_image:
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

