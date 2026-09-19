"""Widget de generación de vídeo: prompt, parámetros, progreso y salida MP4."""

import logging
import shutil
from pathlib import Path

from PyQt6.QtCore import QThread, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from engines.base_engine import Engine, EngineError

logger = logging.getLogger(__name__)


class VideoWorker(QThread):
    """Ejecuta ``engine.generate_video()`` fuera del hilo de la UI."""

    progress = pyqtSignal(int, str)   # porcentaje, mensaje
    video_ready = pyqtSignal(str)     # ruta del MP4 generado
    failed = pyqtSignal(str)

    def __init__(self, engine: Engine, params: dict, parent=None) -> None:
        super().__init__(parent)
        self._engine = engine
        self._params = params

    def run(self) -> None:
        # El progreso llega desde el WebSocket (hilo auxiliar) vía este callback.
        params = dict(self._params)
        params["progress_callback"] = self._on_progress
        try:
            # Duck-typing: el motor debe exponer generate_video() (WanVideoEngine).
            path = self._engine.generate_video(**params)
        except EngineError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001 - red/timeout/JSON inesperados
            logger.debug("Error inesperado en VideoWorker: %r", exc, exc_info=True)
            self.failed.emit(f"Error inesperado: {exc}")
        else:
            self.video_ready.emit(str(path))

    def _on_progress(self, percent: int, message: str) -> None:
        # Emitir desde un hilo distinto es seguro en PyQt (conexión encolada).
        self.progress.emit(percent, message)


class VideoWidget(QWidget):
    """Formulario de generación de vídeo con barra de progreso y salida MP4."""

    # Presets: id -> (label, width, height, duración_s, tooltip).
    _PRESETS = {
        "rapido": ("Rápido", 832, 480, 2, "832×480 · 2 s (~69 s)"),
        "equilibrado": ("Equilibrado", 832, 480, 4, "832×480 · 4 s (~140 s)"),
        "calidad": ("Calidad", 1024, 576, 5, "1024×576 · 5 s (~250 s)"),
    }

    def __init__(self, engine: Engine | None, config_manager=None, parent=None) -> None:
        super().__init__(parent)
        self._engine: Engine | None = engine
        self._config_manager = config_manager
        self._worker: VideoWorker | None = None
        self._last_path: str | None = None
        self._build_ui()
        self._populate_models()
        if self._engine is None:
            self._set_status("Wan Video no está configurado o deshabilitado.")

    def _build_ui(self) -> None:
        self._prompt = QPlainTextEdit()
        self._prompt.setPlaceholderText("Describe el vídeo...")

        self._negative_prompt = QLineEdit()
        self._negative_prompt.setPlaceholderText("(opcional)")

        self._model = QComboBox()
        self._model.currentIndexChanged.connect(self._on_model_changed)

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

        self._fps = QSpinBox()
        self._fps.setRange(1, 60)
        self._fps.setValue(16)

        self._seed = QLineEdit()
        self._seed.setPlaceholderText("12345")
        self._random_seed = QCheckBox("Aleatorio")
        self._random_seed.setChecked(True)
        self._random_seed.toggled.connect(
            lambda checked: self._seed.setEnabled(not checked)
        )
        self._seed.setEnabled(False)

        self._preset_buttons: dict[str, QPushButton] = {}
        for key, (label, _w, _h, _dur, tooltip) in self._PRESETS.items():
            btn = QPushButton(label)
            btn.setToolTip(tooltip)
            btn.clicked.connect(lambda _checked=False, k=key: self._apply_preset(k))
            self._preset_buttons[key] = btn

        self._generate_button = QPushButton("Generar")
        self._save_button = QPushButton("Guardar como...")
        self._save_button.setEnabled(False)
        self._open_button = QPushButton("Abrir carpeta")
        self._open_button.setEnabled(False)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)

        self._status = QLabel("")

        presets_row = QHBoxLayout()
        for key in self._PRESETS:
            presets_row.addWidget(self._preset_buttons[key])

        form = QFormLayout()
        form.addRow("Modelo", self._model)
        form.addRow("Prompt", self._prompt)
        form.addRow("Negativo", self._negative_prompt)
        form.addRow("Ancho", self._width)
        form.addRow("Alto", self._height)
        form.addRow("Duración", self._duration)
        form.addRow("FPS", self._fps)
        form.addRow("Presets", presets_row)
        form.addRow("Seed", self._seed)
        form.addRow("", self._random_seed)

        buttons = QHBoxLayout()
        buttons.addWidget(self._generate_button)
        buttons.addWidget(self._save_button)
        buttons.addWidget(self._open_button)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(buttons)
        layout.addWidget(self._progress)
        layout.addWidget(self._status)
        layout.addStretch(1)

        self._generate_button.clicked.connect(self._generate)
        self._save_button.clicked.connect(self._save_as)
        self._open_button.clicked.connect(self._open_folder)

    def _apply_preset(self, key: str) -> None:
        label, width, height, duration, _tooltip = self._PRESETS[key]
        self._width.setValue(width)
        self._height.setValue(height)
        self._duration.setValue(duration)
        self._set_status(f"Preset aplicado: {label}")

    def _generate(self) -> None:
        if self._engine is None:
            self._set_status("Wan Video no está configurado.")
            return
        prompt = self._prompt.toPlainText().strip()
        if not prompt:
            self._set_status("Escribe un prompt.")
            return
        if self._worker is not None and self._worker.isRunning():
            return

        # length = duración * fps (+1 para el último frame), ej. 2s*16+1 = 33.
        length = round(self._duration.value() * self._fps.value()) + 1
        params = {
            "prompt": prompt,
            "negative_prompt": self._negative_prompt.text().strip(),
            "width": self._width.value(),
            "height": self._height.value(),
            "length": length,
            "fps": self._fps.value(),
            "model": self._model.currentData(),
        }
        if not self._random_seed.isChecked():
            try:
                params["seed"] = int(self._seed.text().strip())
            except ValueError:
                self._set_status(
                    "Seed inválido: usa un número entero o marca 'Aleatorio'."
                )
                return

        self._set_busy(True)
        self._progress.setValue(0)
        self._set_status("Generando... 0%")

        self._worker = VideoWorker(self._engine, params, self)
        self._worker.progress.connect(self._on_progress)
        self._worker.video_ready.connect(self._on_video_ready)
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

    def _on_error(self, message: str) -> None:
        self._set_status(f"Error: {message}")

    def _on_finished(self) -> None:
        self._set_busy(False)
        self._worker.deleteLater()
        self._worker = None

    def _save_as(self) -> None:
        if not self._last_path:
            return
        default_name = Path(self._last_path).name
        target, _ = QFileDialog.getSaveFileName(
            self, "Guardar vídeo como...", default_name, "MP4 (*.mp4);;WebM (*.webm)"
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

    def _set_busy(self, busy: bool) -> None:
        self._generate_button.setEnabled(not busy)
        self._prompt.setEnabled(not busy)
        self._negative_prompt.setEnabled(not busy)
        self._model.setEnabled(not busy)
        self._width.setEnabled(not busy)
        self._height.setEnabled(not busy)
        self._duration.setEnabled(not busy)
        self._fps.setEnabled(not busy)
        self._random_seed.setEnabled(not busy)
        for btn in self._preset_buttons.values():
            btn.setEnabled(not busy)
        self._seed.setEnabled(not busy and not self._random_seed.isChecked())

    def _set_status(self, text: str) -> None:
        self._status.setText(text)

    def set_engine(self, engine: Engine | None) -> None:
        self._engine = engine
        self._populate_models()
        if engine is None:
            self._set_status("Wan Video no está configurado o deshabilitado.")

    def _populate_models(self) -> None:
        """Rellena el selector con los GGUF del motor y restaura el último."""
        if self._engine is None:
            return
        models = list(self._engine.list_models())
        self._model.blockSignals(True)
        self._model.clear()
        for model_id in models:
            self._model.addItem(model_id, model_id)
        last = None
        if self._config_manager is not None:
            last = self._config_manager.load_last_models().get("video")
        if last in models:
            self._model.setCurrentIndex(models.index(last))
        else:
            # Por defecto, el GGUF high-noise (el primero que contenga "HighNoise").
            default = next((m for m in models if "HighNoise" in m), None)
            if default is not None:
                self._model.setCurrentIndex(models.index(default))
        self._model.blockSignals(False)

    def _on_model_changed(self) -> None:
        if self._config_manager is not None and self._model.currentData():
            self._config_manager.save_last_model("video", self._model.currentData())

