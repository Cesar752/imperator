"""Widget de generación de imagen: prompt, parámetros y preview del PNG."""

import logging
import shutil
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from engines.base_engine import Engine, EngineError

logger = logging.getLogger(__name__)


class ImageWorker(QThread):
    """Ejecuta ``engine.generate_image()`` fuera del hilo de la UI."""

    image_ready = pyqtSignal(str)   # ruta del PNG generado
    failed = pyqtSignal(str)

    def __init__(self, engine: Engine, params: dict, parent=None) -> None:
        super().__init__(parent)
        self._engine = engine
        self._params = params

    def run(self) -> None:
        try:
            # Duck-typing: el motor debe exponer generate_image() (ComfyUIEngine).
            path = self._engine.generate_image(**self._params)
        except EngineError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001 - red/timeout/JSON inesperados
            logger.debug("Error inesperado en ImageWorker: %r", exc, exc_info=True)
            self.failed.emit(f"Error inesperado: {exc}")
        else:
            self.image_ready.emit(str(path))


class ImageWidget(QWidget):
    """Formulario de generación con preview de la imagen resultante."""

    def __init__(self, engine: Engine | None, config_manager=None, parent=None) -> None:
        super().__init__(parent)
        self._engine: Engine | None = engine
        self._config_manager = config_manager
        self._worker: ImageWorker | None = None
        self._elapsed = 0
        self._last_path: str | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick)
        self._build_ui()
        self._populate_models()
        if self._engine is None:
            self._set_status("ComfyUI no está configurado o deshabilitado.")

    def _build_ui(self) -> None:
        self._prompt = QPlainTextEdit()
        self._prompt.setPlaceholderText("Describe la imagen...")

        self._negative_prompt = QLineEdit()
        self._negative_prompt.setPlaceholderText("(opcional)")

        self._model = QComboBox()
        self._model.currentIndexChanged.connect(self._on_model_changed)

        self._width = QSpinBox()
        self._width.setRange(256, 2048)
        self._width.setValue(1024)
        self._height = QSpinBox()
        self._height.setRange(256, 2048)
        self._height.setValue(1024)

        self._steps = QSpinBox()
        self._steps.setRange(1, 150)
        self._steps.setValue(20)

        self._guidance = QDoubleSpinBox()
        self._guidance.setRange(0.0, 30.0)
        self._guidance.setSingleStep(0.1)
        self._guidance.setValue(3.5)

        self._seed = QLineEdit()
        self._seed.setPlaceholderText("12345")
        self._random_seed = QCheckBox("Aleatorio")
        self._random_seed.setChecked(True)
        self._random_seed.toggled.connect(
            lambda checked: self._seed.setEnabled(not checked)
        )
        self._seed.setEnabled(False)

        self._generate_button = QPushButton("Generar")
        self._save_button = QPushButton("Guardar como...")
        self._save_button.setEnabled(False)

        self._status = QLabel("")
        self._preview = QLabel("La imagen aparecerá aquí.")
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setMinimumSize(480, 320)

        form = QFormLayout()
        form.addRow("Modelo:", self._model)
        form.addRow("Prompt:", self._prompt)
        form.addRow("Prompt negativo:", self._negative_prompt)
        form.addRow("Ancho:", self._width)
        form.addRow("Alto:", self._height)
        form.addRow("Steps:", self._steps)
        form.addRow("Guidance:", self._guidance)

        seed_row = QHBoxLayout()
        seed_row.addWidget(self._seed)
        seed_row.addWidget(self._random_seed)
        form.addRow("Seed:", seed_row)

        buttons = QHBoxLayout()
        buttons.addWidget(self._generate_button)
        buttons.addWidget(self._save_button)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(buttons)
        layout.addWidget(self._status)
        layout.addWidget(self._preview)

        self._generate_button.clicked.connect(self._generate)
        self._save_button.clicked.connect(self._save_as)

    def set_engine(self, engine: Engine | None) -> None:
        """Cambia el motor activo."""
        self._engine = engine
        self._populate_models()

    def _populate_models(self) -> None:
        """Rellena el selector con los modelos del motor y restaura el último."""
        if self._engine is None:
            return
        models = list(self._engine.list_models())
        self._model.blockSignals(True)
        self._model.clear()
        for model_id in models:
            self._model.addItem(self._display_model(model_id), model_id)
        last = None
        if self._config_manager is not None:
            last = self._config_manager.load_last_models().get("image")
        if last in models:
            self._model.setCurrentIndex(models.index(last))
        self._model.blockSignals(False)

    @staticmethod
    def _display_model(model_id: str) -> str:
        return model_id.replace("_", " ").title()

    def _on_model_changed(self) -> None:
        if self._config_manager is not None and self._model.currentData():
            self._config_manager.save_last_model("image", self._model.currentData())

    def _generate(self) -> None:
        if self._engine is None:
            self._set_status("Error: no hay motor ComfyUI configurado.")
            return
        prompt = self._prompt.toPlainText().strip()
        if not prompt:
            self._set_status("Escribe un prompt.")
            return
        if self._worker is not None and self._worker.isRunning():
            return

        params = {
            "prompt": prompt,
            "negative_prompt": self._negative_prompt.text().strip(),
            "width": self._width.value(),
            "height": self._height.value(),
            "steps": self._steps.value(),
            "guidance": self._guidance.value(),
            "model": self._model.currentData(),
        }
        if not self._random_seed.isChecked():
            try:
                params["seed"] = int(self._seed.text().strip())
            except ValueError:
                self._set_status("Seed inválido: usa un número entero o marca 'Aleatorio'.")
                return

        self._set_busy(True)
        self._elapsed = 0
        self._set_status("Generando... 0s")

        self._worker = ImageWorker(self._engine, params, self)
        self._worker.image_ready.connect(self._on_image_ready)
        self._worker.failed.connect(self._on_error)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()
        self._timer.start()

    def _tick(self) -> None:
        self._elapsed += 1
        self._set_status(f"Generando... {self._elapsed}s")

    def _on_image_ready(self, path: str) -> None:
        self._last_path = path
        self._show_image(path)
        self._save_button.setEnabled(True)
        self._set_status(f"Listo: {path}")

    def _on_error(self, message: str) -> None:
        self._set_status(f"Error: {message}")

    def _on_finished(self) -> None:
        self._timer.stop()
        self._set_busy(False)
        self._worker.deleteLater()
        self._worker = None

    def _show_image(self, path: str) -> None:
        pixmap = QPixmap(path)
        if pixmap.isNull():
            self._preview.setText("No se pudo cargar la imagen.")
            return
        scaled = pixmap.scaled(
            self._preview.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._preview.setPixmap(scaled)

    def _save_as(self) -> None:
        if not self._last_path:
            return
        default_name = Path(self._last_path).name
        target, _ = QFileDialog.getSaveFileName(
            self, "Guardar imagen como...", default_name, "PNG (*.png)"
        )
        if not target:
            return
        try:
            shutil.copyfile(self._last_path, target)
        except OSError as exc:
            QMessageBox.critical(self, "Error", f"No se pudo guardar la imagen: {exc}")
            return
        self._set_status(f"Imagen guardada en: {target}")

    def _set_busy(self, busy: bool) -> None:
        self._generate_button.setEnabled(not busy)
        self._prompt.setEnabled(not busy)
        self._negative_prompt.setEnabled(not busy)
        self._model.setEnabled(not busy)
        self._width.setEnabled(not busy)
        self._height.setEnabled(not busy)
        self._steps.setEnabled(not busy)
        self._guidance.setEnabled(not busy)
        self._random_seed.setEnabled(not busy)
        # El campo de seed solo se habilita si no está marcado "Aleatorio".
        self._seed.setEnabled(not busy and not self._random_seed.isChecked())

    def _set_status(self, text: str) -> None:
        self._status.setText(text)