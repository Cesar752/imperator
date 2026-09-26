"""Widget de generación de imagen: prompt, parámetros y preview del PNG."""

import logging
import shutil

from PyQt6.QtCore import Qt, QTimer
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
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
)

from engines.base_engine import Engine
from services import image_params as ip
from services import preferences as prefs
from ui.task_widget import TaskWidget
from ui.workers import ThreadWorker, run_pool_task

logger = logging.getLogger(__name__)


class ImageWorker(ThreadWorker):
    """Ejecuta ``engine.generate_image()`` fuera del hilo de la UI."""

    def __init__(self, engine: Engine, params: dict, parent=None) -> None:
        super().__init__(parent)
        self._engine = engine
        self._params = params

    def work(self) -> str:
        params = dict(self._params)
        params["cancel_check"] = self.isInterruptionRequested
        # Duck-typing: el motor debe exponer generate_image() (ImageEngine).
        return str(self._engine.generate_image(**params))


class ImageWidget(TaskWidget):
    """Formulario de generación con preview de la imagen resultante."""

    def __init__(
        self,
        engines: dict[str, Engine | None] | None,
        config_manager=None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._engines: dict[str, Engine | None] = engines or {}
        self._engine: Engine | None = None
        self._model_engine: dict[str, str] = {}
        self._config_manager = config_manager
        self._prefs = prefs.PrefsStore(config_manager)
        self._worker: ImageWorker | None = None
        self._credits_task = None
        self._elapsed = 0
        self._last_path: str | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick)
        self._build_ui()
        self._populate_models()
        if not any(e is not None for e in self._engines.values()):
            self._set_status("No hay motor de imagen configurado o habilitado.")

    def _build_ui(self) -> None:
        self._prompt = QPlainTextEdit()
        self._prompt.setPlaceholderText("Describe la imagen...")

        self._negative_prompt = QLineEdit()
        self._negative_prompt.setPlaceholderText("(opcional)")

        self._ref_image = QLineEdit()
        self._ref_image.setPlaceholderText("Imagen de referencia (edición)...")
        self._ref_image.setReadOnly(True)
        self._ref_button = QPushButton("Examinar...")
        self._ref_clear_button = QPushButton("Quitar")

        self._model = QComboBox()
        self._model.currentIndexChanged.connect(self._on_model_changed)

        self._width = QSpinBox()
        self._width.setRange(256, 2048)
        self._width.setValue(1024)
        self._height = QSpinBox()
        self._height.setRange(256, 2048)
        self._height.setValue(1024)

        self._preset_buttons: dict[str, QPushButton] = {}
        for key, (label, _w, _h, tooltip) in ip.PRESETS.items():
            btn = QPushButton(label)
            btn.setToolTip(tooltip)
            btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            btn.clicked.connect(lambda _checked=False, k=key: self._apply_preset(k))
            self._preset_buttons[key] = btn

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
        self._cancel_button = QPushButton("Cancelar")
        self._cancel_button.setEnabled(False)
        self._save_button = QPushButton("Guardar como...")
        self._save_button.setEnabled(False)

        self._status = QLabel("")
        self._preview = QLabel("La imagen aparecerá aquí.")
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setMinimumSize(480, 320)

        presets_row = QHBoxLayout()
        for key in ip.PRESETS:
            presets_row.addWidget(self._preset_buttons[key])
        presets_row.addStretch(1)

        form = QFormLayout()
        form.addRow("Modelo:", self._model)
        self._credits_label = QLabel("")
        form.addRow("Créditos:", self._credits_label)
        form.addRow("Prompt:", self._prompt)
        form.addRow("Prompt negativo:", self._negative_prompt)

        ref_row = QHBoxLayout()
        ref_row.addWidget(self._ref_image)
        ref_row.addWidget(self._ref_button)
        ref_row.addWidget(self._ref_clear_button)
        form.addRow("Referencia:", ref_row)

        form.addRow("Ancho:", self._width)
        form.addRow("Alto:", self._height)
        form.addRow("Presets:", presets_row)
        form.addRow("Steps:", self._steps)
        form.addRow("Guidance:", self._guidance)

        seed_row = QHBoxLayout()
        seed_row.addWidget(self._seed)
        seed_row.addWidget(self._random_seed)
        form.addRow("Seed:", seed_row)

        buttons = QHBoxLayout()
        buttons.addWidget(self._generate_button)
        buttons.addWidget(self._cancel_button)
        buttons.addWidget(self._save_button)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(buttons)
        layout.addWidget(self._status)
        layout.addWidget(self._preview)

        self._generate_button.clicked.connect(self._generate)
        self._cancel_button.clicked.connect(self._cancel)
        self._save_button.clicked.connect(self._save_as)
        self._ref_button.clicked.connect(self._pick_ref_image)
        self._ref_clear_button.clicked.connect(self._clear_ref_image)

    def set_engines(self, engines: dict[str, Engine | None]) -> None:
        """Actualiza los motores disponibles (ComfyUI local + BFL de pago)."""
        self._engines = engines or {}
        self._populate_models()

    def _populate_models(self) -> None:
        """Rellena el selector con los modelos de todos los motores."""
        self._model.blockSignals(True)
        self._model.clear()
        self._model_engine = {}
        for engine_id, engine in self._engines.items():
            if engine is None:
                continue
            for model_id in engine.list_models():
                self._model.addItem(ip.display_model(model_id, engine_id), model_id)
                self._model_engine[model_id] = engine_id
        last = self._prefs.get_str(prefs.IMAGE)
        index = self._model.findData(last)
        if index >= 0:
            self._model.setCurrentIndex(index)
        self._model.blockSignals(False)
        self._update_ref_enabled()

    def _on_model_changed(self) -> None:
        self._prefs.set(prefs.IMAGE, self._model.currentData())
        self._update_ref_enabled()
        self._refresh_credits()

    def _current_engine(self) -> Engine | None:
        model = self._model.currentData()
        engine_id = self._model_engine.get(model)
        return self._engines.get(engine_id)

    def _refresh_credits(self) -> None:
        """Consulta los créditos de la API de pago en segundo plano (M5)."""
        engine = self._current_engine()
        get_credits = (
            getattr(engine, "get_credits", None) if engine is not None else None
        )
        if getattr(engine, "id", "") != "bfl" or not callable(get_credits):
            self._credits_label.setText("")
            return
        self._credits_label.setText("Consultando…")
        self._credits_task = run_pool_task(
            get_credits, on_result=self._on_credits, on_error=self._on_credits_error
        )

    def _on_credits_error(self, message: str) -> None:
        logger.debug("No se pudieron consultar los créditos: %s", message)
        self._credits_label.setText("No disponible")

    def _on_credits(self, credits: object) -> None:
        if credits is None:
            self._credits_label.setText("No disponible")
        else:
            self._credits_label.setText(
                f"{float(credits):,.1f} créditos (≈ ${float(credits) / 100:.2f})"
            )

    def _update_ref_enabled(self, busy: bool = False) -> None:
        model = self._model.currentData()
        required = ip.requires_reference(model)
        optional = ip.optional_reference(model)
        enabled = (not busy) and (required or optional)
        self._ref_image.setEnabled(enabled)
        self._ref_button.setEnabled(enabled)
        self._ref_clear_button.setEnabled(enabled)
        if required:
            self._ref_image.setPlaceholderText("Imagen de referencia (obligatoria)...")
        elif optional:
            self._ref_image.setPlaceholderText(
                "Imagen de referencia (opcional): con foto edita, sin foto crea desde cero..."
            )
        else:
            self._ref_image.setPlaceholderText("Imagen de referencia (no aplica a este modelo)")

    def _pick_ref_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Selecciona la imagen de referencia", "", "Imágenes (*.png *.jpg *.jpeg)"
        )
        if path:
            self._ref_image.setText(path)

    def _clear_ref_image(self) -> None:
        self._ref_image.clear()

    def _apply_preset(self, key: str) -> None:
        label, width, height, _tooltip = ip.PRESETS[key]
        self._width.setValue(width)
        self._height.setValue(height)
        self._set_status(f"Preset aplicado: {label}")

    def _generate(self) -> None:
        model = self._model.currentData()
        engine = self._engines.get(self._model_engine.get(model))
        if engine is None:
            self._set_status("Error: no hay motor configurado para este modelo.")
            return
        prompt = self._prompt.toPlainText().strip()
        if not prompt:
            self._set_status("Escribe un prompt.")
            return
        if self._worker is not None and self._worker.isRunning():
            return

        ref_path = self._ref_image.text().strip()
        if ip.requires_reference(model) and not ref_path:
            self._set_status("Selecciona una imagen de referencia para este modelo.")
            return

        seed = None
        if not self._random_seed.isChecked():
            try:
                seed = ip.parse_seed(self._seed.text())
            except ValueError:
                self._set_status("Seed inválido: usa un número entero o marca 'Aleatorio'.")
                return
        params = ip.build_generate_params(
            prompt=prompt,
            negative_prompt=self._negative_prompt.text().strip(),
            width=self._width.value(),
            height=self._height.value(),
            steps=self._steps.value(),
            guidance=self._guidance.value(),
            model=model,
            ref_path=ref_path,
            seed=seed,
        )

        self._engine = engine
        self._set_busy(True)
        self._elapsed = 0
        self._set_status("Generando... 0s")

        self._worker = ImageWorker(engine, params, self)
        self._worker.result.connect(self._on_image_ready)
        self._worker.failed.connect(self._on_error)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()
        self._timer.start()

    def _tick(self) -> None:
        self._elapsed += 1
        status = f"Generando... {self._elapsed}s"
        if self._is_paid_engine() and self._elapsed >= 30 and self._elapsed % 60 == 30:
            status += " (FLUX.2 Pro es lento: puede tardar varios minutos, es normal)"
        self._set_status(status)

    def _on_image_ready(self, path: str) -> None:
        self._last_path = path
        self._show_image(path)
        self._save_button.setEnabled(True)
        self._set_status(f"Listo: {path}")

    def _on_finished(self) -> None:
        self._timer.stop()
        super()._on_finished()

    def _is_paid_engine(self) -> bool:
        """True si el motor actual es de pago (Black Forest Labs)."""
        return getattr(self._engine, "id", "") == "bfl"

    def _cancel(self) -> None:
        if self._worker is None or not self._worker.isRunning():
            return
        if self._is_paid_engine():
            answer = QMessageBox.question(
                self,
                "Cancelar generación de pago",
                "La solicitud ya se envió a la API de pago (Black Forest Labs). "
                "Cancelar aquí NO detiene la generación en sus servidores y es "
                "posible que se consuman créditos igualmente.\n\n"
                "¿Quieres cancelar la espera?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self._worker.requestInterruption()
        cancel_remote = getattr(self._engine, "cancel_remote", None)
        if callable(cancel_remote):
            cancel_remote()
        self._set_status("Cancelando...")

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
        target, _ = QFileDialog.getSaveFileName(
            self, "Guardar imagen como...", self._last_path, "PNG (*.png)"
        )
        if not target:
            return
        try:
            shutil.copyfile(self._last_path, target)
        except OSError as exc:
            QMessageBox.critical(self, "Error", f"No se pudo guardar la imagen: {exc}")
            return
        self._set_status(f"Imagen guardada en: {target}")

    def _busy_widgets(self) -> tuple:
        return (
            self._generate_button,
            self._prompt,
            self._negative_prompt,
            self._model,
            self._width,
            self._height,
            self._steps,
            self._guidance,
            self._random_seed,
            *self._preset_buttons.values(),
        )

    def _busy_only_widgets(self) -> tuple:
        return (self._cancel_button,)

    def _on_busy_changed(self, busy: bool) -> None:
        # El campo de seed solo se habilita si no está marcado "Aleatorio".
        self._seed.setEnabled(not busy and not self._random_seed.isChecked())
        self._update_ref_enabled(busy=busy)

    def _focus_widget(self):
        """El prompt es el control que debe recuperar el foco."""
        return self._prompt