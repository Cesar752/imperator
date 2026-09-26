"""Diálogo de gestión de motores locales (lista + CRUD + checks en paralelo)."""

import json
import logging

from PyQt6.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from core.engine_manager import EngineManager
from engines.base_engine import Engine

logger = logging.getLogger(__name__)

COL_NOMBRE = 0
COL_ID = 1
COL_ACTIVO = 2
COL_CONEXION = 3

# Plantilla de ``parametros`` por tipo de motor, usada al AÑADIR uno nuevo para
# no partir de un dict vacío (evita configuraciones incompletas imposibles de
# construir, p. ej. ComfyUI sin workflows). El usuario puede editarla como JSON.
_TYPE_TEMPLATES: dict[str, dict] = {
    "ollama": {"model": "qwen3.5:9b"},
    "ollama_vision": {"model": "qwen3-vl:8b"},
    "comfyui": {
        "workflows": {"flux_dev": "C:/CComfyUI/flux_dev_workflow_api.json"},
        "default_model": "flux_dev",
    },
    "bfl": {"models": ["flux-kontext-pro", "flux-kontext-max", "flux-2-pro"]},
    "wanvideo": {
        "workflows": {
            "t2v": "C:/CComfyUI/wan22_t2v_workflow_api.json",
            "i2v": "C:/CComfyUI/wan22_i2v_workflow_api.json",
        },
        "default_model": "t2v",
    },
    "piper_tts": {"voices_dir": "voices", "default_voice": ""},
    "kokoro_tts": {"default_voice": "ef_dora"},
    "whisper": {"model": "small", "device": "cpu", "compute_type": "int8"},
}

# URL (ruta) por defecto al añadir un motor de cada tipo.
_TYPE_URLS: dict[str, str] = {
    "ollama": "http://localhost:11434",
    "ollama_vision": "http://localhost:11434",
    "comfyui": "http://127.0.0.1:8188",
    "bfl": "https://api.bfl.ai",
    "wanvideo": "http://127.0.0.1:8188",
    "piper_tts": "voices",
    "kokoro_tts": "local",
    "whisper": "local",
}


class _WorkerSignals(QObject):
    availability = pyqtSignal(str, bool)
    models = pyqtSignal(str, list)
    error = pyqtSignal(str, str)


class _AvailabilityWorker(QRunnable):
    """Comprueba ``engine.is_available()`` fuera del hilo de la UI."""

    def __init__(self, engine_id: str, engine: Engine) -> None:
        super().__init__()
        self._engine_id = engine_id
        self._engine = engine
        self.signals = _WorkerSignals()

    def run(self) -> None:
        try:
            ok = self._engine.is_available()
        except Exception as exc:  # noqa: BLE001 - red/timeout inesperados
            self.signals.error.emit(self._engine_id, str(exc))
            return
        self.signals.availability.emit(self._engine_id, ok)


class _ModelsWorker(QRunnable):
    """Obtiene ``engine.list_models()`` fuera del hilo de la UI."""

    def __init__(self, engine_id: str, engine: Engine) -> None:
        super().__init__()
        self._engine_id = engine_id
        self._engine = engine
        self.signals = _WorkerSignals()

    def run(self) -> None:
        try:
            models = self._engine.list_models()
        except Exception as exc:  # noqa: BLE001 - red/timeout inesperados
            self.signals.error.emit(self._engine_id, str(exc))
            return
        self.signals.models.emit(self._engine_id, models)


class EngineFormDialog(QDialog):
    """Formulario para añadir o editar un motor.

    El ``id`` (tipo) es inmutable al editar. Los ``parametros`` se editan como
    JSON para no perder campos específicos de cada motor (workflows, device,
    voces...), que un formulario plano destruiría.
    """

    def __init__(
        self,
        parent=None,
        engine_config: dict | None = None,
        supported_types: list[str] | None = None,
    ) -> None:
        super().__init__(parent)
        self._is_edit = engine_config is not None
        types = supported_types or list(_TYPE_TEMPLATES)
        self.setWindowTitle("Editar motor" if self._is_edit else "Añadir motor")

        form = QFormLayout(self)

        self._type_combo = QComboBox()
        if self._is_edit:
            # Al editar, el tipo queda fijo: solo el id original, deshabilitado.
            self._type_combo.addItem(engine_config["id"])
            self._type_combo.setEnabled(False)
        else:
            self._type_combo.addItems(types)
            self._type_combo.currentIndexChanged.connect(self._on_type_changed)

        self._nombre_edit = QLineEdit()
        self._ruta_edit = QLineEdit()
        self._enabled_check = QCheckBox("Habilitado")

        self._params_edit = QPlainTextEdit()
        self._params_edit.setPlaceholderText('{"clave": "valor"}')
        self._params_edit.setMinimumHeight(140)
        self._params_hint = QLabel(
            "Parámetros del motor en JSON (modelos, workflows, voces, device...)."
        )
        self._params_hint.setWordWrap(True)

        if self._is_edit:
            self._nombre_edit.setText(engine_config.get("nombre", ""))
            self._ruta_edit.setText(engine_config.get("ruta", ""))
            self._enabled_check.setChecked(engine_config.get("enabled", True))
            self._params_edit.setPlainText(
                json.dumps(
                    engine_config.get("parametros", {}),
                    indent=2,
                    ensure_ascii=False,
                )
            )
        else:
            self._enabled_check.setChecked(False)
            self._apply_type_defaults()

        form.addRow("Tipo:", self._type_combo)
        form.addRow("Nombre:", self._nombre_edit)
        form.addRow("URL (ruta):", self._ruta_edit)
        form.addRow("", self._enabled_check)
        form.addRow("Parámetros (JSON):", self._params_hint)
        form.addRow("", self._params_edit)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _engine_type(self) -> str:
        return self._type_combo.currentText()

    def _apply_type_defaults(self) -> None:
        """Rellena URL/parametros con la plantilla del tipo seleccionado."""
        engine_type = self._engine_type()
        if not self._ruta_edit.text().strip():
            self._ruta_edit.setText(_TYPE_URLS.get(engine_type, ""))
        template = _TYPE_TEMPLATES.get(engine_type, {})
        self._params_edit.setPlainText(
            json.dumps(template, indent=2, ensure_ascii=False)
        )

    def _on_type_changed(self) -> None:
        # Solo en alta: al cambiar de tipo se refrescan la URL y los parámetros.
        if not self._is_edit:
            self._ruta_edit.clear()
            self._apply_type_defaults()

    def _parsed_params(self) -> dict | None:
        raw = self._params_edit.toPlainText().strip()
        if not raw:
            return {}
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            QMessageBox.warning(
                self,
                "Parámetros inválidos",
                f"Los parámetros no son JSON válido:\n{exc}",
            )
            return None
        if not isinstance(data, dict):
            QMessageBox.warning(
                self,
                "Parámetros inválidos",
                "Los parámetros deben ser un objeto JSON ({...}).",
            )
            return None
        return data

    def _on_accept(self) -> None:
        if not self._nombre_edit.text().strip():
            QMessageBox.warning(self, "Datos incompletos", "El nombre no puede estar vacío.")
            return
        if not self._ruta_edit.text().strip():
            QMessageBox.warning(self, "Datos incompletos", "La URL (ruta) no puede estar vacía.")
            return
        if self._parsed_params() is None:
            return
        self.accept()

    def result_config(self) -> dict:
        return {
            "id": self._engine_type(),
            "nombre": self._nombre_edit.text().strip(),
            "ruta": self._ruta_edit.text().strip(),
            "enabled": self._enabled_check.isChecked(),
            "parametros": self._parsed_params() or {},
        }


class EnginesDialog(QDialog):
    """Lista los motores configurados y permite gestionarlos."""

    def __init__(self, engine_manager: EngineManager, parent=None) -> None:
        super().__init__(parent)
        self._manager = engine_manager
        self.setWindowTitle("Motores")
        self.resize(640, 380)
        self._row_to_id: dict[int, str] = {}
        self._build_ui()
        self._populate()
        self._check_all_availability()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["Nombre", "ID", "Activo", "Conexión"])
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(COL_NOMBRE, QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setVisible(False)
        layout.addWidget(self._table)

        buttons = QHBoxLayout()
        self._add_btn = QPushButton("Añadir")
        self._edit_btn = QPushButton("Editar")
        self._delete_btn = QPushButton("Eliminar")
        self._toggle_btn = QPushButton("Activar/Desactivar")
        self._test_btn = QPushButton("Probar conexión")
        self._models_btn = QPushButton("Ver modelos")
        for btn in (
            self._add_btn,
            self._edit_btn,
            self._delete_btn,
            self._toggle_btn,
            self._test_btn,
            self._models_btn,
        ):
            buttons.addWidget(btn)
        layout.addLayout(buttons)

        self._status = QLabel("")
        layout.addWidget(self._status)

        self._add_btn.clicked.connect(self._on_add)
        self._edit_btn.clicked.connect(self._on_edit)
        self._delete_btn.clicked.connect(self._on_delete)
        self._toggle_btn.clicked.connect(self._on_toggle_enabled)
        self._test_btn.clicked.connect(self._on_test_connection)
        self._models_btn.clicked.connect(self._on_show_models)

    # ---- Poblado de la tabla ------------------------------------

    def _populate(self) -> None:
        self._table.setRowCount(0)
        self._row_to_id.clear()
        for config in self._manager.list_configs():
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._row_to_id[row] = config["id"]
            self._table.setItem(row, COL_NOMBRE, QTableWidgetItem(config.get("nombre", "")))
            self._table.setItem(row, COL_ID, QTableWidgetItem(config["id"]))
            self._table.setItem(
                row, COL_ACTIVO, QTableWidgetItem("Sí" if config.get("enabled", False) else "No")
            )
            self._table.setItem(row, COL_CONEXION, QTableWidgetItem("—"))

    def _row_for(self, engine_id: str) -> int | None:
        for row, eid in self._row_to_id.items():
            if eid == engine_id:
                return row
        return None

    def _set_connection(self, engine_id: str, text: str) -> None:
        row = self._row_for(engine_id)
        if row is not None:
            item = self._table.item(row, COL_CONEXION)
            if item is not None:
                item.setText(text)

    def _selected_id(self) -> str | None:
        row = self._table.currentRow()
        if row < 0:
            return None
        return self._row_to_id.get(row)

    # ---- Disponibilidad / modelos (asíncronos) ------------------

    def _check_all_availability(self) -> None:
        for config in self._manager.list_configs():
            engine_id = config["id"]
            instance = self._manager.get_instance(engine_id)
            if instance is None:
                self._set_connection(engine_id, "Error (no se pudo construir)")
                continue
            self._set_connection(engine_id, "Comprobando…")
            worker = _AvailabilityWorker(engine_id, instance)
            worker.signals.availability.connect(self._on_availability)
            worker.signals.error.connect(self._on_availability_error)
            QThreadPool.globalInstance().start(worker)

    def _on_availability(self, engine_id: str, ok: bool) -> None:
        if not self.isVisible():
            return
        self._set_connection(engine_id, "Conectado" if ok else "Desconectado")

    def _on_availability_error(self, engine_id: str, message: str) -> None:
        if not self.isVisible():
            return
        self._set_connection(engine_id, f"Error: {message}")

    # ---- Acciones (CRUD + pruebas) ------------------------------

    def _on_add(self) -> None:
        configured = {c["id"] for c in self._manager.list_configs()}
        available = [t for t in self._manager.supported_types() if t not in configured]
        if not available:
            QMessageBox.information(
                self,
                "Añadir motor",
                "Todos los tipos de motor soportados ya están configurados.",
            )
            return
        dialog = EngineFormDialog(self, supported_types=available)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                self._manager.add_engine(dialog.result_config())
            except Exception as exc:  # noqa: BLE001 - ConfigError / validación
                QMessageBox.critical(self, "Error", f"No se pudo añadir el motor: {exc}")
                return
            self._refresh()

    def _on_edit(self) -> None:
        engine_id = self._selected_id()
        if engine_id is None:
            QMessageBox.information(self, "Editar motor", "Selecciona un motor para editar.")
            return
        config = self._manager.get_config(engine_id)
        if config is None:
            return
        # Pasar supported_types=[engine_id] mantiene el tipo original en el combo
        # (antes quedaba en "ollama" y reescribía el id del motor editado).
        dialog = EngineFormDialog(
            self, engine_config=config, supported_types=[engine_id]
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                self._manager.update_engine(engine_id, dialog.result_config())
            except Exception as exc:  # noqa: BLE001 - ConfigError / validación
                QMessageBox.critical(self, "Error", f"No se pudo actualizar el motor: {exc}")
                return
            self._refresh()

    def _on_delete(self) -> None:
        engine_id = self._selected_id()
        if engine_id is None:
            QMessageBox.information(self, "Eliminar motor", "Selecciona un motor para eliminar.")
            return
        reply = QMessageBox.question(
            self,
            "Eliminar motor",
            f"¿Seguro que quieres eliminar el motor '{engine_id}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            self._manager.remove_engine(engine_id)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Error", f"No se pudo eliminar el motor: {exc}")
            return
        self._refresh()

    def _on_toggle_enabled(self) -> None:
        engine_id = self._selected_id()
        if engine_id is None:
            QMessageBox.information(self, "Activar/Desactivar", "Selecciona un motor.")
            return
        config = self._manager.get_config(engine_id)
        if config is None:
            return
        try:
            self._manager.set_enabled(engine_id, not config.get("enabled", False))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Error", f"No se pudo cambiar el estado: {exc}")
            return
        self._refresh()

    def _on_test_connection(self) -> None:
        engine_id = self._selected_id()
        if engine_id is None:
            QMessageBox.information(self, "Probar conexión", "Selecciona un motor.")
            return
        instance = self._manager.get_instance(engine_id)
        if instance is None:
            QMessageBox.warning(self, "Probar conexión", f"No se pudo construir el motor '{engine_id}'.")
            return
        self._set_connection(engine_id, "Comprobando…")
        self._status.setText(f"Probando conexión de '{engine_id}'…")
        worker = _AvailabilityWorker(engine_id, instance)
        worker.signals.availability.connect(self._on_test_result)
        worker.signals.error.connect(self._on_availability_error)
        QThreadPool.globalInstance().start(worker)

    def _on_test_result(self, engine_id: str, ok: bool) -> None:
        if not self.isVisible():
            return
        self._set_connection(engine_id, "Conectado" if ok else "Desconectado")
        self._status.setText("")
        if ok:
            QMessageBox.information(self, "Probar conexión", f"'{engine_id}' responde correctamente.")
        else:
            QMessageBox.warning(self, "Probar conexión", f"'{engine_id}' no responde.")

    def _on_show_models(self) -> None:
        engine_id = self._selected_id()
        if engine_id is None:
            QMessageBox.information(self, "Ver modelos", "Selecciona un motor.")
            return
        instance = self._manager.get_instance(engine_id)
        if instance is None:
            QMessageBox.warning(self, "Ver modelos", f"No se pudo construir el motor '{engine_id}'.")
            return
        self._status.setText(f"Obteniendo modelos de '{engine_id}'…")
        worker = _ModelsWorker(engine_id, instance)
        worker.signals.models.connect(self._on_models)
        worker.signals.error.connect(self._on_models_error)
        QThreadPool.globalInstance().start(worker)

    def _on_models(self, engine_id: str, models: list) -> None:
        if not self.isVisible():
            return
        self._status.setText("")
        if not models:
            QMessageBox.information(self, "Ver modelos", f"'{engine_id}' no expone modelos.")
            return
        QMessageBox.information(
            self,
            f"Modelos de '{engine_id}'",
            "\n".join(models),
        )

    def _on_models_error(self, engine_id: str, message: str) -> None:
        if not self.isVisible():
            return
        self._status.setText("")
        QMessageBox.warning(
            self,
            "Ver modelos",
            f"No se pudieron listar los modelos de '{engine_id}': {message}",
        )

    def _refresh(self) -> None:
        self._populate()
        self._check_all_availability()
