"""Widget de chat mínimo: entrada del usuario y respuesta del motor local."""

import logging

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from engines.base_engine import Engine, EngineError

logger = logging.getLogger(__name__)


class ChatWorker(QThread):
    """Ejecuta ``engine.chat()`` fuera del hilo de la UI."""

    reply_ready = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(
        self, engine: Engine, messages: list[dict], model: str | None, parent=None
    ) -> None:
        super().__init__(parent)
        self._engine = engine
        self._messages = messages
        self._model = model

    def run(self) -> None:
        try:
            reply = self._engine.chat(self._messages, model=self._model)
        except EngineError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001 - red/timeout/JSON inesperados
            logger.debug("Error inesperado en ChatWorker: %r", exc, exc_info=True)
            self.failed.emit(f"Error inesperado: {exc}")
        else:
            self.reply_ready.emit(reply)


class ChatWidget(QWidget):
    """Área de conversación, indicador de actividad, input y botón Enviar."""

    def __init__(self, engine: Engine, config_manager=None, parent=None) -> None:
        super().__init__(parent)
        self._engine: Engine | None = engine
        self._config_manager = config_manager
        # TODO(fase 5): limitar el historial a N mensajes.
        self._messages: list[dict] = []
        self._worker: ChatWorker | None = None
        self._build_ui()
        self._populate_models()

    def _build_ui(self) -> None:
        self._model = QComboBox()
        self._model.currentIndexChanged.connect(self._on_model_changed)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Modelo:"))
        model_row.addWidget(self._model, 1)

        self._conversation = QPlainTextEdit()
        self._conversation.setReadOnly(True)
        self._conversation.setPlaceholderText("La conversación aparecerá aquí.")

        self._thinking_label = QLabel("Pensando...")
        self._thinking_label.setVisible(False)

        self._input = QLineEdit()
        self._input.setPlaceholderText("Escribe tu mensaje y pulsa Enter...")

        self._send_button = QPushButton("Enviar")

        input_row = QHBoxLayout()
        input_row.addWidget(self._input)
        input_row.addWidget(self._send_button)

        layout = QVBoxLayout(self)
        layout.addLayout(model_row)
        layout.addWidget(self._conversation)
        layout.addWidget(self._thinking_label)
        layout.addLayout(input_row)

        self._input.returnPressed.connect(self._send_message)
        self._send_button.clicked.connect(self._send_message)

    def set_engine(self, engine: Engine | None) -> None:
        """Cambia el motor activo. Limpia el historial y el área."""
        self._engine = engine
        self._messages.clear()
        self._conversation.clear()
        self._set_busy(False)
        self._populate_models()

    # ---- Modelos ------------------------------------------------

    def _populate_models(self) -> None:
        """Rellena el selector con los modelos de Ollama y restaura el último."""
        if self._engine is None:
            return
        details = self._fetch_model_details()
        names = [detail["name"] for detail in details]
        default = getattr(self._engine, "default_model", None)
        if default and default not in names:
            details.append({"name": default, "size": None, "installed": False})
            logger.warning("default_model '%s' no está instalado en Ollama.", default)

        self._model.blockSignals(True)
        self._model.clear()
        for detail in details:
            name = detail["name"]
            installed = detail.get("installed", True)
            size = detail.get("size")
            if installed:
                display = name
                tooltip = f"{name} ({self._format_size(size)})" if size else name
            else:
                display = f"{name} (no instalado)"
                tooltip = f"{name} (no instalado en Ollama)"
            self._model.addItem(display, name)
            self._model.setItemData(
                self._model.count() - 1, tooltip, Qt.ItemDataRole.ToolTipRole
            )

        last = None
        if self._config_manager is not None:
            last = self._config_manager.load_last_models().get("chat")
        if last is not None:
            idx = self._model.findData(last)
            if idx >= 0:
                self._model.setCurrentIndex(idx)
            else:
                logger.warning(
                    "Modelo guardado '%s' ya no existe en Ollama; "
                    "usando el primero del combo.",
                    last,
                )
        self._model.blockSignals(False)

    def _fetch_model_details(self) -> list[dict]:
        """Detalles de los modelos; lista vacía si falla la red."""
        if self._engine is None:
            return []
        try:
            return list(self._engine.list_model_details())
        except Exception as exc:  # noqa: BLE001 - red/timeout/JSON inesperados
            logger.debug("No se pudieron listar los modelos de Ollama: %r", exc)
            return []

    @staticmethod
    def _format_size(size: int) -> str:
        return f"{size / (1024 ** 3):.1f} GB"

    def _on_model_changed(self) -> None:
        model = self._model.currentData()
        if not model:
            return
        if self._config_manager is not None:
            self._config_manager.save_last_model("chat", model)
        if self._messages:
            answer = QMessageBox.question(
                self,
                "Cambiar de modelo",
                "¿Limpiar la conversación al cambiar de modelo?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if answer == QMessageBox.StandardButton.Yes:
                self._messages.clear()
                self._conversation.clear()

    def _send_message(self) -> None:
        if self._engine is None:
            self._append_line("Error: no hay motor activo.")
            return
        text = self._input.text().strip()
        if not text:
            return
        if self._worker is not None and self._worker.isRunning():
            return

        self._append_line(f"Tú: {text}")
        self._messages.append({"role": "user", "content": text})
        self._input.clear()
        self._input.setFocus()

        self._thinking_label.setVisible(True)
        self._set_busy(True)

        self._worker = ChatWorker(
            self._engine, list(self._messages), self._model.currentData(), self
        )
        self._worker.reply_ready.connect(self._on_reply)
        self._worker.failed.connect(self._on_error)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.start()

    def _on_reply(self, reply: str) -> None:
        self._messages.append({"role": "assistant", "content": reply})
        self._thinking_label.setVisible(False)
        self._set_busy(False)
        self._append_line(f"Ollama: {reply}")
        self._worker = None

    def _on_error(self, message: str) -> None:
        self._thinking_label.setVisible(False)
        self._set_busy(False)
        self._append_line(f"Error: {message}")
        self._worker = None

    def _append_line(self, text: str) -> None:
        self._conversation.appendPlainText(text)

    def _set_busy(self, busy: bool) -> None:
        self._model.setEnabled(not busy)
        self._input.setEnabled(not busy)
        self._send_button.setEnabled(not busy)
