"""Widget de chat mínimo: entrada del usuario y respuesta del motor local."""

import logging

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
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

    def __init__(self, engine: Engine, messages: list[dict], parent=None) -> None:
        super().__init__(parent)
        self._engine = engine
        self._messages = messages

    def run(self) -> None:
        try:
            reply = self._engine.chat(self._messages)
        except EngineError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001 - red/timeout/JSON inesperados
            logger.debug("Error inesperado en ChatWorker: %r", exc, exc_info=True)
            self.failed.emit(f"Error inesperado: {exc}")
        else:
            self.reply_ready.emit(reply)


class ChatWidget(QWidget):
    """Área de conversación, indicador de actividad, input y botón Enviar."""

    def __init__(self, engine: Engine, parent=None) -> None:
        super().__init__(parent)
        self._engine: Engine | None = engine
        # TODO(fase 5): limitar el historial a N mensajes.
        self._messages: list[dict] = []
        self._worker: ChatWorker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
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

        self._worker = ChatWorker(self._engine, list(self._messages), self)
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
        self._input.setEnabled(not busy)
        self._send_button.setEnabled(not busy)
