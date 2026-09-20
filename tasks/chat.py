"""Tarea de chat: envuelve el ChatWidget y lo expone como tarea registrable."""

import logging

from PyQt6.QtWidgets import QWidget

from core.engine_manager import EngineManager
from tasks.base_task import Task
from ui.chat_widget import ChatWidget

logger = logging.getLogger(__name__)


class ChatTask(Task):
    """Tarea de conversación con el motor local (Ollama).

    Construye un :class:`ui.chat_widget.ChatWidget` con el motor Ollama
    actual y lo expone como widget de tarea.
    """

    def __init__(self, engine_manager: EngineManager) -> None:
        self._engine_manager = engine_manager
        self._widget: ChatWidget | None = None

    @property
    def id(self) -> str:
        return "chat"

    @property
    def nombre(self) -> str:
        return "Chat"

    def build_widget(self, parent: QWidget | None = None) -> QWidget:
        engine = self._engine_manager.get_engine("ollama")
        self._widget = ChatWidget(engine, self._engine_manager.config_manager, parent)
        return self._widget

    def refresh_engine(self) -> None:
        """Relee el motor Ollama y lo propaga al widget del chat.

        Método concreto de esta tarea (no forma parte de la interfaz
        :class:`Task`): se invoca cuando cambia la configuración de motores
        para que el chat siga usando el motor correcto.
        """
        if self._widget is not None:
            self._widget.set_engine(self._engine_manager.get_engine("ollama"))
