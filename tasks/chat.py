"""Tarea de chat: envuelve el ChatWidget y lo expone como tarea registrable."""

import logging

from PyQt6.QtWidgets import QWidget

from core.engine_manager import EngineManager
from services import preferences as prefs
from tasks.base_task import Task
from ui.chat_widget import ChatWidget

logger = logging.getLogger(__name__)


class ChatTask(Task):
    """Tarea de conversación con el motor local (Ollama).

    Construye un :class:`ui.chat_widget.ChatWidget` con el motor Ollama
    actual y lo expone como widget de tarea. Recibe además el orquestador (si
    el ``TaskManager`` lo inyecta) para que el chat pueda desviar un mensaje a
    la tarea que le corresponde.
    """

    engine_id = "ollama"

    def __init__(
        self, engine_manager: EngineManager, orchestrator: object | None = None
    ) -> None:
        self._engine_manager = engine_manager
        self._orchestrator = orchestrator
        self._widget: ChatWidget | None = None

    @property
    def id(self) -> str:
        return "chat"

    @property
    def nombre(self) -> str:
        return "Chat"

    def build_widget(self, parent: QWidget | None = None) -> QWidget:
        config_manager = self._engine_manager.config_manager
        saved = prefs.PrefsStore(config_manager).get_str(prefs.CHAT_ENGINE) or "ollama"
        engine = self._engine_manager.get_engine(saved) or self._engine_manager.get_engine(
            "ollama"
        )
        self._widget = ChatWidget(
            engine,
            config_manager,
            parent,
            orchestrator=self._orchestrator,
            engine_manager=self._engine_manager,
        )
        return self._widget

    def refresh_engine(self) -> None:
        """Relee los motores de chat y los propaga al widget.

        Método concreto de esta tarea (no forma parte de la interfaz
        :class:`Task`): se invoca cuando cambia la configuración de motores
        para que el chat siga usando un motor válido.
        """
        if self._widget is not None:
            self._widget.set_engine_manager(self._engine_manager)
