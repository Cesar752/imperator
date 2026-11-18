"""Tarea **Crónica**: envuelve el ChronicleWidget (vídeo comentado por IA)."""

import logging

from PyQt6.QtWidgets import QWidget

from core.engine_manager import EngineManager
from tasks.base_task import Task
from ui.chronicle_widget import ChronicleWidget

logger = logging.getLogger(__name__)


class ChronicleTask(Task):
    """Crónica locutada de un vídeo: mira, cuenta, locuta y monta, todo en local.

    Reúne motores que ya existían sueltos (visión, chat, Whisper, TTS y MusicGen)
    y deja el resultado montado en un MP4 con voz, subtítulos y música.
    """

    #: El análisis y el guion se piden a Ollama: al abandonar la pestaña se libera
    #: su modelo de la VRAM, como en Chat, Documentos, Comprensión y Cuaderno.
    engine_id = "ollama"

    def __init__(self, engine_manager: EngineManager) -> None:
        self._engine_manager = engine_manager
        self._widget: ChronicleWidget | None = None

    @property
    def id(self) -> str:
        return "chronicle"

    @property
    def nombre(self) -> str:
        return "Crónica"

    def build_widget(self, parent: QWidget | None = None) -> QWidget:
        self._widget = ChronicleWidget(self._engine_manager, parent)
        return self._widget
