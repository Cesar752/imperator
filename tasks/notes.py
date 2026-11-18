"""Tarea **Cuaderno**: envuelve el NotesWidget (análisis cinematográfico local)."""

import logging

from PyQt6.QtWidgets import QWidget

from core.engine_manager import EngineManager
from tasks.base_task import Task
from ui.notes_widget import NotesWidget

logger = logging.getLogger(__name__)


class NotesTask(Task):
    """Cuaderno de análisis cinematográfico asistido por IA, todo en local.

    Construye un :class:`ui.notes_widget.NotesWidget` como pestaña propia.
    """

    #: El análisis final se pide a Ollama: al abandonar la pestaña se libera su
    #: modelo de la VRAM, igual que en Chat, Documentos y Comprensión.
    engine_id = "ollama"

    def __init__(self, engine_manager: EngineManager) -> None:
        self._engine_manager = engine_manager
        self._widget: NotesWidget | None = None

    @property
    def id(self) -> str:
        return "notes"

    @property
    def nombre(self) -> str:
        return "Cuaderno"

    def build_widget(self, parent: QWidget | None = None) -> QWidget:
        self._widget = NotesWidget(self._engine_manager, parent)
        return self._widget

    def refresh_engine(self) -> None:
        """Relee los motores (Ollama/Vision/Whisper) tras cambiar la config."""
        if self._widget is not None:
            self._widget.set_engine_manager(self._engine_manager)
