"""Tarea Ensamblador: combina MP4 + audio + subtítulos (Whisper) en un MP4 final."""

import logging

from PyQt6.QtWidgets import QWidget

from core.engine_manager import EngineManager
from tasks.base_task import Task
from ui.assembler_widget import AssemblerWidget

logger = logging.getLogger(__name__)


class AssemblerTask(Task):
    """Tarea de ensamblaje de vídeo (MP4 + audio + subtítulos).

    Construye un :class:`ui.assembler_widget.AssemblerWidget` con el motor
    Whisper actual (para los subtítulos) y lo expone como widget de tarea.
    """

    def __init__(self, engine_manager: EngineManager) -> None:
        self._engine_manager = engine_manager
        self._widget: AssemblerWidget | None = None

    @property
    def id(self) -> str:
        return "assembler"

    @property
    def nombre(self) -> str:
        return "Ensamblador"

    def build_widget(self, parent: QWidget | None = None) -> QWidget:
        engine = self._engine_manager.get_engine("whisper")
        self._widget = AssemblerWidget(
            engine, self._engine_manager.config_manager, parent
        )
        return self._widget

    def refresh_engine(self) -> None:
        """Relee el motor Whisper y lo propaga al widget del ensamblador."""
        if self._widget is not None:
            self._widget.set_engine(self._engine_manager.get_engine("whisper"))
