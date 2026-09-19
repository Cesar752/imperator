"""Tarea de vídeo: envuelve el VideoWidget y lo expone como tarea registrable."""

import logging

from PyQt6.QtWidgets import QWidget

from core.engine_manager import EngineManager
from tasks.base_task import Task
from ui.video_widget import VideoWidget

logger = logging.getLogger(__name__)


class VideoTask(Task):
    """Tarea de generación de vídeo con el motor Wan Video local.

    Construye un :class:`ui.video_widget.VideoWidget` con el motor WanVideo
    actual y lo expone como widget de tarea.
    """

    def __init__(self, engine_manager: EngineManager) -> None:
        self._engine_manager = engine_manager
        self._widget: VideoWidget | None = None

    @property
    def id(self) -> str:
        return "video"

    @property
    def nombre(self) -> str:
        return "Vídeo"

    def build_widget(self, parent: QWidget | None = None) -> QWidget:
        engine = self._engine_manager.get_engine("wanvideo")
        self._widget = VideoWidget(engine, self._engine_manager.config_manager, parent)
        return self._widget

    def refresh_engine(self) -> None:
        """Relee el motor WanVideo y lo propaga al widget de vídeo."""
        if self._widget is not None:
            self._widget.set_engine(self._engine_manager.get_engine("wanvideo"))
