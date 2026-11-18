"""Tarea **Vídeo IA**: envuelve el VideoEnhanceWidget (retoque de vídeo)."""

import logging

from PyQt6.QtWidgets import QWidget

from core.engine_manager import EngineManager
from tasks.base_task import Task
from ui.video_enhance_widget import VideoEnhanceWidget

logger = logging.getLogger(__name__)


class VideoEnhanceTask(Task):
    """Reescalado, ajuste de color y cambio de fotogramas de un vídeo.

    La parte rápida (color/efectos/FPS) usa ffmpeg; la parte IA reescala con
    Real-ESRGAN fotograma a fotograma (worker en ``venv-photo``). No usa motores
    del registro, así que no libera VRAM al cambiar de pestaña.
    """

    def __init__(self, engine_manager: EngineManager) -> None:
        self._engine_manager = engine_manager
        self._widget: VideoEnhanceWidget | None = None

    @property
    def id(self) -> str:
        return "video_enhance"

    @property
    def nombre(self) -> str:
        return "Vídeo IA"

    def build_widget(self, parent: QWidget | None = None) -> QWidget:
        self._widget = VideoEnhanceWidget(parent)
        return self._widget
