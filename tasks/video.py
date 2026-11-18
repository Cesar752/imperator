"""Tarea de vídeo: envuelve el VideoWidget y lo expone como tarea registrable."""

import logging

from PyQt6.QtWidgets import QWidget

from core.engine_manager import EngineManager
from services import chat_engine as ce
from tasks.base_task import Task
from ui.video_widget import VideoWidget

logger = logging.getLogger(__name__)


class VideoTask(Task):
    """Tarea de generación de vídeo con el motor Wan Video local.

    Construye un :class:`ui.video_widget.VideoWidget` con el motor WanVideo
    actual y lo expone como widget de tarea. También le pasa los motores de
    imagen (ComfyUI local y BFL de pago) que usa el modo «Fotos primero»: cada
    escena se genera como foto y esa foto entra como primer fotograma del clip.
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

    def _image_engines(self) -> dict:
        """Motores de imagen disponibles para las fotos de escena."""
        return {
            "comfyui": self._engine_manager.get_engine("comfyui"),
            "bfl": self._engine_manager.get_engine("bfl"),
        }

    def build_widget(self, parent: QWidget | None = None) -> QWidget:
        engine = self._engine_manager.get_engine("wanvideo")
        # El motor de chat (Ollama) es opcional: solo se usa para generar el
        # guion de escenas. Sin él, la pestaña sigue generando clip a clip.
        chat = ce.preferred_chat_engine(self._engine_manager)
        self._widget = VideoWidget(
            engine,
            self._engine_manager.config_manager,
            parent,
            chat_engine=chat,
            image_engines=self._image_engines(),
        )
        return self._widget

    def refresh_engine(self) -> None:
        """Relee los motores de vídeo, chat e imagen y los propaga al widget."""
        if self._widget is not None:
            self._widget.set_engine(self._engine_manager.get_engine("wanvideo"))
            self._widget.set_chat_engine(ce.preferred_chat_engine(self._engine_manager))
            self._widget.set_engines(self._image_engines())
