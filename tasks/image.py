"""Tarea de imagen: envuelve el ImageWidget y lo expone como tarea registrable."""

import logging

from PyQt6.QtWidgets import QWidget

from core.engine_manager import EngineManager
from tasks.base_task import Task
from ui.image_widget import ImageWidget

logger = logging.getLogger(__name__)


class ImageTask(Task):
    """Tarea de generación de imágenes con el motor ComfyUI local.

    Construye un :class:`ui.image_widget.ImageWidget` con el motor ComfyUI
    actual y lo expone como widget de tarea.
    """

    def __init__(self, engine_manager: EngineManager) -> None:
        self._engine_manager = engine_manager
        self._widget: ImageWidget | None = None

    @property
    def id(self) -> str:
        return "image"

    @property
    def nombre(self) -> str:
        return "Imagen"

    def build_widget(self, parent: QWidget | None = None) -> QWidget:
        engine = self._engine_manager.get_engine("comfyui")
        self._widget = ImageWidget(engine, self._engine_manager.config_manager, parent)
        return self._widget

    def refresh_engine(self) -> None:
        """Relee el motor ComfyUI y lo propaga al widget de imagen.

        Método concreto de esta tarea (no forma parte de la interfaz
        :class:`Task`), simétrico a :meth:`ChatTask.refresh_engine`.
        """
        if self._widget is not None:
            self._widget.set_engine(self._engine_manager.get_engine("comfyui"))
