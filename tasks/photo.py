"""Tarea **Retoque**: envuelve el PhotoWidget (mejora y conversión de fotos)."""

import logging

from PyQt6.QtWidgets import QWidget

from core.engine_manager import EngineManager
from tasks.base_task import Task
from ui.photo_widget import PhotoWidget

logger = logging.getLogger(__name__)


class PhotoTask(Task):
    """Retoque de fotos: ajustes de color, mejora con IA y cambio de formato.

    Construye un :class:`ui.photo_widget.PhotoWidget` como pestaña propia (al lado
    de Imagen). No depende de un motor del registro: la mejora con IA la hace un
    worker por subprocess en ``venv-photo`` (Real-ESRGAN/GFPGAN).
    """

    def __init__(self, engine_manager: EngineManager) -> None:
        self._engine_manager = engine_manager
        self._widget: PhotoWidget | None = None

    @property
    def id(self) -> str:
        return "photo"

    @property
    def nombre(self) -> str:
        return "Retoque"

    def build_widget(self, parent: QWidget | None = None) -> QWidget:
        self._widget = PhotoWidget(parent)
        return self._widget
