"""Tarea de comprensión de vídeo: envuelve el UnderstandingWidget."""

import logging

from PyQt6.QtWidgets import QWidget

from core.engine_manager import EngineManager
from tasks.base_task import Task
from ui.understanding_widget import UnderstandingWidget

logger = logging.getLogger(__name__)


class UnderstandingTask(Task):
    """Analiza un vídeo (fotogramas + audio) y resume su contenido.

    Construye un :class:`ui.understanding_widget.UnderstandingWidget` y lo expone
    como widget de tarea.
    """

    #: El resumen final se pide a Ollama: al abandonar la pestaña se libera su
    #: modelo de la VRAM, igual que en Chat y Documentos.
    engine_id = "ollama"

    def __init__(self, engine_manager: EngineManager) -> None:
        self._engine_manager = engine_manager
        self._widget: UnderstandingWidget | None = None

    @property
    def id(self) -> str:
        return "understanding"

    @property
    def nombre(self) -> str:
        return "Comprensión"

    def build_widget(self, parent: QWidget | None = None) -> QWidget:
        self._widget = UnderstandingWidget(self._engine_manager, parent)
        return self._widget

    def refresh_engine(self) -> None:
        """Relee los motores (YOLO/CLIP/OCR/Whisper/Ollama) tras cambiar la config."""
        if self._widget is not None:
            self._widget.set_engine_manager(self._engine_manager)
