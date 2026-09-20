"""Tarea de documentos: OCR, resumen y extracción de tablas con Ollama Vision."""

import logging

from PyQt6.QtWidgets import QWidget

from core.engine_manager import EngineManager
from tasks.base_task import Task
from ui.document_widget import DocumentWidget

logger = logging.getLogger(__name__)


class DocumentTask(Task):
    """Tarea de lectura de documentos con el motor Ollama Vision local.

    Construye un :class:`ui.document_widget.DocumentWidget` con el motor
    ``ollama_vision`` actual y lo expone como widget de tarea.
    """

    def __init__(self, engine_manager: EngineManager) -> None:
        self._engine_manager = engine_manager
        self._widget: DocumentWidget | None = None

    @property
    def id(self) -> str:
        return "document"

    @property
    def nombre(self) -> str:
        return "Documentos"

    def build_widget(self, parent: QWidget | None = None) -> QWidget:
        engine = self._engine_manager.get_engine("ollama_vision")
        self._widget = DocumentWidget(
            engine, self._engine_manager.config_manager, parent
        )
        return self._widget

    def refresh_engine(self) -> None:
        """Relee el motor Ollama Vision y lo propaga al widget de documentos."""
        if self._widget is not None:
            self._widget.set_engine(
                self._engine_manager.get_engine("ollama_vision")
            )
