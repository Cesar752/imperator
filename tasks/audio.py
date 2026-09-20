"""Tarea de audio: texto a voz local con Piper TTS."""

import logging

from PyQt6.QtWidgets import QWidget

from core.engine_manager import EngineManager
from tasks.base_task import Task
from ui.audio_widget import AudioWidget

logger = logging.getLogger(__name__)


class AudioTask(Task):
    """Tarea de síntesis de voz con el motor Piper TTS local.

    Construye un :class:`ui.audio_widget.AudioWidget` con el motor
    ``piper_tts`` actual y lo expone como widget de tarea.
    """

    def __init__(self, engine_manager: EngineManager) -> None:
        self._engine_manager = engine_manager
        self._widget: AudioWidget | None = None

    @property
    def id(self) -> str:
        return "audio"

    @property
    def nombre(self) -> str:
        return "Audio"

    def build_widget(self, parent: QWidget | None = None) -> QWidget:
        engine = self._engine_manager.get_engine("piper_tts")
        self._widget = AudioWidget(engine, self._engine_manager.config_manager, parent)
        return self._widget

    def refresh_engine(self) -> None:
        """Relee el motor Piper TTS y lo propaga al widget de audio."""
        if self._widget is not None:
            self._widget.set_engine(self._engine_manager.get_engine("piper_tts"))
