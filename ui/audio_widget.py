"""Widget de audio: texto a voz con Piper TTS y reproductor embebido."""

import logging
import os
import shutil
import subprocess
from pathlib import Path

from PyQt6.QtCore import QThread, QUrl, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

try:
    from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer

    _HAS_QT_MULTIMEDIA = True
except ImportError:  # pragma: no cover - QtMultimedia opcional
    QAudioOutput = None
    QMediaPlayer = None
    _HAS_QT_MULTIMEDIA = False

try:
    import imageio_ffmpeg
except ImportError:  # pragma: no cover - dependencia declarada en requirements.txt
    imageio_ffmpeg = None

from engines.base_engine import Engine, EngineError

logger = logging.getLogger(__name__)

# Rango y valor por defecto del control de velocidad (length_scale de Piper).
_SPEED_MIN = 0.5
_SPEED_MAX = 2.0
_SPEED_DEFAULT = 1.0


class AudioWorker(QThread):
    """Ejecuta ``engine.synthesize()`` fuera del hilo de la UI."""

    audio_ready = pyqtSignal(str)   # ruta del WAV generado
    failed = pyqtSignal(str)

    def __init__(self, engine, text, voice, length_scale, parent=None) -> None:
        super().__init__(parent)
        self._engine = engine
        self._text = text
        self._voice = voice
        self._length_scale = length_scale

    def run(self) -> None:
        try:
            path = self._engine.synthesize(
                self._text, self._voice, self._length_scale
            )
        except EngineError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001 - red/IO inesperados
            logger.debug("Error inesperado en AudioWorker: %r", exc, exc_info=True)
            self.failed.emit(f"Error inesperado: {exc}")
        else:
            self.audio_ready.emit(str(path))


class AudioWidget(QWidget):
    """Formulario de texto a voz con reproductor embebido y guardado WAV/MP3."""

    def __init__(
        self, engine: Engine | None, config_manager=None, parent=None
    ) -> None:
        super().__init__(parent)
        self._engine = engine
        self._config_manager = config_manager
        self._worker: AudioWorker | None = None
        self._last_path: str | None = None
        self._build_ui()
        self._setup_player()
        self._populate_voices()
        if engine is None:
            self._set_status("Piper TTS no está configurado o deshabilitado.")
            self._generate_button.setEnabled(False)

    def _build_ui(self) -> None:
        self._text_area = QPlainTextEdit()
        self._text_area.setPlaceholderText(
            "Escribe el texto que quieres convertir a voz..."
        )

        self._voice = QComboBox()
        self._voice.currentIndexChanged.connect(self._on_voice_changed)

        self._speed = QDoubleSpinBox()
        self._speed.setRange(_SPEED_MIN, _SPEED_MAX)
        self._speed.setSingleStep(0.1)
        self._speed.setValue(_SPEED_DEFAULT)
        self._speed.setSuffix(" ×")

        form = QFormLayout()
        form.addRow("Voz:", self._voice)
        form.addRow("Velocidad:", self._speed)

        self._generate_button = QPushButton("Generar voz")
        self._play_button = QPushButton("Reproducir")
        self._play_button.setEnabled(False)
        self._open_button = QPushButton("Abrir con el reproductor del sistema")
        self._open_button.setEnabled(False)
        self._save_button = QPushButton("Guardar como...")
        self._save_button.setEnabled(False)

        buttons = QHBoxLayout()
        buttons.addWidget(self._generate_button)
        buttons.addWidget(self._play_button)
        buttons.addWidget(self._open_button)
        buttons.addWidget(self._save_button)

        self._status = QLabel("")

        layout = QVBoxLayout(self)
        layout.addWidget(self._text_area, 1)
        layout.addLayout(form)
        layout.addLayout(buttons)
        layout.addWidget(self._status)

        self._generate_button.clicked.connect(self._generate)
        self._play_button.clicked.connect(self._toggle_play)
        self._open_button.clicked.connect(self._open_system)
        self._save_button.clicked.connect(self._save_as)

    def _setup_player(self) -> None:
        self._player = None
        self._audio_output = None
        if not _HAS_QT_MULTIMEDIA:
            return
        try:
            self._audio_output = QAudioOutput(self)
            self._player = QMediaPlayer(self)
            self._player.setAudioOutput(self._audio_output)
            self._player.playbackStateChanged.connect(self._on_playback_state)
        except Exception as exc:  # noqa: BLE001 - backend de audio no disponible
            logger.warning("No se pudo crear el reproductor embebido: %s", exc)
            self._player = None
            self._audio_output = None

    def _on_playback_state(self, state) -> None:
        playing = state == QMediaPlayer.PlaybackState.PlayingState
        self._play_button.setText("Pausa" if playing else "Reproducir")

    # ---- Voces y persistencia ------------------------------------

    def _populate_voices(self) -> None:
        self._voice.blockSignals(True)
        self._voice.clear()
        voices: list[str] = []
        if self._engine is not None:
            try:
                voices = list(self._engine.list_voices())
            except EngineError as exc:
                logger.warning("No se pudieron listar las voces: %s", exc)
                self._set_status(f"Piper TTS no disponible: {exc}")
            except Exception as exc:  # noqa: BLE001
                logger.debug("Error listando voces: %r", exc, exc_info=True)
        for voice_name in voices:
            self._voice.addItem(voice_name, voice_name)
        last = None
        if self._config_manager is not None:
            last = self._config_manager.load_last_models().get("audio")
        if last in voices:
            self._voice.setCurrentIndex(voices.index(last))
        self._voice.blockSignals(False)

    def _on_voice_changed(self) -> None:
        if self._config_manager is not None and self._voice.currentData():
            self._config_manager.save_last_model("audio", self._voice.currentData())

    # ---- Generación ----------------------------------------------

    def _generate(self) -> None:
        if self._engine is None:
            self._set_status("Error: no hay motor Piper TTS configurado.")
            return
        text = self._text_area.toPlainText().strip()
        if not text:
            self._set_status("Escribe algún texto antes de generar.")
            return
        if self._worker is not None and self._worker.isRunning():
            return

        voice = self._voice.currentData() or None
        length_scale = self._speed.value()

        self._set_busy(True)
        self._set_status("Generando voz...")

        self._worker = AudioWorker(self._engine, text, voice, length_scale, self)
        self._worker.audio_ready.connect(self._on_audio_ready)
        self._worker.failed.connect(self._on_error)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _on_audio_ready(self, path: str) -> None:
        self._last_path = path
        self._play_button.setEnabled(True)
        self._open_button.setEnabled(True)
        self._save_button.setEnabled(True)
        if self._player is not None:
            self._player.setSource(QUrl.fromLocalFile(path))
        self._set_status(f"Listo: {path}")

    def _on_error(self, message: str) -> None:
        self._set_status(f"Error: {message}")

    def _on_finished(self) -> None:
        self._set_busy(False)
        self._worker.deleteLater()
        self._worker = None

    # ---- Reproductor ---------------------------------------------

    def _toggle_play(self) -> None:
        if self._player is None or not self._last_path:
            return
        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
        else:
            self._player.play()

    def _open_system(self) -> None:
        if not self._last_path:
            return
        startfile = getattr(os, "startfile", None)
        if startfile is None:
            self._set_status("Abrir con el reproductor del sistema no está soportado.")
            return
        try:
            startfile(self._last_path)
        except OSError as exc:
            self._set_status(f"No se pudo abrir el reproductor: {exc}")

    # ---- Guardado -------------------------------------------------

    def _save_as(self) -> None:
        if not self._last_path:
            return
        default = str(Path(self._last_path).with_suffix(".wav"))
        target, _selected = QFileDialog.getSaveFileName(
            self, "Guardar audio como...", default, "WAV (*.wav);;MP3 (*.mp3)"
        )
        if not target:
            return
        try:
            if target.lower().endswith(".mp3"):
                self._convert_to_mp3(self._last_path, target)
            else:
                shutil.copyfile(self._last_path, target)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Error", f"No se pudo guardar el audio: {exc}")
            return
        self._set_status(f"Guardado en: {target}")

    @staticmethod
    def _convert_to_mp3(wav_path: str, mp3_path: str) -> None:
        if imageio_ffmpeg is None:
            raise EngineError(
                "imageio-ffmpeg no está instalado. "
                "Instálalo con: pip install imageio-ffmpeg>=0.6.0"
            )
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run(
            [
                ffmpeg, "-y", "-i", str(wav_path),
                "-codec:a", "libmp3lame", "-q:a", "2", str(mp3_path),
            ],
            check=True,
            capture_output=True,
        )

    # ---- Utilidades ----------------------------------------------

    def _set_busy(self, busy: bool) -> None:
        self._generate_button.setEnabled(not busy and self._engine is not None)
        self._text_area.setEnabled(not busy)
        self._voice.setEnabled(not busy)
        self._speed.setEnabled(not busy)

    def _set_status(self, text: str) -> None:
        self._status.setText(text)

    def set_engine(self, engine: Engine | None) -> None:
        self._engine = engine
        self._populate_voices()
        if engine is None:
            self._set_status("Piper TTS no está configurado o deshabilitado.")
        self._generate_button.setEnabled(engine is not None)
