"""Widget de audio: texto a voz (Piper / Kokoro) con reproductor embebido."""

import logging
import os
import shutil
import subprocess
from pathlib import Path

from PyQt6.QtCore import QUrl
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
from services import audio_params as ap
from services import preferences as prefs
from ui.task_widget import TaskWidget
from ui.workers import ThreadWorker

logger = logging.getLogger(__name__)

# Timeout de la conversión a MP3 (ffmpeg -q:a 2): holgado, pero finito, para que
# un ffmpeg atascado no bloquee el hilo del worker.
_MP3_TIMEOUT = 300


class AudioWorker(ThreadWorker):
    """Ejecuta ``engine.synthesize()`` fuera del hilo de la UI.

    Piper/Kokoro son bloqueantes (no se pueden interrumpir a mitad): la
    comprobación de cancelación antes/después la aporta ``ThreadWorker``.
    """

    def __init__(self, engine, text, voice, speed, parent=None) -> None:
        super().__init__(parent)
        self._engine = engine
        self._text = text
        self._voice = voice
        self._speed = speed

    def work(self) -> str:
        return str(self._engine.synthesize(self._text, self._voice, self._speed))


class AudioWidget(TaskWidget):
    """Formulario de texto a voz (Piper/Kokoro) con reproductor y guardado."""

    def __init__(
        self, piper_engine=None, kokoro_engine=None, config_manager=None, parent=None
    ) -> None:
        super().__init__(parent)
        self._engines = {"piper": piper_engine, "kokoro": kokoro_engine}
        self._engine = None
        self._current_motor = None
        self._config_manager = config_manager
        self._prefs = prefs.PrefsStore(config_manager)
        self._worker: AudioWorker | None = None
        self._last_path: str | None = None
        self._build_ui()
        self._setup_player()
        self._populate_motors()

    def _build_ui(self) -> None:
        self._text_area = QPlainTextEdit()
        self._text_area.setPlaceholderText(
            "Escribe el texto que quieres convertir a voz..."
        )

        self._motor = QComboBox()
        self._motor.setToolTip(ap.MOTOR_TOOLTIP)
        self._motor.currentIndexChanged.connect(self._on_motor_changed)

        self._voice = QComboBox()
        self._voice.currentIndexChanged.connect(self._on_voice_changed)

        self._speed = QDoubleSpinBox()
        self._speed.setRange(ap.SPEED_MIN, ap.SPEED_MAX)
        self._speed.setSingleStep(ap.SPEED_STEP)
        self._speed.setValue(ap.SPEED_DEFAULT)
        self._speed.setSuffix(" ×")

        form = QFormLayout()
        form.addRow("Motor:", self._motor)
        form.addRow("Voz:", self._voice)
        form.addRow("Velocidad:", self._speed)

        self._generate_button = QPushButton("Generar voz")
        self._cancel_button = QPushButton("Cancelar")
        self._cancel_button.setEnabled(False)
        self._play_button = QPushButton("Reproducir")
        self._play_button.setEnabled(False)
        self._open_button = QPushButton("Abrir con el reproductor del sistema")
        self._open_button.setEnabled(False)
        self._save_button = QPushButton("Guardar como...")
        self._save_button.setEnabled(False)

        buttons = QHBoxLayout()
        buttons.addWidget(self._generate_button)
        buttons.addWidget(self._cancel_button)
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
        self._cancel_button.clicked.connect(self._cancel)
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

    # ---- Motores y voces -----------------------------------------

    def _populate_motors(self) -> None:
        self._motor.blockSignals(True)
        self._motor.clear()
        for label, motor_id in ap.MOTORS:
            self._motor.addItem(label, motor_id)
        last = prefs.AudioPrefs.from_store(self._prefs).engine
        idx = self._motor.findData(last)
        if idx < 0:
            idx = self._motor.findData(ap.DEFAULT_MOTOR)
        self._motor.setCurrentIndex(idx)
        self._motor.blockSignals(False)
        self._on_motor_changed()

    def _on_motor_changed(self) -> None:
        motor = self._motor.currentData()
        if not motor or motor == self._current_motor:
            return
        # Guardar la voz actual en la clave del motor que se abandona.
        previous_key = ap.voice_key(self._current_motor)
        if previous_key and self._voice.currentData():
            self._prefs.set(previous_key, self._voice.currentData())
        self._current_motor = motor
        self._engine = self._engines.get(motor)
        self._prefs.set(prefs.AUDIO_ENGINE, motor)
        self._populate_voices()
        self._generate_button.setEnabled(self._engine is not None)
        if self._engine is None:
            self._set_status(f"El motor '{motor}' no está disponible.")

    def _populate_voices(self) -> None:
        self._voice.blockSignals(True)
        self._voice.clear()
        voices: list[str] = []
        if self._engine is not None:
            try:
                voices = list(self._engine.list_voices())
            except EngineError as exc:
                logger.warning("No se pudieron listar las voces: %s", exc)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Error listando voces: %r", exc, exc_info=True)
        for voice_name in voices:
            self._voice.addItem(voice_name, voice_name)
        key = ap.voice_key(self._current_motor)
        saved = self._prefs.get_str(key) if key else None
        selected = ap.resolve_voice(self._current_motor, saved, voices)
        if selected is not None:
            self._voice.setCurrentIndex(voices.index(selected))
        self._voice.blockSignals(False)

    def _on_voice_changed(self) -> None:
        key = ap.voice_key(self._current_motor)
        if key and self._voice.currentData():
            self._prefs.set(key, self._voice.currentData())

    # ---- Generación ----------------------------------------------

    def _generate(self) -> None:
        if self._engine is None:
            self._set_status("Error: no hay motor TTS configurado.")
            return
        text = self._text_area.toPlainText().strip()
        if not text:
            self._set_status("Escribe algún texto antes de generar.")
            return
        if self._worker is not None and self._worker.isRunning():
            return

        voice = self._voice.currentData() or None
        speed = self._speed.value()

        self._set_busy(True)
        self._set_status("Generando voz...")

        self._worker = AudioWorker(self._engine, text, voice, speed, self)
        self._worker.result.connect(self._on_audio_ready)
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

    def _cancel(self) -> None:
        if self._worker is None or not self._worker.isRunning():
            return
        self._worker.requestInterruption()
        self._set_status("Cancelando...")

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
            timeout=_MP3_TIMEOUT,
        )

    # ---- Utilidades ----------------------------------------------

    def _busy_widgets(self) -> tuple:
        return (self._text_area, self._motor, self._voice, self._speed)

    def _busy_only_widgets(self) -> tuple:
        return (self._cancel_button,)

    def _on_busy_changed(self, busy: bool) -> None:
        self._generate_button.setEnabled(not busy and self._engine is not None)

    def _focus_widget(self):
        """El área de texto es el control que debe recuperar el foco."""
        return self._text_area

    def set_engines(self, piper_engine=None, kokoro_engine=None) -> None:
        self._engines = {"piper": piper_engine, "kokoro": kokoro_engine}
        if self._current_motor:
            self._engine = self._engines.get(self._current_motor)
            self._populate_voices()
        self._generate_button.setEnabled(self._engine is not None)


