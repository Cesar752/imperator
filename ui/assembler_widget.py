"""Widget del Ensamblador: combina MP4 + audio + subtítulos (Whisper) en un MP4."""

import logging
import os
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.video_assembler import VideoAssembler
from engines.base_engine import Engine, EngineError
from engines.whisper_engine import WHISPER_MODELS

logger = logging.getLogger(__name__)

# Filtros de diálogo de archivos.
VIDEO_FILTER = "Vídeo (*.mp4)"
AUDIO_FILTER = "Audio (*.wav *.mp3 *.m4a *.ogg *.flac)"

# Idiomas disponibles para la transcripción (código -> etiqueta).
_LANGUAGES = [
    ("auto", "Auto"),
    ("es", "Español"),
    ("en", "Inglés"),
    ("eu", "Euskera"),
    ("fr", "Francés"),
    ("de", "Alemán"),
    ("it", "Italiano"),
    ("pt", "Portugués"),
]

# Posición y color de los subtítulos (clave interna -> etiqueta).
_POSITIONS = [("bottom", "Abajo"), ("middle", "Centro"), ("top", "Arriba")]
_COLORS = [("white", "Blanco"), ("yellow", "Amarillo"), ("green", "Verde")]

# Diferencia de duración (s) a partir de la cual se avisa al usuario.
_DURATION_GAP_WARNING = 2.0

# Grosor del contorno del subtítulo (px), fijo por ahora (no expuesto en la UI).
_DEFAULT_OUTLINE = 2

# Carpeta de biblioteca de música de fondo (el usuario mete sus MP3 aquí).
MUSIC_DIR = Path(__file__).resolve().parent.parent / "music"
MUSIC_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}


class AssemblerWorker(QThread):
    """Orquesta transcripción (Whisper) + ensamblaje (ffmpeg) fuera de la UI."""

    progress = pyqtSignal(int, str)   # porcentaje, mensaje
    result_ready = pyqtSignal(str)    # ruta del MP4 final
    failed = pyqtSignal(str)

    def __init__(
        self,
        whisper_engine,
        assembler,
        video_path,
        audio_path,
        use_subtitles,
        model_size,
        language,
        subtitle_style,
        music_path,
        music_volume,
        voice_volume,
        ducking,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._whisper = whisper_engine
        self._assembler = assembler
        self._video_path = video_path
        self._audio_path = audio_path
        self._use_subtitles = use_subtitles
        self._model_size = model_size
        self._language = language
        self._subtitle_style = subtitle_style
        self._music_path = music_path
        self._music_volume = music_volume
        self._voice_volume = voice_volume
        self._ducking = ducking

    def run(self) -> None:
        try:
            srt_path = None
            if self._use_subtitles and self._audio_path:
                if self._whisper is None:
                    raise EngineError(
                        "El motor Whisper no está configurado o deshabilitado."
                    )
                self.progress.emit(0, "Transcribiendo audio...")
                srt_path = self._whisper.transcribe(
                    self._audio_path,
                    language=self._language,
                    model_size=self._model_size,
                    progress_callback=self._on_transcribe_progress,
                )
            self.progress.emit(40, "Ensamblando vídeo...")
            output = self._assembler.assemble(
                self._video_path,
                audio_path=self._audio_path,
                srt_path=srt_path,
                subtitle_style=self._subtitle_style,
                music_path=self._music_path,
                music_volume=self._music_volume,
                voice_volume=self._voice_volume,
                ducking=self._ducking,
            )
            self.progress.emit(100, "Listo")
            self.result_ready.emit(str(output))
        except EngineError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001 - IO/ffmpeg inesperados
            logger.debug("Error inesperado en AssemblerWorker: %r", exc, exc_info=True)
            self.failed.emit(f"Error inesperado: {exc}")

    def _on_transcribe_progress(self, percent: int, message: str) -> None:
        # La transcripción ocupa el tramo 0-40 de la barra de progreso.
        self.progress.emit(int(percent * 0.4), message)


class AssemblerWidget(QWidget):
    """Formulario del Ensamblador: vídeo + audio + subtítulos -> MP4 final."""

    def __init__(
        self, whisper_engine: Engine | None, config_manager=None, parent=None
    ) -> None:
        super().__init__(parent)
        self._whisper: Engine | None = whisper_engine
        self._config_manager = config_manager
        self._assembler = VideoAssembler()
        self._worker: AssemblerWorker | None = None
        self._last_path: str | None = None
        self._video_path: str | None = None
        self._audio_path: str | None = None
        self._video_duration: float | None = None
        self._audio_duration: float | None = None
        self._build_ui()
        self._restore_preferences()
        self._subtitles_section.setVisible(self._subtitles_check.isChecked())
        if self._whisper is None:
            self._subtitles_check.setChecked(False)
            self._subtitles_check.setEnabled(False)
            self._set_status(
                "Whisper no está configurado o deshabilitado; "
                "los subtítulos no estarán disponibles."
            )

    # ---- Construcción de la UI -----------------------------------

    def _build_ui(self) -> None:
        # Vídeo
        self._video_button = QPushButton("Seleccionar vídeo...")
        self._video_button.clicked.connect(self._select_video)
        self._video_label = QLabel("Sin vídeo")
        self._video_label.setWordWrap(True)
        self._video_info = QLabel("")

        video_group = QGroupBox("Vídeo (MP4)")
        video_layout = QVBoxLayout(video_group)
        video_layout.addWidget(self._video_button)
        video_layout.addWidget(self._video_label)
        video_layout.addWidget(self._video_info)

        # Audio
        self._audio_button = QPushButton("Seleccionar audio...")
        self._audio_button.clicked.connect(self._select_audio)
        self._audio_label = QLabel("Sin audio")
        self._audio_label.setWordWrap(True)
        self._audio_info = QLabel("")

        audio_group = QGroupBox("Audio (opcional)")
        audio_layout = QVBoxLayout(audio_group)
        audio_layout.addWidget(self._audio_button)
        audio_layout.addWidget(self._audio_label)
        audio_layout.addWidget(self._audio_info)

        # Música de fondo
        self._music_combo = QComboBox()
        self._music_combo.currentIndexChanged.connect(self._on_music_changed)
        self._music_refresh = QPushButton("Refrescar")
        self._music_refresh.clicked.connect(self._populate_music)

        music_row = QHBoxLayout()
        music_row.addWidget(self._music_combo)
        music_row.addWidget(self._music_refresh)

        self._music_volume = QSpinBox()
        self._music_volume.setRange(0, 100)
        self._music_volume.setValue(30)
        self._music_volume.setSuffix("%")
        self._music_volume.valueChanged.connect(self._on_music_volume_changed)

        self._voice_volume = QSpinBox()
        self._voice_volume.setRange(0, 100)
        self._voice_volume.setValue(100)
        self._voice_volume.setSuffix("%")
        self._voice_volume.valueChanged.connect(self._on_voice_volume_changed)

        self._ducking = QCheckBox("Ducking automático")
        self._ducking.setChecked(True)
        self._ducking.toggled.connect(self._on_ducking_toggled)

        music_form = QFormLayout()
        music_form.addRow("Pista:", music_row)
        music_form.addRow("Vol. música:", self._music_volume)
        music_form.addRow("Vol. voz:", self._voice_volume)
        music_form.addRow("", self._ducking)

        music_group = QGroupBox("Música de fondo")
        music_group.setLayout(music_form)

        self._populate_music()

        # Subtítulos
        self._subtitles_check = QCheckBox("Generar subtítulos con Whisper")
        self._subtitles_check.setChecked(True)
        self._subtitles_check.toggled.connect(self._on_subtitles_toggled)

        self._model_combo = QComboBox()
        for model in WHISPER_MODELS:
            self._model_combo.addItem(model, model)
        self._model_combo.currentIndexChanged.connect(self._on_model_changed)

        self._language_combo = QComboBox()
        for code, label in _LANGUAGES:
            self._language_combo.addItem(label, code)
        self._language_combo.currentIndexChanged.connect(self._on_language_changed)

        self._fontsize = QSpinBox()
        self._fontsize.setRange(12, 72)
        self._fontsize.setValue(24)
        self._fontsize.setToolTip(
            "Tamaño del texto. 24 es apropiado para 832x480; para 720x1280 usa "
            "36-44; para 896x1600 usa 48-56."
        )
        self._fontsize.valueChanged.connect(self._on_fontsize_changed)

        self._position_combo = QComboBox()
        for key, label in _POSITIONS:
            self._position_combo.addItem(label, key)
        self._position_combo.currentIndexChanged.connect(self._on_position_changed)

        self._color_combo = QComboBox()
        for key, label in _COLORS:
            self._color_combo.addItem(label, key)
        self._color_combo.currentIndexChanged.connect(self._on_color_changed)

        subtitles_form = QFormLayout()
        subtitles_form.addRow("Modelo Whisper:", self._model_combo)
        subtitles_form.addRow("Idioma:", self._language_combo)

        style_group = QGroupBox("Estilo de subtítulos")
        style_form = QFormLayout(style_group)
        style_form.addRow("Tamaño:", self._fontsize)
        style_form.addRow("Posición:", self._position_combo)
        style_form.addRow("Color:", self._color_combo)

        self._subtitles_section = QWidget()
        subtitles_layout = QVBoxLayout(self._subtitles_section)
        subtitles_layout.setContentsMargins(0, 0, 0, 0)
        subtitles_layout.addLayout(subtitles_form)
        subtitles_layout.addWidget(style_group)

        # Acciones y estado
        self._assemble_button = QPushButton("Ensamblar")
        self._open_button = QPushButton("Abrir carpeta")
        self._open_button.setEnabled(False)

        buttons = QHBoxLayout()
        buttons.addWidget(self._assemble_button)
        buttons.addWidget(self._open_button)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)

        self._status = QLabel("")

        layout = QVBoxLayout(self)
        layout.addWidget(video_group)
        layout.addWidget(audio_group)
        layout.addWidget(music_group)
        layout.addWidget(self._subtitles_check)
        layout.addWidget(self._subtitles_section)
        layout.addLayout(buttons)
        layout.addWidget(self._progress)
        layout.addWidget(self._status)
        layout.addStretch(1)

        self._assemble_button.clicked.connect(self._assemble)
        self._open_button.clicked.connect(self._open_folder)

    # ---- Selección de archivos -----------------------------------

    def _select_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar vídeo", "", VIDEO_FILTER
        )
        if not path:
            return
        self._video_path = path
        self._video_label.setText(path)
        info = self._assembler.probe(path)
        self._video_duration = info.get("duration")
        parts = []
        if info.get("width") and info.get("height"):
            parts.append(f"Resolución: {info['width']}×{info['height']}")
        if self._video_duration is not None:
            parts.append(f"Duración: {self._video_duration:.1f} s")
        self._video_info.setText(" · ".join(parts))
        self._check_duration_gap()

    def _select_audio(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar audio", "", AUDIO_FILTER
        )
        if not path:
            return
        self._audio_path = path
        self._audio_label.setText(path)
        info = self._assembler.probe(path)
        self._audio_duration = info.get("duration")
        self._audio_info.setText(
            f"Duración: {self._audio_duration:.1f} s"
            if self._audio_duration is not None
            else ""
        )
        self._check_duration_gap()

    def _check_duration_gap(self) -> None:
        video = self._video_duration
        audio = self._audio_duration
        if video is None or audio is None:
            return
        if abs(video - audio) > _DURATION_GAP_WARNING:
            self._set_status(
                f"Aviso: el audio dura {audio:.1f} s y el vídeo {video:.1f} s. "
                "Se cortará al más corto."
            )

    # ---- Ensamblado ----------------------------------------------

    def _assemble(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        if not self._video_path:
            self._set_status("Selecciona un vídeo (MP4).")
            return
        use_subtitles = (
            self._subtitles_check.isChecked() and self._subtitles_check.isEnabled()
        )
        if use_subtitles and not self._audio_path:
            self._set_status("Selecciona un audio para generar subtítulos.")
            return
        if use_subtitles and self._whisper is None:
            self._set_status("Whisper no está disponible.")
            return

        subtitle_style = {
            "fontsize": self._fontsize.value(),
            "position": self._position_combo.currentData(),
            "color": self._color_combo.currentData(),
            "outline": _DEFAULT_OUTLINE,
        }

        music_path = self._music_combo.currentData()
        music_volume = self._music_volume.value() / 100.0
        voice_volume = self._voice_volume.value() / 100.0
        ducking = self._ducking.isChecked()

        self._set_busy(True)
        self._progress.setValue(0)
        self._set_status("Iniciando...")

        self._worker = AssemblerWorker(
            self._whisper,
            self._assembler,
            self._video_path,
            self._audio_path,
            use_subtitles,
            self._model_combo.currentData(),
            self._language_combo.currentData(),
            subtitle_style,
            music_path,
            music_volume,
            voice_volume,
            ducking,
            self,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.result_ready.connect(self._on_ready)
        self._worker.failed.connect(self._on_error)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _on_progress(self, percent: int, message: str) -> None:
        self._progress.setValue(percent)
        self._set_status(f"{message} ({percent}%)")

    def _on_ready(self, path: str) -> None:
        self._last_path = path
        self._progress.setValue(100)
        self._open_button.setEnabled(True)
        self._set_status(f"Listo: {path}")

    def _on_error(self, message: str) -> None:
        self._set_status(f"Error: {message}")

    def _on_finished(self) -> None:
        self._set_busy(False)
        self._worker.deleteLater()
        self._worker = None

    def _open_folder(self) -> None:
        startfile = getattr(os, "startfile", None)
        if startfile is None:
            self._set_status("No se puede abrir la carpeta en este sistema.")
            return
        output_dir = self._assembler.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            startfile(str(output_dir))
        except OSError as exc:
            self._set_status(f"No se pudo abrir la carpeta: {exc}")

    # ---- Subtítulos y persistencia -------------------------------

    def _on_subtitles_toggled(self, checked: bool) -> None:
        self._subtitles_section.setVisible(checked)

    def _on_model_changed(self) -> None:
        self._save_pref("assembler", self._model_combo.currentData())

    def _on_language_changed(self) -> None:
        self._save_pref("assembler_idioma", self._language_combo.currentData())

    def _on_fontsize_changed(self) -> None:
        self._save_pref("assembler_fontsize", str(self._fontsize.value()))

    def _on_position_changed(self) -> None:
        self._save_pref("assembler_posicion", self._position_combo.currentData())

    def _on_color_changed(self) -> None:
        self._save_pref("assembler_color", self._color_combo.currentData())

    # ---- Música de fondo -----------------------------------------

    def _populate_music(self) -> None:
        """Rellena el combo con las pistas disponibles en music/."""
        self._music_combo.blockSignals(True)
        current = self._music_combo.currentData()
        self._music_combo.clear()
        self._music_combo.addItem("(ninguna)", None)
        for path in sorted(MUSIC_DIR.glob("*")):
            if path.is_file() and path.suffix.lower() in MUSIC_EXTS:
                self._music_combo.addItem(path.stem, str(path))
        if current is not None:
            idx = self._music_combo.findData(current)
            if idx >= 0:
                self._music_combo.setCurrentIndex(idx)
        self._music_combo.blockSignals(False)

    def _on_music_changed(self) -> None:
        self._save_pref("assembler_musica", self._music_combo.currentData() or "")

    def _on_music_volume_changed(self) -> None:
        self._save_pref("assembler_musica_vol", str(self._music_volume.value()))

    def _on_voice_volume_changed(self) -> None:
        self._save_pref("assembler_voz_vol", str(self._voice_volume.value()))

    def _on_ducking_toggled(self, checked: bool) -> None:
        self._save_pref("assembler_ducking", "1" if checked else "0")

    def _save_pref(self, key: str, value: str) -> None:
        if self._config_manager is not None and value is not None:
            self._config_manager.save_last_model(key, value)

    def _restore_preferences(self) -> None:
        if self._config_manager is None:
            return
        last = self._config_manager.load_last_models()

        combos = (
            (self._model_combo, "assembler"),
            (self._language_combo, "assembler_idioma"),
            (self._position_combo, "assembler_posicion"),
            (self._color_combo, "assembler_color"),
        )
        all_combos = (
            self._model_combo,
            self._language_combo,
            self._position_combo,
            self._color_combo,
            self._music_combo,
        )
        for combo in all_combos:
            combo.blockSignals(True)
        for spin in (self._fontsize, self._music_volume, self._voice_volume):
            spin.blockSignals(True)
        self._ducking.blockSignals(True)
        try:
            for combo, key in combos:
                value = last.get(key)
                if value is not None:
                    idx = combo.findData(value)
                    if idx >= 0:
                        combo.setCurrentIndex(idx)
            fontsize = last.get("assembler_fontsize")
            if fontsize is not None:
                try:
                    self._fontsize.setValue(int(fontsize))
                except ValueError:
                    pass
            music_path = last.get("assembler_musica")
            if music_path:
                idx = self._music_combo.findData(music_path)
                if idx >= 0:
                    self._music_combo.setCurrentIndex(idx)
            music_vol = last.get("assembler_musica_vol")
            if music_vol is not None:
                try:
                    self._music_volume.setValue(int(music_vol))
                except ValueError:
                    pass
            voice_vol = last.get("assembler_voz_vol")
            if voice_vol is not None:
                try:
                    self._voice_volume.setValue(int(voice_vol))
                except ValueError:
                    pass
            ducking = last.get("assembler_ducking")
            if ducking is not None:
                self._ducking.setChecked(ducking == "1")
        finally:
            for combo in all_combos:
                combo.blockSignals(False)
            for spin in (self._fontsize, self._music_volume, self._voice_volume):
                spin.blockSignals(False)
            self._ducking.blockSignals(False)

    # ---- Utilidades ----------------------------------------------

    def _set_busy(self, busy: bool) -> None:
        self._assemble_button.setEnabled(not busy)
        self._video_button.setEnabled(not busy)
        self._audio_button.setEnabled(not busy)
        self._music_combo.setEnabled(not busy)
        self._music_refresh.setEnabled(not busy)
        self._music_volume.setEnabled(not busy)
        self._voice_volume.setEnabled(not busy)
        self._ducking.setEnabled(not busy)
        self._subtitles_check.setEnabled(not busy and self._whisper is not None)
        self._subtitles_section.setEnabled(not busy)

    def _set_status(self, text: str) -> None:
        self._status.setText(text)

    def set_engine(self, engine: Engine | None) -> None:
        self._whisper = engine
        if engine is None:
            self._subtitles_check.setChecked(False)
            self._subtitles_check.setEnabled(False)
            self._set_status(
                "Whisper no está configurado o deshabilitado; "
                "los subtítulos no estarán disponibles."
            )
        else:
            self._subtitles_check.setEnabled(True)
            self._set_status("")
