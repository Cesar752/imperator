"""Widget del Ensamblador: combina MP4 + audio + subtítulos (Whisper) en un MP4."""

import logging
import os
import shutil
from pathlib import Path

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.video_assembler import VideoAssembler
from engines.base_engine import CANCEL_MESSAGE, Engine, EngineError
from engines.whisper_engine import WHISPER_MODELS
from services import assembler_params as asp
from services import preferences as prefs
from ui.task_widget import TaskWidget
from ui.workers import ThreadWorker

logger = logging.getLogger(__name__)

# Filtros de diálogo de archivos.
VIDEO_FILTER = "Vídeo (*.mp4)"
AUDIO_FILTER = "Audio (*.wav *.mp3 *.m4a *.ogg *.flac)"

# Carpeta de biblioteca de música de fondo (el usuario mete sus MP3 aquí).
MUSIC_DIR = Path(__file__).resolve().parent.parent / "music"
MUSIC_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}


class AssemblerWorker(ThreadWorker):
    """Orquesta transcripción (Whisper) + ensamblaje (ffmpeg) fuera de la UI."""

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
        voice_offset,
        music_offset,
        parent=None,
        *,
        keep_original: bool = False,
        original_volume: float = 1.0,
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
        self._voice_offset = voice_offset
        self._music_offset = music_offset
        #: Conservar el audio del propio vídeo (ambiente) bajo la voz elegida.
        self._keep_original = keep_original
        #: Volumen de ese audio del vídeo (1.0 = tal cual; menos, lo baja bajo la voz).
        self._original_volume = original_volume

    def work(self) -> str:
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
        if self.isInterruptionRequested():
            raise EngineError(CANCEL_MESSAGE)
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
            voice_offset=self._voice_offset,
            music_offset=self._music_offset,
            keep_original=self._keep_original,
            original_volume=self._original_volume,
        )
        self.progress.emit(100, "Listo")
        return str(output)

    def _on_transcribe_progress(self, percent: int, message: str) -> None:
        # La transcripción ocupa el tramo 0-40 de la barra de progreso.
        self.progress.emit(int(percent * 0.4), message)


class AssemblerWidget(TaskWidget):
    """Formulario del Ensamblador: vídeo + audio + subtítulos -> MP4 final."""

    def __init__(
        self, whisper_engine: Engine | None, config_manager=None, parent=None
    ) -> None:
        super().__init__(parent)
        self._whisper: Engine | None = whisper_engine
        self._config_manager = config_manager
        self._prefs = prefs.PrefsStore(config_manager)
        self._assembler = VideoAssembler()
        self._worker: AssemblerWorker | None = None
        self._last_path: str | None = None
        self._video_path: str | None = None
        self._audio_path: str | None = None
        self._video_duration: float | None = None
        self._video_has_audio = False
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

        self._original_volume = QSpinBox()
        self._original_volume.setRange(0, 100)
        self._original_volume.setValue(100)
        self._original_volume.setSuffix("%")
        self._original_volume.setToolTip(
            "Volumen del audio que trae el vídeo (p. ej. el ambiente de una crónica).\n"
            "Bájalo si el sonido del vídeo tapa a la voz."
        )

        self._ducking = QCheckBox("Ducking automático")
        self._ducking.setChecked(True)
        self._ducking.toggled.connect(self._on_ducking_toggled)

        self._voice_offset = QDoubleSpinBox()
        self._voice_offset.setRange(0.0, 30.0)
        self._voice_offset.setSingleStep(0.5)
        self._voice_offset.setDecimals(1)
        self._voice_offset.setValue(0.0)
        self._voice_offset.setToolTip(
            "Cuándo empieza a sonar la voz, en segundos."
        )
        self._voice_offset.valueChanged.connect(self._on_voice_offset_changed)

        self._music_offset = QDoubleSpinBox()
        self._music_offset.setRange(0.0, 30.0)
        self._music_offset.setSingleStep(0.5)
        self._music_offset.setDecimals(1)
        self._music_offset.setValue(0.0)
        self._music_offset.setToolTip(
            "Cuándo empieza a sonar la música, en segundos."
        )
        self._music_offset.valueChanged.connect(self._on_music_offset_changed)

        music_form = QFormLayout()
        music_form.addRow("Pista:", music_row)
        music_form.addRow("Vol. música:", self._music_volume)
        music_form.addRow("Vol. voz:", self._voice_volume)
        music_form.addRow("Vol. vídeo:", self._original_volume)
        music_form.addRow("Inicio voz (s):", self._voice_offset)
        music_form.addRow("Inicio música (s):", self._music_offset)
        music_form.addRow("", self._ducking)

        self._original_check = QCheckBox("Conservar el audio del vídeo")
        self._original_check.setToolTip(
            "Mezcla el sonido original de la grabación (público, silbato, balón) con\n"
            "la voz en vez de sustituirlo. Sin voz ni música el audio se copia igual."
        )
        music_form.addRow("", self._original_check)

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
        for code, label in asp.LANGUAGES:
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
        for key, label in asp.POSITIONS:
            self._position_combo.addItem(label, key)
        self._position_combo.currentIndexChanged.connect(self._on_position_changed)

        self._color_combo = QComboBox()
        for key, label in asp.COLORS:
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
        self._cancel_button = QPushButton("Cancelar")
        self._cancel_button.setEnabled(False)
        self._open_button = QPushButton("Abrir carpeta")
        self._open_button.setEnabled(False)
        self._save_button = QPushButton("Guardar como...")
        self._save_button.setEnabled(False)

        buttons = QHBoxLayout()
        buttons.addWidget(self._assemble_button)
        buttons.addWidget(self._cancel_button)
        buttons.addWidget(self._save_button)
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
        self._cancel_button.clicked.connect(self._cancel)
        self._save_button.clicked.connect(self._save_as)
        self._open_button.clicked.connect(self._open_folder)

    # ---- Selección de archivos -----------------------------------

    def _select_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar vídeo", "", VIDEO_FILTER
        )
        if not path:
            return
        self.set_video(path)

    def set_video(self, path: str) -> None:
        """Precarga un vídeo (lo usa la pestaña Vídeo al enviar la secuencia).

        Es la puerta pública para que la ventana principal pueda empalmar la
        salida de la pestaña Vídeo con esta, sin que el widget de vídeo conozca
        el Ensamblador.
        """
        if not path:
            return
        self._video_path = str(path)
        self._video_label.setText(self._video_path)
        try:
            info = self._assembler.probe(self._video_path)
        except EngineError as exc:
            self._set_status(f"No se pudo inspeccionar el vídeo: {exc}")
            info = {}
        self._video_duration = info.get("duration")
        self._video_has_audio = bool(info.get("has_audio"))
        parts = []
        if info.get("width") and info.get("height"):
            parts.append(f"Resolución: {info['width']}×{info['height']}")
        if self._video_duration is not None:
            parts.append(f"Duración: {self._video_duration:.1f} s")
        # Se dice explícitamente: sin esto no se sabía si el vídeo traía audio (una
        # crónica ya lo trae, con la narración mezclada).
        parts.append("Audio: sí" if self._video_has_audio else "Audio: no")
        self._video_info.setText(" · ".join(parts))
        self._sugerir_audio_original()
        self._check_duration_gap()

    def _sugerir_audio_original(self) -> None:
        """Marca «Conservar el audio del vídeo» según lo que traiga el vídeo.

        Al enviar una crónica, su audio (narración + ambiente) **es** el que se quiere
        oír: si se dejara sin marcar, añadir música o subtítulos lo tiraría. Y al
        revés: si el vídeo no tiene audio, se desmarca, porque intentar conservarlo
        haría fallar el ensamblado (no hay pista ``[0:a]`` que mezclar).
        """
        if not self._video_has_audio:
            self._original_check.setChecked(False)
            return
        if self._audio_path is None:
            self._original_check.setChecked(True)
            self._set_status(
                "El vídeo ya trae audio (narración/ambiente): se conservará. "
                "Desmárcalo si le vas a poner otra voz."
            )

    def _select_audio(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar audio", "", AUDIO_FILTER
        )
        if not path:
            return
        self._audio_path = path
        self._audio_label.setText(path)
        try:
            info = self._assembler.probe(path)
        except EngineError as exc:
            self._set_status(f"No se pudo inspeccionar el audio: {exc}")
            info = {}
        self._audio_duration = info.get("duration")
        self._audio_info.setText(
            f"Duración: {self._audio_duration:.1f} s"
            if self._audio_duration is not None
            else ""
        )
        self._check_duration_gap()

    def _check_duration_gap(self) -> None:
        message = asp.duration_gap_message(self._video_duration, self._audio_duration)
        if message is not None:
            self._set_status(message)

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

        subtitle_style = asp.build_subtitle_style(
            self._fontsize.value(),
            self._position_combo.currentData(),
            self._color_combo.currentData(),
        )

        music_path = self._music_combo.currentData()
        music_volume = self._music_volume.value() / 100.0
        voice_volume = self._voice_volume.value() / 100.0
        original_volume = self._original_volume.value() / 100.0
        ducking = self._ducking.isChecked()
        voice_offset = self._voice_offset.value()
        music_offset = self._music_offset.value()
        keep_original = self._original_check.isChecked()

        video_duration = self._video_duration
        if video_duration is None and self._video_path:
            try:
                video_duration = self._assembler.probe(self._video_path).get("duration")
            except EngineError as exc:
                self._set_status(f"No se pudo inspeccionar el vídeo: {exc}")
                return
        if video_duration is not None:
            if self._audio_path and asp.offset_too_late(voice_offset, video_duration):
                self._set_status(
                    f"Aviso: el inicio de la voz ({voice_offset:.1f}s) supera la "
                    f"duración del vídeo ({video_duration:.1f}s). La voz no se "
                    "oirá. Cambia el offset o genera un vídeo más largo antes de "
                    "continuar."
                )
                return
            if music_path and asp.offset_too_late(music_offset, video_duration):
                self._set_status(
                    f"Aviso: el inicio de la música ({music_offset:.1f}s) supera "
                    f"la duración del vídeo ({video_duration:.1f}s). La música no "
                    "se oirá. Cambia el offset o genera un vídeo más largo antes "
                    "de continuar."
                )
                return

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
            voice_offset,
            music_offset,
            self,
            keep_original=keep_original,
            original_volume=original_volume,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.result.connect(self._on_ready)
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
        self._save_button.setEnabled(True)
        self._set_status(f"Listo: {path}")

    def _cancel(self) -> None:
        if self._worker is None or not self._worker.isRunning():
            return
        self._worker.requestInterruption()
        self._set_status("Cancelando...")

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

    def _save_as(self) -> None:
        if not self._last_path:
            return
        target, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar vídeo como...",
            self._last_path,
            "Vídeo MP4 (*.mp4)",
        )
        if not target:
            return
        if not target.lower().endswith(".mp4"):
            target += ".mp4"
        try:
            shutil.copyfile(self._last_path, target)
        except OSError as exc:
            QMessageBox.critical(self, "Error", f"No se pudo guardar el vídeo: {exc}")
            return
        self._set_status(f"Copiado a: {target}")

    # ---- Subtítulos y persistencia -------------------------------

    def _on_subtitles_toggled(self, checked: bool) -> None:
        self._subtitles_section.setVisible(checked)

    def _on_model_changed(self) -> None:
        self._prefs.set(prefs.ASSEMBLER, self._model_combo.currentData())

    def _on_language_changed(self) -> None:
        self._prefs.set(prefs.ASSEMBLER_IDIOMA, self._language_combo.currentData())

    def _on_fontsize_changed(self) -> None:
        self._prefs.set(prefs.ASSEMBLER_FONTSIZE, self._fontsize.value())

    def _on_position_changed(self) -> None:
        self._prefs.set(prefs.ASSEMBLER_POSICION, self._position_combo.currentData())

    def _on_color_changed(self) -> None:
        self._prefs.set(prefs.ASSEMBLER_COLOR, self._color_combo.currentData())

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
        self._prefs.set(prefs.ASSEMBLER_MUSICA, self._music_combo.currentData() or "")

    def _on_music_volume_changed(self) -> None:
        self._prefs.set(prefs.ASSEMBLER_MUSICA_VOL, self._music_volume.value())

    def _on_voice_volume_changed(self) -> None:
        self._prefs.set(prefs.ASSEMBLER_VOZ_VOL, self._voice_volume.value())

    def _on_voice_offset_changed(self) -> None:
        self._prefs.set(prefs.ASSEMBLER_VOICE_OFFSET, self._voice_offset.value())

    def _on_music_offset_changed(self) -> None:
        self._prefs.set(prefs.ASSEMBLER_MUSIC_OFFSET, self._music_offset.value())

    def _on_ducking_toggled(self, checked: bool) -> None:
        self._prefs.set_bool(prefs.ASSEMBLER_DUCKING, checked)

    def _restore_preferences(self) -> None:
        p = prefs.AssemblerPrefs.from_store(self._prefs)

        combos = (
            (self._model_combo, p.model),
            (self._language_combo, p.language),
            (self._position_combo, p.position),
            (self._color_combo, p.color),
        )
        all_combos = (
            self._model_combo,
            self._language_combo,
            self._position_combo,
            self._color_combo,
            self._music_combo,
        )
        spins = (
            self._fontsize,
            self._music_volume,
            self._voice_volume,
            self._original_volume,
            self._voice_offset,
            self._music_offset,
        )
        for combo in all_combos:
            combo.blockSignals(True)
        for spin in spins:
            spin.blockSignals(True)
        self._ducking.blockSignals(True)
        try:
            for combo, value in combos:
                if value is not None:
                    idx = combo.findData(value)
                    if idx >= 0:
                        combo.setCurrentIndex(idx)
            self._fontsize.setValue(p.fontsize)
            if p.music:
                idx = self._music_combo.findData(p.music)
                if idx >= 0:
                    self._music_combo.setCurrentIndex(idx)
            self._music_volume.setValue(p.music_volume)
            self._voice_volume.setValue(p.voice_volume)
            self._voice_offset.setValue(p.voice_offset)
            self._music_offset.setValue(p.music_offset)
            self._ducking.setChecked(p.ducking)
        finally:
            for combo in all_combos:
                combo.blockSignals(False)
            for spin in spins:
                spin.blockSignals(False)
            self._ducking.blockSignals(False)

    # ---- Utilidades ----------------------------------------------

    def _busy_widgets(self) -> tuple:
        return (
            self._assemble_button,
            self._video_button,
            self._audio_button,
            self._music_combo,
            self._music_refresh,
            self._music_volume,
            self._voice_volume,
            self._original_volume,
            self._voice_offset,
            self._music_offset,
            self._ducking,
            self._subtitles_section,
        )

    def _busy_only_widgets(self) -> tuple:
        return (self._cancel_button,)

    def _on_busy_changed(self, busy: bool) -> None:
        self._save_button.setEnabled(not busy and self._last_path is not None)
        self._subtitles_check.setEnabled(not busy and self._whisper is not None)

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
