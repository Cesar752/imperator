"""Preferencias persistidas (settings.json -> ``last_models``) con tipos.

``ConfigManager.last_models`` es un ``dict[str, str]`` sin tipos: cada widget
hacía ``int(raw)``/``float(raw)``/``== "1"`` a mano, con ``try/except``
dispersos y strings sueltos. Aquí se centraliza en:

- :class:`PrefsStore`: lectura/escritura tipada (``get_int``, ``get_float``,
  ``get_bool``, ``set``) sobre las claves de :mod:`services.preferences`.
- Dataclasses por dominio (``from_store``) que evitan conversiones repetidas.

Sin dependencias de Qt (testeable sin GUI).
"""

from dataclasses import dataclass

# ---- Claves (evitan strings sueltos repartidos por los widgets) ----

CHAT = "chat"
DOCUMENT = "document"
DOCUMENT_MODE = "document_mode"
DOCUMENT_LANGUAGE = "document_language"
IMAGE = "image"
VIDEO = "video"
VIDEO_LAST_IMAGE = "video_last_image"
VIDEO_TIMEOUT = "video_timeout"
AUDIO_ENGINE = "audio_engine"
AUDIO_VOICE_PIPER = "audio_voice_piper"
AUDIO_VOICE_KOKORO = "audio_voice_kokoro"
ASSEMBLER = "assembler"
ASSEMBLER_IDIOMA = "assembler_idioma"
ASSEMBLER_FONTSIZE = "assembler_fontsize"
ASSEMBLER_POSICION = "assembler_posicion"
ASSEMBLER_COLOR = "assembler_color"
ASSEMBLER_MUSICA = "assembler_musica"
ASSEMBLER_MUSICA_VOL = "assembler_musica_vol"
ASSEMBLER_VOZ_VOL = "assembler_voz_vol"
ASSEMBLER_VOICE_OFFSET = "assembler_voice_offset"
ASSEMBLER_MUSIC_OFFSET = "assembler_music_offset"
ASSEMBLER_DUCKING = "assembler_ducking"


class PrefsStore:
    """Acceso tipado a las preferencias (sobre ``ConfigManager.last_models``)."""

    def __init__(self, config_manager=None) -> None:
        self._config = config_manager

    def as_map(self) -> dict[str, str]:
        if self._config is None:
            return {}
        return self._config.load_last_models()

    def get_str(self, key: str, default: str | None = None) -> str | None:
        value = self.as_map().get(key)
        return value if value is not None else default

    def get_int(self, key: str, default: int) -> int:
        raw = self.as_map().get(key)
        if raw is None:
            return default
        try:
            return int(raw)
        except (TypeError, ValueError):
            return default

    def get_float(self, key: str, default: float) -> float:
        raw = self.as_map().get(key)
        if raw is None:
            return default
        try:
            return float(raw)
        except (TypeError, ValueError):
            return default

    def get_bool(self, key: str, default: bool = False) -> bool:
        raw = self.as_map().get(key)
        if raw is None:
            return default
        return str(raw) == "1"

    def set(self, key: str, value) -> None:
        """Guarda ``value`` como string (ignora ``None`` y sin config)."""
        if self._config is None or value is None:
            return
        self._config.save_last_model(key, str(value))

    def set_bool(self, key: str, value: bool) -> None:
        self.set(key, "1" if value else "0")


@dataclass
class VideoPrefs:
    """Preferencias de la pestaña Vídeo."""

    mode: str = "t2v"
    last_image: str | None = None
    timeout: int = 1800

    @classmethod
    def from_store(cls, store: PrefsStore) -> "VideoPrefs":
        return cls(
            mode=store.get_str(VIDEO, "t2v") or "t2v",
            last_image=store.get_str(VIDEO_LAST_IMAGE),
            timeout=store.get_int(VIDEO_TIMEOUT, 1800),
        )


@dataclass
class AudioPrefs:
    """Preferencias de la pestaña Audio (motor y voz por motor)."""

    engine: str = "piper"
    voice_piper: str | None = None
    voice_kokoro: str | None = None

    @classmethod
    def from_store(cls, store: PrefsStore) -> "AudioPrefs":
        return cls(
            engine=store.get_str(AUDIO_ENGINE, "piper") or "piper",
            voice_piper=store.get_str(AUDIO_VOICE_PIPER),
            voice_kokoro=store.get_str(AUDIO_VOICE_KOKORO),
        )


@dataclass
class AssemblerPrefs:
    """Preferencias de la pestaña Ensamblador."""

    model: str | None = None
    language: str | None = None
    fontsize: int = 24
    position: str | None = None
    color: str | None = None
    music: str | None = None
    music_volume: int = 30
    voice_volume: int = 100
    voice_offset: float = 0.0
    music_offset: float = 0.0
    ducking: bool = True

    @classmethod
    def from_store(cls, store: PrefsStore) -> "AssemblerPrefs":
        return cls(
            model=store.get_str(ASSEMBLER),
            language=store.get_str(ASSEMBLER_IDIOMA),
            fontsize=store.get_int(ASSEMBLER_FONTSIZE, 24),
            position=store.get_str(ASSEMBLER_POSICION),
            color=store.get_str(ASSEMBLER_COLOR),
            music=store.get_str(ASSEMBLER_MUSICA),
            music_volume=store.get_int(ASSEMBLER_MUSICA_VOL, 30),
            voice_volume=store.get_int(ASSEMBLER_VOZ_VOL, 100),
            voice_offset=store.get_float(ASSEMBLER_VOICE_OFFSET, 0.0),
            music_offset=store.get_float(ASSEMBLER_MUSIC_OFFSET, 0.0),
            ducking=store.get_bool(ASSEMBLER_DUCKING, True),
        )
