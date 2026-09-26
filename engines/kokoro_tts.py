"""Motor local Kokoro TTS (texto a voz con voces en español, en CPU)."""

import logging
import wave
from datetime import datetime
from pathlib import Path

try:
    import kokoro  # noqa: F401
    from kokoro import KPipeline
except Exception:  # noqa: BLE001 - cadena de imports (torch/misaki/espeak)
    kokoro = None  # noqa: F811
    KPipeline = None

import numpy as np

from engines.base_engine import Engine, EngineError

logger = logging.getLogger(__name__)

# Raíz del proyecto (C:\\Imperator); output/ cuelga de aquí.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Voces en español disponibles en Kokoro-82M.
KOKORO_VOICES = ["ef_dora", "em_alex", "em_santa"]

_SAMPLE_RATE = 24000
_REPO_ID = "hexgrad/Kokoro-82M"


class KokoroTTSEngine(Engine):
    """Sintetiza texto a voz con Kokoro (voces en español, CPU)."""

    def __init__(self, default_voice: str = "ef_dora", output_dir=None) -> None:
        self._default_voice = default_voice
        self._output_dir = (
            Path(output_dir) if output_dir is not None else PROJECT_ROOT / "output"
        )
        self._pipeline = None  # KPipeline (carga diferida: descarga el modelo)

    @property
    def id(self) -> str:
        return "kokoro_tts"

    @property
    def nombre(self) -> str:
        return "Kokoro TTS"

    def is_available(self) -> bool:
        return KPipeline is not None

    def list_models(self) -> list[str]:
        return self.list_voices()

    def list_voices(self) -> list[str]:
        return list(KOKORO_VOICES)

    def chat(self, messages: list[dict], model: str | None = None) -> str:
        raise EngineError("Kokoro TTS es un motor de texto a voz, no de chat.")

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
        output_path: Path | None = None,
    ) -> Path:
        """Sintetiza ``text`` con la voz ``voice`` y devuelve el WAV generado."""
        if KPipeline is None:
            raise EngineError(
                "kokoro no está instalado. Instálalo con: pip install kokoro soundfile"
            )
        text = (text or "").strip()
        if not text:
            raise EngineError("El texto a sintetizar está vacío.")

        voice_name = voice or self._default_voice
        if voice_name not in KOKORO_VOICES:
            raise EngineError(f"Kokoro TTS: la voz no existe: {voice_name}")

        output_path = (
            Path(output_path) if output_path is not None else self._default_output_path()
        )

        # La UI maneja "velocidad" como Piper (1.5 = más lento). Kokoro usa speed
        # al revés (1.5 = más rápido), así que invertimos.
        kokoro_speed = 1.0 / max(float(speed), 0.1)

        pipeline = self._get_pipeline()
        try:
            self._write_wav(pipeline, text, voice_name, kokoro_speed, output_path)
        except EngineError:
            raise
        except Exception as exc:  # noqa: BLE001
            output_path.unlink(missing_ok=True)
            raise EngineError(f"Kokoro TTS: error al sintetizar: {exc}") from exc

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise EngineError("Kokoro TTS: la síntesis no produjo audio.")
        logger.info("Kokoro TTS: audio guardado en '%s'.", output_path)
        return output_path

    # ---- Internos ------------------------------------------------

    def _default_output_path(self) -> Path:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"kokoro_{datetime.now():%Y%m%d_%H%M%S}.wav"
        return self._output_dir / filename

    def _get_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        logger.info("Kokoro TTS: cargando modelo '%s'...", _REPO_ID)
        try:
            self._pipeline = KPipeline(lang_code="e", repo_id=_REPO_ID)
        except Exception as exc:  # noqa: BLE001
            raise EngineError(f"Kokoro TTS: no se pudo cargar el modelo: {exc}") from exc
        return self._pipeline

    @staticmethod
    def _write_wav(pipeline, text: str, voice: str, speed: float, output_path: Path) -> None:
        """Convierte el generador de Kokoro a un WAV PCM 16-bit mono (24 kHz)."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        wrote_any = False
        with wave.open(str(output_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(_SAMPLE_RATE)
            for result in pipeline(text, voice=voice, speed=speed):
                if result.audio is None:
                    continue
                audio = result.audio.numpy()
                audio_bytes = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
                wav_file.writeframes(audio_bytes)
                wrote_any = True
        if not wrote_any:
            output_path.unlink(missing_ok=True)
            raise EngineError("Kokoro TTS: la síntesis no produjo audio.")

