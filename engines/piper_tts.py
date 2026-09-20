"""Motor local Piper TTS (texto a voz con modelos ONNX en voices/)."""

import logging
import re
import wave
from datetime import datetime
from pathlib import Path

try:
    from piper import PiperVoice, SynthesisConfig
except ImportError:  # pragma: no cover - dependencia declarada en requirements.txt
    PiperVoice = None
    SynthesisConfig = None

from engines.base_engine import Engine, EngineError

logger = logging.getLogger(__name__)

# Raíz del proyecto (C:\Imperator); voices/ y output/ cuelgan de aquí.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# TODO(docs): documentar en README/PLAN cómo descargar voces de Piper para
# usuarios nuevos (p. ej. `python -m piper.download_voices es_ES-davefx-medium`).


class PiperTTSEngine(Engine):
    """Sintetiza texto a WAV usando voces de Piper (ONNX locales en voices/)."""

    def __init__(
        self,
        voices_dir: str | Path = "voices",
        default_voice: str | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self._voices_dir = self._resolve(voices_dir)
        self._default_voice = default_voice
        self._output_dir = (
            Path(output_dir) if output_dir is not None else PROJECT_ROOT / "output"
        )
        # Cache de voces ya cargadas (clave: nombre de voz) para no recargar
        # el .onnx en cada síntesis.
        self._loaded: dict[str, PiperVoice] = {}

    @staticmethod
    def _resolve(path) -> Path:
        p = Path(path)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def id(self) -> str:
        return "piper_tts"

    @property
    def nombre(self) -> str:
        return "Piper TTS"

    def is_available(self) -> bool:
        if PiperVoice is None:
            return False
        if not self._voices_dir.is_dir():
            return False
        return bool(self.list_voices())

    def list_models(self) -> list[str]:
        # Alias del contrato Engine (list_models) hacia list_voices.
        return self.list_voices()

    def list_voices(self) -> list[str]:
        if not self._voices_dir.is_dir():
            return []
        return sorted(
            p.stem for p in self._voices_dir.glob("*.onnx") if p.is_file()
        )

    def chat(self, messages: list[dict]) -> str:
        raise EngineError("Piper TTS es un motor de texto a voz, no de chat.")

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        length_scale: float = 1.0,
        output_path: Path | None = None,
    ) -> Path:
        """Sintetiza ``text`` y devuelve la ruta del WAV generado."""
        if PiperVoice is None:
            raise EngineError(
                "Piper no está instalado. Instálalo con: pip install piper-tts>=1.8.0"
            )
        text = (text or "").strip()
        if not text:
            raise EngineError("El texto a sintetizar está vacío.")

        voice_name = voice or self._default_voice or self._first_voice()
        if voice_name is None:
            raise EngineError("No hay voces disponibles en voices/.")

        model_path = self._voices_dir / f"{voice_name}.onnx"
        if not model_path.exists():
            raise EngineError(f"Piper TTS: la voz no existe: {model_path}")

        if output_path is None:
            output_path = self._default_output_path(voice_name)

        piper_voice = self._get_voice(voice_name, model_path)
        syn_config = SynthesisConfig(length_scale=length_scale)
        self._write_wav(piper_voice, text, syn_config, output_path)

        logger.info("Piper TTS: audio guardado en '%s'.", output_path)
        return output_path

    # ---- Internos ------------------------------------------------

    def _first_voice(self) -> str | None:
        voices = self.list_voices()
        return voices[0] if voices else None

    def _default_output_path(self, voice_name: str) -> Path:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        safe_voice = re.sub(r"[^A-Za-z0-9_-]", "_", voice_name)
        filename = f"plataformaia_{datetime.now():%Y%m%d_%H%M%S}_{safe_voice}.wav"
        return self._output_dir / filename

    def _get_voice(self, voice_name: str, model_path: Path) -> PiperVoice:
        cached = self._loaded.get(voice_name)
        if cached is not None:
            return cached
        logger.info("Piper TTS: cargando voz '%s'...", voice_name)
        try:
            piper_voice = PiperVoice.load(model_path)
        except Exception as exc:  # noqa: BLE001
            raise EngineError(
                f"Piper TTS: no se pudo cargar la voz '{voice_name}': {exc}"
            ) from exc
        self._loaded[voice_name] = piper_voice
        return piper_voice

    @staticmethod
    def _write_wav(piper_voice, text: str, syn_config, output_path: Path) -> None:
        """Convierte el generador de AudioChunk de Piper a un WAV en disco."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        wrote_any = False
        try:
            with wave.open(str(output_path), "wb") as wav_file:
                for chunk in piper_voice.synthesize(text, syn_config):
                    if not wrote_any:
                        wav_file.setframerate(chunk.sample_rate)
                        wav_file.setsampwidth(chunk.sample_width)
                        wav_file.setnchannels(chunk.sample_channels)
                        wrote_any = True
                    wav_file.writeframes(chunk.audio_int16_bytes)
        except EngineError:
            raise
        except Exception as exc:  # noqa: BLE001
            output_path.unlink(missing_ok=True)
            raise EngineError(f"Piper TTS: error al sintetizar: {exc}") from exc

        if not wrote_any:
            output_path.unlink(missing_ok=True)
            raise EngineError("Piper TTS: la síntesis no produjo audio.")
