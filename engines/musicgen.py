"""Motor local MusicGen: música a partir de una descripción de texto.

Meta MusicGen (``facebook/musicgen-*``) genera audio musical desde un prompt.
Es un motor **opcional**: ``audiocraft`` (y por debajo torch y ffmpeg) es una
dependencia pesada, así que el import va protegido y la app arranca igual sin
ella; el motor simplemente aparece como no disponible.

El modelo pesa cientos de MB y se descarga en la primera generación, así que se
carga de forma diferida y se cachea por tamaño.
"""

import logging
from datetime import datetime
from pathlib import Path

try:
    from audiocraft.data.audio import audio_write
    from audiocraft.models import MusicGen
except Exception:  # noqa: BLE001 - cadena de imports (torch/ffmpeg pueden fallar)
    MusicGen = None
    audio_write = None

from engines.base_engine import EngineError, MusicEngine

logger = logging.getLogger(__name__)

# Raíz del proyecto; output/ cuelga de aquí.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Tamaños publicados de MusicGen. El pequeño es el único que corre en CPU con una
# paciencia razonable, así que es el que se usa por defecto.
MODEL_SIZES = ("small", "medium", "large")
DEFAULT_MODEL = "small"

#: Repositorio de HuggingFace de cada tamaño.
REPO_IDS = {
    "small": "facebook/musicgen-small",
    "medium": "facebook/musicgen-medium",
    "large": "facebook/musicgen-large",
}

DEFAULT_DURATION = 8.0
MIN_DURATION = 1.0
MAX_DURATION = 30.0


class MusicGenEngine(MusicEngine):
    """Genera música con MusicGen (audiocraft) y la guarda como WAV."""

    def __init__(
        self,
        model_size: str = DEFAULT_MODEL,
        output_dir=None,
        duration: float | None = None,
    ) -> None:
        self._model_size = model_size if model_size in MODEL_SIZES else DEFAULT_MODEL
        self._output_dir = (
            Path(output_dir) if output_dir is not None else PROJECT_ROOT / "output"
        )
        self._duration = (
            float(duration) if duration is not None else DEFAULT_DURATION
        )
        # Cache de modelos ya cargados (clave: tamaño) para no volver a descargar
        # ni reconstruir la red en cada generación.
        self._loaded: dict[str, object] = {}

    @property
    def id(self) -> str:
        return "musicgen"

    @property
    def nombre(self) -> str:
        return "MusicGen"

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    def is_available(self) -> bool:
        return MusicGen is not None

    def list_models(self) -> list[str]:
        return list(MODEL_SIZES)

    def generate_music(
        self,
        prompt: str,
        model: str | None = None,
        duration: float | None = None,
        output_path: Path | str | None = None,
    ) -> Path:
        """Genera música según ``prompt`` y devuelve la ruta del WAV creado.

        ``model`` elige el tamaño (``small``/``medium``/``large``) y ``duration``
        los segundos de audio; ambos caen a los valores del motor si no se pasan.
        """
        if MusicGen is None:
            raise EngineError(
                "audiocraft no está instalado. Instálalo con: "
                "pip install audiocraft (requiere torch y ffmpeg)."
            )
        prompt = (prompt or "").strip()
        if not prompt:
            raise EngineError("La descripción de la música está vacía.")

        size = model if model in MODEL_SIZES else self._model_size
        seconds = self._clamp_duration(
            self._duration if duration is None else duration
        )
        target = self._resolve_output_path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        music_model = self._get_model(size)
        try:
            music_model.set_generation_params(duration=seconds)
            generated = music_model.generate([prompt])
            self._write_wav(music_model, generated, target)
        except EngineError:
            raise
        except Exception as exc:  # noqa: BLE001
            target.unlink(missing_ok=True)
            raise EngineError(f"MusicGen: error al generar: {exc}") from exc

        if not target.exists() or target.stat().st_size == 0:
            raise EngineError("MusicGen: la generación no produjo audio.")
        logger.info("MusicGen: audio guardado en '%s'.", target)
        return target

    # ---- Internos ------------------------------------------------

    @staticmethod
    def _clamp_duration(seconds) -> float:
        """Acota la duración al rango que MusicGen maneja (1-30 s)."""
        try:
            value = float(seconds)
        except (TypeError, ValueError):
            return DEFAULT_DURATION
        return min(max(value, MIN_DURATION), MAX_DURATION)

    def _resolve_output_path(self, output_path) -> Path:
        """Normaliza el destino a un ``.wav`` (``audio_write`` lo añade él)."""
        path = (
            Path(output_path)
            if output_path is not None
            else self._default_output_path()
        )
        if path.suffix.lower() != ".wav":
            path = path.with_suffix(".wav")
        return path

    def _default_output_path(self) -> Path:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        return self._output_dir / f"musicgen_{datetime.now():%Y%m%d_%H%M%S}.wav"

    def _get_model(self, size: str):
        cached = self._loaded.get(size)
        if cached is not None:
            return cached
        logger.info(
            "MusicGen: cargando '%s' (la primera vez descarga el modelo)...",
            REPO_IDS[size],
        )
        try:
            model = MusicGen.get_pretrained(REPO_IDS[size])
        except Exception as exc:  # noqa: BLE001
            raise EngineError(
                f"MusicGen: no se pudo cargar el modelo '{size}': {exc}"
            ) from exc
        self._loaded[size] = model
        return model

    @staticmethod
    def _write_wav(music_model, generated, target: Path) -> None:
        """Escribe el tensor generado como WAV con ``audiocraft.audio_write``.

        ``audio_write`` recibe la ruta *sin* extensión y añade ``.wav``, así que
        se le pasa la base del destino ya normalizado en :meth:`_resolve_output_path`.
        """
        audio_write(
            str(target.with_suffix("")),
            generated[0].cpu(),
            music_model.sample_rate,
            strategy="loudness",
        )
