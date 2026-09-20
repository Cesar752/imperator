"""Motor local Whisper (faster-whisper) para transcripción de audio a SRT."""

import logging
from pathlib import Path
from typing import Callable

try:
    from faster_whisper import WhisperModel
except ImportError:  # pragma: no cover - dependencia declarada en requirements.txt
    WhisperModel = None

from engines.base_engine import Engine, EngineError

logger = logging.getLogger(__name__)

# Tamaños de modelo disponibles (faster-whisper los descarga/cachea en disco).
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3"]


class WhisperEngine(Engine):
    """Transcribe audio a SRT con faster-whisper (backend CTranslate2).

    Por defecto usa ``device="cpu"`` y ``compute_type="int8"``: CTranslate2 puede
    fallar con CUDA en GPUs muy nuevas (p. ej. RTX 5080, sm_120), así que ambos
    se dejan configurables para activar GPU más adelante cuando sea estable.
    """

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        # Caché en memoria: no recargar el modelo entre transcripciones.
        self._model = None           # WhisperModel cargado (None hasta el primer uso)
        self._loaded_size: str | None = None

    @property
    def id(self) -> str:
        return "whisper"

    @property
    def nombre(self) -> str:
        return "Whisper"

    def is_available(self) -> bool:
        return WhisperModel is not None

    def list_models(self) -> list[str]:
        return list(WHISPER_MODELS)

    def chat(self, messages: list[dict]) -> str:
        raise EngineError("Whisper es un motor de transcripción, no de chat.")

    def transcribe(
        self,
        audio_path: str | Path,
        language: str | None = None,
        output_srt: str | Path | None = None,
        progress_callback: Callable[[int, str], None] | None = None,
        model_size: str | None = None,
    ) -> Path:
        """Transcribe ``audio_path`` y devuelve la ruta del SRT generado.

        ``language=None`` deja que faster-whisper autodetecte el idioma; pasar
        ``"es"`` (u otro código ISO 639-1) fuerza ese idioma. Si ``output_srt``
        es None, el SRT se guarda junto al audio con extensión ``.srt``.
        ``model_size`` permite cambiar el modelo sin reconstruir el motor (la UI
        lo expone como selector); el modelo se cachea en memoria y solo se
        recarga si cambia el tamaño.
        """
        if WhisperModel is None:
            raise EngineError(
                "faster-whisper no está instalado. "
                "Instálalo con: pip install faster-whisper>=1.0.0"
            )
        audio_path = Path(audio_path)
        if not audio_path.is_file():
            raise EngineError(f"El audio a transcribir no existe: {audio_path}")

        if model_size is not None:
            self._model_size = model_size
        model = self._get_model()

        logger.info(
            "Whisper: transcribiendo '%s' (modelo=%s, idioma=%s)...",
            audio_path.name, self._model_size, language or "auto",
        )
        try:
            segments, info = model.transcribe(str(audio_path), language=language)
        except Exception as exc:  # noqa: BLE001 - CTranslate2/IO
            raise EngineError(f"Whisper: no se pudo transcribir: {exc}") from exc

        srt_path = (
            Path(output_srt)
            if output_srt is not None
            else audio_path.with_suffix(".srt")
        )
        srt_path.parent.mkdir(parents=True, exist_ok=True)

        total_duration = float(getattr(info, "duration", 0.0) or 0.0)
        wrote = self._write_srt(segments, srt_path, total_duration, progress_callback)
        if not wrote:
            srt_path.unlink(missing_ok=True)
            raise EngineError("Whisper: la transcripción no produjo subtítulos.")

        logger.info("Whisper: subtítulos guardados en '%s'.", srt_path)
        return srt_path

    # ---- Internos ------------------------------------------------

    def _get_model(self):
        if self._model is not None and self._loaded_size == self._model_size:
            return self._model
        logger.info(
            "Whisper: cargando modelo '%s' (device=%s, compute_type=%s)...",
            self._model_size, self._device, self._compute_type,
        )
        try:
            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
            )
        except Exception as exc:  # noqa: BLE001
            raise EngineError(
                f"Whisper: no se pudo cargar el modelo '{self._model_size}': {exc}"
            ) from exc
        self._loaded_size = self._model_size
        return self._model

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """Formatea segundos como ``HH:MM:SS,mmm`` (estándar SRT)."""
        total_ms = max(0, int(round(seconds * 1000)))
        hours, total_ms = divmod(total_ms, 3_600_000)
        minutes, total_ms = divmod(total_ms, 60_000)
        secs, ms = divmod(total_ms, 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"

    def _write_srt(
        self,
        segments,
        srt_path: Path,
        total_duration: float,
        progress_callback: Callable[[int, str], None] | None,
    ) -> bool:
        wrote_any = False
        try:
            with srt_path.open("w", encoding="utf-8") as fh:
                written = 0
                for segment in segments:
                    text = (segment.text or "").strip()
                    if not text:
                        continue
                    written += 1
                    start = float(segment.start)
                    end = float(segment.end)
                    fh.write(f"{written}\n")
                    fh.write(
                        f"{self._format_timestamp(start)} --> "
                        f"{self._format_timestamp(end)}\n"
                    )
                    fh.write(f"{text}\n\n")
                    wrote_any = True
                    if progress_callback is not None and total_duration > 0:
                        percent = min(99, int(100 * end / total_duration))
                        progress_callback(percent, "Transcribiendo...")
        except Exception as exc:  # noqa: BLE001
            raise EngineError(f"Whisper: error al escribir el SRT: {exc}") from exc
        return wrote_any
