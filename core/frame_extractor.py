"""Extracción de fotogramas y audio de un vídeo con ffmpeg.

Utilidad local (no es un ``Engine``), hermana de
:class:`core.video_assembler.VideoAssembler`: la usa la tarea de Comprensión
para convertir un MP4 en fotogramas PNG y en un WAV mono que pasar a los
motores de percepción y a Whisper.

Convenciones heredadas del ensamblador:

- **Binario de ``imageio-ffmpeg``.** No se añade una dependencia nueva.
- **``timeout`` y kill.** ``subprocess.run`` mata el proceso al expirar: un
  ffmpeg atascado no puede colgar el hilo del worker ni hacer que la cancelación
  de la UI no sirva de nada.
- **Se comprueba la salida.** Un ffmpeg que devuelve 0 pero no escribe nada se
  trata como error, no como fotograma válido.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

try:
    import imageio_ffmpeg
except ImportError:  # pragma: no cover - dependencia declarada en requirements.txt
    imageio_ffmpeg = None

from core.video_assembler import VideoAssembler
from engines.base_engine import EngineError

logger = logging.getLogger(__name__)

#: Timeouts (segundos) por operación. Extraer un fotograma es rápido; separar el
#: audio de un vídeo largo puede tardar, de ahí la diferencia.
FRAME_TIMEOUT = 60
AUDIO_TIMEOUT = 900

#: Frecuencia de muestreo del audio extraído. 16 kHz mono es lo que espera
#: Whisper (y lo que menos pesa en disco).
AUDIO_SAMPLE_RATE = 16000


class FrameExtractor:
    """Extrae fotogramas y audio de un vídeo usando el ffmpeg de imageio."""

    def __init__(self, work_dir: str | Path | None = None) -> None:
        self._work_dir = Path(work_dir) if work_dir is not None else None

    @property
    def work_dir(self) -> Path | None:
        return self._work_dir

    # ---- Consultas -------------------------------------------------

    @staticmethod
    def _ffmpeg() -> str:
        if imageio_ffmpeg is None:
            raise EngineError(
                "imageio-ffmpeg no está instalado. "
                "Instálalo con: pip install imageio-ffmpeg>=0.6.0"
            )
        return imageio_ffmpeg.get_ffmpeg_exe()

    def probe(self, video_path: str | Path) -> dict:
        """``{duration, width, height}`` del vídeo (delega en el ensamblador)."""
        return VideoAssembler().probe(video_path)

    def duration(self, video_path: str | Path) -> float | None:
        """Duración en segundos, o ``None`` si ffmpeg no la detecta."""
        dato = self.probe(video_path).get("duration")
        try:
            return float(dato) if dato is not None else None
        except (TypeError, ValueError):
            return None

    # ---- Extracción ------------------------------------------------

    def extract_frame(
        self,
        video_path: str | Path,
        timestamp: float,
        output_path: str | Path,
    ) -> Path:
        """Escribe en ``output_path`` el fotograma del instante ``timestamp``.

        El ``-ss`` va **antes** de ``-i`` (búsqueda rápida por índice); para el
        muestreo del análisis es suficiente y evita decodificar desde el
        principio en cada fotograma.
        """
        video = self._require_file(video_path)
        destino = Path(output_path)
        destino.parent.mkdir(parents=True, exist_ok=True)
        instante = max(0.0, float(timestamp))

        self._run(
            [
                "-ss",
                f"{instante:.3f}",
                "-i",
                str(video),
                "-frames:v",
                "1",
                "-y",
                str(destino),
            ],
            timeout=FRAME_TIMEOUT,
            what="extraer el fotograma",
        )
        if not destino.exists() or destino.stat().st_size == 0:
            raise EngineError(
                f"ffmpeg no generó el fotograma de t={instante:g}s en '{destino.name}'."
            )
        logger.debug("Fotograma de t=%gs guardado en '%s'.", instante, destino)
        return destino

    def extract_audio(
        self, video_path: str | Path, output_path: str | Path
    ) -> Path:
        """Escribe en ``output_path`` el audio mono a ``AUDIO_SAMPLE_RATE`` Hz."""
        video = self._require_file(video_path)
        destino = Path(output_path)
        destino.parent.mkdir(parents=True, exist_ok=True)

        self._run(
            [
                "-i",
                str(video),
                "-vn",                    # sin vídeo
                "-ac",
                "1",                      # mono
                "-ar",
                str(AUDIO_SAMPLE_RATE),
                "-y",
                str(destino),
            ],
            timeout=AUDIO_TIMEOUT,
            what="extraer el audio",
        )
        if not destino.exists() or destino.stat().st_size == 0:
            raise EngineError(
                f"ffmpeg no generó el audio en '{destino.name}' "
                "(¿el vídeo no tiene pista de sonido?)."
            )
        return destino

    # ---- Internos --------------------------------------------------

    @staticmethod
    def _require_file(video_path: str | Path) -> Path:
        path = Path(video_path)
        if not path.is_file():
            raise EngineError(f"El vídeo no existe: {path}")
        return path

    def _run(self, args: list[str], *, timeout: int, what: str) -> None:
        ffmpeg = self._ffmpeg()
        comando = [ffmpeg, "-hide_banner", "-loglevel", "error", *args]
        try:
            subprocess.run(
                comando, capture_output=True, text=True, timeout=timeout, check=False
            )
        except subprocess.TimeoutExpired as exc:
            raise EngineError(
                f"ffmpeg no terminó de {what} en {timeout} s."
            ) from exc
        except OSError as exc:
            raise EngineError(f"No se pudo ejecutar ffmpeg para {what}: {exc}") from exc
