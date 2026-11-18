"""Caché de fotogramas extraídos de un vídeo.

Extraer un fotograma con ffmpeg cuesta (decodificación) y en Comprensión y en el
Cuaderno se extraen **los mismos** instantes una y otra vez (reintentos, varios
análisis del mismo vídeo). Esta caché guarda cada fotograma en disco, indexado
por vídeo (ruta + tamaño + mtime) y tiempo, y lo reutiliza.

Es lógica pura de ficheros (sin Qt y sin ffmpeg): quien extrae es
:class:`core.frame_extractor.FrameExtractor`; aquí sólo se decide *dónde* vive
cada fotograma y si ya existe.

La clave incluye el ``mtime``/tamaño del vídeo: si el fichero cambia, la clave
cambia y no se sirven fotogramas de una versión anterior.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from core.paths import CACHE

logger = logging.getLogger(__name__)

#: Carpeta por defecto de la caché de fotogramas (gitignored).
DEFAULT_CACHE_DIR = CACHE / "frames"


class FrameCache:
    """Guarda y localiza fotogramas de vídeo por (vídeo, instante)."""

    def __init__(self, root: str | Path | None = None) -> None:
        self._root = Path(root) if root is not None else DEFAULT_CACHE_DIR

    @property
    def root(self) -> Path:
        return self._root

    @staticmethod
    def video_key(video: str | Path) -> str:
        """Clave estable del vídeo (ruta + tamaño + mtime), en hexadecimal corto."""
        p = Path(video)
        try:
            st = p.stat()
            firma = f"{p.resolve()}|{st.st_size}|{st.st_mtime_ns}"
        except OSError:
            # Vídeo inexistente o inaccesible: clave por ruta (aún así determinista).
            firma = str(p)
        return hashlib.sha1(firma.encode("utf-8")).hexdigest()[:16]

    def dir_for(self, video: str | Path) -> Path:
        return self._root / self.video_key(video)

    def path_for(self, video: str | Path, t_s: float, suffix: str = ".png") -> Path:
        """Ruta (determinista) del fotograma de ``t_s`` para ``video``."""
        try:
            ms = int(round(max(0.0, float(t_s)) * 1000))
        except (TypeError, ValueError):
            ms = 0
        return self.dir_for(video) / f"t{ms:010d}{suffix}"

    def get(self, video: str | Path, t_s: float, suffix: str = ".png") -> Path | None:
        """Fotograma cacheado, o ``None`` si no existe (o está vacío)."""
        ruta = self.path_for(video, t_s, suffix)
        try:
            if ruta.is_file() and ruta.stat().st_size > 0:
                return ruta
        except OSError:
            return None
        return None

    def ensure_dir(self, video: str | Path) -> Path:
        carpeta = self.dir_for(video)
        carpeta.mkdir(parents=True, exist_ok=True)
        return carpeta

    # ---- Mantenimiento --------------------------------------------

    def stats(self) -> dict:
        """``{files, bytes}`` de toda la caché (tolerante a inexistencia)."""
        files = 0
        total = 0
        if not self._root.is_dir():
            return {"files": 0, "bytes": 0}
        for p in self._root.rglob("*.png"):
            try:
                total += p.stat().st_size
                files += 1
            except OSError:
                continue
        return {"files": files, "bytes": total}

    def prune(self, max_files: int = 20000) -> int:
        """Borra los fotogramas más antiguos si se supera ``max_files``.

        Devuelve cuántos se borraron. Pensada para llamarse al arrancar (la caché
        puede crecer con muchos vídeos); no borra nada por debajo del tope.
        """
        if not self._root.is_dir():
            return 0
        ficheros = [p for p in self._root.rglob("*.png") if p.is_file()]
        exceso = len(ficheros) - max(1, int(max_files))
        if exceso <= 0:
            return 0
        ficheros.sort(key=lambda p: p.stat().st_mtime)
        borrados = 0
        for p in ficheros[:exceso]:
            try:
                p.unlink()
                borrados += 1
            except OSError:
                continue
        if borrados:
            logger.info("Caché de fotogramas: %d ficheros antiguos borrados.", borrados)
        return borrados
