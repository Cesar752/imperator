"""Retención de lo que la app genera en ``output/``.

Cada imagen, vídeo, audio o SRT se queda ahí para siempre: la auditoría encontró
180 ficheros (110,9 MB) sin ninguna política de limpieza y el único tope era el
disco. Aquí vive la decisión (lógica pura, sin Qt y testeable) y su ejecución.

Reglas:

- Solo se miran ficheros **de la carpeta de salidas** con las extensiones que
  escribe la app (``.png .jpg .jpeg .webp .mp4 .wav .mp3 .srt``). Cualquier otra
  extensión se ignora por completo: si alguien guarda algo suyo ahí, no se toca.
- Nada de las últimas ``keep_recent_hours`` horas se borra (por defecto 24): lo
  que acabas de generar sigue ahí aunque el límite esté superado.
- Primero se borra lo que supera ``max_age_days``; después, lo más antiguo
  primero, hasta cumplir ``max_files`` y ``max_mb``.

Los valores por defecto son una **red de seguridad**, no una purga: con 500
ficheros, 90 días y 8 GB, la primera ejecución no borra nada en un ``output/``
normal (el de la auditoría tenía 180 ficheros, 11 días y 110 MB).
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

#: Extensiones que escribe la plataforma (todo lo demás se respeta).
APP_EXTENSIONS: tuple[str, ...] = (
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".mp4",
    ".wav",
    ".mp3",
    ".srt",
)

#: Máximo de ficheros que se conservan (los más antiguos se borran).
DEFAULT_MAX_FILES = 500
#: Edad máxima de un fichero de salida.
DEFAULT_MAX_AGE_DAYS = 90
#: Tamaño máximo del conjunto (MB).
DEFAULT_MAX_MB = 8192
#: Gracia para lo recién generado: nunca se borra.
DEFAULT_KEEP_RECENT_HOURS = 24


def _as_int(valor, por_defecto: int) -> int:
    """Convierte ``valor`` a int; si no se puede, devuelve ``por_defecto``.

    ``settings.json`` es editable a mano, así que aquí puede llegar cualquier
    cosa (texto, ``None``); la limpieza no debe romper el arranque por eso.
    """
    try:
        return int(valor)
    except (TypeError, ValueError):
        return por_defecto


@dataclass(frozen=True)
class CleanupPlan:
    """Qué se va a borrar (o se ha borrado) y por qué no se borra el resto."""

    to_delete: tuple[Path, ...] = ()
    kept: int = 0
    ignored: int = 0
    bytes_to_free: int = 0

    @property
    def count(self) -> int:
        return len(self.to_delete)

    @property
    def mb(self) -> float:
        return self.bytes_to_free / (1024 * 1024)

    def describe(self) -> str:
        """Frase corta para el log o la barra de estado."""
        if not self.count:
            return (
                f"nada que borrar ({self.kept} ficheros conservados, "
                f"{self.ignored} ignorados)"
            )
        return (
            f"{self.count} ficheros antiguos ({self.mb:.1f} MB), "
            f"{self.kept} conservados, {self.ignored} ignorados"
        )


def plan_cleanup(
    files: Iterable[Path],
    *,
    max_files: int = DEFAULT_MAX_FILES,
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    max_mb: int = DEFAULT_MAX_MB,
    keep_recent_hours: int = DEFAULT_KEEP_RECENT_HOURS,
    now: datetime | None = None,
) -> CleanupPlan:
    """Decide qué ficheros de ``files`` se borran (función pura)."""
    momento = now if now is not None else datetime.now()
    gracia = timedelta(hours=max(0, _as_int(keep_recent_hours, DEFAULT_KEEP_RECENT_HOURS)))
    edad_maxima = timedelta(days=max(0, _as_int(max_age_days, DEFAULT_MAX_AGE_DAYS)))
    tope_ficheros = max(0, _as_int(max_files, DEFAULT_MAX_FILES))
    tope_bytes = max(0, _as_int(max_mb, DEFAULT_MAX_MB)) * 1024 * 1024

    candidatos: list[tuple[datetime, int, Path]] = []
    ignorados = 0
    for ruta in files:
        ruta = Path(ruta)
        if ruta.suffix.lower() not in APP_EXTENSIONS:
            ignorados += 1
            continue
        try:
            info = ruta.stat()
        except OSError:
            # Ya no está (o no se puede leer): no hay nada que planificar.
            ignorados += 1
            continue
        fecha = datetime.fromtimestamp(info.st_mtime)
        if momento - fecha < gracia:
            ignorados += 1
            continue
        candidatos.append((fecha, info.st_size, ruta))

    # Del más antiguo al más nuevo: el orden en el que se decide borrar.
    candidatos.sort(key=lambda item: (item[0], str(item[2])))

    to_delete: list[Path] = []
    liberados = 0
    restantes: list[tuple[datetime, int, Path]] = []
    for fecha, tamano, ruta in candidatos:
        if momento - fecha > edad_maxima:
            to_delete.append(ruta)
            liberados += tamano
        else:
            restantes.append((fecha, tamano, ruta))

    total = sum(tamano for _, tamano, _ in restantes)
    indice = 0
    while indice < len(restantes) and (
        len(restantes) - indice > tope_ficheros or total > tope_bytes
    ):
        _, tamano, ruta = restantes[indice]
        to_delete.append(ruta)
        liberados += tamano
        total -= tamano
        indice += 1

    return CleanupPlan(
        to_delete=tuple(to_delete),
        kept=len(restantes) - indice,
        ignored=ignorados,
        bytes_to_free=liberados,
    )


def cleanup_output(
    directory: str | Path,
    *,
    max_files: int = DEFAULT_MAX_FILES,
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    max_mb: int = DEFAULT_MAX_MB,
    keep_recent_hours: int = DEFAULT_KEEP_RECENT_HOURS,
    now: datetime | None = None,
) -> CleanupPlan:
    """Aplica la política a ``directory`` y borra lo que toque.

    Nunca lanza: si la carpeta no existe o un fichero está bloqueado, se avisa en
    el log y se sigue (la limpieza no puede impedir el arranque).
    """
    carpeta = Path(directory)
    if not carpeta.is_dir():
        return CleanupPlan()

    try:
        ficheros = [p for p in carpeta.iterdir() if p.is_file()]
    except OSError as exc:
        logger.warning("No se pudo leer la carpeta de salidas '%s': %s", carpeta, exc)
        return CleanupPlan()

    plan = plan_cleanup(
        ficheros,
        max_files=max_files,
        max_age_days=max_age_days,
        max_mb=max_mb,
        keep_recent_hours=keep_recent_hours,
        now=now,
    )
    borrados = 0
    for ruta in plan.to_delete:
        try:
            ruta.unlink()
            borrados += 1
        except OSError as exc:
            logger.warning("No se pudo borrar '%s': %s", ruta, exc)

    if borrados:
        logger.info(
            "Limpieza de salidas en '%s': %d ficheros, %.1f MB liberados.",
            carpeta,
            borrados,
            plan.mb,
        )
    return plan
