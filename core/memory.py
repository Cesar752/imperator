"""Higiene de memoria del proceso entre tareas pesadas (sin Qt).

La generación de vídeo por escenas encadena varios clips **en el mismo proceso**.
ComfyUI libera sus modelos tras cada clip (``POST /free`` dentro de
``generate_video``), pero el proceso de Python no devuelve al sistema operativo
la memoria que deja de usar: el recolector libera objetos, pero las arenas de
``pymalloc`` y los búferes grandes de ``requests`` (el MP4 descargado de cada
escena) quedan en el *working set*. En una secuencia larga el proceso parece —y,
para el planificador de Windows, es— cada vez más grande, hasta que la máquina
se queda sin RAM y el sistema mata la aplicación.

:func:`release_memory` cierra ese hueco entre escena y escena:

1. ``gc.collect()`` fuerza la recolección de ciclos (los búferes grandes sueltan
   sus páginas de inmediato, pero los ciclos necesitan un pase completo).
2. En Windows, ``EmptyWorkingSet(GetCurrentProcess())`` devuelve al SO las
   páginas **ociosas** del proceso (no libera memoria en uso, solo la que ya no
   se referencia). Es el equivalente a «limpiar el búfer».

Es *best-effort* y **nunca lanza**: no poder limpiar no debe romper una
generación. La parte dependiente del SO es inyectable para poder testear el
contrato sin tocar el working set real.
"""

from __future__ import annotations

import ctypes
import gc
import logging
import sys
from typing import Callable

logger = logging.getLogger(__name__)


def trim_working_set() -> bool:
    """Devuelve al SO la RAM ociosa del proceso en Windows; ``False`` si no se pudo.

    Se usan dos vías porque no siempre está disponible la misma API:
    ``psapi.EmptyWorkingSet`` (lo más directo) y, como respaldo,
    ``kernel32.SetProcessWorkingSetSize(handle, -1, -1)``. En otros sistemas no
    se hace nada y se devuelve ``False``.
    """
    if sys.platform != "win32":
        return False
    try:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetCurrentProcess()
    except (AttributeError, OSError) as exc:  # pragma: no cover - depende del SO
        logger.debug("Sin acceso a kernel32 para recortar el working set: %s", exc)
        return False
    try:
        if ctypes.windll.psapi.EmptyWorkingSet(handle):
            return True
    except (AttributeError, OSError):
        pass
    try:
        # ``-1`` en ambos tamaños es la forma documentada de "lo mínimo posible".
        return bool(
            kernel32.SetProcessWorkingSetSize(
                handle, ctypes.c_size_t(-1), ctypes.c_size_t(-1)
            )
        )
    except (AttributeError, OSError, OverflowError) as exc:  # pragma: no cover
        logger.debug("No se pudo recortar el working set: %s", exc)
        return False


def release_memory(*, trim: Callable[[], bool] | None = None) -> int:
    """Recolecta ciclos y devuelve al SO la RAM ociosa del proceso.

    Devuelve el número de objetos recolectados (útil para logs y tests). Nunca
    lanza: el recorte del working set es una optimización, no un requisito.
    ``trim`` permite inyectar un doble en los tests para no tocar el proceso.
    """
    collected = int(gc.collect())
    trimer = trim if trim is not None else trim_working_set
    try:
        trimer()
    except Exception as exc:  # noqa: BLE001 - liberar no debe romper nada
        logger.debug("Fallo al recortar el working set: %s", exc)
    return collected
