"""Sondeo de recursos de ComfyUI (sin Qt).

Antes vivía dentro de ``ui/video_widget.py`` (con la URL de ComfyUI hardcodeada
en dos sitios). Aquí queda como función pura y testeable con ``requests``
mockeado, reutilizable por otras pestañas que hablen con ComfyUI.
"""

import requests

# Endpoint por defecto de ComfyUI (coincide con el de engines/comfyui_base.py).
DEFAULT_BASE_URL = "http://127.0.0.1:8188"

# Timeout de la consulta y antigüedad máxima aceptable de un dato cacheado.
STATS_TIMEOUT = 2
STALE_AFTER = 120


def read_ram_free(
    base_url: str | None = None, timeout: float = STATS_TIMEOUT
) -> float | None:
    """RAM libre en bytes según ``/system_stats``, o ``None`` si no se pudo leer.

    Nunca lanza: un ComfyUI caído o una respuesta corrupta se traducen en
    ``None`` para que la UI decida sin datos (y sin bloquear).
    """
    url = base_url or DEFAULT_BASE_URL
    try:
        resp = requests.get(f"{url}/system_stats", timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError):
        return None
    ram_free = (data.get("system") or {}).get("ram_free")
    return ram_free if isinstance(ram_free, (int, float)) else None


def is_fresh(
    timestamp: float | None, now: float, max_age: float = STALE_AFTER
) -> bool:
    """¿El dato leído en ``timestamp`` es lo bastante reciente para decidir?"""
    if timestamp is None:
        return False
    return (now - timestamp) < max_age
