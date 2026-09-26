"""Ayuda compartida para hablar con la API HTTP de Ollama.

Ollama tarda ~2 s en responder ``GET /api/tags`` (con 13 modelos, recalcula
tamaños en disco en cada llamada). Sin caché, el arranque de Imperator lo
llama 3 veces de forma síncrona (Chat, Documentos y barra de estado) y bloquea
~6 s. Esta caché corta (TTL) colapsa esas llamadas en una sola.
"""

import time

import requests

# Caché a nivel de módulo de /api/tags, clave por base_url.
_TAGS_CACHE: dict[str, tuple[float, dict]] = {}
_TAGS_TTL = 30.0  # segundos


def fetch_tags(base_url: str, timeout: float = 2.0) -> dict:
    """Devuelve el JSON de ``GET /api/tags`` con una caché corta (TTL)."""
    base_url = base_url.rstrip("/")
    now = time.monotonic()
    cached = _TAGS_CACHE.get(base_url)
    if cached is not None and (now - cached[0]) < _TAGS_TTL:
        return cached[1]
    resp = requests.get(f"{base_url}/api/tags", timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    _TAGS_CACHE[base_url] = (now, data)
    return data
