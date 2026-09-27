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

#: Ventana de contexto que se pide a Ollama por defecto. La de Ollama (4096) se
#: agota con conversaciones largas y con documentos (una página de PDF con capa de
#: texto, o el resumen global de un documento entero): cuando el prompt no cabe,
#: Ollama responde HTTP 400 y la tarea muere. Se pide desde el payload
#: (``options.num_ctx``); el servidor lo respeta (comprobado con /api/ps).
DEFAULT_NUM_CTX = 8192


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


def fetch_loaded(base_url: str, timeout: float = 2.0) -> list[str]:
    """Nombres de los modelos cargados en memoria (``GET /api/ps``)."""
    resp = requests.get(f"{base_url.rstrip('/')}/api/ps", timeout=timeout)
    resp.raise_for_status()
    names: list[str] = []
    for model in resp.json().get("models", []):
        name = model.get("name") or model.get("model")
        if name:
            names.append(name)
    return names


def is_loaded(base_url: str, model: str, timeout: float = 2.0) -> bool:
    """¿``model`` está cargado en memoria ahora mismo?

    Ollama no tiene un "descarga si procede": pedir ``/api/generate`` con
    ``keep_alive: 0`` sobre un modelo que **no** está cargado hace que lo cargue
    para acto seguido liberarlo, con el coste de tiempo y VRAM que eso supone.

    Ante la duda (``/api/ps`` no responde) se devuelve ``False``: no se descarga
    nada, que es el comportamiento seguro.
    """
    try:
        loaded = fetch_loaded(base_url, timeout)
    except (requests.RequestException, ValueError):
        return False
    return any(
        name == model or name.startswith(f"{model}:") or model.startswith(f"{name}:")
        for name in loaded
    )


def http_error_message(resp, num_ctx: int | None = None) -> str:
    """Mensaje de error útil a partir de una respuesta de error de Ollama.

    El texto de ``raise_for_status()`` ("400 Client Error: Bad Request for url:
    ...") no dice nada: el motivo real viaja en el cuerpo (``{"error": "..."}``).
    El caso habitual con documentos es que el prompt no quepa en el contexto.
    """
    detalle = _error_detail(resp)
    if "exceeds the available context size" in detalle.lower():
        ventana = f"num_ctx={num_ctx}" if num_ctx is not None else "el contexto del servidor"
        return (
            "El texto no cabe en el contexto de Ollama: el prompt es más largo "
            f"que la ventana disponible ({ventana}). Acorta el mensaje o el "
            "documento, o sube el contexto ('num_ctx' en engines.json y/o "
            f"OLLAMA_CONTEXT_LENGTH en el servidor). Detalle de Ollama: {detalle}"
        )
    if detalle:
        return f"Error al hablar con Ollama: {detalle}"
    codigo = getattr(resp, "status_code", None)
    return f"Error al hablar con Ollama: HTTP {codigo if codigo else 'desconocido'}"


def _error_detail(resp) -> str:
    """Motivo que Ollama puso en el cuerpo de la respuesta (o ``text`` crudo)."""
    if resp is None:
        return ""
    try:
        data = resp.json()
    except ValueError:
        return (getattr(resp, "text", "") or "").strip()
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, str):
            return error.strip()
        if error is not None:
            return str(error)
    return ""
