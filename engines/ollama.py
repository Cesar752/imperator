"""Motor local Ollama (HTTP API en http://localhost:11434)."""

import logging

import requests

from engines.base_engine import ChatEngine, EngineError
from engines.ollama_api import fetch_tags

logger = logging.getLogger(__name__)


class OllamaEngine(ChatEngine):
    """Implementación de :class:`Engine` para un servidor local de Ollama."""

    #: Ventana de contexto que se pide al servidor. El valor por defecto de
    #: Ollama (4096) se agota con modelos que "piensan" y respuestas largas; al
    #: truncarse la generación, el contenido llega vacío.
    _NUM_CTX = 8192

    #: Tokens máximos de salida. ``-1`` = sin límite explícito: el techo real es
    #: la ventana de contexto (``num_ctx``). Un tope bajo corta las respuestas a
    #: media frase (el defecto del servidor, 128, incluso las dejaba vacías).
    _NUM_PREDICT = -1

    #: Aviso que se añade cuando el servidor cortó la respuesta por límite de
    #: tokens (``done_reason == "length"``), para que el corte no pase inadvertido.
    TRUNCATED_NOTE = "\n\n[respuesta incompleta: se alcanzó el límite de tokens]"

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        default_model: str = "qwen3.5:9b",
        num_ctx: int = _NUM_CTX,
        num_predict: int = _NUM_PREDICT,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._default_model = default_model
        self._num_ctx = int(num_ctx)
        self._num_predict = int(num_predict)
        self._last_used_model: str | None = None
        self._short_timeout = 2.0   # para /api/tags
        self._chat_timeout = 120.0  # para /api/chat

    @property
    def id(self) -> str:
        return "ollama"

    @property
    def nombre(self) -> str:
        return "Ollama"

    @property
    def default_model(self) -> str:
        """Modelo por defecto (de engines.json) usado cuando no se elige otro."""
        return self._default_model

    def is_available(self) -> bool:
        try:
            fetch_tags(self._base_url, self._short_timeout)
        except requests.RequestException:
            return False
        return True

    def list_models(self) -> list[str]:
        """Nombres de los modelos de chat (sin embeddings)."""
        return [model["name"] for model in self._fetch_models()]

    def list_model_details(self) -> list[dict]:
        """Detalles ``[{name, size}]`` (bytes) de los modelos de chat."""
        return [
            {"name": model["name"], "size": model.get("size")}
            for model in self._fetch_models()
        ]

    def _fetch_models(self) -> list[dict]:
        """Modelos de ``/api/tags`` excluyendo los de embeddings."""
        data = fetch_tags(self._base_url, self._short_timeout)
        return [
            model
            for model in data.get("models", [])
            if not self._is_embedding(model.get("name", ""))
        ]

    @staticmethod
    def _is_embedding(name: str) -> bool:
        """Devuelve ``True`` para modelos de embeddings (no de chat)."""
        return name.startswith("nomic-embed-") or "-embed-" in name

    def chat(self, messages: list[dict], model: str | None = None) -> str:
        model = model or self._default_model
        self._last_used_model = model
        logger.debug("Ollama chat: %d mensajes con '%s'", len(messages), model)
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_ctx": self._num_ctx,
                "num_predict": self._num_predict,
            },
        }
        try:
            resp = requests.post(
                f"{self._base_url}/api/chat",
                json=payload,
                timeout=self._chat_timeout,
            )
            resp.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            raise EngineError("No se pudo conectar a Ollama. ¿Está arrancado?") from exc
        except requests.exceptions.Timeout as exc:
            raise EngineError("Ollama tardó demasiado en responder (timeout).") from exc
        except requests.RequestException as exc:
            raise EngineError(f"Error al hablar con Ollama: {exc}") from exc

        try:
            data = resp.json()
        except ValueError as exc:
            raise EngineError(
                "Ollama devolvió una respuesta que no es JSON válido."
            ) from exc

        message = data.get("message") or {}
        content = (message.get("content") or "").strip()
        if not content:
            raise EngineError(self._empty_reply_reason(data, message))
        if str(data.get("done_reason") or "") == "length":
            logger.warning("Ollama: respuesta cortada por el límite de tokens")
            return f"{content}{self.TRUNCATED_NOTE}"
        logger.debug("Ollama chat OK: %d chars", len(content))
        return content

    def _empty_reply_reason(self, data: dict, message: dict) -> str:
        """Explica por qué Ollama respondió sin contenido.

        Ocurría en silencio: con la ventana de contexto llena (``truncated``) o
        con el presupuesto de salida gastado en el razonamiento, Ollama devuelve
        HTTP 200 con ``message.content`` vacío y el chat pintaba un turno en
        blanco.
        """
        if str(data.get("done_reason") or "") == "length":
            motivo = (
                f"se agotó el espacio de la respuesta "
                f"(num_predict={self._num_predict}) o de la ventana de contexto"
            )
        elif (message.get("thinking") or "").strip():
            motivo = "el modelo gastó la respuesta pensando (thinking)"
        else:
            motivo = "la respuesta llegó vacía"
        return (
            f"Ollama: {motivo}. Sube num_ctx/num_predict (ahora "
            f"num_ctx={self._num_ctx}, num_predict={self._num_predict}) o usa un "
            f"modelo sin razonamiento."
        )

    def unload_model(self) -> bool:
        """Descarga el modelo en uso de la VRAM de Ollama (``keep_alive: 0``)."""
        model = self._last_used_model or self._default_model
        if not model:
            return True
        try:
            requests.post(
                f"{self._base_url}/api/generate",
                json={"model": model, "keep_alive": 0},
                timeout=2,
            )
            logger.info("Ollama: modelo '%s' descargado de VRAM", model)
            return True
        except requests.RequestException as exc:
            logger.warning("No se pudo descargar '%s' de Ollama: %s", model, exc)
            return False
