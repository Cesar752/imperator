"""Motor local Ollama (HTTP API en http://localhost:11434)."""

import logging

import requests

from engines.base_engine import Engine, EngineError
from engines.ollama_api import fetch_tags

logger = logging.getLogger(__name__)


class OllamaEngine(Engine):
    """Implementación de :class:`Engine` para un servidor local de Ollama."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        default_model: str = "qwen3.5:9b",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._default_model = default_model
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
        payload = {"model": model, "messages": messages, "stream": False}
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

        content = data.get("message", {}).get("content", "")
        logger.debug("Ollama chat OK: %d chars", len(content))
        return content

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
