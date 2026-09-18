"""Motor local Ollama (HTTP API en http://localhost:11434)."""

import logging

import requests

from engines.base_engine import Engine, EngineError

logger = logging.getLogger(__name__)


class OllamaEngine(Engine):
    """Implementación de :class:`Engine` para un servidor local de Ollama."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3.5:9b",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._short_timeout = 2.0   # para /api/tags
        self._chat_timeout = 120.0  # para /api/chat

    @property
    def id(self) -> str:
        return "ollama"

    @property
    def nombre(self) -> str:
        return "Ollama"

    def is_available(self) -> bool:
        try:
            resp = requests.get(
                f"{self._base_url}/api/tags", timeout=self._short_timeout
            )
            resp.raise_for_status()
        except requests.RequestException:
            return False
        return True

    def list_models(self) -> list[str]:
        resp = requests.get(
            f"{self._base_url}/api/tags", timeout=self._short_timeout
        )
        resp.raise_for_status()
        data = resp.json()
        return [model["name"] for model in data.get("models", [])]

    def chat(self, messages: list[dict]) -> str:
        logger.debug("Ollama chat: %d mensajes", len(messages))
        payload = {"model": self._model, "messages": messages, "stream": False}
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
