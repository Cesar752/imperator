"""Motor de chat para APIs **compatibles con OpenAI** (OpenAI, DeepSeek, …).

Un único motor cubre varios proveedores: todos hablan ``POST {base_url}/chat/
completions`` con ``Authorization: Bearer <clave>``. La clave se lee **solo del
entorno** (``{API_ID}_API_KEY``, p. ej. ``OPENAI_API_KEY``), nunca de un JSON en
claro — misma política que el resto de la plataforma.

Es un plugin más del registro de motores: se configura en ``engines.json`` y se
elige como el resto (pestaña Chat). ``is_available()`` sólo comprueba que haya
clave; no hace red, para no bloquear el arranque.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Callable

import requests

from engines.base_engine import CANCEL_MESSAGE, ChatEngine, EngineError

logger = logging.getLogger(__name__)

#: Timeout por defecto de una petición de chat (segundos).
DEFAULT_TIMEOUT = 300.0

#: Timeout corto para listar modelos de un servidor local (``GET /models``): si
#: tarda o falla, se usan los modelos configurados sin cortar la UI.
MODELS_TIMEOUT = 4.0


class OpenAICompatibleChatEngine(ChatEngine):
    """Chat contra una API compatible con OpenAI (chat/completions)."""

    def __init__(
        self,
        api_id: str,
        base_url: str,
        default_model: str,
        *,
        nombre: str | None = None,
        models: list[str] | None = None,
        api_key_env: str | None = None,
        api_key_required: bool = True,
        timeout: float | None = None,
    ) -> None:
        self._api_id = api_id
        self._base_url = base_url.rstrip("/")
        self._default_model = default_model
        self._nombre = nombre or api_id.capitalize()
        # Modelos que se ofrecen en el selector. Si no se configura ninguno, sólo
        # aparece el modelo por defecto.
        self._models = list(models or ([default_model] if default_model else []))
        self._api_key_env = api_key_env or f"{api_id.upper()}_API_KEY"
        # Los servidores locales compatibles (llama.cpp, LM Studio, vLLM…) no
        # piden clave; las nubes (OpenAI/DeepSeek) sí.
        self._api_key_required = bool(api_key_required)
        self._timeout = float(timeout) if timeout is not None else DEFAULT_TIMEOUT

    # ---- Contrato Engine -------------------------------------------

    @property
    def id(self) -> str:
        return self._api_id

    @property
    def nombre(self) -> str:
        return self._nombre

    @property
    def default_model(self) -> str:
        return self._default_model

    def api_key(self) -> str:
        """Clave leída del entorno (``{API_ID}_API_KEY``)."""
        return os.getenv(self._api_key_env, "").strip()

    def is_available(self) -> bool:
        """``True`` si el motor puede usarse. No se hace red (no bloquear arranque).

        Con clave obligatoria exige que exista; en un servidor local (sin clave)
        se da por disponible.
        """
        if not self._api_key_required:
            return True
        return bool(self.api_key())

    def list_models(self) -> list[str]:
        """Modelos que ofrece el motor.

        En la **nube** (clave obligatoria) manda la lista curada de
        ``engines.json``. En un **servidor local** (llama.cpp, LM Studio, vLLM…,
        sin clave) se añaden los modelos que **sirve de verdad** el servidor
        (``GET {base_url}/models``): así el modelo cargado con ``llama-server``
        aparece en los selectores sin tener que escribirlo en la configuración.
        """
        nombres = list(self._models)
        if self._api_key_required:
            return nombres
        for nombre in self._server_models():
            if nombre not in nombres:
                nombres.append(nombre)
        return nombres

    def list_model_details(self) -> list[dict]:
        """Mismos detalles que espera el chat: ``[{name, size, installed}]``."""
        return [
            {"name": n, "size": None, "installed": True} for n in self.list_models()
        ]

    def _server_models(self) -> list[str]:
        """Ids que ofrece el servidor en ``GET {base_url}/models`` (o ``[]``).

        Tolerante a fallos: un servidor apagado, un endpoint sin ``/models`` o una
        respuesta rara devuelven lista vacía y se sigue con los configurados.
        """
        try:
            resp = requests.get(
                f"{self._base_url}/models",
                headers=self._headers(),
                timeout=MODELS_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError):
            return []
        items = data.get("data") if isinstance(data, dict) else None
        salida: list[str] = []
        for item in items or []:
            nombre = item.get("id") if isinstance(item, dict) else item
            if isinstance(nombre, str) and nombre.strip():
                salida.append(nombre.strip())
        return salida

    # ---- Chat ------------------------------------------------------

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        clave = self.api_key()
        if clave:
            headers["Authorization"] = f"Bearer {clave}"
        return headers

    def _endpoint(self) -> str:
        return f"{self._base_url}/chat/completions"

    def _require_key(self) -> None:
        if self._api_key_required and not self.api_key():
            raise EngineError(
                f"{self._nombre}: falta la clave de API. Defínela en el entorno "
                f"como {self._api_key_env} (fichero .env)."
            )

    def chat(self, messages: list[dict], model: str | None = None) -> str:
        """Petición única (sin streaming) a ``/chat/completions``."""
        self._require_key()
        model = model or self._default_model
        payload = {"model": model, "messages": messages, "stream": False}
        try:
            resp = requests.post(
                self._endpoint(), json=payload, headers=self._headers(), timeout=self._timeout
            )
            resp.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            raise EngineError(self._http_error(exc)) from exc
        except requests.exceptions.Timeout as exc:
            raise EngineError(f"{self._nombre}: timeout esperando la respuesta.") from exc
        except requests.RequestException as exc:
            raise EngineError(f"{self._nombre}: error de red: {exc}") from exc

        try:
            data = resp.json()
        except ValueError as exc:
            raise EngineError(f"{self._nombre}: respuesta no válida (no es JSON).") from exc
        return self._content_of(data)

    def chat_stream(
        self,
        messages: list[dict],
        model: str | None = None,
        *,
        on_token: Callable[[str, int], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> str:
        """Como :meth:`chat` pero con streaming (SSE) y cancelación."""
        self._require_key()
        model = model or self._default_model
        payload = {"model": model, "messages": messages, "stream": True}
        partes: list[str] = []
        total = 0
        try:
            with requests.post(
                self._endpoint(),
                json=payload,
                headers=self._headers(),
                timeout=self._timeout,
                stream=True,
            ) as resp:
                resp.raise_for_status()
                for linea in resp.iter_lines(decode_unicode=True):
                    if cancel_check is not None and cancel_check():
                        raise EngineError(CANCEL_MESSAGE)
                    if not linea or not linea.startswith("data:"):
                        continue
                    dato = linea[5:].strip()
                    if dato == "[DONE]":
                        break
                    delta = self._delta_of(dato)
                    if delta:
                        partes.append(delta)
                        total += len(delta)
                        if on_token is not None:
                            on_token(delta, total)
        except requests.exceptions.HTTPError as exc:
            raise EngineError(self._http_error(exc)) from exc
        except requests.exceptions.Timeout as exc:
            raise EngineError(f"{self._nombre}: timeout esperando la respuesta.") from exc
        except requests.RequestException as exc:
            raise EngineError(f"{self._nombre}: error de red: {exc}") from exc
        return "".join(partes)

    # ---- Internos --------------------------------------------------

    @staticmethod
    def _content_of(data: dict) -> str:
        try:
            return (data["choices"][0]["message"]["content"] or "").strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise EngineError(
                f"Respuesta inesperada de la API de chat: {exc}"
            ) from exc

    def _delta_of(self, dato: str) -> str:
        try:
            chunk = json.loads(dato)
        except ValueError:
            return ""
        try:
            return chunk["choices"][0]["delta"].get("content") or ""
        except (KeyError, IndexError, TypeError):
            return ""

    def _http_error(self, exc) -> str:
        """Mensaje útil: el motivo real viaja en el cuerpo de la respuesta."""
        detalle = ""
        resp = getattr(exc, "response", None)
        if resp is not None:
            try:
                detalle = (resp.json() or {}).get("error", {}).get("message", "")
            except (ValueError, AttributeError):
                detalle = (getattr(resp, "text", "") or "").strip()
        codigo = getattr(resp, "status_code", "?")
        return f"{self._nombre}: error HTTP {codigo}. {detalle}".strip()
