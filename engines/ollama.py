"""Motor local Ollama (HTTP API en http://127.0.0.1:11434).

Se usa ``127.0.0.1`` y no ``localhost``: en Windows ``localhost`` intenta primero
``::1`` (Ollama no escucha ahí), la conexión se rechaza y el fallback a IPv4
cuesta ~2 s por petición (medido en la auditoría: 2 040 ms frente a 15 ms).
"""

import json
import logging
import threading
from typing import Callable

import requests

from engines.base_engine import CANCEL_MESSAGE, ChatEngine, EngineError
from engines.ollama_api import (
    DEFAULT_NUM_CTX,
    _OllamaCommon,
    fetch_tags,
    http_error_message,
)

logger = logging.getLogger(__name__)

#: URL y modelo por defecto del motor. Tienen que coincidir con
#: ``core.engine_manager._OLLAMA_DEFAULT_BASE_URL`` / ``_OLLAMA_DEFAULT_MODEL``
#: (y con ``core.config_manager.DEFAULT_ENGINES`` y la plantilla de la UI):
#: tests/test_ollama_defaults.py lo comprueba para que no vuelvan a divergir.
#: Divergían: aquí ponía ``qwen3.5:9b`` mientras el resto del proyecto usaba
#: ``qwen3.6:35b``, así que construir el motor sin modelo daba un 404 de Ollama.
_DEFAULT_BASE_URL = "http://127.0.0.1:11434"
_DEFAULT_MODEL = "qwen3.6:35b"


class OllamaEngine(_OllamaCommon, ChatEngine):
    """Implementación de :class:`Engine` para un servidor local de Ollama."""

    #: Ventana de contexto que se pide al servidor. El valor por defecto de
    #: Ollama (4096) se agota con modelos que "piensan" y respuestas largas; al
    #: truncarse la generación, el contenido llega vacío.
    _NUM_CTX = DEFAULT_NUM_CTX

    #: Tokens máximos de salida. ``-1`` = sin límite explícito: el techo real es
    #: la ventana de contexto (``num_ctx``). Un tope bajo corta las respuestas a
    #: media frase (el defecto del servidor, 128, incluso las dejaba vacías).
    _NUM_PREDICT = -1

    #: Aviso que se añade cuando el servidor cortó la respuesta por límite de
    #: tokens (``done_reason == "length"``), para que el corte no pase inadvertido.
    TRUNCATED_NOTE = "\n\n[respuesta incompleta: se alcanzó el límite de tokens]"

    #: Timeout de una petición de chat (``/api/chat``). Generoso a propósito:
    #: los modelos que "piensan" (qwen3.x) gastan la mayor parte del tiempo en
    #: ``thinking``, y una tarea como el **guion de escenas** (prompt de sistema
    #: largo + esquema JSON) genera miles de tokens antes de responder. Medido
    #: con ``qwen3.6:35b``: a los 120 s iba por **6709 tokens a ~100 t/s** y el
    #: cliente lo cortó, así que la generación del guion moría con "Ollama tardó
    #: demasiado en responder (timeout)". Configurable por motor en
    #: ``engines.json`` (``ollama.parametros.timeout``). Contrapartida: la
    #: petición es bloqueante, así que mientras dura no atiende a la cancelación
    #: (se resolvería con streaming, pendiente).
    _CHAT_TIMEOUT = 600.0

    #: Marcador «soft switch» de Qwen3 para apagar el razonamiento desde el prompt.
    #: Se añade cuando ``think=False`` por si la versión de Ollama ignora el campo.
    NO_THINK_HINT = "/no_think"

    def __init__(
        self,
        base_url: str = _DEFAULT_BASE_URL,
        default_model: str = _DEFAULT_MODEL,
        num_ctx: int = _NUM_CTX,
        num_predict: int = _NUM_PREDICT,
        timeout: float | None = None,
        think: bool | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._default_model = default_model
        self._num_ctx = int(num_ctx)
        self._num_predict = int(num_predict)
        # ``think=False`` desactiva el razonamiento de los modelos qwen3.x (que
        # gastan la ventana «pensando» y devuelven vacío). ``None`` deja el
        # comportamiento por defecto del modelo. Configurable por motor.
        self._think = think
        #: Último modelo usado (lo lee ``unload_model``). El motor es una instancia
        #: única compartida entre workers y el hilo de UI, así que el campo va bajo
        #: lock: *Liberar VRAM ahora* puede leerlo mientras un chat lo escribe
        #: (auditoría del 30/09/2026).
        self._last_used_model: str | None = None
        self._lock = threading.Lock()
        self._short_timeout = 2.0  # para /api/tags
        # para /api/chat; configurable por motor en engines.json
        self._chat_timeout = (
            float(timeout) if timeout is not None else self._CHAT_TIMEOUT
        )

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
        with self._lock:
            self._last_used_model = model
        logger.debug("Ollama chat: %d mensajes con '%s'", len(messages), model)
        if self._think is False:
            messages = self._apply_no_think(messages)
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_ctx": self._num_ctx,
                "num_predict": self._num_predict,
            },
        }
        if self._think is not None:
            payload["think"] = bool(self._think)
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
            raise EngineError(
                "Ollama tardó demasiado en responder (timeout tras "
                f"{self._chat_timeout:.0f} s). Los modelos que razonan pueden "
                "necesitar más tiempo: súbelo con 'timeout' en los 'parametros' "
                "del motor 'ollama' de engines.json."
            ) from exc
        except requests.exceptions.HTTPError as exc:
            # El motivo real viaja en el cuerpo de la respuesta (ver ollama_api).
            raise EngineError(
                http_error_message(exc.response, self._num_ctx)
            ) from exc
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

    def chat_stream(
        self,
        messages: list[dict],
        model: str | None = None,
        *,
        on_token: "Callable[[str, int], None] | None" = None,
        cancel_check: "Callable[[], bool] | None" = None,
    ) -> str:
        """``chat`` con **streaming**: informa del avance y cancela al instante.

        Con ``stream: true`` Ollama manda la respuesta por líneas NDJSON. Al leer
        cada línea se consulta ``cancel_check``: si devuelve ``True`` se lanza
        :data:`CANCEL_MESSAGE` y, al cerrarse la conexión, **Ollama detiene la
        generación**. Antes, cancelar el chat no hacía nada hasta que llegara la
        respuesta completa (o el timeout).

        ``on_token(delta, total_chars)`` recibe cada fragmento de ``content`` (el
        ``thinking`` no se emite: no forma parte de la respuesta).
        """
        model = model or self._default_model
        with self._lock:
            self._last_used_model = model
        logger.debug(
            "Ollama chat_stream: %d mensajes con '%s'", len(messages), model
        )
        if self._think is False:
            messages = self._apply_no_think(messages)
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "num_ctx": self._num_ctx,
                "num_predict": self._num_predict,
            },
        }
        if self._think is not None:
            payload["think"] = bool(self._think)
        chunks: list[str] = []
        total = 0
        done_reason = ""
        try:
            with requests.post(
                f"{self._base_url}/api/chat",
                json=payload,
                timeout=self._chat_timeout,
                stream=True,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines(decode_unicode=True):
                    if cancel_check is not None and cancel_check():
                        raise EngineError(CANCEL_MESSAGE)
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except ValueError:
                        continue
                    message = data.get("message") or {}
                    delta = message.get("content") or ""
                    if delta:
                        chunks.append(delta)
                        total += len(delta)
                        if on_token is not None:
                            on_token(delta, total)
                    if data.get("done"):
                        done_reason = str(data.get("done_reason") or "")
                        break
        except requests.exceptions.ConnectionError as exc:
            raise EngineError("No se pudo conectar a Ollama. ¿Está arrancado?") from exc
        except requests.exceptions.Timeout as exc:
            raise EngineError(
                "Ollama tardó demasiado en responder (timeout tras "
                f"{self._chat_timeout:.0f} s). Los modelos que razonan pueden "
                "necesitar más tiempo: súbelo con 'timeout' en los 'parametros' "
                "del motor 'ollama' de engines.json."
            ) from exc
        except requests.exceptions.HTTPError as exc:
            raise EngineError(
                http_error_message(exc.response, self._num_ctx)
            ) from exc
        except requests.RequestException as exc:
            raise EngineError(f"Error al hablar con Ollama: {exc}") from exc

        content = "".join(chunks).strip()
        if not content:
            raise EngineError(
                "Ollama: la respuesta llegó vacía. Puede que el modelo gastara la "
                "ventana 'pensando' (sube num_ctx en engines.json) o que se cortara "
                "la conexión."
            )
        if done_reason == "length":
            logger.warning("Ollama: respuesta cortada por el límite de tokens")
            return f"{content}{self.TRUNCATED_NOTE}"
        logger.debug("Ollama chat_stream OK: %d chars", len(content))
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

