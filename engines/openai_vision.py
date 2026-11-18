"""Motor de **visión** para APIs compatibles con OpenAI (GPT-4o, Gemini…).

Los motores de chat de la plataforma (``openai``, ``deepseek``, ``llama_cpp``…) sólo
escriben texto. Para que la **Crónica** (y los documentos) cuenten con una IA de
pago o con «más visión» —que mire los fotogramas de verdad en vez de conformarse
con un modelo local pequeño— hace falta un motor **multimodal** que sepa enviar
imágenes a ``/chat/completions``.

Este plugin implementa el mismo contrato que :class:`~engines.ollama_vision.
OllamaVisionEngine` (:class:`~engines.base_engine.VisionEngine`), así que se puede
elegir en cualquier sitio que pida «motor de visión» (Crónica, Documentos…):

- ``analyze_image()``: codifica la imagen como *data URL* (``data:image/png;base64``)
  y la manda en el mensaje del usuario, el formato que aceptan OpenAI y los
  proveedores compatibles (Google Gemini por su endpoint compatible, xAI, vLLM…).
- ``analyze_text()``: análisis sin imagen, para hojas que ya traen texto.

OJO: el **modelo** tiene que ser **multimodal**. Los modelos de sólo texto no ven
imágenes: ``deepseek-chat`` / ``deepseek-reasoner`` (DeepSeek v4) **no tienen
visión**, así que no valen aquí; para ellos está el motor de **chat**, que sí
escribe la crónica. Alternativas de visión de pago, sin cambiar de motor: Google
Gemini por su endpoint compatible, xAI Grok (con visión), Qwen-VL por su endpoint
compatible… basta con apuntar ``ruta`` a su URL base y poner en ``models`` un
nombre con visión. En local, DeepSeek-VL sí se detecta como visión en Ollama
(``ollama_vision``), porque ahí sí es multimodal.

La clave se lee **sólo del entorno** (``{API_ID}_API_KEY``, p. ej.
``OPENAI_API_KEY``), nunca de un JSON en claro: misma política que el resto de la
plataforma. ``is_available()`` no hace red (no bloquea el arranque); sólo comprueba
que la clave exista.

Aviso de coste: a diferencia de Ollama, una API de pago **cobra por fotograma
analizado**. La Crónica avisa de esto al elegir uno de estos motores; cuantos más
fotogramas y más «máx.», más caro sale generar la crónica.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
from pathlib import Path

import requests

from engines.base_engine import EngineError, VisionEngine

logger = logging.getLogger(__name__)

#: Timeout por defecto de una imagen (segundos). Generoso: algunas APIs tardan en
#: analizar una imagen grande y en encadenar reintentos.
DEFAULT_TIMEOUT = 300.0

#: ``detail`` de OpenAI (``low`` | ``high`` | ``auto``). ``low`` abarata mucho el
#: análisis de un fotograma (y basta para «¿qué se ve?»); ``high`` mira con más
#: detalle y cuesta más. Configurable por motor (``parametros.detail``).
DEFAULT_DETAIL = "auto"

#: Timeout corto para listar modelos de un servidor local compatible (``GET /models``).
MODELS_TIMEOUT = 4.0

#: Extensiones que se mandan como ``image/jpeg`` aunque el sufijo no lo diga.
_MIME_FALLBACK = "image/png"


class OpenAICompatibleVisionEngine(VisionEngine):
    """Visión contra una API compatible con OpenAI (``/chat/completions``)."""

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
        detail: str = DEFAULT_DETAIL,
        max_tokens: int | None = None,
    ) -> None:
        self._api_id = api_id
        self._base_url = base_url.rstrip("/")
        self._default_model = default_model
        self._nombre = nombre or f"{api_id.replace('_', ' ').title()} Vision"
        # Modelos que se ofrecen en el selector. Si no se configura ninguno, sólo
        # aparece el modelo por defecto.
        self._models = list(models or ([default_model] if default_model else []))
        self._api_key_env = api_key_env or f"{api_id.upper()}_API_KEY"
        # Los servidores locales compatibles (vLLM, LM Studio multimodal) no piden
        # clave; las nubes (OpenAI, Gemini-compat, xAI) sí.
        self._api_key_required = bool(api_key_required)
        self._timeout = float(timeout) if timeout is not None else DEFAULT_TIMEOUT
        self._detail = str(detail or DEFAULT_DETAIL)
        self._max_tokens = int(max_tokens) if max_tokens else None
        self._last_used_model: str | None = None

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
        """``True`` si el motor puede usarse. No se hace red (no bloquear arranque)."""
        if not self._api_key_required:
            return True
        return bool(self.api_key())

    def list_models(self) -> list[str]:
        """Modelos del motor: los configurados y, en un servidor LOCAL con visión
        (vLLM/LM Studio multimodal, sin clave), los que sirve de verdad.

        En la nube (clave obligatoria) manda la lista configurada.
        """
        nombres = list(self._models)
        if self._default_model and self._default_model not in nombres:
            nombres.insert(0, self._default_model)
        if not self._api_key_required:
            for nombre in self._server_models():
                if nombre not in nombres:
                    nombres.append(nombre)
        return nombres

    def list_model_details(self) -> list[dict]:
        """Mismos detalles que espera el chat: ``[{name, size, installed}]``."""
        return [{"name": n, "size": None, "installed": True} for n in self.list_models()]

    def _server_models(self) -> list[str]:
        """Ids que ofrece el servidor en ``GET {base_url}/models`` (o ``[]``)."""
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

    # ---- Visión ----------------------------------------------------

    def analyze_image(
        self, image_path: Path, prompt: str, model: str | None = None
    ) -> str:
        """Envía ``image_path`` (como *data URL*) y devuelve el texto del modelo."""
        image_path = Path(image_path)
        if not image_path.is_file():
            raise EngineError(f"{self._nombre}: la imagen no existe: {image_path}")
        try:
            encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        except OSError as exc:
            raise EngineError(
                f"{self._nombre}: no se pudo leer la imagen: {exc}"
            ) from exc

        mime = mimetypes.guess_type(image_path.name)[0] or _MIME_FALLBACK
        logger.debug(
            "%s: analizando '%s' (%s, %d KB)",
            self._nombre,
            image_path.name,
            mime,
            len(encoded) // 1024,
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{encoded}",
                            "detail": self._detail,
                        },
                    },
                ],
            }
        ]
        return self._chat(messages, model)

    def analyze_text(self, text: str, prompt: str, model: str | None = None) -> str:
        """Analiza texto ya extraído (capa de texto de un PDF) sin imagen."""
        text = (text or "").strip()
        if not text:
            raise EngineError(f"{self._nombre}: no hay texto que analizar.")
        messages = [{"role": "user", "content": f"{prompt}\n\n{text}"}]
        return self._chat(messages, model)

    # ---- Internos ---------------------------------------------------

    def _chat(self, messages: list[dict], model: str | None) -> str:
        self._require_key()
        modelo = model or self._default_model
        self._last_used_model = modelo
        payload: dict = {"model": modelo, "messages": messages, "stream": False}
        if self._max_tokens:
            payload["max_tokens"] = self._max_tokens
        try:
            resp = requests.post(
                self._endpoint(),
                json=payload,
                headers=self._headers(),
                timeout=self._timeout,
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
            raise EngineError(
                f"{self._nombre}: respuesta no válida (no es JSON)."
            ) from exc
        return self._content_of(data)

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

    @staticmethod
    def _content_of(data: dict) -> str:
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise EngineError(
                f"Respuesta inesperada de la API de visión: {exc}"
            ) from exc
        # Algunos proveedores devuelven la respuesta como lista de partes.
        if isinstance(content, list):
            return "".join(
                str(parte.get("text", ""))
                for parte in content
                if isinstance(parte, dict)
            ).strip()
        return str(content or "").strip()

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
