"""Motor local Ollama Vision (HTTP API en http://127.0.0.1:11434).

``127.0.0.1`` en vez de ``localhost``: el fallback de IPv6 de ``localhost``
cuesta ~2 s por petición (medido: 2 040 ms frente a 15 ms), y aquí se paga una
vez por página.
"""

import base64
import logging
from pathlib import Path

import requests

from engines.base_engine import EngineError, VisionEngine
from engines.ollama_api import (
    DEFAULT_VISION_NUM_CTX,
    fetch_tags,
    http_error_message,
    is_loaded,
)

logger = logging.getLogger(__name__)

#: URL y modelo por defecto del motor. Deben coincidir con
#: ``core.engine_manager._OLLAMA_VISION_DEFAULT_BASE_URL`` /
#: ``_OLLAMA_VISION_DEFAULT_MODEL`` (test-guarda: tests/test_ollama_defaults.py).
_DEFAULT_BASE_URL = "http://127.0.0.1:11434"
_DEFAULT_MODEL = "qwen2.5vl:7b"

# Marcadores (subcadenas, en minúsculas) para detectar modelos con capacidad de
# visión entre los expuestos por /api/tags.
VISION_MODEL_MARKERS = (
    "qwen3-vl",
    "qwen2.5vl",
    "qwen2-vl",
    "qwen-vl",
    "glm4v",
    "glm-4v",
    "glm-ocr",
    "minicpm-v",
    "llava",
    "gemma3",
    "llama3.2-vision",
    "deepseek-vl",
    "moondream",
    "vision",
)


class OllamaVisionEngine(VisionEngine):
    """Motor multimodal de Ollama: analiza una imagen y devuelve texto.

    Expone ``analyze_image()``, que codifica la imagen en base64 y la envía a
    ``/api/chat`` con el campo ``images`` (formato nativo de Ollama para modelos
    de visión). El OCR/resumen/extracción de tablas se orquesta desde la capa de
    UI (ver ``ui/document_widget.py``); aquí solo vive el primitivo "dime el
    texto que ves en esta imagen".
    """

    _SHORT_TIMEOUT = 2.0       # /api/tags

    #: Timeout de una petición de análisis. Generoso a propósito: los modelos de
    #: visión con razonamiento tardan minutos por página. Medido con
    #: ``qwen3-vl:8b`` y una página de 3000 caracteres: 41,5 s y 4143 tokens
    #: generados... para una respuesta de 390 caracteres, porque 13859 caracteres
    #: se fueron en "thinking". Con 180 s la tarea moría a mitad de documento.
    _ANALYZE_TIMEOUT = 600.0

    #: Aviso cuando Ollama cortó la respuesta por llenarse la ventana de contexto.
    TRUNCATED_NOTE = "\n\n[respuesta incompleta: se agotó la ventana de contexto]"

    #: Ventana de contexto que se pide al servidor. Sin pedirla, Ollama usa su
    #: 4096: el análisis de una página con capa de texto —o el resumen global de
    #: un documento entero— no cabía y Ollama respondía HTTP 400 ("request (4135
    #: tokens) exceeds the available context size (4096 tokens)"). Se pide más que
    #: en el chat porque el modelo razona y el razonamiento consume ventana.
    _NUM_CTX = DEFAULT_VISION_NUM_CTX

    def __init__(
        self,
        base_url: str = _DEFAULT_BASE_URL,
        default_model: str = _DEFAULT_MODEL,
        num_ctx: int = _NUM_CTX,
        timeout: float | None = None,
        think: bool | None = None,
        num_predict: int | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._default_model = default_model
        self._num_ctx = int(num_ctx)
        self._timeout = (
            float(timeout) if timeout is not None else self._ANALYZE_TIMEOUT
        )
        # ``think=False`` desactiva el razonamiento (los modelos qwen3-vl gastan
        # ~90 % de los tokens en «thinking»): se pide el campo ``think`` y se añade
        # el marcador ``/no_think`` por si la versión de Ollama lo ignora. ``None``
        # deja el comportamiento por defecto del modelo.
        self._think = think
        # Tope de tokens de salida, para acotar el coste del razonamiento.
        self._num_predict = int(num_predict) if num_predict is not None else None
        self._last_used_model: str | None = None

    @property
    def id(self) -> str:
        return "ollama_vision"

    @property
    def nombre(self) -> str:
        return "Ollama Vision"

    def is_available(self) -> bool:
        try:
            fetch_tags(self._base_url, self._SHORT_TIMEOUT)
        except requests.RequestException:
            return False
        return True

    def list_models(self) -> list[str]:
        data = fetch_tags(self._base_url, self._SHORT_TIMEOUT)
        return [
            model["name"]
            for model in data.get("models", [])
            if self._is_vision_model(model.get("name", ""))
        ]

    @staticmethod
    def _is_vision_model(name: str) -> bool:
        lowered = name.lower()
        return any(marker in lowered for marker in VISION_MODEL_MARKERS)

    def analyze_image(
        self, image_path: Path, prompt: str, model: str | None = None
    ) -> str:
        """Codifica la imagen en base64 y pide a Ollama que la analice."""
        image_path = Path(image_path)
        if not image_path.exists():
            raise EngineError(f"Ollama Vision: la imagen no existe: {image_path}")
        try:
            encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        except OSError as exc:
            raise EngineError(
                f"Ollama Vision: no se pudo leer la imagen: {exc}"
            ) from exc

        logger.debug("Ollama Vision: analizando '%s'", image_path.name)
        return self._chat(
            [{"role": "user", "content": prompt, "images": [encoded]}], model
        )

    def analyze_text(self, text: str, prompt: str, model: str | None = None) -> str:
        """Analiza texto ya extraído (capa de texto de un PDF) sin OCR.

        Mismo endpoint que :meth:`analyze_image` pero sin ``images``: resumir o
        extraer tablas de un PDF nativo no necesita rasterizar la página.
        """
        text = (text or "").strip()
        if not text:
            raise EngineError("Ollama Vision: no hay texto que analizar.")
        return self._chat([{"role": "user", "content": f"{prompt}\n\n{text}"}], model)

    def _chat(self, messages: list[dict], model: str | None) -> str:
        """Envía ``messages`` a ``/api/chat`` y devuelve el contenido de la respuesta."""
        model = model or self._default_model
        self._last_used_model = model
        options: dict = {"num_ctx": self._num_ctx}
        if self._num_predict is not None:
            options["num_predict"] = self._num_predict
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            # Sin esto Ollama usa su contexto por defecto (4096) y el documento
            # no cabe: ver _NUM_CTX.
            "options": options,
        }
        if self._think is not None:
            payload["think"] = bool(self._think)
            if self._think is False:
                messages = self._apply_no_think(messages)
                payload["messages"] = messages
        try:
            resp = requests.post(
                f"{self._base_url}/api/chat",
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            raise EngineError(
                "No se pudo conectar a Ollama. ¿Está arrancado?"
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise EngineError(
                "Ollama Vision: timeout al procesar el documento."
            ) from exc
        except requests.exceptions.HTTPError as exc:
            # El motivo real viaja en el cuerpo (ver ollama_api).
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
        razon = str(data.get("done_reason") or "")

        if not content:
            if (message.get("thinking") or "").strip() or razon == "length":
                # El modelo razonó y no le quedó espacio para escribir: mejor
                # explicarlo que devolver una página en blanco sin más.
                raise EngineError(self._no_content_reason(message, razon))
            # Página sin texto (en blanco): no es un error.
            return ""
        if razon == "length":
            logger.warning("Ollama Vision: respuesta cortada por el contexto")
            return f"{content}{self.TRUNCATED_NOTE}"
        logger.debug("Ollama Vision OK: %d chars", len(content))
        return content

    def _no_content_reason(self, message: dict, razon: str) -> str:
        """Explica por qué no hay respuesta (el razonamiento se comió el espacio).

        Tres casos, de más a menos frecuente:
        - ``num_predict`` puesto y el modelo razona: gasta el presupuesto de salida
          «pensando» y no le queda nada para responder (por fotograma).
        - se llena la ventana (``num_ctx``) antes de escribir.
        - el modelo pensó y no escribió (``done_reason`` distinto de ``length``).
        """
        pensado = len((message.get("thinking") or "").strip())
        if razon == "length" and pensado and self._num_predict:
            motivo = (
                f"el modelo gastó los {self._num_predict} tokens de salida "
                f"razonando ({pensado} caracteres de «thinking») y no le quedó "
                "espacio para responder"
            )
            pista = (
                "Quita 'num_predict' del motor o usa un modelo de visión sin "
                "razonamiento (p. ej. qwen2.5vl): los qwen3-vl pueden ignorar "
                "'think': false."
            )
        elif razon == "length":
            motivo = (
                f"se agotó la ventana de contexto (num_ctx={self._num_ctx}) antes "
                "de escribir la respuesta"
            )
            pista = (
                "Usa un modelo de visión sin razonamiento (p. ej. qwen2.5vl) o "
                "sube num_ctx en engines.json."
            )
        else:
            motivo = (
                "el modelo gastó la respuesta razonando "
                f"({pensado} caracteres de «thinking»)"
            )
            pista = (
                "Usa un modelo de visión sin razonamiento (p. ej. qwen2.5vl): los "
                "qwen3-vl pueden ignorar 'think': false."
            )
        return f"Ollama Vision: {motivo}. {pista}"

    #: Marcador «soft switch» de Qwen3 para apagar el razonamiento en el prompt.
    NO_THINK_HINT = "/no_think"

    def _apply_no_think(self, messages: list[dict]) -> list[dict]:
        """Añade el marcador ``/no_think`` al último turno (sin duplicarlo)."""
        if not messages:
            return messages
        copia = [dict(m) for m in messages]
        ultimo = copia[-1]
        contenido = str(ultimo.get("content") or "")
        if self.NO_THINK_HINT not in contenido:
            ultimo["content"] = f"{contenido} {self.NO_THINK_HINT}".strip()
        return copia

    def is_loaded(self, model: str | None = None) -> bool:
        """¿``model`` (o el del último análisis) está ya cargado en la VRAM?

        Mismo uso que en el chat: si el modelo de visión ya está cargado, la
        tarea no reserva memoria nueva y no debe bloquearse por falta de hueco.
        """
        target = model or self._last_used_model or self._default_model
        if not target:
            return False
        return is_loaded(self._base_url, target, self._SHORT_TIMEOUT)

    def unload_model(self) -> bool:
        """Descarga el modelo en uso de la VRAM de Ollama (``keep_alive: 0``).

        Solo si está realmente cargado: ``keep_alive: 0`` sobre un modelo que no
        está en memoria hace que Ollama lo cargue para descargarlo después.
        """
        model = self._last_used_model or self._default_model
        if not model:
            return True
        if not is_loaded(self._base_url, model, self._SHORT_TIMEOUT):
            logger.debug(
                "Ollama Vision: '%s' no está cargado; no hay nada que descargar",
                model,
            )
            return True
        try:
            requests.post(
                f"{self._base_url}/api/generate",
                json={"model": model, "keep_alive": 0},
                timeout=2,
            )
            logger.info("Ollama Vision: modelo '%s' descargado de VRAM", model)
            return True
        except requests.RequestException as exc:
            logger.warning("No se pudo descargar '%s' de Ollama: %s", model, exc)
            return False
