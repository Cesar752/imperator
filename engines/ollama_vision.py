"""Motor local Ollama Vision (HTTP API en http://localhost:11434)."""

import base64
import logging
from pathlib import Path

import requests

from engines.base_engine import EngineError, VisionEngine
from engines.ollama_api import fetch_tags

logger = logging.getLogger(__name__)

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
    _ANALYZE_TIMEOUT = 180.0   # /api/chat (los PDFs grandes tardan)

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        default_model: str = "qwen3-vl:8b",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._default_model = default_model
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
        payload = {"model": model, "messages": messages, "stream": False}
        try:
            resp = requests.post(
                f"{self._base_url}/api/chat",
                json=payload,
                timeout=self._ANALYZE_TIMEOUT,
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
        except requests.RequestException as exc:
            raise EngineError(f"Error al hablar con Ollama: {exc}") from exc

        try:
            data = resp.json()
        except ValueError as exc:
            raise EngineError(
                "Ollama devolvió una respuesta que no es JSON válido."
            ) from exc

        content = data.get("message", {}).get("content", "") or ""
        logger.debug("Ollama Vision OK: %d chars", len(content))
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
            logger.info("Ollama Vision: modelo '%s' descargado de VRAM", model)
            return True
        except requests.RequestException as exc:
            logger.warning("No se pudo descargar '%s' de Ollama: %s", model, exc)
            return False
