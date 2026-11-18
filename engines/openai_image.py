"""Motor de imagen para la API de imágenes de OpenAI (p. ej. ``gpt-image-1``).

Genera una imagen a partir de un prompt (``POST {base_url}/images/generations``)
y la guarda en ``output/``. La respuesta puede traer la imagen en base64
(``b64_json``, lo de ``gpt-image-1``) o como URL (modelos tipo ``dall-e``); se
soportan las dos.

La clave se lee **solo del entorno** (``OPENAI_API_KEY``, la misma que el motor de
chat ``openai``), nunca de un JSON en claro. Al ser opcional, si no hay clave el
motor aparece como no disponible y no rompe el arranque.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import requests

from engines.base_engine import EngineError, ImageEngine

logger = logging.getLogger(__name__)

#: Raíz del proyecto (C:\Imperator); output/ cuelga de aquí.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-image-1"
DEFAULT_SIZE = "1024x1024"
DEFAULT_TIMEOUT = 300.0


class OpenAIImageEngine(ImageEngine):
    """Genera imágenes con la API de imágenes de OpenAI."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        default_model: str = DEFAULT_MODEL,
        *,
        nombre: str = "OpenAI Imagen",
        models: list[str] | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        output_dir: str | Path | None = None,
        size: str = DEFAULT_SIZE,
        timeout: float | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._default_model = default_model
        self._nombre = nombre
        self._models = list(models or ([default_model] if default_model else []))
        self._api_key_env = api_key_env
        self._output_dir = (
            Path(output_dir) if output_dir is not None else PROJECT_ROOT / "output"
        )
        self._size = size
        self._timeout = float(timeout) if timeout is not None else DEFAULT_TIMEOUT

    # ---- Contrato Engine -------------------------------------------

    @property
    def id(self) -> str:
        return "openai_image"

    @property
    def nombre(self) -> str:
        return self._nombre

    @property
    def default_model(self) -> str:
        return self._default_model

    def api_key(self) -> str:
        return os.getenv(self._api_key_env, "").strip()

    def is_available(self) -> bool:
        """``True`` si hay clave. No hace red (no bloquear el arranque)."""
        return bool(self.api_key())

    def list_models(self) -> list[str]:
        return list(self._models)

    # ---- Generación ------------------------------------------------

    def generate_image(self, prompt: str, **kwargs) -> Path:
        """Genera una imagen y devuelve la ruta del fichero creado."""
        clave = self.api_key()
        if not clave:
            raise EngineError(
                f"{self._nombre}: falta la clave de API. Defínela en el entorno "
                f"como {self._api_key_env} (fichero .env)."
            )
        prompt = (prompt or "").strip()
        if not prompt:
            raise EngineError("El prompt de la imagen está vacío.")

        model = kwargs.get("model") or self._default_model
        size = kwargs.get("size") or self._size
        payload = {"model": model, "prompt": prompt, "size": size, "n": 1}
        headers = {"Authorization": f"Bearer {clave}", "Content-Type": "application/json"}
        try:
            resp = requests.post(
                f"{self._base_url}/images/generations",
                json=payload,
                headers=headers,
                timeout=self._timeout,
            )
            resp.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            raise EngineError(self._http_error(exc)) from exc
        except requests.exceptions.Timeout as exc:
            raise EngineError(f"{self._nombre}: timeout generando la imagen.") from exc
        except requests.RequestException as exc:
            raise EngineError(f"{self._nombre}: error de red: {exc}") from exc

        destino = self._output_path(kwargs.get("output_path"))
        self._save_response(resp, destino)
        logger.info("%s: imagen guardada en '%s'.", self._nombre, destino)
        return destino

    # ---- Internos --------------------------------------------------

    def _output_path(self, output_path) -> Path:
        if output_path is not None:
            destino = Path(output_path)
            destino.parent.mkdir(parents=True, exist_ok=True)
            return destino
        self._output_dir.mkdir(parents=True, exist_ok=True)
        marca = f"{datetime.now():%Y%m%d_%H%M%S}"
        return self._output_dir / f"plataformaia_{marca}_{self.id}.png"

    def _save_response(self, resp, destino: Path) -> None:
        try:
            data = resp.json()
        except ValueError as exc:
            raise EngineError(f"{self._nombre}: respuesta no válida (no es JSON).") from exc
        items = data.get("data") or []
        if not items:
            raise EngineError(f"{self._nombre}: la API no devolvió ninguna imagen.")
        primero = items[0]
        if primero.get("b64_json"):
            try:
                destino.write_bytes(base64.b64decode(primero["b64_json"]))
            except (ValueError, TypeError) as exc:
                raise EngineError(f"{self._nombre}: base64 inválido.") from exc
            return
        if primero.get("url"):
            self._download(primero["url"], destino)
            return
        raise EngineError(f"{self._nombre}: la respuesta no trae 'b64_json' ni 'url'.")

    def _download(self, url: str, destino: Path) -> None:
        try:
            resp = requests.get(url, timeout=self._timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise EngineError(f"{self._nombre}: no se pudo descargar la imagen: {exc}") from exc
        destino.write_bytes(resp.content)

    def _http_error(self, exc) -> str:
        detalle = ""
        resp = getattr(exc, "response", None)
        if resp is not None:
            try:
                cuerpo = resp.json()
                detalle = (cuerpo or {}).get("error", {}).get("message", "") or json.dumps(cuerpo)
            except (ValueError, AttributeError):
                detalle = (getattr(resp, "text", "") or "").strip()
        codigo = getattr(resp, "status_code", "?")
        return f"{self._nombre}: error HTTP {codigo}. {detalle}".strip()
