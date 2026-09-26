"""Motor de imagen Black Forest Labs (API de pago, https://api.bfl.ai)."""

import base64
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import requests
from dotenv import load_dotenv

from engines.base_engine import Engine, EngineError

logger = logging.getLogger(__name__)

# Modelos de imagen expuestos por defecto (todos de pago).
_DEFAULT_MODELS: tuple[str, ...] = (
    "flux-kontext-pro",
    "flux-kontext-max",
    "flux-2-pro",
)

# Campo donde viaja la imagen de referencia según el modelo.
_IMAGE_FIELD = {
    "flux-kontext-pro": "image",
    "flux-kontext-max": "image",
    "flux-2-pro": "input_image",
}


class BFLApiEngine(Engine):
    """Genera imágenes llamando a la API de pago de Black Forest Labs."""

    _SHORT_TIMEOUT = 5.0
    _SUBMIT_TIMEOUT = 30.0
    _POLL_TIMEOUT = 10.0
    _DOWNLOAD_TIMEOUT = 120.0
    _GENERATE_TIMEOUT = 600.0
    _POLL_INTERVAL = 1.0

    def __init__(
        self,
        base_url: str = "https://api.bfl.ai",
        models: list[str] | tuple[str, ...] | None = None,
        api_key: str | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._models = list(models) if models else list(_DEFAULT_MODELS)
        self._api_key = api_key
        self._output_dir = (
            Path(output_dir)
            if output_dir is not None
            else Path(__file__).resolve().parent.parent / "output"
        )

    @property
    def id(self) -> str:
        return "bfl"

    @property
    def nombre(self) -> str:
        return "Black Forest Labs (API de pago)"

    def _get_api_key(self) -> str:
        if self._api_key:
            return self._api_key
        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
        return os.getenv("BFL_API_KEY", "")

    def _headers(self) -> dict[str, str]:
        return {
            "x-key": self._get_api_key(),
            "accept": "application/json",
            "Content-Type": "application/json",
        }

    def is_available(self) -> bool:
        return bool(self._get_api_key())

    def list_models(self) -> list[str]:
        return list(self._models)

    def get_credits(self) -> float | None:
        """Devuelve los créditos restantes de la cuenta (None si no se puede consultar)."""
        if not self._get_api_key():
            return None
        try:
            resp = requests.get(
                f"{self._base_url}/v1/credits",
                headers=self._headers(),
                timeout=self._SHORT_TIMEOUT,
            )
            resp.raise_for_status()
            credits = resp.json().get("credits")
            return float(credits) if credits is not None else None
        except (requests.RequestException, ValueError) as exc:
            logger.warning("Black Forest Labs: no se pudo consultar los créditos: %s", exc)
            return None

    def chat(self, messages: list[dict], model: str | None = None) -> str:
        raise EngineError("Black Forest Labs es un motor de imagen, no de chat.")

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        steps: int = 20,
        seed: int | None = None,
        guidance: float = 3.5,
        model: str | None = None,
        image_path: str | Path | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> Path:
        if not self._get_api_key():
            raise EngineError(
                "Black Forest Labs: falta la clave BFL_API_KEY (API de pago). "
                "Añádela al archivo .env del proyecto."
            )
        model = model or self._models[0]
        if model not in self._models:
            raise EngineError(f"Black Forest Labs: modelo desconocido '{model}'.")

        payload: dict = {"prompt": prompt}
        if width and height:
            payload["width"] = width
            payload["height"] = height
        if seed is not None:
            payload["seed"] = seed

        if image_path is not None:
            field = _IMAGE_FIELD.get(model, "image")
            payload[field] = self._image_value(image_path)

        polling_url = self._submit(model, payload)
        sample_url = self._wait(polling_url, cancel_check=cancel_check)
        return self._download_and_save(sample_url, seed or 0)

    # ---- Envío ----------------------------------------------------

    def _submit(self, model: str, payload: dict) -> str:
        try:
            resp = requests.post(
                f"{self._base_url}/v1/{model}",
                headers=self._headers(),
                json=payload,
                timeout=self._SUBMIT_TIMEOUT,
            )
        except requests.RequestException as exc:
            raise EngineError(f"Black Forest Labs: error al enviar: {exc}") from exc
        if resp.status_code >= 400:
            raise EngineError(
                f"Black Forest Labs: la API devolvió {resp.status_code}: "
                f"{resp.text[:300]}"
            )
        try:
            data = resp.json()
        except ValueError as exc:
            raise EngineError("Black Forest Labs: respuesta inválida al enviar.") from exc
        polling_url = data.get("polling_url")
        if not polling_url:
            raise EngineError("Black Forest Labs: la API no devolvió 'polling_url'.")
        return polling_url

    # ---- Polling --------------------------------------------------

    def _wait(
        self,
        polling_url: str,
        cancel_check: Callable[[], bool] | None = None,
    ) -> str:
        start = time.monotonic()
        while True:
            if cancel_check is not None and cancel_check():
                raise EngineError("Cancelado por el usuario")
            if time.monotonic() - start >= self._GENERATE_TIMEOUT:
                raise EngineError("Black Forest Labs: timeout esperando el resultado.")
            try:
                resp = requests.get(
                    polling_url, headers=self._headers(), timeout=self._POLL_TIMEOUT
                )
            except requests.RequestException as exc:
                raise EngineError(
                    f"Black Forest Labs: error consultando el estado: {exc}"
                ) from exc
            resp.raise_for_status()
            result = resp.json()
            status = result.get("status")
            if status == "Ready":
                sample = (result.get("result") or {}).get("sample")
                if not sample:
                    raise EngineError("Black Forest Labs: resultado sin URL de imagen.")
                return sample
            if status in ("Error", "Failed"):
                raise EngineError(
                    f"Black Forest Labs: la generación falló ({status})."
                )
            if status == "Request Moderated":
                raise EngineError(
                    "Black Forest Labs: la solicitud fue moderada (contenido no permitido)."
                )
            if status == "Task not found":
                raise EngineError(
                    "Black Forest Labs: la tarea no se encontró en el servidor."
                )
            time.sleep(self._POLL_INTERVAL)

    # ---- Imagen de referencia y descarga --------------------------

    def _image_value(self, image_path: str | Path) -> str:
        """Devuelve la imagen como URL (si ya es URL) o como data URI base64."""
        text = str(image_path)
        if text.startswith(("http://", "https://")):
            return text
        path = Path(image_path)
        if not path.exists():
            raise EngineError(f"Black Forest Labs: la imagen no existe: {path}")
        ext = path.suffix.lower()
        mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(
            ext, "image/png"
        )
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    def _download_and_save(self, url: str, seed: int) -> Path:
        try:
            resp = requests.get(url, timeout=self._DOWNLOAD_TIMEOUT)
            resp.raise_for_status()
            content = resp.content
        except requests.RequestException as exc:
            raise EngineError(
                f"Black Forest Labs: error descargando la imagen: {exc}"
            ) from exc
        self._output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"bfl_{datetime.now():%Y%m%d_%H%M%S}_{seed}.png"
        path = self._output_dir / filename
        with path.open("wb") as fh:
            fh.write(content)
        logger.info("Black Forest Labs: imagen guardada en '%s'.", path)
        return path
