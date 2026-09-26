"""Tronco común para motores que hablan con la API HTTP de ComfyUI.

``ComfyUIEngine`` (imagen) y ``WanVideoEngine`` (vídeo) comparten el mismo ciclo
de vida: cargar un workflow JSON, parchear sus nodos, subir una imagen opcional,
enviar ``POST /prompt``, esperar el resultado (cada uno a su manera), descargar
la salida vía ``/view`` y guardarla en ``output/``. Esta clase aloja ese tronco
y deja a cada motor solo lo que le es propio: el parcheo de nodos y la espera.

No se instancia directamente (es abstracta mientras no se implementen ``id`` y
``nombre``): se usa como mixin junto a ``ImageEngine``/``VideoEngine``.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import requests

from engines.base_engine import Engine, EngineError

logger = logging.getLogger(__name__)


class ComfyUIBaseEngine(Engine):
    """Tronco compartido de los motores basados en ComfyUI."""

    #: Etiqueta usada en los mensajes de error (la sobrescribe cada motor).
    _LABEL = "ComfyUI"

    # Timeouts por defecto (los motores pueden sobrescribirlos).
    _SHORT_TIMEOUT = 2.0        # /system_stats, /object_info, /history (lectura)
    _SUBMIT_TIMEOUT = 30.0      # POST /prompt
    _VIEW_TIMEOUT = 120.0       # GET /view (descarga del resultado)
    _UPLOAD_TIMEOUT = 60.0      # POST /upload/image

    def __init__(
        self,
        base_url: str,
        workflows: dict[str, str] | None = None,
        default_model: str | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        if not workflows:
            raise EngineError(
                f"{self._LABEL}: falta 'workflows' en la config de engines.json"
            )
        self._base_url = base_url.rstrip("/")
        self._workflows = {name: Path(path) for name, path in workflows.items()}
        self._default_model = default_model or next(iter(self._workflows))
        # Carpeta de salida por defecto: <raíz del proyecto>/output.
        self._output_dir = (
            Path(output_dir)
            if output_dir is not None
            else Path(__file__).resolve().parent.parent / "output"
        )

    @property
    def base_url(self) -> str:
        return self._base_url

    # ---- Ciclo de vida ------------------------------------------

    def list_models(self) -> list[str]:
        # Cada modelo/modo es una clave del diccionario "workflows".
        return list(self._workflows.keys())

    def is_available(self) -> bool:
        try:
            resp = requests.get(
                f"{self._base_url}/system_stats", timeout=self._SHORT_TIMEOUT
            )
            resp.raise_for_status()
        except requests.RequestException:
            return False
        return True

    def cancel_remote(self) -> None:
        """Solicita a ComfyUI detener la ejecución en curso (POST /interrupt)."""
        try:
            requests.post(f"{self._base_url}/interrupt", timeout=2)
            logger.info("ComfyUI: interrupción solicitada")
        except requests.RequestException as exc:
            logger.warning("No se pudo interrumpir ComfyUI: %s", exc)

    def unload_model(self) -> bool:
        """Libera VRAM/RAM de ComfyUI (POST /free)."""
        try:
            requests.post(
                f"{self._base_url}/free",
                json={"unload_models": True, "free_memory": True},
                timeout=5,
            )
            logger.info("ComfyUI: modelos liberados")
            return True
        except requests.RequestException as exc:
            logger.warning("No se pudo liberar ComfyUI: %s", exc)
            return False

    # ---- Workflow ------------------------------------------------

    def _load_workflow(self, model: str) -> dict:
        path = self._workflows.get(model)
        if path is None:
            raise EngineError(f"{self._LABEL}: modelo desconocido '{model}'.")
        if not path.exists():
            raise EngineError(f"{self._LABEL}: workflow no encontrado: {path}")
        try:
            with path.open("r", encoding="utf-8") as fh:
                workflow = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            raise EngineError(
                f"{self._LABEL}: no se pudo cargar el workflow '{path}': {exc}"
            ) from exc
        self._validate_workflow(workflow, path)
        return workflow

    def _validate_workflow(self, workflow: dict, path: Path) -> None:
        """Hook de validación: los motores comprueban los nodos que van a parchear."""

    # ---- Subida, envío, descarga y guardado ---------------------

    def _upload_image(self, image_path: Path) -> str:
        """Sube una imagen de referencia a ComfyUI (``POST /upload/image``)."""
        if not image_path.exists():
            raise EngineError(
                f"{self._LABEL}: la imagen de referencia no existe: {image_path}"
            )
        ext = image_path.suffix.lower()
        mime = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
        }.get(ext, "image/png")
        with image_path.open("rb") as fh:
            data = fh.read()
        try:
            resp = requests.post(
                f"{self._base_url}/upload/image",
                files={"image": (image_path.name, data, mime)},
                data={"overwrite": "true"},
                timeout=self._UPLOAD_TIMEOUT,
            )
        except requests.RequestException as exc:
            raise EngineError(
                f"{self._LABEL}: error subiendo la imagen: {exc}"
            ) from exc
        resp.raise_for_status()
        try:
            info = resp.json()
        except ValueError as exc:
            raise EngineError(
                f"{self._LABEL}: respuesta inválida al subir la imagen."
            ) from exc
        name = info.get("name")
        if not name:
            raise EngineError(
                f"{self._LABEL}: ComfyUI no devolvió el nombre de la imagen subida."
            )
        return name

    def _submit(self, workflow: dict, client_id: str | None = None) -> str:
        """Envía el workflow a ``POST /prompt`` y devuelve el ``prompt_id``."""
        payload = {"prompt": workflow, "client_id": client_id or str(uuid4())}
        try:
            resp = requests.post(
                f"{self._base_url}/prompt", json=payload, timeout=self._SUBMIT_TIMEOUT
            )
        except requests.RequestException as exc:
            raise EngineError(
                f"{self._LABEL}: error al enviar el prompt: {exc}"
            ) from exc

        logger.debug("ComfyUI POST /prompt status=%d", resp.status_code)
        if resp.status_code >= 400:
            try:
                data = resp.json()
                err = data.get("error") or {}
                msg = err.get("message") or err.get("type") or resp.text
            except ValueError:
                msg = resp.text
            raise EngineError(f"{self._LABEL}: ComfyUI rechazó el workflow: {msg}")

        try:
            data = resp.json()
        except ValueError as exc:
            raise EngineError(
                f"{self._LABEL}: respuesta inválida al enviar el prompt."
            ) from exc

        prompt_id = data.get("prompt_id")
        if not prompt_id:
            raise EngineError(
                f"{self._LABEL}: ComfyUI no devolvió 'prompt_id' al enviar el prompt."
            )
        return prompt_id

    def _download(self, media_meta: dict) -> bytes:
        """Descarga un resultado de ComfyUI (``GET /view``)."""
        params = {
            "filename": media_meta["filename"],
            "type": media_meta.get("type", "output"),
        }
        subfolder = media_meta.get("subfolder", "")
        if subfolder:
            params["subfolder"] = subfolder
        resp = requests.get(
            f"{self._base_url}/view", params=params, timeout=self._VIEW_TIMEOUT
        )
        resp.raise_for_status()
        return resp.content

    def _save(self, content: bytes, seed: int, ext: str = ".png") -> Path:
        """Guarda el resultado en ``output/`` con nombre único."""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"plataformaia_{datetime.now():%Y%m%d_%H%M%S}_{seed}{ext}"
        path = self._output_dir / filename
        with path.open("wb") as fh:
            fh.write(content)
        logger.info("%s: resultado guardado en '%s'.", self._LABEL, path)
        return path
