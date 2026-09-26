"""Motor local ComfyUI (API HTTP en http://127.0.0.1:8188)."""

import logging
import random
import time
from pathlib import Path
from typing import Callable

import requests

from engines.base_engine import EngineError, ImageEngine
from engines.comfyui_base import ComfyUIBaseEngine

logger = logging.getLogger(__name__)


def _flatten_messages(messages: list) -> str:
    """Aplana los mensajes de error de ComfyUI (listas de traceback) a texto."""
    parts: list[str] = []
    for item in messages:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, (list, tuple)):
            parts.extend(str(part) for part in item)
        else:
            parts.append(str(item))
    return " ".join(parts)


class ComfyUIEngine(ComfyUIBaseEngine, ImageEngine):
    """Generación de imagen vía ComfyUI con polling de ``/history``.

    Comparte el tronco HTTP con :class:`engines.comfyui_base.ComfyUIBaseEngine`
    (cargar/parchear/subir/enviar/descargar/guardar). Aquí solo vive lo propio de
    la imagen: el parcheo de nodos y la espera por polling del historial.
    """

    _LABEL = "ComfyUI"
    _SUBMIT_TIMEOUT = 30.0      # POST /prompt
    _POLL_TIMEOUT = 5.0         # GET /history/{id}
    _GENERATE_TIMEOUT = 300.0   # espera máxima total por imagen

    @property
    def id(self) -> str:
        return "comfyui"

    @property
    def nombre(self) -> str:
        return "ComfyUI"

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
        """Genera una imagen y devuelve la ruta del PNG guardado en ``output/``.

        ``image_path`` (opcional) activa la imagen de referencia para modelos de
        edición como ``flux_kontext``: se sube a ComfyUI y se parchea en el nodo
        ``LoadImage`` del workflow.
        """
        if seed is None:
            seed = random.randint(0, 2**31 - 1)

        workflow = self._load_workflow(model or self._default_model)

        image_name = None
        if image_path is not None:
            image_name = self._upload_image(Path(image_path))

        self._patch_workflow(
            workflow,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            seed=seed,
            guidance=guidance,
            image_name=image_name,
        )

        prompt_id = self._submit(workflow)
        image_meta = self._wait(
            prompt_id, timeout=self._GENERATE_TIMEOUT, cancel_check=cancel_check
        )
        try:
            content = self._download(image_meta)
            return self._save(content, seed)
        finally:
            self.unload_model()

    # ---- Parcheo del workflow ------------------------------------

    def _patch_workflow(
        self,
        workflow: dict,
        *,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        seed: int,
        guidance: float,
        image_name: str | None = None,
    ) -> None:
        """Parchea los nodos por tipo (los workflows Flux tienen IDs distintos).

        - CLIPTextEncode con texto = prompt positivo; vacío = negativo.
        - FluxGuidance (solo Flux Dev) = guidance.
        - EmptySD3LatentImage = dimensiones; KSampler = seed/steps.
        - LoadImage (flux_kontext) = imagen de referencia subida.
        El ``unet_name`` ya está embebido en cada workflow (no se parchea).
        """
        for node in workflow.values():
            inputs = node.get("inputs")
            if not isinstance(inputs, dict):
                continue
            class_type = node.get("class_type")
            if class_type == "CLIPTextEncode":
                if inputs.get("text", "").strip():
                    inputs["text"] = prompt
                else:
                    inputs["text"] = negative_prompt
            elif class_type == "FluxGuidance":
                inputs["guidance"] = guidance
            elif class_type == "EmptySD3LatentImage":
                inputs["width"] = width
                inputs["height"] = height
            elif class_type == "KSampler":
                inputs["seed"] = seed
                inputs["steps"] = steps
            elif class_type == "LoadImage" and image_name is not None:
                inputs["image"] = image_name

    # ---- Espera por polling --------------------------------------

    def _wait(
        self,
        prompt_id: str,
        timeout: float,
        cancel_check: Callable[[], bool] | None = None,
    ) -> dict:
        start = time.monotonic()
        last_log = 0.0
        while True:
            if cancel_check is not None and cancel_check():
                raise EngineError("Cancelado por el usuario")
            elapsed = time.monotonic() - start
            if elapsed >= timeout:
                raise EngineError(
                    f"ComfyUI: timeout tras {timeout:.0f}s esperando la imagen."
                )
            if elapsed - last_log >= 5:
                logger.info("ComfyUI: generando... (%.0fs transcurridos)", elapsed)
                last_log = elapsed

            try:
                resp = requests.get(
                    f"{self._base_url}/history/{prompt_id}", timeout=self._POLL_TIMEOUT
                )
                resp.raise_for_status()
            except requests.RequestException as exc:
                raise EngineError(
                    f"ComfyUI: error consultando el estado: {exc}"
                ) from exc

            data = resp.json()
            entry = data.get(prompt_id)
            if entry is not None:
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    detail = _flatten_messages(status.get("messages", []))[:500]
                    raise EngineError(
                        f"ComfyUI: error en la ejecución. {detail}".strip()
                    )
                if status.get("completed"):
                    return self._first_image_meta(entry)

            time.sleep(1)

    @staticmethod
    def _first_image_meta(entry: dict) -> dict:
        outputs = entry.get("outputs", {})
        for node_output in outputs.values():
            images = node_output.get("images", [])
            if images:
                return images[0]
        raise EngineError("ComfyUI terminó sin devolver ninguna imagen.")
