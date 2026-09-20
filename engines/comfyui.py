"""Motor local ComfyUI (API HTTP en http://127.0.0.1:8188)."""

import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Callable
from uuid import uuid4

import requests

from engines.base_engine import Engine, EngineError

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


class ComfyUIEngine(Engine):
    """Implementación de :class:`Engine` para un servidor local de ComfyUI.

    Genera imágenes vía la API HTTP de ComfyUI a partir de un workflow JSON
    exportado (``workflow_path``). El flujo es: cargar el workflow, parchear los
    nodos con los parámetros (prompt, steps, dimensiones, seed, guidance),
    enviarlo a ``/prompt``, hacer polling de ``/history/{id}`` hasta completar
    y descargar el PNG resultante vía ``/view``.

    ``workflow_path`` es obligatorio: la configuración (``engines.json``) es la
    única fuente de verdad sobre la ubicación del workflow del usuario.
    """

    # Timeouts en segundos.
    _SHORT_TIMEOUT = 2.0        # /system_stats, /object_info
    _SUBMIT_TIMEOUT = 30.0      # POST /prompt
    _POLL_TIMEOUT = 5.0         # GET /history/{id}
    _VIEW_TIMEOUT = 60.0        # GET /view (descarga del PNG)
    _GENERATE_TIMEOUT = 300.0   # espera máxima total por imagen

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8188",
        workflows: dict[str, str] | None = None,
        default_model: str | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        if not workflows:
            raise EngineError("ComfyUI: falta 'workflows' en la config de engines.json")
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
    def id(self) -> str:
        return "comfyui"

    @property
    def nombre(self) -> str:
        return "ComfyUI"

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

    def is_available(self) -> bool:
        try:
            resp = requests.get(
                f"{self._base_url}/system_stats", timeout=self._SHORT_TIMEOUT
            )
            resp.raise_for_status()
        except requests.RequestException:
            return False
        return True

    def list_models(self) -> list[str]:
        # Cada modelo es una clave del diccionario "workflows" (un workflow
        # JSON por modelo); la config es la fuente de verdad.
        return list(self._workflows.keys())

    def chat(self, messages: list[dict]) -> str:
        raise EngineError("ComfyUI es un motor de generación de imagen, no de chat.")

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
        cancel_check: Callable[[], bool] | None = None,
    ) -> Path:
        """Genera una imagen y devuelve la ruta del PNG guardado en ``output/``."""
        if seed is None:
            seed = random.randint(0, 2**31 - 1)

        workflow = self._load_workflow(model or self._default_model)
        self._patch_workflow(
            workflow,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            seed=seed,
            guidance=guidance,
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

    # ---- Carga y parcheo del workflow -----------------------------

    def _load_workflow(self, model: str) -> dict:
        if model not in self._workflows:
            raise EngineError(f"ComfyUI: modelo desconocido '{model}'.")
        path = self._workflows[model]
        try:
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            raise EngineError(
                f"ComfyUI: no se pudo cargar el workflow '{path}': {exc}"
            ) from exc

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
    ) -> None:
        """Parchea los nodos por tipo (los workflows Flux tienen IDs distintos).

        - CLIPTextEncode con texto = prompt positivo; vacío = negativo.
        - FluxGuidance (solo Flux Dev) = guidance.
        - EmptySD3LatentImage = dimensiones; KSampler = seed/steps.
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

    # ---- Envío, polling y descarga --------------------------------

    def _submit(self, workflow: dict) -> str:
        payload = {"prompt": workflow, "client_id": str(uuid4())}
        try:
            resp = requests.post(
                f"{self._base_url}/prompt", json=payload, timeout=self._SUBMIT_TIMEOUT
            )
        except requests.RequestException as exc:
            raise EngineError(f"ComfyUI: error al enviar el prompt: {exc}") from exc

        if resp.status_code >= 400:
            try:
                data = resp.json()
                err = data.get("error") or {}
                msg = err.get("message") or err.get("type") or resp.text
            except ValueError:
                msg = resp.text
            raise EngineError(f"ComfyUI rechazó el workflow: {msg}")

        try:
            data = resp.json()
        except ValueError as exc:
            raise EngineError(
                "ComfyUI devolvió una respuesta inválida al enviar el prompt."
            ) from exc

        prompt_id = data.get("prompt_id")
        if not prompt_id:
            raise EngineError("ComfyUI no devolvió 'prompt_id' al enviar el prompt.")
        return prompt_id

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

    def _download(self, image_meta: dict) -> bytes:
        params = {
            "filename": image_meta["filename"],
            "type": image_meta.get("type", "output"),
        }
        subfolder = image_meta.get("subfolder", "")
        if subfolder:
            params["subfolder"] = subfolder
        resp = requests.get(
            f"{self._base_url}/view", params=params, timeout=self._VIEW_TIMEOUT
        )
        resp.raise_for_status()
        return resp.content

    def _save(self, content: bytes, seed: int) -> Path:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"plataformaia_{datetime.now():%Y%m%d_%H%M%S}_{seed}.png"
        path = self._output_dir / filename
        with path.open("wb") as fh:
            fh.write(content)
        logger.info("ComfyUI: imagen guardada en '%s'.", path)
        return path