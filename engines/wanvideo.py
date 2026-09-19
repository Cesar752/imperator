"""Motor local Wan Video (ComfyUI + Wan2.2-Lightning T2V)."""

import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Callable
from uuid import uuid4

import requests

try:
    import websocket  # websocket-client (progreso y finalización en tiempo real)
except ImportError:  # pragma: no cover - dependencia declarada en requirements.txt
    websocket = None

from engines.base_engine import Engine, EngineError

logger = logging.getLogger(__name__)


class WanVideoEngine(Engine):
    """Motor de generación de vídeo con Wan2.2-Lightning vía ComfyUI.

    Genera clips a partir del workflow ``wan22_t2v_workflow_api.json``. El flujo
    replica a :class:`engines.comfyui.ComfyUIEngine` (cargar, parchear, ``/prompt``
    y descargar vía ``/view``). A diferencia del motor de imagen, la finalización
    y el progreso se detectan por el WebSocket ``/ws`` (fuente única de verdad);
    ``/history`` solo se lee una vez tras el ``execution_success`` para obtener la
    salida. Si el WebSocket falla, la generación falla (no hay fallback).
    """

    # Timeouts en segundos.
    _SHORT_TIMEOUT = 2.0        # /system_stats, /object_info, /history (lectura)
    _SUBMIT_TIMEOUT = 30.0      # POST /prompt
    _WS_TIMEOUT = 5.0           # recv() del WebSocket
    _VIEW_TIMEOUT = 120.0       # GET /view (descarga del MP4)
    _GENERATE_TIMEOUT = 600.0   # espera máxima total por clip
    _HISTORY_RETRIES = 10       # reintentos al leer /history tras el éxito

    # IDs de los KSampler en el workflow Lightning (para mapear el progreso).
    _KSAMPLER_HIGH = "57"       # etapa high-noise (pasos 0→2)
    _KSAMPLER_LOW = "58"        # etapa low-noise (pasos 2→4)

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8188",
        workflow_path: str | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        if not workflow_path:
            raise EngineError(
                "WanVideo: falta 'workflow_path' en la config de engines.json"
            )
        self._base_url = base_url.rstrip("/")
        self._workflow_path = Path(workflow_path)
        # Carpeta de salida por defecto: <raíz del proyecto>/output.
        self._output_dir = (
            Path(output_dir)
            if output_dir is not None
            else Path(__file__).resolve().parent.parent / "output"
        )

    @property
    def id(self) -> str:
        return "wanvideo"

    @property
    def nombre(self) -> str:
        return "Wan Video"

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
        # El workflow Lightning carga los UNet con UnetLoaderGGUF; su input
        # "unet_name" expone los .gguf de models/unet.
        resp = requests.get(
            f"{self._base_url}/object_info/UnetLoaderGGUF", timeout=self._SHORT_TIMEOUT
        )
        resp.raise_for_status()
        data = resp.json()
        unet_name = data["UnetLoaderGGUF"]["input"]["required"]["unet_name"]
        return list(unet_name[0])

    def chat(self, messages: list[dict]) -> str:
        raise EngineError("WanVideo es un motor de generación de vídeo, no de chat.")

    def generate_video(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 832,
        height: int = 480,
        length: int = 33,
        fps: int = 16,
        seed: int | None = None,
        model: str | None = None,
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> Path:
        """Genera un clip y devuelve la ruta del MP4 guardado en ``output/``.

        ``progress_callback`` recibe ``(porcentaje, mensaje)`` y se invoca desde
        el hilo que ejecuta esta llamada (el worker de la UI).
        """
        if seed is None:
            seed = random.randint(0, 2**31 - 1)

        workflow = self._load_workflow()
        self._patch_workflow(
            workflow,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            length=length,
            fps=fps,
            seed=seed,
            model=model,
        )

        client_id = str(uuid4())
        prompt_id = self._submit(workflow, client_id)
        video_meta = self._await_result(client_id, prompt_id, progress_callback)

        content = self._download(video_meta)
        return self._save(content, video_meta, seed)

    # ---- Carga y parcheo del workflow -----------------------------

    def _load_workflow(self) -> dict:
        try:
            with self._workflow_path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            raise EngineError(
                f"WanVideo: no se pudo cargar el workflow '{self._workflow_path}': {exc}"
            ) from exc

    def _patch_workflow(
        self,
        workflow: dict,
        *,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        length: int,
        fps: int,
        seed: int,
        model: str | None = None,
    ) -> None:
        """Parchea los nodos del workflow con los parámetros de la petición.

        Los IDs corresponden al workflow exportado (``wan22_t2v_workflow_api.json``).
        """
        workflow["6"]["inputs"]["text"] = prompt
        workflow["7"]["inputs"]["text"] = negative_prompt
        workflow["59"]["inputs"]["width"] = width
        workflow["59"]["inputs"]["height"] = height
        workflow["59"]["inputs"]["length"] = length
        workflow["60"]["inputs"]["fps"] = fps
        # La misma semilla en ambas etapas (high y low) para un clip coherente.
        workflow["57"]["inputs"]["noise_seed"] = seed
        workflow["58"]["inputs"]["noise_seed"] = seed
        # ``model`` es el GGUF high-noise; el low-noise se deriva por nombre.
        if model is not None:
            workflow["37"]["inputs"]["unet_name"] = model
            workflow["56"]["inputs"]["unet_name"] = model.replace(
                "HighNoise", "LowNoise"
            )

    # ---- Envío ----------------------------------------------------

    def _submit(self, workflow: dict, client_id: str) -> str:
        payload = {"prompt": workflow, "client_id": client_id}
        try:
            resp = requests.post(
                f"{self._base_url}/prompt", json=payload, timeout=self._SUBMIT_TIMEOUT
            )
        except requests.RequestException as exc:
            raise EngineError(f"WanVideo: error al enviar el prompt: {exc}") from exc

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

    # ---- Espera vía WebSocket (sin polling de /history) ----------

    def _await_result(
        self,
        client_id: str,
        prompt_id: str,
        progress_callback: Callable[[int, str], None] | None,
    ) -> dict:
        """Espera la notificación de fin por WebSocket y lee la salida.

        ``/history`` solo se consulta una vez, tras el ``execution_success``
        (con reintentos por la posible carrera entre el evento y el historial).
        Si el WebSocket no conecta o se corta, se lanza :class:`EngineError`.
        """
        if websocket is None:
            raise EngineError("WanVideo: falta la dependencia 'websocket-client'.")
        ws_url = self._base_url.replace("http://", "ws://").replace("https://", "wss://")
        ws_url = f"{ws_url}/ws?clientId={client_id}"
        try:
            ws = websocket.create_connection(ws_url, timeout=self._WS_TIMEOUT)
        except Exception as exc:  # noqa: BLE001 - fallo de red
            raise EngineError(
                f"WanVideo: no se pudo conectar al WebSocket: {exc}"
            ) from exc

        start = time.monotonic()
        try:
            while True:
                elapsed = time.monotonic() - start
                if elapsed >= self._GENERATE_TIMEOUT:
                    raise EngineError(
                        f"WanVideo: timeout tras {self._GENERATE_TIMEOUT:.0f}s "
                        f"esperando el vídeo."
                    )
                try:
                    raw = ws.recv()
                except websocket.WebSocketTimeoutException:
                    continue
                except Exception as exc:  # noqa: BLE001 - fin de conexión
                    raise EngineError(
                        f"WanVideo: la conexión WebSocket se interrumpió: {exc}"
                    ) from exc
                if not raw:
                    continue
                try:
                    message = json.loads(raw)
                except ValueError:
                    continue
                state = self._handle_ws_message(message, progress_callback)
                if state == "success":
                    return self._read_output(prompt_id)
                if state == "error":
                    raise EngineError(
                        f"ComfyUI: error en la ejecución. "
                        f"{self._ws_error_detail(message)}".strip()
                    )
        finally:
            ws.close()

    def _handle_ws_message(
        self, message: dict, progress_callback: Callable[[int, str], None] | None
    ) -> str | None:
        """Procesa un mensaje del WebSocket; devuelve ``'success'``, ``'error'`` o ``None``."""
        msg_type = message.get("type")
        data = message.get("data", {})
        if msg_type == "progress" and progress_callback is not None:
            value = data.get("value", 0)
            max_value = data.get("max", 0)
            node = str(data.get("node", ""))
            if max_value > 0:
                ratio = value / max_value
                if node == self._KSAMPLER_HIGH:
                    progress_callback(int(ratio * 50), f"High-noise {value}/{max_value}")
                elif node == self._KSAMPLER_LOW:
                    progress_callback(50 + int(ratio * 50), f"Low-noise {value}/{max_value}")
        elif msg_type == "executing" and progress_callback is not None:
            node = str(data.get("node", ""))
            if node == self._KSAMPLER_HIGH:
                progress_callback(0, "Iniciando etapa high-noise")
            elif node == self._KSAMPLER_LOW:
                progress_callback(50, "Iniciando etapa low-noise")
        elif msg_type in ("execution_success", "execution_cached"):
            return "success"
        elif msg_type == "execution_error":
            return "error"
        return None

    @staticmethod
    def _ws_error_detail(message: dict) -> str:
        data = message.get("data", {})
        exc_type = data.get("exception_type", "")
        exc_message = data.get("exception_message", "")
        detail = f"{exc_type}: {exc_message}".strip()
        return detail or "error desconocido"

    def _read_output(self, prompt_id: str) -> dict:
        """Lee ``/history/{id}`` tras el ``execution_success`` (con reintentos)."""
        last_error = None
        for _ in range(self._HISTORY_RETRIES):
            try:
                resp = requests.get(
                    f"{self._base_url}/history/{prompt_id}", timeout=self._SHORT_TIMEOUT
                )
                resp.raise_for_status()
                entry = resp.json().get(prompt_id)
                if entry is not None and entry.get("status", {}).get("completed"):
                    return self._first_video_meta(entry)
            except requests.RequestException as exc:
                last_error = exc
            time.sleep(0.5)
        raise EngineError(
            f"WanVideo: no se pudo leer la salida del vídeo desde /history: {last_error}"
        )

    # ---- Descarga y guardado --------------------------------------

    @staticmethod
    def _first_video_meta(entry: dict) -> dict:
        outputs = entry.get("outputs", {})
        for node_output in outputs.values():
            # El nodo SaveVideo devuelve el clip bajo "images" (con "animated").
            for key in ("images", "videos", "gifs"):
                items = node_output.get(key, [])
                if items:
                    return items[0]
        raise EngineError("ComfyUI terminó sin devolver ningún vídeo.")

    def _download(self, video_meta: dict) -> bytes:
        params = {
            "filename": video_meta["filename"],
            "type": video_meta.get("type", "output"),
        }
        subfolder = video_meta.get("subfolder", "")
        if subfolder:
            params["subfolder"] = subfolder
        resp = requests.get(
            f"{self._base_url}/view", params=params, timeout=self._VIEW_TIMEOUT
        )
        resp.raise_for_status()
        return resp.content

    def _save(self, content: bytes, video_meta: dict, seed: int) -> Path:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        ext = Path(video_meta.get("filename", "video.mp4")).suffix or ".mp4"
        filename = f"plataformaia_{datetime.now():%Y%m%d_%H%M%S}_{seed}{ext}"
        path = self._output_dir / filename
        with path.open("wb") as fh:
            fh.write(content)
        logger.info("WanVideo: vídeo guardado en '%s'.", path)
        return path


