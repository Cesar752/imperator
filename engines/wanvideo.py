"""Motor local Wan Video (ComfyUI + Wan2.2-Lightning T2V/I2V)."""

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

    Soporta varios modos (``t2v``, ``i2v``...) seleccionados por la clave del
    diccionario ``workflows``. Cada modo es un workflow JSON que embebe sus
    propios modelos; el dict puede crecer en el futuro (``t2v_q5``, ``i2v_q4``...)
    sin tocar el motor. ``image_path`` activa el modo imagen-a-vídeo (``i2v``).

    El flujo replica a :class:`engines.comfyui.ComfyUIEngine` (cargar, parchear,
    ``/prompt`` y descargar vía ``/view``). La finalización y el progreso se
    detectan por el WebSocket ``/ws`` (fuente única de verdad); ``/history`` solo
    se lee una vez tras ``execution_success``. Si el WebSocket falla, la
    generación falla (no hay fallback).
    """

    # Timeouts en segundos.
    _SHORT_TIMEOUT = 2.0        # /system_stats, /object_info, /history (lectura)
    _SUBMIT_TIMEOUT = 30.0      # POST /prompt
    _WS_TIMEOUT = 5.0           # recv() del WebSocket
    _VIEW_TIMEOUT = 120.0       # GET /view (descarga del MP4)
    _GENERATE_TIMEOUT = 1800.0  # espera máxima total por clip (default)
    _HISTORY_RETRIES = 30       # reintentos al leer /history tras el éxito (30 s)
    _HISTORY_RETRY_DELAY = 1.0  # segundos entre reintentos de /history
    _UPLOAD_TIMEOUT = 60.0      # POST /upload/image

    # IDs de los KSampler en el workflow Lightning (para mapear el progreso).
    _KSAMPLER_HIGH = "57"       # etapa high-noise (pasos 0→2)
    _KSAMPLER_LOW = "58"        # etapa low-noise (pasos 2→4)

    # class_types del nodo de latent en cada modo.
    _LATENT_TYPES = ("EmptyHunyuanLatentVideo", "WanImageToVideo")

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8188",
        workflows: dict[str, str] | None = None,
        default_model: str = "t2v",
        output_dir: str | Path | None = None,
        timeout: float = 1800.0,
    ) -> None:
        if not workflows:
            raise EngineError("WanVideo: falta 'workflows' en la config de engines.json")
        self._base_url = base_url.rstrip("/")
        self._workflows = {name: Path(path) for name, path in workflows.items()}
        self._default_model = default_model or next(iter(self._workflows))
        self._generate_timeout = float(timeout)
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

    @property
    def base_url(self) -> str:
        return self._base_url

    def set_timeout(self, timeout: float) -> None:
        """Actualiza el tiempo máximo de espera por clip (segundos)."""
        self._generate_timeout = float(timeout)

    def cancel_remote(self) -> None:
        """Solicita a ComfyUI detener la ejecución en curso (POST /interrupt)."""
        try:
            requests.post(f"{self._base_url}/interrupt", timeout=2)
            logger.info("ComfyUI: interrupción solicitada")
        except requests.RequestException as exc:
            logger.warning("No se pudo interrumpir ComfyUI: %s", exc)

    def _free_models(self) -> None:
        """Libera VRAM/RAM de ComfyUI tras finalizar una generación (POST /free)."""
        try:
            requests.post(
                f"{self._base_url}/free",
                json={"unload_models": True, "free_memory": True},
                timeout=5,
            )
            logger.info("ComfyUI: modelos liberados")
        except requests.RequestException as exc:
            logger.warning("No se pudo liberar ComfyUI: %s", exc)

    def unload_model(self) -> bool:
        """WanVideo ya libera tras cada generación; no-op para consistencia."""
        return True

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
        # Cada modelo es una clave del diccionario "workflows" (t2v, i2v, ...).
        return list(self._workflows.keys())

    def chat(self, messages: list[dict], model: str | None = None) -> str:
        raise EngineError("WanVideo es un motor de generación de vídeo, no de chat.")

    def generate_video(
        self,
        prompt: str,
        negative_prompt: str = "",
        image_path: str | Path | None = None,
        model: str | None = None,
        width: int = 832,
        height: int = 480,
        length: int = 33,
        fps: int = 16,
        seed: int | None = None,
        progress_callback: Callable[[int, str], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> Path:
        """Genera un clip y devuelve la ruta del MP4 guardado en ``output/``.

        ``model`` es el modo (``t2v``/``i2v``); ``image_path`` activa el modo
        imagen-a-vídeo. ``progress_callback`` recibe ``(porcentaje, mensaje)``
        y se invoca desde el hilo que ejecuta esta llamada (el worker de la UI).
        """
        if seed is None:
            seed = random.randint(0, 2**31 - 1)

        model = model or self._default_model
        if image_path is not None and model == "t2v":
            logger.warning("WanVideo: image_path con modo 't2v'; cambiando a 'i2v'.")
            model = "i2v"
        if model == "i2v" and image_path is None:
            raise EngineError("WanVideo: el modo 'i2v' requiere image_path")

        workflow = self._load_workflow(model)

        image_name = None
        if image_path is not None:
            image_name = self._upload_image(Path(image_path))

        self._patch_workflow(
            workflow,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            length=length,
            fps=fps,
            seed=seed,
            image_name=image_name,
        )

        client_id = str(uuid4())
        prompt_id = self._submit(workflow, client_id)
        video_meta = self._await_result(
            client_id, prompt_id, progress_callback, cancel_check
        )

        try:
            content = self._download(video_meta)
            return self._save(content, video_meta, seed)
        finally:
            self._free_models()

    # ---- Subida de imagen -----------------------------------------

    def _upload_image(self, image_path: Path) -> str:
        """Sube la imagen a ComfyUI (``POST /upload/image``) y devuelve su nombre."""
        if not image_path.exists():
            raise EngineError(f"WanVideo: la imagen no existe: {image_path}")
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
            raise EngineError(f"WanVideo: error subiendo la imagen: {exc}") from exc
        resp.raise_for_status()
        try:
            info = resp.json()
        except ValueError as exc:
            raise EngineError("WanVideo: respuesta inválida al subir la imagen.") from exc
        name = info.get("name")
        if not name:
            raise EngineError("WanVideo: ComfyUI no devolvió el nombre de la imagen subida.")
        return name

    # ---- Carga y parcheo del workflow -----------------------------

    def _load_workflow(self, model: str) -> dict:
        path = self._workflows.get(model)
        if path is None:
            raise EngineError(f"WanVideo: modo desconocido '{model}'")
        if not path.exists():
            raise EngineError(f"WanVideo: workflow no encontrado: {path}")
        try:
            with path.open("r", encoding="utf-8") as fh:
                wf = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            raise EngineError(
                f"WanVideo: no se pudo cargar el workflow '{path}': {exc}"
            ) from exc
        self._validate_workflow(wf, path)
        return wf

    def _validate_workflow(self, wf: dict, path: Path) -> None:
        """Valida que el workflow tenga los nodos que el motor va a parchear."""
        required = {"6": "CLIPTextEncode", "7": "CLIPTextEncode"}
        for node_id, expected_ct in required.items():
            node = wf.get(node_id)
            if node is None:
                raise EngineError(f"WanVideo: workflow '{path.name}' no tiene el nodo {node_id}")
            if node.get("class_type") != expected_ct:
                raise EngineError(
                    f"WanVideo: nodo {node_id} es '{node.get('class_type')}', esperado '{expected_ct}'"
                )
        if not any(n.get("class_type") in self._LATENT_TYPES for n in wf.values()):
            raise EngineError(f"WanVideo: workflow '{path.name}' no tiene nodo de latent")

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
        image_name: str | None = None,
    ) -> None:
        """Parchea por ID (nodos estables) y por class_type (nodos que difieren)."""
        # Prompt: IDs 6 (positivo) / 7 (negativo), idénticos en t2v e i2v.
        workflow["6"]["inputs"]["text"] = prompt
        workflow["7"]["inputs"]["text"] = negative_prompt

        # Latent + LoadImage: por class_type (difieren entre modos).
        for node in workflow.values():
            inputs = node.get("inputs")
            if not isinstance(inputs, dict):
                continue
            ct = node.get("class_type")
            if ct in self._LATENT_TYPES:
                inputs["width"] = width
                inputs["height"] = height
                inputs["length"] = length
            elif ct == "LoadImage" and image_name is not None:
                inputs["image"] = image_name

        # Samplers y fps: IDs estables.
        workflow["57"]["inputs"]["noise_seed"] = seed
        workflow["58"]["inputs"]["noise_seed"] = seed
        workflow["60"]["inputs"]["fps"] = fps

    # ---- Envío ----------------------------------------------------

    def _submit(self, workflow: dict, client_id: str) -> str:
        payload = {"prompt": workflow, "client_id": client_id}
        try:
            resp = requests.post(
                f"{self._base_url}/prompt", json=payload, timeout=self._SUBMIT_TIMEOUT
            )
        except requests.RequestException as exc:
            raise EngineError(f"WanVideo: error al enviar el prompt: {exc}") from exc

        logger.info("ComfyUI POST /prompt status=%d", resp.status_code)
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

        logger.info("ComfyUI response: %s", json.dumps(data)[:500])
        prompt_id = data.get("prompt_id")
        logger.info("ComfyUI prompt_id: %s", prompt_id)
        if not prompt_id:
            raise EngineError(f"ComfyUI no devolvió 'prompt_id' al enviar el prompt: {data}")
        return prompt_id

    # ---- Espera vía WebSocket (sin polling de /history) ----------

    def _await_result(
        self,
        client_id: str,
        prompt_id: str,
        progress_callback: Callable[[int, str], None] | None,
        cancel_check: Callable[[], bool] | None = None,
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
                if cancel_check is not None and cancel_check():
                    raise EngineError("Cancelado por el usuario")
                elapsed = time.monotonic() - start
                if elapsed >= self._generate_timeout:
                    raise EngineError(
                        f"WanVideo: timeout tras {self._generate_timeout:.0f}s "
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
                    ws_prompt_id = (message.get("data") or {}).get("prompt_id")
                    if ws_prompt_id and ws_prompt_id != prompt_id:
                        logger.warning(
                            "WanVideo: prompt_id del WebSocket (%s) != prompt_id "
                            "del POST /prompt (%s); usando el del WebSocket.",
                            ws_prompt_id,
                            prompt_id,
                        )
                        prompt_id = ws_prompt_id
                    return self._read_output(prompt_id, cancel_check)
                if state == "error":
                    raise EngineError(
                        f"ComfyUI: error en la ejecución. "
                        f"{self._ws_error_detail(message)}".strip()
                    )
                if state == "interrupted":
                    raise EngineError("Cancelado por el usuario")
        finally:
            ws.close()

    def _handle_ws_message(
        self, message: dict, progress_callback: Callable[[int, str], None] | None
    ) -> str | None:
        """Procesa un mensaje del WebSocket.

        Devuelve ``'success'``, ``'error'``, ``'interrupted'`` o ``None``.
        """
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
        elif msg_type == "execution_success":
            # Solo execution_success marca el final de la generación.
            # execution_cached se emite al INICIO (ComfyUI >=0.3.x, tras
            # execution_start) indicando qué nodos salen de caché; tratarlo
            # como éxito hacía leer /history demasiado pronto (historial vacío).
            return "success"
        elif msg_type == "execution_error":
            return "error"
        elif msg_type == "execution_interrupted":
            return "interrupted"
        return None

    @staticmethod
    def _ws_error_detail(message: dict) -> str:
        data = message.get("data", {})
        exc_type = data.get("exception_type", "")
        exc_message = data.get("exception_message", "")
        detail = f"{exc_type}: {exc_message}".strip()
        return detail or "error desconocido"

    def _read_output(
        self, prompt_id: str, cancel_check: Callable[[], bool] | None = None
    ) -> dict:
        """Lee ``/history/{id}`` tras el ``execution_success`` (con reintentos).

        El WebSocket avisa de ``execution_success`` antes de que ``/history``
        esté completamente actualizado (carrera conocida en ComfyUI), así que
        reintentamos durante ``_HISTORY_RETRIES * _HISTORY_RETRY_DELAY``
        segundos. La salida se busca en cualquier nodo y en cualquier clave
        multimedia conocida (``videos``/``gifs``/``images``). Detecta además la
        cancelación del usuario y las ejecuciones interrumpidas/erróneas para
        no esperar el ciclo completo de reintentos (30 s).
        """
        logger.info("WanVideo: esperando salida en /history/%s", prompt_id)
        last_error = None
        for attempt in range(1, self._HISTORY_RETRIES + 1):
            if cancel_check is not None and cancel_check():
                raise EngineError("Cancelado por el usuario")
            try:
                resp = requests.get(
                    f"{self._base_url}/history/{prompt_id}", timeout=self._SHORT_TIMEOUT
                )
                resp.raise_for_status()
                payload = resp.json()
                entry = payload.get(prompt_id)
                logger.info(
                    "WanVideo: /history intento %d/%d -> keys=%s, completed=%s",
                    attempt,
                    self._HISTORY_RETRIES,
                    list(payload.keys()),
                    bool(entry and entry.get("status", {}).get("completed")),
                )
                if entry is not None:
                    status = entry.get("status", {})
                    if status.get("status_str") == "error":
                        raise EngineError(
                            "ComfyUI: ejecución interrumpida o con error."
                        )
                    if status.get("completed"):
                        media = self._find_media(entry.get("outputs", {}))
                        if media is not None:
                            logger.info(
                                "WanVideo: salida encontrada en key '%s': %s",
                                media["key"],
                                media["data"],
                            )
                            return media["data"]
                        logger.warning(
                            "WanVideo: entrada completada pero sin salida multimedia "
                            "(intento %d).",
                            attempt,
                        )
            except requests.RequestException as exc:
                last_error = exc
                logger.warning(
                    "WanVideo: error HTTP leyendo /history (intento %d): %s",
                    attempt,
                    exc,
                )
            time.sleep(self._HISTORY_RETRY_DELAY)
        detail = (
            f"último error: {last_error}" if last_error else "el historial no quedó disponible"
        )
        raise EngineError(
            f"WanVideo: no se pudo leer la salida del vídeo desde /history/{prompt_id}: {detail}"
        )

    # ---- Descarga y guardado --------------------------------------

    @staticmethod
    def _find_media(outputs: dict) -> dict | None:
        """Busca el primer clip en cualquier nodo y en cualquier clave multimedia.

        ComfyUI guarda el vídeo bajo ``videos``, ``gifs`` o ``images`` (con el
        flag ``animated``) según el nodo de salida; el nodo ``SaveVideo`` nativo
        lo publica bajo ``gifs``. Devuelve ``{"key": key, "data": item}`` o
        ``None`` si todavía no hay salida.
        """
        for node_output in outputs.values():
            for key in ("videos", "gifs", "images"):
                if key in node_output and node_output[key]:
                    return {"key": key, "data": node_output[key][0]}
        return None

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

