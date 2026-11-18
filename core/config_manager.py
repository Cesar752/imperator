"""Gestión de configuración basada en JSON con validación.

Este módulo lee y escribe tres tipos de configuración de la plataforma
—motores, APIs y ajustes generales— almacenados como JSON en el directorio
``config/`` (raíz del proyecto). Es lógica pura, sin dependencias de PyQt6.

Los archivos reales (``config/*.json``) se crean automáticamente con valores
por defecto si no existen y se ignoran por git; las plantillas versionadas
viven en ``config/*.example.json``.
"""

import json
import logging
import os
import tempfile
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Error de configuración: archivo corrupto o datos no válidos."""


# Directorio de configuración por defecto: raíz del proyecto / config.
CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

# Esquemas de validación. Cada campo declara:
#   type     -> tipo esperado (str, bool, dict, ...)
#   required -> si el campo debe estar presente
#   min_len  -> longitud mínima para str/list/dict (opcional)
#   allowed  -> valores permitidos (opcional)
ENGINE_SCHEMA: dict[str, dict[str, Any]] = {
    "id": {"type": str, "required": True, "min_len": 1},
    "nombre": {"type": str, "required": True, "min_len": 1},
    "ruta": {"type": str, "required": True, "min_len": 1},
    "enabled": {"type": bool, "required": True},
    "parametros": {"type": dict, "required": True},
}

# NOTA: 'key' es opcional y está DEPRECADO. Las claves de API se leen SIEMPRE
# de variables de entorno (.env); nunca de este JSON en claro (ver
# :meth:`ConfigManager.resolve_api_key`).
API_SCHEMA: dict[str, dict[str, Any]] = {
    "id": {"type": str, "required": True, "min_len": 1},
    "nombre": {"type": str, "required": True, "min_len": 1},
    "enabled": {"type": bool, "required": True},
    "key": {"type": str, "required": False},
    "base_url": {"type": str, "required": True, "min_len": 1},
    "tipo": {
        "type": str,
        "required": True,
        "allowed": ("llm", "image", "video", "audio"),
    },
}

SETTINGS_SCHEMA: dict[str, dict[str, Any]] = {
    "log_level": {
        "type": str,
        "required": True,
        "allowed": ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
    },
    "rutas": {"type": dict, "required": True},
    "last_models": {"type": dict, "required": False},
    "unload_on_tab_change": {"type": bool, "required": False},
    "theme": {"type": str, "required": False, "allowed": ("dark", "light")},
    "geometry": {"type": dict, "required": False},
    "font_size": {"type": int, "required": False},
    # Ajustes del planificador de GPU (core/gpu_scheduler.py). El detalle de los
    # campos lo valida SchedulerConfig.from_settings, que es tolerante a fallos.
    "gpu_scheduler": {"type": dict, "required": False},
    # Retención de lo que se genera en output/ (services/output_cleanup.py). El
    # detalle lo valida plan_cleanup, que también es tolerante a fallos.
    "output_limpieza": {"type": dict, "required": False},
}

# Valores por defecto (se escriben en disco cuando el archivo no existe).
#
# OJO con la URL de Ollama: se usa ``127.0.0.1`` y **no** ``localhost``. En
# Windows ``localhost`` resuelve a ``::1`` y a ``127.0.0.1``; Ollama sólo escucha
# en IPv4, así que la conexión a ``::1`` se rechaza (WinError 10061) y el
# fallback a IPv4 cuesta ~2 s **por petición** (medido con la auditoría: 2 040 ms
# con ``localhost`` frente a 15 ms con ``127.0.0.1``). Con ``localhost``, cada
# página de un documento y cada mensaje del chat pagaban ese peaje.
DEFAULT_ENGINES: list[dict[str, Any]] = [
    {
        "id": "ollama",
        "nombre": "Ollama",
        "ruta": "http://127.0.0.1:11434",
        "enabled": True,
        "parametros": {"model": "qwen3.6:35b"},
    },
    {
        "id": "ollama_vision",
        "nombre": "Ollama Vision",
        "ruta": "http://127.0.0.1:11434",
        "enabled": True,
        "parametros": {"model": "qwen2.5vl:7b"},
    },
    # APIs externas compatibles con OpenAI (chat). Deshabilitadas y sin clave por
    # defecto: la clave se lee de .env (OPENAI_API_KEY / DEEPSEEK_API_KEY).
    {
        "id": "openai",
        "nombre": "OpenAI",
        "ruta": "https://api.openai.com/v1",
        "enabled": False,
        "parametros": {
            "model": "gpt-4o-mini",
            "models": ["gpt-4o-mini", "gpt-4o"],
        },
    },
    {
        "id": "deepseek",
        "nombre": "DeepSeek",
        "ruta": "https://api.deepseek.com/v1",
        "enabled": False,
        "parametros": {
            "model": "deepseek-chat",
            "models": ["deepseek-chat", "deepseek-reasoner"],
        },
    },
    # Servidores locales compatibles con OpenAI (sin clave). Opcionales.
    {
        "id": "llama_cpp",
        "nombre": "llama.cpp (llama-server)",
        "ruta": "http://127.0.0.1:8080/v1",
        "enabled": False,
        "parametros": {"model": "local-model"},
    },
    {
        "id": "lm_studio",
        "nombre": "LM Studio",
        "ruta": "http://127.0.0.1:1234/v1",
        "enabled": False,
        "parametros": {"model": "local-model"},
    },
    # Imagen externa (API de imágenes de OpenAI). Clave: OPENAI_API_KEY (.env).
    {
        "id": "openai_image",
        "nombre": "OpenAI Imagen",
        "ruta": "https://api.openai.com/v1",
        "enabled": False,
        "parametros": {
            "model": "gpt-image-1",
            "models": ["gpt-image-1", "dall-e-3"],
            "size": "1024x1024",
        },
    },
    # Visión de pago compatible con OpenAI (GPT-4o): mira los fotogramas con más
    # detalle que un modelo local. Pensada para la Crónica. Clave: OPENAI_API_KEY.
    {
        "id": "openai_vision",
        "nombre": "OpenAI Vision",
        "ruta": "https://api.openai.com/v1",
        "enabled": False,
        "parametros": {
            "model": "gpt-4o-mini",
            "models": ["gpt-4o-mini", "gpt-4o"],
            "detail": "auto",
        },
    },
    {
        "id": "comfyui",
        "nombre": "ComfyUI",
        "ruta": "http://127.0.0.1:8188",
        "enabled": False,
        "parametros": {
            "workflows": {
                "flux_dev": "C:/CComfyUI/flux_dev_workflow_api.json",
                "flux_schnell": "C:/CComfyUI/flux_schnell_workflow_api.json",
                "z_image_turbo": "C:/CComfyUI/z_image_turbo_workflow_api.json",
            },
            "default_model": "flux_dev",
        },
    },
    {
        "id": "wanvideo",
        "nombre": "Wan Video",
        "ruta": "http://127.0.0.1:8188",
        "enabled": False,
        "parametros": {
            "workflows": {
                "t2v": "C:/CComfyUI/wan22_t2v_workflow_api.json",
                "i2v": "C:/CComfyUI/wan22_i2v_workflow_api.json",
            },
            "default_model": "t2v",
        },
    },
    {
        "id": "bfl",
        "nombre": "Black Forest Labs (API de pago)",
        "ruta": "https://api.bfl.ai",
        "enabled": False,
        "parametros": {
            "models": ["flux-kontext-pro", "flux-kontext-max", "flux-2-pro"],
        },
    },
    # Herramientas externas de línea de comandos (engines/cli_tool.py). Se
    # invocan por subprocess; 'ruta' (o 'parametros.executable') apunta al
    # ejecutable y 'parametros.args' es la plantilla. Deshabilitadas por
    # defecto porque no vienen instaladas; con el ejecutable ausente el motor
    # aparece como no disponible (is_available() -> False).
    {
        "id": "bkingfilm",
        "nombre": "BkingFilm (CLI)",
        "ruta": "bkingfilm",
        "enabled": False,
        "parametros": {
            "args": ["--input", "{input}", "--prompt", "{prompt}", "--output", "{output}"],
            "output_ext": ".mp4",
        },
    },
    {
        "id": "lapian_notes",
        "nombre": "Lapian Notes (CLI)",
        "ruta": "lapian-notes",
        "enabled": False,
        "parametros": {
            "args": ["--video", "{input}", "--package", "{output}"],
            "output_ext": ".zip",
        },
    },
    {
        "id": "piper_tts",
        "nombre": "Piper TTS",
        "ruta": "voices",
        "enabled": True,
        "parametros": {
            "voices_dir": "voices",
            "default_voice": "es_ES-davefx-medium",
        },
    },
    {
        "id": "kokoro_tts",
        "nombre": "Kokoro TTS",
        "ruta": "local",
        "enabled": True,
        "parametros": {
            "default_voice": "ef_dora",
        },
    },
    {
        "id": "whisper",
        "nombre": "Whisper",
        "ruta": "local",
        "enabled": True,
        "parametros": {
            "model": "small",
            "device": "cpu",
            "compute_type": "int8",
        },
    },
    {
        "id": "musicgen",
        "nombre": "MusicGen (música local)",
        "ruta": "local",
        "enabled": True,
        "parametros": {
            "model": "small",
            "duration": 8,
        },
    },
    # Clonado de voz (XTTS-v2). Deshabilitado por defecto: requiere coqui-tts +
    # torch (pesados) y una muestra de voz en voices/clones/. Sin la dependencia
    # el motor aparece como no disponible, sin romper el arranque.
    {
        "id": "voice_clone",
        "nombre": "Clonar voz (XTTS-v2)",
        "ruta": "voices/clones",
        "enabled": False,
        "parametros": {
            "model": "tts_models/multilingual/multi-dataset/xtts_v2",
            "language": "es",
            "device": "cuda",
            "backend": "subprocess",
            "python": "venv-tts/Scripts/python.exe",
        },
    },
    # Percepción estructurada. Deshabilitados por defecto porque arrastran
    # dependencias pesadas y opcionales (ultralytics, transformers/torch,
    # paddleocr): sin ellas el motor aparece como no disponible, no rompe.
    {
        "id": "yolo",
        "nombre": "YOLO (detección)",
        "ruta": "local",
        "enabled": False,
        "parametros": {"model": "yolov8n.pt", "confidence": 0.25},
    },
    {
        "id": "clip",
        "nombre": "CLIP (etiquetado)",
        "ruta": "local",
        "enabled": False,
        "parametros": {"model": "openai/clip-vit-base-patch32", "top_k": 4},
    },
    {
        "id": "paddle_ocr",
        "nombre": "OCR local (PaddleOCR)",
        "ruta": "local",
        "enabled": False,
        "parametros": {"lang": "latin", "use_gpu": False},
    },
]
DEFAULT_APIS: list[dict[str, Any]] = []
DEFAULT_SETTINGS: dict[str, Any] = {
    "log_level": "INFO",
    "rutas": {},
    "last_models": {},
    "unload_on_tab_change": True,
    "theme": "dark",
    "font_size": 11,
    "gpu_scheduler": {
        "enabled": True,
        "max_concurrent": 2,
        "vram_headroom_mb": 1024,
        "defer_batch_above_util": 70,
        "defer_batch_below_free_mb": 2048,
    },
    # Red de seguridad contra el crecimiento sin límite de output/ (ver
    # services/output_cleanup.py). Con estos valores no se borra nada en un
    # output/ normal: 500 ficheros, 90 días y 8 GB, con 24 h de gracia para lo
    # recién generado. Deben coincidir con los DEFAULT_* del servicio.
    "output_limpieza": {
        "enabled": True,
        "max_files": 500,
        "max_age_days": 90,
        "max_mb": 8192,
        "keep_recent_hours": 24,
    },
}


def _type_name(expected: type | tuple[type, ...]) -> str:
    if isinstance(expected, tuple):
        return " o ".join(t.__name__ for t in expected)
    return expected.__name__


def _validate_object(obj: Any, schema: dict[str, dict[str, Any]], source: str) -> None:
    if not isinstance(obj, dict):
        raise ConfigError(
            f"{source}: se esperaba un objeto JSON, se recibió {type(obj).__name__}."
        )

    for field, spec in schema.items():
        if field not in obj:
            if spec.get("required", False):
                raise ConfigError(
                    f"{source}: falta el campo requerido '{field}'."
                )
            continue

        value = obj[field]
        expected = spec["type"]
        if not isinstance(value, expected):
            raise ConfigError(
                f"{source}: el campo '{field}' debe ser de tipo "
                f"{_type_name(expected)}, se recibió {type(value).__name__}."
            )

        if "allowed" in spec and value not in spec["allowed"]:
            allowed = ", ".join(repr(v) for v in spec["allowed"])
            raise ConfigError(
                f"{source}: el campo '{field}' tiene el valor {value!r}, "
                f"no permitido (valores válidos: {allowed})."
            )

        min_len = spec.get("min_len")
        if min_len is not None and isinstance(value, (str, list, dict, tuple)):
            if len(value) < min_len:
                if min_len <= 1:
                    detalle = "no puede estar vacío"
                else:
                    detalle = f"debe tener al menos {min_len} elementos"
                raise ConfigError(f"{source}: el campo '{field}' {detalle}.")


def _validate_list(data: Any, schema: dict[str, dict[str, Any]], source: str) -> None:
    if not isinstance(data, list):
        raise ConfigError(
            f"{source}: se esperaba una lista JSON, se recibió {type(data).__name__}."
        )

    seen: set[str] = set()
    for index, item in enumerate(data):
        item_source = f"{source}[{index}]"
        _validate_object(item, schema, item_source)
        item_id = item.get("id")
        if isinstance(item_id, str):
            lowered = item_id.lower()
            if lowered in seen:
                raise ConfigError(
                    f"{source}: id duplicado '{item_id}' "
                    f"(la comparación no distingue mayúsculas)."
                )
            seen.add(lowered)


def _validate(
    data: Any,
    schema: dict[str, dict[str, Any]],
    *,
    is_list: bool,
    source: str,
) -> None:
    if is_list:
        _validate_list(data, schema, source)
    else:
        _validate_object(data, schema, source)


class ConfigManager:
    """Lee y escribe la configuración JSON de la plataforma."""

    def __init__(self, config_dir: str | Path | None = None) -> None:
        self._config_dir = Path(config_dir) if config_dir is not None else CONFIG_DIR
        self._config_dir.mkdir(parents=True, exist_ok=True)
        self._dotenv_loaded = False
        # Serializa las escrituras: hay varios widgets (y workers) guardando
        # preferencias en settings.json a la vez.
        self._lock = threading.RLock()

    # ---- Motores -------------------------------------------------

    def load_engines(self) -> list[dict[str, Any]]:
        return self._load_json(
            "engines.json", ENGINE_SCHEMA, DEFAULT_ENGINES, is_list=True
        )

    def save_engines(self, engines: list[dict[str, Any]]) -> None:
        self._save_json("engines.json", engines, ENGINE_SCHEMA, is_list=True)

    # ---- APIs ----------------------------------------------------

    def load_apis(self) -> list[dict[str, Any]]:
        return self._load_json("apis.json", API_SCHEMA, DEFAULT_APIS, is_list=True)

    def save_apis(self, apis: list[dict[str, Any]]) -> None:
        self._save_json("apis.json", apis, API_SCHEMA, is_list=True)

    # ---- Ajustes -------------------------------------------------

    def load_settings(self) -> dict[str, Any]:
        return self._load_json(
            "settings.json", SETTINGS_SCHEMA, DEFAULT_SETTINGS, is_list=False
        )

    def save_settings(self, settings: dict[str, Any]) -> None:
        self._save_json("settings.json", settings, SETTINGS_SCHEMA, is_list=False)

    def update_settings(self, changes: dict[str, Any]) -> dict[str, Any]:
        """Aplica ``changes`` sobre settings.json y devuelve el resultado.

        Read-modify-write **bajo lock**: antes cada llamante hacía
        ``load_settings()`` + modificar + ``save_settings()`` por su cuenta, así
        que dos cambios simultáneos (p. ej. la geometría al cerrar mientras un
        worker guarda una preferencia) podían perder uno de los dos.
        """
        with self._lock:
            settings = self.load_settings()
            settings.update(changes)
            self._save_json(
                "settings.json", settings, SETTINGS_SCHEMA, is_list=False
            )
            return settings

    # ---- Último modelo seleccionado por tarea --------------------

    def load_last_models(self) -> dict[str, str]:
        """Devuelve el mapa tarea -> último modelo seleccionado (settings.json)."""
        settings = self.load_settings()
        return dict(settings.get("last_models", {}))

    def save_last_model(self, task_id: str, model: str) -> None:
        """Guarda el último modelo seleccionado para una tarea (settings.json).

        Bajo el mismo lock que :meth:`update_settings`: el merge del mapa
        ``last_models`` no puede pisar lo que guarde otro widget a la vez.
        """
        with self._lock:
            settings = self.load_settings()
            settings.setdefault("last_models", {})[task_id] = model
            self._save_json(
                "settings.json", settings, SETTINGS_SCHEMA, is_list=False
            )

    # ---- Resolución de clave de API ------------------------------

    def resolve_api_key(self, api: dict[str, Any]) -> str:
        """Devuelve la clave de API, leída SIEMPRE del entorno (nunca del JSON).

        El campo ``key`` de ``apis.json`` está deprecado y se ignora: los secretos
        viven solo en ``.env`` (p. ej. ``BFL_API_KEY``). Así se evita guardar
        claves en claro en disco.
        """
        api_id = api.get("id")
        if not api_id:
            return ""
        self._ensure_dotenv()
        env_var = f"{str(api_id).upper()}_API_KEY"
        return os.getenv(env_var, "")

    # ---- Internos -------------------------------------------------

    def _file_path(self, filename: str) -> Path:
        return self._config_dir / filename

    def _ensure_dotenv(self) -> None:
        if self._dotenv_loaded:
            return
        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
        self._dotenv_loaded = True

    def _load_json(
        self,
        filename: str,
        schema: dict[str, dict[str, Any]],
        default: Any,
        *,
        is_list: bool,
    ) -> Any:
        path = self._file_path(filename)
        if not path.exists():
            with self._lock:
                # Otra instancia puede haberlo creado mientras tanto.
                if not path.exists():
                    self._atomic_dump(path, default)
                    logger.debug("Configuración por defecto creada en '%s'.", path)
            return deepcopy(default)

        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ConfigError(
                f"El archivo de configuración '{path}' no es un JSON válido: {exc}"
            ) from exc
        except OSError as exc:
            raise ConfigError(f"No se pudo leer el archivo '{path}': {exc}") from exc

        _validate(data, schema, is_list=is_list, source=str(path))
        logger.debug("Configuración cargada desde '%s'.", path)
        return data

    def _save_json(
        self,
        filename: str,
        data: Any,
        schema: dict[str, dict[str, Any]],
        *,
        is_list: bool,
    ) -> None:
        path = self._file_path(filename)
        _validate(data, schema, is_list=is_list, source=str(path))
        with self._lock:
            self._atomic_dump(path, data)
        logger.debug("Configuración guardada en '%s'.", path)

    def _atomic_dump(self, path: Path, data: Any) -> None:
        """Escribe ``data`` en ``path`` de forma atómica (vía archivo temporal).

        El temporal es **único por escritura** (``mkstemp``). Antes se usaba un
        nombre fijo (``<archivo>.tmp``): dos escrituras simultáneas compartían el
        mismo temporal y en Windows eso acaba en fichero en uso a medias (o en un
        JSON truncado si una de las dos hace ``os.replace`` mientras la otra
        escribe). El ``os.replace`` final sigue siendo atómico.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, tmp_name = tempfile.mkstemp(
            dir=str(path.parent), prefix=f"{path.name}.", suffix=".tmp"
        )
        tmp = Path(tmp_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2)
                fh.write("\n")
            os.replace(tmp, path)
        except (OSError, TypeError, ValueError) as exc:
            # Limpiar el temporal ante cualquier fallo (I/O o datos no
            # serializables) para no dejar basura en config/.
            tmp.unlink(missing_ok=True)
            raise ConfigError(f"No se pudo escribir el archivo '{path}': {exc}") from exc
