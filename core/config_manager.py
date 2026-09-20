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

API_SCHEMA: dict[str, dict[str, Any]] = {
    "id": {"type": str, "required": True, "min_len": 1},
    "nombre": {"type": str, "required": True, "min_len": 1},
    "enabled": {"type": bool, "required": True},
    "key": {"type": str, "required": True},
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
}

# Valores por defecto (se escriben en disco cuando el archivo no existe).
DEFAULT_ENGINES: list[dict[str, Any]] = [
    {
        "id": "ollama",
        "nombre": "Ollama",
        "ruta": "http://localhost:11434",
        "enabled": True,
        "parametros": {"model": "qwen3.5:9b"},
    },
    {
        "id": "ollama_vision",
        "nombre": "Ollama Vision",
        "ruta": "http://localhost:11434",
        "enabled": True,
        "parametros": {"model": "qwen2.5vl:7b"},
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
]
DEFAULT_APIS: list[dict[str, Any]] = []
DEFAULT_SETTINGS: dict[str, Any] = {"log_level": "INFO", "rutas": {}, "last_models": {}}


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
                raise ConfigError(
                    f"{source}: el campo '{field}' no puede estar vacío."
                )


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

    # ---- Último modelo seleccionado por tarea --------------------

    def load_last_models(self) -> dict[str, str]:
        """Devuelve el mapa tarea -> último modelo seleccionado (settings.json)."""
        settings = self.load_settings()
        return dict(settings.get("last_models", {}))

    def save_last_model(self, task_id: str, model: str) -> None:
        """Guarda el último modelo seleccionado para una tarea (settings.json)."""
        settings = self.load_settings()
        settings.setdefault("last_models", {})[task_id] = model
        self.save_settings(settings)

    # ---- Resolución de clave de API ------------------------------

    def resolve_api_key(self, api: dict[str, Any]) -> str:
        """Devuelve la clave efectiva: la del JSON si existe, o la del entorno."""
        key = api.get("key", "")
        if key:
            return key
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
        self._atomic_dump(path, data)
        logger.debug("Configuración guardada en '%s'.", path)

    def _atomic_dump(self, path: Path, data: Any) -> None:
        """Escribe ``data`` en ``path`` de forma atómica (vía archivo temporal)."""
        tmp = path.with_name(path.name + ".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2)
                fh.write("\n")
            os.replace(tmp, path)
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            raise ConfigError(f"No se pudo escribir el archivo '{path}': {exc}") from exc
