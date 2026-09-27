"""Registro y ciclo de vida de los motores locales de la plataforma."""

import logging
from copy import deepcopy

from core.config_manager import DEFAULT_ENGINES, ConfigManager
from engines.base_engine import Engine
from engines.bfl import BFLApiEngine
from engines.comfyui import ComfyUIEngine
from engines.kokoro_tts import KokoroTTSEngine
from engines.musicgen import MusicGenEngine
from engines.ollama import OllamaEngine
from engines.ollama_vision import OllamaVisionEngine
from engines.piper_tts import PiperTTSEngine
from engines.wanvideo import WanVideoEngine
from engines.whisper_engine import WhisperEngine

logger = logging.getLogger(__name__)

# Mapa id de motor -> clase Engine. Los ids que no estén aquí se ignoran
# al cargar la configuración (ver :meth:`EngineManager.reload`).
ENGINE_FACTORY: dict[str, type[Engine]] = {
    "ollama": OllamaEngine,
    "ollama_vision": OllamaVisionEngine,
    "comfyui": ComfyUIEngine,
    "bfl": BFLApiEngine,
    "kokoro_tts": KokoroTTSEngine,
    "musicgen": MusicGenEngine,
    "wanvideo": WanVideoEngine,
    "piper_tts": PiperTTSEngine,
    "whisper": WhisperEngine,
}

# Defaults por tipo de motor (si faltan en la configuración).
_OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434"
_OLLAMA_DEFAULT_MODEL = "qwen3.5:9b"
_OLLAMA_VISION_DEFAULT_BASE_URL = "http://localhost:11434"
_OLLAMA_VISION_DEFAULT_MODEL = "qwen3-vl:8b"
_COMFYUI_DEFAULT_BASE_URL = "http://127.0.0.1:8188"
_WANVIDEO_DEFAULT_BASE_URL = "http://127.0.0.1:8188"
_BFL_DEFAULT_BASE_URL = "https://api.bfl.ai"
_PIPER_DEFAULT_VOICES_DIR = "voices"
_KOKORO_DEFAULT_VOICE = "ef_dora"
_MUSICGEN_DEFAULT_MODEL = "small"
_MUSICGEN_DEFAULT_DURATION = 8.0
_WHISPER_DEFAULT_MODEL = "small"
_WHISPER_DEFAULT_DEVICE = "cpu"
_WHISPER_DEFAULT_COMPUTE_TYPE = "int8"


class EngineManager:
    """Carga los motores desde ``config/engines.json`` y gestiona su ciclo de vida.

    Mantiene dos estructuras paralelas:
    - ``_configs``: configuraciones reconocidas (aunque el motor no llegue a
      construirse). Es la fuente de verdad para la UI.
    - ``_instances``: solo los motores que se construyeron correctamente.
    """

    def __init__(self, config_manager: ConfigManager | None = None) -> None:
        self._config = config_manager if config_manager is not None else ConfigManager()
        self._configs: dict[str, dict] = {}
        self._instances: dict[str, Engine] = {}
        self.reload()

    @property
    def config_manager(self) -> ConfigManager:
        """ConfigManager asociado (para persistencia de preferencias de UI)."""
        return self._config

    # ---- Carga ---------------------------------------------------

    def reload(self) -> None:
        """Recarga los motores desde la configuración, tolerante a fallos por entrada."""
        self._configs = {}
        self._instances = {}
        try:
            entries = self._config.load_engines()
        except Exception as exc:  # noqa: BLE001 - config corrupta no debe romper el arranque
            logger.error(
                "No se pudo cargar engines.json (%s); se usan los motores por "
                "defecto. Corrige o borra config/engines.json.",
                exc,
            )
            entries = deepcopy(DEFAULT_ENGINES)
        for entry in entries:
            engine_id = entry["id"]
            if ENGINE_FACTORY.get(engine_id) is None:
                logger.warning("Motor '%s' no soportado; se ignora.", engine_id)
                continue
            self._configs[engine_id] = entry
            try:
                self._instances[engine_id] = self._build(engine_id, entry)
            except Exception as exc:  # noqa: BLE001 - fallo de construcción aislado
                logger.error("No se pudo construir el motor '%s': %s", engine_id, exc)

    @staticmethod
    def _build(engine_id: str, entry: dict) -> Engine:
        """Construye la instancia de un motor a partir de su configuración."""
        params = entry.get("parametros", {})
        if engine_id == "ollama":
            return OllamaEngine(
                base_url=entry.get("ruta", _OLLAMA_DEFAULT_BASE_URL),
                default_model=params.get("model", _OLLAMA_DEFAULT_MODEL),
            )
        if engine_id == "ollama_vision":
            return OllamaVisionEngine(
                base_url=entry.get("ruta", _OLLAMA_VISION_DEFAULT_BASE_URL),
                default_model=params.get("model", _OLLAMA_VISION_DEFAULT_MODEL),
            )
        if engine_id == "comfyui":
            return ComfyUIEngine(
                base_url=entry.get("ruta", _COMFYUI_DEFAULT_BASE_URL),
                workflows=params.get("workflows", {}),
                default_model=params.get("default_model"),
            )
        if engine_id == "bfl":
            return BFLApiEngine(
                base_url=entry.get("ruta", _BFL_DEFAULT_BASE_URL),
                models=params.get("models"),
            )
        if engine_id == "wanvideo":
            workflows = params.get("workflows")
            if not workflows and params.get("workflow_path"):
                # Backward-compat: formato antiguo con un solo workflow_path (T2V).
                workflows = {"t2v": params["workflow_path"]}
            return WanVideoEngine(
                base_url=entry.get("ruta", _WANVIDEO_DEFAULT_BASE_URL),
                workflows=workflows,
                default_model=params.get("default_model", "t2v"),
            )
        if engine_id == "piper_tts":
            return PiperTTSEngine(
                voices_dir=params.get("voices_dir", entry.get("ruta", _PIPER_DEFAULT_VOICES_DIR)),
                default_voice=params.get("default_voice"),
            )
        if engine_id == "whisper":
            return WhisperEngine(
                model_size=params.get("model", _WHISPER_DEFAULT_MODEL),
                device=params.get("device", _WHISPER_DEFAULT_DEVICE),
                compute_type=params.get("compute_type", _WHISPER_DEFAULT_COMPUTE_TYPE),
            )
        if engine_id == "kokoro_tts":
            return KokoroTTSEngine(
                default_voice=params.get("default_voice", _KOKORO_DEFAULT_VOICE),
            )
        if engine_id == "musicgen":
            return MusicGenEngine(
                model_size=params.get("model", _MUSICGEN_DEFAULT_MODEL),
                output_dir=params.get("output_dir"),
                duration=params.get("duration", _MUSICGEN_DEFAULT_DURATION),
            )
        # Genérico para futuros motores: los parametros se pasan tal cual.
        cls = ENGINE_FACTORY[engine_id]
        return cls(**params)

    # ---- Consultas ----------------------------------------------

    def supported_types(self) -> list[str]:
        """Tipos de motor que la plataforma sabe construir."""
        return list(ENGINE_FACTORY)

    def list_configs(self) -> list[dict]:
        """Configuraciones de todos los motores reconocidos (incluye los rotos)."""
        return [dict(c) for c in self._configs.values()]

    def get_config(self, engine_id: str) -> dict | None:
        cfg = self._configs.get(engine_id)
        return dict(cfg) if cfg is not None else None

    def get_instance(self, engine_id: str) -> Engine | None:
        """Instancia construida (independiente de ``enabled``); ``None`` si está rota."""
        return self._instances.get(engine_id)

    def get_engine(self, engine_id: str) -> Engine | None:
        """Motor listo para ejecutar tareas: construido y habilitado."""
        instance = self._instances.get(engine_id)
        if instance is None:
            return None
        cfg = self._configs.get(engine_id)
        if cfg is None or not cfg.get("enabled", False):
            return None
        return instance

    # ---- Mutación (persisten vía ConfigManager) -----------------

    def add_engine(self, config: dict) -> None:
        engines = self._config.load_engines()
        engines.append(config)
        self._config.save_engines(engines)
        self.reload()

    def update_engine(self, engine_id: str, config: dict) -> None:
        engines = self._config.load_engines()
        for index, entry in enumerate(engines):
            if entry["id"] == engine_id:
                engines[index] = config
                break
        else:
            raise KeyError(f"Motor '{engine_id}' no encontrado.")
        self._config.save_engines(engines)
        self.reload()

    def remove_engine(self, engine_id: str) -> None:
        engines = [e for e in self._config.load_engines() if e["id"] != engine_id]
        self._config.save_engines(engines)
        self.reload()

    def set_enabled(self, engine_id: str, enabled: bool) -> None:
        engines = self._config.load_engines()
        for entry in engines:
            if entry["id"] == engine_id:
                entry["enabled"] = enabled
                break
        self._config.save_engines(engines)
        self.reload()
