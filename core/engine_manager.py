"""Registro y ciclo de vida de los motores locales de la plataforma."""

import logging
from copy import deepcopy
from typing import Callable

from core.config_manager import DEFAULT_ENGINES, ConfigError, ConfigManager
from engines.base_engine import Engine, EngineError
from engines.bfl import BFLApiEngine
from engines.cli_tool import CliVideoEngine
from engines.comfyui import ComfyUIEngine
from engines.kokoro_tts import KokoroTTSEngine
from engines.musicgen import MusicGenEngine
from engines.ollama import OllamaEngine
from engines.ollama_vision import OllamaVisionEngine
from engines.openai_compatible import OpenAICompatibleChatEngine
from engines.openai_image import OpenAIImageEngine
from engines.openai_vision import OpenAICompatibleVisionEngine
from engines.perception import ClipEngine, PaddleOcrEngine, YoloEngine
from engines.piper_tts import PiperTTSEngine
from engines.voice_clone import VoiceCloneEngine
from engines.wanvideo import WanVideoEngine
from engines.whisper_engine import WhisperEngine

logger = logging.getLogger(__name__)

# Mapa id de motor -> clase Engine. Los ids que no estén aquí se ignoran
# al cargar la configuración (ver :meth:`EngineManager.reload`).
ENGINE_FACTORY: dict[str, type[Engine]] = {
    "ollama": OllamaEngine,
    "ollama_vision": OllamaVisionEngine,
    # Chat compatible con OpenAI: nube (openai/deepseek, clave en .env) y
    # servidores locales (llama.cpp/llama-server y LM Studio, sin clave).
    "openai": OpenAICompatibleChatEngine,
    "deepseek": OpenAICompatibleChatEngine,
    "llama_cpp": OpenAICompatibleChatEngine,
    "lm_studio": OpenAICompatibleChatEngine,
    # Visión de pago / con «más ojos» (GPT-4o, Gemini-compat, xAI…): mira los
    # fotogramas de verdad. Es la alternativa a Ollama Vision para la Crónica,
    # donde la calidad de la descripción por fotograma decide la crónica entera.
    "openai_vision": OpenAICompatibleVisionEngine,
    # Imagen externa (API de imágenes de OpenAI, p. ej. gpt-image-1).
    "openai_image": OpenAIImageEngine,
    "comfyui": ComfyUIEngine,
    "bfl": BFLApiEngine,
    # Herramientas externas de línea de comandos: se invocan por subprocess
    # (ver engines/cli_tool.py y core/cli_runner.py). El ejecutable lo fija
    # 'parametros.executable' (o 'ruta') en engines.json.
    "bkingfilm": CliVideoEngine,
    "lapian_notes": CliVideoEngine,
    "kokoro_tts": KokoroTTSEngine,
    "musicgen": MusicGenEngine,
    "wanvideo": WanVideoEngine,
    "piper_tts": PiperTTSEngine,
    "whisper": WhisperEngine,
    # Clonado de voz (XTTS-v2). Opcional: sin coqui-tts/torch queda como no
    # disponible, igual que musicgen sin audiocraft.
    "voice_clone": VoiceCloneEngine,
    # Percepción estructurada (opcional; sin sus dependencias quedan como no
    # disponibles, igual que musicgen sin audiocraft).
    "yolo": YoloEngine,
    "clip": ClipEngine,
    "paddle_ocr": PaddleOcrEngine,
}

# Defaults por tipo de motor (si faltan en la configuración).
#
# Se usa ``127.0.0.1`` en vez de ``localhost`` a propósito: con ``localhost`` la
# conexión intenta primero ``::1`` (donde Ollama no escucha), es rechazada y el
# fallback a IPv4 cuesta ~2 s por petición (medido: 2 040 ms frente a 15 ms).
_OLLAMA_DEFAULT_BASE_URL = "http://127.0.0.1:11434"
_OLLAMA_DEFAULT_MODEL = "qwen3.6:35b"
_OLLAMA_VISION_DEFAULT_BASE_URL = "http://127.0.0.1:11434"
_OLLAMA_VISION_DEFAULT_MODEL = "qwen2.5vl:7b"
_COMFYUI_DEFAULT_BASE_URL = "http://127.0.0.1:8188"
_WANVIDEO_DEFAULT_BASE_URL = "http://127.0.0.1:8188"
_BFL_DEFAULT_BASE_URL = "https://api.bfl.ai"
_OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_OPENAI_DEFAULT_MODEL = "gpt-4o-mini"
_OPENAI_DEFAULT_MODELS = ["gpt-4o-mini", "gpt-4o"]
_DEEPSEEK_DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
_DEEPSEEK_DEFAULT_MODEL = "deepseek-chat"
_DEEPSEEK_DEFAULT_MODELS = ["deepseek-chat", "deepseek-reasoner"]
_OPENAI_IMAGE_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_OPENAI_IMAGE_DEFAULT_MODEL = "gpt-image-1"
_OPENAI_IMAGE_DEFAULT_MODELS = ["gpt-image-1", "dall-e-3"]
_OPENAI_IMAGE_DEFAULT_SIZE = "1024x1024"
# Visión de pago compatible con OpenAI (GPT-4o). Misma clave que el chat de
# OpenAI (OPENAI_API_KEY en .env). Se usa, sobre todo, en la Crónica.
_OPENAI_VISION_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_OPENAI_VISION_DEFAULT_MODEL = "gpt-4o-mini"
_OPENAI_VISION_DEFAULT_MODELS = ["gpt-4o-mini", "gpt-4o"]
# Servidores locales compatibles (llama.cpp `llama-server` y LM Studio): no piden
# clave. Son opcionales y vienen deshabilitados.
_LLAMACPP_DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"
_LMSTUDIO_DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"

#: (base_url, modelo por defecto, modelos ofrecidos, ¿clave obligatoria?).
_OPENAI_COMPATIBLE_DEFAULTS: dict[str, tuple] = {
    "openai": (_OPENAI_DEFAULT_BASE_URL, _OPENAI_DEFAULT_MODEL, _OPENAI_DEFAULT_MODELS, True),
    "deepseek": (_DEEPSEEK_DEFAULT_BASE_URL, _DEEPSEEK_DEFAULT_MODEL, _DEEPSEEK_DEFAULT_MODELS, True),
    "llama_cpp": (_LLAMACPP_DEFAULT_BASE_URL, "local-model", [], False),
    "lm_studio": (_LMSTUDIO_DEFAULT_BASE_URL, "local-model", [], False),
}
_PIPER_DEFAULT_VOICES_DIR = "voices"
_KOKORO_DEFAULT_VOICE = "ef_dora"
_MUSICGEN_DEFAULT_MODEL = "small"
_MUSICGEN_DEFAULT_DURATION = 8.0
_YOLO_DEFAULT_MODEL = "yolov8n.pt"
_YOLO_DEFAULT_CONFIDENCE = 0.25
_CLIP_DEFAULT_MODEL = "openai/clip-vit-base-patch32"
_CLIP_DEFAULT_TOP_K = 4
_OCR_DEFAULT_LANG = "latin"
_WHISPER_DEFAULT_MODEL = "small"
_WHISPER_DEFAULT_DEVICE = "cpu"
_WHISPER_DEFAULT_COMPUTE_TYPE = "int8"
_VOICE_CLONE_DEFAULT_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
_VOICE_CLONE_DEFAULT_LANGUAGE = "es"
_VOICE_CLONE_DEFAULT_DEVICE = "cuda"
_VOICE_CLONE_DEFAULT_VOICES_DIR = "voices/clones"
# Coqui no soporta Python 3.12 (el de la app): por defecto la síntesis se hace en
# el venv de Python 3.11 vía subprocess (tools/xtts_worker.py).
_VOICE_CLONE_DEFAULT_BACKEND = "subprocess"
_VOICE_CLONE_DEFAULT_PYTHON = "venv-tts/Scripts/python.exe"

# Herramientas externas de línea de comandos (engines/cli_tool.py). Los valores
# por defecto son solo un punto de partida: se ajustan en engines.json. Se
# marcan deshabilitadas por defecto en DEFAULT_ENGINES (no están instaladas).
_CLI_DEFAULTS: dict[str, dict] = {
    "bkingfilm": {
        "nombre": "BkingFilm (CLI)",
        "executable": "bkingfilm",
        "args": ["--input", "{input}", "--prompt", "{prompt}", "--output", "{output}"],
        "output_ext": ".mp4",
    },
    "lapian_notes": {
        "nombre": "Lapian Notes (CLI)",
        "executable": "lapian-notes",
        "args": ["--video", "{input}", "--package", "{output}"],
        "output_ext": ".zip",
    },
}



def _params(entry: dict) -> dict:
    """Parámetros del motor en ``engines.json`` (``{}`` si no hay)."""
    return entry.get("parametros", {})


def _build_ollama(engine_id: str, entry: dict) -> Engine:
    params = _params(entry)
    # ``num_ctx`` (y ``num_predict``) se pueden fijar por motor en engines.json;
    # si no están, se usan los valores del propio motor, que son la única fuente
    # de verdad. ``timeout`` permite dar más margen a los modelos que razonan.
    return OllamaEngine(
        base_url=entry.get("ruta", _OLLAMA_DEFAULT_BASE_URL),
        default_model=params.get("model", _OLLAMA_DEFAULT_MODEL),
        num_ctx=params.get("num_ctx", OllamaEngine._NUM_CTX),
        num_predict=params.get("num_predict", OllamaEngine._NUM_PREDICT),
        timeout=params.get("timeout"),
        # ``think=False`` apaga el razonamiento de los modelos qwen3.x, que
        # gastaban la ventana «pensando» (Cuaderno/Crónica salían vacíos).
        think=params.get("think"),
    )


def _build_ollama_vision(engine_id: str, entry: dict) -> Engine:
    params = _params(entry)
    return OllamaVisionEngine(
        base_url=entry.get("ruta", _OLLAMA_VISION_DEFAULT_BASE_URL),
        default_model=params.get("model", _OLLAMA_VISION_DEFAULT_MODEL),
        num_ctx=params.get("num_ctx", OllamaVisionEngine._NUM_CTX),
        timeout=params.get("timeout"),
        # Rendimiento: apagar el razonamiento (qwen3-vl gasta ~90 % de los tokens
        # en «thinking») y acotar la salida. Ver README.
        think=params.get("think"),
        num_predict=params.get("num_predict"),
    )


def _build_openai_compatible(engine_id: str, entry: dict) -> Engine:
    params = _params(entry)
    base, model, models, need_key = _OPENAI_COMPATIBLE_DEFAULTS[engine_id]
    return OpenAICompatibleChatEngine(
        api_id=engine_id,
        base_url=entry.get("ruta", base),
        default_model=params.get("model", model),
        nombre=entry.get("nombre", engine_id.replace("_", " ").title()),
        models=params.get("models", models),
        api_key_env=params.get("api_key_env"),
        api_key_required=params.get("api_key_required", need_key),
        timeout=params.get("timeout"),
    )


def _build_openai_vision(engine_id: str, entry: dict) -> Engine:
    params = _params(entry)
    # Comparte la clave del chat de OpenAI (OPENAI_API_KEY en .env) y acepta
    # ``detail`` y ``max_tokens`` para acotar el coste por fotograma.
    return OpenAICompatibleVisionEngine(
        api_id=engine_id,
        base_url=entry.get("ruta", _OPENAI_VISION_DEFAULT_BASE_URL),
        default_model=params.get("model", _OPENAI_VISION_DEFAULT_MODEL),
        nombre=entry.get("nombre", "OpenAI Vision"),
        models=params.get("models", _OPENAI_VISION_DEFAULT_MODELS),
        api_key_env=params.get("api_key_env", "OPENAI_API_KEY"),
        api_key_required=params.get("api_key_required", True),
        timeout=params.get("timeout"),
        detail=params.get("detail", "auto"),
        max_tokens=params.get("max_tokens"),
    )


def _build_comfyui(engine_id: str, entry: dict) -> Engine:
    params = _params(entry)
    return ComfyUIEngine(
        base_url=entry.get("ruta", _COMFYUI_DEFAULT_BASE_URL),
        workflows=params.get("workflows", {}),
        default_model=params.get("default_model"),
    )


def _build_bfl(engine_id: str, entry: dict) -> Engine:
    params = _params(entry)
    return BFLApiEngine(
        base_url=entry.get("ruta", _BFL_DEFAULT_BASE_URL),
        models=params.get("models"),
    )


def _build_openai_image(engine_id: str, entry: dict) -> Engine:
    params = _params(entry)
    return OpenAIImageEngine(
        base_url=entry.get("ruta", _OPENAI_IMAGE_DEFAULT_BASE_URL),
        default_model=params.get("model", _OPENAI_IMAGE_DEFAULT_MODEL),
        nombre=entry.get("nombre", "OpenAI Imagen"),
        models=params.get("models", _OPENAI_IMAGE_DEFAULT_MODELS),
        api_key_env=params.get("api_key_env", "OPENAI_API_KEY"),
        output_dir=params.get("output_dir"),
        size=params.get("size", _OPENAI_IMAGE_DEFAULT_SIZE),
        timeout=params.get("timeout"),
    )


def _build_cli(engine_id: str, entry: dict) -> Engine:
    params = _params(entry)
    # CLI externa (subprocess). El ejecutable se toma de
    # 'parametros.executable' o, si no, de 'ruta'; la plantilla de argumentos de
    # 'parametros.args'. Si no está instalada, ``is_available()`` da False.
    defaults = _CLI_DEFAULTS[engine_id]
    return CliVideoEngine(
        engine_id=engine_id,
        nombre=entry.get("nombre", defaults["nombre"]),
        executable=params.get("executable") or entry.get("ruta") or defaults["executable"],
        args=params.get("args", defaults["args"]),
        output_dir=params.get("output_dir"),
        output_ext=params.get("output_ext", defaults["output_ext"]),
        output_glob=params.get("output_glob"),
        timeout=params.get("timeout"),
        models=params.get("models"),
        cwd=params.get("cwd"),
    )


def _build_wanvideo(engine_id: str, entry: dict) -> Engine:
    params = _params(entry)
    workflows = params.get("workflows")
    if not workflows and params.get("workflow_path"):
        # Backward-compat: formato antiguo con un solo workflow_path (T2V).
        workflows = {"t2v": params["workflow_path"]}
    return WanVideoEngine(
        base_url=entry.get("ruta", _WANVIDEO_DEFAULT_BASE_URL),
        workflows=workflows,
        default_model=params.get("default_model", "t2v"),
    )


def _build_piper_tts(engine_id: str, entry: dict) -> Engine:
    params = _params(entry)
    return PiperTTSEngine(
        voices_dir=params.get("voices_dir", entry.get("ruta", _PIPER_DEFAULT_VOICES_DIR)),
        default_voice=params.get("default_voice"),
    )


def _build_whisper(engine_id: str, entry: dict) -> Engine:
    params = _params(entry)
    return WhisperEngine(
        model_size=params.get("model", _WHISPER_DEFAULT_MODEL),
        device=params.get("device", _WHISPER_DEFAULT_DEVICE),
        compute_type=params.get("compute_type", _WHISPER_DEFAULT_COMPUTE_TYPE),
    )


def _build_voice_clone(engine_id: str, entry: dict) -> Engine:
    params = _params(entry)
    return VoiceCloneEngine(
        model_name=params.get("model", _VOICE_CLONE_DEFAULT_MODEL),
        language=params.get("language", _VOICE_CLONE_DEFAULT_LANGUAGE),
        device=params.get("device", _VOICE_CLONE_DEFAULT_DEVICE),
        voices_dir=params.get(
            "voices_dir", entry.get("ruta", _VOICE_CLONE_DEFAULT_VOICES_DIR)
        ),
        output_dir=params.get("output_dir"),
        backend=params.get("backend", _VOICE_CLONE_DEFAULT_BACKEND),
        python=params.get("python", _VOICE_CLONE_DEFAULT_PYTHON),
        worker=params.get("worker"),
        timeout=params.get("timeout"),
    )


def _build_kokoro_tts(engine_id: str, entry: dict) -> Engine:
    params = _params(entry)
    return KokoroTTSEngine(
        default_voice=params.get("default_voice", _KOKORO_DEFAULT_VOICE),
    )


def _build_musicgen(engine_id: str, entry: dict) -> Engine:
    params = _params(entry)
    return MusicGenEngine(
        model_size=params.get("model", _MUSICGEN_DEFAULT_MODEL),
        output_dir=params.get("output_dir"),
        duration=params.get("duration", _MUSICGEN_DEFAULT_DURATION),
    )


def _build_yolo(engine_id: str, entry: dict) -> Engine:
    params = _params(entry)
    return YoloEngine(
        model_path=params.get("model", _YOLO_DEFAULT_MODEL),
        confidence=params.get("confidence", _YOLO_DEFAULT_CONFIDENCE),
    )


def _build_clip(engine_id: str, entry: dict) -> Engine:
    params = _params(entry)
    return ClipEngine(
        model_name=params.get("model", _CLIP_DEFAULT_MODEL),
        top_k=params.get("top_k", _CLIP_DEFAULT_TOP_K),
    )


def _build_paddle_ocr(engine_id: str, entry: dict) -> Engine:
    params = _params(entry)
    return PaddleOcrEngine(
        lang=params.get("lang", _OCR_DEFAULT_LANG),
        use_gpu=bool(params.get("use_gpu", False)),
    )


def _build_generico(engine_id: str, entry: dict) -> Engine:
    """Genérico para futuros motores: los parametros se pasan tal cual."""
    cls = ENGINE_FACTORY[engine_id]
    try:
        return cls(**_params(entry))
    except TypeError as exc:
        # Antes salía un ``TypeError`` crudo que la UI mostraba como «error
        # inesperado»; el problema real es de la configuración del motor.
        raise ConfigError(
            f"El motor '{engine_id}' no acepta esos 'parametros' en "
            f"engines.json ({exc})."
        ) from exc


#: Mapa id de motor -> función que lo construye. Añadir un motor nuevo es añadir
#: una función aquí, no tocar la cadena de decisión del gestor.
#:
#: Los builders reciben ``(engine_id, entry)`` y no ``self`` porque ``_build`` es
#: un ``@staticmethod`` que los tests llaman como ``EngineManager._build(id, cfg)``;
#: convertirlos en métodos obligaría a tocar ``reload`` y diez llamadas de test.
_BUILDERS: dict[str, Callable[[str, dict], Engine]] = {
    "ollama": _build_ollama,
    "ollama_vision": _build_ollama_vision,
    "openai": _build_openai_compatible,
    "deepseek": _build_openai_compatible,
    "llama_cpp": _build_openai_compatible,
    "lm_studio": _build_openai_compatible,
    "openai_vision": _build_openai_vision,
    "openai_image": _build_openai_image,
    "comfyui": _build_comfyui,
    "bfl": _build_bfl,
    "bkingfilm": _build_cli,
    "lapian_notes": _build_cli,
    "wanvideo": _build_wanvideo,
    "piper_tts": _build_piper_tts,
    "whisper": _build_whisper,
    "voice_clone": _build_voice_clone,
    "kokoro_tts": _build_kokoro_tts,
    "musicgen": _build_musicgen,
    "yolo": _build_yolo,
    "clip": _build_clip,
    "paddle_ocr": _build_paddle_ocr,
}


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
    def _avisar_si_usa_localhost(engine_id: str, ruta: str | None) -> None:
        """Avisa si la URL del motor usa ``localhost`` (cuesta ~2 s por petición).

        En Windows ``localhost`` resuelve a ``::1`` y a ``127.0.0.1``; Ollama y
        ComfyUI sólo escuchan en IPv4, así que se paga el fallback en cada
        llamada (medido en la auditoría: 2 040 ms frente a 15 ms). No se
        reescribe la configuración del usuario: sólo se avisa.
        """
        if isinstance(ruta, str) and "localhost" in ruta:
            logger.warning(
                "El motor '%s' usa 'localhost' (%s): cada petición pagará ~2 s de "
                "fallback a IPv4. Cambia la 'ruta' a '127.0.0.1'.",
                engine_id,
                ruta,
            )

    @staticmethod
    def _build(engine_id: str, entry: dict) -> Engine:
        """Construye la instancia de un motor a partir de su configuración."""
        EngineManager._avisar_si_usa_localhost(engine_id, entry.get("ruta"))
        builder = _BUILDERS.get(engine_id)
        if builder is not None:
            return builder(engine_id, entry)
        if engine_id not in ENGINE_FACTORY:
            raise EngineError(f"Motor desconocido: {engine_id}")
        return _build_generico(engine_id, entry)

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
        """Añade un motor a la configuración (no admite ids repetidos)."""
        engine_id = config.get("id", "")
        engines = self._config.load_engines()
        if any(entry.get("id") == engine_id for entry in engines):
            # El esquema ya rechaza ids duplicados al guardar, pero el mensaje
            # genérico («id duplicado») no decía qué motor ni por qué.
            raise ConfigError(f"Ya existe un motor con el id '{engine_id}'.")
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
