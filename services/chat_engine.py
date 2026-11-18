"""Motor de chat preferido, compartido por todas las tareas.

El usuario elige el motor de chat en la pestaña **Chat** (Ollama, OpenAI,
DeepSeek, llama.cpp, LM Studio…) y esa elección se guarda en ``last_models``
(clave ``chat_engine``). Documentos, el guion de Vídeo, el Cuaderno, Comprensión
y la barra de estado resuelven aquí **el mismo** motor, en vez de clavar Ollama.

Lógica pura, sin Qt.
"""

from services import preferences as prefs

#: Motor por defecto cuando no hay ninguna elección guardada.
DEFAULT_CHAT_ENGINE = "ollama"


def preferred_chat_engine_id(config_manager=None) -> str:
    """Id del motor de chat elegido (o ``ollama`` si no hay preferencia)."""
    elegido = prefs.PrefsStore(config_manager).get_str(prefs.CHAT_ENGINE)
    return elegido or DEFAULT_CHAT_ENGINE


def preferred_chat_engine(engine_manager):
    """Instancia del motor de chat preferido, o la de por defecto.

    Devuelve ``None`` si no hay ninguno construido/habilitado.
    """
    if engine_manager is None:
        return None
    config_manager = getattr(engine_manager, "config_manager", None)
    elegido = preferred_chat_engine_id(config_manager)
    return engine_manager.get_engine(elegido) or engine_manager.get_engine(
        DEFAULT_CHAT_ENGINE
    )
