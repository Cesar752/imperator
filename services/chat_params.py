"""Presets y utilidades puras de la tarea de chat (sin Qt)."""

# Número máximo de mensajes que se envían al motor. Acota el contexto: sin
# límite, una conversación larga degrada latencia, memoria y coste del modelo de
# forma silenciosa (la UI sigue mostrando la conversación completa).
MAX_HISTORY_MESSAGES = 20


def trim_history(
    messages: list[dict], max_messages: int = MAX_HISTORY_MESSAGES
) -> list[dict]:
    """Devuelve los últimos ``max_messages`` mensajes, conservando el orden.

    No muta ``messages`` (devuelve una copia recortada). Un ``max_messages``
    menor o igual que cero se interpreta como "sin recorte", útil para tests y
    para depuración.
    """
    if max_messages <= 0:
        return list(messages)
    return list(messages[-max_messages:])
