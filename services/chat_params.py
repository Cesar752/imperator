"""Presets y utilidades puras de la tarea de chat (sin Qt)."""

# Número máximo de mensajes que se envían al motor. Acota el contexto: sin
# límite, una conversación larga degrada latencia, memoria y coste del modelo de
# forma silenciosa (la UI sigue mostrando la conversación completa).
MAX_HISTORY_MESSAGES = 20

# Presupuesto aproximado de caracteres del historial (≈4 caracteres por token).
# Contar mensajes no basta: unas pocas respuestas largas llenan la ventana de
# contexto del motor, que trunca la generación y devuelve una respuesta vacía.
MAX_HISTORY_CHARS = 12000


def trim_history(
    messages: list[dict],
    max_messages: int = MAX_HISTORY_MESSAGES,
    max_chars: int = MAX_HISTORY_CHARS,
) -> list[dict]:
    """Últimos mensajes que caben en los límites, conservando el orden.

    Recorta por número de mensajes y por tamaño. No muta ``messages`` (devuelve
    una copia). Un ``max_messages`` menor o igual que cero desactiva el recorte
    por número; ``max_chars`` menor o igual que cero desactiva el de tamaño.
    Siempre se conserva el último mensaje (el que acaba de escribir el usuario)
    aunque por sí solo supere el presupuesto.
    """
    recortado = list(messages) if max_messages <= 0 else list(messages[-max_messages:])
    if max_chars <= 0:
        return recortado
    total = 0
    inicio = len(recortado)
    for indice in range(len(recortado) - 1, -1, -1):
        total += _message_chars(recortado[indice])
        if total > max_chars and indice < len(recortado) - 1:
            break
        inicio = indice
    return recortado[inicio:]


def _message_chars(message: dict) -> int:
    """Tamaño aproximado del mensaje en caracteres (0 si no hay contenido)."""
    contenido = message.get("content")
    return len(contenido) if isinstance(contenido, str) else 0


def with_system_prompt(messages: list[dict], system_prompt: str) -> list[dict]:
    """Devuelve los mensajes con un mensaje de ``system`` al frente.

    El chat libre no usa prompt de sistema (conversa sin rol impuesto). Cuando el
    motor debe comportarse como **guionista** (ver ``services.video_script``), se
    le antepone ese rol. Si ``messages`` ya empieza por un ``system``, se
    reemplaza: nunca se envían dos mensajes de sistema seguidos, que algunos
    motores interpretan mal. No muta ``messages``.
    """
    system_prompt = (system_prompt or "").strip()
    if not system_prompt:
        return list(messages)
    if messages and messages[0].get("role") == "system":
        return [{"role": "system", "content": system_prompt}, *messages[1:]]
    return [{"role": "system", "content": system_prompt}, *messages]
