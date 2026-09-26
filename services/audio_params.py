"""Presets y validaciones de la tarea de audio / TTS (sin Qt).

Centraliza lo que antes vivía disperso en ``ui/audio_widget.py``: la lista de
motores, el rango de velocidad y la resolución de la voz (guardada -> por
defecto -> primera disponible).
"""

from services import preferences as prefs

# Motores soportados: (etiqueta visible, id del motor).
MOTORS: tuple[tuple[str, str], ...] = (("Piper", "piper"), ("Kokoro", "kokoro"))
DEFAULT_MOTOR = "piper"

# Aviso de la UI sobre la cobertura de idiomas de cada motor.
MOTOR_TOOLTIP = "Kokoro no soporta euskera. Para leyendas en euskera, usa Piper."

# Rango y valor por defecto de la velocidad. Semántica común (Piper): >1 = más
# lento, <1 = más rápido. KokoroTTSEngine invierte internamente el valor.
SPEED_MIN = 0.5
SPEED_MAX = 2.0
SPEED_DEFAULT = 1.0
SPEED_STEP = 0.1

# Voz por defecto de cada motor (si no hay ninguna guardada).
DEFAULT_VOICES: dict[str, str] = {
    "piper": "es_ES-davefx-medium",
    "kokoro": "ef_dora",
}

# Clave de persistencia de la voz elegida en cada motor.
VOICE_KEYS: dict[str, str] = {
    "piper": prefs.AUDIO_VOICE_PIPER,
    "kokoro": prefs.AUDIO_VOICE_KOKORO,
}


def voice_key(motor: str | None) -> str | None:
    """Clave de preferencias donde se guarda la voz de ``motor``.

    Devuelve ``None`` si no hay motor (o es desconocido), para que el llamante
    pueda omitir la lectura/escritura sin comprobar el diccionario a mano.
    """
    if not motor:
        return None
    return VOICE_KEYS.get(motor)


def resolve_voice(
    motor: str | None, saved: str | None, available: list[str]
) -> str | None:
    """Voz a seleccionar: la guardada, la del motor o la primera disponible.

    Devuelve ``None`` si el motor no expone ninguna voz.
    """
    if saved in available:
        return saved
    default = DEFAULT_VOICES.get(motor or "")
    if default in available:
        return default
    return available[0] if available else None
