"""Presets y validaciones de la tarea de audio / TTS (sin Qt).

Centraliza lo que antes vivía disperso en ``ui/audio_widget.py``: la lista de
motores, el rango de velocidad y la resolución de la voz (guardada -> por
defecto -> primera disponible).
"""

from services import preferences as prefs

# Motores soportados: (etiqueta visible, id del motor).
MOTORS: tuple[tuple[str, str], ...] = (
    ("Piper", "piper"),
    ("Kokoro", "kokoro"),
    ("Clonar voz (XTTS)", "voice_clone"),
)
DEFAULT_MOTOR = "piper"

# Aviso de la UI sobre la cobertura de idiomas de cada motor.
MOTOR_TOOLTIP = (
    "Kokoro no soporta euskera. Para leyendas en euskera, usa Piper. "
    "«Clonar voz» necesita una muestra de audio en voices/clones/."
)

# Rango y valor por defecto de la velocidad. Semántica común (Piper): >1 = más
# lento, <1 = más rápido. KokoroTTSEngine invierte internamente el valor.
SPEED_MIN = 0.5
SPEED_MAX = 2.0
SPEED_DEFAULT = 1.0
SPEED_STEP = 0.1

# Voz por defecto de cada motor (si no hay ninguna guardada). En «voice_clone» el
# valor es vacío a propósito: su "voz" es una muestra de referencia del usuario
# (no hay una por defecto), pero el guardián exige entrada para todos los motores.
DEFAULT_VOICES: dict[str, str] = {
    "piper": "es_ES-davefx-medium",
    "kokoro": "ef_dora",
    "voice_clone": "",
}

# Clave de persistencia de la voz elegida en cada motor. En «voice_clone» se
# guarda el nombre de la muestra de referencia elegida.
VOICE_KEYS: dict[str, str] = {
    "piper": prefs.AUDIO_VOICE_PIPER,
    "kokoro": prefs.AUDIO_VOICE_KOKORO,
    "voice_clone": prefs.AUDIO_VOICE_VOICE_CLONE,
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


#: De dónde es cada voz, por su prefijo de idioma (``es_ES-…`` → ``es_ES``). Se
#: muestra en el desplegable para no elegir sin querer una voz de otro país
#: (p. ej. la argentina ``es_AR-daniela-high``).
LOCALE_LABELS: dict[str, str] = {
    "es_ES": "castellano (España)",
    "es_MX": "español (México)",
    "es_AR": "español (Argentina)",
    "es_CO": "español (Colombia)",
    "eu_ES": "euskera",
    "ca_ES": "catalán",
    "gl_ES": "gallego",
    "en_US": "inglés (EE. UU.)",
    "en_GB": "inglés (Reino Unido)",
}

#: Idioma que se pone primero en el desplegable de voces. La app está en español y
#: la liga del usuario también, así que las voces de España van arriba; el resto
#: sigue estando, sólo que ordenado detrás.
PREFERRED_LOCALE = "es_ES"


def voice_locale(voice: str | None) -> str:
    """Prefijo de idioma/país de una voz (``es_ES-davefx-medium`` → ``es_ES``)."""
    texto = str(voice or "")
    if "-" not in texto or "_" not in texto:
        return ""
    return texto.partition("-")[0]


def voice_label(voice: str | None, motor: str | None = None) -> str:
    """Etiqueta del desplegable: nombre de la voz + de dónde es.

    Los motores que no llevan el idioma en el nombre (Kokoro: ``ef_dora``) se
    anuncian como «español (Kokoro)», porque no son castellano de España.
    """
    nombre = str(voice or "")
    if not nombre:
        return ""
    if motor == "kokoro":
        return f"{nombre} · español (Kokoro)"
    etiqueta = LOCALE_LABELS.get(voice_locale(nombre))
    return f"{nombre} · {etiqueta}" if etiqueta else nombre


def sort_voices(voices, *, prefer: str = PREFERRED_LOCALE) -> list[str]:
    """Voces ordenadas: primero las de ``prefer`` y luego el resto por nombre.

    Piper devuelve los nombres alfabéticamente, así que ``es_AR-daniela-high``
    salía **antes** que ``es_ES-davefx-medium``: en un desplegable eso invita a
    elegir la voz equivocada.
    """

    def clave(voz: str) -> tuple[int, str]:
        return (0 if voice_locale(voz) == prefer else 1, voz)

    return sorted(list(voices), key=clave)
