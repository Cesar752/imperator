"""Listas y helpers puros del Ensamblador (sin Qt)."""

# Idiomas disponibles para la transcripción (código -> etiqueta).
LANGUAGES: list[tuple[str, str]] = [
    ("auto", "Auto"),
    ("es", "Español"),
    ("en", "Inglés"),
    ("eu", "Euskera"),
    ("fr", "Francés"),
    ("de", "Alemán"),
    ("it", "Italiano"),
    ("pt", "Portugués"),
]

# Posición y color de los subtítulos (clave interna -> etiqueta).
POSITIONS: list[tuple[str, str]] = [
    ("bottom", "Abajo"),
    ("middle", "Centro"),
    ("top", "Arriba"),
]
COLORS: list[tuple[str, str]] = [
    ("white", "Blanco"),
    ("yellow", "Amarillo"),
    ("green", "Verde"),
]

# Grosor del contorno del subtítulo (px), fijo por ahora (no expuesto en la UI).
DEFAULT_OUTLINE = 2

#: Tamaño de letra por defecto de los subtítulos (px, con la resolución del vídeo).
DEFAULT_FONTSIZE = 34

# Diferencia de duración (s) a partir de la cual se avisa al usuario.
DURATION_GAP_WARNING = 2.0


def build_subtitle_style(fontsize: int, position: str, color: str) -> dict:
    """Estilo de subtítulos para ``VideoAssembler.assemble``."""
    return {
        "fontsize": fontsize,
        "position": position,
        "color": color,
        "outline": DEFAULT_OUTLINE,
    }


def duration_gap_message(
    video_duration: float | None,
    audio_duration: float | None,
    gap: float = DURATION_GAP_WARNING,
) -> str | None:
    """Aviso si el audio y el vídeo difieren demasiado; ``None`` si no aplica."""
    if video_duration is None or audio_duration is None:
        return None
    if abs(video_duration - audio_duration) <= gap:
        return None
    return (
        f"Aviso: el audio dura {audio_duration:.1f} s y el vídeo "
        f"{video_duration:.1f} s. El vídeo es el maestro: el audio se cortará o "
        "completará con silencio para ajustarse a su duración."
    )


def offset_too_late(offset: float, video_duration: float, margin: float = 0.5) -> bool:
    """¿El inicio (voz/música) cae después del final del vídeo?"""
    return offset > video_duration - margin
