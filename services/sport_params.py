"""Reglas y vocabulario por disciplina (balonmano, genérico…).

Vive en su propio módulo **neutral** para que lo compartan las tres pestañas que
hablan con la IA (Cuaderno, Crónica y Comprensión) sin crear dependencias
cruzadas: ``notes_params`` ya importa ``video_understanding``, así que las reglas
no podían vivir en ``notes_params``.

Sin esto, el análisis no sabe de qué deporte se trata y confunde disciplinas (un
balonmano se describió como «voleibol/fútbol sala»). El **primero** de la lista es
el valor por defecto que ofrecen los selectores.
"""

from __future__ import annotations

#: Vocabulario y reglas por disciplina. ``clave -> (etiqueta, primer)``. El
#: ``primer`` es el texto que se le pasa al modelo; puede ser vacío (genérico).
SPORT_PRESETS: dict[str, tuple[str, str]] = {
    "balonmano": (
        "Balonmano",
        "Balonmano (reglas y vocabulario para interpretar lo que se ve):\n"
        "- 7 contra 7 (6 de campo + portero) en pista de 40x20 m.\n"
        "- Dos tiempos de 30 min con el cronómetro PARADO (no corre el tiempo).\n"
        "- El área de 6 m es solo del portero; ningún otro jugador puede pisarla.\n"
        "- Un jugador da como máximo 3 pasos sin botar; botar es «regate».\n"
        "- Solo se permite contacto frontal con el torso; no se puede retener, "
        "empujar, golpear ni zancadillear.\n"
        "- Golpe franco desde donde fue la falta; si fue entre 6 y 9 m, desde la "
        "línea de 9 m. Lanzamiento de 7 m (penalti) si se frustra una ocasión clara.\n"
        "- Exclusiones de 2 minutos; tres exclusiones = descalificación (roja).\n"
        "- «Juego pasivo»: tras varias faltas/pases el árbitro obliga a lanzar.\n"
        "- Saques: de centro (inicio y tras gol), de banda, de portería y golpe franco.\n"
        "- Vocabulario: central, lateral, extremo, pivote, portero, contraataque, "
        "exclusión, 7 metros, falta en ataque, bloqueo, área de 6 m, línea de 9 m.\n"
        "- Gol: el balón rebasa por completo la línea entre los postes.",
    ),
    "generico": ("Genérico (sin reglas específicas)", ""),
}

#: Clave de la disciplina por defecto (la primera de :data:`SPORT_PRESETS`).
DEFAULT_SPORT = next(iter(SPORT_PRESETS))


#: Etiqueta con la que las tres pestañas (Comprensión, Cuaderno y Crónica) piden al
#: usuario sus **instrucciones**: qué quiere que cuente la IA y con qué prioridad.
#: Sin esto, la IA decide por su cuenta y el resultado no es el que uno quiere.
INSTRUCTIONS_LABEL = "INSTRUCCIONES DEL USUARIO (lo que quiere contar; respétalas)"

#: Ejemplo de instrucciones (balonmano) que se ofrece como punto de partida en las
#: tres pestañas: presentar la alineación al principio (mientras calientan) y luego
#: centrarse en goles, paradas del portero y lanzamientos de 7 metros. Es editable.
INSTRUCTIONS_EXAMPLE = (
    "Empieza presentando el POLIDEPORTIVO y la FECHA del partido y, justo después, "
    "la ALINEACIÓN DE LOS DOS EQUIPOS (dorsal y nombre), mientras calientan "
    "(momento PREVIO/CALENTAMIENTO).\n"
    "Después céntrate en lo importante: GOLES, PARADAS del portero y LANZAMIENTOS "
    "DE 7 METROS (penaltis).\n"
    "En cada gol, di quién lo marca y cómo (contraataque, extremo, pivote…).\n"
    "En cada parada, di qué portero fue y qué lanzamiento detuvo.\n"
    "Menciona exclusiones de 2 minutos y tiempos muertos solo si cambian el partido."
)


def sport_primer(sport: str | None) -> str:
    """Texto de reglas/vocabulario de la disciplina ``sport`` (o cadena vacía)."""
    preset = SPORT_PRESETS.get((sport or "").strip().lower())
    return preset[1] if preset else ""
