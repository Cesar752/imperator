"""Marcador y goles a partir del texto que el OCR lee en pantalla.

Lógica **pura** (sin Qt, sin ffmpeg, sin motores): convierte el texto OCR
(«MUSKIZ 12 - 9 CORAZONISTAS») en una **serie temporal del marcador** y, de ahí,
en la **lista de goles** (cada vez que un lado sube exactamente 1).

Decisiones (pensadas para no inventar nada):

- Solo se aceptan tanteos separados por **guion** (``-``/``–``/``—``). El reloj usa
  dos puntos («25:30») y confundirlo con el marcador fue justo el error del modelo
  de visión; las medidas de pista (``40x20``) se evitan al no aceptar ``x``.
- Un **gol** es un incremento de **exactamente +1 en un solo lado** entre dos
  lecturas consecutivas. Los saltos (OCR sucio, «12-9» → «17-13») **no** son
  goles: se descartan, no se cuentan de más.
- No se decide qué lado es local y cuál visitante: el OCR lee de izquierda a
  derecha y ese orden no tiene por qué coincidir con «local primero». Se informa
  como **izquierda/derecha**, que es lo único que el dato permite afirmar.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from services.video_understanding import format_timestamp

#: Dos números de 1-2 cifras separados por guion, aislados de otros dígitos.
#: A propósito **no** acepta «:» (es el reloj) ni «x» (medidas de pista, «40x20»).
_SCORE_RE = re.compile(r"(?<!\d)(\d{1,2})\s*[-–—]\s*(\d{1,2})(?!\d)")

#: Marcador con los goles separados por **espacios** («LOCAL VISITANTE 08 07»).
#: El OCR de muchas cámaras de pabellón no mete guion: lee las dos cifras de la
#: pantalla pegadas tras las etiquetas. Se exige una etiqueta LOCAL/VISITANTE
#: (una o repetidas: «LOCAL VISITANTE») delante, con a lo sumo tres palabras de
#: adorno en medio. Se captura la **racha** de números que sigue (1 a 3) y en
#: :func:`extract_score` solo se acepta si son **exactamente dos**: una racha de
#: tres («… VISITANTE 2 42 29») es una lectura sucia y se descarta antes que
#: inventar un marcador.
_SCORE_LABEL_RE = re.compile(
    r"\b(?:local|visitante)\b(?:\s+\b(?:local|visitante)\b)*"
    r"(?:\s+[^\d\s]+){0,3}?\s+"
    r"((?:\d{1,2})(?:\s+\d{1,2}){0,2})(?!\d)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ScorePoint:
    """Una lectura del marcador en un instante (izquierda-derecha, como se lee)."""

    t_s: float
    izquierda: int
    derecha: int

    @property
    def marcador(self) -> str:
        return f"{self.izquierda}-{self.derecha}"


@dataclass(frozen=True)
class Goal:
    """Un gol detectado: qué lado lo hizo y el marcador resultante."""

    t_s: float
    lado: str  #: "izquierda" o "derecha"
    marcador: str
    dudoso: bool = False

    @property
    def label(self) -> str:
        return format_timestamp(self.t_s)


def extract_score(text: str) -> tuple[int, int] | None:
    """Primer tanteo plausible de ``text`` (o ``None`` si no hay).

    Acepta dos formatos, en orden:

    1. Pareja separada por guion («12-9», «28 – 25»).
    2. Pareja **separada por espacios** tras las etiquetas LOCAL/VISITANTE
       («LOCAL VISITANTE 08 07»): es como lee el marcador electrónico la cámara
       cuando el OCR no capta el guion.

    Ignora el reloj («25:30») y las medidas de pista («40x20») en ambos casos.
    """
    texto = str(text or "")
    match = _SCORE_RE.search(texto)
    if match:
        return int(match.group(1)), int(match.group(2))
    for match in _SCORE_LABEL_RE.finditer(texto):
        numeros = match.group(1).split()
        if len(numeros) == 2:
            return int(numeros[0]), int(numeros[1])
    return None


#: Reloj del marcador electrónico «MM:SS» (minuto de 1-3 cifras, segundos 00-59).
#: A propósito NO acepta el tanteo (usa guion) ni las medidas de pista: el reloj
#: es lo que dice CUÁNDO corre el partido.
_CLOCK_RE = re.compile(r"(?<!\d)(\d{1,3}):([0-5]\d)(?!\d)")


def extract_clock(text: str) -> int | None:
    """Segundos del primer reloj «MM:SS» de ``text`` (o ``None`` si no hay)."""
    match = _CLOCK_RE.search(str(text or ""))
    if not match:
        return None
    return int(match.group(1)) * 60 + int(match.group(2))


def game_start(lecturas: Sequence[tuple[float, str]]) -> float | None:
    """Instante en que el juego ARRANCA: cuando el reloj empieza a moverse.

    Recorre las lecturas OCR con su reloj y devuelve el ``t_s`` de la última
    lectura en la que el reloj aún estaba **parado** justo antes de que empezara a
    correr (la primera lectura cuyo reloj supera al anterior): ahí empieza el
    partido, no en el minuto que «estime» el modelo. Antes de ese instante es
    calentamiento/saludo (PREVIO). ``None`` si no se leyó reloj o nunca se movió.
    """
    previo: int | None = None
    ultimo_parado: float | None = None
    for t_s, texto in lecturas or ():
        reloj = extract_clock(texto)
        if reloj is None:
            continue
        if previo is not None and reloj > previo:
            return ultimo_parado if ultimo_parado is not None else float(t_s)
        previo = reloj
        ultimo_parado = float(t_s)
    return None


def half_time(
    lecturas: Sequence[tuple[float, str]], *, stall_s: float = 120.0
) -> float | None:
    """Instante del descanso (entretiempo), o ``None`` si no se detecta.

    Se detecta de dos formas:
    1. **Reinicio**: el reloj retrocede más de un minuto (p. ej. de 30:00 a 00:00):
       ahí acaba el primer tiempo.
    2. **Reloj parado**: el reloj queda igual durante ``stall_s`` segundos: el
       descanso empieza cuando el reloj dejó de avanzar.
    """
    previo: int | None = None
    ultimo_reloj: int | None = None
    ultimo_cambio_t: float | None = None
    for t_s, texto in lecturas or ():
        reloj = extract_clock(texto)
        if reloj is None:
            continue
        if previo is not None and reloj < previo - 60:
            return float(t_s)
        if ultimo_reloj is None or reloj != ultimo_reloj:
            ultimo_reloj = reloj
            ultimo_cambio_t = float(t_s)
        elif ultimo_cambio_t is not None and (float(t_s) - ultimo_cambio_t) >= stall_s:
            return float(ultimo_cambio_t)
        previo = reloj
    return None


def score_series(lecturas: Sequence[tuple[float, str]]) -> list[ScorePoint]:
    """Serie del marcador a partir de ``(t_s, texto OCR)``.

    Descarta las lecturas sin tanteo y las **repeticiones** del mismo par: la
    serie marca solo los cambios, que es lo que importa para los goles.
    """
    serie: list[ScorePoint] = []
    for t_s, texto in lecturas or ():
        par = extract_score(texto)
        if par is None:
            continue
        if serie and (serie[-1].izquierda, serie[-1].derecha) == par:
            continue
        serie.append(ScorePoint(float(t_s), par[0], par[1]))
    return serie


def goals_from_series(serie: Sequence[ScorePoint]) -> list[Goal]:
    """Goles de la serie: +1 exacto en un solo lado entre lecturas consecutivas.

    Los cambios que no encajan (saltos por OCR sucio, o bajadas) **no** se cuentan
    como goles: se prefieren menos goles verdaderos que goles inventados.
    """
    goles: list[Goal] = []
    for anterior, actual in zip(serie, serie[1:]):
        delta_izq = actual.izquierda - anterior.izquierda
        delta_der = actual.derecha - anterior.derecha
        if delta_izq == 1 and delta_der == 0:
            goles.append(Goal(actual.t_s, "izquierda", actual.marcador))
        elif delta_der == 1 and delta_izq == 0:
            goles.append(Goal(actual.t_s, "derecha", actual.marcador))
    return goles


def marcador_block(lecturas: Sequence[tuple[float, str]]) -> str:
    """Bloque «Marcador leído por OCR» (una línea por cambio) para el prompt.

    Devuelve cadena vacía si no se leyó ningún tanteo: así el prompt no se llena
    de un bloque vacío cuando el OCR no está disponible.
    """
    return series_block(score_series(lecturas))


def series_block(serie: Sequence[ScorePoint]) -> str:
    """Bloque «Marcador leído por OCR» a partir de una serie ya calculada.

    Permite reutilizar la serie leída por otra pestaña (p. ej. Comprensión) sin
    volver a hacer OCR: el Cuaderno la recibe como datos ya leídos.
    """
    if not serie:
        return ""
    lineas = [f"[{p.t_s:.0f} s] {p.marcador}" for p in serie]
    goles = goals_from_series(serie)
    if goles:
        lineas.append(f"Total de goles detectados: {len(goles)}.")
    return "\n".join(lineas)


def scoreboard_summary(lecturas: Sequence[tuple[float, str]]) -> str:
    """Resumen de una línea para la UI (o cadena vacía si no hay tanteos)."""
    return series_summary(score_series(lecturas))


def series_summary(serie: Sequence[ScorePoint]) -> str:
    """Resumen de una línea a partir de una serie ya calculada."""
    if not serie:
        return ""
    goles = goals_from_series(serie)
    izq = sum(1 for gol in goles if gol.lado == "izquierda")
    der = sum(1 for gol in goles if gol.lado == "derecha")
    return (
        f"Marcador OCR: {len(serie)} cambios leídos, de {serie[0].marcador} a "
        f"{serie[-1].marcador}; {len(goles)} goles detectados ({izq} · {der})."
    )
