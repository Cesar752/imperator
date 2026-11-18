"""Lectura del **acta** de un partido para alimentar la crónica.

El acta (PDF o foto, la que suben todos los partidos) es la fuente que ya trae lo
que la imagen no puede dar: los nombres y los dorsales de cada equipo, y el
marcador. Aquí vive todo lo decidible sin Qt ni subprocess:

1. El prompt de OCR con el que la visión **transcribe** un acta escaneada.
2. El prompt con el que el chat **estructura** esa transcripción en JSON.
3. El parseo tolerante de la respuesta (JSON del modelo o, si no, líneas
   «dorsal nombre» de la transcripción).
4. El bloque final que se escribe en la caja «Datos del partido» de la crónica.

Dos caminos, y el segundo es sólo un respaldo:

- **Con chat**: se le pasa el texto del acta y devuelve equipos, alineaciones y
  marcador en JSON. Es lo robusto, porque cada federación maqueta el acta a su
  manera.
- **Sin chat**: un lector por líneas saca los pares «dorsal nombre» de las
  transcripciones típicas y adivina el marcador. Mejor eso que nada, pero peor.

El acta no sustituye al vídeo: la crónica sigue contando lo que se ve. Lo que
hace el acta es poner **nombres** donde la visión pondría «un jugador».
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from services import notes_params as nts

#: Extensiones que se intentan leer como acta (PDF con capa de texto o foto).
ACTA_EXTENSIONS = (".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".tif")
#: Filtro del selector de archivo del acta.
ACTA_FILTER = "Acta (*.pdf *.png *.jpg *.jpeg *.bmp *.webp *.tiff *.tif)"

#: Prompt de OCR para una página escaneada: **transcribir**, no resumir.
ACTA_OCR_PROMPT = (
    "Transcribe literalmente el texto de esta imagen, que es el ACTA de un partido "
    "de balonmano. Copia los equipos, los dorsales, los nombres de los jugadores y "
    "el marcador tal y como aparecen, una cosa por línea. No resumas, no traduzcas "
    "y no interpretes: sólo transcribe."
)

#: Prompt con el que el chat convierte el texto del acta en JSON. Los nombres de
#: equipo y el marcador se usan tal cual; lo que no esté en el acta queda vacío.
ACTA_PROMPT = (
    "Te doy el texto de un ACTA de un partido de balonmano. Extrae las alineaciones "
    "y el resultado. Devuelve SÓLO un JSON con esta forma exacta:\n"
    '{"local": "Equipo local", "visitante": "Equipo visitante", '
    '"color_local": "azul", "color_visitante": "roja", "marcador": "28-25", '
    '"jugadores": [{"equipo": "local", "dorsal": 1, "nombre": "Unai", "portero": true}, '
    '{"equipo": "visitante", "dorsal": 10, "nombre": "Samu"}]}\n'
    'En "equipo" pon "local" o "visitante". Si un dato no aparece en el acta, deja '
    "el campo vacío; no inventes nombres ni dorsales.\n\nTEXTO DEL ACTA:\n"
)

#: Un dorsal y un nombre al principio de una línea (respaldo sin chat).
_JUGADOR_RE = re.compile(r"^(\d{1,2})[\s.\-]+([^\d].{2,60})$")
#: Un marcador «28-25», «28:25» o «28 – 25» (respaldo sin chat).
_MARCADOR_RE = re.compile(r"(?<!\d)(\d{1,3})\s*[-–:]\s*(\d{1,3})(?!\d)")
#: Cabecera de la tabla de jugadores de un acta de federación (columnas A y B).
_COL_A = re.compile(r"A:\s*NOMBRE", re.IGNORECASE)
_COL_B = re.compile(r"B:\s*NOMBRE", re.IGNORECASE)
#: Una fila de la tabla: dorsal + NOMBRE, con los goles/parciales detrás (opcional).
#: El ``*`` va en la clase porque marca sanciones («…MIGUENS**  5  45») y se limpia
#: después; si no estuviera, esos jugadores se perderían.
_FILA_JUGADOR = re.compile(
    r"^(\d{1,2})\s+([A-ZÑÁÉÍÓÚÜ][A-Za-zÑñÁÉÍÓÚÜáéíóúü .'*\-]{2,}?)(?:\s+\d.*)?$"
)


def _limpio(valor: object) -> str:
    """Texto de una sola línea (colapsa espacios y saltos)."""
    return " ".join(str(valor or "").split())


def _norm_equipo(valor: object) -> str:
    """Normaliza a ``"local"`` o ``"visitante"`` (por defecto, local)."""
    texto = str(valor or "").strip().lower()
    if texto[:1] in {"v", "2", "b"} or "visit" in texto:
        return "visitante"
    return "local"


def _norm_dorsal(valor: object) -> int | None:
    """Dorsal válido (1-99) o ``None`` si no lo es (evita basura del modelo)."""
    if valor is None or isinstance(valor, bool):
        return None
    try:
        numero = int(float(str(valor).strip()))
    except (TypeError, ValueError):
        return None
    return numero if 0 < numero <= 99 else None


@dataclass(frozen=True)
class ActaJugador:
    """Un jugador del acta: en qué equipo juega, dorsal, nombre y si es portero."""

    equipo: str
    dorsal: int
    nombre: str
    portero: bool = False


@dataclass(frozen=True)
class Acta:
    """Lo que se saca de un acta: equipos, colores, marcador y alineaciones."""

    local: str = ""
    visitante: str = ""
    color_local: str = ""
    color_visitante: str = ""
    marcador: str = ""
    jugadores: tuple[ActaJugador, ...] = ()

    def __post_init__(self) -> None:
        # El orden (local primero y por dorsal) es un invariante del acta, no algo
        # que dependa de quién la construya: así lo lea el usuario o el modelo, sale
        # siempre igual.
        object.__setattr__(self, "jugadores", _ordenar(self.jugadores))

    def jugadores_de(self, equipo: str) -> tuple[ActaJugador, ...]:
        """Jugadores de un equipo (``"local"`` o ``"visitante"``)."""
        clave = _norm_equipo(equipo)
        return tuple(jugador for jugador in self.jugadores if jugador.equipo == clave)


def _ordenar(jugadores: Sequence[ActaJugador]) -> tuple[ActaJugador, ...]:
    """Local primero y, dentro de cada equipo, por dorsal (orden de lectura)."""
    return tuple(sorted(jugadores, key=lambda j: (j.equipo != "local", j.dorsal, j.nombre)))


def _acta_desde_json(datos: dict) -> Acta:
    """Construye el acta desde el JSON del chat (claves alternativas incluidas)."""
    jugadores: list[ActaJugador] = []
    entradas = datos.get("jugadores") or datos.get("alineacion") or datos.get("plantilla") or []
    if isinstance(entradas, list):
        for entrada in entradas:
            if not isinstance(entrada, dict):
                continue
            dorsal = _norm_dorsal(entrada.get("dorsal", entrada.get("numero")))
            nombre = _limpio(entrada.get("nombre") or entrada.get("jugador"))
            if dorsal is None or len(nombre) < 2:
                continue
            jugadores.append(
                ActaJugador(
                    equipo=_norm_equipo(entrada.get("equipo")),
                    dorsal=dorsal,
                    nombre=nombre,
                    portero=bool(entrada.get("portero")),
                )
            )
    return Acta(
        local=_limpio(datos.get("local") or datos.get("equipo_local")),
        visitante=_limpio(datos.get("visitante") or datos.get("equipo_visitante")),
        color_local=_limpio(datos.get("color_local")),
        color_visitante=_limpio(datos.get("color_visitante")),
        marcador=_limpio(datos.get("marcador") or datos.get("resultado")),
        jugadores=_ordenar(jugadores),
    )


def _acta_desde_texto(raw: str) -> Acta:
    """Respaldo sin chat: dos lecturas, de la más fiable a la menos.

    Primero la **tabla** de un acta de federación (columnas «A: NOMBRE» y «B:
    NOMBRE» con filas «dorsal NOMBRE», y el marcador junto a «FINAL»). Si no hay
    tabla, se cae a las **líneas sueltas** «dorsal nombre». Es sólo un respaldo: lo
    bueno es el chat, que entiende cualquier maquetación.
    """
    texto = str(raw or "")
    lineas = texto.splitlines()
    local, visitante, marcador = _equipos_y_marcador(texto)
    jugadores = _jugadores_de_tabla(lineas)
    if not jugadores:
        jugadores = _jugadores_sueltos(lineas)
    if not marcador:
        marcador = _marcador_con_contexto(texto)
    return Acta(
        local=local,
        visitante=visitante,
        marcador=marcador,
        jugadores=_ordenar(jugadores),
    )


def _equipos_y_marcador(texto: str) -> tuple[str, str, str]:
    """Equipos y marcador del acta (junto a «A: EQUIPO», «B: EQUIPO» y «FINAL»)."""
    local = ""
    visitante = ""
    marcador = ""
    # El nombre del local va JUSTO antes de «A: EQUIPO ORGANIZADOR:» y, al extraer
    # el texto, puede quedar pegado («…MUSKIZA: EQUIPO…»): el `\s*` lo tolera.
    hallado = re.search(
        r"([A-ZÑÁÉÍÓÚÜ][A-Za-zÑñ0-9. '&/-]{2,40}?)\s*A:\s*EQUIPO", texto
    )
    if hallado:
        local = _limpio(hallado.group(1))
    hallado = re.search(r"B:\s*EQUIPO VISITANTE:?\s*(.+)", texto)
    if hallado:
        # El nombre va al principio de esa línea; lo que venga tras 2+ espacios es
        # otra columna (resultado, club…), así que se corta ahí.
        visitante = _limpio(re.split(r"\s{2,}", hallado.group(1), maxsplit=1)[0])
    hallado = re.search(r"FINAL:?\s*(\d{1,3})\s*[-–:]\s*(\d{1,3})", texto)
    if hallado:
        marcador = f"{hallado.group(1)}-{hallado.group(2)}"
    return local, visitante, marcador


def _parsear_jugador(parte: str) -> tuple[int, str] | None:
    """``(dorsal, nombre)`` de media fila de la tabla, o ``None`` si no hay jugador."""
    coincidencia = _FILA_JUGADOR.match(_limpio(parte))
    if not coincidencia:
        return None
    dorsal = _norm_dorsal(coincidencia.group(1))
    nombre = _limpio(re.sub(r"[*]", " ", coincidencia.group(2)))
    if dorsal is None or len(nombre) < 3:
        return None
    return dorsal, nombre


def _jugadores_de_tabla(lineas: Sequence[str]) -> list[ActaJugador]:
    """Jugadores de una tabla de acta de dos columnas (A = local, B = visitante).

    La cabecera «Nº … A: NOMBRE … Nº … B: NOMBRE …» marca dónde empieza cada
    columna; se corta cada fila por el «Nº» de la B y cada mitad se lee aparte.
    """
    corte = 0
    inicio = None
    for posicion, linea in enumerate(lineas):
        columna_b = _COL_B.search(linea)
        if _COL_A.search(linea) and columna_b:
            pos_b = columna_b.start()
            previo = linea.rfind("Nº", 0, pos_b)
            corte = previo if previo > 0 else pos_b
            inicio = posicion + 1
            break
    if inicio is None or corte <= 0:
        return []
    jugadores: list[ActaJugador] = []
    for linea in lineas[inicio:]:
        izquierda = _parsear_jugador(linea[:corte])
        if izquierda:
            jugadores.append(ActaJugador("local", izquierda[0], izquierda[1]))
        derecha = _parsear_jugador(linea[corte:])
        if derecha:
            jugadores.append(ActaJugador("visitante", derecha[0], derecha[1]))
    return jugadores


def _jugadores_sueltos(lineas: Sequence[str]) -> list[ActaJugador]:
    """Respaldo simple: líneas «dorsal nombre», con el equipo de la última cabecera."""
    jugadores: list[ActaJugador] = []
    equipo = "local"
    for linea in lineas:
        texto = _limpio(linea)
        if not texto:
            continue
        bajo = texto.lower()
        if "visitante" in bajo:
            equipo = "visitante"
        elif "local" in bajo:
            equipo = "local"
        coincidencia = _JUGADOR_RE.match(texto)
        if not coincidencia:
            continue
        dorsal = _norm_dorsal(coincidencia.group(1))
        nombre = _limpio(re.sub(r"\bportero\b|[()]", " ", coincidencia.group(2)))
        if dorsal is None or len(nombre) < 3:
            continue
        jugadores.append(ActaJugador(equipo, dorsal, nombre, portero="portero" in bajo))
    return jugadores


def _marcador_con_contexto(texto: str) -> str:
    """Marcador «a-b» de una línea que hable de resultado/final (evita la hora)."""
    for linea in texto.splitlines():
        bajo = linea.lower()
        if not any(clave in bajo for clave in ("resultado", "final", "marcador")):
            continue
        hallado = _MARCADOR_RE.search(linea)
        if hallado:
            return f"{hallado.group(1)}-{hallado.group(2)}"
    return ""


def parse_acta(raw: str) -> Acta:
    """Convierte la respuesta (JSON del chat o texto del acta) en un :class:`Acta`.

    Se intenta primero el JSON; si no hay o no trae nada útil, se cae al lector por
    líneas. Nunca lanza: un acta ilegible devuelve un :class:`Acta` vacío y quien
    llama decide qué hacer.
    """
    datos = None
    try:
        datos = nts.extract_json(raw)
    except nts.NotesError:
        datos = None
    if isinstance(datos, dict):
        acta = _acta_desde_json(datos)
        if acta.jugadores or acta.local or acta.visitante or acta.marcador:
            return acta
    return _acta_desde_texto(raw)


def _equipo_descr(nombre: str, rol: str, color: str) -> str:
    """«Muskis (local, camiseta azul)»; con lo que falte, se describe igual."""
    etiqueta = nombre or ("equipo local" if rol == "local" else "equipo visitante")
    detalle = rol + (f", camiseta {color}" if color else "")
    return f"{etiqueta} ({detalle})"


def _lista_jugadores(jugadores: Sequence[ActaJugador]) -> str:
    """«1 Unai (portero), 4 Álex, 7 Marc» en una sola línea."""
    partes = [
        f"{jugador.dorsal} {jugador.nombre}" + (" (portero)" if jugador.portero else "")
        for jugador in jugadores
    ]
    return ", ".join(partes)


def acta_context(acta: Acta) -> str:
    """Bloque de texto para la caja «Datos del partido» de la crónica.

    Mismo formato que el ejemplo de :data:`chronicle_params.CONTEXT_EXAMPLE`, para
    que lo lea igual el usuario y el modelo: una línea por dato, sin adornos.
    """
    lineas: list[str] = []
    if acta.local or acta.visitante:
        lineas.append(
            "Equipos: "
            + _equipo_descr(acta.local, "local", acta.color_local)
            + " vs "
            + _equipo_descr(acta.visitante, "visitante", acta.color_visitante)
            + "."
        )
    for equipo, etiqueta in (("local", acta.local), ("visitante", acta.visitante)):
        jugadores = acta.jugadores_de(equipo)
        if not jugadores:
            continue
        nombre_equipo = etiqueta or ("el local" if equipo == "local" else "el visitante")
        lineas.append(f"Alineación {nombre_equipo}: {_lista_jugadores(jugadores)}")
    if acta.marcador:
        lineas.append(f"Marcador final: {acta.marcador}")
    return "\n".join(lineas)
