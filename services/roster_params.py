"""Alineaciones del usuario: de «dorsal 7» al **nombre de pila**.

La visión y el OCR sólo pueden dar **dorsales y colores**: no ven nombres. Los
nombres viven en los «Datos del partido» que aporta el usuario (o el acta), con la
forma habitual::

    Equipos: Muskis (local, camiseta azul) vs Corazonistas (visitante, roja).
    Alineación Muskis: 1 Unai (portero), 4 Álex, 7 Marc, 9 Iván.

Este módulo —lógica pura, sin Qt ni motores— convierte esa alineación en tres cosas:

- La lista de jugadoras (:func:`parse_roster`), cada una con su **nombre de pila**
  (``AMANDA OTERO BENITO`` → ``Amanda``).
- El bloque de prompt (:func:`roster_block`) que pide llamarlas por su nombre de
  pila. Se inyecta en los prompts del Cuaderno y de la Crónica.
- Un **reescritor determinista** (:func:`resolve_roster_names`) que traduce
  «dorsal 7» / «el 7» a «Marc» en el texto que produce la IA. Es la red de
  seguridad: un modelo local no siempre obedece al prompt, y así el dorsal y el
  color no llegan al relato.

Sin alineación no se inventa nada: los nombres salen **sólo** de lo que escribió el
usuario (o de lo que trajo el acta).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

#: Claves de línea que NO son alineaciones: «Equipos:», «Marcador final:»… Si no se
#: filtran, su parte derecha se intentaría leer como «dorsal nombre».
_CLAVES_SIN_JUGADORAS = (
    "equipo",
    "marcador",
    "resultado",
    "tanteo",
    "nota",
    "deporte",
    "disciplina",
    "competicion",
    "fecha",
    "hora",
    "arbitro",
    "lugar",
    "pabellon",
    "color",
    "cancha",
    "pista",
    "categoria",
    "observacion",
    "comentario",
)

#: Palabras que introducen la lista y NO forman parte del nombre del equipo
#: («Alineación Muskis: …» → equipo «Muskis»).
_PREFIJOS_ALINEACION = (
    "alineacion",
    "alineaciones",
    "jugadores",
    "jugadoras",
    "plantilla",
    "convocatoria",
    "lista",
)

#: Colores de camiseta reconocidos (para ligar cada equipo con su color). Incluye
#: las formas de femenino y plural que se escriben de verdad («camisetas rojas»).
_COLORES = (
    "azul",
    "azules",
    "rojo",
    "roja",
    "rojos",
    "rojas",
    "verde",
    "verdes",
    "blanco",
    "blanca",
    "blancos",
    "blancas",
    "negro",
    "negra",
    "negros",
    "negras",
    "amarillo",
    "amarilla",
    "amarillos",
    "amarillas",
    "naranja",
    "naranjas",
    "gris",
    "grises",
    "morado",
    "morada",
    "morados",
    "moradas",
    "violeta",
    "violetas",
    "rosa",
    "rosas",
    "celeste",
    "celestes",
    "granate",
    "granates",
    "burdeos",
    "grana",
    "granas",
    "turquesa",
    "marron",
    "marrones",
)

#: Color en su forma canónica: «rojo»/«roja» son el mismo color, para no fallar al
#: cruzar «el 1 rojo» con «camiseta roja».
_RAIZ_COLOR = {
    "azul": "azul",
    "azules": "azul",
    "rojo": "rojo",
    "roja": "rojo",
    "rojos": "rojo",
    "rojas": "rojo",
    "verde": "verde",
    "verdes": "verde",
    "blanco": "blanco",
    "blanca": "blanco",
    "blancos": "blanco",
    "blancas": "blanco",
    "negro": "negro",
    "negra": "negro",
    "negros": "negro",
    "negras": "negro",
    "amarillo": "amarillo",
    "amarilla": "amarillo",
    "amarillos": "amarillo",
    "amarillas": "amarillo",
    "naranja": "naranja",
    "naranjas": "naranja",
    "gris": "gris",
    "grises": "gris",
    "morado": "morado",
    "morada": "morado",
    "morados": "morado",
    "moradas": "morado",
    "violeta": "violeta",
    "violetas": "violeta",
    "rosa": "rosa",
    "rosas": "rosa",
    "celeste": "celeste",
    "celestes": "celeste",
    "granate": "granate",
    "granates": "granate",
    "burdeos": "burdeos",
    "grana": "granate",
    "granas": "granate",
    "turquesa": "turquesa",
    "marron": "marron",
    "marrones": "marron",
}

#: Palabras que no son nombre de equipo en la línea «Equipos:».
_ROL_EQUIPO = ("local", "visitante", "camiseta", "equipacion", "de", "equipo")

#: Palabras de tiempo que, tras un número, delatan que NO es un dorsal
#: («la 2 parte», «el 7 minuto»): no se debe poner el nombre de nadie.
_PALABRAS_TIEMPO = (
    "parte",
    "tiempo",
    "periodo",
    "minuto",
    "min",
    "minutos",
    "segundo",
    "segundos",
    "hora",
    "horas",
    "manga",
    "set",
    "acto",
)

#: Una línea «clave: resto» (la clave suele ser el equipo).
_LINEA_RE = re.compile(r"^(?P<clave>[^:]{2,60}?)\s*:\s*(?P<resto>\S.*)$")
#: Un elemento «dorsal nombre» de la lista de una alineación.
_ITEM_RE = re.compile(r"^(\d{1,2})\s*[.)\-–]?\s+(.+)$")

#: Alternancia de colores, para consumir «el 7 azul» entero al poner el nombre.
#: Ordenada de más larga a más corta: si «roja» va antes que «rojas», el regex
#: casaría «roja» y dejaría la «s» suelta («el 15 rojas» -> «Beas»).
_COLORES_RE = "|".join(sorted(_COLORES, key=len, reverse=True))

#: Patrones de referencia a un dorsal en un texto ya redactado, en orden. El 1.º
#: cubre «con el dorsal 6», «del dorsal 9»…; el 2.º, «(dorsal 3)» sin artículo; el
#: 3.º y el 4.º, «nº 3» y «#7»; el 5.º, «el 7» con su color opcional «el 7 azul».
_PATRONES_DORSAL = tuple(
    re.compile(patron, re.IGNORECASE)
    for patron in (
        r"\b(?:el|la|los|las|del|al)\s+dorsal(?:es)?\b[^0-9]{0,6}(\d{1,2})(?!\d)",
        r"\bdorsal(?:es)?\b[^0-9]{0,6}(\d{1,2})(?!\d)",
        r"\bn[ºo°]\.?\s*(\d{1,2})(?!\d)",
        r"#\s*(\d{1,2})(?!\d)",
        rf"\b(?:el|la)\s+(\d{{1,2}})(?!\d)(?:\s+({_COLORES_RE}))?",
    )
)


def raices_color(texto: str) -> set[str]:
    """Colores (canónicos) que aparecen en ``texto``.

    «rojo», «roja» y «rojas» devuelven el mismo color: sirve para comparar el
    contexto del usuario con lo que describe el vídeo sin fallar por el género o
    el número.
    """
    bajo = _sin_acentos(texto)
    return {
        _RAIZ_COLOR[color] for color in _COLORES if re.search(rf"\b{color}\b", bajo)
    }


def _sin_acentos(texto: str) -> str:
    """Minúsculas y sin acentos: para comparar nombres sin falsos negativos."""
    base = unicodedata.normalize("NFKD", str(texto or ""))
    return "".join(c for c in base if not unicodedata.combining(c)).lower().strip()


def primer_nombre(nombre_completo: str) -> str:
    """Nombre de pila de un nombre completo: «AMANDA OTERO BENITO» → «Amanda»."""
    partes = str(nombre_completo or "").split()
    if not partes:
        return ""
    return partes[0].capitalize()


@dataclass(frozen=True)
class RosterPlayer:
    """Una jugadora de la alineación: su equipo, su dorsal y su nombre completo."""

    equipo: str
    dorsal: int
    nombre_completo: str

    @property
    def nombre(self) -> str:
        """Nombre de pila (lo que se usa para llamarla)."""
        return primer_nombre(self.nombre_completo)

    @property
    def equipo_clave(self) -> str:
        """Nombre del equipo normalizado (sin acentos, minúsculas)."""
        return _sin_acentos(self.equipo)


def _limpia_nombre(resto: str) -> str:
    """Nombre utilizable de la parte derecha de un elemento de la alineación."""
    texto = " ".join(str(resto or "").split())
    texto = re.sub(r"\([^)]*\)", " ", texto)  # (portero)
    texto = re.sub(r"\bporter[oa]s?\b", " ", texto, flags=re.IGNORECASE)
    texto = texto.replace("*", " ").strip(" .,-–")
    texto = re.sub(r"\s{2,}", " ", texto).strip()
    if not texto or texto[:1].isdigit():
        return ""
    return texto


def _items_de_lista(resto: str) -> list[tuple[int, str]]:
    """Pares ``(dorsal, nombre)`` de la parte derecha de una línea de alineación.

    Se corta por comas cuando las hay y, si no, por el dorsal siguiente: así
    funciona tanto el formato «1 Unai, 4 Álex, 7 Marc» como una lista pegada
    «1 Unai 4 Álex 7 Marc».
    """
    texto = re.sub(r"[;|]", ",", str(resto or ""))
    if "," in texto:
        trozos = texto.split(",")
    else:
        trozos = re.split(r"(?=(?<!\d)\d{1,2}\s+\D)", texto)
    pares: list[tuple[int, str]] = []
    for trozo in trozos:
        match = _ITEM_RE.match(trozo.strip())
        if not match:
            continue
        dorsal = int(match.group(1))
        nombre = _limpia_nombre(match.group(2))
        if nombre and 0 < dorsal <= 99:
            pares.append((dorsal, nombre))
    return pares


def _nombre_equipo(clave: str) -> str:
    """Equipo de una clave «Alineación Muskis» → «Muskis»."""
    palabras = " ".join(str(clave or "").split()).split()
    while palabras and _sin_acentos(palabras[0]).rstrip(":.") in _PREFIJOS_ALINEACION:
        palabras.pop(0)
    return " ".join(palabras).strip(" .:-")


def _es_clave_sin_jugadoras(clave: str) -> bool:
    """¿La clave de la línea es de un dato que no son jugadoras?"""
    bajo = _sin_acentos(clave)
    return any(bajo.startswith(generico) for generico in _CLAVES_SIN_JUGADORAS)


def parse_roster(context: str) -> tuple[RosterPlayer, ...]:
    """Jugadoras de las líneas «Alineación <equipo>: <dorsal> <nombre>, …».

    Devuelve ``()`` si el contexto no trae alineaciones: entonces los nombres no
    salen de ninguna parte y no hay que inventarlos.
    """
    jugadores: list[RosterPlayer] = []
    vistos: set[tuple[str, int, str]] = set()
    for linea in str(context or "").splitlines():
        limpia = " ".join(linea.split())
        if not limpia:
            continue
        match = _LINEA_RE.match(limpia)
        if not match or _es_clave_sin_jugadoras(match.group("clave")):
            continue
        equipo = _nombre_equipo(match.group("clave"))
        for dorsal, nombre in _items_de_lista(match.group("resto")):
            clave = (_sin_acentos(equipo), dorsal, _sin_acentos(nombre))
            if clave in vistos:
                continue
            vistos.add(clave)
            jugadores.append(RosterPlayer(equipo, dorsal, nombre))
    return tuple(jugadores)


def _equipo_colores(context: str) -> dict[str, str]:
    """``{equipo_normalizado: color canónico}`` de la línea «Equipos: …»."""
    colores: dict[str, str] = {}
    for linea in str(context or "").splitlines():
        limpia = " ".join(linea.split())
        if not limpia:
            continue
        match = _LINEA_RE.match(limpia)
        if not match or not _sin_acentos(match.group("clave")).startswith("equipo"):
            continue
        lados = re.split(
            r"\bvs\b|\bcontra\b|/|--| - ", match.group("resto"), flags=re.IGNORECASE
        )
        for lado in lados:
            raices = raices_color(lado)
            color = next(
                (_RAIZ_COLOR[c] for c in _COLORES if _RAIZ_COLOR[c] in raices), ""
            )
            sin_paren = re.sub(r"\([^)]*\)", " ", lado)
            palabras = [
                palabra
                for palabra in sin_paren.split()
                if _sin_acentos(palabra) not in _COLORES
                and _sin_acentos(palabra) not in _ROL_EQUIPO
            ]
            equipo = _sin_acentos(" ".join(palabras)).strip(" .,-")
            # El equipo se guarda aunque no se le encuentre color: sirve para
            # desambiguar dorsales por nombre de equipo («el 15 de Corazonistas»).
            if equipo:
                colores[equipo] = color
    return colores


def _color_de_equipo(equipo_clave: str, colores: dict[str, str]) -> str:
    """Color de un equipo, tolerando «Corazonistas» vs «Corazonistas Senior»."""
    if not equipo_clave:
        return ""
    if equipo_clave in colores:
        return colores[equipo_clave]
    for clave, color in colores.items():
        if clave.startswith(equipo_clave) or equipo_clave.startswith(clave):
            return color
    return ""


def team_color_summary(context: str) -> str:
    """Mapa breve «color → equipo (LOCAL/VISITANTE)» de la línea «Equipos:».

    La visión sólo ve colores: con esto puede decir a qué equipo —y si es local o
    visitante— pertenece quien tiene el balón, en vez de «el equipo rojo» (que es
    lo que confundía ataque y defensa y hacía decir que ganaba quien no iba por
    delante). Devuelve "" si no hay línea de equipos.
    """
    linea = ""
    for cruda in str(context or "").splitlines():
        limpia = " ".join(cruda.split())
        coincide = re.match(r"^equipos?\b[^:]{0,40}?(?:[:\-]\s*|\s+)(.+)$", limpia, re.IGNORECASE)
        if coincide:
            linea = coincide.group(1).strip()
            break
    if not linea:
        return ""
    trozos = re.split(r"\s+vs\.?\s+|\s+contra\s+", linea, flags=re.IGNORECASE)
    if len(trozos) < 2:
        return ""
    partes: list[str] = []
    for indice, trozo in enumerate(trozos[:2]):
        limpio = " ".join(trozo.split())
        bajo = limpio.lower()
        if "visitante" in bajo:
            rol = "VISITANTE"
        elif "local" in bajo:
            rol = "LOCAL"
        else:
            rol = "LOCAL" if indice == 0 else "VISITANTE"
        colores = sorted(raices_color(limpio))
        nombre = re.sub(r"\([^)]*\)", " ", limpio).strip(" .,-")
        etiqueta = ", ".join(color.capitalize() for color in colores)
        prefijo = f"{etiqueta} → " if etiqueta else ""
        partes.append(f"{prefijo}{nombre} ({rol})")
    return "; ".join(partes)


def roster_block(context: str) -> str:
    """Bloque de prompt que traduce cada dorsal a su nombre de pila (o "").

    Se añade a los prompts que redactan (Cuaderno y Crónica) para que el modelo
    llame a las jugadoras por su nombre y no por «dorsal 7» o «la del rojo».
    """
    jugadores = parse_roster(context)
    if not jugadores:
        return ""
    colores = _equipo_colores(context)
    lineas: list[str] = []
    for jugador in jugadores:
        detalle: list[str] = []
        if jugador.equipo:
            detalle.append(jugador.equipo)
        color = _color_de_equipo(jugador.equipo_clave, colores)
        if color:
            detalle.append(color)
        sufijo = f" ({', '.join(detalle)})" if detalle else ""
        lineas.append(f"- Dorsal {jugador.dorsal}: {jugador.nombre}{sufijo}")
    return (
        "CÓMO LLAMAR A LAS JUGADORAS (usa el NOMBRE DE PILA, nunca el dorsal ni el "
        "color; el dorsal y el color sólo sirven para identificarlas. Si dos "
        "jugadoras comparten dorsal, fíjate en el equipo o el color):\n"
        + "\n".join(lineas)
    )


def resolve_roster_names(texto: str, context: str) -> str:
    """Traduce «dorsal 7» / «el 7» al **nombre de pila** de la alineación.

    Reescribe el texto que ha producido la IA para que el relato hable de las
    jugadoras por su nombre. Nunca adivina: si el dorsal no está en la alineación,
    o hay dos jugadoras con ese dorsal y el texto no dice a qué equipo se refiere,
    la referencia se deja tal cual.
    """
    jugadores = parse_roster(context)
    if not jugadores or not texto:
        return texto

    por_dorsal: dict[int, list[RosterPlayer]] = {}
    for jugador in jugadores:
        por_dorsal.setdefault(jugador.dorsal, []).append(jugador)
    colores = _equipo_colores(context)
    equipos = tuple(colores)

    def elegir(dorsal: int, frase: str) -> str | None:
        candidatos = por_dorsal.get(dorsal)
        if not candidatos:
            return None
        bajo = _sin_acentos(frase)
        pistas_color = raices_color(frase)
        pistas_equipo = {e for e in equipos if e and e in bajo}
        if pistas_color or pistas_equipo:
            aciertos = [
                c
                for c in candidatos
                if _color_de_equipo(c.equipo_clave, colores) in pistas_color
                or any(
                    c.equipo_clave
                    and (
                        c.equipo_clave == clave
                        or c.equipo_clave.startswith(clave)
                        or clave.startswith(c.equipo_clave)
                    )
                    for clave in pistas_equipo
                )
            ]
            # Un solo acierto: es ella. Ninguno o varios: no se fuerza el nombre
            # (mejor el dorsal que un nombre equivocado).
            return aciertos[0].nombre if len(aciertos) == 1 else None
        return candidatos[0].nombre if len(candidatos) == 1 else None

    def sustituir(match: re.Match, frase: str) -> str:
        dorsal = int(match.group(1))
        if dorsal <= 0:
            return match.group(0)
        # «el 2ª parte» / «la 2 parte»: no es una jugadora.
        resto = frase[match.end() :]
        if resto[:1] in {"ª", "º", "°"}:
            return match.group(0)
        siguiente = re.match(r"\s*([A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+)", resto)
        if siguiente and _sin_acentos(siguiente.group(1)) in _PALABRAS_TIEMPO:
            return match.group(0)
        nombre = elegir(dorsal, frase)
        return nombre if nombre else match.group(0)

    resultado = texto
    for patron in _PATRONES_DORSAL:
        actual = resultado
        resultado = patron.sub(lambda m, frase=actual: sustituir(m, frase), actual)
    resultado = re.sub(r"\(\s*\)", "", resultado)
    resultado = re.sub(r"\s{2,}", " ", resultado)
    resultado = re.sub(r"\s+([,.;:!?])", r"\1", resultado)
    return resultado.strip()
