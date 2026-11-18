"""Cuaderno de análisis cinematográfico («Lapian Notes») — lógica pura.

Reúne, sin Qt y sin ffmpeg, todo lo que la pestaña **Cuaderno** necesita para
analizar una película en local:

- El **plan de muestreo** (un fotograma por segundo, con tope de seguridad).
- La **instrucciones y el esquema** que viajan en el paquete ZIP para una IA
  externa, y el **paquete** en sí (fotogramas + metadatos + esquema).
- El **modelo** del análisis (líneas narrativas, estructura, curva de emoción y
  segmentos) y su **parseo tolerante** del JSON devuelto por la IA.
- La **exportación a Markdown**.

Quien ejecuta (``ui.notes_widget``) extrae los fotogramas con ffmpeg, llama a los
motores y muestra el resultado; aquí solo se decide *qué* se pide y *cómo* se
interpreta y se compone.
"""

from __future__ import annotations

import json
import re
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path

from services import roster_params as rp
from services.atomic_io import atomic_write_json
from services.sport_params import (  # noqa: F401 - SPORT_PRESETS/sport_primer se reexportan
    INSTRUCTIONS_LABEL,
    SPORT_PRESETS,
    sport_primer,
)
from services.video_understanding import format_timestamp


class NotesError(Exception):
    """Error del cuaderno: JSON inválido, paquete incompleto o datos ilegibles."""


#: Fotogramas por segundo del paquete (lo que describe Lapian Notes: 1 fps).
DEFAULT_FPS = 1.0

#: Tope de seguridad de fotogramas del paquete (evita ZIPs gigantes). A 1 fps son
#: 30 min; se puede subir desde la UI para películas más largas.
DEFAULT_MAX_FRAMES = 1800

#: Tope duro admitido por la UI.
MAX_MAX_FRAMES = 10800  # 3 h a 1 fps

#: Número de fotogramas que se envían a la visión local (coste lineal en
#: inferencias), independiente del paquete completo de 1 fps. Por defecto se apunta
#: a la **máxima calidad**: 120 fotogramas repartidos por todo el vídeo.
DEFAULT_VISION_FRAMES = 120

#: Escala de la curva de emoción.
EMOTION_MIN = 0.0
EMOTION_MAX = 1.0

#: Paleta por defecto para las líneas narrativas sin color propio.
DEFAULT_LANE_COLORS = (
    "#4e79a7",
    "#f28e2b",
    "#e15759",
    "#76b7b2",
    "#59a14f",
    "#edc948",
    "#b07aa1",
    "#ff9da7",
)


def clamp_fps(value) -> float:
    """Acota los fotogramas por segundo a un valor razonable."""
    try:
        fps = float(value)
    except (TypeError, ValueError):
        return DEFAULT_FPS
    return min(max(0.1, fps), 60.0)


def clamp_max_frames(value) -> int:
    """Acota el tope de fotogramas del paquete."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        return DEFAULT_MAX_FRAMES
    return min(max(1, n), MAX_MAX_FRAMES)


def sample_frames(
    duration_s, fps: float = DEFAULT_FPS, max_frames: int = DEFAULT_MAX_FRAMES
) -> list[float]:
    """Instantes (segundos) a extraer: uno por fotograma del paquete.

    Con 1 fps y un tope de ``max_frames``, un vídeo más largo que ese tope se
    **reparte uniformemente por toda su duración** (el centro de cada tramo): antes
    se cortaba a los ``max_frames`` segundos y el resto del vídeo no se veía (por
    eso un vídeo de 54 min solo analizaba los primeros 30 y el modelo extrapolaba
    el final). Devuelve ``[]`` si la duración no es utilizable.
    """
    try:
        duration = float(duration_s)
    except (TypeError, ValueError):
        return []
    if duration <= 0:
        return []

    paso = 1.0 / clamp_fps(fps)
    limite = clamp_max_frames(max_frames)
    natural = int(duration / paso)
    if natural <= limite:
        instantes = [round(paso / 2.0 + i * paso, 3) for i in range(natural)]
        return instantes or [round(duration / 2.0, 3)]
    # Más fotogramas que el tope: se reparten por TODO el vídeo (no solo el inicio).
    ancho = duration / limite
    return [round(ancho * (i + 0.5), 3) for i in range(limite)]


def spread(items: list, count: int) -> list:
    """Reparte ``count`` elementos de ``items`` de forma uniforme (con extremos)."""
    if count <= 0:
        return []
    if len(items) <= count:
        return list(items)
    if count == 1:
        return [items[len(items) // 2]]
    paso = (len(items) - 1) / (count - 1)
    return [items[round(i * paso)] for i in range(count)]


# ---- Modelo del análisis -------------------------------------------

@dataclass(frozen=True)
class NarrativeEvent:
    t_s: float
    descripcion: str

    @property
    def label(self) -> str:
        return format_timestamp(self.t_s)


@dataclass(frozen=True)
class NarrativeLane:
    nombre: str
    color: str = ""
    eventos: tuple[NarrativeEvent, ...] = ()


@dataclass(frozen=True)
class StructureNode:
    titulo: str
    inicio_s: float | None = None
    fin_s: float | None = None
    hijos: tuple["StructureNode", ...] = ()


@dataclass(frozen=True)
class EmotionPoint:
    t_s: float
    intensidad: float
    etiqueta: str = ""


@dataclass(frozen=True)
class Segment:
    inicio_s: float
    fin_s: float
    titulo: str
    resumen: str = ""
    escenas: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        return f"{format_timestamp(self.inicio_s)}–{format_timestamp(self.fin_s)}"


@dataclass(frozen=True)
class Analysis:
    """Análisis completo de una película."""

    titulo: str = ""
    duracion_s: float | None = None
    lineas: tuple[NarrativeLane, ...] = ()
    estructura: StructureNode | None = None
    emocion: tuple[EmotionPoint, ...] = ()
    segmentos: tuple[Segment, ...] = ()
    notas: str = ""

    @property
    def is_empty(self) -> bool:
        return not (self.lineas or self.estructura or self.emocion or self.segmentos)


#: Palabras que delatan que un evento narra un gol (verificación de cobertura).
_GOAL_WORDS_RE = re.compile(
    r"\b(gol|marca|anota|empata|adelanta|ampl[íi]a|recorta|acorta|suma|"
    r"remata|convierte|transforma)\b",
    re.IGNORECASE,
)


def analysis_coverage_warnings(analysis: Analysis, total_goles: int) -> list[str]:
    """Avisos si el cuaderno se dejó goles del marcador sin cubrir.

    ``analysis`` es el análisis ya parseado; ``total_goles``, los goles que el
    marcador OCR declaró. Compara cuántos eventos mencionan un gol y avisa si
    faltan.
    """
    if not total_goles:
        return []
    narrados = sum(
        1
        for linea in analysis.lineas
        for evento in linea.eventos
        if _GOAL_WORDS_RE.search(evento.descripcion)
    )
    if narrados < total_goles:
        return [
            f"El cuaderno narra {narrados} goles de los {total_goles} que constan "
            "en el marcador OCR: faltan goles por cubrir."
        ]
    return []


# ---- Parseo tolerante ----------------------------------------------

def _as_float(value, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def _as_list(value) -> list:
    return value if isinstance(value, list) else []


def _clamp_t(value, limite: float | None = None) -> float:
    """Instante acotado a ``[0, limite]`` (o solo ``>= 0`` si no hay límite).

    La visión y el chat son libres de devolver cualquier segundo, pero el vídeo
    tiene una duración real: acotar evita que el cuaderno feche momentos que no
    existen (p. ej. 4835 s en un vídeo de 3126 s).
    """
    t = _as_float(value, 0.0) or 0.0
    if t < 0:
        t = 0.0
    if limite is not None and t > limite:
        t = float(limite)
    return t


def _parse_event(raw, context: str = "", limite: float | None = None) -> NarrativeEvent:
    raw = _as_dict(raw)
    return NarrativeEvent(
        t_s=_clamp_t(raw.get("t_s", raw.get("t", raw.get("inicio_s"))), limite),
        descripcion=rp.resolve_roster_names(
            str(raw.get("descripcion", raw.get("texto", ""))).strip(), context
        ),
    )


def _parse_lane(raw, index: int, context: str = "", limite: float | None = None) -> NarrativeLane:
    raw = _as_dict(raw)
    eventos = tuple(_parse_event(e, context, limite) for e in _as_list(raw.get("eventos")))
    color = str(raw.get("color", "")).strip()
    if not color:
        color = DEFAULT_LANE_COLORS[index % len(DEFAULT_LANE_COLORS)]
    return NarrativeLane(
        nombre=str(raw.get("nombre", raw.get("titulo", f"Línea {index + 1}"))).strip(),
        color=color,
        eventos=eventos,
    )


def _parse_node(raw, limite: float | None = None) -> StructureNode | None:
    raw = _as_dict(raw)
    if not raw:
        return None
    hijos = tuple(
        nodo
        for nodo in (_parse_node(h, limite) for h in _as_list(raw.get("hijos")))
        if nodo
    )
    inicio = _as_float(raw.get("inicio_s"))
    fin = _as_float(raw.get("fin_s"))
    return StructureNode(
        titulo=str(raw.get("titulo", raw.get("nombre", ""))).strip() or "(sin título)",
        inicio_s=_clamp_t(inicio, limite) if inicio is not None else None,
        fin_s=_clamp_t(fin, limite) if fin is not None else None,
        hijos=hijos,
    )


def _parse_emotion(raw, limite: float | None = None) -> EmotionPoint:
    raw = _as_dict(raw)
    valor = _as_float(raw.get("intensidad", raw.get("valor")), 0.0) or 0.0
    valor = min(max(valor, EMOTION_MIN), EMOTION_MAX)
    return EmotionPoint(
        t_s=_clamp_t(raw.get("t_s", raw.get("t")), limite),
        intensidad=valor,
        etiqueta=str(raw.get("etiqueta", "")).strip(),
    )


def _parse_segment(raw, context: str = "", limite: float | None = None) -> Segment:
    raw = _as_dict(raw)
    escenas = tuple(
        rp.resolve_roster_names(str(e).strip(), context)
        for e in _as_list(raw.get("escenas"))
        if str(e).strip()
    )
    return Segment(
        inicio_s=_clamp_t(raw.get("inicio_s"), limite),
        fin_s=_clamp_t(raw.get("fin_s"), limite),
        titulo=str(raw.get("titulo", "")).strip(),
        resumen=rp.resolve_roster_names(str(raw.get("resumen", "")).strip(), context),
        escenas=escenas,
    )


def parse_analysis(
    raw: dict | str, context: str = "", *, duration_s: float | None = None
) -> Analysis:
    """Convierte el JSON (dict o texto) en :class:`Analysis`, tolerante a huecos.

    ``context`` son los «Datos del partido» (equipos, alineación, marcador): con la
    alineación, las descripciones que nombran a una jugadora por su **dorsal** se
    reescriben con su **nombre de pila** (:mod:`services.roster_params`). Sin ella
    el análisis se devuelve tal cual.

    ``duration_s`` es la duración **real** del vídeo (la que midió ffmpeg). Si se
    pasa, manda sobre la que traiga el JSON: el modelo suele redondearla o
    equivocarla, y todos los ``t_s`` (líneas, emoción, estructura y segmentos) se
    acotan a ``[0, duration_s]`` para que el cuaderno nunca feche un momento que no
    existe.

    Lanza :class:`NotesError` solo si no hay ni dict ni JSON legible.
    """
    data = raw
    if isinstance(raw, str):
        data = extract_json(raw)
    if not isinstance(data, dict):
        raise NotesError("El análisis debe ser un objeto JSON.")
    if not any(k in data for k in ("lineas", "lineas_narrativas", "estructura", "emocion", "segmentos")):
        raise NotesError(
            "El JSON no tiene el formato esperado (líneas, estructura, emoción o segmentos)."
        )

    dur_json = _as_float(data.get("duracion_s", data.get("duracion")))
    limite = _as_float(duration_s) if duration_s is not None else dur_json

    lineas = tuple(
        _parse_lane(linea, i, context, limite)
        for i, linea in enumerate(_as_list(data.get("lineas_narrativas", data.get("lineas"))))
    )
    emocion = tuple(_parse_emotion(p, limite) for p in _as_list(data.get("emocion", data.get("emocion_puntos"))))
    emocion = tuple(sorted(emocion, key=lambda p: p.t_s))
    segmentos = tuple(_parse_segment(s, context, limite) for s in _as_list(data.get("segmentos")))
    segmentos = tuple(sorted(segmentos, key=lambda s: s.inicio_s))

    return Analysis(
        titulo=str(data.get("titulo", "")).strip(),
        duracion_s=limite if limite is not None else dur_json,
        lineas=lineas,
        estructura=_parse_node(data.get("estructura"), limite) if data.get("estructura") else None,
        emocion=emocion,
        segmentos=segmentos,
        notas=rp.resolve_roster_names(str(data.get("notas", "")).strip(), context),
    )


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str):
    """Extrae el primer objeto JSON de un texto (tolera vallas ```json```)."""
    if not isinstance(text, str):
        raise NotesError("Se esperaba texto con JSON.")
    candidato = text.strip()
    match = _FENCE_RE.search(candidato)
    if match:
        candidato = match.group(1).strip()
    try:
        return json.loads(candidato)
    except json.JSONDecodeError:
        pass
    # Segundo intento: recortar desde la primera '{' hasta la última '}'.
    inicio, fin = candidato.find("{"), candidato.rfind("}")
    if inicio != -1 and fin != -1 and fin > inicio:
        try:
            return json.loads(candidato[inicio : fin + 1])
        except json.JSONDecodeError as exc:
            raise NotesError(f"JSON inválido: {exc}") from exc
    raise NotesError("No se encontró ningún objeto JSON en la respuesta.")


# ---- Instrucciones y esquema (para la IA) --------------------------

ANALYSIS_SCHEMA: dict = {
    "titulo": "string",
    "duracion_s": "number",
    "lineas_narrativas": [
        {
            "nombre": "string",
            "color": "#RRGGBB",
            "eventos": [{"t_s": "number (segundos)", "descripcion": "string"}],
        }
    ],
    "estructura": {
        "titulo": "string",
        "inicio_s": "number",
        "fin_s": "number",
        "hijos": ["(mismo objeto, anidado)"],
    },
    "emocion": [{"t_s": "number", "intensidad": "number 0..1", "etiqueta": "string"}],
    "segmentos": [
        {
            "inicio_s": "number",
            "fin_s": "number",
            "titulo": "string",
            "resumen": "string",
            "escenas": ["string"],
        }
    ],
    "notas": "string",
}


def package_instructions(
    video_name: str = "",
    duration_s: float | None = None,
    *,
    context: str = "",
    sport: str = "",
) -> str:
    """Texto que acompaña al paquete para que la IA produzca el análisis."""
    cabecera = f"Vídeo: {video_name}\n" if video_name else ""
    dur = f"Duración: {format_timestamp(duration_s)}\n" if duration_s else ""
    contexto = _contexto_prompt(context, sport)
    extra = ("\n".join(contexto) + "\n") if contexto else ""
    return (
        "Eres un analista de contenido audiovisual (deporte o cine). Te doy un "
        "paquete con fotogramas (uno por segundo) y su marca de tiempo, y la "
        "transcripción si la hay.\n"
        f"{cabecera}{dur}{extra}"
        "Devuelve **solo** un JSON (sin texto alrededor) con esta forma:\n\n"
        f"{json.dumps(ANALYSIS_SCHEMA, ensure_ascii=False, indent=2)}\n\n"
        "Criterios:\n"
        "- 'lineas_narrativas': hilos argumentales en paralelo (story-lane), cada "
        "uno con sus momentos (t_s en segundos y una descripción breve).\n"
        "- 'estructura': árbol del relato (actos, subtramas) anidado en 'hijos'.\n"
        "- 'emocion': puntos (t_s, intensidad 0..1, etiqueta) que dibujan la curva "
        "del espectador.\n"
        "- 'segmentos': bloques con inicio/fin, título, resumen y escenas.\n"
        "Básate solo en lo que se ve/lee; si algo no está claro, dilo en 'notas'."
    )


def analysis_example() -> dict:
    """Ejemplo mínimo válido (lo incluye el paquete como plantilla)."""
    return {
        "titulo": "Título de la película",
        "duracion_s": 0,
        "lineas_narrativas": [
            {"nombre": "Trama principal", "color": "#4e79a7", "eventos": [{"t_s": 0, "descripcion": "…"}]}
        ],
        "estructura": {"titulo": "Película", "inicio_s": 0, "fin_s": 0, "hijos": []},
        "emocion": [{"t_s": 0, "intensidad": 0.5, "etiqueta": "calma"}],
        "segmentos": [{"inicio_s": 0, "fin_s": 0, "titulo": "…", "resumen": "…", "escenas": ["…"]}],
        "notas": "",
    }


def build_package(
    zip_path: str | Path,
    *,
    frames: list[tuple[float, Path]],
    video_name: str = "",
    duration_s: float | None = None,
    transcript: str = "",
    origin: str = "",
    context: str = "",
    sport: str = "",
) -> Path:
    """Escribe el ZIP de análisis (fotogramas + instrucciones + esquema).

    ``frames`` es una lista ``(t_s, ruta_png)``. Se incluye ``manifiesto.json``
    con la marca de tiempo de cada fotograma para que la IA pueda referenciarla.
    """
    destino = Path(zip_path)
    destino.parent.mkdir(parents=True, exist_ok=True)
    manifiesto = {
        "pelicula": video_name,
        "duracion_s": duration_s,
        "origen": origin,
        "fotogramas": [
            {"fichero": f"fotogramas/frame_{i:05d}.png", "t_s": t}
            for i, (t, _) in enumerate(frames)
        ],
    }
    with zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "instrucciones.txt",
            package_instructions(video_name, duration_s, context=context, sport=sport),
        )
        zf.writestr("esquema.json", json.dumps(ANALYSIS_SCHEMA, ensure_ascii=False, indent=2))
        zf.writestr("ejemplo.json", json.dumps(analysis_example(), ensure_ascii=False, indent=2))
        zf.writestr("manifiesto.json", json.dumps(manifiesto, ensure_ascii=False, indent=2))
        limpio = context_block(context)
        if limpio:
            zf.writestr("contexto.txt", limpio)
        primer = sport_primer(sport)
        if primer:
            zf.writestr("reglas_disciplina.txt", primer)
        if transcript.strip():
            zf.writestr("transcripcion.txt", transcript.strip())
        for i, (_, ruta) in enumerate(frames):
            ruta = Path(ruta)
            if ruta.is_file():
                zf.write(ruta, f"fotogramas/frame_{i:05d}.png")
    return destino


# ---- Informe reutilizable (lo consume la pestaña Crónica) ----------

#: Fichero con lo que el Cuaderno «vio»: descripciones por fotograma y audio.
#: Permite que la Crónica escriba la crónica **sin volver a mirar el vídeo**.
INFORME_FILENAME = "informe.json"


def informe_to_dict(
    *,
    video_path: str = "",
    video_name: str = "",
    duration_s: float | None = None,
    frames: list[tuple[float, str]] | None = None,
    transcript: str = "",
    sport: str = "",
    context: str = "",
) -> dict:
    """Empaqueta lo que la visión/Whisper han visto en un informe reutilizable."""
    nombre = video_name or (Path(str(video_path)).name if video_path else "")
    return {
        "video_path": str(video_path or ""),
        "video_name": nombre,
        "duracion_s": duration_s,
        "disciplina": sport or "",
        "contexto": context or "",
        "fotogramas": [
            {
                "t_s": float(t),
                "descripcion": rp.resolve_roster_names(str(texto).strip(), context),
            }
            for t, texto in (frames or [])
        ],
        "transcripcion": transcript or "",
    }


def save_informe(path: str | Path, data: dict) -> Path:
    """Escribe el informe (JSON) en ``path`` y devuelve su ruta."""
    destino = Path(path)
    atomic_write_json(destino, data)
    return destino


def load_informe(path: str | Path) -> dict:
    """Lee un informe del Cuaderno (``informe.json``); lanza ``NotesError`` si no vale."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise NotesError(f"No se pudo leer el informe: {exc}") from exc
    if not isinstance(data, dict) or "fotogramas" not in data:
        raise NotesError(
            "Ese fichero no es un informe del Cuaderno (falta el campo 'fotogramas')."
        )
    return data


def informe_frames(data: dict) -> list[tuple[float, str]]:
    """Descripciones ``(t_s, texto)`` de un informe (tolerante a huecos)."""
    return [
        (_as_float(f.get("t_s"), 0.0) or 0.0, str(f.get("descripcion", "")).strip())
        for f in _as_list(data.get("fotogramas"))
        if isinstance(f, dict)
    ]


def frame_annotation(descripcion: str, perception_summary: str = "") -> str:
    """Texto de un fotograma: descripción de la visión + datos de percepción.

    Los motores de percepción (YOLO/CLIP/OCR) aportan datos **estructurados** del
    mismo instante (objetos, etiquetas del vocabulario, texto en pantalla). Se
    guardan pegados a la descripción, así el cuaderno y la crónica (que lee este
    informe) cuentan con ambos.
    """
    partes: list[str] = []
    limpio = " ".join(str(descripcion or "").split())
    if limpio:
        partes.append(limpio)
    percepcion = " ".join(str(perception_summary or "").split())
    if percepcion:
        partes.append(percepcion)
    return " | ".join(partes)


# ---- Contraste contexto ↔ percepción --------------------------------
#: Colores canónicos de camiseta. El contraste usa ``roster_params.raices_color``,
#: que reconoce género y número («rojo» = «roja» = «rojas») para no fallar.
COLORES_CAMISETA: tuple[str, ...] = (
    "azul", "rojo", "verde", "blanco", "negro", "amarillo", "naranja",
    "gris", "morado", "violeta", "rosa", "celeste", "granate", "burdeos",
)

#: Tanteo «42-29», «42 --29», «42 : 29»… (tolera separadores repetidos y espacios).
_MARCADOR_RE = re.compile(r"(?<!\d)(\d{1,3})(?:\s*[-–:]\s*){1,2}(\d{1,3})(?!\d)")
_EQUIPOS_RE = re.compile(r"equipos?\s*[:\-]\s*(.+)", re.IGNORECASE)

#: Palabras que NO son nombre de equipo en la línea «Equipos:» (colores y adornos).
_PALABRAS_NO_EQUIPO = frozenset(
    {
        "azul", "azules", "rojo", "roja", "rojos", "rojas", "verde", "verdes",
        "blanco", "blanca", "blancos", "blancas", "negro", "negra", "negros",
        "negras", "amarillo", "amarilla", "amarillos", "amarillas", "naranja",
        "naranjas", "gris", "grises", "morado", "morada", "morados", "moradas",
        "violeta", "violetas", "rosa", "rosas", "celeste", "celestes", "granate",
        "granates", "burdeos", "grana", "granas", "turquesa", "marron",
        "camiseta", "camisetas", "equipacion", "equipaciones", "equipo", "equipos",
        "local", "visitante", "vs", "contra", "color", "colores",
        "de", "del", "el", "la", "los", "las", "y", "o",
    }
)

#: Palabras funcionales del castellano. Si la transcripción no trae ninguna, no es
#: habla en español (es ruido de grada: «APPLAUSE, WHISTLE…») y no se juzgan los
#: nombres de equipo: avisaría siempre y sería un falso positivo.
_PALABRAS_FUNCIONALES_ES = frozenset(
    {
        "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "que",
        "y", "o", "en", "por", "con", "para", "al", "es", "son", "se", "no", "lo",
        "su", "sus", "muy", "mas", "ya", "pero", "como", "cuando", "donde", "hay",
        "aqui", "alli", "todo", "todos", "tambien", "esto", "esta", "este",
    }
)


def _habla_en_espanol(transcripcion: str) -> bool:
    """¿La transcripción parece habla en español? (filtro de ruido de grada)."""
    palabras = re.findall(r"[a-zñ]+", _sin_acentos(transcripcion))
    return any(palabra in _PALABRAS_FUNCIONALES_ES for palabra in palabras)


def _sin_acentos(texto: str) -> str:
    """Minúsculas y sin acentos: para comparar nombres sin falsos negativos."""
    base = unicodedata.normalize("NFKD", str(texto or ""))
    return "".join(c for c in base if not unicodedata.combining(c)).lower()


def _frame_texts(data: dict) -> list[str]:
    """Textos por fotograma, sea un informe del Cuaderno o una percepción."""
    textos: list[str] = []
    for frame in _as_list(data.get("fotogramas")):
        if isinstance(frame, dict):
            textos.append(str(frame.get("descripcion") or frame.get("anotacion") or ""))
    return textos


def _nombres_de_equipo(context: str) -> list[str]:
    """Nombres de equipo de la línea «Equipos: …» (limpios de colores y adornos).

    La línea del usuario se escribe de mil maneras («de azul PAN MENESA MUSKIZ
    (local) vs de rojo BAIGENE…»); aquí se quitan colores, roles y adornos para que
    el nombre sea de verdad el del equipo y el contraste no avise por «camiseta
    azul» (que no es un equipo).
    """
    for linea in str(context or "").splitlines():
        match = _EQUIPOS_RE.search(linea)
        if not match:
            continue
        resto = re.sub(r"\([^)]*\)", " ", match.group(1))
        partes = re.split(
            r"\bvs\b|\bcontra\b|[,/;&]|\s-\s|\sy\s", resto, flags=re.IGNORECASE
        )
        nombres: list[str] = []
        for parte in partes:
            palabras = [
                palabra
                for palabra in parte.split()
                if _sin_acentos(palabra).strip(" .-") not in _PALABRAS_NO_EQUIPO
            ]
            limpio = " ".join(palabras).strip(" .-")
            if len(limpio) >= 4 and _sin_acentos(limpio) not in {
                _sin_acentos(nombre) for nombre in nombres
            }:
                nombres.append(limpio)
        return nombres
    return []


def contrast_context(context: str, data: dict) -> list[str]:
    """Avisos (no correcciones) si el contexto choca con lo que se ve/oye.

    Comprobaciones **deterministas** sobre el informe del Cuaderno o la percepción
    de Comprensión: marcador leído en pantalla, colores de camiseta y nombres de
    equipo en la transcripción. Cada aviso se redacta como «no lo he encontrado»,
    nunca como «está mal»: la ausencia de una señal no prueba un error. Nada se
    reescribe solo; el usuario decide.
    """
    avisos: list[str] = []
    context = str(context or "").strip()
    if not context:
        return avisos
    percepcion = _sin_acentos("\n".join(_frame_texts(data)))
    transcripcion = _sin_acentos(str(data.get("transcripcion", "")))
    if not percepcion and not transcripcion:
        return avisos  # sin datos que contrastar

    # Marcador: solo si el contexto lo declara y en pantalla se leyó ALGÚN tanteo
    # (si no se leyó ninguno, la ausencia no significa nada).
    if _hay_tanteo(percepcion):
        for linea in context.splitlines():
            if "marcador" not in _sin_acentos(linea):
                continue
            match = _MARCADOR_RE.search(linea)
            if not match:
                continue
            izquierda, derecha = match.group(1), match.group(2)
            marcador = f"{izquierda}-{derecha}"
            variantes = (
                marcador,
                marcador.replace("-", ":"),
                marcador.replace("-", " "),
                f"{izquierda}{derecha}",
            )
            # Además de las variantes exactas se acepta «42 …29» seguidos: el OCR a
            # veces separa los goles con espacios o guiones de más («42 29», «42 --29»).
            cerca = re.search(
                rf"(?<!\d){izquierda}\D{{0,4}}{derecha}(?!\d)", percepcion
            )
            if not cerca and not any(variante in percepcion for variante in variantes):
                avisos.append(
                    f"Marcador: tu contexto dice {marcador}, pero no aparece leído "
                    "en pantalla. Compruébalo (puede que la visión no lo captara)."
                )
            break

    # Colores de camiseta: solo si el vídeo describe ALGÚN color. Se comparan los
    # canónicos («rojo» = «roja» = «rojas»): el contexto real suele decir
    # «camisetas rojas» y no debe contarse como un color que falta.
    colores_contexto = rp.raices_color(context)
    colores_vistos = rp.raices_color(percepcion)
    if colores_vistos:
        for color in sorted(colores_contexto - colores_vistos):
            avisos.append(
                f"Color «{color}»: está en tu contexto, pero no aparece en las "
                "descripciones ni etiquetas del vídeo. Revísalo."
            )

    # Nombres de equipo: si NINGUNO asoma en la transcripción EN ESPAÑOL. Con ruido
    # de grada (silbatos, aplausos, inglés suelto) no se juzga: avisaría siempre.
    nombres = _nombres_de_equipo(context)
    if transcripcion and nombres and _habla_en_espanol(transcripcion):
        presentes = [nombre for nombre in nombres if _sin_acentos(nombre) in transcripcion]
        if not presentes:
            avisos.append(
                "Nombres de equipo: ninguno de "
                + ", ".join(nombres)
                + " aparece en la transcripción (si no se dijeron en el audio, es normal)."
            )
    return avisos


def _hay_tanteo(percepcion: str) -> bool:
    """¿Hay algún tanteo tipo ``28-25``/``28:25`` en el texto de percepción?"""
    return bool(_MARCADOR_RE.search(str(percepcion or "")))


# ---- Exportación a Markdown ----------------------------------------

def _node_lines(node: StructureNode, depth: int = 0) -> list[str]:
    rango = ""
    if node.inicio_s is not None and node.fin_s is not None:
        rango = f" ({format_timestamp(node.inicio_s)}–{format_timestamp(node.fin_s)})"
    lineas = [f"{'  ' * depth}- {node.titulo}{rango}"]
    for hijo in node.hijos:
        lineas.extend(_node_lines(hijo, depth + 1))
    return lineas


# ---- Prompts -------------------------------------------------------
#: Prompt por fotograma para la visión local (Ollama). A propósito NO incluye el
#: contexto del usuario (ver :func:`vision_frame_prompt`): si se le pasa el
#: marcador final a cada fotograma, el modelo lo repite en todo el vídeo.
VISION_FRAME_PROMPT = (
    "Eres un anotador de un partido. Describe este fotograma en 1-2 frases, en "
    "español, y EMPIEZA indicando el ESTADO: «JUGANDO», «SAQUE», «GOL», "
    "«PARADA», «CELEBRACIÓN», «TIEMPO MUERTO», «DESCANSO», «PREVIO/SALUDO» o "
    "«FINAL». "
    "Di qué se ve y qué está pasando (quién tiene el balón —ese equipo es el que "
    "ATACA y el otro DEFIENDE—, hacia dónde va la jugada, dónde están las "
    "jugadoras). Si es un lanzamiento, di si acaba en GOL o en PARADA de la portera "
    "(una parada es cuando la portera detiene el balón antes de que entre). Si "
    "distingues el equipo por el color de la camiseta o el DORSAL de quien tiene "
    "el balón (y, en un lanzamiento o parada, el dorsal de quien lanza o de la "
    "portera), dilo en esa misma frase. Si NO lo distingues, no lo menciones: no "
    "escribas frases como «no se ve el dorsal» ni rellenes con evasivas. Usa el "
    "vocabulario del deporte (portería, área, saque, 7 metros). NO escribas el "
    "marcador ni el tiempo del reloj (de eso se encarga el OCR). El ESTADO es el "
    "de ESTE instante: usa «PREVIO/SALUDO» solo antes del saque inicial o al "
    "saludar, y «DESCANSO»/«FINAL» solo cuando el juego esté parado por eso; si el "
    "balón está en juego, NO digas «PREVIO/SALUDO» (usa «JUGANDO»). No inventes "
    "nombres, dorsales ni acciones que no se vean."
)

# Las reglas/vocabulario por disciplina viven en ``services.sport_params`` (módulo
# neutral) para compartirlas con Comprensión y Crónica sin ciclos; aquí se
# reexportan (``nts.SPORT_PRESETS`` / ``nts.sport_primer``) por compatibilidad.

#: Etiqueta y ejemplo del contexto que aporta el usuario (equipos, marcador...).
CONTEXT_LABEL = "DATOS DEL PARTIDO O DE LA GRABACIÓN (los aporta el usuario y son correctos)"
CONTEXT_EXAMPLE = (
    "Fecha: 9/10/2026 a las 20:30\n"
    "Lugar: Polideportivo de Muskiz\n"
    "Equipos: PAN MENESA MUSKIZ camiseta azul (local) vs BAIGENE CORAZONISTAS "
    "camisetas rojas (visitante).\n"
    "Alineación PAN MENESA MUSKIZ: 2 ARANE PEREZ HERNANDEZ, 5 IRIA MORAGA GARCIA, "
    "6 JUGATX FERNANDEZ GONZALEZ, 8 ALAITZ SAN MARTIN HERVELLA...\n"
    "Alineación BAIGENE CORAZONISTAS: 3 AMANDA OTERO BENITO, 8 PAULA GARCIA "
    "VALENCIA, 9 AITANA ESTEBAN GARCIA, 10 JULENE RANEA SIERRA...\n"
    "Marcador final: 42-29\n"
    "Nota: faltas del segundo tiempo del minuto 30 al 40, por error de cámara.\n"
    "Nota: la 31 (OSKIA ELCARTE CORTES) lleva camiseta roja con una manga azul "
    "oscuro y el dorsal 1 pegado con esparadrapo; puede caerse."
)


def context_block(context: str) -> str:
    """Normaliza los datos del usuario: una línea por dato, sin huecos ni sobras."""
    lineas = [" ".join(linea.split()) for linea in str(context or "").splitlines()]
    return "\n".join(linea for linea in lineas if linea)


def _contexto_prompt(context: str, sport: str) -> list[str]:
    """Bloques de prompt con las reglas de la disciplina y los datos del usuario.

    Si el contexto trae la **alineación** (dorsal + nombre), se añade además el
    bloque que traduce cada dorsal a su **nombre de pila**: es lo que hace que el
    cuaderno llame a las jugadoras por su nombre en vez de por «dorsal 7».
    """
    bloques: list[str] = []
    primer = sport_primer(sport)
    if primer:
        bloques.append(primer)
    limpio = context_block(context)
    if limpio:
        bloques.append(f"{CONTEXT_LABEL}:\n{limpio}")
    roster = rp.roster_block(context)
    if roster:
        bloques.append(roster)
    return bloques


def vision_frame_prompt(
    sport: str = "", instructions: str = "", context: str = ""
) -> str:
    """Prompt por fotograma: la disciplina y el mapa color→equipo.

    A propósito **no** se inyecta aquí el contexto del usuario (marcador, alineación):
    al pasarle «Marcador final: 42-29» a cada fotograma, el modelo lo repetía en dos
    tercios de las anotaciones, incluso en el minuto 1. El contexto se reserva para
    :func:`analysis_prompt`, que compone el relato, y el marcador/reloj reales los
    lee el OCR.

    Sí se pasa el **mapa «color → equipo (LOCAL/VISITANTE)»** de ``context``: la
    visión sólo ve colores y sin el mapa confunde ataque y defensa (decía «el equipo
    rojo ataca» sin saber quién era). ``instructions`` se mantiene por compatibilidad,
    pero las instrucciones del usuario describen el RELATO: si se pasan por fotograma,
    el modelo intenta «presentar la alineación» en todos y alucina listas enormes.
    """
    partes = [VISION_FRAME_PROMPT]
    primer = sport_primer(sport)
    if primer:
        partes.append(primer)
    equipos = rp.team_color_summary(context)
    if equipos:
        partes.append(
            "EQUIPOS SEGÚN EL COLOR DE CAMISETA: "
            f"{equipos}. Ese mapa es correcto: NO cambies qué color es LOCAL y qué "
            "color es VISITANTE. El que TIENE EL BALÓN ataca; el otro defiende. "
            "Nómbralos por su equipo, no por el color."
        )
    pedido = context_block(instructions)
    if pedido:
        partes.append(f"{INSTRUCTIONS_LABEL}:\n{pedido}")
    return "\n\n".join(partes)


#: Presupuesto (caracteres) del bloque de anotaciones que recibe el chat para
#: componer el cuaderno. Con 120 fotogramas, meterlos enteros no cabía en
#: ``num_ctx`` y el modelo solo usaba el final (el análisis empezaba en 32:10).
MAX_ANOTACIONES_CHARS = 9000

#: Tope de la transcripción que entra en el prompt del cuaderno: la de un partido
#: de 50 min son decenas de miles de caracteres y llenaba la ventana («se agotó el
#: espacio de la respuesta»). El informe sigue guardándola entera; solo se acota al
#: prompt.
MAX_TRANSCRIPT_CHARS = 4000


def clamp_transcript(transcript: str, *, limit: int = MAX_TRANSCRIPT_CHARS) -> str:
    """Acota la transcripción que entra en el prompt (avisa del corte con » […]»)."""
    texto = str(transcript or "")
    if len(texto) <= limit:
        return texto
    corte = texto[:limit]
    if " " in corte:
        corte = corte[: corte.rfind(" ")]
    return corte.rstrip() + " […]"


#: Largo mínimo de cada línea de anotación. Con muchos fotogramas (p. ej. 500) el
#: presupuesto no da para todos: antes se recortaba cada línea a 30 caracteres
#: (ilegible) y aun así el bloque se pasaba del tope. Ahora se **muestrea**: se
#: conservan la primera y la última anotación y se reparten las del medio, con
#: líneas legibles.
MIN_ANOTACION_LINEA = 100


def _anotaciones_prompt(
    frame_notes: list[tuple[float, str]], tope: int = MAX_ANOTACIONES_CHARS
) -> str:
    """Bloque de anotaciones acotado: una línea por instante, descripción recortada.

    ``tope`` es el presupuesto en caracteres. Si hay más fotogramas que líneas
    caben, se muestrean (primera, última y reparto uniforme del medio) en vez de
    dejar cada descripción en 30 caracteres. ``tope`` permite además pedir un
    bloque más corto para los reintentos.
    """
    notas = [
        (float(t), " ".join(str(texto).split()))
        for t, texto in frame_notes
        if str(texto).strip()
    ]
    if not notas:
        return ""
    tope = max(1000, int(tope))
    max_lineas = max(5, tope // MIN_ANOTACION_LINEA)
    if len(notas) > max_lineas:
        notas = spread(notas, max_lineas)
    por_linea = max(60, tope // len(notas) - 14)
    lineas = [f"[t={format_timestamp(t)}] {texto[:por_linea]}" for t, texto in notas]
    return "\n".join(lineas)


def analysis_prompt(
    video_name: str,
    duration_s: float | None,
    frame_notes: list[tuple[float, str]],
    transcript: str = "",
    *,
    context: str = "",
    sport: str = "",
    marcador: str = "",
    juego_inicio_s: float | None = None,
    descanso_s: float | None = None,
    max_annotations: int = MAX_ANOTACIONES_CHARS,
    max_transcript: int = MAX_TRANSCRIPT_CHARS,
    instructions: str = "",
) -> str:
    """Prompt con el que el chat local compone el JSON del cuaderno.

    ``marcador`` es el marcador **real** leído por OCR a lo largo del vídeo (una
    línea por cambio, con su minuto). Es dato duro, no contexto del usuario: se le
    pasa al modelo para que sitúe los goles, sin volver a inventarlos.

    ``juego_inicio_s`` es el instante en que el **reloj empieza a correr** (lo
    detecta :func:`services.scoreboard.game_start`): ahí arranca el partido. Es
    dato duro y evita que el modelo feche el inicio del juego a ojo.
    ``max_annotations``/``max_transcript`` acotan los bloques grandes: el motor
    reintenta con valores menores cuando el modelo se queda sin espacio.
    """
    cabecera = f"Vídeo: {video_name}\n" if video_name else ""
    dur = f"Duración: {format_timestamp(duration_s)}\n" if duration_s else ""
    # Las anotaciones vienen de la visión, que sólo ve dorsales y colores: si hay
    # alineación, se traducen los dorsales a nombres de pila ANTES de dárselas al
    # modelo, para que éste ya lea «Marc» y no «dorsal 7».
    frame_notes = [
        (t, rp.resolve_roster_names(texto, context)) for t, texto in frame_notes
    ]
    anotaciones = _anotaciones_prompt(frame_notes, max_annotations)
    pedido = context_block(instructions)
    extra = [f"{INSTRUCTIONS_LABEL}:\n{pedido}"] if pedido else []
    bloques = [
        "Eres un analista de contenido audiovisual (deporte o cine). A partir de las "
        "anotaciones de fotogramas (y la transcripción, si la hay) reconstruye el "
        "análisis del relato. Identifica la disciplina y usa su vocabulario y sus "
        "reglas; NO inventes otras disciplinas ni acciones que las anotaciones no "
        "muestren. Las anotaciones cubren TODO el vídeo (desde el segundo 0): no "
        "digas que faltan datos del principio.",
        f"{cabecera}{dur}",
        *_contexto_prompt(context, sport),
        *extra,
    ]
    if marcador.strip():
        bloques += [
            "## Marcador real leído por OCR (por minuto)",
            "Este marcador lo leyó el OCR de la pantalla: es correcto y son los "
            "únicos goles que constan (izquierda/derecha según el marcador de "
            "pantalla). No añadas goles que no estén aquí.",
            marcador.strip(),
            "VERIFICA: crea UN evento por CADA gol de este marcador (sin agruparlos "
            "ni saltarte ninguno), y un evento por cada PARADA, 7 METROS, EXCLUSIÓN "
            "o TIEMPO MUERTO que muestren las anotaciones.",
        ]
    if juego_inicio_s is not None:
        bloques += [
            "## Inicio del juego (reloj del marcador)",
            "El reloj de la pantalla EMPIEZA A CORRER en "
            f"{format_timestamp(juego_inicio_s)}: ahí ARRANCA el partido, no antes. "
            "Todo lo anterior es calentamiento/saludo (PREVIO/SALUDO). En cuanto el "
            "reloj avanza el juego está EN MARCHA: usa JUGANDO/SAQUE/GOL/FALTA, "
            "NUNCA «PREVIO/SALUDO». Sitúa el fin de la fase previa y el inicio del "
            "partido en ese instante.",
        ]
    if descanso_s is not None:
        bloques += [
            "## Descanso (reloj del marcador)",
            "El reloj se detiene/retrocede en "
            f"{format_timestamp(descanso_s)}: ahí empieza el DESCANSO/entretiempo. "
            "Lo que pase entre ese instante y la reanudación es DESCANSO (usa ese "
            "estado), NO juego. No narres jugadas ni goles durante el descanso.",
        ]
    bloques += [
        "## Anotaciones por fotograma",
        anotaciones or "(sin anotaciones)",
    ]
    if transcript.strip():
        bloques += [
            "## Transcripción del audio",
            clamp_transcript(transcript, limit=max_transcript),
        ]
    bloques += [
        "Devuelve **solo** un JSON (sin texto alrededor) con esta forma:",
        json.dumps(ANALYSIS_SCHEMA, ensure_ascii=False, indent=2),
        "DISTINGUE JUEGO de NO-JUEGO: cada anotación empieza por su estado "
        "(JUGANDO/SAQUE/GOL/CELEBRACIÓN/TIEMPO MUERTO/DESCANSO/PREVIO/FINAL). Cuenta "
        "jugadas solo en los momentos de juego; los descansos, saludos y tiempos "
        "muertos no son jugadas. Cuando quieras decir quién marca, llama a la "
        "jugadora por su NOMBRE DE PILA: míralo en la alineación de los datos del "
        "usuario cruzando el dorsal (o el color) que aparezca en las anotaciones; "
        "no la llames por su dorsal ni por el color de la camiseta. Si su dorsal no "
        "está en la alineación, NO le pongas nombre.",
        "IDENTIDAD: si en una anotación se identifica a una jugadora por su dorsal "
        "(y su nombre de la alineación), reutiliza ese nombre cuando las anotaciones "
        "siguientes describan a la misma jugadora aunque no se le lea el dorsal; no "
        "le pongas nombre a quien no puedas identificar.",
        "Cada 't_s' debe ser EXACTAMENTE el segundo de una anotación [t=MM:SS] de "
        "abajo (el momento real del fotograma); no inventes ni interpoles tiempos, "
        "y no uses instantes que no aparezcan en las anotaciones. Así los minutos "
        "del Cuaderno coinciden con los de Comprensión y Crónica.",
        "Básate solo en los datos. Si las anotaciones no cubren todo el vídeo, NO "
        "extrapoles el resto: indica en 'notas' hasta qué minuto tienes datos. Si "
        "algo no está claro, dilo también en 'notas'.",
        "Limita el TAMAÑO de la respuesta (si no, se corta): como máximo 6 líneas "
        "narrativas con 12 eventos cada una, 6 segmentos con hasta 6 escenas, 12 "
        "puntos de emoción y descripciones de una frase corta (máx. 12 palabras). "
        "Y NO te quedes en generalidades: entre 20 y 60 eventos en total (los "
        "momentos de juego de las anotaciones), 3-6 segmentos con título y resumen "
        "CONCRETOS (qué se juega, no «desarrollo táctico»), una estructura de DOS "
        "niveles (partes del partido con sus bloques) y un punto de emoción por "
        "momento relevante. No repitas la misma idea en dos eventos.",
    ]
    return "\n\n".join(bloques)


# ---- Exportación a Markdown ----------------------------------------

def build_markdown(analysis: Analysis) -> str:
    """Cuaderno en Markdown (estructura, emoción, líneas y segmentos)."""
    titulo = analysis.titulo or "Cuaderno de análisis"
    partes: list[str] = [f"# {titulo}", ""]
    if analysis.duracion_s:
        partes += [f"**Duración:** {format_timestamp(analysis.duracion_s)}", ""]

    if analysis.estructura is not None:
        partes += ["## Estructura narrativa", ""]
        partes += _node_lines(analysis.estructura)
        partes.append("")

    if analysis.emocion:
        partes += ["## Curva de emoción", "", "| Momento | Intensidad | Etiqueta |", "| --- | --- | --- |"]
        for p in analysis.emocion:
            partes.append(f"| {format_timestamp(p.t_s)} | {p.intensidad:.2f} | {p.etiqueta} |")
        partes.append("")

    if analysis.lineas:
        partes += ["## Líneas narrativas", ""]
        for linea in analysis.lineas:
            partes.append(f"### {linea.nombre}")
            for ev in linea.eventos:
                partes.append(f"- **{ev.label}** — {ev.descripcion}")
            partes.append("")

    if analysis.segmentos:
        partes += ["## Segmentos y escenas", ""]
        for seg in analysis.segmentos:
            partes.append(f"### {seg.label} · {seg.titulo or '(sin título)'}")
            if seg.resumen:
                partes.append(seg.resumen)
            for escena in seg.escenas:
                partes.append(f"- {escena}")
            partes.append("")

    if analysis.notas:
        partes += ["## Notas", "", analysis.notas, ""]

    return "\n".join(partes).rstrip() + "\n"
