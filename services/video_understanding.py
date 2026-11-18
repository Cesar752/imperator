"""Comprensión de vídeo: plan de muestreo, prompts y montaje del informe.

Lógica pura (sin Qt y sin ffmpeg): aquí se decide *qué* fotogramas se analizan,
*qué* se le pide al LLM y *cómo* se compone el informe final. Quien ejecuta
(``ui.understanding_widget``) extrae los fotogramas y llama a los motores.

Decisiones de diseño:

- **Muestreo por segundos, no por milisegundos.** Un fotograma «cada N ms» es
  inviable (60 fps = 60 inferencias por segundo). Se toma uno cada
  :data:`DEFAULT_EVERY_S` segundos, en el **centro** de cada intervalo: así no se
  cae siempre en el primer fotograma, que suele ser un fundido o un negro.
- **Tope de fotogramas.** Si el vídeo es largo, se reparten uniformemente
  :data:`DEFAULT_MAX_FRAMES` en vez de analizar cientos: el coste es lineal en
  inferencias y el resumen no mejora por tener más de ~24 anotaciones.
- **Sin señales inventadas.** Si no hay motor de objeto/etiqueta/OCR disponible,
  el informe lo dice; y si no hay transcripción, también (no se rellena el
  hueco con texto imaginado).
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from services.sport_params import INSTRUCTIONS_LABEL, sport_primer

#: Separación por defecto entre fotogramas analizados (segundos). Por defecto se
#: apunta a la **máxima calidad**: pocos segundos entre fotogramas y el tope de
#: fotogramas al máximo (se reparten por todo el vídeo si no caben).
DEFAULT_EVERY_S = 2.0

#: Separación mínima admitida (por debajo, el coste se dispara sin aportar).
MIN_EVERY_S = 1.0

#: Número de fotogramas analizados como máximo (y por defecto). El tope por defecto
#: es el máximo: para un vídeo largo se reparten uniformemente (un fotograma cada
#: pocos segundos), que es la opción de más calidad.
DEFAULT_MAX_FRAMES = 300

#: Tope duro del selector (evita que la UI permita un análisis inviable).
MAX_MAX_FRAMES = 600

#: Cadencia del muestreo **denso del marcador**. El marcador cambia justo al gol y
#: es la fuente más fiable de "minuto a minuto": se lee por OCR con esta cadencia
#: (mucho más fina que la visión, que describe la jugada) para captar CADA gol y su
#: instante real, en vez de depender del fotograma de visión más cercano.
SCOREBOARD_EVERY_S = 2.0

#: Tope de lecturas densas del marcador (red de seguridad para vídeos muy largos).
SCOREBOARD_MAX_READINGS = 2000

#: Filtro ``crop`` de ffmpeg para leer el marcador: la franja superior (30 %), donde
#: suele estar el marcador/reloj. Recortar reduce el área y acelera el OCR.
SCOREBOARD_CROP = "crop=in_w:in_h*0.30:0:0"

#: Extensiones de vídeo que ffmpeg lee y que la app ofrece en sus diálogos. Incluye
#: los formatos de cámara (``.mts``/``.m2ts`` de AVCHD): si no están aquí, el
#: fichero existe pero no aparece en el selector y parece que la app no lo soporta.
VIDEO_EXTENSIONS: tuple[str, ...] = (
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".mts", ".m2ts",
)

#: Filtro de diálogo de vídeo, con «Todos» detrás para formatos raros.
VIDEO_FILTER = "Vídeo (" + " ".join(f"*{ext}" for ext in VIDEO_EXTENSIONS) + ")"
VIDEO_FILTER_ALL = f"{VIDEO_FILTER};;Todos (*)"

#: Vocabulario sugerido para CLIP en un partido de balonmano (la pestaña lo usa
#: como valor inicial y editable). CLIP **no** describe: puntúa una lista cerrada,
#: así que la calidad depende de cómo se elija.
#:
#: Criterios (por qué esta lista):
#: - **Concreta y visible**: lo que un fotograma puede mostrar de un vistazo
#:   («lanzamiento a portería», «público en la grada»), no conceptos
#:   («intensidad», «partido»).
#: - **Separables entre sí**: dos etiquetas que suelen ir juntas confunden a CLIP.
#:   Cada etiqueta debe describir un instante distinto.
#: - **Cortas y en español, con el vocabulario del deporte** (portero, siete
#:   metros, tiempo muerto): es lo que el usuario reconoce y edita.
#: - **Sin nombres ni colores**: CLIP no lee dorsales ni distingue «azul» con luz
#:   de pabellón; el nombre y el color los pone la alineación (roster_params).
#:
#: Nota: CLIP se entrenó sobre todo en inglés; si en tu pabellón sale ruido,
#: prueba la versión inglesa (p. ej. «handball players on court, ball, goal,
#: goalkeeper, crowd, scoreboard, referee, coach, celebration, timeout»).
SPORT_CLIP_LABELS: tuple[str, ...] = (
    "jugadores en la pista",
    "balón",
    "portería",
    "portero",
    "lanzamiento a portería",
    "gol",
    "siete metros",
    "público en la grada",
    "marcador",
    "árbitro",
    "entrenador",
    "celebración",
    "saludo o fila inicial",
    "tiempo muerto",
    "banquillo",
    "pista cubierta",
)


def clip_labels_text(labels: tuple[str, ...] = SPORT_CLIP_LABELS) -> str:
    """Vocabulario de CLIP como texto (coma entre etiquetas)."""
    return ", ".join(labels)


def format_timestamp(seconds: float) -> str:
    """``95.2`` → ``01:35`` (y ``3725`` → ``1:02:05``)."""
    total = max(0, int(seconds))
    horas, resto = divmod(total, 3600)
    minutos, segundos = divmod(resto, 60)
    if horas:
        return f"{horas}:{minutos:02d}:{segundos:02d}"
    return f"{minutos:02d}:{segundos:02d}"


#: Marca de tiempo ``MM:SS`` o ``H:MM:SS`` (con o sin corchetes alrededor).
_TIMESTAMP_RE = re.compile(r"(?<!\d)(\d{1,3}):([0-5]?\d)(?::([0-5]?\d))?(?!\d)")


def parse_timestamp(texto: object) -> float | None:
    """Marca de tiempo de un texto → segundos (inversa de :func:`format_timestamp`).

    Acepta ``12:34`` (MM:SS) y ``1:02:03`` (H:MM:SS), con o sin corchetes/etiquetas
    («[00:12]», «t=00:12»). Devuelve ``None`` si no hay ninguna marca: así no se
    confunde un texto cualquiera con un instante.
    """
    match = _TIMESTAMP_RE.search(str(texto or ""))
    if not match:
        return None
    primero, segundo, tercero = match.group(1), match.group(2), match.group(3)
    if tercero is None:
        return float(int(primero) * 60 + int(segundo))
    return float(int(primero) * 3600 + int(segundo) * 60 + int(tercero))


def clamp_every(seconds) -> float:
    """Acota la separación al rango admitido (tolera valores basura)."""
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return DEFAULT_EVERY_S
    return max(MIN_EVERY_S, value)


def clamp_max_frames(value) -> int:
    """Acota el número de fotogramas al rango admitido."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return DEFAULT_MAX_FRAMES
    return min(max(1, number), MAX_MAX_FRAMES)


def sample_timestamps(
    duration_s,
    every_s: float = DEFAULT_EVERY_S,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> list[float]:
    """Instantes (segundos) en los que tomar fotograma.

    Devuelve ``[]`` si la duración no es utilizable. Si ``every_s`` produce más
    de ``max_frames`` instantes, se reparten uniformemente ``max_frames``.
    """
    try:
        duration = float(duration_s)
    except (TypeError, ValueError):
        return []
    if duration <= 0:
        return []

    step = clamp_every(every_s)
    limite = clamp_max_frames(max_frames)

    if duration < step:
        # Vídeo más corto que un paso: un único fotograma, en su centro real
        # (no en el centro del intervalo de muestreo, que se saldría del vídeo).
        return [round(duration / 2.0, 3)]

    instantes: list[float] = []
    t = step / 2.0
    while t < duration:
        instantes.append(round(t, 3))
        t += step
    if not instantes:  # red de seguridad (no debería alcanzarse)
        return [round(duration / 2.0, 3)]

    if len(instantes) > limite:
        instantes = _spread(instantes, limite)
    return instantes


def _spread(items: list[float], count: int) -> list[float]:
    """Reparte ``count`` elementos de ``items`` de forma uniforme (incluye extremos)."""
    if count <= 1:
        return [items[len(items) // 2]]
    paso = (len(items) - 1) / (count - 1)
    return [items[round(i * paso)] for i in range(count)]


def clamp_scoreboard_every(seconds) -> float:
    """Acota la cadencia del muestreo del marcador (tolera valores basura)."""
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return SCOREBOARD_EVERY_S
    return max(0.5, value)


def scoreboard_timestamps(
    duration_s,
    every_s: float = SCOREBOARD_EVERY_S,
    max_readings: int = SCOREBOARD_MAX_READINGS,
) -> list[float]:
    """Instantes densos en los que leer el marcador por OCR.

    A diferencia de :func:`sample_timestamps`, aquí NO se reparte a un tope de
    fotogramas pequeño: el marcador hay que leerlo con la cadencia pedida (cada
    ~2 s) para no saltarse ningún gol. ``max_readings`` (alto) es solo una red de
    seguridad para vídeos muy largos.
    """
    try:
        duration = float(duration_s)
    except (TypeError, ValueError):
        return []
    if duration <= 0:
        return []
    step = clamp_scoreboard_every(every_s)
    instantes: list[float] = []
    t = step / 2.0
    while t < duration:
        instantes.append(round(t, 3))
        t += step
    if not instantes:
        return [round(duration / 2.0, 3)]
    if len(instantes) > max_readings:
        instantes = _spread(instantes, max_readings)
    return instantes


def srt_to_text(raw: str) -> str:
    """Convierte un SRT a texto plano (sin índices ni marcas de tiempo)."""
    lineas: list[str] = []
    for linea in (raw or "").splitlines():
        texto = linea.strip()
        if not texto or texto.isdigit() or "-->" in texto:
            continue
        lineas.append(texto)
    return "\n".join(lineas)


def srt_timestamp(seconds: float) -> str:
    """Segundos → marca de tiempo SRT (``95.2`` → ``00:01:35,200``)."""
    milisegundos = max(0, int(round(float(seconds) * 1000)))
    horas, resto = divmod(milisegundos, 3_600_000)
    minutos, resto = divmod(resto, 60_000)
    segs, ms = divmod(resto, 1000)
    return f"{horas:02d}:{minutos:02d}:{segs:02d},{ms:03d}"


def build_srt(entries: Sequence[tuple[float, float, str]]) -> str:
    """SRT a partir de ``(inicio, fin, texto)``.

    Es el *escritor* que faltaba: el proyecto sabía leer SRT (:func:`srt_to_text`,
    ``VideoAssembler._parse_srt``) pero no generarlos. Lo usan los rótulos de Vídeo
    IA y la crónica, que ya conocen sus propios tiempos (no hace falta Whisper
    para subtitular texto que uno mismo escribe).
    """
    bloques: list[str] = []
    for inicio, fin, texto in entries:
        limpio = " ".join((texto or "").split())
        if not limpio:
            # Se numera al vuelo: una entrada sin texto no debe dejar un hueco en
            # la numeración (el bloque «útil» siguiente es el 1, no el 2).
            continue
        bloques.append(
            f"{len(bloques) + 1}\n{srt_timestamp(inicio)} --> {srt_timestamp(fin)}\n"
            f"{limpio}"
        )
    return "\n\n".join(bloques) + ("\n" if bloques else "")


@dataclass(frozen=True)
class UnderstandingPlan:
    """Qué se analiza de un vídeo y con qué opciones."""

    every_s: float = DEFAULT_EVERY_S
    max_frames: int = DEFAULT_MAX_FRAMES
    do_objects: bool = True
    do_labels: bool = False
    do_ocr: bool = False
    transcribe: bool = True
    #: Disciplina (reglas/vocabulario). Sin ella el resumen confunde deportes.
    sport: str = ""
    #: Idioma del resumen (``None`` = el del vídeo, que fija el LLM).
    language: str | None = None

    def normalized(self) -> "UnderstandingPlan":
        """Devuelve el plan con los valores acotados (lo que aplica el worker)."""
        return UnderstandingPlan(
            every_s=clamp_every(self.every_s),
            max_frames=clamp_max_frames(self.max_frames),
            do_objects=bool(self.do_objects),
            do_labels=bool(self.do_labels),
            do_ocr=bool(self.do_ocr),
            transcribe=bool(self.transcribe),
            sport=str(self.sport or ""),
            language=self.language,
        )

    @property
    def wants_perception(self) -> bool:
        """¿Pide algún análisis de imagen (objetos, etiquetas u OCR)?"""
        return self.do_objects or self.do_labels or self.do_ocr

    def describe(self) -> str:
        """Frase legible de la configuración efectiva."""
        partes = [f"un fotograma cada {clamp_every(self.every_s):g} s"]
        partes.append(f"máx. {clamp_max_frames(self.max_frames)}")
        señales = []
        if self.do_objects:
            señales.append("objetos")
        if self.do_labels:
            señales.append("etiquetas")
        if self.do_ocr:
            señales.append("texto en pantalla")
        if self.transcribe:
            señales.append("transcripción")
        if señales:
            partes.append("con " + ", ".join(señales))
        return " · ".join(partes)


@dataclass(frozen=True)
class FrameReport:
    """Anotación de un fotograma analizado."""

    timestamp: float
    summary: str = ""

    @property
    def label(self) -> str:
        return format_timestamp(self.timestamp)


SUMMARY_HEADING = "=== Resumen ==="
FRAMES_HEADING = "=== Anotaciones por fotograma ==="
TRANSCRIPT_HEADING = "=== Transcripción del audio ==="

#: Avisos honestos: se emiten en vez de rellenar el hueco con texto inventado.
NO_PERCEPTION_NOTICE = (
    "Aviso: no hay ningún motor de percepción activo (YOLO/CLIP/OCR); no se "
    "anotó ningún fotograma."
)
NO_TRANSCRIPT_NOTICE = (
    "Aviso: no hay motor de transcripción (Whisper); el informe no incluye audio."
)
NO_SUMMARY_NOTICE = (
    "Aviso: no hay motor de chat (Ollama) para resumir; se entregan solo las "
    "anotaciones."
)


#: Etiqueta del bloque con los datos que aporta el usuario (fecha, lugar, equipos,
#: alineación…). Se usa igual en el resumen, en el informe y en comprension.json.
CONTEXT_LABEL = "DATOS DEL PARTIDO O DE LA GRABACIÓN (los aporta el usuario y son correctos)"

#: Topes del prompt del resumen, para que quepa en ``num_ctx``. La transcripción
#: de un partido de 50 min son decenas de miles de caracteres: con ella entera el
#: prompt llenaba la ventana y Ollama devolvía «se agotó el espacio de la
#: respuesta» (aunque el modelo de resumen sea pequeño).
MAX_ANNOTATIONS_CHARS = 9000
MAX_TRANSCRIPT_CHARS = 4000


def clamp_text(texto: str, tope: int) -> str:
    """Recorta ``texto`` a ``tope`` caracteres por palabra y avisa del corte."""
    texto = str(texto or "")
    if len(texto) <= tope:
        return texto
    corte = texto[:tope]
    if " " in corte:
        corte = corte[: corte.rfind(" ")]
    return corte.rstrip() + " […]"


def context_block(context: str) -> str:
    """Normaliza los datos del usuario: una línea por dato, sin huecos ni sobras."""
    lineas = [" ".join(linea.split()) for linea in str(context or "").splitlines()]
    return "\n".join(linea for linea in lineas if linea)


def aggregate_prompt(
    plan: UnderstandingPlan,
    frame_count: int,
    has_transcript: bool,
    context: str = "",
    marcador: str = "",
    instructions: str = "",
) -> str:
    """Prompt con el que el LLM resume las anotaciones y la transcripción.

    ``context`` son los datos del usuario (fecha, lugar, equipos, alineación…):
    van como verdad para que el resumen nombre en vez de describir «un jugador».
    ``marcador`` es la serie real de tanteos leída por OCR: el único dato fiable de
    goles (sin ella, el modelo «ve» un gol en cada fotograma).
    ``instructions`` es lo que el usuario quiere destacar; el resumen lo prioriza.
    """
    croquis = ""
    primer = sport_primer(plan.sport)
    if primer:
        croquis = (
            f"{primer}\n"
            "Usa ese vocabulario y esas reglas para interpretar lo que se ve.\n\n"
        )
    limpio = context_block(context)
    if limpio:
        croquis += (
            f"{CONTEXT_LABEL}:\n{limpio}\n"
            "Esos datos son correctos: úsalos para nombrar (fecha, lugar, equipos, "
            "personas). No los contradigas ni los inventes.\n\n"
        )
    if marcador.strip():
        croquis += (
            "MARCADOR REAL LEÍDO POR OCR (izquierda-derecha según el marcador de "
            "pantalla; son los ÚNICOS goles que constan):\n"
            f"{marcador.strip()}\n"
            "No cuentes un gol por fotograma ni por cada anotación: un gol solo "
            "existe si aparece aquí. Si un momento no consta, no lo afirmes.\n\n"
        )
    pedido = context_block(instructions)
    if pedido:
        croquis += (
            f"{INSTRUCTIONS_LABEL}:\n{pedido}\n"
            "Resume dando prioridad a lo que pide el usuario (por delante de "
            "cualquier otro detalle), sin inventar nada que no esté en los bloques."
            "\n\n"
        )
    croquis += (
        "Eres un analista de vídeo. Te doy dos bloques: (1) anotaciones de "
        "fotogramas del vídeo (objetos detectados, etiquetas y texto leído en "
        "pantalla, con su marca de tiempo) y (2) la transcripción del audio.\n"
        "Escribe un resumen del vídeo con: qué ocurre y en qué orden, quién o qué "
        "aparece, y los momentos clave con su marca de tiempo. No inventes nada "
        "que no esté en los bloques: si falta información, dilo.\n"
        "Las etiquetas de CLIP y los objetos son SUGERENCIAS aproximadas: una "
        "etiqueta «gol» o una transcripción con ruido de la grada NO prueban que "
        "haya pasado nada; no afirmes un gol ni una acción por una etiqueta.\n"
        "VERIFICA antes de terminar: lista CADA gol del MARCADOR REAL (con su "
        "minuto), y cada PARADA, 7 METROS, EXCLUSIÓN o TIEMPO MUERTO que muestren "
        "las anotaciones. No te dejes ningún gol del marcador fuera del resumen.\n"
        "Sé conciso: resume en 300-500 palabras, con los momentos clave y su marca "
        "de tiempo; NO escribas un párrafo por fotograma ni repitas la misma idea.\n"
        "IDENTIDAD: si una jugadora se identifica en un momento (dorsal y nombre de "
        "la alineación), usa ese nombre para ella en los momentos siguientes cuando "
        "el equipo y el contexto encajen; no inventes nombres.\n"
    )
    if not has_transcript:
        croquis += "No hay transcripción de audio disponible.\n"
    if frame_count == 0:
        croquis += "No hay anotaciones de fotogramas disponibles.\n"
    if plan.language:
        croquis += f"Responde en {plan.language}, en texto plano.\n"
    else:
        croquis += "Responde en el mismo idioma que la transcripción, en texto plano.\n"
    return croquis


def build_report(
    plan: UnderstandingPlan,
    frames: list[FrameReport],
    transcript: str = "",
    summary: str = "",
    notices: tuple[str, ...] = (),
    context: str = "",
) -> str:
    """Compone el informe final que ve el usuario.

    Un único punto de montaje para que el worker no repita el formato en cada
    rama (mismo criterio que la FASE 5 del pipeline de documentos). El bloque de
    datos del usuario se incluye en el propio informe: así el texto (y la Crónica,
    si reutiliza este informe) lleva la alineación y el lugar sin depender de que
    los vuelvan a escribir.
    """
    plan = plan.normalized()
    bloques: list[str] = [f"Plan: {plan.describe()}"]

    limpio = context_block(context)
    if limpio:
        bloques.append(f"{CONTEXT_LABEL}:\n{limpio}")

    if notices:
        bloques.append("\n".join(notices))

    bloques.append(SUMMARY_HEADING)
    bloques.append(summary.strip() or NO_SUMMARY_NOTICE)

    bloques.append(FRAMES_HEADING)
    if frames:
        bloques.extend(
            f"t={frame.label} → {frame.summary.strip()}"
            if frame.summary.strip()
            else f"t={frame.label} → (sin hallazgos)"
            for frame in frames
        )
    else:
        bloques.append(NO_PERCEPTION_NOTICE)

    bloques.append(TRANSCRIPT_HEADING)
    bloques.append(transcript.strip() or NO_TRANSCRIPT_NOTICE)

    return "\n".join(bloques)


#: Valor por defecto del plan (evita crear uno nuevo en cada llamada).
DEFAULT_PLAN = UnderstandingPlan()

#: Nombre del fichero con la percepción estructurada que deja esta pestaña. El
#: Cuaderno lo reutiliza con «Usar percepción de Comprensión…» para no repetir
#: YOLO/CLIP/OCR/Whisper (y así el informe y la crónica salen más exactos).
UNDERSTANDING_FILENAME = "comprension.json"


class UnderstandingError(ValueError):
    """Fallo al leer/escribir el informe estructurado de Comprensión."""


def understanding_to_dict(
    *,
    video_path: str | Path = "",
    video_name: str = "",
    duration_s: float | None = None,
    plan: UnderstandingPlan | None = None,
    frames: Sequence[FrameReport] = (),
    transcript: str = "",
    summary: str = "",
    notices: Sequence[str] = (),
    context: str = "",
    marcador: Sequence = (),
    goles: Sequence = (),
    juego_inicio: float | None = None,
    descanso: float | None = None,
) -> dict:
    """Empaqueta la percepción de Comprensión en un informe reutilizable (JSON).

    Guarda lo **estructurado** (una anotación por fotograma, con su marca de
    tiempo, más la transcripción y el resumen) para que otra pestaña lo consuma
    sin volver a mirar el vídeo. ``context`` (fecha, lugar, equipos, alineación…)
    se guarda en ``contexto``: el Cuaderno lo adopta al reutilizar esta percepción.
    """
    nombre = video_name or (Path(str(video_path)).name if video_path else "")
    plan = (plan or DEFAULT_PLAN).normalized()
    return {
        "video_path": str(video_path or ""),
        "video_name": nombre,
        "duracion_s": duration_s,
        "plan": plan.describe(),
        "contexto": str(context or ""),
        "resumen": str(summary or "").strip(),
        "avisos": [str(aviso) for aviso in notices],
        "fotogramas": [
            {"t_s": float(frame.timestamp), "anotacion": str(frame.summary).strip()}
            for frame in frames
        ],
        "transcripcion": str(transcript or ""),
        "marcador": [dict(p) for p in marcador],
        "goles": [dict(g) for g in goles],
        "juego_inicio": juego_inicio,
        "descanso": descanso,
    }


def save_understanding(path: str | Path, data: dict) -> Path:
    """Escribe el informe estructurado (JSON) en ``path`` y devuelve su ruta."""
    destino = Path(path)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return destino


def load_understanding(path: str | Path) -> dict:
    """Lee un informe de Comprensión; lanza ``UnderstandingError`` si no vale."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise UnderstandingError(
            f"No se pudo leer el informe de Comprensión: {exc}"
        ) from exc
    if not isinstance(data, dict) or "fotogramas" not in data:
        raise UnderstandingError(
            "Ese fichero no es un informe de Comprensión (falta el campo "
            "'fotogramas')."
        )
    return data


def understanding_frames(data: dict) -> list[tuple[float, str]]:
    """Anotaciones ``(t_s, texto)`` de un informe de Comprensión (tolerante)."""
    salida: list[tuple[float, str]] = []
    for frame in data.get("fotogramas") or []:
        if not isinstance(frame, dict):
            continue
        try:
            instante = float(frame.get("t_s") or 0.0)
        except (TypeError, ValueError):
            instante = 0.0
        salida.append((instante, str(frame.get("anotacion", "")).strip()))
    return salida


def nearest_annotation(
    seconds: float,
    perception: Sequence[tuple[float, str]],
    tolerance_s: float = 2.5,
) -> str:
    """Anotación de percepción más cercana a ``seconds`` (dentro de la tolerancia).

    Devuelve ``""`` si no hay ninguna lo bastante cerca: es preferible un hueco a
    pegar a un fotograma la anotación de otro momento.
    """
    try:
        objetivo = float(seconds)
    except (TypeError, ValueError):
        return ""
    mejor_texto = ""
    mejor_distancia: float | None = None
    for instante, texto in perception or []:
        try:
            distancia = abs(float(instante) - objetivo)
        except (TypeError, ValueError):
            continue
        if mejor_distancia is None or distancia < mejor_distancia:
            mejor_distancia = distancia
            mejor_texto = str(texto or "").strip()
    if mejor_distancia is None or mejor_distancia > tolerance_s:
        return ""
    return mejor_texto
