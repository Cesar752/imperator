"""Router de intención: a qué tarea de la plataforma apunta un mensaje del chat.

Lógica pura (sin Qt): dado el texto del usuario, dice a qué pestaña corresponde
la petición y con qué acción («resume este PDF», «analiza este vídeo»). Con eso
la ventana principal puede **ofrecer** abrir la pestaña adecuada en lugar de que
el usuario la busque a mano: es la pieza que faltaba para que el chat deje de
ser una isla y pueda orquestar el resto de tareas.

Decisiones de diseño:

- **No es un LLM.** Son reglas léxicas deterministas, multilingües (es/eu/en) y
  testeables. Un router con LLM costaría una llamada y sería impredecible para
  una decisión de UI.
- **Se exige señal compuesta.** Un sustantivo de tipo de contenido vale 1.5 y un
  verbo de acción 1.2, pero el umbral es 2.5: «pdf» a secas **no** enruta
  (podría ser una pregunta sobre PDFs), «resume este pdf» sí. Así se evita
  secuestrar conversaciones normales.
- **Empate → chat.** Sin señal suficiente devuelve ``KIND_CHAT`` y el mensaje se
  queda donde estaba.
- **Se detecta el idioma por palabras funcionales del propio texto**, como
  ``document_params.detect_language``: los términos ya van en los tres idiomas,
  así que no hace falta adivinar nada.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# ---- Tipos de destino (coinciden con los ids de tarea) ----

KIND_CHAT = "chat"
KIND_DOCUMENT = "document"
KIND_IMAGE = "image"
KIND_VIDEO = "video"
KIND_UNDERSTANDING = "understanding"
KIND_AUDIO = "audio"
KIND_ASSEMBLER = "assembler"

#: Etiquetas legibles por tipo (la UI no debe depender de Task para esto).
TASK_LABELS: dict[str, str] = {
    KIND_CHAT: "Chat",
    KIND_DOCUMENT: "Documentos",
    KIND_IMAGE: "Imagen",
    KIND_VIDEO: "Vídeo",
    KIND_UNDERSTANDING: "Comprensión",
    KIND_AUDIO: "Audio",
    KIND_ASSEMBLER: "Ensamblador",
}

# ---- Acciones ----

ACTION_NONE = ""
ACTION_SUMMARY = "summary"
ACTION_OCR = "ocr"
ACTION_TABLES = "tables"
ACTION_GENERATE = "generate"
ACTION_TRANSCRIBE = "transcribe"
ACTION_ANALYZE = "analyze"
ACTION_ASSEMBLE = "assemble"
#: Pedir un **guion** (no la generación directa): el chat planifica una historia
#: coherente y la pestaña Vídeo la muestra como escenas editables.
ACTION_PLAN_SCRIPT = "plan_script"

ACTION_LABELS: dict[str, str] = {
    ACTION_NONE: "abrir la tarea",
    ACTION_SUMMARY: "resumen",
    ACTION_OCR: "transcripción/OCR",
    ACTION_TABLES: "extracción de tablas",
    ACTION_GENERATE: "generación",
    ACTION_TRANSCRIBE: "transcripción",
    ACTION_ANALYZE: "análisis",
    ACTION_ASSEMBLE: "montaje",
    ACTION_PLAN_SCRIPT: "guion",
}

#: Puntuación mínima para enrutar. Con 2.5 hacen falta dos señales (un tipo de
#: contenido y una acción), no una palabra suelta.
MIN_SCORE = 2.5

#: Puntuación a la que la confianza se considera total.
CONFIDENCE_SATURATION = 5.0

#: Pesos de las señales.
W_CONTENT = 1.5       # sustantivo de tipo de contenido ("pdf", "imagen")
W_CONTENT_PLURAL = 1.5
W_VERB = 1.2          # verbo de acción ("resume", "analiza")
W_PHRASE = 2.5        # frase explícita ("genera una imagen", "analiza este video")
W_WEAK = 0.6          # señal ambigua ("logo", "hablar")


@dataclass(frozen=True)
class _Rule:
    """Una señal léxica: término normalizado → tipo, acción y peso."""

    term: str
    kind: str
    action: str = ACTION_NONE
    weight: float = W_CONTENT


def _r(term: str, kind: str, action: str = ACTION_NONE, weight: float = W_CONTENT) -> _Rule:
    return _Rule(term, kind, action, weight)


#: Tabla de señales. Las frases (con espacio) se buscan como subcadena; las
#: palabras sueltas, con límite de palabra. Todo va normalizado (minúsculas y sin
#: acentos) porque :func:`_normalize` limpia el texto de entrada igual.
RULES: tuple[_Rule, ...] = (
    # ---- Documentos -------------------------------------------------
    _r("pdf", KIND_DOCUMENT),
    _r("pdfs", KIND_DOCUMENT),
    _r("documento", KIND_DOCUMENT),
    _r("documentos", KIND_DOCUMENT),
    _r("document", KIND_DOCUMENT),
    _r("documents", KIND_DOCUMENT),
    _r("dokumentu", KIND_DOCUMENT),
    _r("escaneado", KIND_DOCUMENT, weight=W_VERB),
    _r("escaneados", KIND_DOCUMENT, weight=W_VERB),
    _r("scanned", KIND_DOCUMENT, weight=W_VERB),
    _r("este pdf", KIND_DOCUMENT, weight=W_PHRASE),
    _r("este documento", KIND_DOCUMENT, weight=W_PHRASE),
    _r("resume este pdf", KIND_DOCUMENT, ACTION_SUMMARY, W_PHRASE),
    _r("resume", KIND_DOCUMENT, ACTION_SUMMARY, W_VERB),
    _r("resumen", KIND_DOCUMENT, ACTION_SUMMARY, W_VERB),
    _r("resumelo", KIND_DOCUMENT, ACTION_SUMMARY, W_VERB),
    _r("summarize", KIND_DOCUMENT, ACTION_SUMMARY, W_VERB),
    _r("summarise", KIND_DOCUMENT, ACTION_SUMMARY, W_VERB),
    _r("laburpen", KIND_DOCUMENT, ACTION_SUMMARY, W_VERB),
    _r("ocr", KIND_DOCUMENT, ACTION_OCR),
    _r("transcribe", KIND_DOCUMENT, ACTION_OCR, W_VERB),
    _r("transcribir", KIND_DOCUMENT, ACTION_OCR, W_VERB),
    _r("tabla", KIND_DOCUMENT, ACTION_TABLES, W_VERB),
    _r("tablas", KIND_DOCUMENT, ACTION_TABLES, W_VERB),
    _r("tables", KIND_DOCUMENT, ACTION_TABLES, W_VERB),
    # ---- Imagen -----------------------------------------------------
    _r("imagen", KIND_IMAGE),
    _r("imagenes", KIND_IMAGE),
    _r("image", KIND_IMAGE),
    _r("images", KIND_IMAGE),
    _r("irudi", KIND_IMAGE),
    _r("foto", KIND_IMAGE),
    _r("ilustracion", KIND_IMAGE),
    _r("logo", KIND_IMAGE, weight=W_WEAK),
    _r("genera una imagen", KIND_IMAGE, ACTION_GENERATE, W_PHRASE),
    _r("crea una imagen", KIND_IMAGE, ACTION_GENERATE, W_PHRASE),
    _r("dibuja", KIND_IMAGE, ACTION_GENERATE, W_VERB),
    _r("generate an image", KIND_IMAGE, ACTION_GENERATE, W_PHRASE),
    # ---- Vídeo (generación) -----------------------------------------
    _r("video", KIND_VIDEO),
    _r("videos", KIND_VIDEO),
    _r("bideo", KIND_VIDEO),
    _r("genera un video", KIND_VIDEO, ACTION_GENERATE, W_PHRASE),
    _r("crea un video", KIND_VIDEO, ACTION_GENERATE, W_PHRASE),
    _r("generate a video", KIND_VIDEO, ACTION_GENERATE, W_PHRASE),
    # ---- Vídeo (guion: historia coherente con escenas) --------------
    # Frases explícitas (2.5) para no secuestrar conversaciones: «guion» a secas
    # pesa 1.2 y por debajo del umbral no enruta.
    _r("guion de un video", KIND_VIDEO, ACTION_PLAN_SCRIPT, W_PHRASE),
    _r("guion de video", KIND_VIDEO, ACTION_PLAN_SCRIPT, W_PHRASE),
    _r("hazme un guion", KIND_VIDEO, ACTION_PLAN_SCRIPT, W_PHRASE),
    _r("escribe un guion", KIND_VIDEO, ACTION_PLAN_SCRIPT, W_PHRASE),
    _r("guion", KIND_VIDEO, ACTION_PLAN_SCRIPT, W_VERB),
    _r("guiones", KIND_VIDEO, ACTION_PLAN_SCRIPT, W_VERB),
    _r("storyboard", KIND_VIDEO, ACTION_PLAN_SCRIPT, W_CONTENT),
    _r("cortometraje", KIND_VIDEO, ACTION_PLAN_SCRIPT, W_CONTENT),
    _r("corto animado", KIND_VIDEO, ACTION_PLAN_SCRIPT, W_PHRASE),
    _r("historia cinematografica", KIND_VIDEO, ACTION_PLAN_SCRIPT, W_PHRASE),
    _r("video con escenas", KIND_VIDEO, ACTION_PLAN_SCRIPT, W_PHRASE),
    _r("escenas", KIND_VIDEO, ACTION_PLAN_SCRIPT, W_VERB),
    _r("movie script", KIND_VIDEO, ACTION_PLAN_SCRIPT, W_PHRASE),
    _r("short film", KIND_VIDEO, ACTION_PLAN_SCRIPT, W_PHRASE),
    _r("gidoia", KIND_VIDEO, ACTION_PLAN_SCRIPT, W_VERB),
    # ---- Comprensión de vídeo ---------------------------------------
    _r("analiza este video", KIND_UNDERSTANDING, ACTION_ANALYZE, W_PHRASE),
    _r("analiza el video", KIND_UNDERSTANDING, ACTION_ANALYZE, W_PHRASE),
    _r("describe el video", KIND_UNDERSTANDING, ACTION_ANALYZE, W_PHRASE),
    _r("describe este video", KIND_UNDERSTANDING, ACTION_ANALYZE, W_PHRASE),
    _r("resume el video", KIND_UNDERSTANDING, ACTION_SUMMARY, W_PHRASE),
    _r("que sale en el video", KIND_UNDERSTANDING, ACTION_ANALYZE, W_PHRASE),
    _r("comprension de video", KIND_UNDERSTANDING, ACTION_ANALYZE, W_PHRASE),
    _r("understand video", KIND_UNDERSTANDING, ACTION_ANALYZE, W_PHRASE),
    # ---- Audio ------------------------------------------------------
    _r("audio", KIND_AUDIO),
    _r("audios", KIND_AUDIO),
    _r("voz", KIND_AUDIO),
    _r("voice", KIND_AUDIO),
    _r("narra", KIND_AUDIO, ACTION_GENERATE, W_VERB),
    _r("narracion", KIND_AUDIO, ACTION_GENERATE, W_VERB),
    _r("tts", KIND_AUDIO, ACTION_GENERATE, W_VERB),
    _r("text to speech", KIND_AUDIO, ACTION_GENERATE, W_PHRASE),
    _r("ponle voz", KIND_AUDIO, ACTION_GENERATE, W_PHRASE),
    _r("hablar", KIND_AUDIO, ACTION_GENERATE, W_WEAK),
    _r("whisper", KIND_AUDIO, ACTION_TRANSCRIBE, W_VERB),
    _r("transcribe el audio", KIND_AUDIO, ACTION_TRANSCRIBE, W_PHRASE),
    _r("subtitulos", KIND_AUDIO, ACTION_TRANSCRIBE, W_VERB),
    # ---- Ensamblador ------------------------------------------------
    _r("ensambla", KIND_ASSEMBLER, ACTION_ASSEMBLE, W_PHRASE),
    _r("assemble", KIND_ASSEMBLER, ACTION_ASSEMBLE, W_PHRASE),
    _r("monta un video", KIND_ASSEMBLER, ACTION_ASSEMBLE, W_PHRASE),
    _r("montar el video", KIND_ASSEMBLER, ACTION_ASSEMBLE, W_PHRASE),
    _r("unir clips", KIND_ASSEMBLER, ACTION_ASSEMBLE, W_PHRASE),
    _r("unir los clips", KIND_ASSEMBLER, ACTION_ASSEMBLE, W_PHRASE),
    _r("quemar subtitulos", KIND_ASSEMBLER, ACTION_ASSEMBLE, W_PHRASE),
    _r("srt", KIND_ASSEMBLER, ACTION_ASSEMBLE, W_VERB),
    _r("monta", KIND_ASSEMBLER, ACTION_ASSEMBLE, W_VERB),
    _r("montaje", KIND_ASSEMBLER, ACTION_ASSEMBLE, W_VERB),
    _r("ensamblador", KIND_ASSEMBLER, ACTION_ASSEMBLE, W_VERB),
)


@dataclass(frozen=True)
class Intent:
    """Resultado del router."""

    kind: str = KIND_CHAT
    action: str = ACTION_NONE
    score: float = 0.0
    matched: tuple[str, ...] = field(default_factory=tuple)
    raw: str = ""

    @property
    def is_routed(self) -> bool:
        """``True`` si la petición apunta a una tarea distinta del chat."""
        return self.kind != KIND_CHAT

    @property
    def confidence(self) -> float:
        """Confianza normalizada (0-1) a partir de la puntuación."""
        if self.score <= 0:
            return 0.0
        return min(1.0, self.score / CONFIDENCE_SATURATION)

    @property
    def label(self) -> str:
        return TASK_LABELS.get(self.kind, self.kind)

    @property
    def action_label(self) -> str:
        return ACTION_LABELS.get(self.action, self.action)

    def describe(self) -> str:
        """Frase para mostrar al usuario (avisos, diálogos, logs)."""
        if not self.is_routed:
            return "El mensaje se queda en el chat."
        return f"Parece una tarea de «{self.label}» ({self.action_label})."


def _normalize(text: str) -> str:
    """Minúsculas, sin acentos y con espacios colapsados."""
    decomposed = unicodedata.normalize("NFKD", text or "")
    sin_acentos = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", sin_acentos.lower()).strip()


def _matches(normalized: str, term: str) -> bool:
    """¿Aparece ``term``? Frases como subcadena; palabras con límite de palabra."""
    if " " in term:
        return term in normalized
    return re.search(rf"\b{re.escape(term)}\b", normalized) is not None


def route(text: str) -> Intent:
    """Decide a qué tarea apunta ``text``.

    Devuelve ``Intent(KIND_CHAT)`` si no hay señal suficiente o si el mejor
    candidato no supera :data:`MIN_SCORE`.
    """
    normalized = _normalize(text)
    if not normalized:
        return Intent(raw=text or "")

    scores: dict[str, float] = {}
    actions: dict[str, tuple[float, str]] = {}
    matched: dict[str, list[str]] = {}

    for rule in RULES:
        if not _matches(normalized, rule.term):
            continue
        scores[rule.kind] = scores.get(rule.kind, 0.0) + rule.weight
        matched.setdefault(rule.kind, []).append(rule.term)
        if rule.action != ACTION_NONE:
            previous = actions.get(rule.kind)
            if previous is None or rule.weight > previous[0]:
                actions[rule.kind] = (rule.weight, rule.action)

    if not scores:
        return Intent(raw=text or "")

    kind = max(scores, key=lambda k: scores[k])
    score = scores[kind]
    if score < MIN_SCORE:
        return Intent(raw=text or "")

    action = actions.get(kind, (0.0, ACTION_NONE))[1]
    return Intent(
        kind=kind,
        action=action,
        score=score,
        matched=tuple(matched.get(kind, ())),
        raw=text or "",
    )
