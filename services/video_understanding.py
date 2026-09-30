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

from dataclasses import dataclass, field

#: Separación por defecto entre fotogramas analizados (segundos).
DEFAULT_EVERY_S = 5.0

#: Separación mínima admitida (por debajo, el coste se dispara sin aportar).
MIN_EVERY_S = 1.0

#: Número de fotogramas analizados como máximo (y por defecto).
DEFAULT_MAX_FRAMES = 24

#: Tope duro del selector (evita que la UI permita un análisis inviable).
MAX_MAX_FRAMES = 120


def format_timestamp(seconds: float) -> str:
    """``95.2`` → ``01:35`` (y ``3725`` → ``1:02:05``)."""
    total = max(0, int(seconds))
    horas, resto = divmod(total, 3600)
    minutos, segundos = divmod(resto, 60)
    if horas:
        return f"{horas}:{minutos:02d}:{segundos:02d}"
    return f"{minutos:02d}:{segundos:02d}"


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


def srt_to_text(raw: str) -> str:
    """Convierte un SRT a texto plano (sin índices ni marcas de tiempo)."""
    lineas: list[str] = []
    for linea in (raw or "").splitlines():
        texto = linea.strip()
        if not texto or texto.isdigit() or "-->" in texto:
            continue
        lineas.append(texto)
    return "\n".join(lineas)


@dataclass(frozen=True)
class UnderstandingPlan:
    """Qué se analiza de un vídeo y con qué opciones."""

    every_s: float = DEFAULT_EVERY_S
    max_frames: int = DEFAULT_MAX_FRAMES
    do_objects: bool = True
    do_labels: bool = False
    do_ocr: bool = False
    transcribe: bool = True
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


def aggregate_prompt(
    plan: UnderstandingPlan, frame_count: int, has_transcript: bool
) -> str:
    """Prompt con el que el LLM resume las anotaciones y la transcripción."""
    croquis = (
        "Eres un analista de vídeo. Te doy dos bloques: (1) anotaciones de "
        "fotogramas del vídeo (objetos detectados, etiquetas y texto leído en "
        "pantalla, con su marca de tiempo) y (2) la transcripción del audio.\n"
        "Escribe un resumen del vídeo con: qué ocurre y en qué orden, quién o qué "
        "aparece, y los momentos clave con su marca de tiempo. No inventes nada "
        "que no esté en los bloques: si falta información, dilo.\n"
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
) -> str:
    """Compone el informe final que ve el usuario.

    Un único punto de montaje para que el worker no repita el formato en cada
    rama (mismo criterio que la FASE 5 del pipeline de documentos).
    """
    plan = plan.normalized()
    bloques: list[str] = [f"Plan: {plan.describe()}"]

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
