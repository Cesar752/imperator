"""Presets y utilidades puros de la tarea de documentos (sin Qt).

Reúne lo que antes vivía en ``ui/document_widget.py``: formatos soportados,
modelo por defecto, prompts por modo y el formateo de la previsualización
(tipo y tamaño legible del archivo).

Sobre idiomas: los modelos locales de Ollama que se usan aquí (``qwen3-vl`` y la
familia ``qwen3.x``) sí trabajan en euskera, así que los prompts piden conservar
el idioma original. La limitación "sin euskera" del proyecto es exclusiva de TTS:
Kokoro no lo cubre y para voces en euskera se usa Piper.

El idioma de la *respuesta* es una opción de la UI (:data:`LANGUAGES`): ``auto``
respeta el del documento y ``es``/``eu``/``en`` lo fuerzan a español, euskera o
inglés, que es lo que permite pedir el resumen en euskera aunque el documento
venga en castellano (o al revés). El OCR nunca se traduce. Las instrucciones van
en inglés porque los modelos las siguen mejor.

El *alcance* del resumen es la otra opción: por defecto cada página se analiza
por separado, pero con el resumen global (:func:`global_prompt_for`) se recoge el
texto de todas las páginas y se hace una única llamada al modelo, que así ve el
documento completo en vez de páginas sueltas.

Los resúmenes llevan además reglas de estilo propias (:data:`SUMMARY_STYLE`):
apartados separados, frases de longitud parecida, sin repetir datos ya dados y
con una frase de cierre por bloque. Son el resultado de revisar un resumen real
que mezclaba temas, alternaba frases largas con telegráficas y repetía cifras.
"""

# Formatos de imagen soportados (además de PDF).
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".tif")

# Filtro del selector de archivo.
FILE_FILTER = "Documentos (*.pdf *.png *.jpg *.jpeg *.bmp *.webp *.tiff *.tif)"

# Modelo por defecto (fallback si /api/tags no responde o no hay modelos de visión).
DEFAULT_MODEL = "qwen3-vl:8b"

# Prompts por modo (imagen: página rasterizada o foto). El idioma de salida se
# añade aparte, en :func:`prompt_for`.
PROMPTS = {
    "ocr": (
        "Extract and transcribe all visible text from this image. Preserve the "
        "original layout, line breaks, and the original language (the text may be "
        "in Basque/euskera or any other language). Do NOT translate, summarise or "
        "normalise it. Do not add any commentary or explanations."
    ),
    "resumen": (
        "Summarize the content of this document in 3-5 sentences. Focus on the "
        "key information and main points."
    ),
    "tablas": (
        "Extract all tables from this image and return them as HTML tables, "
        "keeping the cell text in the original language. If there is no table, "
        "return 'No tables found'."
    ),
}

# Prompts para texto *ya extraído* (p. ej. la capa de texto de un PDF nativo).
# En ``ocr`` no hacen falta: el texto extraído ya es la transcripción.
TEXT_PROMPTS = {
    "resumen": (
        "Summarize the following document content in 3-5 sentences. Focus on the "
        "key information and main points."
    ),
    "tablas": (
        "Extract all tables from the following text and return them as HTML "
        "tables, keeping the cell text in the original language. If there is no "
        "table, return 'No tables found'."
    ),
}

# Prompt del resumen global: el mismo análisis, pero sobre el documento completo
# (todas las páginas juntas) en vez de página a página.
GLOBAL_PROMPT = (
    "Summarize the following complete document in 5-10 sentences. Cover the key "
    "information, the main points and the overall conclusions."
)

# Modos que *analizan* el contenido: en ellos el texto nativo también pasa por el
# modelo (y el idioma de salida es configurable). En OCR basta transcribir.
ANALYSIS_MODES = ("resumen", "tablas")

#: Modo de resumen: el único que admite alcance global y reglas de estilo
#: propias (el OCR transcribe y las tablas devuelven celdas, no prosa).
SUMMARY_MODE = "resumen"

# Modos que admiten resumir el documento *entero* de una vez. En OCR no hay nada
# que resumir y en tablas cada página trae las suyas: unirlas no es un resumen.
GLOBAL_MODES = (SUMMARY_MODE,)

# Reglas de estilo de los resúmenes, por página y globales. Salen de revisar tres
# resúmenes reales:
# - el primero mezclaba apartados en un mismo párrafo, alternaba frases
#   larguísimas con telegráficas y repetía la misma cifra en dos apartados;
# - el segundo enseñó que pedir un "cierre" por apartado con "→" era un error:
#   el modelo lo usaba para *interpretar* y llegó a inventar datos (hablaba de
#   "equipamiento" donde el texto decía acumular la hora de lactancia). Por eso
#   las reglas de fidelidad prohíben interpretar, inferir e inventar;
# - el tercero pedía títulos: sin ellos el resumen se lee como un texto corrido y
#   cuesta localizar cada apartado. Los títulos salen de los epígrafes del propio
#   documento y en texto plano, porque el área de resultado no pinta markdown.
SUMMARY_STYLE = (
    "Style: organize the answer in short blocks, one per distinct topic, and "
    "start every block with a one-line title followed by its paragraph. Take "
    "that title from the document's own headings, keep it short, write it in the "
    "target language like the rest of the answer and in plain text (no markdown "
    "symbols); do not invent titles. Keep the sentences of similar length (do "
    "not mix very long sentences with very short ones); do not repeat data "
    "already given in another block; and make sure every sentence adds "
    "information taken from the document. "
    "Fidelity: summarize only what the document states. Do not interpret, "
    "infer, deduce, give opinions or add context that is not written in the "
    "text, and do not invent terms, numbers or data. Do not add closing lines, "
    "arrows or extra commentary."
)

# Presupuesto de caracteres del texto que se manda al resumen global. El contexto
# de Ollama está acotado, así que un documento largo se recorta por el final y se
# avisa en el resultado. Está atado al contexto que piden los motores
# (``ollama_api.DEFAULT_NUM_CTX`` = 8192): 12000 caracteres son ~3000 tokens en
# castellano y hasta ~5000 en euskera, que es el idioma que peor se tokeniza. Si
# se sube este presupuesto, hay que subir también ``num_ctx``.
MAX_GLOBAL_CHARS = 12_000

# Aviso que se añade al resultado cuando el documento no cupo entero.
TRUNCATED_NOTICE = (
    "Aviso: el documento es más largo de lo que cabe en una sola consulta; "
    "este resumen cubre únicamente su primera parte."
)

# Idiomas de salida para resumen y tablas. El OCR no se traduce nunca: una
# transcripción debe conservar el idioma original del documento.
LANGUAGES: dict[str, str] = {
    "auto": "Igual que el documento",
    "es": "Español",
    "eu": "Euskera",
    "en": "Inglés",
}
DEFAULT_LANGUAGE = "auto"

_LANGUAGE_INSTRUCTIONS = {
    "auto": (
        "Answer in the SAME language as the document: if it is written in "
        "Basque/euskera, answer in Basque."
    ),
    "es": "Answer in Spanish, regardless of the language of the document.",
    "eu": "Answer in Basque (euskera), regardless of the language of the document.",
    "en": "Answer in English, regardless of the language of the document.",
}

#: Recordatorio para que el modelo no repita sus instrucciones en la respuesta
#: (se le escapaba un "(3 sentences)" copiado del prompt).
_NO_META = "Do not add commentary, notes or mentions of these instructions."

# Etiquetas legibles por modo.
MODE_LABELS = {
    "ocr": "OCR",
    "resumen": "Resumen",
    "tablas": "Extraer tablas",
}

# DPI al rasterizar páginas de PDF para OCR.
PDF_DPI = 200

# Mínimo de caracteres de texto nativo para saltarse el OCR de una página.
TEXT_THRESHOLD = 50


# Marcadores de modelos de visión con cobertura amplia de idiomas (euskera
# incluido). Los demás pueden transcribir, pero al resumir o extraer tablas de
# un idioma minoritario tienden a traducir o directamente a inventar.
MULTILINGUAL_VISION_MARKERS = ("qwen3", "qwen2.5", "gemma3", "glm4v", "glm-4v")

# Aviso al elegir un modelo sin esa cobertura en los modos que exigen *entender*
# el documento (en OCR no aplica: transcribir no requiere comprender).
LANGUAGE_WARNING = (
    "Ojo: este modelo puede no cubrir el euskera; al resumir o extraer tablas "
    "podría traducir o inventar. Para documentos en euskera, usa qwen3-vl."
)


def prompt_for(mode: str | None, language: str | None = None) -> str:
    """Prompt de imagen del modo, con la instrucción de idioma si aplica."""
    base = PROMPTS.get(mode or "", PROMPTS["ocr"])
    return _with_language(mode, base, language)


def language_instruction(language: str | None) -> str:
    """Instrucción de idioma de salida (``auto`` = el del documento)."""
    return _LANGUAGE_INSTRUCTIONS.get(
        language or DEFAULT_LANGUAGE, _LANGUAGE_INSTRUCTIONS[DEFAULT_LANGUAGE]
    )


def _with_language(mode: str | None, base: str, language: str | None) -> str:
    """Añade las reglas de estilo, el idioma de salida y la regla de no comentar.

    En ``ocr`` no se toca: una transcripción debe conservar el idioma original.
    Las reglas de estilo (y de fidelidad al texto) son solo del resumen; ver
    :data:`SUMMARY_STYLE`.
    """
    if not needs_text_analysis(mode):
        return base
    parts = [base]
    if (mode or "") == SUMMARY_MODE:
        parts.append(SUMMARY_STYLE)
    parts.append(language_instruction(language))
    parts.append(_NO_META)
    return " ".join(parts)


def supports_minority_languages(model: str | None) -> bool:
    """¿El modelo de visión cubre idiomas minoritarios como el euskera?"""
    if not model:
        return False
    name = str(model).lower()
    return any(marker in name for marker in MULTILINGUAL_VISION_MARKERS)


def language_warning(model: str | None, mode: str | None) -> str | None:
    """Aviso a mostrar, o ``None`` si ese modelo/modo no lo necesitan.

    Solo se avisa en los modos que analizan el contenido (resumen, tablas): en
    ``ocr`` basta con transcribir, y para eso no hace falta conocer el idioma.
    """
    if not needs_text_analysis(mode):
        return None
    if supports_minority_languages(model):
        return None
    return LANGUAGE_WARNING


def text_prompt_for(mode: str | None, language: str | None = None) -> str | None:
    """Prompt para analizar texto ya extraído, o ``None`` si el modo no lo necesita.

    ``ocr`` no necesita modelo cuando el documento ya trae capa de texto: esa
    capa *es* la transcripción. Resumen y tablas, en cambio, deben analizarla.
    """
    base = TEXT_PROMPTS.get(mode or "")
    if base is None:
        return None
    return _with_language(mode, base, language)


def needs_text_analysis(mode: str | None) -> bool:
    """¿El modo debe analizar el texto nativo en vez de devolverlo tal cual?

    Se comprueba contra ``ANALYSIS_MODES`` en vez de contra ``text_prompt_for``
    para no crear un ciclo (esa función consulta este mismo predicado).
    """
    return (mode or "") in ANALYSIS_MODES


def supports_global(mode: str | None) -> bool:
    """¿El modo admite resumir el documento completo de una vez?"""
    return (mode or "") in GLOBAL_MODES


def wants_global(mode: str | None, requested: bool) -> bool:
    """¿Procede el resumen global con este modo y esta opción de la UI?"""
    return bool(requested) and supports_global(mode)


def global_prompt_for(mode: str | None, language: str | None = None) -> str | None:
    """Prompt del resumen global, o ``None`` si el modo no lo admite."""
    if not supports_global(mode):
        return None
    return _with_language(mode, GLOBAL_PROMPT, language)


def merge_pages(pages: list[str], max_chars: int = MAX_GLOBAL_CHARS) -> tuple[str, bool]:
    """Une las páginas para el resumen global: ``(texto, recortado)``.

    Las páginas vacías se descartan para no gastar presupuesto en separadores.
    Si el resultado no cabe en ``max_chars`` se recorta por el final y se marca
    como recortado, para que quien lo mande pueda avisar.
    """
    texto = "\n\n".join(page.strip() for page in pages if page and page.strip())
    if len(texto) <= max_chars:
        return texto, False
    return texto[:max_chars], True


def is_pdf(suffix: str) -> bool:
    """¿La extensión corresponde a un PDF? (admite mayúsculas)."""
    return suffix.lower() == ".pdf"


def has_native_text(text: str) -> bool:
    """¿La página trae texto suficiente como para no necesitar OCR?"""
    return len(text.strip()) > TEXT_THRESHOLD


def file_kind(suffix: str) -> str:
    """Tipo legible del archivo a partir de su extensión."""
    suffix = suffix.lower()
    if suffix == ".pdf":
        return "PDF"
    if suffix in IMAGE_EXTENSIONS:
        return "Imagen"
    return suffix.lstrip(".").upper() or "Archivo"


def human_size(num: int | float) -> str:
    """Tamaño legible (B/KB/MB/GB) para la previsualización del archivo."""
    value = float(num)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def describe_file(name: str, suffix: str, size: int) -> str:
    """Línea de previsualización: ``nombre · tipo · tamaño``."""
    return f"{name} · {file_kind(suffix)} · {human_size(size)}"
