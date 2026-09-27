"""Presets y utilidades puros de la tarea de documentos (sin Qt).

Reúne lo que antes vivía en ``ui/document_widget.py``: formatos soportados,
modelo por defecto, prompts por modo y el formateo de la previsualización
(tipo y tamaño legible del archivo).

Sobre idiomas: los modelos locales de Ollama que se usan aquí (``qwen3-vl`` y la
familia ``qwen3.x``) sí trabajan en euskera, así que los prompts piden conservar
el idioma original y responder en él. La limitación "sin euskera" del proyecto
es exclusiva de TTS: Kokoro no lo cubre y para voces en euskera se usa Piper.
Las instrucciones van en inglés (los modelos las siguen mejor), pero el idioma
de la *respuesta* se fija explícitamente.
"""

# Formatos de imagen soportados (además de PDF).
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".tif")

# Filtro del selector de archivo.
FILE_FILTER = "Documentos (*.pdf *.png *.jpg *.jpeg *.bmp *.webp *.tiff *.tif)"

# Modelo por defecto (fallback si /api/tags no responde o no hay modelos de visión).
DEFAULT_MODEL = "qwen3-vl:8b"

# Prompts por modo (imagen: página rasterizada o foto).
PROMPTS = {
    "ocr": (
        "Extract and transcribe all visible text from this image. Preserve the "
        "original layout, line breaks, and the original language (the text may be "
        "in Basque/euskera or any other language). Do NOT translate, summarise or "
        "normalise it. Do not add any commentary or explanations."
    ),
    "resumen": (
        "Summarize the content of this document in 3-5 sentences, in the SAME "
        "language as the document: if it is written in Basque/euskera, answer in "
        "Basque. Focus on the key information and main points."
    ),
    "tablas": (
        "Extract all tables from this image and return them as HTML tables. Keep "
        "the cell text in the original language. If there is no table, return "
        "'No tables found'."
    ),
}

# Prompts para texto *ya extraído* (p. ej. la capa de texto de un PDF nativo).
# En ``ocr`` no hacen falta: el texto extraído ya es la transcripción.
TEXT_PROMPTS = {
    "resumen": (
        "Summarize the following document content in 3-5 sentences, in the SAME "
        "language as the text: if it is written in Basque/euskera, answer in "
        "Basque. Focus on the key information and main points."
    ),
    "tablas": (
        "Extract all tables from the following text and return them as HTML "
        "tables. Keep the cell text in the original language. If there is no "
        "table, return 'No tables found'."
    ),
}

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


def prompt_for(mode: str | None) -> str:
    """Prompt del modo indicado (cae a OCR si el modo falta o es desconocido)."""
    return PROMPTS.get(mode or "", PROMPTS["ocr"])


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


def text_prompt_for(mode: str | None) -> str | None:
    """Prompt para analizar texto ya extraído, o ``None`` si el modo no lo necesita.

    ``ocr`` no necesita modelo cuando el documento ya trae capa de texto: esa
    capa *es* la transcripción. Resumen y tablas, en cambio, deben analizarla.
    """
    return TEXT_PROMPTS.get(mode or "")


def needs_text_analysis(mode: str | None) -> bool:
    """¿El modo debe analizar el texto nativo en vez de devolverlo tal cual?"""
    return text_prompt_for(mode) is not None


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
