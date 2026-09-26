"""Presets y utilidades puros de la tarea de documentos (sin Qt).

Reúne lo que antes vivía en ``ui/document_widget.py``: formatos soportados,
modelo por defecto, prompts por modo y el formateo de la previsualización
(tipo y tamaño legible del archivo).
"""

# Formatos de imagen soportados (además de PDF).
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".tif")

# Filtro del selector de archivo.
FILE_FILTER = "Documentos (*.pdf *.png *.jpg *.jpeg *.bmp *.webp *.tiff *.tif)"

# Modelo por defecto (fallback si /api/tags no responde o no hay modelos de visión).
DEFAULT_MODEL = "qwen3-vl:8b"

# Prompts por modo.
PROMPTS = {
    "ocr": (
        "Extract and transcribe all visible text from this image. Preserve the "
        "original layout, line breaks, and the original language. Do NOT translate. "
        "Do not add any commentary or explanations."
    ),
    "resumen": (
        "Summarize the content of this document in 3-5 sentences. Focus on the "
        "key information and main points."
    ),
    "tablas": (
        "Extract all tables from this image and return them as HTML tables. "
        "If there is no table, return 'No tables found'."
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
        "Extract all tables from the following text and return them as HTML tables. "
        "If there is no table, return 'No tables found'."
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


def prompt_for(mode: str | None) -> str:
    """Prompt del modo indicado (cae a OCR si el modo falta o es desconocido)."""
    return PROMPTS.get(mode or "", PROMPTS["ocr"])


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
