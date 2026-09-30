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

import re

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
#
# NO lleva un número fijo de frases a propósito. El «5-10 sentences» anterior
# obligaba al modelo a mezclar apartados en una misma frase y a resumir en
# rangos («de 107.137 a 221.329 euros»), que es justo como se pierden los datos
# de cada apartado en un documento largo.
GLOBAL_PROMPT = (
    "Summarize the following complete document. Keep every section it contains, "
    "in the order they appear: one block per section or numbered item, with the "
    "key figure or decision each one states, and with every item of every list "
    "it gives —compressed into a sentence or two, not copied—. This is a summary, "
    "not a transcription: say it with your own words and much shorter, but never "
    "drop a rule, an item or a figure, and never move information from one "
    "section to another. If the document repeats the same section over and over "
    "with different figures, group those repetitions into a single block with "
    "the values the text states."
)

#: Prompt de un bloque cuando el documento no cabe en una sola consulta: se
#: resume por partes y después se juntan (resumen de resúmenes). Es el mismo
#: encargo que :data:`GLOBAL_PROMPT` sobre una parte del documento.
CHUNK_PROMPT = (
    "Summarize this part of a longer document. Keep every section it contains, "
    "in the order they appear: one block per section or numbered item, with the "
    "key figure or decision each one states, and with every item of every list "
    "it gives —compressed into a sentence or two, not copied—. Keep each block "
    "to one or two sentences. This is not a transcription: rewrite it with your "
    "own words and much shorter, but never drop a rule, an item or a figure. If "
    "the document repeats the same section over "
    "and over with different figures, group those repetitions into a single "
    "block with the values the text states."
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

# Alcance del resumen (selector de la UI). El valor se guarda en la preferencia
# ``document_global``, que antes era un booleano ("1" = global, "0" = por
# página): :func:`normalize_scope` sigue leyendo esos valores heredados.
SCOPE_AUTO = "auto"
SCOPE_GLOBAL = "global"
SCOPE_PAGES = "pages"

# Estilo del resumen (otro selector de la UI). El **ejecutivo** es el de por
# defecto: medido, el detallado por apartado devolvía el 89 % del documento
# —fiel, sin cifras inventadas ni omitidas, pero no resumía—. El usuario eligió
# 20-30 % aceptando perder detalle secundario.
STYLE_EXECUTIVE = "executive"
STYLE_DETAILED = "detailed"
#: Las dos cosas en la misma entrega: resumen ejecutivo breve **y** detalle por
#: apartado debajo. Es la única forma de tener un resumen corto de verdad sin
#: perder datos: medido, un documento denso en cifras no baja del ~50 % si se
#: conserva todo, y bajar de ahí exige omitir.
STYLE_TWO_PARTS = "two"

DETAIL_LABELS: dict[str, str] = {
    STYLE_TWO_PARTS: "Ejecutivo + detalle por apartados",
    STYLE_EXECUTIVE: "Ejecutivo (breve)",
    STYLE_DETAILED: "Por apartado (detallado)",
}

DEFAULT_DETAIL = STYLE_TWO_PARTS

#: Encabezados de la entrega en dos partes.
EXECUTIVE_HEADING = "== Resumen ejecutivo =="
DETAIL_HEADING = "== Detalle por apartados =="

SCOPE_LABELS: dict[str, str] = {
    SCOPE_AUTO: "Automático (global si el documento es largo)",
    SCOPE_GLOBAL: "Global (una sola llamada)",
    SCOPE_PAGES: "Por página",
}

DEFAULT_SCOPE = SCOPE_AUTO

#: A partir de este número de páginas, el alcance «automático» resume el
#: documento completo en una sola llamada. Medido con ``qwen3-vl:8b`` y páginas
#: de ~900 caracteres: una llamada por página cuesta ~50 s (19 páginas ≈ 16 min),
#: mientras que el resumen global del mismo documento tarda **47,7 s**. El
#: motivo del coste por página es el razonamiento del modelo: ~5 900 tokens
#: generados por página para ~650 caracteres de respuesta, el resto «thinking».
AUTO_GLOBAL_MIN_PAGES = 6

# Reglas de estilo de los resúmenes, por página y globales. Salen de revisar
# resúmenes reales. Los cuatro fallos que se repitieron en el resumen de un
# convenio laboral, y la regla que los ataca:
#  1. «24 horas» resumido como «24 días», y una cifra traída de otro apartado
#     («no supera 24 horas en dos semanas») → cada cifra con su concepto y su
#     unidad, y ninguna cifra que no esté en el texto.
#  2. El trabajo nocturno aparecía dentro de «Ordutegia», y el descanso semanal
#     dentro del descanso entre jornadas → un apartado, un bloque: nada se mueve
#     de sitio.
#  3. «prohibido para los nocturnos, salvo fuerza mayor» resumido como «excepto
#     en trabajos nocturnos», justo al revés → no invertir el sentido: decir a
#     quién se prohíbe y a quién se exceptúa.
#  4. La lista de permisos retribuidos quedó en dos de nueve → las listas se
#     conservan enteras, aunque el bloque crezca.
#
# Y de revisar tres resúmenes anteriores:
# - el primero mezclaba apartados en un mismo párrafo, alternaba frases
#   larguísimas con telegráficas y repetía la misma cifra en dos apartados;
# - el segundo enseñó que pedir un "cierre" por apartado con "→" era un error si
#   se pide a secas: el modelo lo usaba para *interpretar* y llegó a inventar
#   datos (hablaba de "equipamiento" donde el texto decía acumular la hora de
#   lactancia). Por eso las reglas de fidelidad prohíben interpretar e inventar, y
#   el cierre pasó a ser opcional y extractivo (ver CLOSING_STYLE);
# - el tercero pedía títulos: sin ellos el resumen se lee como un texto corrido y
#   cuesta localizar cada apartado. Los títulos salen de los epígrafes del propio
#   documento y en texto plano, porque el área de resultado no pinta markdown.
SUMMARY_STYLE = (
    "Style: organize the answer in short blocks, one per distinct topic, and "
    "start every block with a one-line title followed by its paragraph. Take "
    "that title from the document's own headings —use every heading, one block "
    "each—, keep it short, write it in the target language like the rest of the "
    "answer and in plain text (no markdown symbols); do not invent titles. "
    "Rewrite, do not transcribe: never copy the document's sentences. Say the "
    "same with your own words and shorter. Write the body of each block as "
    "complete sentences of similar length (roughly the same number of words "
    "each), not as a list of telegraphic fragments, and make sure every "
    "sentence adds information taken from the document. "
    "Never translate: keep the document's own language in the body of every "
    "block, as in its headings. "
    "One section, one block: a block must contain only what its own section "
    "states. Never move a statement, a figure or a rule to another section, "
    "never merge two different sections into one block, and do not repeat data "
    "already given in another block. Keep every figure with the concept and the "
    "unit it belongs to —do not turn hours into days, do not reuse a figure "
    "taken from a different section—. "
    "Lists and conditions: a list is summarised, not copied. Keep **every** "
    "item, but compress the list into one or two sentences with each item's "
    "figure in brackets —e.g. «Permisos retribuidos: entre 1 y 15 días según el "
    "caso (matrimonio 15, hospitalización 5, fallecimiento 2, mudanza 1)»—. "
    "Never drop an item and never drop a figure, but never transcribe the "
    "original wording either. If the document forbids something to a group or "
    "exempts it, state exactly who is forbidden and who is exempt, and never "
    "invert the sense of the rule. "
    "Length: one or two sentences per block, and the whole answer clearly "
    "shorter than the document —a fraction of its length—: drop the wording, "
    "never the rules, the items or the figures. Only if the document repeats the "
    "same section over and over with different figures (the same record repeated "
    "once per page) group those repetitions into one block; never group sections "
    "that state different things. "
    "Fidelity: summarize only what the document states. Do not interpret, "
    "infer, deduce, give opinions or add context that is not written in the "
    "text, and do not invent terms, numbers or data. Never quote a figure that "
    "is not literally in the text: do not continue series or extrapolate "
    "patterns, and never complete the values of items the document does not "
    "state. A range is allowed only if both ends appear in the text."
)

#: Estilo del resumen **ejecutivo**. El objetivo es un quinto del documento: se
#: conservan las reglas y cifras que cambian una decisión y se deja fuera el
#: detalle secundario. Nada de «un bloque por apartado» (eso es el detallado).
#: El tope de palabras se calcula del documento (`max_words_for`) y se inyecta
#: aquí: pedir «un quinto» no basta —medido, el modelo devolvía el 83 %—, pero
#: con un número explícito de palabras sí se ajusta.
EXECUTIVE_STYLE = (
    "Length: this is an executive summary of at most {max_words} words —around "
    "a fifth of the document—. Write a few paragraphs —or one short block per "
    "major topic at most—, not one block per section. "
    "Keep the rules and figures that decide something: limits, deadlines, "
    "entitlements, prohibitions and who they apply to. Drop the wording, the "
    "examples, the explanatory clauses and whatever does not change a decision. "
    "Rewrite, do not transcribe: never copy the document's sentences, and do "
    "not list every value of a series —say its range or how it evolves—. "
    "Fidelity: summarize only what the document states. Do not interpret, "
    "infer, deduce or invent terms, numbers or data. Keep each figure with the "
    "concept and the unit it belongs to, do not reuse a figure taken from a "
    "different section, and never invert the sense of a rule: state who is "
    "forbidden and who is exempt. A range is allowed only if both ends appear "
    "in the text."
)

#: Prompt del **segundo pase** del resumen ejecutivo: si el primer resumen se
#: pasa del tope de palabras, se le pide comprimirse a sí mismo. Reescribir el
#: resumen (y no el documento) conserva las cifras ya seleccionadas y evita
#: volver a leer el original. Medido: con el tope en el prompt el resumen
#: ejecutivo bajó del 83 % al 54 %; con este pase baja al objetivo.
COMPRESS_PROMPT = (
    "The text below is a summary and it is still too long. Rewrite it in at "
    "most {max_words} words, in the same language, keeping every rule, item and "
    "figure it contains and dropping only the wording. Do not add anything new "
    "and do not merge figures into ranges."
)

#: Regla por defecto sobre los cierres: son lo que invitaba a interpretar.
NO_CLOSING_STYLE = (
    "Do not add closing lines, arrows, wrap-ups or extra commentary."
)

#: Frase de cierre por bloque («→ …»), **opcional** (casilla de la UI, apagada
#: por defecto). Se pide extractiva a propósito: al pedirla sin más, el modelo
#: inventaba datos para "concluir". Si el resumen es para un examen, mejor
#: dejarla apagada: una frase interpretada resta puntos.
CLOSING_STYLE = (
    "After each block's paragraph, add one closing line that starts with '→'. "
    "That line must recap the block using information already written in it "
    "(quote or rephrase its own words): never add claims, opinions, advice or "
    "data that the block does not state."
)

# Presupuesto de caracteres **por llamada** del resumen global. El contexto de
# Ollama está acotado (16384 tokens por petición con la configuración actual:
# `num_ctx` 32768 repartido entre dos slots), así que un documento que no quepa
# no se recorta: se resume **por bloques** y luego se resume el conjunto
# (ver `services/document_pipeline.summary_blocks`). 16000 caracteres son ~4600
# tokens en castellano y hasta ~8000 en euskera, que es el idioma que peor se
# tokeniza; con el razonamiento del modelo (~4500 tokens medidos) sigue habiendo
# margen. Si se sube, hay que subir también `num_ctx`.
MAX_GLOBAL_CHARS = 16_000

#: Aviso que se añade al resultado cuando el documento hubo que resumirlo por
#: bloques. No es un error: es la garantía de que no se ha perdido nada.
BLOCKS_NOTICE = (
    "Nota: el documento no cabía en una sola consulta; el resumen se ha "
    "elaborado por bloques (resumen de resúmenes) y cubre el documento completo."
)

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
        "Write the answer in the SAME language as the document text, whatever "
        "the language of these instructions is: if the document is written in "
        "Spanish, answer in Spanish; if it is in Basque/euskera, answer in "
        "Basque. Never switch to English unless the document is in English."
    ),
    "es": (
        "Write the answer in Spanish, regardless of the language of the "
        "document and of the language of these instructions."
    ),
    "eu": (
        "Write the answer in Basque (euskera), regardless of the language of "
        "the document and of the language of these instructions."
    ),
    "en": (
        "Write the answer in English, regardless of the language of the "
        "document and of the language of these instructions."
    ),
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


#: Fracción del documento que puede ocupar el resumen ejecutivo, y suelo de
#: palabras para que un documento corto no quede mutilado.
EXECUTIVE_RATIO = 0.25
EXECUTIVE_MIN_WORDS = 80


def max_words_for(text: str, ratio: float = EXECUTIVE_RATIO) -> int:
    """Tope de palabras del resumen ejecutivo para un documento dado.

    Se cuenta por palabras del documento (no caracteres) porque es lo que el
    modelo puede controlar al escribir. Medido: sin tope explícito, el
    «ejecutivo» devolvía el 83 % del documento aunque el prompt pidiera un
    quinto.
    """
    palabras = len(text.split())
    return max(EXECUTIVE_MIN_WORDS, int(palabras * ratio))


#: Margen sobre el tope antes de pedir el segundo pase: los modelos no miden
#: palabras, así que se comprime cuando se pasa de verdad, no por una línea.
EXECUTIVE_WORD_MARGIN = 1.2


def excede_palabras(text: str, max_words: int) -> bool:
    """¿El texto se pasa del tope (con margen) y toca comprimirlo otra vez?"""
    return len(text.split()) > max_words * EXECUTIVE_WORD_MARGIN


#: Segundo pase de compresión del resumen ejecutivo, **apagado por defecto**.
#: Medido con un convenio denso en datos: encenderlo subía el resumen del 54 %
#: al 65 % del documento (el modelo vuelve a conservarlo todo) y doblaba el
#: tiempo. En un documento que es una lista de datos, un resumen corto solo se
#: consigue **omitiendo** datos; no hay prompt que lo evite. Se deja aquí por si
#: algún día interesa comprimir a costa de perder detalle.
EXECUTIVE_COMPRESS = False


def style_for(detail: str | None, max_words: int | None = None) -> str:
    """Reglas de estilo del resumen según el detalle elegido.

    El **ejecutivo** (por defecto) resume de verdad —un quinto del documento—;
    el detallado conserva todos los ítems y cifras y, medido, ocupaba el 89 %
    del documento.
    """
    if (detail or DEFAULT_DETAIL) in (STYLE_EXECUTIVE, STYLE_TWO_PARTS):
        # En «dos partes», este estilo es el de la parte breve (la primera).
        return EXECUTIVE_STYLE.format(max_words=max_words or EXECUTIVE_MIN_WORDS)
    return SUMMARY_STYLE


def prompt_for(
    mode: str | None,
    language: str | None = None,
    closing: bool = False,
    detail: str | None = None,
    max_words: int | None = None,
) -> str:
    """Prompt de imagen del modo, con la instrucción de idioma si aplica."""
    base = PROMPTS.get(mode or "", PROMPTS["ocr"])
    return _with_language(mode, base, language, closing, detail, max_words)


#: Palabras funcionales para reconocer el idioma del documento sin dependencias
#: externas. Se comparan por **prefijo** (no solo por igualdad) porque el
#: euskera declina por sufijos: «lan», «lanaldia», «laneko» son la misma raíz.
_LANGUAGE_MARKERS: dict[str, tuple[str, ...]] = {
    "eu": (
        "eta", "edo", "dira", "izan", "behar", "egun", "lan", "ordu", "aste",
        "hilabete", "urte", "baimen", "opor", "ordutegi", "kontratu", "txanda",
        "gaueko", "gutxienez", "gehienez", "ordain", "egokit", "langile",
        "emandako", "arautegia", "jardun", "denbora", "errepos", "jaiegun",
    ),
    "es": (
        "del", "los", "las", "que", "con", "para", "una", "por", "trabajo",
        "jornada", "dias", "días", "horas", "permiso", "vacaciones", "contrato",
        "empresa", "minimo", "mínimo", "maximo", "máximo", "trabajadores",
        "horario", "descanso", "anuales", "debe", "puede", "tiene",
    ),
    "en": (
        "the", "and", "with", "that", "days", "hours", "work", "leave",
        "annual", "employees", "must", "shall", "may",
    ),
}


def detect_language(text: str) -> str | None:
    """Idioma probable del texto (``eu``/``es``/``en``), o ``None`` si no está claro.

    Hace falta para el alcance «auto»: pedir «responde en el mismo idioma» **no
    basta** —medido con un documento en euskera, el resumen global salía en
    castellano con los títulos en euskera—; diciéndole al modelo *cuál* es el
    idioma del documento, sí. Es una heurística de palabras funcionales, no un
    detector serio: si la diferencia no es clara devuelve ``None``, porque
    equivocarse aquí es peor que no decir nada.
    """
    if not text or not text.strip():
        return None
    palabras = re.findall(r"[^\W\d_]+", text.lower())
    if not palabras:
        return None
    puntuacion = {
        idioma: sum(
            1
            for palabra in palabras
            if any(
                palabra == marcador
                or (len(marcador) >= 4 and palabra.startswith(marcador))
                for marcador in marcadores
            )
        )
        for idioma, marcadores in _LANGUAGE_MARKERS.items()
    }
    ordenados = sorted(puntuacion.values(), reverse=True)
    mejor = max(puntuacion, key=lambda idioma: puntuacion[idioma])
    if ordenados[0] < 5 or ordenados[0] < 2 * max(ordenados[1], 1):
        return None
    return mejor


def language_instruction(language: str | None) -> str:
    """Instrucción de idioma de salida (``auto`` = el del documento)."""
    return _LANGUAGE_INSTRUCTIONS.get(
        language or DEFAULT_LANGUAGE, _LANGUAGE_INSTRUCTIONS[DEFAULT_LANGUAGE]
    )


def _with_language(
    mode: str | None,
    base: str,
    language: str | None,
    closing: bool = False,
    detail: str | None = None,
    max_words: int | None = None,
) -> str:
    """Añade las reglas de estilo, el idioma de salida y la regla de no comentar.

    En ``ocr`` no se toca: una transcripción debe conservar el idioma original.
    Las reglas de estilo (y de fidelidad al texto) son solo del resumen; ver
    :data:`SUMMARY_STYLE`. ``closing`` decide entre prohibir las líneas de cierre
    (:data:`NO_CLOSING_STYLE`, por defecto) o pedirlas de forma extractiva
    (:data:`CLOSING_STYLE`).
    """
    if not needs_text_analysis(mode):
        return base
    # El idioma va **antes** de las reglas de estilo a propósito: con todas las
    # instrucciones en inglés, el modelo tendía a responder en inglés aunque el
    # documento fuese en castellano (medido: el mismo documento salía unas veces
    # en castellano y otras en inglés, con los títulos en un idioma y el cuerpo
    # en otro).
    parts = [base, language_instruction(language)]
    if (mode or "") == SUMMARY_MODE:
        parts.append(style_for(detail, max_words))
        parts.append(CLOSING_STYLE if closing else NO_CLOSING_STYLE)
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


def text_prompt_for(
    mode: str | None,
    language: str | None = None,
    closing: bool = False,
    detail: str | None = None,
    max_words: int | None = None,
) -> str | None:
    """Prompt para analizar texto ya extraído, o ``None`` si el modo no lo necesita.

    ``ocr`` no necesita modelo cuando el documento ya trae capa de texto: esa
    capa *es* la transcripción. Resumen y tablas, en cambio, deben analizarla.
    """
    base = TEXT_PROMPTS.get(mode or "")
    if base is None:
        return None
    return _with_language(mode, base, language, closing, detail, max_words)


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


def normalize_scope(raw: str | None) -> str:
    """Alcance guardado → uno de :data:`SCOPE_AUTO`/:data:`SCOPE_GLOBAL`/:data:`SCOPE_PAGES`.

    Acepta los valores heredados de cuando el alcance era una casilla: ``"1"``
    (global) y ``"0"`` (por página, que ahora es el automático). Cualquier otra
    cosa cae en el automático.
    """
    if raw in (SCOPE_GLOBAL, "1"):
        return SCOPE_GLOBAL
    if raw == SCOPE_PAGES:
        return SCOPE_PAGES
    return SCOPE_AUTO


def normalize_detail(raw: str | None) -> str:
    """Detalle guardado → uno de los tres estilos.

    Por defecto :data:`STYLE_TWO_PARTS` (ejecutivo + apartados), que es la forma
    de dar un resumen corto sin perder datos.
    """
    if raw in (STYLE_DETAILED, STYLE_EXECUTIVE):
        return raw
    return STYLE_TWO_PARTS


def auto_global(page_count: int, min_pages: int = AUTO_GLOBAL_MIN_PAGES) -> bool:
    """¿El alcance automático debe resumir el documento completo?

    A partir de :data:`AUTO_GLOBAL_MIN_PAGES` páginas compensa: el resumen por
    página multiplica las llamadas al modelo (y con él, el razonamiento) por el
    número de páginas.
    """
    return page_count >= min_pages


def global_prompt_for(
    mode: str | None,
    language: str | None = None,
    closing: bool = False,
    detail: str | None = None,
    max_words: int | None = None,
) -> str | None:
    """Prompt del resumen global, o ``None`` si el modo no lo admite."""
    if not supports_global(mode):
        return None
    return _with_language(mode, GLOBAL_PROMPT, language, closing, detail, max_words)


def chunk_prompt_for(
    mode: str | None,
    language: str | None = None,
    closing: bool = False,
    detail: str | None = None,
    max_words: int | None = None,
) -> str | None:
    """Prompt del resumen de **un bloque** de un documento largo.

    ``None`` si el modo no resume: el resumen por bloques solo existe para el
    alcance global (documento completo).
    """
    if not supports_global(mode):
        return None
    return _with_language(mode, CHUNK_PROMPT, language, closing, detail, max_words)


def compress_prompt_for(
    mode: str | None,
    language: str | None = None,
    closing: bool = False,
    detail: str | None = None,
    max_words: int | None = None,
) -> str | None:
    """Prompt del segundo pase del resumen ejecutivo (comprimir sin perder datos)."""
    if not supports_global(mode):
        return None
    return _with_language(
        mode, COMPRESS_PROMPT, language, closing, detail, max_words
    )


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
