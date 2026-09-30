"""Pipeline por fases del procesado de documentos (lógica pura, sin Qt).

Es la parte *de decisión* del diagrama; quien lo ejecuta (``ui.document_widget``)
le pregunta qué fases tocan, cómo va cada página y cómo se monta la entrega:

    PDF
     ↓
    [FASE 0: Detección] ─── ¿el usuario pide OCR? ─┬─ sí → FASE 1 OCR (VL)
                                                   └─ no → texto embebido
     ↓
    ¿el usuario pide tablas? ─┬─ sí → FASE 1 Tablas (VL)
                              └─ no → saltar
     ↓
    [FASE 2: Normalización] ── ¿el usuario pide resumen?
                                 ├─ global (documento completo)
                                 └─ por apartados (por página)
     ↓
    [FASE 4: Validación ligera]
     ↓
    [FASE 5: Entrega]

Decisiones de diseño:

- **FASE 0 (detección)**: se decide *por página* si hay capa de texto suficiente
  (:func:`services.document_params.has_native_text`). Sin texto no hay
  alternativa: esa página va por OCR aunque no se haya pedido.
- **Forzar OCR**: si el usuario lo pide (:attr:`PipelinePlan.force_ocr`), la
  capa de texto se ignora y se transcribe con el modelo de visión. Sirve para
  PDFs cuyo OCR embebido es malo. Es lo que separa «¿pide OCR? sí» de «no».
- **Tablas**: pueden ser el objetivo único (modo ``tablas``, comportamiento de
  siempre) o añadirse a otra cosa (:attr:`PipelinePlan.tables` con el resumen o
  la transcripción), en cuyo caso la entrega lleva un bloque extra por página.
- **FASE 2 (resumen)**: ``global`` hace una sola llamada con el documento
  unido; ``apartados`` resume cada página en su propia llamada. El alcance es
  ``auto`` por defecto y se resuelve al conocer las páginas
  (:func:`PipelinePlan.with_page_count`): a partir de
  :data:`services.document_params.AUTO_GLOBAL_MIN_PAGES` páginas pasar a global
  no es una preferencia, es lo que evita multiplicar el coste por página (una
  llamada por página con ``qwen3-vl:8b`` son ~50 s, casi todo ``thinking``).
  La normalización (:func:`normalize_pages`) solo tiene sentido cuando hay
  resumen: el OCR y la transcripción no se normalizan.
- **FASE 4 (validación ligera)**: no bloquea nada; añade avisos al resultado
  (documento recortado, respuesta vacía). Es la red de seguridad que evita
  entregar en blanco en silencio.
- **FASE 5 (entrega)**: un único punto que compone el texto final, para que el
  formato no se repita en cada rama del worker.

Los atajos del ejecutor (una llamada al modelo en vez de dos cuando el OCR no
se ha pedido y la página no tiene texto) no viven aquí: aquí solo se declara
*qué* toca hacer.
"""

import re
from dataclasses import dataclass, replace

from services import document_params as dp

# ---- Fases (nombres tal cual aparecen en el diagrama) ----

FASE_DETECCION = "FASE 0: Detección"
FASE_OCR = "FASE 1: OCR (visión)"
FASE_TABLAS = "FASE 1: Tablas (visión)"
FASE_NORMALIZACION = "FASE 2: Normalización"
FASE_RESUMEN_GLOBAL = "FASE 2: Resumen global (documento completo)"
FASE_RESUMEN_APARTADOS = "FASE 2: Resumen por apartados"
FASE_VALIDACION = "FASE 4: Validación ligera"
FASE_ENTREGA = "FASE 5: Entrega"

# ---- Alcance del resumen (FASE 2) ----

SUMMARY_NONE = "none"
SUMMARY_SECTIONS = "sections"
SUMMARY_GLOBAL = "global"

# ---- Etiquetas de ruta de página (FASE 0/1) ----
#: Coinciden con las que ya usaba el resultado («--- Página 1 (texto) ---»).
ROUTE_TEXT = "texto"
ROUTE_OCR = "OCR"

# ---- Avisos de la FASE 4 (validación ligera) ----

#: El modelo no devolvió nada (ni resumen ni cuerpo de página).
EMPTY_RESULT_NOTICE = (
    "Aviso: el modelo no devolvió texto; revisa el modelo elegido o vuelve a "
    "intentarlo."
)

#: El documento no dio contenido (vacío, cifrado o ilegible).
NO_CONTENT_NOTICE = (
    "Aviso: no se pudo extraer contenido del documento (¿está vacío, cifrado o "
    "ilegible?)."
)

#: Etiqueta del bloque de tablas cuando acompañan a un resumen.
TABLES_HEADING = "--- Tablas (Página {index}) ---"

#: Aviso cuando el resultado trae cifras que no están en el documento.
INVENTED_NOTICE = (
    "Aviso de fidelidad: el resultado incluye cifras que no aparecen en el "
    "documento ({cifras}); puede haber datos inventados."
)

#: Cifras que el documento trae y el resultado no. **No** se avisa de ellas en la
#: entrega (ver :func:`validate`); :func:`check_fidelity` las devuelve para medir
#: la calidad del resumen.
OMITTED_NOTICE = (
    "Aviso de fidelidad: el documento tiene cifras que no aparecen en el "
    "resultado ({cifras}); puede haber omisiones."
)

#: Dígitos mínimos para comprobar una cifra. Con una o dos cifras (numeración de
#: apartados, «30 días») el ruido es mayor que la señal, así que la comprobación
#: es **ligera** a propósito: importes, cantidades y fechas largas, que son los
#: datos cuyo error se nota.
MIN_FIDELITY_DIGITS = 3

#: Cifras omitidas: no van a la entrega (ver :func:`validate`), pero
#: :func:`check_fidelity` las devuelve para las herramientas de medida.


@dataclass(frozen=True)
class PipelinePlan:
    """Qué fases se ejecutan y con qué opciones.

    Es inmutable a propósito: quien ejecuta el pipeline (el worker) no debería
    poder cambiar el plan a mitad de camino. ``mode`` se conserva porque es lo
    que determina los prompts (:func:`image_prompt`, :func:`text_prompt`), que
    ya están cubiertos por los tests de :mod:`services.document_params`.
    """

    mode: str = "ocr"
    force_ocr: bool = False
    tables: bool = False
    #: Alcance pedido (``auto``/``global``/``pages``); ``summary`` es el ya
    #: resuelto para un número de páginas concreto.
    scope: str = dp.DEFAULT_SCOPE
    #: Ejecutivo (breve, por defecto) o por apartado (detallado).
    detail: str = dp.DEFAULT_DETAIL
    summary: str = SUMMARY_NONE
    language: str | None = None
    closing: bool = False
    model: str | None = None
    parallel: int = 1

    # ---- Consultas del plan ----------------------------------------

    @property
    def wants_summary(self) -> bool:
        return self.summary != SUMMARY_NONE

    @property
    def wants_global(self) -> bool:
        return self.summary == SUMMARY_GLOBAL

    @property
    def transcribes_only(self) -> bool:
        """¿Se limita a leer el documento (nada de tablas ni resumen)?"""
        return not self.wants_summary and not self.tables

    def with_page_count(self, page_count: int) -> "PipelinePlan":
        """FASE 2: fija el alcance real del resumen al conocer las páginas.

        El plan se construye antes de abrir el documento, así que el alcance
        «automático» solo se puede resolver aquí: a partir de
        :data:`services.document_params.AUTO_GLOBAL_MIN_PAGES` páginas pasa a
        global y por debajo resume página a página. Los alcances elegidos a
        mano (global o por página) se respetan siempre.
        """
        if not dp.supports_global(self.mode):
            return replace(self, summary=SUMMARY_NONE)
        if self.scope == dp.SCOPE_GLOBAL:
            resumen = SUMMARY_GLOBAL
        elif self.scope == dp.SCOPE_PAGES:
            resumen = SUMMARY_SECTIONS
        else:
            resumen = (
                SUMMARY_GLOBAL if dp.auto_global(page_count) else SUMMARY_SECTIONS
            )
        return replace(self, summary=resumen)

    def phases(self) -> tuple[str, ...]:
        """Fases que se van a ejecutar, en orden (para el log y la UI)."""
        fases = [FASE_DETECCION]
        if self.force_ocr:
            fases.append(FASE_OCR)
        if self.tables:
            fases.append(FASE_TABLAS)
        if self.wants_summary:
            fases.append(FASE_NORMALIZACION)
            fases.append(
                FASE_RESUMEN_GLOBAL if self.wants_global else FASE_RESUMEN_APARTADOS
            )
        fases.append(FASE_VALIDACION)
        fases.append(FASE_ENTREGA)
        return tuple(fases)

    def route(self, has_text: bool) -> str:
        """Ruta de una página (FASE 0): ``texto`` o ``OCR``.

        Con ``force_ocr`` manda el modelo aunque haya capa de texto; sin él,
        una página sin texto suficiente no tiene otra salida que el OCR.
        """
        if self.force_ocr or not has_text:
            return ROUTE_OCR
        return ROUTE_TEXT

    # ---- Prompts ---------------------------------------------------

    def image_prompt(self, max_words: int | None = None) -> str:
        """Prompt de imagen del objetivo (el que ya usaba cada modo)."""
        return dp.prompt_for(self.mode, self.language, self.closing)

    def text_prompt(self, max_words: int | None = None) -> str | None:
        """Prompt para texto ya extraído, o ``None`` si no hay análisis."""
        return dp.text_prompt_for(self.mode, self.language, self.closing)

    def global_prompt(self, max_words: int | None = None) -> str | None:
        """Prompt del resumen global, o ``None`` si no aplica.

        ``max_words`` es el tope de palabras del resumen ejecutivo, que el
        ejecutor calcula del documento (``document_params.max_words_for``).
        """
        return dp.global_prompt_for(self.mode, self.language, self.closing)

    def ocr_prompt(self) -> str:
        """Prompt de transcripción (FASE 1 OCR), sin idioma de salida."""
        return dp.prompt_for("ocr")

    @property
    def auto_language(self) -> bool:
        """¿El idioma de salida es «auto» (el del documento)?"""
        return self.language in (None, dp.DEFAULT_LANGUAGE)

    def with_language(self, language: str | None) -> "PipelinePlan":
        """Fija el idioma efectivo de los prompts.

        El worker lo usa para sustituir el «auto» por el idioma **detectado** en
        el documento (``document_params.detect_language``): con «auto» el modelo
        tendía a responder en el idioma de las instrucciones —medido con un
        documento en euskera: resumen en castellano y títulos en euskera—.
        """
        return replace(self, language=language)

    def with_detail(self, detail: str | None) -> "PipelinePlan":
        """Fija el detalle del resumen (ejecutivo o por apartado)."""
        return replace(self, detail=dp.normalize_detail(detail))

    def chunk_prompt(self, max_words: int | None = None) -> str | None:
        """Prompt del resumen de un bloque (documento que no cabe de una vez)."""
        return dp.chunk_prompt_for(
            self.mode, self.language, self.closing, self.detail, max_words
        )

    @property
    def two_parts(self) -> bool:
        """¿La entrega lleva resumen ejecutivo **y** detalle por apartados?

        Es el estilo por defecto: un resumen corto de verdad solo es posible si
        el detalle va debajo (medido, conservándolo todo el resumen se queda en
        la mitad del documento).
        """
        return self.detail == dp.STYLE_TWO_PARTS

    def prompt_with(
        self, detail: str, max_words: int | None = None
    ) -> str | None:
        """Prompt del resumen forzando un estilo concreto (las dos partes)."""
        return dp.global_prompt_for(
            self.mode, self.language, self.closing, detail, max_words
        )

    def compress_prompt(self, max_words: int | None = None) -> str | None:
        """Prompt del segundo pase: comprimir el resumen sin perder datos."""
        return dp.compress_prompt_for(
            self.mode, self.language, self.closing, self.detail, max_words
        )

    def tables_extra(self) -> bool:
        """¿Las tablas van como bloque *adicional* de la entrega?

        En el modo ``tablas`` ya son el cuerpo principal (y su entrega es la de
        siempre), así que aquí cuentan cuando acompañan a otra cosa: a un
        resumen o a una transcripción.
        """
        return self.tables and self.mode != "tablas"

    def tables_image_prompt(self) -> str:
        return dp.prompt_for("tablas", self.language)

    def tables_text_prompt(self) -> str | None:
        return dp.text_prompt_for("tablas", self.language)


def from_ui(
    mode: str | None,
    force_ocr: bool = False,
    extra_tables: bool = False,
    scope: str | None = None,
    detail: str | None = None,
    global_summary: bool = False,
    closing: bool = False,
    language: str | None = None,
    model: str | None = None,
    parallel: int = 1,
) -> PipelinePlan:
    """Traduce las opciones de la UI a un plan de fases.

    ``mode`` es el objetivo principal (``ocr``/``resumen``/``tablas``), las dos
    casillas añaden capacidades (``force_ocr`` ignora la capa de texto y
    ``extra_tables`` añade un bloque de tablas) y ``scope`` es el alcance del
    resumen. ``global_summary`` se mantiene para los llamadores de la antigua
    casilla booleana (``True`` = global).

    El plan se devuelve resuelto para **una** página: como documento corto, el
    resumen sale por apartados. El worker lo vuelve a resolver con el número
    real de páginas en cuanto abre el PDF (:meth:`PipelinePlan.with_page_count`).
    """
    if scope is None:
        scope = dp.SCOPE_GLOBAL if global_summary else dp.DEFAULT_SCOPE
    objetivo = mode or "ocr"
    tablas = bool(extra_tables) or objetivo == "tablas"
    plan = PipelinePlan(
        mode=objetivo,
        force_ocr=bool(force_ocr),
        tables=tablas,
        scope=dp.normalize_scope(scope),
        detail=dp.normalize_detail(detail),
        language=language,
        closing=closing,
        model=model,
        parallel=max(1, int(parallel)),
    )
    return plan.with_page_count(1)


@dataclass(frozen=True)
class NormalizedText:
    """Resultado de la FASE 2 para el resumen global."""

    text: str
    truncated: bool = False


def normalize_pages(
    pages: list[str], max_chars: int | None = None
) -> NormalizedText:
    """Une y limpia las páginas para el resumen global (FASE 2).

    Por defecto **no recorta**: el recorte silencioso se sustituyó por el
    resumen por bloques (:func:`summary_blocks`), que cubre el documento entero
    en varias llamadas. ``max_chars`` sigue disponible para quien quiera el
    comportamiento antiguo.
    """
    texto, recortado = dp.merge_pages(pages, max_chars or 10**12)
    return NormalizedText(text=texto, truncated=recortado)


def _trocear_parrafo(parrafo: str, max_chars: int) -> list[str]:
    """Parte un párrafo demasiado largo por el último punto antes del corte."""
    trozos: list[str] = []
    resto = parrafo
    while len(resto) > max_chars:
        corte = resto.rfind(". ", 0, max_chars)
        corte = corte + 1 if corte > max_chars // 2 else max_chars
        trozos.append(resto[:corte].strip())
        resto = resto[corte:].strip()
    if resto:
        trozos.append(resto)
    return trozos


def summary_blocks(text: str, max_chars: int = dp.MAX_GLOBAL_CHARS) -> tuple[str, ...]:
    """Parte el texto en bloques que quepan en una llamada (FASE 2).

    Corta por párrafos (líneas en blanco) y, si un párrafo suelto no cabe, por
    la última frase completa: un corte a mitad de frase es peor que un bloque
    algo más corto. Un solo bloque significa que el documento cabe de una vez;
    varios, que hay que resumir por partes y juntar los resúmenes.
    """
    if not text.strip():
        return ()
    trozos: list[str] = []
    for parrafo in (p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()):
        if len(parrafo) <= max_chars:
            trozos.append(parrafo)
        else:
            trozos.extend(_trocear_parrafo(parrafo, max_chars))
    bloques: list[str] = []
    actual: list[str] = []
    largo = 0
    for trozo in trozos:
        if actual and largo + len(trozo) + 2 > max_chars:
            bloques.append("\n\n".join(actual))
            actual, largo = [], 0
        actual.append(trozo)
        largo += len(trozo) + 2
    if actual:
        bloques.append("\n\n".join(actual))
    return tuple(bloques)


@dataclass(frozen=True)
class Section:
    """Bloque de la entrega: un encabezado y su cuerpo."""

    title: str
    body: str

    def render(self) -> str:
        return f"{self.title}\n\n{self.body}"


def page_heading(index: int, route: str) -> str:
    """Encabezado de página de la entrega (formato de siempre)."""
    return f"--- Página {index} ({route}) ---"


def render_page(index: int, route: str, body: str) -> Section:
    """Sección de una página ya procesada."""
    return Section(title=page_heading(index, route), body=body)


def render_tables(index: int, body: str) -> Section:
    """Sección del bloque de tablas que acompaña a un resumen."""
    return Section(title=TABLES_HEADING.format(index=index), body=body)


_NUMERO_RE = re.compile(r"\d[\d.,]*")


def numbers(text: str) -> tuple[str, ...]:
    """Cifras del texto normalizadas a dígitos («1.234,50» → «123450»).

    Quita separadores de millar y de decimal para que «2,5» y «2.5» se comparen
    igual, sin depender del formato que use el modelo.
    """
    salida: list[str] = []
    for bruto in _NUMERO_RE.findall(text or ""):
        digitos = bruto.rstrip(".,").replace(".", "").replace(",", "")
        if digitos:
            salida.append(digitos)
    return tuple(salida)


def significant_numbers(
    text: str, min_digits: int = MIN_FIDELITY_DIGITS
) -> tuple[str, ...]:
    """Cifras con al menos ``min_digits`` dígitos, sin repetir y en orden."""
    return tuple(dict.fromkeys(n for n in numbers(text) if len(n) >= min_digits))


def _lista(cifras: tuple[str, ...], maximo: int = 5) -> str:
    """Lista de cifras para el aviso, recortada si son muchas."""
    visibles = ", ".join(cifras[:maximo])
    if len(cifras) > maximo:
        return f"{visibles} (y {len(cifras) - maximo} más)"
    return visibles


@dataclass(frozen=True)
class FidelityReport:
    """Cifras del resultado que no están en el documento, y al revés."""

    invented: tuple[str, ...] = ()
    omitted: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.invented and not self.omitted

    def warnings(self) -> tuple[str, ...]:
        salida: list[str] = []
        if self.invented:
            salida.append(INVENTED_NOTICE.format(cifras=_lista(self.invented)))
        if self.omitted:
            salida.append(OMITTED_NOTICE.format(cifras=_lista(self.omitted)))
        return tuple(salida)


def check_fidelity(
    result: str, source: str, min_digits: int = MIN_FIDELITY_DIGITS
) -> FidelityReport:
    """Comprobación ligera de fidelidad de las cifras.

    No pretende validar el contenido (eso pediría otro modelo): comprueba lo
    objetivo y barato —que las cifras del resultado estén en el documento y que
    las del documento no se hayan perdido—, que es lo que delata un resumen
    inventado o incompleto.
    """
    base = set(numbers(source))
    producidas = set(numbers(result))
    inventadas = tuple(
        n for n in significant_numbers(result, min_digits) if n not in base
    )
    omitidas = tuple(
        n for n in significant_numbers(source, min_digits) if n not in producidas
    )
    return FidelityReport(invented=inventadas, omitted=omitidas)


@dataclass(frozen=True)
class ValidationReport:
    """Resultado de la FASE 4: avisos que se añaden a la entrega."""

    warnings: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.warnings


def validate(
    sections: tuple[Section, ...] | list[Section] = (),
    summary: str | None = None,
    truncated: bool = False,
    pairs: tuple[tuple[str, str], ...] | list[tuple[str, str]] = (),
    blocks: int = 1,
) -> ValidationReport:
    """Validación ligera: no bloquea, solo avisa.

    - documento recortado por el presupuesto de la FASE 2 → aviso de recorte;
    - entrega sin nada de texto → aviso (distinto si había secciones vacías
      —documento sin contenido— o si el modelo no devolvió nada);
    - ``pairs`` son pares ``(resultado, texto de origen)`` para el chequeo de
      cifras: solo se avisa de las **inventadas** (una cifra que no está en el
      documento es un error). Las omitidas no avisan a propósito: un resumen es
      selectivo por definición —agrupar una serie implica dejar fuera sus
      valores intermedios— y un aviso que sale siempre acaba ignorándose. Ese
      dato lo sigue dando :func:`check_fidelity` para quien mida calidad
      (``tools/verify_calidad_resumen.py``);
    - ``blocks`` (cuántos bloques hubo que resumir) deja una nota informativa:
      el resumen es de resúmenes, pero cubre el documento completo.
    """
    warnings: list[str] = []
    if truncated:
        warnings.append(dp.TRUNCATED_NOTICE)
    if blocks > 1:
        warnings.append(dp.BLOCKS_NOTICE)
    texto = summary if summary is not None else "\n".join(s.body for s in sections)
    if not texto.strip():
        warnings.append(NO_CONTENT_NOTICE if sections else EMPTY_RESULT_NOTICE)
    inventadas: list[str] = []
    for resultado, origen in pairs:
        if not origen.strip():
            continue
        inventadas.extend(check_fidelity(resultado, origen).invented)
    if inventadas:
        warnings.append(
            INVENTED_NOTICE.format(cifras=_lista(tuple(dict.fromkeys(inventadas))))
        )
    return ValidationReport(tuple(dict.fromkeys(warnings)))


def deliver(
    sections: tuple[Section, ...] | list[Section] = (),
    summary: str | None = None,
    extra_sections: tuple[Section, ...] | list[Section] = (),
    report: ValidationReport | None = None,
) -> str:
    """FASE 5: compone el texto final a partir de lo que haya.

    ``summary`` manda cuando existe (resumen global: una sola pieza); si no,
    se entregan las secciones de página. ``extra_sections`` (las tablas que
    acompañan al resumen) van después, y los avisos de la FASE 4 al final.
    """
    parts: list[str] = []
    if summary is not None:
        if summary.strip():
            parts.append(summary.strip())
    else:
        parts.extend(section.render() for section in sections)
    parts.extend(section.render() for section in extra_sections)
    if report is not None:
        parts.extend(aviso for aviso in report.warnings if aviso.strip())
    return "\n\n".join(parts)
