"""Widget de documentos: pipeline por fases con Ollama Vision.

El trabajo se organiza según el diagrama de fases que vive en
:mod:`services.document_pipeline` (FASE 0 detección → FASE 1 OCR/tablas →
FASE 2 normalización y resumen → FASE 4 validación ligera → FASE 5 entrega).
Aquí solo se *ejecuta*: el plan se pide al pipeline y cada fase tiene su método,
para que el flujo se lea igual que el diagrama.
"""

import logging
import tempfile
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from engines.base_engine import CANCEL_MESSAGE, Engine, EngineError
from services import document_params as dp
from services import document_pipeline as dpl
from services import preferences as prefs
from ui.task_widget import TaskWidget
from ui.workers import ThreadWorker, gpu_slot, run_pool_task

logger = logging.getLogger(__name__)

# Import diferido y protegido: si PyMuPDF no está instalado, la app sigue
# arrancando (chat/imagen/vídeo no se rompen); solo fallará al procesar un PDF.
try:
    import pymupdf
except ImportError:  # pragma: no cover - dependencia declarada en requirements.txt
    pymupdf = None

#: Páginas que se analizan a la vez. **1 = en serie**, y es una decisión medida:
#: el paralelismo sólo compensa si Ollama trae varios slots
#: (``OLLAMA_NUM_PARALLEL=2``) y además obliga a doblar ``num_ctx``, porque Ollama
#: reparte el contexto entre las peticiones simultáneas.
#:
#: Medido con ``qwen3-vl:8b``, el ``DocumentWorker`` real y un PDF de 2 páginas
#: (modo resumen por página), dos corridas independientes:
#:
#: - en serie (``1``): 60,1 s y 43,8 s;
#: - en paralelo (``2``): 59,2 s y **70,1 s** → **1,02x** y **0,62x**: ninguna vez
#:   mejor. Con lotes, además, el progreso saltaba (un aviso por lote, no por
#:   página); eso queda corregido en :meth:`DocumentWorker._with_progress`.
#:
#: Con una página por petición ya no hay reparto de contexto, así que
#: ``ollama_api.DEFAULT_VISION_NUM_CTX`` vuelve a 16384 (el doble sólo servía
#: para dar 16384 a cada uno de los dos slots).
MAX_PARALLEL_PAGES = 1


def _chunks(total: int, size: int) -> Iterator[range]:
    """Rangos de índices de página, de ``size`` en ``size`` (el último, más corto)."""
    for inicio in range(0, total, size):
        yield range(inicio, min(inicio + size, total))


_T = TypeVar("_T")


@dataclass
class _PageResult:
    """Lo que devuelve una página: el cuerpo de la entrega y su texto plano.

    ``source`` es el texto del que se puede tirar después (el bloque de tablas
    que acompaña al resumen lo reutiliza en vez de volver a rasterizar).
    """

    body: str
    source: str = ""


class DocumentWorker(ThreadWorker):
    """Procesa una imagen o un PDF fuera del hilo de la UI (FASE 0 → FASE 5).

    El plan de fases lo decide :mod:`services.document_pipeline`; aquí cada
    fase tiene su método, para que el flujo se lea igual que el diagrama.
    """

    page_progress = pyqtSignal(int, int)   # página actual, total de páginas

    def __init__(
        self,
        engine,
        file_path,
        mode,
        model,
        language=None,
        global_summary=False,
        closing=False,
        parallel=MAX_PARALLEL_PAGES,
        force_ocr=False,
        extra_tables=False,
        scope=None,
        detail=None,
        parent=None,
        scheduler=None,
        liberar_otros=None,
    ) -> None:
        super().__init__(parent)
        self._engine = engine
        self._file_path = Path(file_path)
        self._parallel = max(1, int(parallel))
        self._scheduler = scheduler
        self._liberar_otros = liberar_otros
        self._plan = dpl.from_ui(
            mode,
            force_ocr=force_ocr,
            extra_tables=extra_tables,
            scope=scope,
            detail=detail,
            global_summary=global_summary,
            closing=closing,
            language=language,
            model=model,
            parallel=self._parallel,
        )

    # ---- FASE 0 → FASE 5 ------------------------------------------

    def work(self) -> str:
        """Ejecuta el plan con una reserva de GPU (una por documento, no por página)."""
        with gpu_slot(self._scheduler, "document", liberar_otros=self._liberar_otros):
            return self._run_plan()

    def _run_plan(self) -> str:
        """Ejecuta el plan: imagen suelta, páginas, o resumen del documento."""
        plan = self._plan
        if not dp.is_pdf(self._file_path.suffix):
            logger.info("Documentos: %s", " → ".join(plan.phases()))
            return self._process_image(plan)
        # FASE 2: el alcance «automático» no se puede resolver sin saber cuántas
        # páginas trae el documento (a partir de AUTO_GLOBAL_MIN_PAGES compensa
        # resumirlo entero de una vez en vez de página a página) y el idioma
        # «auto» hay que resolverlo con el texto delante. Se guarda en
        # ``self._plan`` para que el plan ejecutado sea el que se puede inspeccionar.
        total, idioma = self._inspect_pdf()
        plan = plan.with_page_count(total)
        if idioma:
            plan = plan.with_language(idioma)
            logger.info("Documentos: idioma detectado en el documento: %s", idioma)
        self._plan = plan
        logger.info(
            "Documentos: %d páginas, resumen %s, idioma %s → %s",
            total,
            plan.summary,
            plan.language,
            " → ".join(plan.phases()),
        )
        if plan.wants_global:
            return self._process_pdf_global(plan)
        return self._process_pdf(plan)

    def _process_image(self, plan: dpl.PipelinePlan) -> str:
        """Imagen suelta: una única llamada al modelo de visión."""
        prompt = plan.ocr_prompt() if plan.force_ocr else plan.image_prompt()
        self.page_progress.emit(1, 1)
        return self._engine.analyze_image(self._file_path, prompt, plan.model)

    def _open_pdf(self):
        """Abre el PDF (PyMuPDF es opcional: sin él la app sigue arrancando)."""
        if pymupdf is None:
            raise EngineError(
                "PyMuPDF no está instalado. Instálalo con: pip install PyMuPDF>=1.24.0"
            )
        return pymupdf.open(str(self._file_path))

    def _inspect_pdf(self) -> tuple[int, str | None]:
        """Cuenta páginas y, si el idioma es «auto», detecta el del documento.

        Se hace en una sola apertura: contar páginas cuesta milisegundos y la
        detección de idioma solo necesita un poco de texto. Devolver ``None``
        deja el «auto» como estaba (mejor eso que decir un idioma equivocado).
        """
        doc = self._open_pdf()
        try:
            total = doc.page_count
            if not self._plan.auto_language:
                return total, None
            for index in range(min(total, 3)):
                texto = (doc.load_page(index).get_text() or "").strip()
                detectado = dp.detect_language(texto)
                if detectado:
                    return total, detectado
            return total, None
        finally:
            doc.close()

    def _process_pdf(self, plan: dpl.PipelinePlan) -> str:
        """Procesa el PDF por lotes: FASE 0/1 por página y FASE 2/4/5 al final.

        Con capa de texto nativa, en ``ocr`` se devuelve tal cual (esa capa *es*
        la transcripción y no hace falta modelo); en ``resumen``/``tablas`` el
        texto es la *entrada* del análisis, así que también pasa por el modelo
        con su prompt de texto. Si no, los tres modos devolvían el mismo texto.

        Las páginas se analizan de ``parallel`` en ``parallel`` (hoy ``1``: ver
        ``MAX_PARALLEL_PAGES``), pero **leer** cada página y montar el resultado se
        hace en este hilo: PyMuPDF no es thread-safe y el orden de las páginas
        debe ser el del PDF.
        """
        image_prompt = plan.image_prompt()
        text_prompt = plan.text_prompt()
        doc = self._open_pdf()
        try:
            total = doc.page_count
            paginas: list[_PageResult] = []
            secciones: list[dpl.Section] = []
            for indices in _chunks(total, plan.parallel):
                etiquetas: list[str] = []
                tareas: list[Callable[[], _PageResult]] = []
                for index in indices:
                    self._ensure_not_cancelled()
                    etiqueta, tarea = self._page_task(
                        doc, index, plan, image_prompt, text_prompt
                    )
                    etiquetas.append(etiqueta)
                    tareas.append(self._with_progress(tarea, index + 1, total))
                for index, etiqueta, resultado in zip(
                    indices, etiquetas, self._run_parallel(tareas)
                ):
                    paginas.append(resultado)
                    secciones.append(
                        dpl.render_page(index + 1, etiqueta, resultado.body)
                    )
            # FASE 2 (resumen por apartados): cada sección ya es el resumen de su
            # página. FASE 4 y FASE 5: validación ligera (con las cifras de cada
            # página frente a su texto) y entrega única.
            extra = self._extra_tables(paginas, plan)
            pares = [(pagina.body, pagina.source) for pagina in paginas]
            report = dpl.validate(secciones, pairs=pares)
            return dpl.deliver(sections=secciones, extra_sections=extra, report=report)
        finally:
            doc.close()

    def _page_task(
        self,
        doc,
        index: int,
        plan: dpl.PipelinePlan,
        image_prompt: str,
        text_prompt: str | None,
    ) -> tuple[str, Callable[[], _PageResult]]:
        """Devuelve ``(ruta, tarea)`` de una página (FASE 0 + FASE 1).

        Aquí solo se *lee* la página; la tarea devuelta contiene la llamada al
        modelo, que es la parte lenta y la que se ejecuta en paralelo.
        """
        page = doc.load_page(index)
        text = (page.get_text() or "").strip()
        if plan.route(dp.has_native_text(text)) == dpl.ROUTE_TEXT:
            if text_prompt is None:
                # OCR: la capa de texto *es* la transcripción, no hay modelo.
                return dpl.ROUTE_TEXT, lambda: _PageResult(body=text, source=text)
            return dpl.ROUTE_TEXT, lambda: _PageResult(
                body=self._engine.analyze_text(text, text_prompt, plan.model),
                source=text,
            )
        png = self._rasterize_page(page)
        if plan.force_ocr or plan.tables_extra():
            # OCR pedido (o hacen falta las tablas): transcripción explícita y,
            # si toca, análisis sobre esa transcripción (dos fases de verdad).
            return dpl.ROUTE_OCR, lambda: self._transcribe_then_analyze(
                png, plan, text_prompt
            )
        # Atajo de siempre: sin OCR forzado, una página sin texto se resuelve en
        # una sola llamada con el prompt del objetivo.
        return dpl.ROUTE_OCR, lambda: _PageResult(
            body=self._analyze_rendered_page(png, image_prompt), source=""
        )

    def _transcribe_then_analyze(
        self, png: Path, plan: dpl.PipelinePlan, text_prompt: str | None
    ) -> _PageResult:
        """FASE 1 (OCR) y, si hay análisis, FASE 2 sobre la transcripción."""
        transcripcion = self._analyze_rendered_page(png, plan.ocr_prompt())
        if text_prompt is None:
            return _PageResult(body=transcripcion, source=transcripcion)
        return _PageResult(
            body=self._engine.analyze_text(transcripcion, text_prompt, plan.model),
            source=transcripcion,
        )

    def _extra_tables(
        self, paginas: list["_PageResult"], plan: dpl.PipelinePlan
    ) -> list[dpl.Section]:
        """FASE 1 (tablas que acompañan al resumen), sobre el texto de cada página.

        Reutiliza el texto ya extraído (capa nativa o transcripción) en vez de
        volver a rasterizar: la extracción de tablas es una consulta al modelo.
        """
        if not plan.tables_extra():
            return []
        prompt = plan.tables_text_prompt()
        secciones: list[dpl.Section] = []
        for numero, pagina in enumerate(paginas, start=1):
            self._ensure_not_cancelled()
            if not pagina.source.strip():
                continue
            tabla = self._engine.analyze_text(pagina.source, prompt, plan.model)
            secciones.append(dpl.render_tables(numero, tabla))
        return secciones

    def _with_progress(
        self, tarea: Callable[[], _T], numero: int, total: int
    ) -> Callable[[], _T]:
        """Envuelve una tarea de página para avisar del progreso al terminarla.

        Se emite una vez por página y desde la propia tarea. Antes se emitía una
        sola vez por *lote* (``indices.stop``): con varias páginas a la vez el
        widget sólo veía el final del lote y la barra saltaba de 0 a N sin mostrar
        avance (medido en la auditoría con ``parallel=2``: un único aviso «2/2»
        para las dos páginas).
        """

        def ejecutar() -> _T:
            resultado = tarea()
            self.page_progress.emit(numero, total)
            return resultado

        return ejecutar

    def _run_parallel(self, tareas: list[Callable[[], _T]]) -> list[_T]:
        """Ejecuta las tareas del lote a la vez y devuelve los resultados en orden.

        Con una sola tarea (o ``parallel=1``) no se abre pool: no aportaría nada
        y solo añadiría ruido.
        """
        if len(tareas) <= 1 or self._parallel <= 1:
            return [tarea() for tarea in tareas]
        with ThreadPoolExecutor(max_workers=self._parallel) as pool:
            return list(pool.map(lambda tarea: tarea(), tareas))

    def _ensure_not_cancelled(self) -> None:
        """Corta el trabajo si el usuario pidió cancelar."""
        if self.isInterruptionRequested():
            raise EngineError(CANCEL_MESSAGE)

    # ---- FASE 1/2/4/5: resumen global ------------------------------

    def _process_pdf_global(self, plan: dpl.PipelinePlan) -> str:
        """Resume el documento completo (en una llamada, o por bloques si no cabe).

        FASE 1: se recoge el texto de cada página (capa nativa u OCR, transcrita
        con el prompt de OCR porque aquí hacen falta las palabras, no un resumen
        de cada página). FASE 2: se normaliza y se resume el conjunto. La
        primera pasada va por lotes, como el modo página a página.
        """
        ocr_prompt = plan.ocr_prompt()
        doc = self._open_pdf()
        try:
            total = doc.page_count
            paginas: list[str] = []
            for indices in _chunks(total, plan.parallel):
                tareas: list[Callable[[], str]] = []
                for index in indices:
                    self._ensure_not_cancelled()
                    page = doc.load_page(index)
                    text = (page.get_text() or "").strip()
                    if dp.has_native_text(text) and not plan.force_ocr:
                        tarea: Callable[[], str] = lambda pagina=text: pagina
                    else:
                        png = self._rasterize_page(page)
                        tarea = lambda imagen=png: self._analyze_rendered_page(
                            imagen, ocr_prompt
                        )
                    tareas.append(self._with_progress(tarea, index + 1, total))
                paginas.extend(self._run_parallel(tareas))

            self._ensure_not_cancelled()
            normalizado = dpl.normalize_pages(paginas)
            bloques = dpl.summary_blocks(normalizado.text)
            ejecutivo, detalle = self._summarize_blocks(bloques, plan)
            extra = [
                self._tabla_de_texto(numero, texto, plan)
                for numero, texto in enumerate(paginas, start=1)
                if plan.tables_extra() and texto.strip()
            ]
            pares = [(ejecutivo, normalizado.text)]
            if detalle is not None:
                pares.append((detalle, normalizado.text))
            report = dpl.validate(
                summary=ejecutivo,
                truncated=normalizado.truncated,
                pairs=pares,
                blocks=len(bloques),
            )
            if detalle is None:
                return dpl.deliver(
                    summary=ejecutivo, extra_sections=extra, report=report
                )
            # Entrega en dos partes: el resumen ejecutivo arriba (breve) y el
            # detalle por apartado debajo (sin perder nada).
            return dpl.deliver(
                sections=[
                    dpl.Section(title=dp.EXECUTIVE_HEADING, body=ejecutivo),
                    dpl.Section(title=dp.DETAIL_HEADING, body=detalle),
                ],
                extra_sections=extra,
                report=report,
            )
        finally:
            doc.close()

    def _summarize_blocks(
        self, bloques: tuple[str, ...], plan: dpl.PipelinePlan
    ) -> tuple[str, str | None]:
        """FASE 2: devuelve ``(resumen ejecutivo, detalle por apartados)``.

        El detalle es ``None`` salvo en el estilo «dos partes». El ejecutivo va
        con su tope de palabras; el detalle, sin tope, para no perder nada.

        Un documento que no cabe en una llamada **no se recorta**: recortarlo
        pierde páginas enteras y, medido, el modelo llegaba a *inventarse* los
        valores de las páginas que faltaban (extendía la serie que veía). Se
        resume cada bloque y después se resume el conjunto; con 19 páginas esto
        son 3-4 llamadas en vez de 19.
        """
        if not bloques:
            return "", None
        tope = dp.max_words_for("\n\n".join(bloques))
        if len(bloques) == 1:
            # El tope de palabras del resumen ejecutivo se calcula del documento
            # (un quinto); con «un quinto» y sin número el modelo no se ajustaba.
            ejecutivo = self._engine.analyze_text(
                bloques[0], plan.global_prompt(tope), plan.model
            )
        else:
            prompt_bloque = plan.chunk_prompt(tope)
            parciales = self._run_parallel(
                [
                    lambda bloque=bloque: self._engine.analyze_text(
                        bloque, prompt_bloque, plan.model
                    )
                    for bloque in bloques
                ]
            )
            self._ensure_not_cancelled()
            ejecutivo = self._engine.analyze_text(
                "\n\n".join(parciales), plan.global_prompt(tope), plan.model
            )
        ejecutivo = self._comprimir_si_hace_falta(ejecutivo, plan, tope)
        if not plan.two_parts:
            return ejecutivo, None
        return ejecutivo, self._detalle_por_apartados(bloques, plan)

    def _detalle_por_apartados(
        self, bloques: tuple[str, ...], plan: dpl.PipelinePlan
    ) -> str:
        """Segunda mitad de la entrega: el detalle por apartado, sin tope.

        Es lo que permite que la primera mitad (el ejecutivo) sea breve sin que
        se pierda nada: el detalle va debajo.
        """
        prompt_global = plan.prompt_with(dp.STYLE_DETAILED)
        if len(bloques) == 1:
            return self._engine.analyze_text(bloques[0], prompt_global, plan.model)
        prompt_bloque = dp.chunk_prompt_for(
            plan.mode, plan.language, plan.closing, dp.STYLE_DETAILED
        )
        parciales = self._run_parallel(
            [
                lambda bloque=bloque: self._engine.analyze_text(
                    bloque, prompt_bloque, plan.model
                )
                for bloque in bloques
            ]
        )
        self._ensure_not_cancelled()
        return "\n\n".join(parciales)

    def _comprimir_si_hace_falta(
        self, resumen: str, plan: dpl.PipelinePlan, tope: int
    ) -> str:
        """Segundo pase del resumen ejecutivo: comprimirlo si se pasa del tope.

        Reescribe **el resumen** (no el documento), así que conserva las cifras
        ya seleccionadas. **Apagado por defecto**: medido, subía el resumen del
        54 % al 65 % del documento (el modelo vuelve a conservarlo todo) y
        doblaba el tiempo. Ver ``document_params.EXECUTIVE_COMPRESS``.
        """
        if not dp.excede_palabras(resumen, tope):
            return resumen
        self._ensure_not_cancelled()
        return self._engine.analyze_text(
            resumen, plan.compress_prompt(tope), plan.model
        )

    def _tabla_de_texto(
        self, numero: int, texto: str, plan: dpl.PipelinePlan
    ) -> dpl.Section:
        """Extrae las tablas de una página ya transcrita (acompañan al resumen)."""
        tabla = self._engine.analyze_text(texto, plan.tables_text_prompt(), plan.model)
        return dpl.render_tables(numero, tabla)

    # ---- PyMuPDF y temporales -------------------------------------

    def _rasterize_page(self, page) -> Path:
        """Rasteriza una página a un PNG temporal y devuelve su ruta.

        Va aparte de la llamada al modelo a propósito: esto toca PyMuPDF (que no
        es thread-safe) y debe quedarse en el hilo del worker.
        """
        png = page.get_pixmap(dpi=dp.PDF_DPI).tobytes("png")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fh:
            fh.write(png)
            return Path(fh.name)

    def _analyze_rendered_page(self, path: Path, prompt: str) -> str:
        """Pasa un PNG temporal al motor de visión y lo borra pase lo que pase."""
        try:
            return self._engine.analyze_image(path, prompt, self._plan.model)
        finally:
            path.unlink(missing_ok=True)


class DocumentWidget(TaskWidget):
    """Selector de archivo, modo, modelo y área de resultado del proceso."""

    def __init__(
        self, engine: Engine | None, config_manager=None, parent=None
    ) -> None:
        super().__init__(parent)
        self._engine = engine
        self._config_manager = config_manager
        self._prefs = prefs.PrefsStore(config_manager)
        self._worker: DocumentWorker | None = None
        self._model_task = None
        self._file_path: Path | None = None
        self._result_text = ""
        #: Aviso de cobertura de idiomas que se está mostrando (o None).
        self._language_hint: str | None = None
        self._build_ui()
        self._populate_models()
        self._restore_mode()
        self._restore_language()
        self._restore_scope()
        self._restore_detail()
        self._restore_closing()
        self._restore_force_ocr()
        self._restore_tables()
        self._update_summary_options()
        if engine is None:
            self._set_status("Ollama Vision no está configurado o deshabilitado.")
            self._process_button.setEnabled(False)

    def _build_ui(self) -> None:
        self._select_button = QPushButton("Seleccionar archivo...")
        self._preview_label = QLabel("Ningún archivo seleccionado")

        file_row = QHBoxLayout()
        file_row.addWidget(self._select_button)
        file_row.addWidget(self._preview_label, 1)

        self._mode = QComboBox()
        for mode_id, label in dp.MODE_LABELS.items():
            self._mode.addItem(label, mode_id)
        self._mode.currentIndexChanged.connect(self._on_mode_changed)

        self._model = QComboBox()
        self._model.currentIndexChanged.connect(self._on_model_changed)

        self._language_combo = QComboBox()
        self._language_combo.setToolTip(
            "Idioma del resumen y de las tablas. El OCR siempre respeta el idioma "
            "original del documento."
        )
        for language_id, label in dp.LANGUAGES.items():
            self._language_combo.addItem(label, language_id)
        self._language_combo.currentIndexChanged.connect(self._on_language_changed)

        self._scope_combo = QComboBox()
        self._scope_combo.setToolTip(
            "Alcance del resumen (solo aplica al modo Resumen).\n"
            "«Automático» resume el documento completo de una vez a partir de "
            f"{dp.AUTO_GLOBAL_MIN_PAGES} páginas: una llamada por página con un "
            "modelo de visión tarda ~50 s (19 páginas ≈ 16 min), mientras que el "
            "documento entero en una sola llamada tarda menos de un minuto.\n"
            "«Global» lo fuerza siempre y «Por página» mantiene un resumen por "
            "página."
        )
        for scope_id, scope_label in dp.SCOPE_LABELS.items():
            self._scope_combo.addItem(scope_label, scope_id)
        self._scope_combo.currentIndexChanged.connect(self._on_scope_changed)

        self._detail_combo = QComboBox()
        self._detail_combo.setToolTip(
            "Cuánto detalle dar al resumen (solo aplica al modo Resumen).\n"
            "«Ejecutivo» escribe un resumen breve (en torno a un quinto del "
            "documento): conserva las reglas y cifras que deciden algo y deja "
            "fuera el detalle secundario.\n"
            "«Por apartado» conserva todos los ítems y cifras; medido, ocupa "
            "casi lo mismo que el documento."
        )
        for detail_id, detail_label in dp.DETAIL_LABELS.items():
            self._detail_combo.addItem(detail_label, detail_id)
        self._detail_combo.currentIndexChanged.connect(self._on_detail_changed)

        self._closing_check = QCheckBox("Frase de cierre por apartado (→)")
        self._closing_check.setToolTip(
            "Añade una línea de cierre a cada apartado («→ …») para fijar la "
            "idea. Se pide que solo repita lo que ya dice ese apartado.\n"
            "Apagada, el resumen se limita a lo que dice el documento (mejor si "
            "es para un examen o para citarlo)."
        )
        self._closing_check.toggled.connect(self._on_closing_changed)

        self._force_ocr_check = QCheckBox("Forzar OCR (FASE 1)")
        self._force_ocr_check.setToolTip(
            "Transcribe las páginas con el modelo de visión aunque el PDF ya "
            "traiga capa de texto. Útil cuando ese texto es malo.\n"
            "Apagada, si la página ya tiene texto se usa tal cual (mucho más "
            "rápido) y solo se hace OCR en las que no lo tienen."
        )
        self._force_ocr_check.toggled.connect(self._on_force_ocr_changed)

        self._tables_check = QCheckBox("Extraer tablas además (FASE 1)")
        self._tables_check.setToolTip(
            "Añade un bloque de tablas por página a la entrega, además del "
            "resultado principal (transcripción o resumen).\n"
            "En el modo Extraer tablas no hace falta: ya es el objetivo."
        )
        self._tables_check.toggled.connect(self._on_tables_changed)

        form = QFormLayout()
        form.addRow("Modo:", self._mode)
        form.addRow("Modelo:", self._model)
        form.addRow("Idioma:", self._language_combo)
        form.addRow("Alcance:", self._scope_combo)
        form.addRow("Detalle:", self._detail_combo)
        form.addRow("", self._closing_check)
        form.addRow("Fases:", self._force_ocr_check)
        form.addRow("", self._tables_check)

        self._process_button = QPushButton("Procesar")
        self._cancel_button = QPushButton("Cancelar")
        self._cancel_button.setEnabled(False)
        self._save_button = QPushButton("Guardar como .txt")
        self._save_button.setEnabled(False)
        self._copy_button = QPushButton("Copiar al portapapeles")
        self._copy_button.setEnabled(False)

        buttons = QHBoxLayout()
        buttons.addWidget(self._process_button)
        buttons.addWidget(self._cancel_button)
        buttons.addWidget(self._save_button)
        buttons.addWidget(self._copy_button)

        self._progress = QProgressBar()
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._progress.setVisible(False)

        self._status = QLabel("")

        self._result_area = QPlainTextEdit()
        self._result_area.setReadOnly(True)
        self._result_area.setPlaceholderText("El resultado aparecerá aquí.")

        layout = QVBoxLayout(self)
        layout.addLayout(file_row)
        layout.addLayout(form)
        layout.addLayout(buttons)
        layout.addWidget(self._progress)
        layout.addWidget(self._status)
        layout.addWidget(self._result_area, 1)

        self._select_button.clicked.connect(self._select_file)
        self._process_button.clicked.connect(self._process)
        self._cancel_button.clicked.connect(self._cancel)
        self._save_button.clicked.connect(self._save_as)
        self._copy_button.clicked.connect(self._copy)

    # ---- Selección de archivo -------------------------------------

    def _select_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar documento", "", dp.FILE_FILTER
        )
        if not path:
            return
        self._file_path = Path(path)
        self._show_preview(self._file_path)
        self._reset_result()

    def _show_preview(self, path: Path) -> None:
        self._preview_label.setText(
            dp.describe_file(path.name, path.suffix, path.stat().st_size)
        )

    # ---- Modelos y persistencia -----------------------------------

    def _populate_models(self) -> None:
        """Lanza la carga de modelos de visión en segundo plano (M5)."""
        self._model.blockSignals(True)
        self._model.clear()
        self._model.blockSignals(False)
        if self._engine is None:
            self._add_model(dp.DEFAULT_MODEL)
            return
        self._model.setEnabled(False)
        self._model.addItem("Cargando modelos…")
        self._model_task = run_pool_task(
            self._engine.list_models,
            on_result=self._on_models,
            on_error=self._on_models_error,
        )

    def _on_models_error(self, message: str) -> None:
        logger.warning("No se pudieron listar los modelos de visión: %s", message)
        self._set_status(f"Ollama Vision no disponible: {message}")
        self._on_models([])

    def _add_model(self, model_id: str) -> None:
        """Añade un modelo al combo con su aviso de cobertura de idiomas."""
        self._model.addItem(model_id, model_id)
        aviso = (
            "Cobertura multilingüe amplia (incluye euskera)."
            if dp.supports_minority_languages(model_id)
            else dp.LANGUAGE_WARNING
        )
        self._model.setItemData(
            self._model.count() - 1, aviso, Qt.ItemDataRole.ToolTipRole
        )

    def _on_models(self, models: list) -> None:
        """Rellena el selector con los modelos recibidos (o con el default)."""
        models = list(models)
        if not models:
            models = [dp.DEFAULT_MODEL]
        self._model.blockSignals(True)
        self._model.clear()
        for model_id in models:
            self._add_model(model_id)
        last = self._prefs.get_str(prefs.DOCUMENT)
        if last in models:
            self._model.setCurrentIndex(models.index(last))
        elif dp.DEFAULT_MODEL in models:
            self._model.setCurrentIndex(models.index(dp.DEFAULT_MODEL))
        self._model.blockSignals(False)
        self._model.setEnabled(True)
        self._update_language_hint()

    def _update_language_hint(self) -> None:
        """Muestra o retira el aviso de idioma del modelo/modo actuales.

        Solo retira el texto de estado si es el propio aviso el que está puesto,
        para no pisar otros mensajes (errores, resultados...).
        """
        aviso = dp.language_warning(self._model.currentData(), self._mode.currentData())
        if aviso == self._language_hint:
            return
        if self._language_hint is not None and self._status.text() == self._language_hint:
            self._set_status("")
        self._language_hint = aviso
        if aviso is not None:
            self._set_status(aviso)

    def _restore_mode(self) -> None:
        last = self._prefs.get_str(prefs.DOCUMENT_MODE)
        if last is None:
            return
        idx = self._mode.findData(last)
        if idx >= 0:
            self._mode.blockSignals(True)
            self._mode.setCurrentIndex(idx)
            self._mode.blockSignals(False)

    def _restore_language(self) -> None:
        """Restaura el idioma de salida guardado (``auto`` si no hay nada)."""
        last = self._prefs.get_str(prefs.DOCUMENT_LANGUAGE)
        idx = self._language_combo.findData(last) if last else -1
        if idx < 0:
            idx = self._language_combo.findData(dp.DEFAULT_LANGUAGE)
        if idx >= 0:
            self._language_combo.blockSignals(True)
            self._language_combo.setCurrentIndex(idx)
            self._language_combo.blockSignals(False)

    def _restore_scope(self) -> None:
        """Restaura el alcance del resumen guardado (``auto`` si no hay nada).

        La preferencia era un booleano («1» global / «0» por página):
        :func:`services.document_params.normalize_scope` sigue leyendo esos
        valores heredados, así que no se pierde nada al actualizar.
        """
        scope = dp.normalize_scope(self._prefs.get_str(prefs.DOCUMENT_GLOBAL))
        idx = self._scope_combo.findData(scope)
        if idx < 0:
            return
        self._scope_combo.blockSignals(True)
        self._scope_combo.setCurrentIndex(idx)
        self._scope_combo.blockSignals(False)

    def _on_scope_changed(self) -> None:
        self._prefs.set(prefs.DOCUMENT_GLOBAL, self._scope_combo.currentData())

    def _restore_detail(self) -> None:
        """Restaura el detalle del resumen (ejecutivo si no se usó nunca)."""
        detail = dp.normalize_detail(self._prefs.get_str(prefs.DOCUMENT_DETAIL))
        idx = self._detail_combo.findData(detail)
        if idx < 0:
            return
        self._detail_combo.blockSignals(True)
        self._detail_combo.setCurrentIndex(idx)
        self._detail_combo.blockSignals(False)

    def _on_detail_changed(self) -> None:
        self._prefs.set(prefs.DOCUMENT_DETAIL, self._detail_combo.currentData())

    def _restore_closing(self) -> None:
        """Restaura la casilla de frase de cierre (apagada si no se usó nunca)."""
        if not self._prefs.get_bool(prefs.DOCUMENT_CLOSING, False):
            return
        self._closing_check.blockSignals(True)
        self._closing_check.setChecked(True)
        self._closing_check.blockSignals(False)

    def _on_closing_changed(self, checked: bool) -> None:
        self._prefs.set_bool(prefs.DOCUMENT_CLOSING, checked)

    def _restore_force_ocr(self) -> None:
        """Restaura la casilla de OCR forzado (apagada si no se usó nunca)."""
        if not self._prefs.get_bool(prefs.DOCUMENT_FORCE_OCR, False):
            return
        self._force_ocr_check.blockSignals(True)
        self._force_ocr_check.setChecked(True)
        self._force_ocr_check.blockSignals(False)

    def _on_force_ocr_changed(self, checked: bool) -> None:
        self._prefs.set_bool(prefs.DOCUMENT_FORCE_OCR, checked)

    def _restore_tables(self) -> None:
        """Restaura la casilla de tablas adicionales (apagada si no se usó nunca)."""
        if not self._prefs.get_bool(prefs.DOCUMENT_TABLES, False):
            return
        self._tables_check.blockSignals(True)
        self._tables_check.setChecked(True)
        self._tables_check.blockSignals(False)

    def _on_tables_changed(self, checked: bool) -> None:
        self._prefs.set_bool(prefs.DOCUMENT_TABLES, checked)

    def _update_summary_options(self, disponible: bool = True) -> None:
        """Qué casillas aplican a cada modo (y ninguna mientras se trabaja).

        - alcance del resumen y frase de cierre: solo en el modo Resumen;
        - OCR forzado (FASE 1): en cualquier modo;
        - tablas adicionales (FASE 1): en todos menos en Extraer tablas, donde
          ya son el objetivo y el bloque extra sería una repetición.
        """
        es_resumen = dp.supports_global(self._mode.currentData())
        es_tablas = self._mode.currentData() == "tablas"
        self._scope_combo.setEnabled(disponible and es_resumen)
        self._detail_combo.setEnabled(disponible and es_resumen)
        self._closing_check.setEnabled(disponible and es_resumen)
        self._force_ocr_check.setEnabled(disponible)
        self._tables_check.setEnabled(disponible and not es_tablas)

    def _on_language_changed(self) -> None:
        self._prefs.set(prefs.DOCUMENT_LANGUAGE, self._language_combo.currentData())

    def _on_mode_changed(self) -> None:
        self._prefs.set(prefs.DOCUMENT_MODE, self._mode.currentData())
        self._update_language_hint()
        self._update_summary_options()

    def _on_model_changed(self) -> None:
        self._prefs.set(prefs.DOCUMENT, self._model.currentData())
        self._update_language_hint()

    # ---- Procesamiento --------------------------------------------

    def _process(self) -> None:
        if self._engine is None:
            self._set_status("Error: no hay motor Ollama Vision configurado.")
            return
        if self._file_path is None:
            self._set_status("Selecciona un archivo primero.")
            return
        if self._worker is not None and self._worker.isRunning():
            return
        if not self._engine.is_available():
            self._set_status("Ollama Vision no está disponible. ¿Está arrancado Ollama?")
            return

        mode = self._mode.currentData()
        model = self._model.currentData() or None

        self._set_busy(True)
        self._reset_result()
        self._progress.setVisible(True)
        self._set_status("Procesando...")

        self._worker = DocumentWorker(
            self._engine,
            self._file_path,
            mode,
            model,
            self._language_combo.currentData(),
            scope=self._scope_combo.currentData(),
            detail=self._detail_combo.currentData(),
            closing=self._closing_check.isChecked(),
            force_ocr=self._force_ocr_check.isChecked(),
            extra_tables=self._tables_check.isChecked(),
            parent=self,
            scheduler=self._gpu_scheduler(),
            liberar_otros=self._gpu_liberar_otros,
        )
        self._worker.page_progress.connect(self._on_progress)
        self._worker.result.connect(self._on_result_ready)
        self._worker.failed.connect(self._on_error)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _on_progress(self, current: int, total: int) -> None:
        self._progress.setRange(0, total)
        self._progress.setValue(current)
        self._set_status(f"Procesando... Página {current} de {total}")

    def _on_result_ready(self, text: str) -> None:
        self._result_text = text
        self._result_area.setPlainText(text)
        self._progress.setRange(0, 1)
        self._progress.setValue(1)
        self._save_button.setEnabled(True)
        self._copy_button.setEnabled(True)
        self._set_status("Listo.")

    def _on_finished(self) -> None:
        self._progress.setVisible(False)
        super()._on_finished()

    def _cancel(self) -> None:
        if self._worker is None or not self._worker.isRunning():
            return
        self._worker.requestInterruption()
        self._set_status("Cancelando...")

    # ---- Resultado -------------------------------------------------

    def _reset_result(self) -> None:
        self._result_text = ""
        self._result_area.clear()
        self._save_button.setEnabled(False)
        self._copy_button.setEnabled(False)

    def _save_as(self) -> None:
        if not self._result_text:
            return
        output_dir = Path(__file__).resolve().parent.parent / "output"
        default = str(output_dir / "resultado.txt")
        target, _ = QFileDialog.getSaveFileName(
            self, "Guardar resultado como...", default, "Texto (*.txt)"
        )
        if not target:
            return
        try:
            Path(target).write_text(self._result_text, encoding="utf-8")
        except OSError as exc:
            QMessageBox.critical(self, "Error", f"No se pudo guardar el archivo: {exc}")
            return
        self._set_status(f"Guardado en: {target}")

    def _copy(self) -> None:
        if not self._result_text:
            return
        QApplication.clipboard().setText(self._result_text)
        self._set_status("Copiado al portapapeles.")

    # ---- Utilidades -----------------------------------------------

    def _busy_widgets(self) -> tuple:
        return (
            self._select_button,
            self._mode,
            self._model,
            self._scope_combo,
            self._detail_combo,
            self._closing_check,
            self._force_ocr_check,
            self._tables_check,
        )

    def _busy_only_widgets(self) -> tuple:
        return (self._cancel_button,)

    def _on_busy_changed(self, busy: bool) -> None:
        self._process_button.setEnabled(not busy and self._engine is not None)
        if busy:
            self._save_button.setEnabled(False)
            self._copy_button.setEnabled(False)
        self._update_summary_options(not busy)

    def set_engine(self, engine: Engine | None) -> None:
        self._engine = engine
        self._populate_models()
        if engine is None:
            self._set_status("Ollama Vision no está configurado o deshabilitado.")
        self._process_button.setEnabled(engine is not None)

