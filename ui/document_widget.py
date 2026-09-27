"""Widget de documentos: OCR, resumen y extracción de tablas con Ollama Vision."""

import logging
import tempfile
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

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
from services import preferences as prefs
from ui.task_widget import TaskWidget
from ui.workers import ThreadWorker, run_pool_task

logger = logging.getLogger(__name__)

# Import diferido y protegido: si PyMuPDF no está instalado, la app sigue
# arrancando (chat/imagen/vídeo no se rompen); solo fallará al procesar un PDF.
try:
    import pymupdf
except ImportError:  # pragma: no cover - dependencia declarada en requirements.txt
    pymupdf = None

#: Páginas que se analizan a la vez. Ollama reparte el contexto entre las
#: peticiones que atiende a la vez, así que esto va de la mano de
#: ``ollama_api.DEFAULT_VISION_NUM_CTX`` (32768 = 16384 por página).
#:
#: Medido con ``qwen3-vl:8b``, en tres escenarios:
#:
#: - peticiones crudas por HTTP, con dos slots disponibles: 140 s en serie frente
#:   a 83 s en paralelo (**1,68x**);
#: - ``DocumentWorker`` completo, con Ollama sin slots forzados (abrió uno):
#:   106 s en serie frente a 126 s en paralelo (**0,84x**: encoladas + sobrecarga);
#: - con ``OLLAMA_NUM_PARALLEL=2`` en el servidor: ver §5.8 del `todo.md`.
#:
#: De ahí la receta: forzar dos slots en el servidor (variable de entorno y
#: reiniciar Ollama) y pedir el doble de contexto, porque con un solo slot activo
#: el paralelismo solo añade sobrecarga.
MAX_PARALLEL_PAGES = 2


def _chunks(total: int, size: int) -> Iterator[range]:
    """Rangos de índices de página, de ``size`` en ``size`` (el último, más corto)."""
    for inicio in range(0, total, size):
        yield range(inicio, min(inicio + size, total))


class DocumentWorker(ThreadWorker):
    """Procesa una imagen o un PDF fuera del hilo de la UI."""

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
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._engine = engine
        self._file_path = Path(file_path)
        self._mode = mode
        self._model = model
        self._language = language
        self._global_summary = global_summary
        self._closing = closing
        self._parallel = max(1, int(parallel))

    def work(self) -> str:
        image_prompt = dp.prompt_for(self._mode, self._language, self._closing)
        if not dp.is_pdf(self._file_path.suffix):
            return self._process_image(image_prompt)
        if dp.wants_global(self._mode, self._global_summary):
            return self._process_pdf_global()
        return self._process_pdf(image_prompt)

    def _process_image(self, prompt: str) -> str:
        self.page_progress.emit(1, 1)
        return self._engine.analyze_image(self._file_path, prompt, self._model)

    def _open_pdf(self):
        """Abre el PDF (PyMuPDF es opcional: sin él la app sigue arrancando)."""
        if pymupdf is None:
            raise EngineError(
                "PyMuPDF no está instalado. Instálalo con: pip install PyMuPDF>=1.24.0"
            )
        return pymupdf.open(str(self._file_path))

    def _process_pdf(self, image_prompt: str) -> str:
        """Procesa el PDF por lotes de páginas según el modo elegido.

        Con capa de texto nativa, en ``ocr`` se devuelve tal cual (esa capa *es*
        la transcripción y no hace falta modelo); en ``resumen``/``tablas`` el
        texto es la *entrada* del análisis, así que también pasa por el modelo
        con su prompt de texto. Si no, los tres modos devolvían el mismo texto.

        Las páginas se analizan de ``_parallel`` en ``_parallel`` (Ollama atiende
        varias peticiones a la vez y la GPU se solapa), pero **leer** cada página
        y montar el resultado se hace en este hilo: PyMuPDF no es thread-safe y el
        orden de las páginas debe ser el del PDF.
        """
        text_prompt = dp.text_prompt_for(self._mode, self._language, self._closing)
        doc = self._open_pdf()
        try:
            total = doc.page_count
            blocks: list[str] = []
            for indices in _chunks(total, self._parallel):
                etiquetas: list[str] = []
                tareas: list[Callable[[], str]] = []
                for index in indices:
                    self._ensure_not_cancelled()
                    etiqueta, tarea = self._page_task(
                        doc, index, image_prompt, text_prompt
                    )
                    etiquetas.append(etiqueta)
                    tareas.append(tarea)
                for index, etiqueta, resultado in zip(
                    indices, etiquetas, self._run_parallel(tareas)
                ):
                    blocks.append(
                        f"--- Página {index + 1} ({etiqueta}) ---\n\n{resultado}"
                    )
                self.page_progress.emit(indices.stop, total)
            return "\n\n".join(blocks)
        finally:
            doc.close()

    def _page_task(
        self, doc, index: int, image_prompt: str, text_prompt: str | None
    ) -> tuple[str, Callable[[], str]]:
        """Devuelve ``(etiqueta, tarea)`` de una página.

        Aquí solo se *lee* la página; la tarea devuelta contiene la llamada al
        modelo, que es la parte lenta y la que se ejecuta en paralelo.
        """
        page = doc.load_page(index)
        text = (page.get_text() or "").strip()
        if dp.has_native_text(text):
            if text_prompt is None:
                # OCR: la capa de texto *es* la transcripción, no hay modelo.
                return "texto", lambda: text
            return "texto", lambda: self._engine.analyze_text(
                text, text_prompt, self._model
            )
        png = self._rasterize_page(page)
        return "OCR", lambda: self._analyze_rendered_page(png, image_prompt)

    def _run_parallel(self, tareas: list[Callable[[], str]]) -> list[str]:
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

    def _process_pdf_global(self) -> str:
        """Resume el documento completo con una sola llamada al modelo.

        Dos pasadas: primero se recoge el texto de cada página (capa nativa u
        OCR, transcrita con el prompt de OCR porque aquí hacen falta las
        palabras, no un resumen de cada página) y después se pide un único
        resumen global. Así el modelo ve el documento entero y no fragmentos.
        La primera pasada va por lotes, como el modo página a página.
        """
        ocr_prompt = dp.prompt_for("ocr")
        doc = self._open_pdf()
        try:
            total = doc.page_count
            pages: list[str] = []
            for indices in _chunks(total, self._parallel):
                tareas: list[Callable[[], str]] = []
                for index in indices:
                    self._ensure_not_cancelled()
                    page = doc.load_page(index)
                    text = (page.get_text() or "").strip()
                    if dp.has_native_text(text):
                        tareas.append(lambda pagina=text: pagina)
                    else:
                        png = self._rasterize_page(page)
                        tareas.append(
                            lambda imagen=png: self._analyze_rendered_page(
                                imagen, ocr_prompt
                            )
                        )
                pages.extend(self._run_parallel(tareas))
                self.page_progress.emit(indices.stop, total)

            self._ensure_not_cancelled()
            joined, truncated = dp.merge_pages(pages)
            summary = self._engine.analyze_text(
                joined,
                dp.global_prompt_for(self._mode, self._language, self._closing),
                self._model,
            )
            if truncated:
                summary = f"{summary}\n\n{dp.TRUNCATED_NOTICE}"
            return summary
        finally:
            doc.close()

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
            return self._engine.analyze_image(path, prompt, self._model)
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
        self._restore_global()
        self._restore_closing()
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

        self._global_check = QCheckBox("Resumen global (documento completo)")
        self._global_check.setToolTip(
            "Resume todas las páginas juntas en una sola consulta, en vez de un "
            "resumen por página. Solo se aplica al modo Resumen."
        )
        self._global_check.toggled.connect(self._on_global_changed)

        self._closing_check = QCheckBox("Frase de cierre por apartado (→)")
        self._closing_check.setToolTip(
            "Añade una línea de cierre a cada apartado («→ …») para fijar la "
            "idea. Se pide que solo repita lo que ya dice ese apartado.\n"
            "Apagada, el resumen se limita a lo que dice el documento (mejor si "
            "es para un examen o para citarlo)."
        )
        self._closing_check.toggled.connect(self._on_closing_changed)

        form = QFormLayout()
        form.addRow("Modo:", self._mode)
        form.addRow("Modelo:", self._model)
        form.addRow("Idioma:", self._language_combo)
        form.addRow("Alcance:", self._global_check)
        form.addRow("", self._closing_check)

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

    def _restore_global(self) -> None:
        """Restaura la casilla de resumen global (apagada si no se usó nunca)."""
        if not self._prefs.get_bool(prefs.DOCUMENT_GLOBAL, False):
            return
        self._global_check.blockSignals(True)
        self._global_check.setChecked(True)
        self._global_check.blockSignals(False)

    def _on_global_changed(self, checked: bool) -> None:
        self._prefs.set_bool(prefs.DOCUMENT_GLOBAL, checked)

    def _restore_closing(self) -> None:
        """Restaura la casilla de frase de cierre (apagada si no se usó nunca)."""
        if not self._prefs.get_bool(prefs.DOCUMENT_CLOSING, False):
            return
        self._closing_check.blockSignals(True)
        self._closing_check.setChecked(True)
        self._closing_check.blockSignals(False)

    def _on_closing_changed(self, checked: bool) -> None:
        self._prefs.set_bool(prefs.DOCUMENT_CLOSING, checked)

    def _update_summary_options(self, disponible: bool = True) -> None:
        """Las casillas del resumen solo aplican al modo Resumen (y no al trabajar)."""
        es_resumen = dp.supports_global(self._mode.currentData())
        self._global_check.setEnabled(disponible and es_resumen)
        self._closing_check.setEnabled(disponible and es_resumen)

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
            global_summary=self._global_check.isChecked(),
            closing=self._closing_check.isChecked(),
            parent=self,
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
            self._global_check,
            self._closing_check,
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

