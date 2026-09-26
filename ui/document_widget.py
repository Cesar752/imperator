"""Widget de documentos: OCR, resumen y extracción de tablas con Ollama Vision."""

import logging
import tempfile
from pathlib import Path

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
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
    QWidget,
)

from engines.base_engine import Engine, EngineError
from services import document_params as dp
from services import preferences as prefs
from ui.workers import ThreadWorker, run_pool_task

logger = logging.getLogger(__name__)

# Import diferido y protegido: si PyMuPDF no está instalado, la app sigue
# arrancando (chat/imagen/vídeo no se rompen); solo fallará al procesar un PDF.
try:
    import pymupdf
except ImportError:  # pragma: no cover - dependencia declarada en requirements.txt
    pymupdf = None


class DocumentWorker(ThreadWorker):
    """Procesa una imagen o un PDF (página a página) fuera del hilo de la UI."""

    page_progress = pyqtSignal(int, int)   # página actual, total de páginas

    def __init__(self, engine, file_path, mode, model, parent=None) -> None:
        super().__init__(parent)
        self._engine = engine
        self._file_path = Path(file_path)
        self._mode = mode
        self._model = model

    def work(self) -> str:
        prompt = dp.prompt_for(self._mode)
        if dp.is_pdf(self._file_path.suffix):
            return self._process_pdf(prompt)
        return self._process_image(prompt)

    def _process_image(self, prompt: str) -> str:
        self.page_progress.emit(1, 1)
        return self._engine.analyze_image(self._file_path, prompt, self._model)

    def _process_pdf(self, prompt: str) -> str:
        if pymupdf is None:
            raise EngineError(
                "PyMuPDF no está instalado. Instálalo con: pip install PyMuPDF>=1.24.0"
            )
        doc = pymupdf.open(str(self._file_path))
        try:
            total = doc.page_count
            blocks: list[str] = []
            for index, page in enumerate(doc):
                if self.isInterruptionRequested():
                    raise EngineError("Cancelado por el usuario")
                self.page_progress.emit(index + 1, total)
                text = (page.get_text() or "").strip()
                if dp.has_native_text(text):
                    blocks.append(f"--- Página {index + 1} (texto) ---\n\n{text}")
                else:
                    ocr = self._ocr_page(page, prompt)
                    blocks.append(f"--- Página {index + 1} (OCR) ---\n\n{ocr}")
            return "\n\n".join(blocks)
        finally:
            doc.close()

    def _ocr_page(self, page, prompt: str) -> str:
        """Rasteriza una página a PNG (200 DPI) y la pasa al motor de visión."""
        png = page.get_pixmap(dpi=dp.PDF_DPI).tobytes("png")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fh:
            fh.write(png)
            tmp_path = Path(fh.name)
        try:
            return self._engine.analyze_image(tmp_path, prompt, self._model)
        finally:
            tmp_path.unlink(missing_ok=True)


class DocumentWidget(QWidget):
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
        self._build_ui()
        self._populate_models()
        self._restore_mode()
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

        form = QFormLayout()
        form.addRow("Modo:", self._mode)
        form.addRow("Modelo:", self._model)

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
            self._model.addItem(dp.DEFAULT_MODEL, dp.DEFAULT_MODEL)
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

    def _on_models(self, models: list) -> None:
        """Rellena el selector con los modelos recibidos (o con el default)."""
        models = list(models)
        if not models:
            models = [dp.DEFAULT_MODEL]
        self._model.blockSignals(True)
        self._model.clear()
        for model_id in models:
            self._model.addItem(model_id, model_id)
        last = self._prefs.get_str(prefs.DOCUMENT)
        if last in models:
            self._model.setCurrentIndex(models.index(last))
        elif dp.DEFAULT_MODEL in models:
            self._model.setCurrentIndex(models.index(dp.DEFAULT_MODEL))
        self._model.blockSignals(False)
        self._model.setEnabled(True)

    def _restore_mode(self) -> None:
        last = self._prefs.get_str(prefs.DOCUMENT_MODE)
        if last is None:
            return
        idx = self._mode.findData(last)
        if idx >= 0:
            self._mode.blockSignals(True)
            self._mode.setCurrentIndex(idx)
            self._mode.blockSignals(False)

    def _on_mode_changed(self) -> None:
        self._prefs.set(prefs.DOCUMENT_MODE, self._mode.currentData())

    def _on_model_changed(self) -> None:
        self._prefs.set(prefs.DOCUMENT, self._model.currentData())

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

        self._worker = DocumentWorker(self._engine, self._file_path, mode, model, self)
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

    def _on_error(self, message: str) -> None:
        if message == "Cancelado por el usuario":
            self._set_status(message)
        else:
            self._set_status(f"Error: {message}")

    def _on_finished(self) -> None:
        self._set_busy(False)
        self._progress.setVisible(False)
        self._worker.deleteLater()
        self._worker = None

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

    def _set_busy(self, busy: bool) -> None:
        self._select_button.setEnabled(not busy)
        self._mode.setEnabled(not busy)
        self._model.setEnabled(not busy)
        self._process_button.setEnabled(not busy and self._engine is not None)
        self._cancel_button.setEnabled(busy)
        if busy:
            self._save_button.setEnabled(False)
            self._copy_button.setEnabled(False)

    def _set_status(self, text: str) -> None:
        self._status.setText(text)

    def set_engine(self, engine: Engine | None) -> None:
        self._engine = engine
        self._populate_models()
        if engine is None:
            self._set_status("Ollama Vision no está configurado o deshabilitado.")
        self._process_button.setEnabled(engine is not None)

