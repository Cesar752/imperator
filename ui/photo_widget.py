"""Pestaña **Retoque**: mejora y conversión de fotos.

- Abre imágenes en **cualquier formato** y las muestra (antes/después).
- **Ajustes clásicos** (brillo, contraste, color, nitidez) con Pillow: funcionan
  siempre, sin instalar nada.
- **Mejora con IA** (Real-ESRGAN + GFPGAN opcional) por **subprocess** en
  ``venv-photo``: escala y quita ruido, y restaura caras si se pide.
- **Guardar como** en PNG/JPEG/WEBP/TIFF/BMP.

La lógica pura (ajustes y conversión) vive en :mod:`services.photo_params`; aquí
solo se ejecuta el subprocess y se pinta.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path

from PyQt6 import sip
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.paths import PROJECT_ROOT
from core.subprocess_utils import no_window_kwargs
from engines.base_engine import EngineError
from services import photo_params as pp
from ui.pool_callbacks import PoolCallbackReceiver
from ui.task_widget import TaskWidget
from ui.workers import ThreadWorker, run_pool_task

logger = logging.getLogger(__name__)

#: Intérprete del venv de IA y rutas del worker/modelos.
PHOTO_PYTHON = "venv-photo/Scripts/python.exe"
PHOTO_WORKER = "tools/photo_worker.py"
PHOTO_MODELS_DIR = "tools/models"

#: Timeout de la mejora con IA (segundos): primera vez descarga pesos y en CPU
#: puede tardar; finito para no colgar el hilo del worker para siempre.
PHOTO_TIMEOUT = 1800

#: Tamaño máximo de las vistas previas (px).
PREVIEW_MAX = 320

#: Espera tras el último cambio de un control antes de reprocesar (C5.5).
ADJUST_DEBOUNCE_MS = 350


class PhotoWorker(ThreadWorker):
    """Mejora una imagen con Real-ESRGAN/GFPGAN (subprocess en venv-photo)."""

    def __init__(
        self,
        *,
        python: Path,
        worker: Path,
        input_path: Path,
        output_path: Path,
        scale: int,
        face: bool,
        model_dir: Path,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._python = python
        self._worker = worker
        self._input = input_path
        self._output = output_path
        self._scale = scale
        self._face = face
        self._model_dir = model_dir

    def work(self) -> Path:
        argv = [
            str(self._python), str(self._worker),
            "--input", str(self._input),
            "--output", str(self._output),
            "--scale", str(self._scale),
            "--model-dir", str(self._model_dir),
        ]
        if self._face:
            argv.append("--face")
        self.progress.emit(5, "Mejorando la foto con IA… (puede tardar)")
        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=PHOTO_TIMEOUT,
                **no_window_kwargs(),
            )
        except subprocess.TimeoutExpired as exc:
            raise EngineError(f"La mejora con IA no terminó en {PHOTO_TIMEOUT}s.") from exc
        except OSError as exc:
            raise EngineError(f"No se pudo ejecutar el worker de fotos: {exc}") from exc

        detalle = self._result(proc.stdout)
        if proc.returncode != 0 or not detalle.get("ok", False):
            motivo = detalle.get("error") or (proc.stderr or "").strip()[-500:]
            raise EngineError(f"Mejora con IA fallida. {motivo}")
        if not self._output.is_file() or self._output.stat().st_size == 0:
            raise EngineError("La IA no generó la imagen de salida.")
        self.progress.emit(100, "Foto mejorada.")
        return self._output

    @staticmethod
    def _result(stdout: str) -> dict:
        for linea in reversed((stdout or "").splitlines()):
            linea = linea.strip()
            if linea.startswith("{"):
                try:
                    data = json.loads(linea)
                except ValueError:
                    continue
                if isinstance(data, dict):
                    return data
        return {}


class PhotoWidget(TaskWidget):
    """Abre, ajusta, mejora con IA y guarda fotos en cualquier formato."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._original = None  # PIL.Image de la foto abierta
        self._result = None    # PIL.Image a guardar (ajustes y/o IA)
        self._source_path: Path | None = None
        #: Token que descarta reprocesos de un ajuste anterior (C5.5).
        self._generation = 0
        #: Debounce: un solo reproceso cuando el usuario deja de mover el control.
        self._adjust_timer = QTimer(self)
        self._adjust_timer.setSingleShot(True)
        self._adjust_timer.setInterval(ADJUST_DEBOUNCE_MS)
        self._adjust_timer.timeout.connect(self._apply_adjustments)
        #: Receptor del reproceso en curso.
        self._adjust_rx: PoolCallbackReceiver | None = None
        self._build_ui()
        self._refresh_ai_availability()

    # ---- UI --------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        fila = QHBoxLayout()
        self._btn_open = QPushButton("Abrir foto…")
        self._btn_open.clicked.connect(self._open)
        self._lbl_info = QLabel("(sin foto)")
        fila.addWidget(self._btn_open)
        fila.addWidget(self._lbl_info, 1)
        root.addLayout(fila)

        vistas = QHBoxLayout()
        self._lbl_before = QLabel("Antes")
        self._lbl_after = QLabel("Después")
        for lbl in (self._lbl_before, self._lbl_after):
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setMinimumSize(PREVIEW_MAX, PREVIEW_MAX)
            lbl.setStyleSheet("border: 1px solid #555;")
            vistas.addWidget(lbl)
        root.addLayout(vistas)

        ajustes = QFormLayout()
        self._spn_brightness = self._adjust_spin()
        self._spn_contrast = self._adjust_spin()
        self._spn_color = self._adjust_spin()
        self._spn_sharpness = self._adjust_spin()
        ajustes.addRow("Brillo:", self._spn_brightness)
        ajustes.addRow("Contraste:", self._spn_contrast)
        ajustes.addRow("Color:", self._spn_color)
        ajustes.addRow("Nitidez:", self._spn_sharpness)
        root.addLayout(ajustes)

        self._btn_reset = QPushButton("Restablecer ajustes")
        self._btn_reset.clicked.connect(self._reset_adjustments)
        root.addWidget(self._btn_reset)

        ia = QHBoxLayout()
        ia.addWidget(QLabel("Mejora IA:"))
        self._cmb_scale = QComboBox()
        for escala in pp.SCALES:
            self._cmb_scale.addItem(f"x{escala}", escala)
        idx = self._cmb_scale.findData(pp.DEFAULT_SCALE)
        if idx >= 0:
            self._cmb_scale.setCurrentIndex(idx)
        ia.addWidget(self._cmb_scale)
        self._chk_face = QCheckBox("Restaurar caras (GFPGAN)")
        ia.addWidget(self._chk_face)
        self._btn_enhance = QPushButton("Mejorar con IA")
        self._btn_enhance.clicked.connect(self._enhance)
        ia.addWidget(self._btn_enhance)
        ia.addStretch(1)
        root.addLayout(ia)

        salida = QHBoxLayout()
        salida.addWidget(QLabel("Formato:"))
        self._cmb_format = QComboBox()
        for fmt in pp.OUTPUT_FORMATS:
            self._cmb_format.addItem(fmt, fmt)
        idx = self._cmb_format.findData(pp.DEFAULT_FORMAT)
        if idx >= 0:
            self._cmb_format.setCurrentIndex(idx)
        salida.addWidget(self._cmb_format)
        salida.addWidget(QLabel("Calidad:"))
        self._spn_quality = QSpinBox()
        self._spn_quality.setRange(1, 100)
        self._spn_quality.setValue(95)
        salida.addWidget(self._spn_quality)
        self._btn_save = QPushButton("Guardar como…")
        self._btn_save.clicked.connect(self._save)
        salida.addWidget(self._btn_save)
        self._btn_cancel = QPushButton("Cancelar")
        self._btn_cancel.clicked.connect(self._cancel)
        salida.addWidget(self._btn_cancel)
        salida.addStretch(1)
        root.addLayout(salida)

        self._status = QLabel("")
        root.addWidget(self._status)
        root.addStretch(1)

        for spn in (
            self._spn_brightness, self._spn_contrast,
            self._spn_color, self._spn_sharpness,
        ):
            spn.valueChanged.connect(self._on_adjust_changed)
        self._btn_cancel.setEnabled(False)

    def _adjust_spin(self) -> QDoubleSpinBox:
        spn = QDoubleSpinBox()
        spn.setRange(pp.ADJUST_MIN, pp.ADJUST_MAX)
        spn.setSingleStep(0.1)
        spn.setValue(pp.ADJUST_DEFAULT)
        return spn

    def _busy_widgets(self) -> tuple:
        return (
            self._btn_open, self._btn_enhance, self._btn_save, self._btn_reset,
            self._cmb_scale, self._chk_face, self._cmb_format, self._spn_quality,
            self._spn_brightness, self._spn_contrast, self._spn_color, self._spn_sharpness,
        )

    def _busy_only_widgets(self) -> tuple:
        return (self._btn_cancel,)

    # ---- Disponibilidad de la IA -----------------------------------

    def _ai_paths(self) -> tuple[Path, Path] | None:
        python = PROJECT_ROOT / PHOTO_PYTHON
        worker = PROJECT_ROOT / PHOTO_WORKER
        if python.is_file() and worker.is_file():
            return python, worker
        return None

    def _refresh_ai_availability(self) -> None:
        disponible = self._ai_paths() is not None
        self._btn_enhance.setEnabled(disponible)
        if not disponible:
            self._btn_enhance.setToolTip(
                "Mejora IA no disponible: falta venv-photo o tools/photo_worker.py. "
                "Instálalo con: py -3.12 -m venv venv-photo && "
                "venv-photo\\Scripts\\python.exe -m pip install realesrgan gfpgan"
            )

    # ---- Acciones --------------------------------------------------

    def _open(self) -> None:
        ruta, _ = QFileDialog.getOpenFileName(
            self, "Abrir foto", "", "Imagen (*);;Todos (*)"
        )
        if not ruta:
            return
        try:
            imagen = pp.load_image(ruta)
        except Exception as exc:  # noqa: BLE001 - PIL/IO
            self._set_status(f"Error al abrir: {exc}")
            return
        self._source_path = Path(ruta)
        self._original = imagen
        self._result = imagen
        self._reset_adjustments()
        self._lbl_info.setText(f"{self._source_path.name} · {imagen.width}×{imagen.height}")
        self._render_previews()
        self._set_status("Foto cargada.")

    def _current_adjustments(self) -> pp.Adjustments:
        return pp.Adjustments(
            brightness=self._spn_brightness.value(),
            contrast=self._spn_contrast.value(),
            color=self._spn_color.value(),
            sharpness=self._spn_sharpness.value(),
        )

    def _on_adjust_changed(self) -> None:
        if self._original is None:
            return
        # El reproceso a resolución completa tarda cientos de ms y se dispara con
        # cada paso del control: se espera a que el usuario pare (C5.5). Mientras
        # tanto se sigue viendo la última versión válida.
        self._generation += 1
        self._adjust_timer.start()

    def _apply_adjustments(self) -> None:
        """Reprocesa los ajustes en el pool y repinta al volver (C5.5)."""
        if self._original is None:
            return
        ajustes = self._current_adjustments()
        rx = PoolCallbackReceiver(
            self, self._on_adjusted, self._on_adjust_error, context=ajustes
        )
        self._adjust_rx = rx
        rx.arm()
        run_pool_task(
            ajustes.apply,
            self._original,
            on_result=rx.handle,
            on_error=rx.handle_error,
        )

    def _on_adjusted(self, _ajustes, imagen) -> None:
        """Recibe la imagen reprocesada (los ajustes viejos ya no llegan)."""
        if sip.isdeleted(self):
            return
        self._result = imagen
        self._render_previews()

    def _on_adjust_error(self, _ajustes, mensaje: str) -> None:
        if sip.isdeleted(self):
            return
        logger.warning("No se pudieron aplicar los ajustes: %s", mensaje)

    def _reset_adjustments(self) -> None:
        for spn in (
            self._spn_brightness, self._spn_contrast,
            self._spn_color, self._spn_sharpness,
        ):
            spn.blockSignals(True)
            spn.setValue(pp.ADJUST_DEFAULT)
            spn.blockSignals(False)
        if self._original is not None:
            self._result = self._original
            self._render_previews()

    def _enhance(self) -> None:
        if self._original is None:
            self._set_status("Abre una foto primero.")
            return
        paths = self._ai_paths()
        if paths is None:
            self._set_status("Mejora IA no disponible (falta venv-photo).")
            return
        python, worker = paths
        base = self._result or self._original
        tmp = Path(tempfile.mkdtemp(prefix="imperator_photo_"))
        entrada = tmp / "entrada.png"
        salida = tmp / "salida.png"
        pp.save_image(base, entrada, "PNG")
        modelo_dir = PROJECT_ROOT / PHOTO_MODELS_DIR

        worker_obj = PhotoWorker(
            python=python,
            worker=worker,
            input_path=entrada,
            output_path=salida,
            scale=int(self._cmb_scale.currentData()),
            face=self._chk_face.isChecked(),
            model_dir=modelo_dir,
        )
        self._worker = worker_obj
        worker_obj.progress.connect(lambda _p, msg: self._set_status(msg))
        worker_obj.result.connect(self._on_enhanced)
        worker_obj.failed.connect(self._on_error)
        worker_obj.finished.connect(self._on_finished)
        self._set_busy(True)
        self._set_status("Mejorando la foto con IA…")
        worker_obj.start()

    def _on_enhanced(self, path) -> None:
        try:
            self._result = pp.load_image(path)
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Error al leer el resultado: {exc}")
            return
        self._render_previews()
        self._set_status("Mejora aplicada. Puedes ajustar o guardar.")

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.requestInterruption()

    def _save(self) -> None:
        if self._result is None:
            self._set_status("No hay nada que guardar.")
            return
        fmt = self._cmb_format.currentData()
        filtro = f"{fmt} (*.{fmt.lower()})"
        ruta, _ = QFileDialog.getSaveFileName(
            self, "Guardar imagen", f"retoque.{fmt.lower()}", filtro
        )
        if not ruta:
            return
        try:
            pp.save_image(self._result, ruta, fmt, self._spn_quality.value())
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Error al guardar: {exc}")
            return
        self._set_status(f"Guardado en {ruta}")

    # ---- Pintado ---------------------------------------------------

    def _render_previews(self) -> None:
        if self._original is not None:
            self._lbl_before.setPixmap(self._pixmap(self._original))
        if self._result is not None:
            self._lbl_after.setPixmap(self._pixmap(self._result))

    @staticmethod
    def _pixmap(imagen) -> QPixmap:
        import io

        buffer = io.BytesIO()
        imagen.convert("RGB").save(buffer, format="PNG")
        pixmap = QPixmap()
        pixmap.loadFromData(buffer.getvalue())
        return pixmap.scaled(
            PREVIEW_MAX, PREVIEW_MAX,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
