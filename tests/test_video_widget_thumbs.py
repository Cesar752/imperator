"""C5.3: las miniaturas de escena se extraen en el pool, no en el hilo de UI.

Con un extractor que duerme, ``_refresh_thumbnails`` debe volver en milisegundos,
dejar un placeholder por escena y rellenarlo cuando llegue el fotograma.
"""

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtGui import QPixmap  # noqa: E402
from PyQt6.QtWidgets import QApplication, QTableWidgetItem  # noqa: E402

from core.config_manager import ConfigManager  # noqa: E402
from services import video_script as vs  # noqa: E402
from ui import video_widget as vw  # noqa: E402

#: Criterio de la ronda: el slot debe volver en <100 ms.
LIMITE_MS = 100


class _Proyecto:
    """Proyecto mínimo: carpeta y clips por fila."""

    def __init__(self, carpeta: Path, filas: int) -> None:
        self.directory = carpeta
        for row in range(filas):
            self.clip_path(row).write_bytes(b"clip falso")

    def clip_path(self, row: int) -> Path:
        return self.directory / f"clip_{row:02d}.mp4"


class MiniaturasTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.cm = ConfigManager(config_dir=self.dir)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _widget(self, filas: int = 2):
        """Widget real, pero sin que el constructor lance tareas de red."""
        with mock.patch.object(vw, "run_pool_task"):
            widget = vw.VideoWidget(None, self.cm)
        widget._script = vs.Script(title="prueba")
        widget._project = _Proyecto(self.dir, filas)
        # Rellenar la tabla emite itemChanged, que programa el recálculo de avisos
        # (necesita un guion real): con las señales bloqueadas no se dispara.
        widget._scene_table.blockSignals(True)
        widget._scene_table.setRowCount(filas)
        for row in range(filas):
            widget._scene_table.setItem(row, 0, QTableWidgetItem(f"escena {row}"))
        widget._scene_table.blockSignals(False)
        return widget

    def _png(self, nombre: str = "miniatura.png") -> Path:
        ruta = self.dir / nombre
        pixmap = QPixmap(48, 27)
        pixmap.fill(Qt.GlobalColor.red)
        pixmap.save(str(ruta))
        return ruta

    def test_refresh_no_bloquea_y_deja_placeholders(self) -> None:
        widget = self._widget(2)
        try:
            with mock.patch.object(vw, "run_pool_task") as pool:
                inicio = time.monotonic()
                widget._refresh_thumbnails()
                transcurrido_ms = (time.monotonic() - inicio) * 1000

            self.assertLess(transcurrido_ms, LIMITE_MS)
            self.assertEqual(pool.call_count, 2, "una extracción por escena sin caché")
            self.assertEqual(widget._thumb_pending, {0, 1})
            self.assertTrue(widget._scene_table.item(0, 0).icon().isNull())
        finally:
            widget.close()

    def test_miniatura_en_disco_se_pinta_sin_ffmpeg(self) -> None:
        widget = self._widget(1)
        try:
            self._png("miniatura_00.png")
            with mock.patch.object(vw, "run_pool_task") as pool:
                widget._refresh_thumbnails()

            self.assertFalse(pool.called, "con la miniatura en disco no se extrae nada")
            self.assertFalse(widget._scene_table.item(0, 0).icon().isNull())
        finally:
            widget.close()

    def test_el_callback_rellena_la_miniatura(self) -> None:
        widget = self._widget(1)
        try:
            with mock.patch.object(vw, "run_pool_task"):
                widget._refresh_thumbnails()
            self.assertEqual(widget._thumb_pending, {0})

            mini = self._png("miniatura_00.png")
            widget._on_thumbnail((0, str(mini)), None)

            self.assertFalse(widget._scene_table.item(0, 0).icon().isNull())
            self.assertEqual(widget._thumb_pending, set(), "la fila ya no está en vuelo")
        finally:
            widget.close()

    def test_no_se_pide_dos_veces_la_misma_fila(self) -> None:
        widget = self._widget(1)
        try:
            with mock.patch.object(vw, "run_pool_task") as pool:
                widget._refresh_thumbnails()
                widget._refresh_thumbnails()

            self.assertEqual(pool.call_count, 1, "la fila en vuelo no se vuelve a pedir")
        finally:
            widget.close()

    def test_un_proyecto_nuevo_reinicia_las_miniaturas(self) -> None:
        widget = self._widget(1)
        try:
            with mock.patch.object(vw, "run_pool_task"):
                widget._refresh_thumbnails()
            generacion = widget._generation

            widget.load_script(widget._script, None)

            self.assertGreater(widget._generation, generacion)
            self.assertEqual(widget._thumb_pending, set())
        finally:
            widget.close()


if __name__ == "__main__":
    unittest.main()
