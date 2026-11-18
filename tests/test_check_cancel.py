"""``_check_cancel`` vive solo en ``ThreadWorker`` (una definición, no cinco).

Estaba reimplementado, idéntico, en cuatro workers de widget. La cancelación es
crítica: el test fija que nadie vuelva a copiarlo y que la versión de la base
sigue lanzando ``EngineError``.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from engines.base_engine import CANCEL_MESSAGE, EngineError  # noqa: E402
from ui import chronicle_widget as chw  # noqa: E402
from ui import understanding_widget as uw  # noqa: E402
from ui import video_enhance_widget as vew  # noqa: E402
from ui.workers import ThreadWorker  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

#: Ficheros que tenían su propia copia (ya no deben tenerla).
SIN_COPIAS = ("chronicle_widget.py", "understanding_widget.py", "video_enhance_widget.py")

#: Workers que usaban la copia y ahora heredan la de la base.
WORKERS = (
    chw.ChronicleWorker,
    chw.ActaWorker,
    uw.UnderstandingWorker,
    vew.AiUpscaleWorker,
)


class CheckCancelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_no_quedan_copias_en_los_widgets(self) -> None:
        for nombre in SIN_COPIAS:
            fuente = (ROOT / "ui" / nombre).read_text(encoding="utf-8")
            self.assertNotIn(
                "def _check_cancel",
                fuente,
                f"{nombre} debe usar la de ThreadWorker",
            )

    def test_solo_esta_definida_una_vez(self) -> None:
        definiciones = []
        for ruta in (ROOT / "ui").glob("*.py"):
            texto = ruta.read_text(encoding="utf-8")
            definiciones.extend([ruta.name] * texto.count("def _check_cancel"))

        self.assertEqual(definiciones, ["workers.py"])

    def test_los_workers_heredan_la_de_la_base(self) -> None:
        for worker in WORKERS:
            with self.subTest(worker=worker.__name__):
                self.assertIs(worker._check_cancel, ThreadWorker._check_cancel)

    def test_lanza_engine_error_si_se_pide_cancelar(self) -> None:
        worker = uw.UnderstandingWorker(Path("v.mp4"), mock.Mock(), extractor=mock.Mock())
        with mock.patch.object(worker, "isInterruptionRequested", return_value=True):
            with self.assertRaises(EngineError) as ctx:
                worker._check_cancel()
        self.assertEqual(str(ctx.exception), CANCEL_MESSAGE)

    def test_no_lanza_si_no_se_ha_cancelado(self) -> None:
        worker = uw.UnderstandingWorker(Path("v.mp4"), mock.Mock(), extractor=mock.Mock())
        with mock.patch.object(worker, "isInterruptionRequested", return_value=False):
            worker._check_cancel()  # no debe lanzar


if __name__ == "__main__":
    unittest.main()
