"""Retención del PoolTask: la tarea no puede morir mientras su hilo corre.

``run_pool_task`` devuelve el ``PoolTask``, pero los llamadores no suelen
guardarlo (``run_pool_task(FrameCache().prune, ...)``). Si el GC se lo llevaba a
mitad de ``run()``, su ``_PoolSignals`` se destruía y la emisión reventaba con
``RuntimeError: wrapped C/C++ object of type _PoolSignals has been deleted`` en
un hilo del pool, donde el ``excepthook`` de la aplicación no llega.
"""

import gc
import os
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QCoreApplication  # noqa: E402

from ui import workers  # noqa: E402

#: Margen para que el hilo del pool termine y entregue la señal.
ESPERA_S = 5.0


class PoolTaskLifetimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QCoreApplication.instance() or QCoreApplication([])

    def tearDown(self) -> None:
        # Ninguna tarea debe quedar retenida entre tests.
        self.assertEqual(workers._TAREAS_VIVAS, set())

    def _esperar(self, condicion, segundos: float = ESPERA_S) -> bool:
        fin = time.monotonic() + segundos
        while time.monotonic() < fin:
            self.app.processEvents()
            if condicion():
                return True
            time.sleep(0.01)
        return False

    def test_la_senal_llega_aunque_el_gc_pase_a_mitad(self) -> None:
        recibidos: list = []

        def lento() -> str:
            time.sleep(0.15)
            return "listo"

        # No se guarda la referencia a propósito: es el caso de los llamadores
        # reales (un `run_pool_task(...)` suelto).
        workers.run_pool_task(lento, on_result=recibidos.append)
        gc.collect()  # fuerza la recolección mientras el hilo sigue

        self.assertTrue(self._esperar(lambda: bool(recibidos)), "la señal no llegó")
        self.assertEqual(recibidos, ["listo"])

    def test_la_tarea_se_retiene_mientras_corre(self) -> None:
        def lento() -> str:
            time.sleep(0.2)
            return "ok"

        workers.run_pool_task(lento)
        self.assertEqual(len(workers._TAREAS_VIVAS), 1)
        self.assertTrue(self._esperar(lambda: not workers._TAREAS_VIVAS))

    def test_el_registro_se_limpia_tras_terminar(self) -> None:
        recibidos: list = []

        workers.run_pool_task(lambda: 42, on_result=recibidos.append)
        self.assertTrue(self._esperar(lambda: bool(recibidos)))

        self.assertEqual(workers._TAREAS_VIVAS, set(), "el registro no puede crecer")

    def test_una_excepcion_de_la_tarea_tambien_libera(self) -> None:
        errores: list = []

        def falla() -> None:
            raise RuntimeError("boom")

        workers.run_pool_task(falla, on_error=errores.append)

        self.assertTrue(self._esperar(lambda: bool(errores)))
        self.assertIn("boom", errores[0])
        self.assertEqual(workers._TAREAS_VIVAS, set())


if __name__ == "__main__":
    unittest.main()
