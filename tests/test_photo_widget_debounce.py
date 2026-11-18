"""C5.5: los ajustes de foto se reprocesan con debounce y en el pool.

Mover un control dispara ``valueChanged`` por cada paso; reprocesar a resolución
completa en cada uno congelaba la ventana. Con debounce, N cambios seguidos
producen UNA sola tarea, y un resultado viejo no pisa al nuevo.
"""

import os
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from ui import photo_widget as pw  # noqa: E402

#: Criterio de la ronda: cada tick debe volver en <100 ms.
LIMITE_MS = 100


class AjustesDebounceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.widget = pw.PhotoWidget()
        self.widget._original = object()  # cualquier cosa no-None sirve aqui
        self.widget._render_previews = mock.Mock()

    def tearDown(self) -> None:
        # Un QTimer armado sigue vivo aunque se cierre el widget: si no se para,
        # dispara durante el test siguiente y descuadra el recuento.
        self.widget._adjust_timer.stop()
        self.widget.close()
        self.widget.deleteLater()
        self.app.processEvents()

    def _ticks(self, veces: int, pausa_s: float = 0.01) -> float:
        """Simula varios pasos seguidos del control y devuelve el tiempo total."""
        inicio = time.monotonic()
        for _ in range(veces):
            self.widget._on_adjust_changed()
            self.app.processEvents()
            time.sleep(pausa_s)
        return (time.monotonic() - inicio) * 1000

    def _esperar(self, segundos: float) -> None:
        fin = time.monotonic() + segundos
        while time.monotonic() < fin:
            self.app.processEvents()
            time.sleep(0.01)

    def test_cada_tick_vuelve_rapido(self) -> None:
        with mock.patch.object(pw, "run_pool_task") as pool:
            transcurrido_ms = self._ticks(5, pausa_s=0.005)

        self.assertLess(transcurrido_ms, 5 * LIMITE_MS)
        self.assertFalse(pool.called, "el pool solo se lanza tras el debounce")

    def test_varios_ticks_producen_una_sola_tarea(self) -> None:
        with mock.patch.object(pw, "run_pool_task") as pool:
            inicio = time.monotonic()
            self._ticks(8, pausa_s=0.0)
            duracion_ms = (time.monotonic() - inicio) * 1000
            self._esperar(0.5)

        self.assertLess(duracion_ms, pw.ADJUST_DEBOUNCE_MS, "los ticks caben en el debounce")
        self.assertEqual(pool.call_count, 1, "8 cambios seguidos = 1 reproceso")

    def test_el_resultado_actualiza_la_imagen_y_repinta(self) -> None:
        imagen = object()
        self.widget._on_adjusted(None, imagen)

        self.assertIs(self.widget._result, imagen)
        self.widget._render_previews.assert_called_once()

    def test_un_resultado_viejo_no_pisa_al_nuevo(self) -> None:
        with mock.patch.object(pw, "run_pool_task"):
            self.widget._on_adjust_changed()
            self._esperar(0.45)  # deja que el debounce lance la tarea
            rx = self.widget._adjust_rx
            self.assertIsNotNone(rx)

            # Llega otro cambio: el resultado anterior deja de valer.
            self.widget._on_adjust_changed()
            viejo = object()
            rx.handle(viejo)

        self.assertIsNot(self.widget._result, viejo)


if __name__ == "__main__":
    unittest.main()
