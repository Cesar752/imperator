"""C5.1: el estado del motor de chat no se comprueba en el hilo de la UI.

``is_available()`` hace red (hasta ~2 s por motor). La barra de estado debe
pintar «comprobando…» y volver en milisegundos; la comprobación real va al pool.
"""

import os
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QMainWindow  # noqa: E402

from ui import main_window as mw  # noqa: E402

#: Margen del criterio de la ronda: el slot debe volver en <100 ms.
LIMITE_MS = 100


class _MotorLento:
    """Motor cuyo ``is_available()`` tarda 1 s (como un Ollama que no responde)."""

    id = "ollama"
    nombre = "Ollama"

    def __init__(self) -> None:
        self.llamadas = 0

    def is_available(self) -> bool:
        self.llamadas += 1
        time.sleep(1.0)
        return True


class EstadoBarraTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _ventana(self):
        """``MainWindow`` sin construir las 11 pestañas: solo lo que usa la barra."""
        win = mw.MainWindow.__new__(mw.MainWindow)
        QMainWindow.__init__(win)
        win._engine_manager = mock.Mock()
        win._status_cache = {}
        win._status_pending = None
        return win

    def test_no_bloquea_el_hilo_de_ui(self) -> None:
        motor = _MotorLento()
        win = self._ventana()
        with mock.patch.object(
            mw.ce, "preferred_chat_engine", return_value=motor
        ), mock.patch.object(mw, "run_pool_task") as pool:
            inicio = time.monotonic()
            win._refresh_status_bar()
            transcurrido_ms = (time.monotonic() - inicio) * 1000

        self.assertLess(transcurrido_ms, LIMITE_MS, "el slot debe volver en <100 ms")
        self.assertEqual(motor.llamadas, 0, "is_available no se llama en el hilo de UI")
        self.assertIn("comprobando", win.statusBar().currentMessage())
        self.assertTrue(pool.called)
        self.assertEqual(win._status_pending, "ollama")

    def test_reutiliza_el_resultado_dentro_del_ttl(self) -> None:
        motor = _MotorLento()
        win = self._ventana()
        win._status_cache["ollama"] = (time.monotonic(), True)
        with mock.patch.object(
            mw.ce, "preferred_chat_engine", return_value=motor
        ), mock.patch.object(mw, "run_pool_task") as pool:
            win._refresh_status_bar()

        self.assertFalse(pool.called, "con caché fresca no se lanza otra comprobación")
        self.assertIn("conectado", win.statusBar().currentMessage())

    def test_el_resultado_del_pool_actualiza_barra_y_cache(self) -> None:
        motor = _MotorLento()
        win = self._ventana()
        win._status_pending = "ollama"
        with mock.patch.object(mw.ce, "preferred_chat_engine", return_value=motor):
            win._on_availability(True)

        self.assertIn("Ollama: conectado", win.statusBar().currentMessage())
        self.assertTrue(win._status_cache["ollama"][1])
        self.assertIsNone(win._status_pending)

    def test_un_resultado_viejo_no_pisa_el_motor_nuevo(self) -> None:
        motor = _MotorLento()
        motor.id = "deepseek"
        motor.nombre = "DeepSeek"
        win = self._ventana()
        win._status_pending = "ollama"  # se lanzó con el motor anterior
        with mock.patch.object(mw.ce, "preferred_chat_engine", return_value=motor):
            win._on_availability(True)

        self.assertNotIn("deepseek", win._status_cache)
        self.assertEqual(win.statusBar().currentMessage(), "")

    def test_error_de_red_deja_desconectado(self) -> None:
        win = self._ventana()
        win._status_pending = "ollama"
        win._on_availability_error("sin conexión")

        self.assertIn("desconectado", win.statusBar().currentMessage())
        self.assertIsNone(win._status_pending)


if __name__ == "__main__":
    unittest.main()
