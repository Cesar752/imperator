"""C5.2/C5.2b: cambiar de pestaña y «Liberar VRAM ahora» no bloquean la UI.

``unload_model()`` y ``release_engines()`` hacen red; en el hilo de la UI eran
varios segundos con la ventana congelada. Ahora van al pool y el slot vuelve en
milisegundos.
"""

import os
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QCheckBox, QMainWindow  # noqa: E402

from ui import main_window as mw  # noqa: E402

#: Criterio de la ronda: el slot debe volver en <100 ms.
LIMITE_MS = 100


class _MotorLento:
    """Motor cuyo ``unload_model()`` tarda 1 s (como un Ollama que no responde)."""

    id = "ollama"
    nombre = "Ollama"

    def __init__(self) -> None:
        self.descargas = 0

    def unload_model(self) -> bool:
        self.descargas += 1
        time.sleep(1.0)
        return True


class VramTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _ventana(self, motor):
        """``MainWindow`` sin construir las 11 pestañas: solo lo que usa C5.2."""
        win = mw.MainWindow.__new__(mw.MainWindow)
        QMainWindow.__init__(win)
        win._engine_manager = mock.Mock()
        win._engine_manager.get_engine.return_value = motor
        win._current_tab_index = 0
        win._tab_to_engine = {0: "ollama", 1: "comfyui"}
        win._unload_toggle = QCheckBox()
        win._unload_toggle.setChecked(True)
        return win

    def test_cambiar_de_pestana_no_bloquea(self) -> None:
        motor = _MotorLento()
        win = self._ventana(motor)
        with mock.patch.object(mw, "run_pool_task") as pool:
            inicio = time.monotonic()
            win._on_tab_changed(1)
            transcurrido_ms = (time.monotonic() - inicio) * 1000

        self.assertLess(transcurrido_ms, LIMITE_MS)
        self.assertEqual(motor.descargas, 0, "unload_model no se llama en el hilo de UI")
        self.assertTrue(pool.called)
        self.assertEqual(win._current_tab_index, 1)

    def test_no_descarga_si_el_toggle_esta_apagado(self) -> None:
        motor = _MotorLento()
        win = self._ventana(motor)
        win._unload_toggle.setChecked(False)
        with mock.patch.object(mw, "run_pool_task") as pool:
            win._on_tab_changed(1)

        self.assertFalse(pool.called)
        self.assertEqual(motor.descargas, 0)

    def test_liberar_vram_ahora_no_bloquea(self) -> None:
        motor = _MotorLento()
        win = self._ventana(motor)
        with mock.patch.object(mw, "run_pool_task") as pool:
            inicio = time.monotonic()
            win._free_all_vram()
            transcurrido_ms = (time.monotonic() - inicio) * 1000

        self.assertLess(transcurrido_ms, LIMITE_MS)
        self.assertEqual(motor.descargas, 0)
        self.assertTrue(pool.called)
        self.assertIn("Liberando VRAM", win.statusBar().currentMessage())

    def test_el_resultado_del_pool_pinta_el_mensaje(self) -> None:
        win = self._ventana(_MotorLento())
        win._on_all_vram_freed(True)
        self.assertIn("VRAM liberada", win.statusBar().currentMessage())

        win._on_all_vram_freed(False)
        self.assertIn("No se pudo liberar VRAM", win.statusBar().currentMessage())

    def test_release_engines_sigue_siendo_sincrono_para_los_workers(self) -> None:
        """Los workers de ``gpu_slot`` necesitan el bool en el momento."""
        motor = _MotorLento()
        win = self._ventana(motor)
        win._engine_manager.supported_types.return_value = ["ollama"]
        motor.unload_model = mock.Mock(return_value=True)  # sin dormir

        self.assertTrue(win.release_engines())
        self.assertTrue(motor.unload_model.called)

    def test_unload_liberado_no_explota(self) -> None:
        self._ventana(_MotorLento())._on_vram_released(True)


if __name__ == "__main__":
    unittest.main()
