"""Smoke test de la GUI (opt-in mediante IMPERATOR_GUI_SMOKE=1).

Construye la MainWindow real en modo offscreen y comprueba que arranca sin
excepciones, con las 6 pestañas registradas y el mapa pestaña->motor correcto.

Se omite salvo que se cumplan las dos condiciones:
- PyQt6 está instalado.
- ``IMPERATOR_GUI_SMOKE=1`` (crear un ``QApplication`` requiere un plugin de
  plataforma que no existe en algunas sesiones sin escritorio; el CI lo activa).
"""

import importlib.util
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Backend sin pantalla (CI / servidores). Se puede sobrescribir desde fuera.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_HAS_QT = importlib.util.find_spec("PyQt6") is not None
_ENABLED = os.environ.get("IMPERATOR_GUI_SMOKE") == "1"


@unittest.skipUnless(
    _HAS_QT and _ENABLED,
    "GUI smoke test desactivado (requiere PyQt6 e IMPERATOR_GUI_SMOKE=1)",
)
class GuiSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from PyQt6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def test_main_window_builds_with_tabs(self) -> None:
        from PyQt6.QtWidgets import QTabWidget

        from ui.main_window import MainWindow

        window = MainWindow()
        try:
            self.assertEqual(window.windowTitle(), "Imperator")
            tabs = window.findChild(QTabWidget)
            self.assertIsNotNone(tabs)
            self.assertEqual(tabs.count(), 6)
            titles = [tabs.tabText(i) for i in range(tabs.count())]
            self.assertEqual(
                titles,
                ["Chat", "Documentos", "Imagen", "Vídeo", "Audio", "Ensamblador"],
            )
            # El mapa pestaña->motor se deriva de Task.engine_id (H4).
            self.assertEqual(window._tab_to_engine.get(0), "ollama")
            self.assertEqual(window._tab_to_engine.get(1), "ollama_vision")
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
