"""Arranque robusto ante un settings.json inválido o con BOM (C4).

Cubre las dos mitades del hallazgo: la lectura tolerante (``utf-8-sig``) y que
``MainWindow`` no muera después del splash cuando el JSON está corrupto.
"""

import json
import os
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from core.config_manager import (  # noqa: E402
    DEFAULT_SETTINGS,
    ConfigError,
    ConfigManager,
    deep_merge,
)

GUI_SMOKE = os.environ.get("IMPERATOR_GUI_SMOKE") == "1"


class SettingsRobustoTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.cm = ConfigManager(config_dir=str(self.dir))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _ventana(self):
        """``MainWindow`` sin construir la GUI: solo lo que usa el helper."""
        from ui.main_window import MainWindow

        win = MainWindow.__new__(MainWindow)
        win._engine_manager = mock.Mock()
        win._engine_manager.config_manager = self.cm
        return win

    def test_deep_merge_rellena_solo_lo_que_falta(self) -> None:
        merged = deep_merge({"a": 1, "b": {"c": 2, "d": 3}}, {"b": {"c": 9}})
        self.assertEqual(merged, {"a": 1, "b": {"c": 9, "d": 3}})

    def test_load_settings_con_bom_no_lanza(self) -> None:
        (self.dir / "settings.json").write_text(
            json.dumps({"log_level": "INFO", "rutas": {"logs": "logs"}}),
            encoding="utf-8-sig",
        )
        # El BOM ya no rompe la lectura (antes: ConfigError de JSON inválido).
        cargados = self.cm.load_settings()
        self.assertEqual(cargados["log_level"], "INFO")
        # Y la ventana lo fusiona con los defaults en vez de morir.
        merged = self._ventana()._load_settings_merged()
        self.assertEqual(merged["theme"], DEFAULT_SETTINGS["theme"])
        self.assertEqual(merged["rutas"]["logs"], "logs")

    def test_load_settings_json_invalido_no_lanza(self) -> None:
        (self.dir / "settings.json").write_text(
            '{"log_level": "INFO", ', encoding="utf-8"
        )
        # El ConfigManager sí avisa; el que no puede caerse es el arranque.
        with self.assertRaises(ConfigError):
            self.cm.load_settings()
        merged = self._ventana()._load_settings_merged()
        self.assertEqual(merged, deepcopy(DEFAULT_SETTINGS))

    @unittest.skipUnless(GUI_SMOKE, "GUI real: requiere IMPERATOR_GUI_SMOKE=1")
    def test_main_arranca_con_settings_invalido(self) -> None:
        """``main()`` termina sin excepción con un settings.json corrupto (A2+C4).

        ``MainWindow`` se sustituye por un doble a propósito: aquí se prueba el
        camino de arranque (settings, log, splash, icono, ``exec``), no la
        construcción de la ventana, que ya cubre ``test_gui_smoke``.
        """
        from PyQt6.QtWidgets import QApplication, QWidget

        # Solo puede existir una QApplication por proceso: si la suite ya creó
        # una, main() debe reutilizarla en vez de construir una segunda (eso
        # abortaba el proceso con 0xC0000005).
        existente = QApplication.instance()
        fabrica = (lambda *a, **k: existente) if existente is not None else QApplication
        (self.dir / "settings.json").write_text("{ esto no es json", encoding="utf-8")

        class _VentanaFalsa(QWidget):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__()

        import main as entrada
        from ui import main_window as mw

        with (
            mock.patch.object(entrada, "QApplication", fabrica),
            mock.patch.object(
                ConfigManager, "_file_path", lambda cm, name: self.dir / name
            ),
            mock.patch.object(QApplication, "exec", lambda *a, **k: 0),
            mock.patch.object(entrada, "_create_splash", return_value=None),
            # La plataforma offscreen da un HWND falso (1) y WM_SETICON sobre él
            # provoca un access violation nativo: el icono de la barra de tareas
            # no es lo que se prueba aquí.
            mock.patch.object(entrada, "_force_taskbar_icon"),
            mock.patch.object(mw, "MainWindow", _VentanaFalsa),
        ):
            codigo = entrada.main()

        self.assertEqual(codigo, 0)


if __name__ == "__main__":
    unittest.main()
