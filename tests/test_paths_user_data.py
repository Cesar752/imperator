"""Datos de usuario en ``%APPDATA%\\Imperator`` (lo que necesita un .exe).

En una instalación normal la carpeta de la app puede ser de solo lectura, así que
modelos, voces, salidas, música y caché no pueden colgar de ``PROJECT_ROOT``.
"""

import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import paths  # noqa: E402


class UserDataTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        # Deja el módulo como estaba (APPDATA real) para el resto de la suite.
        importlib.reload(paths)
        self._tmp.cleanup()

    def _recargar(self):
        return importlib.reload(paths)

    def test_appdata_manda_sobre_las_rutas(self) -> None:
        with mock.patch.dict(os.environ, {"APPDATA": self._tmp.name}):
            recargado = self._recargar()
            esperado = Path(self._tmp.name) / "Imperator"
            self.assertEqual(recargado.USER_DATA_DIR, esperado)
            for nombre in ("MODELS", "VOICES", "OUTPUT", "MUSIC", "CACHE"):
                with self.subTest(ruta=nombre):
                    self.assertTrue(
                        getattr(recargado, nombre).is_relative_to(esperado)
                    )

    def test_ensure_dirs_crea_las_carpetas(self) -> None:
        with mock.patch.dict(os.environ, {"APPDATA": self._tmp.name}):
            recargado = self._recargar()
            recargado.ensure_dirs()
            for carpeta in recargado.DIRECTORIOS:
                self.assertTrue(carpeta.is_dir(), f"no se creó {carpeta}")

    def test_ensure_dirs_es_idempotente(self) -> None:
        with mock.patch.dict(os.environ, {"APPDATA": self._tmp.name}):
            recargado = self._recargar()
            recargado.ensure_dirs()
            recargado.ensure_dirs()  # no debe lanzar

    def test_los_assets_siguen_junto_al_codigo(self) -> None:
        self.assertEqual(paths.ASSETS, ROOT / "assets")
        self.assertNotIn("Imperator/assets", str(paths.ASSETS))

    def test_nadie_construye_la_ruta_de_modelos_a_mano(self) -> None:
        fuente = (ROOT / "tools" / "jarvis_bridge.py").read_text(encoding="utf-8")
        self.assertNotIn('PROJECT_ROOT / "tools" / "models"', fuente)

    def test_los_defaults_de_voces_son_absolutos(self) -> None:
        from core.config_manager import DEFAULT_ENGINES

        for engine_id in ("piper_tts", "voice_clone"):
            entrada = next(e for e in DEFAULT_ENGINES if e["id"] == engine_id)
            with self.subTest(engine=engine_id):
                self.assertTrue(Path(entrada["ruta"]).is_absolute(), entrada["ruta"])

        piper = next(e for e in DEFAULT_ENGINES if e["id"] == "piper_tts")
        self.assertTrue(Path(piper["parametros"]["voices_dir"]).is_absolute())


if __name__ == "__main__":
    unittest.main()
