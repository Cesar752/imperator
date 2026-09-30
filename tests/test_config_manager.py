"""Tests de core.config_manager (validación, persistencia y secretos)."""

import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

# Permitir `import core...` al ejecutar desde la raíz o desde tests/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config_manager import (  # noqa: E402
    DEFAULT_ENGINES,
    ConfigError,
    ConfigManager,
)


class ConfigManagerDefaultsTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.cm = ConfigManager(config_dir=self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_creates_default_engines_file(self) -> None:
        engines = self.cm.load_engines()
        self.assertTrue((Path(self._tmp.name) / "engines.json").exists())
        ids = {e["id"] for e in engines}
        # Los defaults incluyen BFL (H3).
        self.assertIn("ollama", ids)
        self.assertIn("bfl", ids)
        self.assertEqual(len(engines), len(DEFAULT_ENGINES))

    def test_creates_default_settings_file(self) -> None:
        settings = self.cm.load_settings()
        self.assertEqual(settings["log_level"], "INFO")
        self.assertTrue((Path(self._tmp.name) / "settings.json").exists())


class ValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.cm = ConfigManager(config_dir=self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _engine(self, **overrides) -> dict:
        base = {
            "id": "ollama",
            "nombre": "Ollama",
            "ruta": "http://localhost:11434",
            "enabled": True,
            "parametros": {"model": "x"},
        }
        base.update(overrides)
        return base

    def test_missing_field_raises(self) -> None:
        bad = self._engine()
        del bad["nombre"]
        with self.assertRaises(ConfigError):
            self.cm.save_engines([bad])

    def test_wrong_type_raises(self) -> None:
        with self.assertRaises(ConfigError):
            self.cm.save_engines([self._engine(enabled="si")])

    def test_duplicate_id_case_insensitive_raises(self) -> None:
        with self.assertRaises(ConfigError):
            self.cm.save_engines(
                [self._engine(id="Ollama"), self._engine(id="ollama")]
            )

    def test_empty_required_string_raises(self) -> None:
        with self.assertRaises(ConfigError):
            self.cm.save_engines([self._engine(nombre="")])

    def test_json_corrupto_raises(self) -> None:
        (Path(self._tmp.name) / "engines.json").write_text("{no json", encoding="utf-8")
        with self.assertRaises(ConfigError):
            self.cm.load_engines()

    def test_roundtrip_engines(self) -> None:
        data = [self._engine()]
        self.cm.save_engines(data)
        self.assertEqual(self.cm.load_engines(), data)

    def test_atomic_dump_leaves_no_tmp(self) -> None:
        self.cm.save_settings(
            {
                "log_level": "DEBUG",
                "rutas": {},
                "last_models": {},
                "theme": "dark",
                "font_size": 11,
            }
        )
        leftovers = list(Path(self._tmp.name).glob("*.tmp"))
        self.assertEqual(leftovers, [])

    def test_last_models_roundtrip(self) -> None:
        self.cm.save_last_model("chat", "qwen3.5:9b")
        self.assertEqual(self.cm.load_last_models().get("chat"), "qwen3.5:9b")

    def test_update_settings_conserva_lo_que_no_cambia(self) -> None:
        """El read-modify-write va bajo lock: no puede perder otras claves.

        Antes cada llamante hacía ``load_settings()`` + modificar +
        ``save_settings()``; dos cambios a la vez se pisaban (uno de los dos
        desaparecía). Aquí se comprueba que un cambio no borra el resto.
        """
        self.cm.save_last_model("chat", "modelo-x")
        self.cm.update_settings({"theme": "light"})
        self.cm.update_settings({"font_size": 14})
        settings = self.cm.load_settings()
        self.assertEqual(settings["theme"], "light")
        self.assertEqual(settings["font_size"], 14)
        self.assertEqual(settings["last_models"], {"chat": "modelo-x"})
        self.assertEqual(settings["log_level"], "INFO")

    def test_guardar_last_models_no_pisa_otras_claves(self) -> None:
        self.cm.update_settings({"theme": "light"})
        self.cm.save_last_model("video", "i2v")
        settings = self.cm.load_settings()
        self.assertEqual(settings["theme"], "light")
        self.assertEqual(settings["last_models"], {"video": "i2v"})

    def test_escrituras_concurrentes_no_dejan_basura_ni_corrompen(self) -> None:
        """Varias escrituras a la vez: JSON válido, sin temporales huérfanos."""
        hilos = [
            threading.Thread(target=self.cm.save_last_model, args=(f"t{i}", str(i)))
            for i in range(8)
        ]
        for hilo in hilos:
            hilo.start()
        for hilo in hilos:
            hilo.join()
        self.assertEqual(list(Path(self._tmp.name).glob("*.tmp")), [])
        modelos = self.cm.load_last_models()
        self.assertEqual(len(modelos), 8)


class ApiKeyTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.cm = ConfigManager(config_dir=self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_reads_from_environment_not_json(self) -> None:
        api = {"id": "bfl", "key": "clave-del-json-en-claro"}
        with mock.patch.dict(os.environ, {"BFL_API_KEY": "clave-del-entorno"}):
            self.assertEqual(self.cm.resolve_api_key(api), "clave-del-entorno")

    def test_ignores_plaintext_json_key(self) -> None:
        api = {"id": "noexiste", "key": "clave-del-json-en-claro"}
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NOEXISTE_API_KEY", None)
            self.assertEqual(self.cm.resolve_api_key(api), "")

    def test_empty_id_returns_empty(self) -> None:
        self.assertEqual(self.cm.resolve_api_key({"key": "x"}), "")


if __name__ == "__main__":
    unittest.main()
