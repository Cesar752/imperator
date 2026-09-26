"""Tests de core.api_manager (registro de APIs y claves desde entorno)."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.api_manager import ApiManager  # noqa: E402
from core.config_manager import ConfigManager  # noqa: E402


def _api(api_id: str, tipo: str = "llm", enabled: bool = False, key: str = "") -> dict:
    entry = {
        "id": api_id,
        "nombre": api_id.upper(),
        "enabled": enabled,
        "base_url": "https://example.invalid/v1",
        "tipo": tipo,
    }
    if key:
        entry["key"] = key
    return entry


class ApiManagerTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.cm = ConfigManager(config_dir=self._tmp.name)
        self.cm.save_apis(
            [
                _api("openai", "llm", enabled=True, key="clave-en-json"),
                _api("kling", "video"),
            ]
        )
        self.manager = ApiManager(self.cm)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_list_and_get(self) -> None:
        ids = {a["id"] for a in self.manager.list_apis()}
        self.assertEqual(ids, {"openai", "kling"})
        self.assertEqual(self.manager.get_api("openai")["nombre"], "OPENAI")
        self.assertIsNone(self.manager.get_api("noexiste"))

    def test_is_enabled(self) -> None:
        self.assertTrue(self.manager.is_enabled("openai"))
        self.assertFalse(self.manager.is_enabled("kling"))
        self.assertFalse(self.manager.is_enabled("noexiste"))

    def test_supported_types(self) -> None:
        self.assertEqual(self.manager.supported_types(), ["llm", "video"])

    def test_resolve_api_key_uses_environment(self) -> None:
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "clave-entorno"}):
            self.assertEqual(self.manager.resolve_api_key("openai"), "clave-entorno")

    def test_resolve_api_key_ignores_plaintext_json(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            self.assertNotEqual(self.manager.resolve_api_key("openai"), "clave-en-json")

    def test_resolve_api_key_unknown_api(self) -> None:
        self.assertEqual(self.manager.resolve_api_key("noexiste"), "")

    def test_corrupt_apis_json_does_not_crash(self) -> None:
        (Path(self._tmp.name) / "apis.json").write_text("{no json", encoding="utf-8")
        manager = ApiManager(self.cm)  # no debe lanzar
        self.assertEqual(manager.list_apis(), [])


if __name__ == "__main__":
    unittest.main()
