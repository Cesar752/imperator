"""Tests de engines.bfl (API de pago) con requests mockeado."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engines.base_engine import EngineError, ImageEngine  # noqa: E402
from engines.bfl import BFLApiEngine  # noqa: E402


def _response(json_data=None, status_code=200, text="", content=b""):
    resp = mock.MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.content = content
    resp.json.return_value = json_data if json_data is not None else {}
    return resp


class BFLTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.engine = BFLApiEngine(output_dir=self.dir)
        self._env = mock.patch.dict(os.environ, {"BFL_API_KEY": "test-key"})
        self._env.start()

    def tearDown(self) -> None:
        self._env.stop()
        self._tmp.cleanup()

    def test_engine_family_and_models(self) -> None:
        self.assertIsInstance(self.engine, ImageEngine)
        self.assertIn("flux-kontext-pro", self.engine.list_models())

    def test_is_available_with_key(self) -> None:
        self.assertTrue(self.engine.is_available())

    def test_is_available_without_key(self) -> None:
        with mock.patch.dict(os.environ, {"BFL_API_KEY": ""}):
            self.assertFalse(self.engine.is_available())

    def test_generate_without_key_raises(self) -> None:
        with mock.patch.dict(os.environ, {"BFL_API_KEY": ""}):
            with self.assertRaises(EngineError):
                self.engine.generate_image("x")

    @mock.patch("requests.post")
    def test_submit_rejects_http_error(self, post) -> None:
        post.return_value = _response(status_code=401, text="unauthorized")
        with self.assertRaises(EngineError):
            self.engine._submit("flux-kontext-pro", {"prompt": "x"})

    @mock.patch("requests.get")
    def test_el_polling_traduce_el_http_error(self, get) -> None:
        """Antes salía un ``requests.HTTPError`` crudo sin contexto de la API."""
        get.return_value = _response(status_code=403, text="sin cuota")
        with self.assertRaises(EngineError) as ctx:
            self.engine._wait("https://api.bfl.ai/v1/get_result?id=1")
        self.assertIn("403", str(ctx.exception))

    @mock.patch("requests.get")
    def test_descarga_con_http_error(self, get) -> None:
        get.return_value = _response(status_code=500, text="boom")
        with self.assertRaises(EngineError) as ctx:
            self.engine._download_and_save("https://x/y.png", 0)
        self.assertIn("500", str(ctx.exception))

    @mock.patch("requests.get")
    def test_descarga_demasiado_grande(self, get) -> None:
        self.engine._MAX_DOWNLOAD_BYTES = 4
        get.return_value = _response(content=b"x" * 5)
        with self.assertRaises(EngineError) as ctx:
            self.engine._download_and_save("https://x/y.png", 0)
        self.assertIn("tope", str(ctx.exception))

    @mock.patch("requests.get")
    def test_descarga_ok_guarda_el_png(self, get) -> None:
        get.return_value = _response(content=b"png-bytes")
        ruta = self.engine._download_and_save("https://x/y.png", 7)
        self.assertTrue(ruta.exists())
        self.assertEqual(ruta.read_bytes(), b"png-bytes")
        self.assertIn("7", ruta.name)

    @mock.patch("requests.get")
    @mock.patch("requests.post")
    def test_generate_image_end_to_end(self, post, get) -> None:
        post.return_value = _response({"polling_url": "http://poll"})

        def _get(url, **_kwargs):
            if url == "http://poll":
                return _response({"status": "Ready", "result": {"sample": "http://img"}})
            return _response(content=b"BFLPNG")

        get.side_effect = _get
        path = self.engine.generate_image("un gato", model="flux-kontext-pro", seed=3)
        self.assertTrue(path.exists())
        self.assertEqual(path.read_bytes(), b"BFLPNG")
        self.assertTrue(path.name.endswith("_3.png"))

    @mock.patch("requests.get")
    @mock.patch("requests.post")
    def test_generate_error_status(self, post, get) -> None:
        post.return_value = _response({"polling_url": "http://poll"})
        get.return_value = _response({"status": "Error"})
        with self.assertRaises(EngineError):
            self.engine.generate_image("x", model="flux-kontext-pro")

    def test_image_value_url_passthrough(self) -> None:
        self.assertEqual(
            self.engine._image_value("https://example.com/a.png"),
            "https://example.com/a.png",
        )

    def test_image_value_missing_file(self) -> None:
        with self.assertRaises(EngineError):
            self.engine._image_value(self.dir / "no_existe.png")

    @mock.patch("requests.get")
    def test_get_credits(self, get) -> None:
        get.return_value = _response({"credits": 123.5})
        self.assertAlmostEqual(self.engine.get_credits(), 123.5)

    @mock.patch("requests.get", side_effect=requests.exceptions.ConnectionError)
    def test_get_credits_offline(self, _get) -> None:
        self.assertIsNone(self.engine.get_credits())


if __name__ == "__main__":
    unittest.main()
