"""Tests del motor de imagen de OpenAI (engines/openai_image.py). Sin red."""

import base64
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests  # noqa: E402

from engines import openai_image as oi  # noqa: E402
from engines.base_engine import EngineError  # noqa: E402


class _Resp:
    def __init__(self, data=None, status=200, content=b"") -> None:
        self._data = data
        self.status_code = status
        self.content = content
        self.text = ""

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            error = requests.HTTPError("http")
            error.response = self
            raise error

    def json(self):
        if self._data is None:
            raise ValueError("no json")
        return self._data


def _engine(tmp, **kwargs):
    return oi.OpenAIImageEngine(output_dir=tmp, **kwargs)


class OpenAIImageTest(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(os.environ, {"OPENAI_API_KEY": "secret"})
        self.env.start()

    def tearDown(self):
        self.env.stop()

    def test_sin_clave_no_disponible(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            self.assertFalse(oi.OpenAIImageEngine().is_available())

    def test_con_clave_disponible(self):
        self.assertTrue(oi.OpenAIImageEngine().is_available())

    def test_guarda_b64_json(self):
        imagen = base64.b64encode(b"PNGDATA").decode()

        def post(url, json=None, headers=None, timeout=None):
            return _Resp({"data": [{"b64_json": imagen}]})

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(oi.requests, "post", post):
                salida = _engine(tmp).generate_image("un gato")
            self.assertTrue(salida.is_file())
            self.assertEqual(salida.read_bytes(), b"PNGDATA")
            self.assertEqual(salida.parent, Path(tmp))

    def test_descarga_url(self):
        def post(url, json=None, headers=None, timeout=None):
            return _Resp({"data": [{"url": "http://x/y.png"}]})

        def get(url, timeout=None):
            return _Resp(content=b"PNGURL")

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(oi.requests, "post", post), mock.patch.object(
                oi.requests, "get", get
            ):
                salida = _engine(tmp).generate_image("un gato")
            self.assertEqual(salida.read_bytes(), b"PNGURL")

    def test_error_http(self):
        def post(url, json=None, headers=None, timeout=None):
            return _Resp({"error": {"message": "modelo no disponible"}}, status=400)

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(oi.requests, "post", post):
                with self.assertRaises(EngineError) as ctx:
                    _engine(tmp).generate_image("un gato")
            self.assertIn("modelo no disponible", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
