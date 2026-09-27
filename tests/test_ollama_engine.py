"""Tests de engines.ollama (chat, listado y descarga) con requests mockeado."""

import sys
import unittest
from pathlib import Path
from unittest import mock

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engines.base_engine import EngineError  # noqa: E402
from engines.ollama import OllamaEngine  # noqa: E402


def _response(json_data=None, status_code=200, text=""):
    resp = mock.MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.json.return_value = json_data if json_data is not None else {}
    return resp


class OllamaChatTest(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = OllamaEngine(
            base_url="http://127.0.0.1:1", default_model="llama-test"
        )

    @mock.patch("requests.post")
    def test_chat_returns_content(self, post) -> None:
        post.return_value = _response({"message": {"content": "Hola mundo"}})
        self.assertEqual(
            self.engine.chat([{"role": "user", "content": "hi"}]), "Hola mundo"
        )
        _, kwargs = post.call_args
        self.assertEqual(kwargs["json"]["model"], "llama-test")

    @mock.patch("requests.post")
    def test_chat_uses_model_override(self, post) -> None:
        post.return_value = _response({"message": {"content": "x"}})
        self.engine.chat([{"role": "user", "content": "hi"}], model="otro")
        self.assertEqual(post.call_args.kwargs["json"]["model"], "otro")

    @mock.patch("requests.post", side_effect=requests.exceptions.ConnectionError)
    def test_chat_connection_error(self, _post) -> None:
        with self.assertRaises(EngineError):
            self.engine.chat([])

    @mock.patch("requests.post", side_effect=requests.exceptions.Timeout)
    def test_chat_timeout(self, _post) -> None:
        with self.assertRaises(EngineError):
            self.engine.chat([])

    @mock.patch("requests.post")
    def test_chat_invalid_json(self, post) -> None:
        resp = _response()
        resp.json.side_effect = ValueError("no json")
        post.return_value = resp
        with self.assertRaises(EngineError):
            self.engine.chat([])

    @mock.patch("engines.ollama.is_loaded", return_value=True)
    @mock.patch("requests.post")
    def test_unload_model_posts_keep_alive_zero(self, post, _loaded) -> None:
        post.return_value = _response({})
        self.engine._last_used_model = "llama-test"
        self.assertTrue(self.engine.unload_model())
        self.assertEqual(post.call_args.kwargs["json"]["keep_alive"], 0)

    @mock.patch("engines.ollama.is_loaded", return_value=False)
    @mock.patch("requests.post")
    def test_unload_model_skips_when_not_loaded(self, post, _loaded) -> None:
        # Sin nada cargado no se llama a Ollama: ese POST haría que cargase el
        # modelo solo para descargarlo.
        self.assertTrue(self.engine.unload_model())
        post.assert_not_called()


class OllamaModelsTest(unittest.TestCase):
    @mock.patch("engines.ollama.fetch_tags")
    def test_list_models_excludes_embeddings(self, fetch_tags) -> None:
        fetch_tags.return_value = {
            "models": [
                {"name": "llama3:8b", "size": 1},
                {"name": "nomic-embed-text", "size": 2},
                {"name": "algo-embed-x", "size": 3},
            ]
        }
        engine = OllamaEngine()
        self.assertEqual(engine.list_models(), ["llama3:8b"])
        self.assertEqual(
            engine.list_model_details(), [{"name": "llama3:8b", "size": 1}]
        )


if __name__ == "__main__":
    unittest.main()
