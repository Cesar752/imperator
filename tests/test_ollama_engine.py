"""Tests de engines.ollama (chat, listado y descarga) con requests mockeado."""

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engines.base_engine import (  # noqa: E402
    CANCEL_MESSAGE,
    EngineError,
    chat_engine_reply,
)
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
    def test_chat_usa_un_timeout_generoso_por_defecto(self, post) -> None:
        # Los modelos que razonan tardan minutos: con 120 s el guion de escenas
        # moría a mitad (medido: 6709 tokens generados a ~100 t/s y cortado).
        post.return_value = _response({"message": {"content": "x"}})
        self.engine.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(post.call_args.kwargs["timeout"], OllamaEngine._CHAT_TIMEOUT)
        self.assertGreater(OllamaEngine._CHAT_TIMEOUT, 120.0)

    @mock.patch("requests.post")
    def test_chat_timeout_configurable(self, post) -> None:
        post.return_value = _response({"message": {"content": "x"}})
        engine = OllamaEngine(base_url="http://127.0.0.1:1", timeout=45.0)
        engine.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(post.call_args.kwargs["timeout"], 45.0)

    @mock.patch("requests.post")
    def test_chat_invalid_json(self, post) -> None:
        resp = _response()
        resp.json.side_effect = ValueError("no json")
        post.return_value = resp
        with self.assertRaises(EngineError):
            self.engine.chat([])

    @mock.patch("engines.ollama_api.is_loaded", return_value=True)
    @mock.patch("requests.post")
    def test_unload_model_posts_keep_alive_zero(self, post, _loaded) -> None:
        post.return_value = _response({})
        self.engine._last_used_model = "llama-test"
        self.assertTrue(self.engine.unload_model())
        self.assertEqual(post.call_args.kwargs["json"]["keep_alive"], 0)

    @mock.patch("engines.ollama_api.is_loaded", return_value=False)
    @mock.patch("requests.post")
    def test_unload_model_skips_when_not_loaded(self, post, _loaded) -> None:
        # Sin nada cargado no se llama a Ollama: ese POST haría que cargase el
        # modelo solo para descargarlo.
        self.assertTrue(self.engine.unload_model())
        post.assert_not_called()

    @mock.patch("engines.ollama_api.is_loaded", return_value=True)
    def test_is_loaded_delega_en_api_ps(self, check) -> None:
        # Lo pregunta el planificador de GPU para no bloquear el chat por su
        # propio modelo ya cargado.
        self.assertTrue(self.engine.is_loaded("llama-test"))
        self.assertEqual(check.call_args.args[1], "llama-test")

    @mock.patch("engines.ollama_api.is_loaded", return_value=False)
    def test_is_loaded_sin_modelo_usa_el_ultimo_turno(self, check) -> None:
        self.engine._last_used_model = "ultimo"
        self.assertFalse(self.engine.is_loaded())
        self.assertEqual(check.call_args.args[1], "ultimo")


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


class _StreamResponse:
    """Respuesta de ``requests`` con ``iter_lines`` (streaming NDJSON de Ollama)."""

    def __init__(self, lines, status_code=200):
        self._lines = list(lines)
        self.status_code = status_code

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(response=self)

    def iter_lines(self, decode_unicode=False):
        yield from self._lines


def _ndjson(*items) -> list[str]:
    return [json.dumps(item) for item in items]


class OllamaStreamTest(unittest.TestCase):
    """``chat_stream``: avance real y cancelación inmediata."""

    def setUp(self) -> None:
        self.engine = OllamaEngine(
            base_url="http://127.0.0.1:1", default_model="llama-test"
        )

    @mock.patch("requests.post")
    def test_acumula_los_fragmentos_y_avisa(self, post) -> None:
        post.return_value = _StreamResponse(
            _ndjson(
                {"message": {"content": "Ho"}, "done": False},
                {"message": {"content": "la"}, "done": False},
                {"message": {"content": ""}, "done": True, "done_reason": "stop"},
            )
        )
        vistos: list = []
        reply = self.engine.chat_stream(
            [{"role": "user", "content": "hi"}],
            on_token=lambda delta, total: vistos.append((delta, total)),
        )
        self.assertEqual(reply, "Hola")
        self.assertEqual(vistos, [("Ho", 2), ("la", 4)])
        self.assertTrue(post.call_args.kwargs["json"]["stream"])
        self.assertTrue(post.call_args.kwargs["stream"])

    @mock.patch("requests.post")
    def test_cancela_al_instante(self, post) -> None:
        post.return_value = _StreamResponse(
            _ndjson(*[{"message": {"content": "x"}, "done": False}] * 3)
        )
        with self.assertRaises(EngineError) as ctx:
            self.engine.chat_stream([], cancel_check=lambda: True)
        self.assertEqual(str(ctx.exception), CANCEL_MESSAGE)

    @mock.patch("requests.post")
    def test_marca_la_respuesta_cortada(self, post) -> None:
        post.return_value = _StreamResponse(
            _ndjson(
                {"message": {"content": "cort"}, "done": False},
                {"message": {"content": ""}, "done": True, "done_reason": "length"},
            )
        )
        reply = self.engine.chat_stream([])
        self.assertIn("cort", reply)
        self.assertIn("límite de tokens", reply)

    @mock.patch("requests.post")
    def test_respuesta_vacia_es_error(self, post) -> None:
        post.return_value = _StreamResponse(
            _ndjson({"message": {"content": ""}, "done": True, "done_reason": "stop"})
        )
        with self.assertRaises(EngineError):
            self.engine.chat_stream([])


class ChatEngineReplyTest(unittest.TestCase):
    """El helper usa ``chat_stream`` si está y, si no, cae a ``chat``."""

    def test_cae_a_chat_si_no_hay_streaming(self) -> None:
        class _SinStream:
            def chat(self, messages, model=None):
                return "respuesta"

        self.assertEqual(chat_engine_reply(_SinStream(), [], model=None), "respuesta")

    def test_usa_chat_stream_si_esta(self) -> None:
        class _ConStream:
            def chat(self, messages, model=None):  # pragma: no cover - no debe usarse
                raise AssertionError("no debería usarse chat()")

            def chat_stream(
                self, messages, model=None, *, on_token=None, cancel_check=None
            ):
                return "stream"

        self.assertEqual(chat_engine_reply(_ConStream(), []), "stream")


if __name__ == "__main__":
    unittest.main()
