"""Tests del motor de chat compatible con OpenAI (engines/openai_compatible.py).

No hacen red: se sustituye ``requests.post`` por un doble.
"""

import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests  # noqa: E402

from engines import openai_compatible as oc  # noqa: E402
from engines.base_engine import CANCEL_MESSAGE, EngineError  # noqa: E402


class _Resp:
    def __init__(self, data=None, status=200, lines=None) -> None:
        self._data = data
        self.status_code = status
        self._lines = lines or []
        self.text = json.dumps(data) if data is not None else ""

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            error = requests.HTTPError("http")
            error.response = self
            raise error

    def json(self):
        if self._data is None:
            raise ValueError("no json")
        return self._data

    def iter_lines(self, decode_unicode=False):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _engine(**kwargs):
    defaults = dict(
        api_id="openai",
        base_url="https://api.openai.com/v1",
        default_model="gpt-4o-mini",
        nombre="OpenAI",
        models=["gpt-4o-mini", "gpt-4o"],
    )
    defaults.update(kwargs)
    return oc.OpenAICompatibleChatEngine(**defaults)


class AvailabilityTest(unittest.TestCase):
    def test_sin_clave_no_disponible(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            self.assertFalse(_engine().is_available())

    def test_con_clave_disponible(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "k"}):
            self.assertTrue(_engine().is_available())


class ModelsTest(unittest.TestCase):
    def test_lista_modelos_y_detalles(self):
        engine = _engine()
        self.assertEqual(engine.list_models(), ["gpt-4o-mini", "gpt-4o"])
        names = [d["name"] for d in engine.list_model_details()]
        self.assertIn("gpt-4o-mini", names)
        self.assertEqual(engine.default_model, "gpt-4o-mini")


class ChatTest(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(os.environ, {"OPENAI_API_KEY": "secret"})
        self.env.start()

    def tearDown(self):
        self.env.stop()

    def test_chat_envia_endpoint_cabecera_y_modelo(self):
        capturado = {}

        def post(url, json=None, headers=None, timeout=None, stream=False):
            capturado.update(url=url, json=json, headers=headers, stream=stream)
            return _Resp({"choices": [{"message": {"content": "Hola"}}]})

        with mock.patch.object(oc.requests, "post", post):
            salida = _engine().chat([{"role": "user", "content": "hi"}])
        self.assertEqual(salida, "Hola")
        self.assertTrue(capturado["url"].endswith("/chat/completions"))
        self.assertEqual(capturado["headers"]["Authorization"], "Bearer secret")
        self.assertEqual(capturado["json"]["model"], "gpt-4o-mini")
        self.assertFalse(capturado["stream"])

    def test_chat_usa_el_modelo_elegido(self):
        capturado = {}

        def post(url, json=None, headers=None, timeout=None, stream=False):
            capturado.update(json=json)
            return _Resp({"choices": [{"message": {"content": "ok"}}]})

        with mock.patch.object(oc.requests, "post", post):
            _engine().chat([{"role": "user", "content": "hi"}], model="gpt-4o")
        self.assertEqual(capturado["json"]["model"], "gpt-4o")

    def test_error_http_muestra_el_mensaje_de_la_api(self):
        def post(url, json=None, headers=None, timeout=None, stream=False):
            return _Resp({"error": {"message": "clave inválida"}}, status=401)

        with mock.patch.object(oc.requests, "post", post):
            with self.assertRaises(EngineError) as ctx:
                _engine().chat([{"role": "user", "content": "hi"}])
        self.assertIn("clave inválida", str(ctx.exception))

    def test_sin_clave_avisa(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            with self.assertRaises(EngineError):
                _engine().chat([{"role": "user", "content": "hi"}])

    def test_stream_acumula_y_avisa(self):
        lineas = [
            'data: {"choices":[{"delta":{"content":"Ho"}}]}',
            'data: {"choices":[{"delta":{"content":"la"}}]}',
            "data: [DONE]",
        ]

        def post(url, json=None, headers=None, timeout=None, stream=False):
            return _Resp(lines=lineas)

        tokens = []
        with mock.patch.object(oc.requests, "post", post):
            salida = _engine().chat_stream(
                [{"role": "user", "content": "hi"}],
                on_token=lambda delta, total: tokens.append(delta),
            )
        self.assertEqual(salida, "Hola")
        self.assertEqual(tokens, ["Ho", "la"])

    def test_stream_se_cancela(self):
        def post(url, json=None, headers=None, timeout=None, stream=False):
            return _Resp(lines=['data: {"choices":[{"delta":{"content":"x"}}]}'])

        with mock.patch.object(oc.requests, "post", post):
            with self.assertRaises(EngineError) as ctx:
                _engine().chat_stream(
                    [{"role": "user", "content": "hi"}],
                    cancel_check=lambda: True,
                )
        self.assertEqual(str(ctx.exception), CANCEL_MESSAGE)


class LocalServerModelsTest(unittest.TestCase):
    """Un servidor local (llama.cpp/LM Studio, sin clave) anuncia sus modelos."""

    def _local(self, **kwargs):
        defaults = dict(
            api_id="llama_cpp",
            base_url="http://127.0.0.1:8080/v1",
            default_model="local-model",
            models=[],
            api_key_required=False,
            nombre="llama.cpp",
        )
        defaults.update(kwargs)
        return oc.OpenAICompatibleChatEngine(**defaults)

    @mock.patch("requests.get")
    def test_anade_los_modelos_que_sirve_el_servidor(self, get):
        get.return_value = _Resp(
            {"data": [{"id": "Qwen2.5-Coder-14B"}, {"id": "gemma3:27b"}]}
        )
        self.assertEqual(
            self._local().list_models(),
            ["local-model", "Qwen2.5-Coder-14B", "gemma3:27b"],
        )
        self.assertTrue(str(get.call_args.args[0]).endswith("/models"))

    @mock.patch("requests.get")
    def test_si_el_servidor_falla_se_usan_los_configurados(self, get):
        get.side_effect = requests.exceptions.ConnectionError
        self.assertEqual(self._local().list_models(), ["local-model"])

    @mock.patch("requests.get")
    def test_la_nube_no_consulta_el_servidor(self, get):
        # Con clave obligatoria manda la lista curada: no se llama a /models.
        _engine().list_models()
        self.assertFalse(get.called)


if __name__ == "__main__":
    unittest.main()
