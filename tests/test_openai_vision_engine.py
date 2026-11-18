"""Tests del motor de visión compatible con OpenAI (engines/openai_vision.py).

No hacen red: se sustituye ``requests.post`` por un doble. Se comprueba lo que
importa para la Crónica: que la imagen se manda como *data URL*, que se respeta el
modelo elegido y que los errores de la API se explican.
"""

import json
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests  # noqa: E402

from engines import openai_vision as ov  # noqa: E402
from engines.base_engine import EngineError  # noqa: E402


class _Resp:
    def __init__(self, data=None, status=200) -> None:
        self._data = data
        self.status_code = status
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


def _engine(**kwargs):
    defaults = dict(
        api_id="openai_vision",
        base_url="https://api.openai.com/v1",
        default_model="gpt-4o-mini",
        nombre="OpenAI Vision",
        models=["gpt-4o-mini", "gpt-4o"],
        # Igual que el gestor: la visión de pago comparte la clave del chat de
        # OpenAI (OPENAI_API_KEY), no una propia.
        api_key_env="OPENAI_API_KEY",
    )
    defaults.update(kwargs)
    return ov.OpenAICompatibleVisionEngine(**defaults)


class AvailabilityTest(unittest.TestCase):
    def test_sin_clave_no_disponible(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            self.assertFalse(_engine().is_available())

    def test_con_clave_disponible(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "k"}):
            self.assertTrue(_engine().is_available())

    def test_servidor_local_sin_clave_esta_disponible(self):
        self.assertTrue(_engine(api_key_required=False).is_available())


class ModelsTest(unittest.TestCase):
    def test_lista_modelos_y_detalles(self):
        engine = _engine()
        self.assertEqual(engine.list_models(), ["gpt-4o-mini", "gpt-4o"])
        self.assertEqual(engine.default_model, "gpt-4o-mini")
        self.assertTrue(all(d["installed"] for d in engine.list_model_details()))


class LocalServerModelsTest(unittest.TestCase):
    @mock.patch("requests.get")
    def test_un_servidor_local_anade_sus_modelos(self, get):
        get.return_value = _Resp({"data": [{"id": "gemma3:27b"}]})
        engine = _engine(api_key_required=False)
        self.assertIn("gemma3:27b", engine.list_models())


class AnalyzeImageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.env = mock.patch.dict(os.environ, {"OPENAI_API_KEY": "secret"})
        self.env.start()
        self._tmp = TemporaryDirectory()
        self.png = Path(self._tmp.name) / "frame.png"
        self.png.write_bytes(b"\x89PNG\r\n\x1a\n fake")

    def tearDown(self) -> None:
        self.env.stop()
        self._tmp.cleanup()

    def test_analyze_image_manda_data_url_y_devuelve_texto(self):
        capturado = {}

        def post(url, json=None, headers=None, timeout=None):
            capturado.update(url=url, json=json, headers=headers)
            return _Resp({"choices": [{"message": {"content": "Gol de Álex"}}]})

        with mock.patch.object(ov.requests, "post", post):
            salida = _engine().analyze_image(self.png, "¿qué se ve?")

        self.assertEqual(salida, "Gol de Álex")
        self.assertTrue(capturado["url"].endswith("/chat/completions"))
        self.assertEqual(capturado["headers"]["Authorization"], "Bearer secret")
        contenido = capturado["json"]["messages"][0]["content"]
        self.assertEqual(contenido[0]["type"], "text")
        self.assertEqual(contenido[1]["type"], "image_url")
        self.assertTrue(contenido[1]["image_url"]["url"].startswith("data:image/"))
        self.assertIn("base64,", contenido[1]["image_url"]["url"])

    def test_analyze_image_usa_el_modelo_elegido(self):
        capturado = {}

        def post(url, json=None, headers=None, timeout=None):
            capturado.update(json=json)
            return _Resp({"choices": [{"message": {"content": "ok"}}]})

        with mock.patch.object(ov.requests, "post", post):
            _engine().analyze_image(self.png, "prompt", "gpt-4o")
        self.assertEqual(capturado["json"]["model"], "gpt-4o")

    def test_analyze_image_sin_fichero_es_error(self):
        with self.assertRaises(EngineError):
            _engine().analyze_image(Path(self._tmp.name) / "no.png", "prompt")

    def test_block_content_se_concatena(self):
        def post(url, json=None, headers=None, timeout=None):
            return _Resp(
                {"choices": [{"message": {"content": [{"text": "Ho"}, {"text": "la"}]}}]}
            )

        with mock.patch.object(ov.requests, "post", post):
            salida = _engine().analyze_image(self.png, "prompt")
        self.assertEqual(salida, "Hola")

    def test_error_http_muestra_el_mensaje_de_la_api(self):
        def post(url, json=None, headers=None, timeout=None):
            return _Resp({"error": {"message": "cuota agotada"}}, status=429)

        with mock.patch.object(ov.requests, "post", post):
            with self.assertRaises(EngineError) as ctx:
                _engine().analyze_image(self.png, "prompt")
        self.assertIn("cuota agotada", str(ctx.exception))

    def test_sin_clave_avisa(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            with self.assertRaises(EngineError):
                _engine().analyze_image(self.png, "prompt")


class AnalyzeTextTest(unittest.TestCase):
    def setUp(self) -> None:
        self.env = mock.patch.dict(os.environ, {"OPENAI_API_KEY": "secret"})
        self.env.start()

    def tearDown(self) -> None:
        self.env.stop()

    def test_analyze_text_sin_imagen(self):
        capturado = {}

        def post(url, json=None, headers=None, timeout=None):
            capturado.update(json=json)
            return _Resp({"choices": [{"message": {"content": "resumen"}}]})

        with mock.patch.object(ov.requests, "post", post):
            salida = _engine().analyze_text("texto de la página", "resume")
        self.assertEqual(salida, "resumen")
        contenido = capturado["json"]["messages"][0]["content"]
        self.assertIsInstance(contenido, str)
        self.assertIn("texto de la página", contenido)

    def test_analyze_text_vacio_es_error(self):
        with self.assertRaises(EngineError):
            _engine().analyze_text("   ", "resume")


class RegistroTest(unittest.TestCase):
    def test_esta_en_la_factoria_de_motores(self):
        from core.engine_manager import ENGINE_FACTORY

        self.assertIs(ENGINE_FACTORY["openai_vision"], ov.OpenAICompatibleVisionEngine)

    def test_el_gestor_construye_el_motor(self):
        from core.engine_manager import EngineManager
        from core.config_manager import ConfigManager

        with TemporaryDirectory() as tmp:
            manager = EngineManager(ConfigManager(config_dir=tmp))
            instance = manager.get_instance("openai_vision")
        self.assertIsNotNone(instance)
        self.assertEqual(instance.id, "openai_vision")
        self.assertEqual(instance.default_model, "gpt-4o-mini")
        # La clave es la del chat de OpenAI (una sola variables de entorno).
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "k"}):
            self.assertTrue(instance.is_available())
