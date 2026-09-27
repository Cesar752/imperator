"""Tests de engines.ollama_vision (imagen y texto) con ``requests`` mockeado."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests  # noqa: E402

from engines.base_engine import EngineError, VisionEngine  # noqa: E402
from engines import ollama_vision as ov  # noqa: E402
from engines.ollama_vision import OllamaVisionEngine  # noqa: E402


def _response(payload=None, error=None):
    resp = mock.Mock()
    if payload is None:
        resp.json.side_effect = ValueError("no json")
    else:
        resp.json.return_value = payload
    if error is not None:
        resp.raise_for_status.side_effect = error
    return resp


class ImageAnalysisTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.image = Path(self._tmp.name) / "pagina.png"
        self.image.write_bytes(b"\x89PNG falso")
        self.engine = OllamaVisionEngine()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_identidad_y_familia(self) -> None:
        self.assertEqual(self.engine.id, "ollama_vision")
        self.assertEqual(self.engine.nombre, "Ollama Vision")
        self.assertIsInstance(self.engine, VisionEngine)

    def test_analyze_image_envia_la_imagen_en_base64(self) -> None:
        resp = _response({"message": {"content": "texto visto"}})
        with mock.patch.object(ov.requests, "post", return_value=resp) as post:
            result = self.engine.analyze_image(self.image, "transcribe")
        self.assertEqual(result, "texto visto")
        payload = post.call_args.kwargs["json"]
        message = payload["messages"][0]
        self.assertEqual(message["content"], "transcribe")
        self.assertEqual(len(message["images"]), 1)
        self.assertEqual(post.call_args.kwargs["timeout"], self.engine._ANALYZE_TIMEOUT)

    def test_analyze_image_sin_fichero_lanza(self) -> None:
        with self.assertRaises(EngineError):
            self.engine.analyze_image(Path("no-existe.png"), "p")


class TextAnalysisTest(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = OllamaVisionEngine()

    def test_analyze_text_envia_prompt_y_texto_sin_imagen(self) -> None:
        resp = _response({"message": {"content": "resumen"}})
        with mock.patch.object(ov.requests, "post", return_value=resp) as post:
            result = self.engine.analyze_text("contenido", "RESUME", "qwen3-vl:8b")
        self.assertEqual(result, "resumen")
        payload = post.call_args.kwargs["json"]
        message = payload["messages"][0]
        self.assertNotIn("images", message)
        self.assertIn("RESUME", message["content"])
        self.assertIn("contenido", message["content"])
        self.assertEqual(payload["model"], "qwen3-vl:8b")
        self.assertFalse(payload["stream"])
        self.assertEqual(self.engine._last_used_model, "qwen3-vl:8b")

    def test_analyze_text_sin_texto_lanza(self) -> None:
        for vacio in ("", "   ", None):
            with self.assertRaises(EngineError) as ctx:
                self.engine.analyze_text(vacio, "p")
            self.assertIn("no hay texto", str(ctx.exception))

    def test_analyze_text_usa_el_modelo_por_defecto(self) -> None:
        resp = _response({"message": {"content": "ok"}})
        with mock.patch.object(ov.requests, "post", return_value=resp) as post:
            self.engine.analyze_text("c", "p")
        self.assertEqual(post.call_args.kwargs["json"]["model"], "qwen3-vl:8b")

    def test_respuesta_sin_contenido_devuelve_vacio(self) -> None:
        resp = _response({"message": {}})
        with mock.patch.object(ov.requests, "post", return_value=resp):
            self.assertEqual(self.engine.analyze_text("c", "p"), "")

    def test_error_de_conexion_sugiere_arrancar_ollama(self) -> None:
        with mock.patch.object(
            ov.requests, "post", side_effect=requests.exceptions.ConnectionError("x")
        ):
            with self.assertRaises(EngineError) as ctx:
                self.engine.analyze_text("c", "p")
        self.assertIn("¿Está arrancado?", str(ctx.exception))

    def test_timeout_lanza_error_controlado(self) -> None:
        with mock.patch.object(
            ov.requests, "post", side_effect=requests.exceptions.Timeout("t")
        ):
            with self.assertRaises(EngineError) as ctx:
                self.engine.analyze_text("c", "p")
        self.assertIn("timeout", str(ctx.exception))

    def test_json_invalido_lanza_error_controlado(self) -> None:
        with mock.patch.object(ov.requests, "post", return_value=_response(None)):
            with self.assertRaises(EngineError) as ctx:
                self.engine.analyze_text("c", "p")
        self.assertIn("no es JSON válido", str(ctx.exception))

    def test_http_error_lanza_error_controlado(self) -> None:
        resp = _response(
            {}, error=requests.exceptions.HTTPError("500")
        )
        with mock.patch.object(ov.requests, "post", return_value=resp):
            with self.assertRaises(EngineError):
                self.engine.analyze_text("c", "p")

    def test_analyze_text_por_defecto_no_esta_soportado(self) -> None:
        class _SoloImagen(VisionEngine):
            @property
            def id(self) -> str:
                return "solo-imagen"

            @property
            def nombre(self) -> str:
                return "Solo Imagen"

            def is_available(self) -> bool:
                return True

            def list_models(self) -> list[str]:
                return []

            def analyze_image(self, image_path, prompt, model=None) -> str:
                return ""

        with self.assertRaises(EngineError) as ctx:
            _SoloImagen().analyze_text("texto", "prompt")
        self.assertIn("no soporta análisis de texto", str(ctx.exception))


class UnloadTest(unittest.TestCase):
    """Solo se descarga de VRAM lo que está realmente cargado."""

    def setUp(self) -> None:
        self.engine = OllamaVisionEngine()

    def test_no_descarga_si_no_esta_cargado(self) -> None:
        with mock.patch.object(ov, "is_loaded", return_value=False), mock.patch.object(
            ov.requests, "post"
        ) as post:
            self.assertTrue(self.engine.unload_model())
        post.assert_not_called()

    def test_descarga_si_esta_cargado(self) -> None:
        with mock.patch.object(ov, "is_loaded", return_value=True), mock.patch.object(
            ov.requests, "post"
        ) as post:
            self.assertTrue(self.engine.unload_model())
        self.assertEqual(post.call_args.kwargs["json"]["keep_alive"], 0)
        self.assertEqual(
            post.call_args.kwargs["json"]["model"], self.engine._default_model
        )


if __name__ == "__main__":
    unittest.main()
