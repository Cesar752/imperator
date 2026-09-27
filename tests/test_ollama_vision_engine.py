"""Tests de engines.ollama_vision (imagen y texto) con ``requests`` mockeado."""

import json
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


def _respuesta_de_error(payload: dict, status_code: int = 400):
    """Respuesta real de requests con cuerpo, para probar el mensaje del error."""
    resp = requests.models.Response()
    resp.status_code = status_code
    resp._content = json.dumps(payload).encode("utf-8")
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

    def test_pide_ventana_de_contexto(self) -> None:
        # Sin ``options.num_ctx`` Ollama usa 4096 y una página con mucho texto no
        # cabe: responde 400 y la tarea falla.
        resp = _response({"message": {"content": "texto visto"}})
        with mock.patch.object(ov.requests, "post", return_value=resp) as post:
            self.engine.analyze_image(self.image, "transcribe")
        opciones = post.call_args.kwargs["json"]["options"]
        self.assertEqual(opciones["num_ctx"], OllamaVisionEngine._NUM_CTX)


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

    def test_pagina_en_blanco_no_es_un_error(self) -> None:
        # Una página sin texto (o una imagen sin letras) devuelve vacío: no hay
        # nada que transcribir, pero tampoco un fallo.
        resp = _response({"message": {}, "done_reason": "stop"})
        with mock.patch.object(ov.requests, "post", return_value=resp):
            self.assertEqual(self.engine.analyze_text("c", "p"), "")

    def test_el_timeout_es_generoso_y_configurable(self) -> None:
        # Medido: una página con qwen3-vl puede tardar minutos (razona antes de
        # responder); con 180 s la tarea moría a mitad de documento.
        resp = _response({"message": {"content": "ok"}})
        with mock.patch.object(ov.requests, "post", return_value=resp) as post:
            self.engine.analyze_text("c", "p")
        self.assertEqual(post.call_args.kwargs["timeout"], self.engine._ANALYZE_TIMEOUT)
        self.assertGreaterEqual(OllamaVisionEngine._ANALYZE_TIMEOUT, 600.0)

        lento = OllamaVisionEngine(timeout=1800)
        with mock.patch.object(ov.requests, "post", return_value=resp) as post:
            lento.analyze_text("c", "p")
        self.assertEqual(post.call_args.kwargs["timeout"], 1800)

    def test_respuesta_cortada_avisa(self) -> None:
        resp = _response(
            {"message": {"content": "resumen a medias"}, "done_reason": "length"}
        )
        with mock.patch.object(ov.requests, "post", return_value=resp):
            contenido = self.engine.analyze_text("c", "p")
        self.assertTrue(contenido.startswith("resumen a medias"))
        self.assertTrue(contenido.endswith(OllamaVisionEngine.TRUNCATED_NOTE))

    def test_sin_respuesta_por_razonamiento_lo_explica(self) -> None:
        # Es el caso real: el modelo piensa miles de tokens y no escribe nada.
        resp = _response(
            {
                "message": {"content": "", "thinking": "pensando..." * 100},
                "done_reason": "stop",
            }
        )
        with mock.patch.object(ov.requests, "post", return_value=resp):
            with self.assertRaises(EngineError) as ctx:
                self.engine.analyze_text("c", "p")
        mensaje = str(ctx.exception)
        self.assertIn("razonando", mensaje)
        self.assertIn("qwen2.5vl", mensaje)

    def test_sin_respuesta_por_contexto_lleno_lo_explica(self) -> None:
        resp = _response({"message": {"content": ""}, "done_reason": "length"})
        with mock.patch.object(ov.requests, "post", return_value=resp):
            with self.assertRaises(EngineError) as ctx:
                self.engine.analyze_text("c", "p")
        self.assertIn("contexto", str(ctx.exception))

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

    def test_400_por_contexto_explica_el_motivo(self) -> None:
        resp = mock.Mock()
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "400",
            response=_respuesta_de_error(
                {
                    "error": "request (4135 tokens) exceeds the available "
                    "context size (4096 tokens), try increasing it"
                }
            ),
        )
        with mock.patch.object(ov.requests, "post", return_value=resp):
            with self.assertRaises(EngineError) as ctx:
                self.engine.analyze_text("c", "p")
        mensaje = str(ctx.exception)
        self.assertIn("no cabe en el contexto", mensaje)
        self.assertIn(f"num_ctx={OllamaVisionEngine._NUM_CTX}", mensaje)
        self.assertIn("4135", mensaje)

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
