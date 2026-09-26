"""Tests de engines.wanvideo (workflow, progreso, WebSocket y salida).

No usan red ni la dependencia real ``websocket-client``: el módulo se sustituye
por un doble, de modo que la suite corre también en el entorno mínimo del CI.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests  # noqa: E402

from engines.base_engine import CANCEL_MESSAGE, EngineError  # noqa: E402
from engines import wanvideo as wv  # noqa: E402
from engines.wanvideo import WanVideoEngine  # noqa: E402

_WORKFLOW = {
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
    "7": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
    "12": {"class_type": "LoadImage", "inputs": {"image": "ref.png"}},
    "57": {"class_type": "KSampler", "inputs": {"noise_seed": 1}},
    "90": {"class_type": "WanImageToVideo", "inputs": {"width": 1, "height": 1}},
}


class _FakeSocket:
    """Doble del módulo ``websocket``: create_connection devuelve esta instancia."""

    class WebSocketTimeoutException(Exception):
        pass

    def __init__(self, messages=(), connect_error: Exception | None = None) -> None:
        self._messages = list(messages)
        self._connect_error = connect_error
        self.closed = False

    def create_connection(self, url, timeout=None):  # noqa: ANN001 - firma del módulo real
        if self._connect_error is not None:
            raise self._connect_error
        self.url = url
        self.timeout = timeout
        return self

    def recv(self):
        if not self._messages:
            raise self.WebSocketTimeoutException()
        item = self._messages.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def close(self) -> None:
        self.closed = True


class _WanVideoTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.wf_path = self.dir / "t2v.json"
        self.wf_path.write_text(json.dumps(_WORKFLOW), encoding="utf-8")
        self.engine = WanVideoEngine(
            base_url="http://127.0.0.1:8188",
            workflows={"t2v": str(self.wf_path), "i2v": str(self.wf_path)},
            output_dir=self.dir,
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()


class IdentityTest(_WanVideoTest):
    def test_id_y_nombre(self) -> None:
        self.assertEqual(self.engine.id, "wanvideo")
        self.assertEqual(self.engine.nombre, "Wan Video")

    def test_list_models_devuelve_los_modos(self) -> None:
        self.assertEqual(self.engine.list_models(), ["t2v", "i2v"])

    def test_set_timeout(self) -> None:
        self.engine.set_timeout(90)
        self.assertEqual(self.engine._generate_timeout, 90.0)

    def test_es_video_engine(self) -> None:
        from engines.base_engine import VideoEngine

        self.assertIsInstance(self.engine, VideoEngine)


class ValidateWorkflowTest(_WanVideoTest):
    def test_workflow_valido_no_lanza(self) -> None:
        self.engine._validate_workflow(json.loads(json.dumps(_WORKFLOW)), self.wf_path)

    def test_falta_nodo_de_prompt(self) -> None:
        workflow = json.loads(json.dumps(_WORKFLOW))
        del workflow["6"]
        with self.assertRaises(EngineError) as ctx:
            self.engine._validate_workflow(workflow, self.wf_path)
        self.assertIn("nodo 6", str(ctx.exception))

    def test_class_type_inesperado(self) -> None:
        workflow = json.loads(json.dumps(_WORKFLOW))
        workflow["6"]["class_type"] = "Otro"
        with self.assertRaises(EngineError) as ctx:
            self.engine._validate_workflow(workflow, self.wf_path)
        self.assertIn("esperado 'CLIPTextEncode'", str(ctx.exception))

    def test_falta_latent(self) -> None:
        workflow = json.loads(json.dumps(_WORKFLOW))
        del workflow["90"]
        with self.assertRaises(EngineError) as ctx:
            self.engine._validate_workflow(workflow, self.wf_path)
        self.assertIn("latent", str(ctx.exception))

    def test_falta_ksampler(self) -> None:
        workflow = json.loads(json.dumps(_WORKFLOW))
        del workflow["57"]
        with self.assertRaises(EngineError) as ctx:
            self.engine._validate_workflow(workflow, self.wf_path)
        self.assertIn("KSampler", str(ctx.exception))


class PatchWorkflowTest(_WanVideoTest):
    def test_parchea_prompt_dimensiones_seed_y_fps(self) -> None:
        workflow = json.loads(json.dumps(_WORKFLOW))
        workflow["57"]["inputs"]["fps"] = 16
        self.engine._patch_workflow(
            workflow,
            prompt="un gato",
            negative_prompt="borroso",
            width=480,
            height=832,
            length=33,
            fps=12,
            seed=99,
        )
        self.assertEqual(workflow["6"]["inputs"]["text"], "un gato")
        self.assertEqual(workflow["7"]["inputs"]["text"], "borroso")
        self.assertEqual(workflow["90"]["inputs"]["width"], 480)
        self.assertEqual(workflow["90"]["inputs"]["height"], 832)
        self.assertEqual(workflow["90"]["inputs"]["length"], 33)
        self.assertEqual(workflow["57"]["inputs"]["noise_seed"], 99)
        self.assertEqual(workflow["57"]["inputs"]["fps"], 12)

    def test_parchea_imagen_solo_si_se_indica(self) -> None:
        workflow = json.loads(json.dumps(_WORKFLOW))
        self.engine._patch_workflow(
            workflow,
            prompt="p",
            negative_prompt="",
            width=1,
            height=1,
            length=1,
            fps=1,
            seed=1,
        )
        self.assertEqual(workflow["12"]["inputs"]["image"], "ref.png")

        workflow = json.loads(json.dumps(_WORKFLOW))
        self.engine._patch_workflow(
            workflow,
            prompt="p",
            negative_prompt="",
            width=1,
            height=1,
            length=1,
            fps=1,
            seed=1,
            image_name="subida.png",
        )
        self.assertEqual(workflow["12"]["inputs"]["image"], "subida.png")


class HandleMessageTest(_WanVideoTest):
    def _progress(self) -> list:
        return []

    def test_progreso_high_y_low_noise(self) -> None:
        events = self._progress()
        message = {
            "type": "progress",
            "data": {"value": 1, "max": 2, "node": self.engine._KSAMPLER_HIGH},
        }
        self.assertIsNone(self.engine._handle_ws_message(message, lambda p, m: events.append(p)))
        self.assertEqual(events[-1], 25)
        message["data"]["node"] = self.engine._KSAMPLER_LOW
        self.engine._handle_ws_message(message, lambda p, m: events.append(p))
        self.assertEqual(events[-1], 75)

    def test_executing_marca_etapas(self) -> None:
        events = self._progress()
        message = {"type": "executing", "data": {"node": self.engine._KSAMPLER_LOW}}
        self.engine._handle_ws_message(message, lambda p, m: events.append(p))
        self.assertEqual(events[-1], 50)

    def test_estados_relevantes(self) -> None:
        cases = {
            "execution_success": "success",
            "execution_error": "error",
            "execution_interrupted": "interrupted",
        }
        for msg_type, esperado in cases.items():
            message = {"type": msg_type, "data": {}}
            self.assertEqual(
                self.engine._handle_ws_message(message, None), esperado, msg_type
            )

    def test_mensaje_irrelevante_devuelve_none(self) -> None:
        self.assertIsNone(
            self.engine._handle_ws_message({"type": "status", "data": {}}, None)
        )

    def test_ws_error_detail(self) -> None:
        message = {"data": {"exception_type": "ValueError", "exception_message": "boom"}}
        self.assertEqual(self.engine._ws_error_detail(message), "ValueError: boom")
        self.assertEqual(self.engine._ws_error_detail({"data": {}}), "error desconocido")


class FindMediaTest(unittest.TestCase):
    def test_busca_en_videos_gifs_e_images(self) -> None:
        self.assertEqual(
            WanVideoEngine._find_media({"10": {"gifs": [{"filename": "v.mp4"}]}}),
            {"key": "gifs", "data": {"filename": "v.mp4"}},
        )
        self.assertEqual(
            WanVideoEngine._find_media({"10": {"videos": [{"filename": "v.mp4"}]}}),
            {"key": "videos", "data": {"filename": "v.mp4"}},
        )

    def test_sin_salida_devuelve_none(self) -> None:
        self.assertIsNone(WanVideoEngine._find_media({}))
        self.assertIsNone(WanVideoEngine._find_media({"10": {"gifs": []}}))


class AwaitResultTest(_WanVideoTest):
    def test_sin_dependencia_websocket(self) -> None:
        with mock.patch.object(wv, "websocket", None):
            with self.assertRaises(EngineError) as ctx:
                self.engine._await_result("c", "p", None)
        self.assertIn("websocket-client", str(ctx.exception))

    def test_error_de_conexion(self) -> None:
        fake = _FakeSocket(connect_error=OSError("sin ruta"))
        with mock.patch.object(wv, "websocket", fake):
            with self.assertRaises(EngineError) as ctx:
                self.engine._await_result("c", "p", None)
        self.assertIn("no se pudo conectar al WebSocket", str(ctx.exception))

    def test_cancelacion_antes_de_recibir(self) -> None:
        fake = _FakeSocket()
        with mock.patch.object(wv, "websocket", fake):
            with self.assertRaises(EngineError) as ctx:
                self.engine._await_result("c", "p", None, cancel_check=lambda: True)
        self.assertEqual(str(ctx.exception), CANCEL_MESSAGE)
        self.assertTrue(fake.closed)

    def test_mensaje_interrupted_es_cancelacion(self) -> None:
        fake = _FakeSocket(['{"type": "execution_interrupted", "data": {}}'])
        with mock.patch.object(wv, "websocket", fake):
            with self.assertRaises(EngineError) as ctx:
                self.engine._await_result("c", "p", None)
        self.assertEqual(str(ctx.exception), CANCEL_MESSAGE)

    def test_mensaje_de_error_incluye_detalle(self) -> None:
        message = {
            "type": "execution_error",
            "data": {"exception_type": "RuntimeError", "exception_message": "sin VRAM"},
        }
        fake = _FakeSocket([json.dumps(message)])
        with mock.patch.object(wv, "websocket", fake):
            with self.assertRaises(EngineError) as ctx:
                self.engine._await_result("c", "p", None)
        self.assertIn("RuntimeError: sin VRAM", str(ctx.exception))

    def test_exito_delega_en_read_output(self) -> None:
        fake = _FakeSocket(['{"type": "execution_success", "data": {}}'])
        with mock.patch.object(wv, "websocket", fake), mock.patch.object(
            WanVideoEngine, "_read_output", return_value={"filename": "v.mp4"}
        ) as read:
            result = self.engine._await_result("c", "p", None)
        self.assertEqual(result, {"filename": "v.mp4"})
        self.assertEqual(read.call_args.args[0], "p")


class ReadOutputTest(_WanVideoTest):
    def _response(self, payload: dict) -> mock.Mock:
        resp = mock.Mock()
        resp.json.return_value = payload
        return resp

    def test_cancelacion_detectada_antes_de_consultar(self) -> None:
        with self.assertRaises(EngineError) as ctx:
            self.engine._read_output("pid", cancel_check=lambda: True)
        self.assertEqual(str(ctx.exception), CANCEL_MESSAGE)

    def test_devuelve_la_media_del_historial(self) -> None:
        payload = {
            "pid": {
                "status": {"completed": True},
                "outputs": {"10": {"gifs": [{"filename": "v.mp4"}]}},
            }
        }
        with mock.patch.object(wv.requests, "get", return_value=self._response(payload)):
            media = self.engine._read_output("pid")
        self.assertEqual(media, {"filename": "v.mp4"})

    def test_estado_error_lanza(self) -> None:
        payload = {"pid": {"status": {"status_str": "error"}, "outputs": {}}}
        with mock.patch.object(wv.requests, "get", return_value=self._response(payload)):
            with self.assertRaises(EngineError) as ctx:
                self.engine._read_output("pid")
        self.assertIn("error", str(ctx.exception))

    def test_sin_salida_agota_reintentos(self) -> None:
        with mock.patch.object(
            wv.requests, "get", side_effect=requests.RequestException("caido")
        ), mock.patch.object(wv.time, "sleep", return_value=None), mock.patch.object(
            WanVideoEngine, "_HISTORY_RETRIES", 2
        ):
            with self.assertRaises(EngineError) as ctx:
                self.engine._read_output("pid")
        self.assertIn("no se pudo leer la salida", str(ctx.exception))


class GenerateVideoTest(_WanVideoTest):
    def test_i2v_sin_imagen_lanza(self) -> None:
        with self.assertRaises(EngineError) as ctx:
            self.engine.generate_video("p", model="i2v")
        self.assertIn("requiere image_path", str(ctx.exception))

    def test_t2v_end_to_end_parchea_y_guarda(self) -> None:
        salida = self.dir / "out.mp4"
        with mock.patch.object(
            WanVideoEngine, "_submit", return_value="pid"
        ) as submit, mock.patch.object(
            WanVideoEngine, "_await_result", return_value={"filename": "v.mp4"}
        ), mock.patch.object(
            WanVideoEngine, "_download", return_value=b"datos"
        ), mock.patch.object(
            WanVideoEngine, "_save", return_value=salida
        ) as save, mock.patch.object(
            WanVideoEngine, "unload_model", return_value=True
        ) as unload:
            result = self.engine.generate_video(
                "un gato", negative_prompt="borroso", seed=5
            )
        self.assertEqual(result, salida)
        self.assertEqual(submit.call_args.args[0]["6"]["inputs"]["text"], "un gato")
        self.assertEqual(submit.call_args.args[0]["57"]["inputs"]["noise_seed"], 5)
        self.assertEqual(save.call_args.args[1], 5)
        unload.assert_called_once()

    def test_t2v_con_imagen_cambia_a_i2v_y_sube(self) -> None:
        image = self.dir / "ref.png"
        image.write_bytes(b"png")
        with mock.patch.object(
            WanVideoEngine, "_upload_image", return_value="subida.png"
        ) as upload, mock.patch.object(
            WanVideoEngine, "_submit", return_value="pid"
        ) as submit, mock.patch.object(
            WanVideoEngine, "_await_result", return_value={"filename": "v.mp4"}
        ), mock.patch.object(
            WanVideoEngine, "_download", return_value=b"datos"
        ), mock.patch.object(
            WanVideoEngine, "_save", return_value=self.dir / "out.mp4"
        ), mock.patch.object(
            WanVideoEngine, "unload_model", return_value=True
        ):
            self.engine.generate_video("p", image_path=str(image), model="t2v")
        upload.assert_called_once()
        self.assertEqual(submit.call_args.args[0]["12"]["inputs"]["image"], "subida.png")


if __name__ == "__main__":
    unittest.main()
