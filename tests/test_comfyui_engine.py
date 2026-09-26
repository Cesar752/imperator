"""Tests de engines.comfyui (workflow, submit, espera y generación) con mocks."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engines.base_engine import EngineError, ImageEngine  # noqa: E402
from engines.comfyui import ComfyUIEngine, _flatten_messages  # noqa: E402

_WORKFLOW = {
    "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "positivo"}},
    "2": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
    "3": {"class_type": "KSampler", "inputs": {"seed": 0, "steps": 0}},
    "4": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 0, "height": 0}},
}


def _response(json_data=None, status_code=200, text="", content=b""):
    resp = mock.MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.content = content
    resp.json.return_value = json_data if json_data is not None else {}
    return resp


class ComfyUIWorkflowTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.wf_path = self.dir / "wf.json"
        self.wf_path.write_text(json.dumps(_WORKFLOW), encoding="utf-8")
        self.engine = ComfyUIEngine(
            base_url="http://127.0.0.1:1",
            workflows={"flux_dev": str(self.wf_path)},
            output_dir=self.dir,
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_engine_family(self) -> None:
        self.assertIsInstance(self.engine, ImageEngine)
        self.assertEqual(self.engine.list_models(), ["flux_dev"])

    def test_load_workflow_unknown_model(self) -> None:
        with self.assertRaises(EngineError):
            self.engine._load_workflow("noexiste")

    def test_load_workflow_missing_file(self) -> None:
        engine = ComfyUIEngine(
            base_url="http://127.0.0.1:1",
            workflows={"x": str(self.dir / "no_existe.json")},
        )
        with self.assertRaises(EngineError):
            engine._load_workflow("x")

    def test_load_workflow_corrupt_json(self) -> None:
        bad = self.dir / "bad.json"
        bad.write_text("{no json", encoding="utf-8")
        engine = ComfyUIEngine(base_url="http://127.0.0.1:1", workflows={"x": str(bad)})
        with self.assertRaises(EngineError):
            engine._load_workflow("x")

    def test_patch_workflow_sets_fields(self) -> None:
        wf = json.loads(json.dumps(_WORKFLOW))
        self.engine._patch_workflow(
            wf,
            prompt="P",
            negative_prompt="N",
            width=800,
            height=600,
            steps=12,
            seed=42,
            guidance=4.0,
        )
        self.assertEqual(wf["1"]["inputs"]["text"], "P")
        self.assertEqual(wf["2"]["inputs"]["text"], "N")
        self.assertEqual(wf["3"]["inputs"]["seed"], 42)
        self.assertEqual(wf["3"]["inputs"]["steps"], 12)
        self.assertEqual(wf["4"]["inputs"]["width"], 800)
        self.assertEqual(wf["4"]["inputs"]["height"], 600)


class ComfyUISubmitTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.engine = ComfyUIEngine(
            base_url="http://127.0.0.1:1", workflows={"x": "y"}, output_dir=self.dir
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    @mock.patch("requests.post")
    def test_submit_ok(self, post) -> None:
        post.return_value = _response({"prompt_id": "abc"})
        self.assertEqual(self.engine._submit({}), "abc")

    @mock.patch("requests.post")
    def test_submit_rejected(self, post) -> None:
        post.return_value = _response(
            {"error": {"message": "nodo inválido"}}, status_code=400
        )
        with self.assertRaises(EngineError):
            self.engine._submit({})

    @mock.patch("requests.post")
    def test_submit_without_prompt_id(self, post) -> None:
        post.return_value = _response({})
        with self.assertRaises(EngineError):
            self.engine._submit({})

    @mock.patch("requests.get")
    def test_is_available(self, get) -> None:
        get.return_value = _response({}, status_code=200)
        self.assertTrue(self.engine.is_available())

    @mock.patch("requests.get", side_effect=requests.exceptions.ConnectionError)
    def test_is_available_offline(self, _get) -> None:
        self.assertFalse(self.engine.is_available())

    def test_first_image_meta_raises_when_empty(self) -> None:
        with self.assertRaises(EngineError):
            self.engine._first_image_meta({"outputs": {}})

    def test_flatten_messages(self) -> None:
        self.assertEqual(_flatten_messages(["a", ["b", "c"]]), "a b c")


class ComfyUIGenerateTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.wf_path = self.dir / "wf.json"
        self.wf_path.write_text(json.dumps(_WORKFLOW), encoding="utf-8")
        self.engine = ComfyUIEngine(
            base_url="http://127.0.0.1:1",
            workflows={"flux_dev": str(self.wf_path)},
            output_dir=self.dir,
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _get_side_effect(self, url, **_kwargs):
        if "/history/" in url:
            return _response(
                {
                    "pid": {
                        "status": {"completed": True},
                        "outputs": {
                            "9": {"images": [{"filename": "a.png", "type": "output"}]}
                        },
                    }
                }
            )
        if url.endswith("/view"):
            return _response(content=b"PNGDATA")
        return _response({})

    @mock.patch("requests.get")
    @mock.patch("requests.post")
    def test_generate_image_end_to_end(self, post, get) -> None:
        post.return_value = _response({"prompt_id": "pid"})
        get.side_effect = self._get_side_effect
        path = self.engine.generate_image("un gato", model="flux_dev", seed=7)
        self.assertTrue(path.exists())
        self.assertEqual(path.read_bytes(), b"PNGDATA")
        self.assertTrue(path.name.endswith("_7.png"))

    @mock.patch("requests.get")
    @mock.patch("requests.post")
    def test_generate_image_error_status(self, post, get) -> None:
        post.return_value = _response({"prompt_id": "pid"})
        get.side_effect = lambda url, **kw: _response(
            {
                "pid": {
                    "status": {
                        "status_str": "error",
                        "messages": ["fallo de nodo"],
                    },
                    "outputs": {},
                }
            }
        )
        with self.assertRaises(EngineError):
            self.engine.generate_image("x", model="flux_dev")


if __name__ == "__main__":
    unittest.main()
