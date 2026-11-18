"""Tests del control de razonamiento de Ollama Vision (think / num_predict).

Verifican el payload que se envía a ``/api/chat`` sin hablar con Ollama real.
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engines import ollama_vision as ov  # noqa: E402


class _Resp:
    def __init__(self, data: dict) -> None:
        self._data = data

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._data


def _capture():
    """Devuelve (lista_de_payloads, función_post)."""
    capturados: list[dict] = []

    def post(url, json=None, timeout=None):  # noqa: A002 - firma de requests
        capturados.append(json)
        return _Resp({"message": {"content": "ok"}})

    return capturados, post


class ThinkPayloadTest(unittest.TestCase):
    def _engine(self, **kwargs):
        return ov.OllamaVisionEngine(base_url="http://127.0.0.1:11434", **kwargs)

    def test_think_false_apaga_razonamiento(self):
        capturados, post = _capture()
        engine = self._engine(think=False, num_predict=256)
        with mock.patch.object(ov.requests, "post", post):
            self.assertEqual(engine.analyze_text("texto", "resume"), "ok")
        payload = capturados[0]
        self.assertIs(payload["think"], False)
        self.assertEqual(payload["options"]["num_predict"], 256)
        self.assertIn("/no_think", payload["messages"][-1]["content"])

    def test_por_defecto_no_toca_el_razonamiento(self):
        capturados, post = _capture()
        engine = self._engine()
        with mock.patch.object(ov.requests, "post", post):
            engine.analyze_text("texto", "resume")
        payload = capturados[0]
        self.assertNotIn("think", payload)
        self.assertNotIn("num_predict", payload["options"])
        self.assertNotIn("/no_think", payload["messages"][-1]["content"])

    def test_num_predict_solo(self):
        capturados, post = _capture()
        engine = self._engine(num_predict=128)
        with mock.patch.object(ov.requests, "post", post):
            engine.analyze_text("texto", "resume")
        payload = capturados[0]
        self.assertEqual(payload["options"]["num_predict"], 128)
        self.assertNotIn("think", payload)


if __name__ == "__main__":
    unittest.main()
