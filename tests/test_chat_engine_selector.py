"""Test del selector de motor de chat (ui/chat_widget.py)."""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from engines.base_engine import ChatEngine  # noqa: E402
from ui import chat_widget as cw  # noqa: E402


class _FakeChat(ChatEngine):
    def __init__(self, engine_id: str, nombre: str) -> None:
        self._id = engine_id
        self._nombre = nombre

    @property
    def id(self) -> str:
        return self._id

    @property
    def nombre(self) -> str:
        return self._nombre

    def is_available(self) -> bool:
        return True

    def list_models(self) -> list[str]:
        return []

    def list_model_details(self) -> list:
        return []

    def chat(self, messages, model=None) -> str:
        return ""


class _FakeManager:
    def __init__(self, engines) -> None:
        self._engines = engines

    def list_configs(self) -> list[dict]:
        return [{"id": e.id} for e in self._engines]

    def get_engine(self, engine_id: str):
        for e in self._engines:
            if e.id == engine_id:
                return e
        return None


class ChatEngineSelectorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _widget(self):
        engines = [_FakeChat("ollama", "Ollama"), _FakeChat("openai", "OpenAI")]
        manager = _FakeManager(engines)
        with mock.patch.object(cw, "run_pool_task"):
            widget = cw.ChatWidget(engine=engines[0], engine_manager=manager)
        return widget, manager

    def test_lista_los_motores_de_chat(self) -> None:
        widget, _ = self._widget()
        try:
            self.assertTrue(widget._engine_combo.isVisibleTo(widget))
            labels = [
                widget._engine_combo.itemText(i)
                for i in range(widget._engine_combo.count())
            ]
            self.assertEqual(labels, ["Ollama", "OpenAI"])
        finally:
            widget.close()

    def test_cambiar_de_motor_actualiza_el_activo(self) -> None:
        widget, _ = self._widget()
        try:
            idx = widget._engine_combo.findData("openai")
            with mock.patch.object(cw, "run_pool_task"):
                widget._engine_combo.setCurrentIndex(idx)
            self.assertEqual(widget._engine.id, "openai")
        finally:
            widget.close()

    def test_sin_engine_manager_no_hay_selector(self) -> None:
        with mock.patch.object(cw, "run_pool_task"):
            widget = cw.ChatWidget(engine=_FakeChat("ollama", "Ollama"))
        try:
            self.assertFalse(widget._engine_combo.isVisibleTo(widget))
        finally:
            widget.close()


if __name__ == "__main__":
    unittest.main()
