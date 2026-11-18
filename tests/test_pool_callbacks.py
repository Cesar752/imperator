"""Regresión de C2: un callback del pool no puede tocar un widget destruido.

Los callbacks de ``run_pool_task`` llegan por una conexión encolada: si el
widget muere antes, el resultado se invoca igual contra un objeto destruido
(``RuntimeError: wrapped C/C++ object ... has been deleted`` o un ``TypeError``
que PyQt no explica). ``PoolCallbackReceiver`` es el receptor ``QObject`` hijo
del widget: Qt rompe la conexión cuando el widget desaparece.

Este test vigila que no escape ninguna excepción de slot (PyQt las entrega a
``sys.excepthook``, no al test).
"""

import gc
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from core.config_manager import ConfigManager  # noqa: E402
from core.engine_manager import EngineManager  # noqa: E402
from ui import understanding_widget as uw  # noqa: E402

#: Segundos que se deja al pool para entregar el resultado.
ESPERA_S = 0.35


class _MotorQueFalla:
    id = "ollama"
    nombre = "Ollama"
    default_model = "qwen3.6:35b"

    def list_models(self):
        time.sleep(0.1)
        raise RuntimeError("sin conexión con Ollama")


class _MotorQueResponde(_MotorQueFalla):
    def list_models(self):
        time.sleep(0.1)
        return ["qwen3.6:35b", "llava:13b"]


class PoolCallbackReceiverTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.cm = ConfigManager(config_dir=self._tmp.name)
        self.manager = EngineManager(self.cm)
        self._escapadas: list[str] = []
        self._hook_previo = sys.excepthook

        def _hook(t, v, tb):
            self._escapadas.append(f"{t.__name__}: {v}")

        sys.excepthook = _hook

    def tearDown(self) -> None:
        sys.excepthook = self._hook_previo
        self._tmp.cleanup()

    def _spin(self) -> None:
        fin = time.monotonic() + ESPERA_S
        while time.monotonic() < fin:
            self.app.processEvents()
            time.sleep(0.01)

    def _widget(self, motor):
        with mock.patch.object(uw.ce, "preferred_chat_engine", return_value=motor):
            return uw.UnderstandingWidget(self.manager)

    def test_resultado_con_widget_vivo(self) -> None:
        widget = self._widget(_MotorQueResponde())
        try:
            self._spin()
            self.assertEqual(self._escapadas, [])
        finally:
            widget.close()

    def test_error_con_widget_vivo(self) -> None:
        widget = self._widget(_MotorQueFalla())
        try:
            self._spin()
            self.assertEqual(self._escapadas, [])
        finally:
            widget.close()

    def test_widget_destruido_antes_del_resultado(self) -> None:
        """El caso que abortaba el proceso: receptor muerto antes del callback."""
        widget = self._widget(_MotorQueFalla())
        widget.close()
        widget.deleteLater()
        self.app.processEvents()
        del widget
        gc.collect()

        self._spin()
        self.assertEqual(self._escapadas, [])

    def test_receptor_no_entrega_tras_subir_la_generacion(self) -> None:
        widget = self._widget(_MotorQueResponde())
        try:
            rx = widget._models_rx
            token_viejo = rx.arm()
            widget._generation += 1
            rx._token = token_viejo
            recibidos = []
            rx._on_value = lambda t, v: recibidos.append(v)  # noqa: ARG005
            rx.handle(["modelo"])
            self.assertEqual(recibidos, [], "un resultado viejo no debe entregarse")
        finally:
            widget.close()


if __name__ == "__main__":
    unittest.main()
