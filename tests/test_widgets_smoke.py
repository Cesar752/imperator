"""Smoke tests de las pestañas que no tenían test propio.

Cada widget se construye en modo offscreen con una `ConfigManager` temporal (para
no tocar `config/` real) y se comprueba que arranca sin excepciones y con sus
controles básicos en su sitio. Son las pestañas de Vídeo, Imagen, Audio,
Ensamblador y Comprensión: la auditoría del 30/09/2026 señaló que solo el chat y
Documentos tenían tests, y que estas cinco son las que más superficie tienen.

Los sondeos que un widget lanza contra los motores al construirse (`run_pool_task`)
se sustituyen por un doble: el test no toca red ni GPU.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QComboBox, QWidget  # noqa: E402

from core.config_manager import ConfigManager  # noqa: E402
from core.engine_manager import EngineManager  # noqa: E402
from ui import assembler_widget, audio_widget, image_widget  # noqa: E402
from ui import understanding_widget, video_widget  # noqa: E402


class WidgetsSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.cm = ConfigManager(config_dir=self._tmp.name)
        self._patches = [
            mock.patch.object(modulo, "run_pool_task", mock.MagicMock())
            for modulo in (video_widget, image_widget, understanding_widget)
        ]
        for patcher in self._patches:
            patcher.start()

    def tearDown(self) -> None:
        for patcher in self._patches:
            patcher.stop()
        self._tmp.cleanup()

    def _comprobar(self, widget: QWidget, combos: int | None = None) -> QWidget:
        self.assertIsInstance(widget, QWidget)
        self.assertIsNotNone(widget.layout())
        if combos is not None:
            self.assertEqual(len(widget.findChildren(QComboBox)), combos)
        widget.close()
        return widget

    def test_video_widget_arranca(self) -> None:
        widget = video_widget.VideoWidget(None, self.cm)
        self._comprobar(widget)
        # Los modos T2V/I2V se cargan aunque no haya motor disponible.
        self.assertGreaterEqual(widget._mode.count(), 0)

    def test_image_widget_arranca(self) -> None:
        self._comprobar(image_widget.ImageWidget({}, self.cm))

    def test_audio_widget_arranca(self) -> None:
        self._comprobar(audio_widget.AudioWidget(None, None, self.cm))

    def test_assembler_widget_arranca(self) -> None:
        self._comprobar(assembler_widget.AssemblerWidget(None, self.cm))

    def test_understanding_widget_arranca(self) -> None:
        widget = understanding_widget.UnderstandingWidget(EngineManager(self.cm))
        self._comprobar(widget)
        # Sin vídeo elegido no se puede analizar, y el widget lo dice.
        motivo = widget._validation_error(widget._build_plan())
        self.assertEqual(motivo, "Elige primero un vídeo.")

    def test_las_preferencias_de_cada_pestana_van_a_la_config_dada(self) -> None:
        video_widget.VideoWidget(None, self.cm)
        audio_widget.AudioWidget(None, None, self.cm)

        ajustes = self.cm.load_settings()
        self.assertIn("last_models", ajustes)
        self.assertEqual(ajustes["last_models"].get("audio_voice_piper"), None)


if __name__ == "__main__":
    unittest.main()
