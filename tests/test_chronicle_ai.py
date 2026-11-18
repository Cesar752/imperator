"""Pestaña Crónica: elección de IA (visión/chat) y ayuda de Comprensión/Cuaderno.

Se construye el widget en modo offscreen con una ``ConfigManager`` temporal (no se
toca ``config/`` real). No se hace red: los análisis se sustituyen por dobles.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from core.config_manager import ConfigManager  # noqa: E402
from core.engine_manager import EngineManager  # noqa: E402
from ui import chronicle_widget  # noqa: E402


class ChronicleIaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.cm = ConfigManager(config_dir=self._tmp.name)
        self.manager = EngineManager(self.cm)
        # El listado de modelos del motor se hace en segundo plano: en tests se
        # sustituye por un doble para no tocar red.
        self._pool = mock.patch.object(
            chronicle_widget, "run_pool_task", mock.MagicMock()
        )
        self._pool.start()
        self.widget = chronicle_widget.ChronicleWidget(self.manager)

    def tearDown(self) -> None:
        self.widget.close()
        self._pool.stop()
        self._tmp.cleanup()

    def test_los_selectores_de_ia_listan_los_motores_disponibles(self) -> None:
        # Ollama Vision está habilitado por defecto: es la opción local y gratis.
        self.assertGreaterEqual(self.widget._cmb_vision_engine.findData("ollama_vision"), 0)
        # El chat lista Ollama (el motor de chat por defecto).
        self.assertGreaterEqual(self.widget._cmb_chat_engine.findData("ollama"), 0)

    def test_avisa_del_coste_al_elegir_vision_de_pago(self) -> None:
        # Habilitar el motor de visión de pago (OpenAI Vision) y reconstruir.
        self.manager.set_enabled("openai_vision", True)
        widget = chronicle_widget.ChronicleWidget(self.manager)
        try:
            indice = widget._cmb_vision_engine.findData("openai_vision")
            self.assertGreaterEqual(indice, 0)
            widget._cmb_vision_engine.setCurrentIndex(indice)
            self.assertIn("pago", widget._lbl_ia.text())
        finally:
            widget.close()

    def test_los_modelos_elegidos_llegan_al_worker(self) -> None:
        self.widget._video = Path("peli.mp4")
        # Elige el modelo de visión configurado (no «por defecto»).
        self.assertGreater(self.widget._cmb_vision_model.count(), 1)
        self.widget._cmb_vision_model.setCurrentIndex(1)
        self.assertGreater(self.widget._cmb_chat_model.count(), 1)
        self.widget._cmb_chat_model.setCurrentIndex(1)
        with mock.patch.object(self.widget, "_start") as start:
            self.widget._generate()
        worker = start.call_args.args[0]
        self.assertEqual(worker._vision_model, "qwen2.5vl:7b")
        self.assertEqual(worker._chat_model, "qwen3.6:35b")

    def test_la_percepcion_de_comprension_alimenta_la_cronica(self) -> None:
        ruta = Path(self._tmp.name) / "comprension.json"
        ruta.write_text(
            '{"video_path": "", "duracion_s": 100.0, '
            '"contexto": "Alineación Muskis: 4 Álex", '
            '"resumen": "Partido igualado", '
            '"fotogramas": [{"t_s": 12.0, "anotacion": "gol de Álex"}], '
            '"transcripcion": ""}',
            encoding="utf-8",
        )
        self.widget._load_percepcion(str(ruta))
        self.assertTrue(self.widget._chk_informe.isChecked())
        # Con material cargado no se mira el vídeo: la visión queda deshabilitada.
        self.assertFalse(self.widget._cmb_vision_engine.isEnabled())
        self.assertIn("Álex", self.widget._txt_context.toPlainText())
        self.widget._video = Path("peli.mp4")
        with mock.patch.object(self.widget, "_start") as start:
            self.widget._generate()
        worker = start.call_args.args[0]
        self.assertIn("gol de Álex", worker._report)
        self.assertIn("Partido igualado", worker._report)

    def test_sin_material_se_analiza_con_el_motor_de_vision(self) -> None:
        self.widget._video = Path("peli.mp4")
        with mock.patch.object(self.widget, "_start") as start:
            self.widget._generate()
        worker = start.call_args.args[0]
        self.assertEqual(worker._report, "")
        self.assertIsNotNone(worker._vision)
        self.assertTrue(self.widget._cmb_vision_engine.isEnabled())

    def test_los_modelos_reales_del_motor_se_fusionan(self) -> None:
        """Se pueden elegir los modelos instalados/en la API, no solo el configurado."""
        with mock.patch.object(
            chronicle_widget,
            "run_pool_task",
            lambda fn, on_result=None, on_error=None: on_result(
                ["qwen2.5vl:72b", "llava"]
            ),
        ):
            widget = chronicle_widget.ChronicleWidget(self.manager)
        try:
            datos = {
                widget._cmb_vision_model.itemData(i)
                for i in range(widget._cmb_vision_model.count())
            }
            self.assertIn("qwen2.5vl:72b", datos)
            self.assertIn("llava", datos)
        finally:
            widget.close()

    def test_deepseek_escribe_pero_no_ve(self) -> None:
        """DeepSeek es sólo texto: nunca debe ofrecerse como motor de visión."""
        self.manager.set_enabled("deepseek", True)
        widget = chronicle_widget.ChronicleWidget(self.manager)
        try:
            self.assertEqual(widget._cmb_vision_engine.findData("deepseek"), -1)
            self.assertGreaterEqual(widget._cmb_chat_engine.findData("deepseek"), 0)
        finally:
            widget.close()


if __name__ == "__main__":
    unittest.main()
