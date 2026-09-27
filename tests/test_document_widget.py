"""Tests del aviso de idioma del widget de Documentos.

Resumir o extraer tablas exige *entender* el documento: si el modelo de visión
elegido no cubre idiomas minoritarios (euskera), el widget lo avisa en la barra
de estado y en el tooltip del modelo, sin bloquear el trabajo.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import Qt  # noqa: E402

from core.config_manager import ConfigManager  # noqa: E402
from ui import document_widget as docw  # noqa: E402


class DocumentLanguageHintTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from PyQt6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.cm = ConfigManager(config_dir=self._tmp.name)
        self.widget = docw.DocumentWidget(None, self.cm)

    def tearDown(self) -> None:
        self.widget.close()
        self._tmp.cleanup()

    def _tooltip(self, index: int) -> str:
        return self.widget._model.itemData(index, Qt.ItemDataRole.ToolTipRole)

    def _modo(self, mode_id: str) -> None:
        self.widget._mode.setCurrentIndex(self.widget._mode.findData(mode_id))

    def test_marca_en_el_tooltip_los_modelos_con_cobertura(self) -> None:
        self.widget._on_models(["qwen3-vl:30b-a3b", "llava:13b"])
        self.assertIn("multilingüe", self._tooltip(0))
        self.assertIn("euskera", self._tooltip(1))

    def test_avisa_al_resumir_con_un_modelo_sin_cobertura(self) -> None:
        self.widget._on_models(["llava:13b"])
        self._modo("resumen")
        self.assertIn("euskera", self.widget._status.text())

    def test_no_avisa_con_un_modelo_multilingue(self) -> None:
        self.widget._on_models(["qwen3-vl:30b-a3b"])
        self._modo("resumen")
        self.assertNotIn("euskera", self.widget._status.text())

    def test_en_ocr_no_avisa_aunque_el_modelo_no_cubra_el_idioma(self) -> None:
        self.widget._on_models(["llava:13b"])
        self._modo("resumen")
        self.assertIn("euskera", self.widget._status.text())

        self._modo("ocr")
        self.assertEqual(self.widget._status.text(), "")

    def test_no_pisa_otros_mensajes_de_estado(self) -> None:
        self.widget._on_models(["llava:13b"])
        self._modo("resumen")
        self.widget._set_status("Error: se cayó Ollama")

        self._modo("ocr")
        self.assertEqual(self.widget._status.text(), "Error: se cayó Ollama")


if __name__ == "__main__":
    unittest.main()
