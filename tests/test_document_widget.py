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
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import Qt  # noqa: E402

from core.config_manager import ConfigManager  # noqa: E402
from services import document_params as dp  # noqa: E402
from services import preferences as prefs  # noqa: E402
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
        self.widget._on_models(
            self.widget._generation, ["qwen3-vl:30b-a3b", "llava:13b"]
        )
        self.assertIn("multilingüe", self._tooltip(0))
        self.assertIn("euskera", self._tooltip(1))

    def test_avisa_al_resumir_con_un_modelo_sin_cobertura(self) -> None:
        self.widget._on_models(self.widget._generation, ["llava:13b"])
        self._modo("resumen")
        self.assertIn("euskera", self.widget._status.text())

    def test_no_avisa_con_un_modelo_multilingue(self) -> None:
        self.widget._on_models(self.widget._generation, ["qwen3-vl:30b-a3b"])
        self._modo("resumen")
        self.assertNotIn("euskera", self.widget._status.text())

    def test_en_ocr_no_avisa_aunque_el_modelo_no_cubra_el_idioma(self) -> None:
        self.widget._on_models(self.widget._generation, ["llava:13b"])
        self._modo("resumen")
        self.assertIn("euskera", self.widget._status.text())

        self._modo("ocr")
        self.assertEqual(self.widget._status.text(), "")

    def test_no_pisa_otros_mensajes_de_estado(self) -> None:
        self.widget._on_models(self.widget._generation, ["llava:13b"])
        self._modo("resumen")
        self.widget._set_status("Error: se cayó Ollama")

        self._modo("ocr")
        self.assertEqual(self.widget._status.text(), "Error: se cayó Ollama")

    def test_el_selector_de_idioma_tiene_las_opciones(self) -> None:
        combo = self.widget._language_combo
        labels = [combo.itemText(i) for i in range(combo.count())]
        self.assertEqual(list(dp.LANGUAGES.values()), labels)
        self.assertEqual(combo.currentData(), dp.DEFAULT_LANGUAGE)

    def test_el_idioma_elegido_se_guarda(self) -> None:
        combo = self.widget._language_combo
        combo.setCurrentIndex(combo.findData("eu"))
        self.assertEqual(self.cm.load_last_models()[prefs.DOCUMENT_LANGUAGE], "eu")

    def test_restaura_el_idioma_guardado(self) -> None:
        self.cm.save_last_model(prefs.DOCUMENT_LANGUAGE, "es")
        widget = docw.DocumentWidget(None, self.cm)
        try:
            self.assertEqual(widget._language_combo.currentData(), "es")
        finally:
            widget.close()

    def test_el_idioma_elegido_se_pasa_al_worker(self) -> None:
        combo = self.widget._language_combo
        combo.setCurrentIndex(combo.findData("eu"))
        self.widget._file_path = Path("documento.pdf")
        with mock.patch.object(docw, "DocumentWorker") as worker_cls:
            with mock.patch.object(self.widget, "_set_busy"):
                self.widget._engine = mock.Mock(is_available=lambda: True)
                self.widget._process()
        self.assertEqual(worker_cls.call_args.args[4], "eu")


if __name__ == "__main__":
    unittest.main()
