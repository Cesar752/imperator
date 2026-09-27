"""Tests del resumen global de documentos (alcance "documento completo").

Por defecto cada página se resume por separado, así que el modelo nunca ve el
conjunto. Con la opción activada se recoge el texto de todas las páginas (capa
nativa u OCR) y se hace una única llamada de resumen sobre el documento entero.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from core.config_manager import ConfigManager  # noqa: E402
from engines.base_engine import EngineError  # noqa: E402
from services import document_params as dp  # noqa: E402
from services import preferences as prefs  # noqa: E402
from ui import document_widget as dw  # noqa: E402

_TEXTO = "contenido nativo de la pagina " * 4


class _FakeEngine:
    """Motor de visión de mentira que registra las llamadas recibidas."""

    def __init__(self) -> None:
        self.image_calls: list[tuple] = []
        self.text_calls: list[tuple] = []

    def analyze_image(self, path, prompt, model=None) -> str:
        self.image_calls.append((str(path), prompt, model))
        return "TEXTO-OCR"

    def analyze_text(self, text, prompt, model=None) -> str:
        self.text_calls.append((text, prompt, model))
        return "RESULTADO"


class _FakePage:
    def __init__(self, text: str) -> None:
        self._text = text
        self.rasterized = False

    def get_text(self) -> str:
        return self._text

    def get_pixmap(self, dpi=None):
        self.rasterized = True
        return _FakePixmap()


class _FakePixmap:
    def tobytes(self, fmt: str) -> bytes:
        return b"png"


class _FakeDoc:
    def __init__(self, pages) -> None:
        self._pages = list(pages)
        self.page_count = len(self._pages)
        self.closed = False

    def __iter__(self):
        return iter(self._pages)

    def close(self) -> None:
        self.closed = True


class _FakePymupdf:
    def __init__(self, doc: _FakeDoc) -> None:
        self._doc = doc

    def open(self, path: str) -> _FakeDoc:
        return self._doc


class GlobalParamsTest(unittest.TestCase):
    """Funciones puras: qué modos admiten el alcance global y cómo se unen."""

    def test_solo_el_resumen_admite_alcance_global(self) -> None:
        self.assertTrue(dp.supports_global("resumen"))
        for mode in ("ocr", "tablas", "", None):
            self.assertFalse(dp.supports_global(mode))

    def test_hace_falta_que_la_opcion_este_activada(self) -> None:
        self.assertTrue(dp.wants_global("resumen", True))
        self.assertFalse(dp.wants_global("resumen", False))
        self.assertFalse(dp.wants_global("tablas", True))
        self.assertFalse(dp.wants_global(None, True))

    def test_el_prompt_global_no_es_el_de_pagina(self) -> None:
        global_prompt = dp.global_prompt_for("resumen")
        self.assertIsNotNone(global_prompt)
        self.assertNotEqual(global_prompt, dp.text_prompt_for("resumen"))
        self.assertIn("complete document", global_prompt)
        self.assertIn("instructions", global_prompt)  # regla de no comentar

    def test_el_prompt_global_respeta_el_idioma(self) -> None:
        prompt = dp.global_prompt_for("resumen", "eu")
        self.assertIn("Basque (euskera)", prompt)
        self.assertNotIn("SAME language", prompt)

    def test_no_hay_prompt_global_en_los_demas_modos(self) -> None:
        for mode in ("ocr", "tablas", None):
            self.assertIsNone(dp.global_prompt_for(mode))

    def test_merge_pages_une_y_descarta_las_vacias(self) -> None:
        texto, recortado = dp.merge_pages(["uno", "", "  dos  "])
        self.assertEqual(texto, "uno\n\ndos")
        self.assertFalse(recortado)

    def test_merge_pages_recorta_lo_que_no_cabe(self) -> None:
        texto, recortado = dp.merge_pages(["abcdefghij", "klmnopqrst"], max_chars=12)
        self.assertTrue(recortado)
        self.assertEqual(len(texto), 12)
        self.assertTrue(texto.startswith("abcdefghij"))


class GlobalWorkerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from PyQt6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.pdf = Path(self._tmp.name) / "documento.pdf"
        self.pdf.write_bytes(b"%PDF-1.4 (falso para el test)")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run(self, pages, mode="resumen", global_summary=True):
        engine = _FakeEngine()
        doc = _FakeDoc(pages)
        worker = dw.DocumentWorker(
            engine, self.pdf, mode, "modelo-x", global_summary=global_summary
        )
        with mock.patch.object(dw, "pymupdf", _FakePymupdf(doc)):
            result = worker.work()
        return result, engine, doc

    def test_resume_el_documento_completo_en_una_sola_llamada(self) -> None:
        result, engine, doc = self._run([_FakePage(_TEXTO), _FakePage(_TEXTO)])
        self.assertEqual(result, "RESULTADO")
        self.assertEqual(len(engine.text_calls), 1, "un único resumen, no uno por página")
        joined, prompt, model = engine.text_calls[0]
        self.assertEqual(prompt, dp.global_prompt_for("resumen"))
        self.assertEqual(model, "modelo-x")
        self.assertEqual(joined.count(_TEXTO.strip()), 2, "entran las dos páginas")
        self.assertNotIn("--- Página", result, "no se etiqueta página a página")
        self.assertEqual(engine.image_calls, [])
        self.assertTrue(doc.closed)

    def test_las_paginas_sin_texto_se_transcriben_para_el_resumen(self) -> None:
        _, engine, _ = self._run([_FakePage(_TEXTO), _FakePage("")])
        self.assertEqual(len(engine.image_calls), 1, "la página sin texto va por OCR")
        self.assertEqual(engine.image_calls[0][1], dp.prompt_for("ocr"))
        self.assertIn("TEXTO-OCR", engine.text_calls[0][0])

    def test_avisa_cuando_el_documento_no_cabe_entero(self) -> None:
        largo = "palabra " * (dp.MAX_GLOBAL_CHARS // 7 + 100)
        result, engine, _ = self._run([_FakePage(largo)])
        self.assertIn(dp.TRUNCATED_NOTICE, result)
        self.assertEqual(len(engine.text_calls[0][0]), dp.MAX_GLOBAL_CHARS)

    def test_sin_la_opcion_sigue_siendo_pagina_a_pagina(self) -> None:
        result, engine, _ = self._run([_FakePage(_TEXTO)], global_summary=False)
        self.assertIn("--- Página 1 (texto) ---", result)
        self.assertEqual(engine.text_calls[0][1], dp.text_prompt_for("resumen"))

    def test_en_tablas_la_opcion_no_cambia_el_comportamiento(self) -> None:
        result, engine, _ = self._run(
            [_FakePage(_TEXTO)], mode="tablas", global_summary=True
        )
        self.assertIn("--- Página 1 (texto) ---", result)
        self.assertEqual(engine.text_calls[0][1], dp.text_prompt_for("tablas"))

    def test_cancelacion_detectada_en_la_pasada_de_texto(self) -> None:
        class _CancelaDespuesDeLaPrimera(dw.DocumentWorker):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self._comprobaciones = 0

            def isInterruptionRequested(self) -> bool:
                self._comprobaciones += 1
                return self._comprobaciones > 1

        engine = _FakeEngine()
        doc = _FakeDoc([_FakePage(_TEXTO), _FakePage(_TEXTO)])
        worker = _CancelaDespuesDeLaPrimera(
            engine, self.pdf, "resumen", "m", global_summary=True
        )
        with mock.patch.object(dw, "pymupdf", _FakePymupdf(doc)):
            with self.assertRaises(EngineError) as ctx:
                worker.work()
        self.assertEqual(str(ctx.exception), dw.CANCEL_MESSAGE)
        self.assertTrue(doc.closed)


class GlobalWidgetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from PyQt6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.cm = ConfigManager(config_dir=self._tmp.name)
        self.widget = dw.DocumentWidget(None, self.cm)

    def tearDown(self) -> None:
        self.widget.close()
        self._tmp.cleanup()

    def _modo(self, mode_id: str) -> None:
        self.widget._mode.setCurrentIndex(self.widget._mode.findData(mode_id))

    def test_la_casilla_solo_se_activa_en_el_modo_resumen(self) -> None:
        for mode_id, esperado in (("ocr", False), ("tablas", False), ("resumen", True)):
            with self.subTest(modo=mode_id):
                self._modo(mode_id)
                self.assertEqual(self.widget._global_check.isEnabled(), esperado)

    def test_el_estado_de_la_casilla_se_guarda(self) -> None:
        self.widget._global_check.setChecked(True)
        self.assertEqual(self.cm.load_last_models()[prefs.DOCUMENT_GLOBAL], "1")

    def test_restaura_la_casilla_guardada(self) -> None:
        self.cm.save_last_model(prefs.DOCUMENT_GLOBAL, "1")
        widget = dw.DocumentWidget(None, self.cm)
        try:
            self.assertTrue(widget._global_check.isChecked())
        finally:
            widget.close()

    def test_la_casilla_se_bloquea_durante_el_trabajo(self) -> None:
        self._modo("resumen")
        self.widget._set_busy(True)
        self.assertFalse(self.widget._global_check.isEnabled())

        self.widget._set_busy(False)
        self.assertTrue(self.widget._global_check.isEnabled())

    def test_la_opcion_elegida_llega_al_worker(self) -> None:
        self._modo("resumen")
        self.widget._global_check.setChecked(True)
        self.widget._file_path = Path("documento.pdf")
        with mock.patch.object(dw, "DocumentWorker") as worker_cls:
            with mock.patch.object(self.widget, "_set_busy"):
                self.widget._engine = mock.Mock(is_available=lambda: True)
                self.widget._process()
        self.assertTrue(worker_cls.call_args.kwargs["global_summary"])


if __name__ == "__main__":
    unittest.main()
