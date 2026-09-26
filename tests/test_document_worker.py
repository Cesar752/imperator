"""Tests del DocumentWorker: el modo debe aplicarse también a PDFs con texto.

Reproduce el bug reportado: en un PDF con capa de texto nativa, OCR, resumen y
tablas devolvían el mismo resultado porque el texto se entregaba en crudo sin
pasar por el prompt del modo.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# QThread (base del worker) necesita un QApplication en el proceso.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from engines.base_engine import EngineError  # noqa: E402
from services import document_params as dp  # noqa: E402
from ui import document_widget as dw  # noqa: E402

_TEXT = "contenido nativo del pdf " * 4


class _FakeEngine:
    """Motor de visión de mentira que registra las llamadas recibidas."""

    def __init__(self) -> None:
        self.image_calls: list[tuple] = []
        self.text_calls: list[tuple] = []

    def analyze_image(self, path, prompt, model=None) -> str:
        self.image_calls.append((str(path), prompt, model))
        return "RESULTADO-IMAGEN"

    def analyze_text(self, text, prompt, model=None) -> str:
        self.text_calls.append((text, prompt, model))
        return "RESULTADO-TEXTO"


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
        self.opened: list[str] = []

    def open(self, path: str) -> _FakeDoc:
        self.opened.append(path)
        return self._doc


class DocumentWorkerPdfTest(unittest.TestCase):
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

    def _run(self, mode: str, pages):
        engine = _FakeEngine()
        doc = _FakeDoc(pages)
        worker = dw.DocumentWorker(engine, self.pdf, mode, "modelo-x")
        with mock.patch.object(dw, "pymupdf", _FakePymupdf(doc)):
            result = worker.work()
        return result, engine, doc

    def test_resumen_analiza_el_texto_nativo(self) -> None:
        result, engine, doc = self._run("resumen", [_FakePage(_TEXT)])
        self.assertIn("RESULTADO-TEXTO", result)
        self.assertIn("--- Página 1 (texto) ---", result)
        self.assertEqual(len(engine.text_calls), 1)
        text, prompt, model = engine.text_calls[0]
        self.assertEqual(text, _TEXT.strip())
        self.assertEqual(prompt, dp.text_prompt_for("resumen"))
        self.assertEqual(model, "modelo-x")
        self.assertEqual(engine.image_calls, [], "no debe rasterizar con texto nativo")
        self.assertTrue(doc.closed)

    def test_tablas_y_resumen_usan_prompts_distintos(self) -> None:
        _, resumen, _ = self._run("resumen", [_FakePage(_TEXT)])
        _, tablas, _ = self._run("tablas", [_FakePage(_TEXT)])
        self.assertNotEqual(
            resumen.text_calls[0][1], tablas.text_calls[0][1]
        )
        self.assertNotEqual(
            resumen.text_calls[0][1], dp.prompt_for("resumen")
        )

    def test_ocr_devuelve_el_texto_sin_llamar_al_modelo(self) -> None:
        result, engine, _ = self._run("ocr", [_FakePage(_TEXT)])
        self.assertIn(_TEXT.strip(), result)
        self.assertIn("--- Página 1 (texto) ---", result)
        self.assertEqual(engine.text_calls, [])
        self.assertEqual(engine.image_calls, [])

    def test_pagina_sin_texto_va_por_ocr_con_su_prompt(self) -> None:
        result, engine, _ = self._run("resumen", [_FakePage("")])
        self.assertIn("RESULTADO-IMAGEN", result)
        self.assertIn("--- Página 1 (OCR) ---", result)
        self.assertEqual(len(engine.image_calls), 1)
        self.assertEqual(engine.image_calls[0][1], dp.prompt_for("resumen"))
        self.assertEqual(engine.text_calls, [])

    def test_mezcla_de_paginas_separa_texto_y_ocr(self) -> None:
        result, engine, _ = self._run(
            "tablas", [_FakePage(_TEXT), _FakePage("")]
        )
        self.assertIn("--- Página 1 (texto) ---", result)
        self.assertIn("--- Página 2 (OCR) ---", result)
        self.assertEqual(len(engine.text_calls), 1)
        self.assertEqual(len(engine.image_calls), 1)

    def test_imagen_suelta_usa_el_prompt_del_modo(self) -> None:
        image = Path(self._tmp.name) / "foto.png"
        image.write_bytes(b"png")
        engine = _FakeEngine()
        worker = dw.DocumentWorker(engine, image, "tablas", "modelo-y")
        result = worker.work()
        self.assertEqual(result, "RESULTADO-IMAGEN")
        self.assertEqual(engine.image_calls[0][1], dp.prompt_for("tablas"))

    def test_cancelacion_detectada_entre_paginas(self) -> None:
        # QThread.requestInterruption() no marca nada si el hilo no corre, así
        # que se simula la cancelación a partir de la segunda comprobación.
        class _CancelaDespuesDeLaPrimera(dw.DocumentWorker):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self._comprobaciones = 0

            def isInterruptionRequested(self) -> bool:
                self._comprobaciones += 1
                return self._comprobaciones > 1

        engine = _FakeEngine()
        doc = _FakeDoc([_FakePage(_TEXT), _FakePage(_TEXT)])
        worker = _CancelaDespuesDeLaPrimera(engine, self.pdf, "ocr", "m")
        with mock.patch.object(dw, "pymupdf", _FakePymupdf(doc)):
            with self.assertRaises(EngineError) as ctx:
                worker.work()
        self.assertEqual(str(ctx.exception), dw.CANCEL_MESSAGE)
        self.assertTrue(doc.closed)


if __name__ == "__main__":
    unittest.main()
