"""Tests del DocumentWorker: el modo debe aplicarse también a PDFs con texto.

Reproduce el bug reportado: en un PDF con capa de texto nativa, OCR, resumen y
tablas devolvían el mismo resultado porque el texto se entregaba en crudo sin
pasar por el prompt del modo.

Incluye el procesado por lotes: las páginas se analizan de ``parallel`` en
``parallel`` (medido 1,68x con dos), pero el orden del PDF, el progreso y la
cancelación se mantienen.
"""

import os
import sys
import tempfile
import threading
import time
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


class _EchoConcurrentEngine(_FakeEngine):
    """Motor que hace eco del texto y mide el paralelismo real.

    ``max_en_vuelo`` es el máximo de peticiones que coincidieron en el tiempo: es
    la forma de comprobar que las páginas se analizan a la vez de verdad.
    """

    def __init__(self, retardo: float = 0.05) -> None:
        super().__init__()
        self._retardo = retardo
        self._lock = threading.Lock()
        self.en_vuelo = 0
        self.max_en_vuelo = 0

    def _entrar(self) -> None:
        with self._lock:
            self.en_vuelo += 1
            self.max_en_vuelo = max(self.max_en_vuelo, self.en_vuelo)
        time.sleep(self._retardo)

    def _salir(self) -> None:
        with self._lock:
            self.en_vuelo -= 1

    def analyze_text(self, text, prompt, model=None) -> str:
        self._entrar()
        try:
            self.text_calls.append((text, prompt, model))
            return f"ECO[{text[-8:]}]"
        finally:
            self._salir()

    def analyze_image(self, path, prompt, model=None) -> str:
        self._entrar()
        try:
            self.image_calls.append((str(path), prompt, model))
            return "ECO-IMAGEN"
        finally:
            self._salir()


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

    def load_page(self, index: int):
        return self._pages[index]

    def close(self) -> None:
        self.closed = True


class _FakePymupdf:
    def __init__(self, doc: _FakeDoc) -> None:
        self._doc = doc
        self.opened: list[str] = []

    def open(self, path: str) -> _FakeDoc:
        self.opened.append(path)
        return self._doc


class _QtTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from PyQt6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])


class DocumentWorkerPdfTest(_QtTestCase):
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

    def test_el_idioma_elegido_llega_al_prompt(self) -> None:
        engine = _FakeEngine()
        doc = _FakeDoc([_FakePage(_TEXT)])
        worker = dw.DocumentWorker(engine, self.pdf, "resumen", "m", "eu")
        with mock.patch.object(dw, "pymupdf", _FakePymupdf(doc)):
            worker.work()
        prompt = engine.text_calls[0][1]
        self.assertIn("Basque (euskera)", prompt)
        self.assertNotIn("SAME language", prompt)

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


class DocumentWorkerParallelTest(_QtTestCase):
    """Las páginas se analizan por lotes, sin perder orden ni cancelación."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.pdf = Path(self._tmp.name) / "documento.pdf"
        self.pdf.write_bytes(b"%PDF-1.4 (falso para el test)")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _paginas(self, total: int):
        # Cada página lleva su marca para poder comprobar el orden.
        return [_FakePage(f"{_TEXT} pagina {i}") for i in range(total)]

    def _run(self, pages, parallel):
        engine = _EchoConcurrentEngine()
        doc = _FakeDoc(pages)
        # Alcance por página a propósito: este test mide el procesado por lotes,
        # y con el alcance «automático» un documento de 6 páginas se resumiría
        # de una vez (otro camino, con otro número de llamadas).
        worker = dw.DocumentWorker(
            engine,
            self.pdf,
            "resumen",
            "modelo-x",
            parallel=parallel,
            scope=dp.SCOPE_PAGES,
        )
        with mock.patch.object(dw, "pymupdf", _FakePymupdf(doc)):
            result = worker.work()
        return result, engine

    def test_las_paginas_se_analizan_de_dos_en_dos(self) -> None:
        _, engine = self._run(self._paginas(4), parallel=2)
        self.assertEqual(engine.max_en_vuelo, 2, "deben coincidir dos peticiones")
        self.assertEqual(len(engine.text_calls), 4)

    def test_con_parallel_uno_van_de_una_en_una(self) -> None:
        _, engine = self._run(self._paginas(3), parallel=1)
        self.assertEqual(engine.max_en_vuelo, 1)
        self.assertEqual(len(engine.text_calls), 3)

    def test_el_orden_de_las_paginas_se_conserva(self) -> None:
        result, _ = self._run(self._paginas(4), parallel=2)
        posiciones = [result.index(f"--- Página {i} (texto) ---") for i in range(1, 5)]
        self.assertEqual(posiciones, sorted(posiciones))

    def test_cada_pagina_recibe_su_propio_resultado(self) -> None:
        result, _ = self._run(self._paginas(4), parallel=2)
        for i in range(4):
            with self.subTest(pagina=i + 1):
                cuerpo = result.split(f"--- Página {i + 1} (texto) ---")[1]
                self.assertIn(f"ECO[pagina {i}]", cuerpo.split("---")[0])

    def test_la_cancelacion_para_entre_lotes(self) -> None:
        class _CancelaEnElTercerChequeo(dw.DocumentWorker):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self._comprobaciones = 0

            def isInterruptionRequested(self) -> bool:
                self._comprobaciones += 1
                return self._comprobaciones > 2

        engine = _EchoConcurrentEngine()
        doc = _FakeDoc(self._paginas(6))
        worker = _CancelaEnElTercerChequeo(
            engine, self.pdf, "resumen", "m", parallel=2, scope=dp.SCOPE_PAGES
        )
        with mock.patch.object(dw, "pymupdf", _FakePymupdf(doc)):
            with self.assertRaises(EngineError) as ctx:
                worker.work()
        self.assertEqual(str(ctx.exception), dw.CANCEL_MESSAGE)
        self.assertEqual(len(engine.text_calls), 2, "el primer lote sí se analizó")


if __name__ == "__main__":
    unittest.main()
