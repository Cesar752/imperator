"""Tests del pipeline por fases de documentos.

Dos niveles:

- **El plan** (:mod:`services.document_pipeline`): qué fases se ejecutan, por
  dónde va cada página (texto embebido u OCR), la normalización, la validación
  ligera y la entrega. Sin Qt y sin modelo: son funciones puras.
- **La ejecución** (:class:`ui.document_widget.DocumentWorker`): que el worker
  respete ese plan — OCR forzado, tablas que acompañan al resumen y los avisos
  de la validación.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from services import document_params as dp  # noqa: E402
from services import document_pipeline as dpl  # noqa: E402
from services import preferences as prefs  # noqa: E402
from tests import test_document_worker as wd  # noqa: E402
from ui import document_widget as dw  # noqa: E402


class PlanTest(unittest.TestCase):
    """FASE 0: qué decide el plan a partir de las opciones de la UI."""

    def test_solo_leer_no_lleva_resumen_ni_fases_de_analisis(self) -> None:
        plan = dpl.from_ui("ocr")
        self.assertTrue(plan.transcribes_only)
        self.assertEqual(plan.summary, dpl.SUMMARY_NONE)
        self.assertFalse(plan.tables)
        self.assertNotIn(dpl.FASE_NORMALIZACION, plan.phases())

    def test_el_resumen_por_defecto_es_por_apartados(self) -> None:
        plan = dpl.from_ui("resumen")
        self.assertEqual(plan.summary, dpl.SUMMARY_SECTIONS)
        self.assertFalse(plan.wants_global)
        self.assertIn(dpl.FASE_RESUMEN_APARTADOS, plan.phases())
        self.assertIn(dpl.FASE_NORMALIZACION, plan.phases())

    def test_la_casilla_global_hace_una_sola_pasada(self) -> None:
        plan = dpl.from_ui("resumen", global_summary=True)
        self.assertEqual(plan.summary, dpl.SUMMARY_GLOBAL)
        self.assertTrue(plan.wants_global)
        self.assertIn(dpl.FASE_RESUMEN_GLOBAL, plan.phases())
        self.assertNotIn(dpl.FASE_RESUMEN_APARTADOS, plan.phases())

    def test_el_modo_tablas_no_pide_bloque_extra(self) -> None:
        # En modo Extraer tablas, las tablas *son* el cuerpo: no se duplican.
        plan = dpl.from_ui("tablas")
        self.assertTrue(plan.tables)
        self.assertFalse(plan.tables_extra())
        self.assertIn(dpl.FASE_TABLAS, plan.phases())

    def test_las_tablas_pueden_acompanar_al_resumen(self) -> None:
        plan = dpl.from_ui("resumen", extra_tables=True)
        self.assertTrue(plan.tables)
        self.assertTrue(plan.tables_extra())

    def test_las_tablas_pueden_acompanar_a_la_transcripcion(self) -> None:
        plan = dpl.from_ui("ocr", extra_tables=True)
        self.assertTrue(plan.tables_extra(), "en OCR también son un bloque extra")
        self.assertEqual(plan.summary, dpl.SUMMARY_NONE, "siguen sin resumen")

    def test_forzar_ocr_anade_su_fase(self) -> None:
        plan = dpl.from_ui("resumen", force_ocr=True)
        self.assertIn(dpl.FASE_OCR, plan.phases())
        self.assertNotIn(dpl.FASE_OCR, dpl.from_ui("resumen").phases())

    def test_la_validacion_y_la_entrega_siempre_estan(self) -> None:
        for opciones in ({}, {"force_ocr": True}, {"extra_tables": True}):
            with self.subTest(opciones=opciones):
                fases = dpl.from_ui("resumen", **opciones).phases()
                self.assertEqual(fases[0], dpl.FASE_DETECCION)
                self.assertEqual(fases[-2:], (dpl.FASE_VALIDACION, dpl.FASE_ENTREGA))

    def test_parallel_queda_normalizado(self) -> None:
        self.assertEqual(dpl.from_ui("ocr", parallel=0).parallel, 1)
        self.assertEqual(dpl.from_ui("ocr", parallel=2).parallel, 2)


    def test_el_alcance_automatico_resuelve_segun_las_paginas(self) -> None:
        plan = dpl.from_ui("resumen")
        self.assertEqual(plan.with_page_count(2).summary, dpl.SUMMARY_SECTIONS)
        self.assertEqual(
            plan.with_page_count(dp.AUTO_GLOBAL_MIN_PAGES).summary, dpl.SUMMARY_GLOBAL
        )
        self.assertTrue(plan.with_page_count(19).wants_global)

    def test_el_alcance_global_forzado_no_depende_de_las_paginas(self) -> None:
        plan = dpl.from_ui("resumen", scope=dp.SCOPE_GLOBAL)
        self.assertTrue(plan.with_page_count(1).wants_global)

    def test_el_alcance_por_pagina_a_mano_se_respeta(self) -> None:
        plan = dpl.from_ui("resumen", scope=dp.SCOPE_PAGES)
        self.assertEqual(plan.with_page_count(50).summary, dpl.SUMMARY_SECTIONS)

    def test_el_alcance_no_cambia_los_modos_sin_resumen(self) -> None:
        for modo in ("ocr", "tablas"):
            with self.subTest(modo=modo):
                plan = dpl.from_ui(modo, scope=dp.SCOPE_GLOBAL)
                self.assertEqual(plan.with_page_count(50).summary, dpl.SUMMARY_NONE)

    def test_normalize_scope_lee_los_valores_heredados(self) -> None:
        self.assertEqual(dp.normalize_scope("1"), dp.SCOPE_GLOBAL)
        self.assertEqual(dp.normalize_scope(dp.SCOPE_GLOBAL), dp.SCOPE_GLOBAL)
        self.assertEqual(dp.normalize_scope(dp.SCOPE_PAGES), dp.SCOPE_PAGES)
        for heredado in ("0", dp.SCOPE_AUTO, None, "vete a saber"):
            with self.subTest(valor=heredado):
                self.assertEqual(dp.normalize_scope(heredado), dp.SCOPE_AUTO)

    def test_la_antigua_casilla_global_se_sigue_entendiendo(self) -> None:
        self.assertTrue(dpl.from_ui("resumen", global_summary=True).wants_global)


class RouteTest(unittest.TestCase):
    """FASE 0: con texto nativo se usa el texto; sin él, no hay alternativa."""

    def test_con_texto_nativo_va_por_texto(self) -> None:
        self.assertEqual(dpl.from_ui("resumen").route(True), dpl.ROUTE_TEXT)

    def test_sin_texto_nativo_va_por_ocr(self) -> None:
        self.assertEqual(dpl.from_ui("resumen").route(False), dpl.ROUTE_OCR)

    def test_forzar_ocr_ignora_la_capa_de_texto(self) -> None:
        plan = dpl.from_ui("resumen", force_ocr=True)
        self.assertEqual(plan.route(True), dpl.ROUTE_OCR)


class PromptTest(unittest.TestCase):
    """El plan no reescribe prompts: delega en document_params."""

    def test_los_prompts_son_los_de_siempre(self) -> None:
        plan = dpl.from_ui("resumen", language="eu", closing=True)
        self.assertEqual(plan.image_prompt(), dp.prompt_for("resumen", "eu", True))
        self.assertEqual(plan.text_prompt(), dp.text_prompt_for("resumen", "eu", True))
        self.assertEqual(
            plan.global_prompt(), dp.global_prompt_for("resumen", "eu", True)
        )

    def test_el_prompt_de_ocr_no_se_traduce(self) -> None:
        plan = dpl.from_ui("ocr", language="es", force_ocr=True)
        self.assertEqual(plan.ocr_prompt(), dp.prompt_for("ocr"))
        self.assertIsNone(plan.text_prompt())

    def test_el_prompt_de_bloque_no_es_el_del_documento_entero(self) -> None:
        plan = dpl.from_ui("resumen")
        self.assertNotEqual(plan.chunk_prompt(), plan.global_prompt())
        self.assertIn("part of a longer document", plan.chunk_prompt())
        self.assertIsNone(dpl.from_ui("ocr").chunk_prompt())


class SummaryBlocksTest(unittest.TestCase):
    """FASE 2: partir el documento sin perder texto, en vez de recortarlo."""

    def test_un_documento_que_cabe_es_un_solo_bloque(self) -> None:
        texto = "uno\n\ndos"
        self.assertEqual(dpl.summary_blocks(texto, max_chars=100), (texto,))

    def test_el_documento_largo_se_parte_sin_perder_nada(self) -> None:
        texto = "\n\n".join(f"Parrafo {i} con su texto." for i in range(30))
        bloques = dpl.summary_blocks(texto, max_chars=120)
        self.assertGreater(len(bloques), 1)
        self.assertEqual("\n\n".join(bloques), texto, "no se puede perder texto")

    def test_los_bloques_respetan_el_presupuesto(self) -> None:
        for bloque in dpl.summary_blocks("x" * 500, max_chars=100):
            self.assertLessEqual(len(bloque), 100)

    def test_un_parrafo_largo_se_corta_igual(self) -> None:
        # Sin líneas en blanco y sin frases completas: se corta por longitud,
        # que es lo único que evita mandar un bloque que no cabe.
        bloques = dpl.summary_blocks("palabra " * 100, max_chars=80)
        self.assertGreater(len(bloques), 1)
        self.assertLessEqual(max(len(bloque) for bloque in bloques), 80)

    def test_el_corte_busca_el_final_de_frase(self) -> None:
        texto = ("Frase completa. " * 20).strip()
        bloques = dpl.summary_blocks(texto, max_chars=60)
        for bloque in bloques[:-1]:
            self.assertTrue(bloque.endswith("."), bloque[-30:])

    def test_texto_vacio_no_da_bloques(self) -> None:
        self.assertEqual(dpl.summary_blocks("   "), ())


class ReglasDeEstiloTest(unittest.TestCase):
    """Las reglas que salen de corregir un resumen real (convenio laboral).

    Cuatro fallos concretos: cifra con la unidad cambiada («24 horas» → «24
    días»), información movida de apartado, prohibición invertida y listas
    comprimidas. Las reglas están en el prompt; estos tests evitan que se
    pierdan en la siguiente reescritura.
    """

    def test_cada_cifra_con_su_unidad_y_su_apartado(self) -> None:
        estilo = dp.SUMMARY_STYLE
        self.assertIn("do not turn hours into days", estilo)
        self.assertIn("do not reuse a figure taken from a different section", estilo)

    def test_nada_se_mueve_de_apartado(self) -> None:
        estilo = dp.SUMMARY_STYLE
        self.assertIn("One section, one block", estilo)
        self.assertIn("Never move a statement", estilo)
        self.assertIn("never merge two different sections into one block", estilo)

    def test_las_listas_se_resumen_sin_perder_items(self) -> None:
        # Conservar los ítems no puede significar copiarlos: se comprimen a una
        # frase con las cifras entre paréntesis (versión anterior: transcribía).
        estilo = dp.SUMMARY_STYLE
        self.assertIn("a list is summarised, not copied", estilo)
        self.assertIn("Never drop an item and never drop a figure", estilo)

    def test_se_prohibe_copiar_en_vez_de_resumir(self) -> None:
        # Conservar ítems y cifras llevaba a transcribir el documento: hay que
        # pedir explícitamente que se reescriba más corto.
        self.assertIn("Rewrite, do not transcribe", dp.SUMMARY_STYLE)
        for prompt in (
            dp.global_prompt_for("resumen"),
            dp.chunk_prompt_for("resumen"),
        ):
            with self.subTest(prompt=prompt[:40]):
                self.assertIn("not a transcription", prompt)
                self.assertIn("never drop a rule, an item or a figure", prompt)

    def test_no_se_invierte_el_sentido_de_una_regla(self) -> None:
        estilo = dp.SUMMARY_STYLE
        self.assertIn("who is forbidden and who is exempt", estilo)
        self.assertIn("never invert the sense of the rule", estilo)

    def test_los_prompts_globales_tambien_protegen_las_listas(self) -> None:
        for prompt in (
            dp.global_prompt_for("resumen"),
            dp.chunk_prompt_for("resumen"),
        ):
            with self.subTest(prompt=prompt[:40]):
                self.assertIn("every item of every list", prompt)


class LanguageTest(unittest.TestCase):
    """Idioma de salida: «auto» se resuelve detectando el del documento.

    Medido con un documento en euskera: con «auto» a secas el resumen salía en
    castellano y los títulos en euskera. Pedir «el mismo idioma» no basta;
    :func:`services.document_params.detect_language` dice *cuál* es.
    """

    EU = (
        "Lanaldia asteko 40 ordukoa da eta urteko lanaldia 1776 ordukoa da. "
        "Oporrak eta baimenak hitzarmenean arautzen dira, eta gaueko lana "
        "debekatuta dago 18 urte baino gutxiagorentzat."
    )
    ES = (
        "La jornada semanal es de 40 horas de trabajo y las vacaciones son de "
        "30 dias para los trabajadores de la empresa, con un descanso minimo "
        "entre jornadas."
    )
    EN = (
        "The annual leave is 30 days and the working hours are 40 per week for "
        "all employees, with a minimum rest between shifts."
    )

    def test_detecta_los_idiomas(self) -> None:
        self.assertEqual(dp.detect_language(self.EU), "eu")
        self.assertEqual(dp.detect_language(self.ES), "es")
        self.assertEqual(dp.detect_language(self.EN), "en")

    def test_un_texto_ambiguo_no_decide(self) -> None:
        self.assertIsNone(dp.detect_language(""))
        self.assertIsNone(dp.detect_language("123 456"))
        self.assertIsNone(dp.detect_language("hola"))

    def test_auto_se_resuelve_con_el_idioma_detectado(self) -> None:
        plan = dpl.from_ui("resumen")
        self.assertTrue(plan.auto_language)
        resuelto = plan.with_language(dp.detect_language(self.EU))
        self.assertFalse(resuelto.auto_language)
        self.assertIn("Basque (euskera)", resuelto.global_prompt())

    def test_un_idioma_elegido_a_mano_no_es_auto(self) -> None:
        plan = dpl.from_ui("resumen", language="es")
        self.assertFalse(plan.auto_language)
        self.assertIn("Spanish", plan.global_prompt())


class FidelityTest(unittest.TestCase):
    """FASE 4: cifras del resultado que no están en el documento, y al revés."""

    def test_detecta_una_cifra_inventada(self) -> None:
        informe = dpl.check_fidelity(
            "El importe es 9.999 euros.", "El importe es 4.500 euros."
        )
        self.assertIn("9999", informe.invented)
        self.assertFalse(informe.ok)

    def test_detecta_una_cifra_omitida(self) -> None:
        informe = dpl.check_fidelity("Resumen sin datos.", "El importe es 4.500 euros.")
        self.assertIn("4500", informe.omitted)

    def test_ignora_el_formato_de_las_cifras(self) -> None:
        informe = dpl.check_fidelity("1.234,5 euros", "1234.5 euros")
        self.assertTrue(informe.ok, informe.warnings())

    def test_las_cifras_cortas_no_se_comprueban(self) -> None:
        # Numeración de apartados y días sueltos: ruido, no señal.
        informe = dpl.check_fidelity("1. Primer punto del dia 3.", "1. Primer punto.")
        self.assertTrue(informe.ok, informe.warnings())

    def test_validate_avisa_de_las_cifras_inventadas(self) -> None:
        report = dpl.validate(
            summary="El importe es 9.999 euros.",
            pairs=[("El importe es 9.999 euros.", "El importe es 4.500 euros.")],
        )
        self.assertTrue(
            any(aviso.startswith("Aviso de fidelidad") for aviso in report.warnings),
            report.warnings,
        )

    def test_validate_no_avisa_por_omitir_cifras(self) -> None:
        # Un resumen es selectivo: avisar de cada cifra que se queda fuera
        # convertiría el aviso en ruido permanente (y siempre presente al
        # agrupar una serie de apartados).
        report = dpl.validate(
            summary="Los importes van de 1.000 a 9.000 euros.",
            pairs=[
                (
                    "Los importes van de 1.000 a 9.000 euros.",
                    "El importe es 1.000 euros.\nImporte: 5.000 euros.\n"
                    "Importe: 9.000 euros.",
                )
            ],
        )
        self.assertTrue(report.ok, report.warnings)

    def test_validate_avisa_del_resumen_por_bloques(self) -> None:
        report = dpl.validate(summary="resumen", blocks=3)
        self.assertIn(dp.BLOCKS_NOTICE, report.warnings)

    def test_los_avisos_de_bloques_van_al_final(self) -> None:
        report = dpl.validate(summary="resumen", blocks=2)
        salida = dpl.deliver(summary="resumen", report=report)
        self.assertTrue(salida.startswith("resumen"))
        self.assertTrue(salida.endswith(dp.BLOCKS_NOTICE))


class NormalizeTest(unittest.TestCase):
    """FASE 2: unir páginas y saber si hubo que recortar."""

    def test_une_las_paginas_y_descarta_las_vacias(self) -> None:
        normalizado = dpl.normalize_pages(["uno", "", "dos"])
        self.assertEqual(normalizado.text, "uno\n\ndos")
        self.assertFalse(normalizado.truncated)

    def test_marca_el_recorte(self) -> None:
        normalizado = dpl.normalize_pages(["abcdefghij", "klmnopqrst"], max_chars=8)
        self.assertEqual(normalizado.text, "abcdefgh")
        self.assertTrue(normalizado.truncated)


class ValidationTest(unittest.TestCase):
    """FASE 4: avisos ligeros, sin bloquear la entrega."""

    def test_sin_problemas_no_hay_avisos(self) -> None:
        report = dpl.validate([dpl.render_page(1, dpl.ROUTE_TEXT, "contenido")])
        self.assertTrue(report.ok)
        self.assertEqual(report.warnings, ())

    def test_el_recorte_avisa(self) -> None:
        report = dpl.validate(summary="resumen", truncated=True)
        self.assertIn(dp.TRUNCATED_NOTICE, report.warnings)

    def test_una_entrega_vacia_avisa(self) -> None:
        report = dpl.validate([dpl.render_page(1, dpl.ROUTE_OCR, "")])
        self.assertIn(dpl.NO_CONTENT_NOTICE, report.warnings)

    def test_si_el_modelo_no_devuelve_nada_lo_dice(self) -> None:
        report = dpl.validate(summary="")
        self.assertIn(dpl.EMPTY_RESULT_NOTICE, report.warnings)


class DeliverTest(unittest.TestCase):
    """FASE 5: un solo punto de montaje del texto final."""

    def test_el_resumen_manda_y_no_se_etiquetan_paginas(self) -> None:
        salida = dpl.deliver(
            sections=[dpl.render_page(1, dpl.ROUTE_TEXT, "se ignora")],
            summary="resumen global",
        )
        self.assertEqual(salida, "resumen global")

    def test_las_secciones_se_unen_con_el_formato_de_siempre(self) -> None:
        salida = dpl.deliver(
            sections=[
                dpl.render_page(1, dpl.ROUTE_TEXT, "uno"),
                dpl.render_page(2, dpl.ROUTE_OCR, "dos"),
            ]
        )
        self.assertEqual(
            salida, "--- Página 1 (texto) ---\n\nuno\n\n--- Página 2 (OCR) ---\n\ndos"
        )

    def test_las_tablas_van_despues_del_resumen(self) -> None:
        salida = dpl.deliver(
            summary="resumen",
            extra_sections=[dpl.render_tables(2, "<table></table>")],
        )
        self.assertEqual(
            salida, "resumen\n\n--- Tablas (Página 2) ---\n\n<table></table>"
        )

    def test_los_avisos_van_al_final(self) -> None:
        report = dpl.validate(summary="resumen", truncated=True)
        salida = dpl.deliver(summary="resumen", report=report)
        self.assertTrue(salida.startswith("resumen"))
        self.assertTrue(salida.endswith(dp.TRUNCATED_NOTICE))


class _VacioEngine:
    """Motor que no devuelve nada (para la validación)."""

    def analyze_image(self, path, prompt, model=None) -> str:
        return ""

    def analyze_text(self, text, prompt, model=None) -> str:
        return ""


class WorkerFasesTest(unittest.TestCase):
    """Ejecución real de las fases nuevas por el DocumentWorker."""

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

    def _run(self, modo, paginas, **kwargs):
        # Detalle ejecutivo por defecto: estos tests cuentan llamadas y prompt;
        # el estilo por defecto (dos partes) añadiría el detalle.
        kwargs.setdefault("detail", dp.STYLE_EXECUTIVE)
        engine = wd._FakeEngine()
        doc = wd._FakeDoc(paginas)
        worker = dw.DocumentWorker(engine, self.pdf, modo, "modelo-x", **kwargs)
        with mock.patch.object(dw, "pymupdf", wd._FakePymupdf(doc)):
            result = worker.work()
        return result, engine

    def test_forzar_ocr_transcribe_una_pagina_con_texto(self) -> None:
        result, engine = self._run(
            "resumen", [wd._FakePage(wd._TEXT)], force_ocr=True
        )
        # FASE 1: se transcribe con el prompt de OCR (no con el de resumen).
        self.assertEqual(len(engine.image_calls), 1)
        self.assertEqual(engine.image_calls[0][1], dp.prompt_for("ocr"))
        self.assertIn("--- Página 1 (OCR) ---", result)
        # FASE 2: el resumen se pide sobre la transcripción.
        self.assertEqual(len(engine.text_calls), 1)
        self.assertEqual(engine.text_calls[0][0], "RESULTADO-IMAGEN")
        self.assertEqual(engine.text_calls[0][1], dp.text_prompt_for("resumen"))

    def test_sin_forzar_ocr_la_capa_de_texto_se_usa_tal_cual(self) -> None:
        _, engine = self._run("resumen", [wd._FakePage(wd._TEXT)])
        self.assertEqual(engine.image_calls, [])
        self.assertEqual(engine.text_calls[0][0], wd._TEXT.strip())

    def test_las_tablas_acompanan_al_resumen(self) -> None:
        result, engine = self._run(
            "resumen", [wd._FakePage(wd._TEXT)], extra_tables=True
        )
        prompts = [llamada[1] for llamada in engine.text_calls]
        self.assertEqual(
            prompts, [dp.text_prompt_for("resumen"), dp.text_prompt_for("tablas")]
        )
        self.assertIn("--- Tablas (Página 1) ---", result)
        self.assertIn("--- Página 1 (texto) ---", result)

    def test_las_tablas_acompanan_a_la_transcripcion(self) -> None:
        result, engine = self._run(
            "ocr", [wd._FakePage(wd._TEXT)], extra_tables=True
        )
        self.assertIn(wd._TEXT.strip(), result)
        self.assertIn("--- Tablas (Página 1) ---", result)
        self.assertEqual(
            [llamada[1] for llamada in engine.text_calls], [dp.text_prompt_for("tablas")]
        )

    def test_en_el_resumen_global_las_tablas_van_al_final(self) -> None:
        result, engine = self._run(
            "resumen",
            [wd._FakePage(wd._TEXT)],
            global_summary=True,
            extra_tables=True,
        )
        # Los prompts llevan el tope de palabras y el detalle, así que se
        # comprueba su naturaleza (global y de tablas), no su igualdad exacta.
        self.assertIn("complete document", engine.text_calls[0][1])
        self.assertIn("tables", engine.text_calls[1][1])
        self.assertTrue(result.startswith("RESULTADO"))
        self.assertIn("--- Tablas (Página 1) ---", result)

    def test_sin_tablas_extras_solo_se_hace_el_resumen(self) -> None:
        _, engine = self._run(
            "resumen", [wd._FakePage(wd._TEXT)], global_summary=True
        )
        self.assertEqual(len(engine.text_calls), 1)

    def test_la_validacion_avisa_si_el_modelo_no_devuelve_nada(self) -> None:
        # Global: no hay secciones de página, así que el aviso es «el modelo no
        # devolvió texto» (una entrega por páginas avisaría de que el documento
        # no dio contenido: es el caso de abajo).
        result = self._run_con(_VacioEngine(), global_summary=True)
        self.assertIn(dpl.EMPTY_RESULT_NOTICE, result)

    def test_la_validacion_avisa_si_el_documento_no_da_contenido(self) -> None:
        result = self._run_con(_VacioEngine())
        self.assertIn(dpl.NO_CONTENT_NOTICE, result)

    def test_el_idioma_detectado_llega_al_prompt(self) -> None:
        # Sin elegir idioma manda el documento: el prompt lo pide por su nombre
        # (en euskera, no «en el mismo idioma que el documento»).
        engine = wd._FakeEngine()
        doc = wd._FakeDoc([wd._FakePage(LanguageTest.EU)])
        worker = dw.DocumentWorker(engine, self.pdf, "resumen", "modelo-x")
        with mock.patch.object(dw, "pymupdf", wd._FakePymupdf(doc)):
            worker.work()
        self.assertIn("Basque (euskera)", engine.text_calls[0][1])

    def test_un_idioma_elegido_a_mano_no_se_pisa_con_el_detectado(self) -> None:
        engine = wd._FakeEngine()
        doc = wd._FakeDoc([wd._FakePage(LanguageTest.EU)])
        worker = dw.DocumentWorker(engine, self.pdf, "resumen", "modelo-x", "es")
        with mock.patch.object(dw, "pymupdf", wd._FakePymupdf(doc)):
            worker.work()
        self.assertIn("Spanish", engine.text_calls[0][1])

    def _run_con(self, engine, **kwargs) -> str:
        kwargs.setdefault("detail", dp.STYLE_EXECUTIVE)
        doc = wd._FakeDoc([wd._FakePage(wd._TEXT)])
        worker = dw.DocumentWorker(engine, self.pdf, "resumen", "m", **kwargs)
        with mock.patch.object(dw, "pymupdf", wd._FakePymupdf(doc)):
            return worker.work()


class WidgetFasesTest(unittest.TestCase):
    """Las dos casillas de la FASE 1 en la UI: estado, persistencia y uso."""

    @classmethod
    def setUpClass(cls) -> None:
        from PyQt6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        from core.config_manager import ConfigManager

        self._tmp = tempfile.TemporaryDirectory()
        self.cm = ConfigManager(config_dir=self._tmp.name)
        self.widget = dw.DocumentWidget(None, self.cm)

    def tearDown(self) -> None:
        self.widget.close()
        self._tmp.cleanup()

    def _modo(self, mode_id: str) -> None:
        self.widget._mode.setCurrentIndex(self.widget._mode.findData(mode_id))

    def test_las_tablas_extras_no_aplican_al_modo_tablas(self) -> None:
        for mode_id, esperado in (("ocr", True), ("resumen", True), ("tablas", False)):
            with self.subTest(modo=mode_id):
                self._modo(mode_id)
                self.assertEqual(self.widget._tables_check.isEnabled(), esperado)

    def test_el_ocr_forzado_aplica_a_cualquier_modo(self) -> None:
        for mode_id in ("ocr", "resumen", "tablas"):
            with self.subTest(modo=mode_id):
                self._modo(mode_id)
                self.assertTrue(self.widget._force_ocr_check.isEnabled())

    def test_el_estado_de_las_casillas_se_guarda(self) -> None:
        self.widget._force_ocr_check.setChecked(True)
        self.widget._tables_check.setChecked(True)
        last_models = self.cm.load_last_models()
        self.assertEqual(last_models[prefs.DOCUMENT_FORCE_OCR], "1")
        self.assertEqual(last_models[prefs.DOCUMENT_TABLES], "1")

    def test_restaura_las_casillas_guardadas(self) -> None:
        self.cm.save_last_model(prefs.DOCUMENT_FORCE_OCR, "1")
        self.cm.save_last_model(prefs.DOCUMENT_TABLES, "1")
        widget = dw.DocumentWidget(None, self.cm)
        try:
            self.assertTrue(widget._force_ocr_check.isChecked())
            self.assertTrue(widget._tables_check.isChecked())
        finally:
            widget.close()

    def test_las_casillas_se_bloquean_durante_el_trabajo(self) -> None:
        self.widget._set_busy(True)
        self.assertFalse(self.widget._force_ocr_check.isEnabled())
        self.assertFalse(self.widget._tables_check.isEnabled())

        self.widget._set_busy(False)
        self.assertTrue(self.widget._force_ocr_check.isEnabled())
        self.assertTrue(self.widget._tables_check.isEnabled())

    def test_las_casillas_llegan_al_worker(self) -> None:
        self.widget._force_ocr_check.setChecked(True)
        self.widget._tables_check.setChecked(True)
        self.widget._file_path = Path("documento.pdf")
        with mock.patch.object(dw, "DocumentWorker") as worker_cls:
            with mock.patch.object(self.widget, "_set_busy"):
                self.widget._engine = mock.Mock(is_available=lambda: True)
                self.widget._process()
        self.assertTrue(worker_cls.call_args.kwargs["force_ocr"])
        self.assertTrue(worker_cls.call_args.kwargs["extra_tables"])


if __name__ == "__main__":
    unittest.main()
