"""El chequeo de cifras no debe depender del separador de millares.

Caso real medido el 30/09/2026 con un PDF de 10 páginas: el documento escribía
«4 320 000» (espacio, la tipografía española que sale al extraer el texto) y el
resumen «4.320.000» (punto). El regex anterior sólo entendía puntos y comas, así
que la misma cifra aparecía **a la vez** como inventada y como omitida, y la
entrega avisaba de «puede haber datos inventados» sobre un resumen fiel: 8 falsos
positivos en un solo documento. Un aviso que salta siempre acaba ignorándose, que
es justo lo contrario de para lo que existe la FASE 4.
"""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from services import document_pipeline as dpl  # noqa: E402


class SeparadoresDeMillarTest(unittest.TestCase):
    def test_la_misma_cifra_con_cualquier_separador(self) -> None:
        for escrito in (
            "4 320 000",
            "4.320.000",
            "4,320,000",
            "4\u00a0320\u00a0000",  # espacio duro
            "4\u202f320\u202f000",  # espacio fino
        ):
            with self.subTest(escrito=escrito):
                self.assertEqual(dpl.numbers(escrito), ("4320000",))

    def test_no_mezcla_dos_numeros_separados_por_un_espacio(self) -> None:
        # «74 personas y 1 240 horas» son dos cifras, no «74 124».
        self.assertEqual(dpl.numbers("74 personas y 1 240 horas"), ("74", "1240"))

    def test_sigue_normalizando_decimales(self) -> None:
        self.assertEqual(dpl.numbers("1.234,50"), ("123450",))
        self.assertEqual(dpl.numbers("12,4 %"), ("124",))

    def test_cifras_sueltas(self) -> None:
        self.assertEqual(dpl.numbers("2026 y 8,7 sobre 10"), ("2026", "87", "10"))


class CifrasAbreviadasTest(unittest.TestCase):
    """«4,32 millones» y «4 320 000» son la misma cantidad."""

    def test_se_expanden_con_la_palabra_de_escala(self) -> None:
        self.assertEqual(dpl.scaled_numbers("4,32 millones"), ("4320000",))
        self.assertEqual(dpl.scaled_numbers("1,45 millones de euros"), ("1450000",))
        self.assertEqual(dpl.scaled_numbers("2 mil"), ("2000",))
        self.assertEqual(dpl.scaled_numbers("1,5 millon"), ("1500000",))
        self.assertEqual(dpl.scaled_numbers("2 millones"), ("2000000",))

    def test_no_confunde_palabras_que_solo_empiezan_igual(self) -> None:
        self.assertEqual(dpl.scaled_numbers("un milagro de 3 milagros"), ())
        self.assertEqual(dpl.scaled_numbers("sin cifras"), ())

    def test_un_resumen_abreviado_no_es_inventado(self) -> None:
        informe = dpl.check_fidelity(
            "Facturacion: 4,32 millones de euros y 1,45 millones de contratos.",
            "Facturacion: 4 320 000 euros y 1 450 000 contratos.",
        )
        self.assertEqual(informe.invented, ())

    def test_una_cifra_abreviada_distinta_si_se_detecta(self) -> None:
        informe = dpl.check_fidelity(
            "Facturacion: 9,99 millones de euros.",
            "Facturacion: 4 320 000 euros.",
        )
        self.assertIn("9990000", informe.invented)


class FidelidadTest(unittest.TestCase):
    def test_un_resumen_fiel_no_avisa_por_el_formato(self) -> None:
        documento = "La facturacion fue de 4 320 000 euros y el margen del 34,8 %."
        resumen = "Facturacion: 4.320.000 euros (34,8 % de margen)."
        informe = dpl.check_fidelity(resumen, documento)
        self.assertEqual(informe.invented, ())
        self.assertEqual(informe.omitted, ())
        self.assertEqual(informe.warnings(), ())

    def test_sigue_detectando_una_cifra_inventada(self) -> None:
        informe = dpl.check_fidelity(
            "Vendimos 9.999.000 unidades.", "Vendimos 4 320 000 unidades."
        )
        self.assertIn("9999000", informe.invented)
        self.assertTrue(informe.warnings())

    def test_el_validate_no_avisa_en_el_caso_real_medido(self) -> None:
        """El caso exacto de la auditoría: mismo dato, distinto separador."""
        documento = (
            "FACTURACION: 4 320 000 euros y 12,4 %. "
            "MARGEN: 34,8 % y 612 000 euros. PLANTILLA: 74 personas y 1 240 horas."
        )
        resumen = (
            "FACTURACION: 4.320.000 euros y 12,4 %. "
            "MARGEN: 34,8 % y 612.000 euros. PLANTILLA: 74 personas y 1.240 horas."
        )
        informe = dpl.validate(summary=resumen, pairs=[(resumen, documento)])
        self.assertEqual(informe.warnings, ())
        self.assertTrue(informe.ok)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
