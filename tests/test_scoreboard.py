"""Tests del marcador/goles leídos por OCR (``services.scoreboard``)."""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from services import scoreboard as sb  # noqa: E402


class ExtractScoreTest(unittest.TestCase):
    def test_tanteo_con_espacios(self):
        # El OCR real devuelve «MUSKIZ 12 -9 CORAZONISTA» (sin espacio tras el guion).
        self.assertEqual(sb.extract_score("MUSKIZ 12 -9 CORAZONISTA"), (12, 9))

    def test_tanteo_con_guion_largo(self):
        self.assertEqual(sb.extract_score("28 – 25"), (28, 25))

    def test_ignora_el_reloj(self):
        self.assertIsNone(sb.extract_score("1er TIEMPO 25:30"))

    def test_ignora_medidas_de_pista(self):
        self.assertIsNone(sb.extract_score("pista 40x20"))

    def test_sin_tanteo(self):
        self.assertIsNone(sb.extract_score("MUSKIZ CORAZONISTAS"))
        self.assertIsNone(sb.extract_score(""))

    def test_no_parte_numeros_largos(self):
        self.assertIsNone(sb.extract_score("123-456"))

    def test_tanteo_etiquetado_sin_guion(self):
        # OCR real del marcador electrónico: goles separados por ESPACIOS.
        self.assertEqual(
            sb.extract_score("10:45 LOCAL VISITANTE 08 07 PERODO"), (8, 7)
        )

    def test_tanteo_etiquetado_con_adorno(self):
        self.assertEqual(sb.extract_score("41:20 LOCAL 30 20"), (30, 20))

    def test_no_confunde_dorsales_sin_etiquetas(self):
        # Sin LOCAL/VISITANTE, dos números sueltos NO son marcador.
        self.assertIsNone(sb.extract_score("MUSKIZ 15 12"))

    def test_descarta_racha_de_tres_numeros(self):
        # OCR sucio con tres cifras («… 2 42 29»): mejor descartar que inventar.
        self.assertIsNone(
            sb.extract_score("60:00 FALTAS FALTAS LOCAL VISITANTE 2 42 29 PERIODO")
        )


class ScoreSeriesTest(unittest.TestCase):
    def test_descarta_repetidos(self):
        lecturas = [
            (1.0, "0 - 0"),
            (5.0, "0 - 0"),
            (9.0, "1 - 0"),
            (12.0, "1 - 0"),
            (14.0, "1 - 1"),
        ]
        serie = sb.score_series(lecturas)
        self.assertEqual([p.marcador for p in serie], ["0-0", "1-0", "1-1"])

    def test_sin_tanteos(self):
        self.assertEqual(sb.score_series([(1.0, "x"), (2.0, "y")]), [])


class GoalsTest(unittest.TestCase):
    def test_gol_de_cada_lado(self):
        serie = sb.score_series(
            [(0.0, "0-0"), (10.0, "1-0"), (20.0, "1-1"), (30.0, "2-1")]
        )
        goles = sb.goals_from_series(serie)
        self.assertEqual(
            [(g.lado, g.marcador) for g in goles],
            [("izquierda", "1-0"), ("derecha", "1-1"), ("izquierda", "2-1")],
        )

    def test_salto_de_ocr_no_es_gol(self):
        serie = sb.score_series([(0.0, "0-0"), (10.0, "3-0")])
        self.assertEqual(sb.goals_from_series(serie), [])

    def test_bajada_no_es_gol(self):
        serie = sb.score_series([(0.0, "5-3"), (10.0, "4-3")])
        self.assertEqual(sb.goals_from_series(serie), [])


class ReportTest(unittest.TestCase):
    def test_bloque_vacio_sin_tanteos(self):
        self.assertEqual(sb.marcador_block([(1.0, "nada por aquí")]), "")

    def test_bloque_con_tanteos(self):
        bloque = sb.marcador_block([(0.0, "0-0"), (60.0, "1-0")])
        self.assertIn("0-0", bloque)
        self.assertIn("1-0", bloque)
        self.assertIn("Total de goles detectados: 1.", bloque)

    def test_resumen_una_linea(self):
        resumen = sb.scoreboard_summary([(0.0, "0-0"), (60.0, "1-0")])
        self.assertIn("1-0", resumen)
        self.assertIn("goles detectados", resumen)

    def test_resumen_vacio_sin_tanteos(self):
        self.assertEqual(sb.scoreboard_summary([(0.0, "sin marcador")]), "")


class ClockTest(unittest.TestCase):
    def test_extrae_reloj(self):
        self.assertEqual(sb.extract_clock("10:45 LOCAL VISITANTE 08 07"), 645)

    def test_reloj_no_confunde_el_tanteo(self):
        # «12-9» es un tanteo, no un reloj.
        self.assertIsNone(sb.extract_clock("MUSKIZ 12-9 CORAZONISTAS"))

    def test_arranque_cuando_el_reloj_se_mueve(self):
        # El partido arranca en la última lectura con el reloj aún parado.
        lecturas = [(0.0, "00:00"), (27.0, "00:00"), (159.0, "02:05")]
        self.assertEqual(sb.game_start(lecturas), 27.0)

    def test_sin_movimiento_no_hay_arranque(self):
        self.assertIsNone(sb.game_start([(0.0, "00:00"), (10.0, "00:00")]))

    def test_sin_reloj(self):
        self.assertIsNone(sb.game_start([(0.0, "sin reloj")]))


if __name__ == "__main__":
    unittest.main()
