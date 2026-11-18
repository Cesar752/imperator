"""Tests de la lectura del acta (:mod:`services.acta_params`).

El acta es lo que da nombres y dorsales a la crónica; si se parsea mal, la crónica
inventa o se queda sin nombres. Aquí se cubren los dos caminos: la respuesta JSON
del chat y el respaldo por líneas cuando no hay chat.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services import acta_params as acp  # noqa: E402


class ParseActaJsonTest(unittest.TestCase):
    def test_json_del_chat(self):
        raw = (
            '{"local": "Muskis", "visitante": "Corazonistas Senior", '
            '"color_local": "azul", "color_visitante": "roja", "marcador": "28-25", '
            '"jugadores": ['
            '{"equipo": "local", "dorsal": 1, "nombre": "Unai", "portero": true}, '
            '{"equipo": "visitante", "dorsal": 10, "nombre": "Samu"}, '
            '{"equipo": "local", "dorsal": 4, "nombre": "Álex"}]}'
        )
        acta = acp.parse_acta(raw)
        self.assertEqual(acta.local, "Muskis")
        self.assertEqual(acta.visitante, "Corazonistas Senior")
        self.assertEqual(acta.marcador, "28-25")
        self.assertEqual([j.dorsal for j in acta.jugadores_de("local")], [1, 4])
        self.assertTrue(acta.jugadores_de("local")[0].portero)
        self.assertEqual([j.nombre for j in acta.jugadores_de("visitante")], ["Samu"])

    def test_acepta_claves_alternativas_y_vallas(self):
        raw = (
            "```json\n"
            '{"equipo_local": "A", "equipo_visitante": "B", "resultado": "1-2", '
            '"alineacion": [{"numero": "7", "jugador": "Marc"}]}\n'
            "```"
        )
        acta = acp.parse_acta(raw)
        self.assertEqual(acta.local, "A")
        self.assertEqual(acta.visitante, "B")
        self.assertEqual(acta.marcador, "1-2")
        self.assertEqual(acta.jugadores[0].dorsal, 7)
        self.assertEqual(acta.jugadores[0].nombre, "Marc")

    def test_descarta_dorsal_o_nombre_invalidos(self):
        raw = (
            '{"jugadores": [{"dorsal": 200, "nombre": "Malo"}, '
            '{"dorsal": 0, "nombre": "Cero"}, '
            '{"dorsal": 9, "nombre": ""}, '
            '{"dorsal": 5, "nombre": "Bueno"}]}'
        )
        acta = acp.parse_acta(raw)
        self.assertEqual([j.dorsal for j in acta.jugadores], [5])


class ParseActaTextTest(unittest.TestCase):
    def test_respaldo_por_lineas_sin_chat(self):
        texto = (
            "EQUIPO LOCAL\n"
            "1 UNAI PEREZ portero\n"
            "4 ALEX GARCIA\n"
            "EQUIPO VISITANTE\n"
            "10 SAMU LOPEZ\n"
            "Resultado final 28-25\n"
        )
        acta = acp.parse_acta(texto)
        self.assertEqual(acta.marcador, "28-25")
        self.assertEqual(
            [j.nombre for j in acta.jugadores_de("local")], ["UNAI PEREZ", "ALEX GARCIA"]
        )
        self.assertEqual(
            [j.nombre for j in acta.jugadores_de("visitante")], ["SAMU LOPEZ"]
        )
        self.assertTrue(acta.jugadores_de("local")[0].portero)

    def test_texto_inservible(self):
        acta = acp.parse_acta("no hay nada que sacar de aqui")
        self.assertEqual(acta.jugadores, ())


class ActaTablaTest(unittest.TestCase):
    """La tabla real de un acta de federación (dos columnas A/B)."""

    CABECERA = (
        "Nº             A: NOMBRE Y APELLIDOS         G  A  2  2  D  D2  IE  "
        "Nº             B: NOMBRE Y APELLIDOS         G  A  2  2  D  D2  IE"
    )

    def test_dos_columnas_y_oficiales_fuera(self):
        texto = "\n".join(
            [
                self.CABECERA,
                "  2  ARANE PEREZ HERNANDEZ                   3"
                "                        3  AMANDA OTERO BENITO                     3",
                "  5  IRIA MORAGA GARCIA                                               "
                "  8  PAULA GARCIA VALENCIA                    5  2  4",
                " 53  MAIDER FERNANDEZ MIGUENS**              5     45",
                "E      ENDIKA CIEZA OCARANZA                                         "
                "E      DAVID RODRIGUEZ TREPIANA                 10",
            ]
        )
        acta = acp.parse_acta(texto)
        self.assertEqual(
            [j.nombre for j in acta.jugadores_de("local")],
            ["ARANE PEREZ HERNANDEZ", "IRIA MORAGA GARCIA", "MAIDER FERNANDEZ MIGUENS"],
        )
        self.assertEqual(
            [j.nombre for j in acta.jugadores_de("visitante")],
            ["AMANDA OTERO BENITO", "PAULA GARCIA VALENCIA"],
        )
        # Los oficiales («E …») no son jugadores.
        self.assertNotIn("ENDIKA CIEZA OCARANZA", [j.nombre for j in acta.jugadores])

    def test_equipos_y_marcador_del_acta(self):
        texto = (
            "PAN MENESA MUSKIZA: EQUIPO ORGANIZADOR:\n"
            "ESKUBALOIA\n"
            "B: EQUIPO VISITANTE:\n"
            "BAIGENE CORAZONISTAS\n"
            "RESULTADO\nFINAL:\n42-29\n"
        )
        acta = acp.parse_acta(texto)
        self.assertEqual(acta.local, "PAN MENESA MUSKIZ")
        self.assertEqual(acta.visitante, "BAIGENE CORAZONISTAS")
        self.assertEqual(acta.marcador, "42-29")

    def test_la_hora_no_es_el_marcador(self):
        texto = "FECHA\nEl 09/10/2026 a las 20:30\nNUM. ESPECTADORES\n44\n"
        self.assertEqual(acp.parse_acta(texto).marcador, "")


class ActaContextTest(unittest.TestCase):
    def test_bloque_completo(self):
        acta = acp.Acta(
            local="Muskis",
            visitante="Corazonistas Senior",
            color_local="azul",
            color_visitante="roja",
            marcador="28-25",
            jugadores=(
                acp.ActaJugador("local", 1, "Unai", True),
                acp.ActaJugador("local", 4, "Álex"),
                acp.ActaJugador("visitante", 10, "Samu"),
            ),
        )
        bloque = acp.acta_context(acta)
        self.assertIn(
            "Equipos: Muskis (local, camiseta azul) vs "
            "Corazonistas Senior (visitante, camiseta roja).",
            bloque,
        )
        self.assertIn("Alineación Muskis: 1 Unai (portero), 4 Álex", bloque)
        self.assertIn("Alineación Corazonistas Senior: 10 Samu", bloque)
        self.assertIn("Marcador final: 28-25", bloque)

    def test_sin_nombres_de_equipo_usa_el_rol(self):
        acta = acp.Acta(jugadores=(acp.ActaJugador("local", 3, "Luis"),))
        self.assertIn("Alineación el local: 3 Luis", acp.acta_context(acta))

    def test_ordena_local_primero_y_por_dorsal(self):
        acta = acp.Acta(
            jugadores=(
                acp.ActaJugador("visitante", 2, "V"),
                acp.ActaJugador("local", 9, "L9"),
                acp.ActaJugador("local", 3, "L3"),
            )
        )
        self.assertEqual([j.dorsal for j in acta.jugadores], [3, 9, 2])

    def test_vacio(self):
        self.assertEqual(acp.acta_context(acp.Acta()), "")

    def test_filtro_y_extensiones(self):
        self.assertIn("*.pdf", acp.ACTA_FILTER)
        self.assertIn("*.jpg", acp.ACTA_FILTER)
        self.assertIn(".png", acp.ACTA_EXTENSIONS)


if __name__ == "__main__":
    unittest.main()
