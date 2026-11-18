"""Tests de las alineaciones del usuario (services/roster_params.py).

Lo que se cubre: leer la alineación (dorsal + nombre), sacar el nombre de pila y
traducir «dorsal 7» / «el 7» al nombre en el texto de la IA, sin inventar nombres
cuando el dorsal no consta o es ambiguo.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services import chronicle_params as cp  # noqa: E402
from services import notes_params as nts  # noqa: E402
from services import roster_params as rp  # noqa: E402

CONTEXTO = (
    "Deporte: balonmano.\n"
    "Equipos: Muskis (local, camiseta azul) vs Corazonistas Senior (visitante, roja).\n"
    "Alineación Muskis: 1 Unai (portero), 4 Álex, 7 Marc, 9 Iván.\n"
    "Alineación Corazonistas: 1 Diego (portero), 3 Hugo, 6 Pablo, 10 Samu.\n"
    "Marcador final: 28-25."
)


class PrimerNombreTest(unittest.TestCase):
    def test_nombre_completo_en_mayusculas(self):
        self.assertEqual(rp.primer_nombre("AMANDA OTERO BENITO"), "Amanda")

    def test_respeta_acentos(self):
        self.assertEqual(rp.primer_nombre("Álex"), "Álex")

    def test_vacio(self):
        self.assertEqual(rp.primer_nombre(""), "")


class ParseRosterTest(unittest.TestCase):
    def test_lee_las_dos_alineaciones(self):
        jugadores = rp.parse_roster(CONTEXTO)
        self.assertEqual(len(jugadores), 8)
        self.assertEqual(jugadores[0].equipo, "Muskis")
        self.assertEqual(jugadores[0].dorsal, 1)
        self.assertEqual(jugadores[0].nombre, "Unai")
        self.assertEqual(jugadores[0].nombre_completo, "Unai")

    def test_quita_la_marca_de_portero(self):
        jugadores = {j.dorsal: j for j in rp.parse_roster(CONTEXTO) if j.equipo == "Muskis"}
        self.assertEqual(jugadores[1].nombre_completo, "Unai")

    def test_no_confunde_los_equipos_con_la_alineacion(self):
        equipos = {j.equipo for j in rp.parse_roster(CONTEXTO)}
        self.assertEqual(equipos, {"Muskis", "Corazonistas"})

    def test_sin_alineacion_no_hay_jugadoras(self):
        self.assertEqual(rp.parse_roster("Equipos: A vs B\nMarcador final: 1-0"), ())

    def test_lista_sin_comas(self):
        jugadores = rp.parse_roster("Alineación A: 1 Uno 2 Dos 14 Catorce")
        self.assertEqual([j.dorsal for j in jugadores], [1, 2, 14])
        self.assertEqual(jugadores[2].nombre, "Catorce")


class RosterBlockTest(unittest.TestCase):
    def test_traduce_cada_dorsal(self):
        bloque = rp.roster_block(CONTEXTO)
        self.assertIn("Dorsal 7: Marc", bloque)
        self.assertIn("NOMBRE DE PILA", bloque)
        self.assertIn("Corazonistas", bloque)

    def test_sin_alineacion_es_vacio(self):
        self.assertEqual(rp.roster_block("Equipos: A vs B"), "")


class ResolveRosterNamesTest(unittest.TestCase):
    def test_dorsal_a_nombre(self):
        self.assertEqual(
            rp.resolve_roster_names("Jugadora con dorsal 7 en el centro", CONTEXTO),
            "Jugadora con Marc en el centro",
        )

    def test_entre_parentesis(self):
        self.assertEqual(
            rp.resolve_roster_names("Portera (dorsal 10) bloquea", CONTEXTO),
            "Portera (Samu) bloquea",
        )

    def test_el_numero_suelto(self):
        self.assertEqual(
            rp.resolve_roster_names("Lanza el 9 azul", CONTEXTO), "Lanza Iván"
        )

    def test_dorsal_ambiguo_se_resuelve_por_color(self):
        self.assertEqual(rp.resolve_roster_names("el 1 azul", CONTEXTO), "Unai")
        self.assertEqual(rp.resolve_roster_names("el 1 rojo", CONTEXTO), "Diego")

    def test_dorsal_ambiguo_sin_pistas_no_se_toca(self):
        self.assertEqual(rp.resolve_roster_names("el 1", CONTEXTO), "el 1")

    def test_color_en_plural_y_equipo_sin_color(self):
        # «camisetas rojas» debe contar como color rojo (formas de femenino/plural).
        contexto = (
            "Equipos: ALFA camiseta azul (local) vs BRAVO camisetas rojas (visitante).\n"
            "Alineación ALFA: 15 Ana\n"
            "Alineación BRAVO: 15 Bea"
        )
        bloque = rp.roster_block(contexto)
        self.assertIn("Dorsal 15: Ana (ALFA, azul)", bloque)
        self.assertIn("Dorsal 15: Bea (BRAVO, rojo)", bloque)
        self.assertEqual(rp.resolve_roster_names("el 15 rojas", contexto), "Bea")
        self.assertEqual(rp.resolve_roster_names("el 15 azul", contexto), "Ana")

    def test_dorsal_que_no_existe_no_se_toca(self):
        self.assertEqual(
            rp.resolve_roster_names("dorsal 55 avanza", CONTEXTO), "dorsal 55 avanza"
        )

    def test_no_confunde_un_numero_de_tiempo(self):
        self.assertEqual(rp.resolve_roster_names("en la 2 parte", CONTEXTO), "en la 2 parte")

    def test_sin_alineacion_no_cambia_nada(self):
        texto = "gol del dorsal 7"
        self.assertEqual(rp.resolve_roster_names(texto, "Equipos: A vs B"), texto)


class IntegracionTest(unittest.TestCase):
    def test_el_prompt_del_cuaderno_lleva_la_alineacion_y_traduce(self):
        prompt = nts.analysis_prompt(
            "f.mp4", 60, [(1.0, "gol de la jugadora con dorsal 7")], context=CONTEXTO
        )
        self.assertIn("Dorsal 7: Marc", prompt)
        self.assertIn("Marc", prompt)

    def test_el_analisis_traduce_los_dorsales(self):
        analysis = nts.parse_analysis(
            {
                "lineas_narrativas": [
                    {"nombre": "Local", "eventos": [{"t_s": 1, "descripcion": "dorsal 7 marca"}]}
                ]
            },
            context=CONTEXTO,
        )
        self.assertEqual(analysis.lineas[0].eventos[0].descripcion, "Marc marca")

    def test_el_informe_guarda_los_nombres(self):
        informe = nts.informe_to_dict(frames=[(1.0, "dorsal 4 bota")], context=CONTEXTO)
        self.assertEqual(informe["fotogramas"][0]["descripcion"], "Álex bota")

    def test_el_prompt_de_la_cronica_lleva_el_bloque(self):
        prompt = cp.chronicle_prompt("informe", context=CONTEXTO)
        self.assertIn("Dorsal 10: Samu", prompt)

    def test_la_cronica_traduce_los_dorsales(self):
        lineas = cp.parse_chronicle(
            '{"cronica": [{"t_s": 12, "texto": "Gol de la jugadora con dorsal 7"}]}',
            context=CONTEXTO,
        )
        self.assertEqual(lineas[0].texto, "Gol de la jugadora con Marc")


if __name__ == "__main__":
    unittest.main()
