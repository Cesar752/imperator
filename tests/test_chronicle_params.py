"""Tests de la crónica locutada (services/chronicle_params.py)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services import chronicle_params as cp  # noqa: E402
from services import video_understanding as vu  # noqa: E402


class ChronicleLineTest(unittest.TestCase):
    def test_duracion_por_longitud(self):
        corta = cp.ChronicleLine(0, "Gol")
        self.assertEqual(corta.duration_s(), cp.MIN_LINE_S)
        larga = cp.ChronicleLine(0, "x" * 500)
        self.assertEqual(larga.duration_s(), cp.MAX_LINE_S)
        media = cp.ChronicleLine(0, "x" * 70)
        self.assertAlmostEqual(media.duration_s(), 70 / cp.CHARS_PER_SECOND, places=3)

    def test_el_final_no_pisa_la_frase_siguiente(self):
        actual = cp.ChronicleLine(10, "x" * 200)  # duraría 8 s
        siguiente = cp.ChronicleLine(12, "otra")
        self.assertAlmostEqual(
            actual.end_s(siguiente), 12 - cp.MIN_GAP_S, places=3
        )

    def test_el_final_nunca_pasa_del_video(self):
        actual = cp.ChronicleLine(9.5, "x" * 200)
        self.assertAlmostEqual(actual.end_s(None, total_s=10), 10.0, places=3)

    def test_duracion_minima_de_subtitulo(self):
        actual = cp.ChronicleLine(5, "Gol")
        self.assertGreaterEqual(actual.end_s(None) - actual.t_s, 0.2)


class ClampLinesTest(unittest.TestCase):
    def test_ordena_y_limpia(self):
        lineas = cp.clamp_lines(
            [
                cp.ChronicleLine(20, "  segunda   frase  aquí  "),
                cp.ChronicleLine(5, "primera frase aquí"),
            ]
        )
        self.assertEqual([linea.t_s for linea in lineas], [5, 20])
        self.assertEqual(lineas[1].texto, "segunda frase aquí")

    def test_descarta_los_tozudos(self):
        lineas = cp.clamp_lines(
            [
                cp.ChronicleLine(0, "corta"),  # < MIN_LINE_CHARS
                cp.ChronicleLine(120, "frase válida y suficiente"),  # fuera del vídeo
                cp.ChronicleLine(10, "otra frase válida y suficiente"),
            ],
            duration=60,
        )
        self.assertEqual(len(lineas), 1)
        self.assertEqual(lineas[0].t_s, 10)

    def test_recorta_el_texto_larguisimo(self):
        lineas = cp.clamp_lines([cp.ChronicleLine(0, "a" * 900)])
        self.assertEqual(len(lineas[0].texto), cp.MAX_LINE_CHARS)

    def test_acota_el_numero_de_frases(self):
        # Se piden MÁS frases que el tope para que el tope sea lo que recorte.
        muchas = [
            cp.ChronicleLine(i * 5, f"frase numero {i} de la cronica")
            for i in range(cp.MAX_LINES + 10)
        ]
        self.assertEqual(len(cp.clamp_lines(muchas, max_lines=3)), 3)
        self.assertEqual(len(cp.clamp_lines(muchas, max_lines=999)), cp.MAX_LINES)


class DeliverableTest(unittest.TestCase):
    def _lineas(self):
        return cp.clamp_lines(
            [
                cp.ChronicleLine(0, "Arranca el partido"),
                cp.ChronicleLine(12, "Gol de Álex"),
            ]
        )

    def test_srt(self):
        srt = cp.chronicle_srt(self._lineas(), duration=30)
        self.assertIn("00:00:00,000 --> ", srt)
        self.assertIn("00:00:12,000 --> ", srt)
        self.assertIn("Gol de Álex", srt)
        self.assertLess(srt.index("Arranca"), srt.index("Gol de Álex"))

    def test_markdown(self):
        md = cp.chronicle_markdown(
            self._lineas(), video_name="peli.mp4", duration=90
        )
        self.assertIn("# Crónica · peli.mp4", md)
        self.assertIn("Duración: 01:30", md)
        self.assertIn("- **00:12** — Gol de Álex", md)

    def test_markdown_sin_frases(self):
        self.assertIn("_Sin frases._", cp.chronicle_markdown([]))


class PromptTest(unittest.TestCase):
    def test_pide_json_con_tiempos(self):
        prompt = cp.chronicle_prompt("informe", language="euskera", max_lines=5)
        self.assertIn("euskera", prompt)
        self.assertIn("máximo 5 frases", prompt)
        self.assertIn('"t_s"', prompt)
        self.assertIn("informe", prompt)

    def test_acota_las_frases(self):
        self.assertIn(f"máximo {cp.MAX_LINES} frases", cp.chronicle_prompt("x", max_lines=999))

    def test_permite_cronicas_largas(self):
        """Un partido de ~1 h necesita más de 24 frases: el tope da margen."""
        self.assertGreaterEqual(cp.MAX_LINES, 40)
        prompt = cp.chronicle_prompt("informe", max_lines=cp.MAX_LINES + 10)
        self.assertIn(f"máximo {cp.MAX_LINES} frases", prompt)

    def test_mete_los_datos_del_partido(self):
        """La alineación/marcador que da el usuario va al prompt como verdad."""
        prompt = cp.chronicle_prompt("informe", context="Muskis juega de azul")
        self.assertIn(cp.CONTEXT_LABEL, prompt)
        self.assertIn("Muskis juega de azul", prompt)

    def test_sin_contexto_no_hay_bloque_de_datos(self):
        self.assertNotIn(cp.CONTEXT_LABEL, cp.chronicle_prompt("informe"))

    def test_pide_cubrir_los_momentos_de_juego(self):
        informe = "\n".join(
            f"[00:{i:02d}] ESTADO: JUGANDO, ataque del equipo azul" for i in range(30)
        )
        prompt = cp.chronicle_prompt(informe, max_lines=60)
        self.assertIn("~30 momentos de JUEGO", prompt)


class JuegoTest(unittest.TestCase):
    """Distinguir juego de no-juego pese al ruido de las etiquetas."""

    def test_estado_manda_sobre_las_etiquetas(self):
        linea = (
            "[00:10] **ESTADO: JUGANDO** ataque del local | Etiquetas: banquillo, "
            "gol, jugadores en la pista, saludo o fila inicial"
        )
        informe = linea + "\n[00:20] ESTADO: PREVIO/SALUDO calentamiento"
        self.assertEqual(cp.momentos_de_juego(informe), 1)

    def test_prioriza_el_juego_al_muestrear(self):
        # 200 anotaciones, casi todas de previo: al muestrear debe colarse la única
        # de juego (antes se descartaba y la crónica se quedaba sin jugadas).
        notas = [(float(i), f"ESTADO: PREVIO/SALUDO detalle {i}") for i in range(200)]
        notas[50] = (50.0, "ESTADO: JUGANDO ataque peligroso del equipo azul")
        texto = cp.frame_notes_text(notas, max_chars=1000)  # ~10 líneas
        self.assertIn("JUGANDO", texto)


class ContextTest(unittest.TestCase):
    def test_normaliza_espacios_y_lineas_vacias(self):
        self.assertEqual(
            cp.context_block("  Equipos:  A  vs  B \n\n\n Gol de   Álex \n"),
            "Equipos: A vs B\nGol de Álex",
        )

    def test_vacio(self):
        self.assertEqual(cp.context_block(None), "")
        self.assertEqual(cp.context_block("   \n  \n"), "")

    def test_vision_sin_contexto_es_el_prompt_base(self):
        self.assertEqual(cp.vision_frame_prompt(""), cp.VISION_FRAME_PROMPT)

    def test_vision_con_contexto_lo_añade_sin_perder_lo_de_accion(self):
        prompt = cp.vision_frame_prompt("Muskis de azul, Corazonistas de rojo")
        self.assertTrue(prompt.startswith(cp.VISION_FRAME_PROMPT))
        self.assertIn("Muskis de azul", prompt)

    def test_vision_no_pasa_el_marcador_final(self):
        # Regresión: el «Marcador final» del contexto NO debe ir por fotograma
        # (contaminaba casi todas las descripciones, incluso al inicio). Los
        # equipos y colores sí se pasan.
        prompt = cp.vision_frame_prompt(
            "Equipos: A (azul) vs B (rojo)\nMarcador final: 28-25"
        )
        self.assertIn("azul", prompt)
        self.assertNotIn("28-25", prompt)

    def test_vision_con_disciplina_incluye_las_reglas(self):
        prompt = cp.vision_frame_prompt("", "balonmano")
        self.assertIn("Balonmano", prompt)
        self.assertIn("7 metros", prompt)

    def test_vision_sin_disciplina_no_mete_reglas(self):
        self.assertEqual(cp.vision_frame_prompt("", "generico"), cp.VISION_FRAME_PROMPT)

    def test_el_guion_lleva_las_reglas_de_la_disciplina(self):
        prompt = cp.chronicle_prompt("informe x", sport="balonmano")
        self.assertIn("REGLAS Y VOCABULARIO DE LA DISCIPLINA", prompt)
        self.assertIn("juego pasivo", prompt.lower())

    def test_el_guion_sin_disciplina_no_mete_reglas(self):
        prompt = cp.chronicle_prompt("informe x")
        self.assertNotIn("REGLAS Y VOCABULARIO DE LA DISCIPLINA", prompt)


class ParseChronicleTest(unittest.TestCase):
    def test_json_del_modelo(self):
        raw = '{"cronica": [{"t_s": 12, "texto": "Gol de Álex"}, {"t_s": 0, "texto": "Arranca el partido"}]}'
        lineas = cp.parse_chronicle(raw, duration=30)
        self.assertEqual([linea.t_s for linea in lineas], [0, 12])
        self.assertEqual(lineas[1].texto, "Gol de Álex")

    def test_acepta_otras_claves(self):
        raw = '{"lineas": [{"segundo": 3, "frase": "Primera jugada del partido"}]}'
        self.assertEqual(cp.parse_chronicle(raw)[0].t_s, 3)

    def test_ignora_entradas_sin_tiempo_o_sin_texto(self):
        raw = (
            '{"cronica": [{"texto": "sin tiempo"}, {"t_s": 4}, '
            '{"t_s": 6, "texto": "jugada buena del partido"}]}'
        )
        lineas = cp.parse_chronicle(raw)
        self.assertEqual(len(lineas), 1)
        self.assertEqual(lineas[0].t_s, 6)
        self.assertEqual(lineas[0].texto, "jugada buena del partido")

    def test_respuesta_en_texto_plano(self):
        lineas = cp.parse_chronicle("0:05 Primera jugada del partido")
        self.assertEqual(lineas[0].t_s, 5)

    def test_respuesta_inservible(self):
        with self.assertRaises(ValueError):
            cp.parse_chronicle("no pienso devolver nada de eso")
        with self.assertRaises(ValueError):
            cp.parse_chronicle('{"cronica": []}')

    def test_descarta_lo_que_se_sale_del_video(self):
        raw = '{"cronica": [{"t_s": 500, "texto": "muy tarde para el video"}]}'
        with self.assertRaises(ValueError):
            cp.parse_chronicle(raw, duration=60)


class NarrationTest(unittest.TestCase):
    def test_grafo_mezcla_con_retardos(self):
        grafo = cp.narration_filter_complex([(0.0, "a.wav"), (12.5, "b.wav")])
        self.assertIn("[0:a]aresample=48000", grafo)
        self.assertIn("adelay=0:all=1[voz0]", grafo)
        self.assertIn("adelay=12500:all=1[voz1]", grafo)
        self.assertIn("amix=inputs=2:normalize=0:duration=longest[a]", grafo)
        # Un solo canal: los motores de voz devuelven mono y el Ensamblador mezcla.
        self.assertIn("channel_layouts=mono", grafo)

    def test_sin_frases(self):
        with self.assertRaises(ValueError):
            cp.narration_filter_complex([])

    def test_comando(self):
        argv = cp.narration_command(
            "ffmpeg", [(0.0, "a.wav"), (2.0, "b.wav")], "narracion.wav"
        )
        self.assertEqual(argv[0], "ffmpeg")
        self.assertEqual(argv.count("-i"), 2)
        self.assertIn("[a]", argv)
        self.assertIn("pcm_s16le", argv)
        self.assertEqual(argv[-1], "narracion.wav")

    def test_nombre_de_salida(self):
        self.assertEqual(cp.default_output_name("C:/v/peli.mp4"), "peli_cronica")


class SnapLinesTest(unittest.TestCase):
    """Anclaje de los tiempos de la crónica a los momentos vistos."""

    def test_extrae_instantes_del_informe(self):
        inst = cp.report_timestamps("[00:12] gol\n[01:05] parada\n[1:02:03] fin")
        self.assertEqual(inst, [12.0, 65.0, 3723.0])

    def test_alinea_al_instante_mas_cercano(self):
        lineas = (cp.ChronicleLine(14.0, "casi"), cp.ChronicleLine(200.0, "lejos"))
        ajustadas = cp.snap_lines(lineas, [12.0, 60.0], tolerance=45.0)
        self.assertEqual([linea.t_s for linea in ajustadas], [12.0, 200.0])

    def test_deduplica_mismo_instante(self):
        lineas = (cp.ChronicleLine(11.0, "a"), cp.ChronicleLine(13.0, "b"))
        ajustadas = cp.snap_lines(lineas, [12.0], tolerance=45.0)
        self.assertEqual(len(ajustadas), 1)


class ParseTimestampTest(unittest.TestCase):
    """Inversa de format_timestamp: de «[MM:SS]» a segundos."""

    def test_minuto_segundo(self):
        self.assertEqual(vu.parse_timestamp("00:12"), 12.0)
        self.assertEqual(vu.parse_timestamp("[01:05] gol"), 65.0)

    def test_hora_minuto_segundo(self):
        self.assertEqual(vu.parse_timestamp("[1:02:03] fin"), 3723.0)

    def test_sin_marca(self):
        self.assertIsNone(vu.parse_timestamp("sin tiempo"))
        self.assertIsNone(vu.parse_timestamp(None))


class AnclajeTest(unittest.TestCase):
    """La crónica se ancla a la rejilla de instantes de Comprensión/Cuaderno."""

    def test_la_ref_manda_sobre_t_s(self):
        raw = '{"cronica": [{"ref": "00:12", "t_s": 999, "texto": "Gol de Álex"}]}'
        self.assertEqual(cp.parse_chronicle(raw)[0].t_s, 12.0)

    def test_acepta_ref_con_corchetes(self):
        raw = '{"cronica": [{"ref": "[01:05]", "texto": "Parada del portero"}]}'
        self.assertEqual(cp.parse_chronicle(raw)[0].t_s, 65.0)

    def test_anchor_coloca_en_la_rejilla(self):
        lineas = (cp.ChronicleLine(13.0, "casi"), cp.ChronicleLine(58.0, "también"))
        ancladas = cp.anchor_lines(lineas, [12.0, 60.0])
        self.assertEqual([linea.t_s for linea in ancladas], [12.0, 60.0])

    def test_anchor_sin_tolerancia(self):
        # A diferencia de snap_lines, aquí SIEMPRE cae en la anotación más cercana.
        lineas = (cp.ChronicleLine(200.0, "lejos"),)
        self.assertEqual(cp.anchor_lines(lineas, [12.0, 60.0])[0].t_s, 60.0)


    def test_anchor_no_pierde_frases(self):
        # Dos frases que apuntan al mismo instante: la segunda pasa al siguiente
        # libre (antes se descartaba y la crónica se quedaba en pocas líneas).
        lineas = tuple(cp.ChronicleLine(12.0 + i, f"f{i}") for i in range(4))
        ancladas = cp.anchor_lines(lineas, [12.0, 60.0, 120.0, 180.0])
        self.assertEqual(len(ancladas), 4)
        self.assertEqual([linea.t_s for linea in ancladas], [12.0, 60.0, 120.0, 180.0])


class RecorteInformeTest(unittest.TestCase):
    """El informe se acota para que quepa en la ventana de Ollama (num_ctx)."""

    def test_recorta_por_palabra_y_avisa(self):
        recortado = cp.clamp_report("palabra " * 5000, limit=100)
        self.assertLessEqual(len(recortado), 110)
        self.assertTrue(recortado.endswith("[…]"))

    def test_no_recorta_lo_que_cabe(self):
        self.assertEqual(cp.clamp_report("hola", limit=100), "hola")

    def test_la_transcripcion_se_acota(self):
        largo = "a" * (cp.MAX_TRANSCRIPT_CHARS + 100)
        recortado = cp.clamp_transcript(largo)
        self.assertLessEqual(len(recortado), cp.MAX_TRANSCRIPT_CHARS + 4)

    def test_el_informe_cubre_todo_el_video(self):
        # Con 500 fotogramas, el informe se muestrea: cubre del 00:00 al final.
        data = {
            "fotogramas": [
                {"t_s": float(i), "descripcion": "x" * 80} for i in range(500)
            ]
        }
        informe = cp.informe_report(data)
        self.assertLessEqual(len(informe), cp.MAX_REPORT_CHARS)
        self.assertIn("[00:00]", informe)
        self.assertIn("[08:19]", informe)  # último instante (499 s)

    def test_descripciones_largas_no_recortan_el_final(self):
        # Descripciones reales (300-1500 caracteres): el bloque debe acotarse por
        # CARACTERES, no por número de líneas, o clamp_report corta por el final y
        # la crónica solo ve el primer tercio del partido.
        data = {
            "fotogramas": [
                {"t_s": float(i * 10), "descripcion": "ESTADO: JUGANDO " + "y" * 500}
                for i in range(120)
            ]
        }
        informe = cp.informe_report(data)
        self.assertLessEqual(len(informe), cp.MAX_REPORT_CHARS)
        self.assertIn("[00:00]", informe)
        self.assertIn("[19:50]", informe)  # último instante (1190 s)

    def test_el_prompt_no_lleva_el_informe_entero(self):
        prompt = cp.chronicle_prompt("x" * 50000, duration_s=3126)
        # El prompt queda acotado (no se cuela el informe entero) y avisa del corte.
        # El margen cubre el andamiaje del prompt (instrucciones, datos, roster…).
        self.assertLess(len(prompt), cp.MAX_REPORT_CHARS + 3000)
        self.assertIn("[…]", prompt)


class PlaceLinesTest(unittest.TestCase):
    """Las frases se quedan en su momento: sólo se corrige la deriva pequeña."""

    def test_corrige_una_deriva_pequena(self):
        ajustadas = cp.place_lines([cp.ChronicleLine(12.0, "a")], [10.0, 60.0])
        self.assertEqual(ajustadas[0].t_s, 10.0)  # 2 s de deriva: dentro del margen

    def test_no_retrasa_una_frase_hacia_una_anotacion_posterior(self):
        # Una anotación 2 s DESPUÉS no debe empujar la frase: se oiría cuando la
        # acción ya ha pasado (el balón ya es del otro equipo).
        ajustadas = cp.place_lines([cp.ChronicleLine(12.0, "a")], [14.0])
        self.assertEqual(ajustadas[0].t_s, 12.0)

    def test_una_frase_lejana_no_salta_a_otro_instante(self):
        # Antes (anchor_lines) una frase del calentamiento saltaba al instante libre
        # del minuto 3:40; ahora se respeta su momento.
        ajustadas = cp.place_lines([cp.ChronicleLine(5.0, "calientan")], [0.0, 220.0])
        self.assertEqual(ajustadas[0].t_s, 5.0)

    def test_separa_las_frases_que_caen_juntas(self):
        lineas = [cp.ChronicleLine(10.0, "a"), cp.ChronicleLine(10.2, "b")]
        ajustadas = cp.place_lines(lineas, [])
        self.assertGreaterEqual(ajustadas[1].t_s - ajustadas[0].t_s, 0.5)


class InformeReportTest(unittest.TestCase):
    """El informe del Cuaderno se convierte en el informe que usa la crónica."""

    def test_reutiliza_fotogramas_audio_y_contexto(self):
        data = {
            "video_path": "C:/v/peli.mp4",
            "duracion_s": 100.0,
            "contexto": "Equipos: A vs B",
            "fotogramas": [{"t_s": 12.0, "descripcion": "gol de Álex"}],
            "transcripcion": "GOL!",
        }
        informe = cp.informe_report(data)
        self.assertIn("00:12", informe)
        self.assertIn("gol de Álex", informe)
        self.assertIn("GOL!", informe)
        self.assertIn("Equipos: A vs B", informe)

    def test_sin_fotogramas_avisa(self):
        informe = cp.informe_report({"fotogramas": []})
        self.assertIn("no se pudo describir", informe)

    def test_informe_con_marcador_real(self):
        # El informe del Cuaderno puede traer marcador/goles del OCR: la crónica
        # los recibe como datos reales, sin adivinar el resultado.
        data = {
            "fotogramas": [{"t_s": 1.0, "descripcion": "saque"}],
            "marcador": [
                {"t_s": 0.0, "izquierda": 0, "derecha": 0},
                {"t_s": 60.0, "izquierda": 1, "derecha": 0},
            ],
            "goles": [{"t_s": 60.0, "lado": "izquierda", "marcador": "1-0"}],
        }
        informe = cp.informe_report(data)
        self.assertIn("MARCADOR REAL LEÍDO POR OCR", informe)
        self.assertIn("01:00] 1-0", informe)
        self.assertIn("Goles detectados: 1.", informe)


class DatosDelInformeTest(unittest.TestCase):
    """La alineación del informe llega a la crónica aunque la caja esté vacía."""

    INFORME = (
        "DATOS DEL PARTIDO O DE LA GRABACIÓN (los aporta el usuario y son correctos):\n"
        "Equipos: Muskis (local, camiseta azul) vs Corazonistas (visitante, roja).\n"
        "Alineación Muskis: 1 Unai (portero), 4 Álex, 7 Marc.\n"
        "Nota faltan los minutos 30-40 del segundo tiempo\n"
        "LO QUE SE VE EN EL VÍDEO (una línea por instante analizado):\n"
        "[00:12] Jugadora con dorsal 7 marca."
    )

    def test_extrae_solo_los_datos_del_partido(self):
        datos = cp.datos_del_informe(self.INFORME)
        self.assertIn("Alineación Muskis", datos)
        # Las notas del usuario sin «:» también son datos y no se pierden.
        self.assertIn("Nota faltan los minutos", datos)
        self.assertNotIn("[00:12]", datos)
        self.assertNotIn("DATOS DEL PARTIDO", datos)

    def test_el_prompt_usa_la_alineacion_del_informe(self):
        # Sin contexto en la caja: la alineación sale del propio informe.
        prompt = cp.chronicle_prompt(self.INFORME)
        self.assertIn("Dorsal 7: Marc", prompt)

    def test_sin_alineacion_no_hay_bloque(self):
        self.assertEqual(cp.datos_del_informe("[00:12] gol"), "")


class PerceptionReportTest(unittest.TestCase):
    """La percepción de Comprensión alimenta la crónica sin volver a mirar el vídeo."""

    def test_reutiliza_anotaciones_resumen_audio_y_contexto(self):
        data = {
            "video_path": "C:/v/peli.mp4",
            "duracion_s": 100.0,
            "contexto": "Equipos: A vs B",
            "resumen": "Partido igualado con victoria visitante.",
            "fotogramas": [
                {"t_s": 12.0, "anotacion": "jugadores y balón; texto: 1-0"},
            ],
            "transcripcion": "GOL!",
        }
        informe = cp.perception_report(data)
        self.assertIn("00:12", informe)
        self.assertIn("jugadores y balón", informe)
        self.assertIn("RESUMEN DE COMPRENSIÓN", informe)
        self.assertIn("Partido igualado", informe)
        self.assertIn("GOL!", informe)
        self.assertIn("Equipos: A vs B", informe)

    def test_sin_fotogramas_avisa(self):
        informe = cp.perception_report({"fotogramas": []})
        self.assertIn("no se pudo describir", informe)

    def test_la_alineacion_del_contexto_traduce_dorsales(self):
        data = {
            "contexto": "Alineación Muskis: 7 Marc",
            "fotogramas": [{"t_s": 5.0, "anotacion": "marca el dorsal 7"}],
        }
        informe = cp.perception_report(data)
        self.assertIn("Marc", informe)


if __name__ == "__main__":
    unittest.main()
