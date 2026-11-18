"""Tests de la pestaña Cuaderno (services/notes_params.py)."""

import json
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from services import notes_params as nts  # noqa: E402


class SampleFramesTest(unittest.TestCase):
    def test_un_fotograma_por_segundo(self):
        instantes = nts.sample_frames(10.0, fps=1.0, max_frames=999)
        self.assertEqual(len(instantes), 10)
        self.assertAlmostEqual(instantes[0], 0.5, places=3)

    def test_video_mas_corto_que_un_fotograma(self):
        self.assertEqual(nts.sample_frames(0.4, fps=1.0), [0.2])

    def test_duracion_invalida(self):
        self.assertEqual(nts.sample_frames(0), [])
        self.assertEqual(nts.sample_frames("x"), [])

    def test_respeta_el_tope(self):
        instantes = nts.sample_frames(10000.0, fps=1.0, max_frames=5)
        self.assertEqual(len(instantes), 5)

    def test_clamps(self):
        self.assertEqual(nts.clamp_fps("x"), nts.DEFAULT_FPS)
        self.assertGreaterEqual(nts.clamp_fps(0), 0.1)
        self.assertEqual(nts.clamp_max_frames(10**9), nts.MAX_MAX_FRAMES)


class SpreadTest(unittest.TestCase):
    def test_reparte_incluyendo_extremos(self):
        self.assertEqual(nts.spread([1, 2, 3, 4, 5], 3), [1, 3, 5])

    def test_menos_o_igual(self):
        self.assertEqual(nts.spread([1, 2], 5), [1, 2])
        self.assertEqual(nts.spread([1, 2, 3], 0), [])


class ExtractJsonTest(unittest.TestCase):
    def test_directo(self):
        self.assertEqual(nts.extract_json('{"a": 1}'), {"a": 1})

    def test_con_valla_markdown(self):
        texto = 'Aquí tienes:\n```json\n{"a": 2}\n```\nGracias.'
        self.assertEqual(nts.extract_json(texto), {"a": 2})

    def test_con_texto_alrededor(self):
        self.assertEqual(nts.extract_json('bla {"a": 3} bla'), {"a": 3})

    def test_sin_json(self):
        with self.assertRaises(nts.NotesError):
            nts.extract_json("no hay json")


class ParseAnalysisTest(unittest.TestCase):
    _EJEMPLO = {
        "titulo": "Prueba",
        "duracion_s": 120,
        "lineas_narrativas": [
            {"nombre": "Principal", "eventos": [{"t_s": 10, "descripcion": "arranque"}]}
        ],
        "estructura": {"titulo": "Película", "inicio_s": 0, "fin_s": 120, "hijos": [{"titulo": "Acto I"}]},
        "emocion": [
            {"t_s": 60, "intensidad": 0.8, "etiqueta": "clímax"},
            {"t_s": 5, "intensidad": 0.2, "etiqueta": "calma"},
        ],
        "segmentos": [
            {"inicio_s": 0, "fin_s": 60, "titulo": "Inicio", "resumen": "…", "escenas": ["a", "b"]}
        ],
    }

    def test_parsea_y_ordena(self):
        analysis = nts.parse_analysis(self._EJEMPLO)
        self.assertEqual(analysis.titulo, "Prueba")
        self.assertEqual(len(analysis.lineas), 1)
        self.assertEqual(analysis.lineas[0].color, nts.DEFAULT_LANE_COLORS[0])
        # La emoción se ordena por tiempo.
        self.assertEqual([p.t_s for p in analysis.emocion], [5, 60])
        self.assertEqual(analysis.estructura.hijos[0].titulo, "Acto I")
        self.assertEqual(analysis.segmentos[0].escenas, ("a", "b"))

    def test_acepta_texto_json(self):
        analysis = nts.parse_analysis(json.dumps(self._EJEMPLO))
        self.assertEqual(analysis.titulo, "Prueba")

    def test_intensidad_acotada(self):
        analysis = nts.parse_analysis({"emocion": [{"t_s": 0, "intensidad": 5}]})
        self.assertEqual(analysis.emocion[0].intensidad, nts.EMOTION_MAX)

    def test_formato_inesperado(self):
        with self.assertRaises(nts.NotesError):
            nts.parse_analysis({"otra": "cosa"})

    def test_vacio(self):
        analysis = nts.parse_analysis({"lineas_narrativas": []})
        self.assertTrue(analysis.is_empty)

    def test_acota_tiempos_a_la_duracion(self):
        # El modelo devolvía segundos que no existen (más allá del final del vídeo).
        analysis = nts.parse_analysis(
            {
                "duracion_s": 3126,
                "lineas_narrativas": [
                    {"nombre": "L", "eventos": [{"t_s": 4835, "descripcion": "x"}]}
                ],
                "emocion": [{"t_s": 4201, "intensidad": 0.5, "etiqueta": "e"}],
                "segmentos": [{"inicio_s": 3620, "fin_s": 3925, "titulo": "s"}],
            }
        )
        self.assertEqual(analysis.lineas[0].eventos[0].t_s, 3126)
        self.assertEqual(analysis.emocion[0].t_s, 3126)
        self.assertEqual(analysis.segmentos[0].inicio_s, 3126)
        self.assertEqual(analysis.segmentos[0].fin_s, 3126)

    def test_duracion_real_manda_sobre_el_json(self):
        analysis = nts.parse_analysis(
            {"duracion_s": 3126, "lineas_narrativas": []}, duration_s=3126.87
        )
        self.assertEqual(analysis.duracion_s, 3126.87)

    def test_duracion_real_acota_los_tiempos(self):
        analysis = nts.parse_analysis(
            {
                "lineas_narrativas": [
                    {"nombre": "L", "eventos": [{"t_s": 5000, "descripcion": "x"}]}
                ]
            },
            duration_s=3126.87,
        )
        self.assertEqual(analysis.lineas[0].eventos[0].t_s, 3126.87)


class BuildMarkdownTest(unittest.TestCase):
    def test_incluye_secciones(self):
        analysis = nts.parse_analysis(ParseAnalysisTest._EJEMPLO)
        md = nts.build_markdown(analysis)
        for seccion in ("## Estructura narrativa", "## Curva de emoción", "## Líneas narrativas", "## Segmentos y escenas"):
            self.assertIn(seccion, md)


class PackageTest(unittest.TestCase):
    def test_instrucciones_incluyen_esquema(self):
        texto = nts.package_instructions("film.mp4", 90)
        self.assertIn("lineas_narrativas", texto)
        self.assertIn("film.mp4", texto)

    def test_construye_zip_con_miembros(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            frame = tmp / "f.png"
            frame.write_bytes(b"\x89PNG")
            zip_path = nts.build_package(
                tmp / "paquete.zip",
                frames=[(0.5, frame)],
                video_name="film.mp4",
                duration_s=10,
                transcript="hola",
            )
            self.assertTrue(zip_path.is_file())
            with zipfile.ZipFile(zip_path) as zf:
                nombres = set(zf.namelist())
            self.assertIn("instrucciones.txt", nombres)
            self.assertIn("esquema.json", nombres)
            self.assertIn("manifiesto.json", nombres)
            self.assertIn("transcripcion.txt", nombres)
            self.assertIn("fotogramas/frame_00000.png", nombres)


class SampleFramesCoverageTest(unittest.TestCase):
    """El muestreo debe cubrir TODO el vídeo, no solo el principio."""

    def test_reparte_por_todo_el_video_si_supera_el_tope(self):
        # 52 min a 1 fps (3126 s) superan el tope de 1800: debe llegar al final.
        instantes = nts.sample_frames(3126.0, fps=1.0, max_frames=1800)
        self.assertEqual(len(instantes), 1800)
        self.assertGreater(instantes[-1], 3000.0)   # llega cerca del final
        self.assertLess(instantes[-1], 3126.0)
        self.assertLess(instantes[0], 5.0)          # y empieza al principio

    def test_corto_no_reparte(self):
        instantes = nts.sample_frames(10.0, fps=1.0, max_frames=1800)
        self.assertEqual(len(instantes), 10)
        self.assertAlmostEqual(instantes[0], 0.5, places=3)


class PromptDisciplinaTest(unittest.TestCase):
    """La disciplina y el contexto deben llegar a los prompts y al paquete."""

    def test_el_primer_de_balonmano_tiene_reglas(self):
        primer = nts.sport_primer("balonmano")
        for clave in ("7 metros", "6 m", "exclusión", "portero", "juego pasivo"):
            self.assertIn(clave.lower(), primer.lower())

    def test_sin_disciplina_no_hay_reglas(self):
        self.assertEqual(nts.sport_primer("generico"), "")
        self.assertEqual(nts.sport_primer(""), "")

    def test_el_prompt_de_vision_pide_estado_y_dorsal(self):
        # El prompt pide el estado de juego y el dorsal, y prohíbe inventar el
        # marcador/reloj (de eso se encarga el OCR).
        prompt = nts.VISION_FRAME_PROMPT.upper()
        for clave in ("JUGANDO", "DORSAL"):
            self.assertIn(clave, prompt)
        self.assertIn("NO ESCRIBAS EL MARCADOR", prompt)

    def test_las_anotaciones_se_muestrean_si_son_muchas(self):
        # 500 fotogramas no caben con líneas legibles: se muestrean (primera,
        # última y reparto) en vez de dejar cada descripción en 30 caracteres.
        notas = [(float(i), "JUGANDO. Jugadora con el balón en el centro, avanza.") for i in range(500)]
        prompt = nts.analysis_prompt("f.mp4", 3126, notas)
        self.assertIn("[t=00:00]", prompt)
        self.assertIn("[t=08:19]", prompt)  # último instante (499 s)
        self.assertLess(len(prompt), 13000)
        # Cada línea conserva una descripción legible (no 30 caracteres).
        self.assertIn("Jugadora con el balón en el centro", prompt)

    def test_el_prompt_de_vision_evita_evasivas(self):
        prompt = nts.VISION_FRAME_PROMPT
        self.assertIn("no se ve el dorsal", prompt)  # se prohíbe esa muletilla
        self.assertIn("no lo menciones", prompt)

    def test_el_prompt_limita_el_tamano_de_la_respuesta(self):
        # Sin esta instrucción, el modelo escribía un párrafo por fotograma y se
        # cortaba por tokens («se agotó el espacio de la respuesta»).
        prompt = nts.analysis_prompt("f.mp4", 60, [(1.0, "x")])
        self.assertIn("como máximo 6 líneas", prompt)

    def test_los_bloques_grandes_se_pueden_acotar(self):
        notas = [(float(i), "x" * 400) for i in range(120)]
        largo = nts.analysis_prompt("f.mp4", 3126, notas)
        corto = nts.analysis_prompt("f.mp4", 3126, notas, max_annotations=3000)
        self.assertLess(len(corto), len(largo))

    def test_la_transcripcion_del_prompt_se_acota(self):
        # La transcripción de un partido largo no cabe en num_ctx: se recorta.
        prompt = nts.analysis_prompt("f.mp4", 60, [(1.0, "x")], "palabra " * 5000)
        self.assertLess(len(prompt), 9000)
        self.assertTrue(nts.clamp_transcript("x" * 9000).endswith("[…]"))

    def test_las_anotaciones_se_acotan_sin_perder_cobertura(self):
        # 120 notas largas deben caber y conservar el primer y último instante
        # (antes el prompt no cabía en num_ctx y el análisis empezaba en 32:10).
        notas = [(float(i), "x" * 400) for i in range(120)]
        prompt = nts.analysis_prompt("f.mp4", 3126.0, notas)
        self.assertLess(len(prompt), 14000)
        self.assertIn("[t=00:00]", prompt)
        self.assertIn("[t=01:59]", prompt)

    def test_el_prompt_de_vision_incluye_la_disciplina(self):
        prompt = nts.vision_frame_prompt("balonmano")
        self.assertIn("Balonmano", prompt)
        self.assertIn("área de 6 m", prompt)

    def test_el_prompt_de_vision_no_lleva_el_contexto(self):
        # Regresión: el contexto del usuario (p. ej. «Marcador final: 42-29») NO
        # debe ir en el prompt por fotograma; se colaba en casi todas las
        # anotaciones. Solo se usa al COMPONER el cuaderno (analysis_prompt).
        prompt = nts.vision_frame_prompt("balonmano")
        self.assertNotIn(nts.CONTEXT_LABEL, prompt)
        self.assertNotIn("DATOS DEL PARTIDO", prompt)

    def test_el_prompt_de_analisis_incluye_disciplina_y_contexto(self):
        prompt = nts.analysis_prompt(
            "peli.mp4", 120, [(1.0, "saque de centro")],
            context="Equipos: A vs B", sport="balonmano",
        )
        self.assertIn("Balonmano", prompt)
        self.assertIn("7 metros", prompt)
        self.assertIn("Equipos: A vs B", prompt)

    def test_el_prompt_de_analisis_incluye_el_marcador_ocr(self):
        prompt = nts.analysis_prompt(
            "f.mp4", 100, [(1.0, "x")], marcador="[60 s] 1-0"
        )
        self.assertIn("Marcador real leído por OCR", prompt)
        self.assertIn("[60 s] 1-0", prompt)

    def test_sin_marcador_no_hay_bloque(self):
        prompt = nts.analysis_prompt("f.mp4", 100, [(1.0, "x")])
        self.assertNotIn("Marcador real leído por OCR", prompt)

    def test_el_paquete_lleva_reglas_y_contexto(self):
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = nts.build_package(
                Path(tmp) / "p.zip",
                frames=[],
                video_name="f.mp4",
                sport="balonmano",
                context="Equipos: A vs B",
            )
            with zipfile.ZipFile(zip_path) as zf:
                nombres = set(zf.namelist())
                instrucciones = zf.read("instrucciones.txt").decode("utf-8")
        self.assertIn("reglas_disciplina.txt", nombres)
        self.assertIn("contexto.txt", nombres)
        self.assertIn("Balonmano", instrucciones)


class FrameAnnotationTest(unittest.TestCase):
    """Cada fotograma une la descripción de la visión y la percepción."""

    def test_une_vision_y_percepcion(self):
        texto = nts.frame_annotation("gol del 7 azul", "Etiquetas: balón, grada")
        self.assertIn("gol del 7 azul", texto)
        self.assertIn("Etiquetas: balón", texto)

    def test_sin_percepcion_deja_la_vision(self):
        self.assertEqual(nts.frame_annotation("solo visión", ""), "solo visión")


class InformeTest(unittest.TestCase):
    """Informe reutilizable que el Cuaderno deja para la Crónica."""

    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = nts.informe_to_dict(
                video_path="C:/v/peli.mp4",
                duration_s=100.0,
                frames=[(1.0, "saque de centro"), (60.0, "gol")],
                transcript="hola",
                sport="balonmano",
                context="Equipos: A vs B",
            )
            path = nts.save_informe(Path(tmp) / nts.INFORME_FILENAME, data)
            leido = nts.load_informe(path)
        self.assertEqual(leido["video_name"], "peli.mp4")
        self.assertEqual(leido["disciplina"], "balonmano")
        self.assertEqual(
            nts.informe_frames(leido),
            [(1.0, "saque de centro"), (60.0, "gol")],
        )

    def test_informe_invalido(self):
        with tempfile.TemporaryDirectory() as tmp:
            malo = Path(tmp) / "x.json"
            malo.write_text('{"otra": 1}', encoding="utf-8")
            with self.assertRaises(nts.NotesError):
                nts.load_informe(malo)


class ContrasteContextoTest(unittest.TestCase):
    """Avisos deterministas del contexto frente a lo que ve/oye el vídeo."""

    def test_sin_contexto_no_hay_avisos(self):
        data = {"fotogramas": [{"descripcion": "x"}]}
        self.assertEqual(nts.contrast_context("", data), [])

    def test_sin_datos_no_hay_avisos(self):
        self.assertEqual(nts.contrast_context("Marcador final: 28-25", {}), [])

    def test_marcador_que_no_aparece(self):
        data = {"fotogramas": [{"descripcion": "marcador 14-12"}], "transcripcion": ""}
        avisos = nts.contrast_context("Marcador final: 28-25", data)
        self.assertTrue(any("28-25" in aviso for aviso in avisos))

    def test_marcador_coincidente_no_avisa(self):
        data = {"fotogramas": [{"descripcion": "marcador 28-25"}], "transcripcion": ""}
        self.assertEqual(nts.contrast_context("Marcador final: 28-25", data), [])

    def test_color_en_plural_cuenta_como_color(self):
        # El contexto real dice «camisetas rojas»: debe contar como rojo.
        data = {"fotogramas": [{"descripcion": "camiseta azul"}]}
        avisos = nts.contrast_context(
            "Equipos: Muskis camiseta azul vs Corazonistas camisetas rojas", data
        )
        self.assertTrue(any("rojo" in aviso for aviso in avisos))

    def test_color_ausente_avisa_solo_si_hay_colores(self):
        contexto = "Equipos: Muskis (local, azul) vs Corazonistas (visitante, rojo)"
        con_color = {"fotogramas": [{"descripcion": "camiseta azul"}]}
        avisos = nts.contrast_context(contexto, con_color)
        self.assertTrue(any("rojo" in aviso for aviso in avisos))
        self.assertFalse(any("«azul»" in aviso for aviso in avisos))
        # Si el vídeo no describe ningún color, no se puede contrastar: sin avisos.
        sin_color = {"fotogramas": [{"descripcion": "jugadores en la pista"}]}
        self.assertEqual(nts.contrast_context(contexto, sin_color), [])

    def test_nombres_de_equipo_ausentes_en_la_transcripcion(self):
        data = {"fotogramas": [], "transcripcion": "hola y adios"}
        avisos = nts.contrast_context("Equipos: Muskis vs Corazonistas", data)
        self.assertTrue(
            any("Muskis" in aviso and "Corazonistas" in aviso for aviso in avisos)
        )

    def test_percepcion_de_comprension_usa_anotacion(self):
        # comprension.json usa la clave 'anotacion', no 'descripcion'.
        data = {
            "fotogramas": [{"t_s": 1.0, "anotacion": "marcador 14-12"}],
            "transcripcion": "",
        }
        avisos = nts.contrast_context("Marcador: 28-25", data)
        self.assertTrue(any("28-25" in aviso for aviso in avisos))

    def test_marcador_con_separador_repetido_cuenta_igual(self):
        # «42 --29» en el contexto y «42 29» leído por OCR: es el mismo tanteo.
        data = {"fotogramas": [{"descripcion": "marcador 42 29"}], "transcripcion": ""}
        self.assertEqual(nts.contrast_context("Marcador final: 42 --29", data), [])

    def test_nombres_de_equipo_limpian_colores_y_adornos(self):
        contexto = (
            "Equipos: de azul Pan Menesa Muskiz (local). camiseta azul, "
            "Baigene Corazonistas (visitante), camisetas rojas"
        )
        data = {"fotogramas": [{"descripcion": "x"}], "transcripcion": "hola y adios"}
        avisos = nts.contrast_context(contexto, data)
        nombre_aviso = next(aviso for aviso in avisos if "Nombres de equipo" in aviso)
        self.assertIn("Pan Menesa Muskiz", nombre_aviso)
        self.assertIn("Baigene Corazonistas", nombre_aviso)
        self.assertNotIn("camiseta", nombre_aviso.lower())

    def test_no_juzga_nombres_con_ruido_de_grada(self):
        # Transcripción de silbatos/aplausos (sin habla en español): no avisa.
        data = {
            "fotogramas": [{"descripcion": "marcador 42 29"}],
            "transcripcion": "APPLAUSE WHISTLE CHEERING That's it Very good",
        }
        self.assertEqual(
            nts.contrast_context("Equipos: Muskis vs Corazonistas", data), []
        )


class ComponerCuadernoTest(unittest.TestCase):
    """Reintento corto cuando el modelo se queda sin espacio (Cuaderno)."""

    _JSON = (
        '{"lineas_narrativas": [{"nombre": "L", '
        '"eventos": [{"t_s": 1, "descripcion": "x"}]}]}'
    )

    def test_reintenta_mas_corto_si_se_agota(self):
        from engines.base_engine import EngineError
        from ui import notes_widget as nw

        llamadas: list[str] = []

        def chat(prompt: str) -> str:
            llamadas.append(prompt)
            if len(llamadas) == 1:
                raise EngineError(
                    "Ollama: se agotó el espacio de la respuesta (num_predict=-1)…"
                )
            return self._JSON

        avisos: list[int] = []
        analysis = nw.componer_cuaderno(
            chat,
            ["prompt largo " * 200, "prompt corto"],
            parsear=nts.parse_analysis,
            aviso=avisos.append,
        )
        self.assertEqual(len(llamadas), 2)
        self.assertLess(len(llamadas[1]), len(llamadas[0]))
        self.assertEqual(avisos, [1])
        self.assertEqual(analysis.lineas[0].nombre, "L")

    def test_no_reintenta_con_otros_errores(self):
        from engines.base_engine import EngineError
        from ui import notes_widget as nw

        def chat(_prompt: str) -> str:
            raise EngineError("No se pudo conectar a Ollama. ¿Está arrancado?")

        with self.assertRaises(EngineError):
            nw.componer_cuaderno(chat, ["a", "b"], parsear=nts.parse_analysis)


class WorkerCancelTest(unittest.TestCase):
    """Regresión: los workers del Cuaderno deben poder comprobar la cancelación.

    ``PackageWorker`` y ``AnalyzeWorker`` llamaban a ``self._check_cancel()`` sin
    definirla, así que el paquete/el análisis morían con «'PackageWorker' object
    has no attribute '_check_cancel'» en el primer fotograma.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_los_workers_del_cuaderno_tienen_check_cancel(self):
        from ui import notes_widget

        for clase in (notes_widget.PackageWorker, notes_widget.AnalyzeWorker):
            self.assertTrue(
                callable(getattr(clase, "_check_cancel", None)), clase.__name__
            )

    def test_check_cancel_lanza_al_interrumpir(self):
        from unittest import mock

        from engines.base_engine import CANCEL_MESSAGE, EngineError
        from ui.workers import ThreadWorker

        class _Sencillo(ThreadWorker):
            def work(self):  # pragma: no cover - no se ejecuta aquí
                return None

        worker = _Sencillo()
        worker._check_cancel()  # sin interrupción: no lanza
        # ``requestInterruption`` no marca nada en un hilo que no se ha arrancado,
        # así que se simula la interrupción para probar el atajo de cancelación.
        with mock.patch.object(worker, "isInterruptionRequested", return_value=True):
            with self.assertRaises(EngineError) as ctx:
                worker._check_cancel()
        self.assertEqual(str(ctx.exception), CANCEL_MESSAGE)
        worker.deleteLater()


if __name__ == "__main__":
    unittest.main()
