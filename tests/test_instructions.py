"""Instrucciones del usuario en las tres pestañas (qué contar y con qué prioridad).

Se comprueba que:
- el ejemplo compartido (balonmano) existe y pide alineación, goles, paradas y 7 m;
- las instrucciones entran en los prompts de Comprensión, Cuaderno y Crónica;
- cada pestaña las pasa a su worker.

Sin red ni GPU: los workers se sustituyen por dobles.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from core.config_manager import ConfigManager  # noqa: E402
from core.engine_manager import EngineManager  # noqa: E402
from services import chronicle_params as cp  # noqa: E402
from services import notes_params as nts  # noqa: E402
from services import sport_params as sp  # noqa: E402
from services import video_understanding as vu  # noqa: E402
from ui import chronicle_widget, notes_widget, understanding_widget  # noqa: E402


class EjemploTest(unittest.TestCase):
    def test_el_ejemplo_pide_lo_que_quiere_el_usuario(self):
        texto = sp.INSTRUCTIONS_EXAMPLE.lower()
        self.assertIn("polideportivo", texto)
        self.assertIn("fecha", texto)
        self.assertIn("alineación", texto)
        self.assertIn("goles", texto)
        self.assertIn("paradas", texto)
        self.assertIn("7 metros", texto)
        self.assertIn("calientan", texto)


class AperturaTest(unittest.TestCase):
    """La crónica abre con sede, fecha y alineación de los DOS equipos."""

    CONTEXTO = (
        "Lugar: Polideportivo de Muskiz\n"
        "Fecha: 9/10/2026\n"
        "Equipos: Muskis (azul) vs Corazonistas (rojo)\n"
        "Alineación Muskis: 1 Unai, 4 Álex\n"
        "Alineación Corazonistas: 3 Amanda, 8 Paula"
    )

    def test_la_apertura_lleva_sede_fecha_y_los_dos_equipos(self):
        lineas = cp.opening_lines(self.CONTEXTO)
        unido = " ".join(linea.texto for linea in lineas)
        self.assertIn("Polideportivo de Muskiz", unido)
        self.assertIn("9/10/2026", unido)
        self.assertIn("Muskis", unido)
        self.assertIn("Corazonistas", unido)
        self.assertIn("Alineación del Muskis", unido)
        self.assertIn("1 Unai", unido)
        self.assertEqual(lineas[0].t_s, 0.0)
        self.assertTrue(
            all(b.t_s > a.t_s for a, b in zip(lineas, lineas[1:])),
            "las frases de apertura deben ir separadas en el tiempo",
        )

    def test_sin_datos_no_hay_apertura(self):
        self.assertEqual(cp.opening_lines(""), ())
        self.assertEqual(cp.opening_lines("Marcador final: 1-0"), ())

    def test_reconoce_la_fecha_sin_dos_puntos(self):
        # En el cuaderno real la fecha va como «Fecha 09/10/2026» (sin «:»).
        lineas = cp.opening_lines("Fecha 09/10/2026\nLugar: Polideportivo de Muskiz")
        unido = " ".join(linea.texto for linea in lineas)
        self.assertIn("09/10/2026", unido)
        self.assertIn("Polideportivo de Muskiz", unido)

    def test_el_mapa_de_equipos_por_color_llega_a_la_vision(self):
        contexto = "Equipos: MUSKIZ camiseta azul (local) vs CORAZONISTAS roja (visitante)"
        prompt = cp.vision_frame_prompt(contexto, "balonmano")
        self.assertIn("Azul", prompt)
        self.assertIn("LOCAL", prompt)
        self.assertIn("VISITANTE", prompt)
        self.assertIn("tiene el balón", prompt.lower())

    def test_prepend_opening_quita_las_frases_que_se_pisan(self):
        apertura = cp.opening_lines(self.CONTEXTO)
        fin = apertura[-1].t_s + cp._duracion_hablada(apertura[-1].texto)
        dentro = cp.ChronicleLine(fin - 1.0, "frase que se pisaría")
        fuera = cp.ChronicleLine(fin + 1.0, "frase posterior")
        unidas = cp.prepend_opening(apertura, [dentro, fuera])
        textos = [linea.texto for linea in unidas]
        self.assertNotIn("frase que se pisaría", textos)
        self.assertIn("frase posterior", textos)
        self.assertEqual(unidas[: len(apertura)], apertura)


class PromptsTest(unittest.TestCase):
    def test_cronica_lleva_las_instrucciones(self):
        vision = cp.vision_frame_prompt("", "balonmano", "céntrate en las paradas")
        self.assertIn("céntrate en las paradas", vision)
        guion = cp.chronicle_prompt("informe", instructions="céntrate en las paradas")
        self.assertIn("céntrate en las paradas", guion)
        self.assertIn(sp.INSTRUCTIONS_LABEL, guion)

    def test_el_prompt_de_vision_de_la_cronica_pide_el_estado(self):
        self.assertIn("ESTADO", cp.vision_frame_prompt())
        self.assertIn("PARADA", cp.VISION_FRAME_PROMPT)

    def test_la_cronica_prioriza_goles_paradas_y_7_metros(self):
        guion = cp.chronicle_prompt("informe")
        self.assertIn("PRIORIDAD", guion)
        self.assertIn("PARADAS", guion)
        self.assertIn("7 metros", guion)

    def test_cuaderno_lleva_las_instrucciones(self):
        vision = nts.vision_frame_prompt("balonmano", "cuenta los 7 metros")
        self.assertIn("cuenta los 7 metros", vision)
        prompt = nts.analysis_prompt(
            "peli.mp4", 60.0, [(1.0, "gol")], instructions="cuenta los 7 metros"
        )
        self.assertIn("cuenta los 7 metros", prompt)
        self.assertIn(nts.INSTRUCTIONS_LABEL, prompt)

    def test_comprension_lleva_las_instrucciones(self):
        prompt = vu.aggregate_prompt(
            vu.UnderstandingPlan(), 3, True, instructions="destaca las paradas"
        )
        self.assertIn("destaca las paradas", prompt)
        self.assertIn(vu.INSTRUCTIONS_LABEL, prompt)


class EquiposMarcadorTest(unittest.TestCase):
    """Local/visitante y marcador real, para no decir que gana quien no va delante."""

    CONTEXTO = (
        "Equipos: Muskis camiseta azul (local) vs Corazonistas camisetas rojas "
        "(visitante)"
    )

    def test_detecta_local_y_visitante(self):
        local, visitante = cp.teams_order(self.CONTEXTO)
        self.assertIn("Muskis", local)
        self.assertIn("Corazonistas", visitante)

    def test_el_marcador_lleva_nombres_y_aviso(self):
        data = {
            "contexto": self.CONTEXTO,
            "marcador": [{"t_s": 60.0, "izquierda": 1, "derecha": 0}],
        }
        bloque = cp.marcador_del_informe(data, context=self.CONTEXTO)
        self.assertIn("01:00] 1-0", bloque)
        self.assertIn("Muskis", bloque)
        self.assertIn("IZQUIERDA", bloque)
        self.assertIn("NO digas que va por delante", bloque)

    def test_el_guion_avisa_del_liderato(self):
        prompt = cp.chronicle_prompt("informe", context=self.CONTEXTO)
        self.assertIn("LOCAL", prompt)
        self.assertIn("VISITANTE", prompt)
        self.assertIn("TIENE EL BALÓN ataca", prompt)
        self.assertIn("NO digas que va por delante", prompt)


class WidgetsInstruccionesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.manager = EngineManager(ConfigManager(config_dir=self._tmp.name))
        self._pool = mock.patch.object(
            chronicle_widget, "run_pool_task", mock.MagicMock()
        )
        self._pool.start()

    def tearDown(self) -> None:
        self._pool.stop()
        self._tmp.cleanup()

    def test_la_cronica_pasa_las_instrucciones_al_worker(self) -> None:
        widget = chronicle_widget.ChronicleWidget(self.manager)
        try:
            widget._txt_instructions.setPlainText("céntrate en las paradas")
            widget._video = Path("peli.mp4")
            with mock.patch.object(widget, "_start") as start:
                widget._generate()
            self.assertEqual(
                start.call_args.args[0]._instructions, "céntrate en las paradas"
            )
        finally:
            widget.close()

    def test_la_cronica_arranca_con_el_ejemplo(self) -> None:
        widget = chronicle_widget.ChronicleWidget(self.manager)
        try:
            texto = widget._txt_instructions.toPlainText().lower()
            self.assertIn("7 metros", texto)
            self.assertIn("alineación", texto)
        finally:
            widget.close()

    def test_el_cuaderno_pasa_las_instrucciones_al_worker(self) -> None:
        widget = notes_widget.NotesWidget(self.manager)
        try:
            widget._video = Path("peli.mp4")
            widget._txt_instructions.setPlainText("presenta la alineación")
            with mock.patch.object(widget, "_start_worker") as start:
                widget._analyze()
            worker = start.call_args.args[0]
            self.assertEqual(worker._instructions, "presenta la alineación")
        finally:
            widget.close()

    def test_comprension_pasa_las_instrucciones_al_worker(self) -> None:
        widget = understanding_widget.UnderstandingWidget(self.manager)
        try:
            widget._video_path = Path("peli.mp4")
            widget._txt_instructions.setPlainText("destaca las paradas")
            # Solo transcripción: no exige motores de percepción (YOLO/CLIP/OCR).
            widget._objects.setChecked(False)
            widget._labels.setChecked(False)
            widget._ocr.setChecked(False)
            widget._transcribe.setChecked(True)
            with mock.patch.object(
                understanding_widget, "UnderstandingWorker"
            ) as worker_cls:
                widget._analyze()
            self.assertEqual(
                worker_cls.call_args.kwargs["instructions"], "destaca las paradas"
            )
        finally:
            widget.close()


if __name__ == "__main__":
    unittest.main()
