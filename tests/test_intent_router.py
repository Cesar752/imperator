"""Tests del router de intención y del orquestador de tareas.

Cubren las dos piezas de la Fase 2: ``services.intent_router`` decide a qué
tarea apunta un mensaje del chat y ``services.task_orchestrator`` lo resuelve
contra las tareas registradas. Además se comprueba que el ``ChatWidget`` emite
la señal ``routed`` sin abrir nada por su cuenta.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from services import intent_router as ir  # noqa: E402
from services.task_orchestrator import Dispatch, TaskOrchestrator  # noqa: E402


class RouteTest(unittest.TestCase):
    # ---- Sin señal: se queda en el chat -----------------------------

    def test_texto_vacio(self) -> None:
        self.assertFalse(ir.route("").is_routed)

    def test_conversacion_normal(self) -> None:
        intent = ir.route("Hola, ¿me explicas qué es una función recursiva?")
        self.assertFalse(intent.is_routed)
        self.assertEqual(intent.kind, ir.KIND_CHAT)

    def test_sustantivo_suelto_no_enruta(self) -> None:
        """«pdf» a secas puede ser una pregunta: no debe secuestrar el chat."""
        self.assertFalse(ir.route("pdf").is_routed)
        self.assertFalse(ir.route("imagen").is_routed)
        self.assertFalse(ir.route("¿qué es una imagen vectorial?").is_routed)

    def test_resumen_sin_documento_no_enruta(self) -> None:
        self.assertFalse(ir.route("hazme un resumen de lo que te he contado").is_routed)

    # ---- Documentos -------------------------------------------------

    def test_resume_este_pdf(self) -> None:
        intent = ir.route("resume este pdf, por favor")
        self.assertEqual(intent.kind, ir.KIND_DOCUMENT)
        self.assertEqual(intent.action, ir.ACTION_SUMMARY)
        self.assertGreater(intent.score, 0)

    def test_mayusculas_y_acentos_no_importan(self) -> None:
        self.assertEqual(
            ir.route("RESUME ESTE PDF").kind, ir.KIND_DOCUMENT
        )
        self.assertEqual(ir.route("vídeo").kind, ir.KIND_CHAT)

    def test_ocr(self) -> None:
        intent = ir.route("hazme ocr de este documento escaneado")
        self.assertEqual(intent.kind, ir.KIND_DOCUMENT)
        self.assertEqual(intent.action, ir.ACTION_OCR)

    def test_tablas(self) -> None:
        intent = ir.route("extrae las tablas de este documento pdf")
        self.assertEqual(intent.kind, ir.KIND_DOCUMENT)
        self.assertEqual(intent.action, ir.ACTION_TABLES)

    def test_ingles(self) -> None:
        intent = ir.route("summarize this document")
        self.assertEqual(intent.kind, ir.KIND_DOCUMENT)
        self.assertEqual(intent.action, ir.ACTION_SUMMARY)

    def test_euskera(self) -> None:
        intent = ir.route("laburpen dokumentu hau")
        self.assertEqual(intent.kind, ir.KIND_DOCUMENT)

    # ---- Imagen / vídeo / audio / ensamblador ----------------------

    def test_genera_una_imagen(self) -> None:
        intent = ir.route("genera una imagen de un gato con sombrero")
        self.assertEqual(intent.kind, ir.KIND_IMAGE)
        self.assertEqual(intent.action, ir.ACTION_GENERATE)

    def test_genera_un_video(self) -> None:
        intent = ir.route("genera un vídeo de un gato saltando")
        self.assertEqual(intent.kind, ir.KIND_VIDEO)
        self.assertEqual(intent.action, ir.ACTION_GENERATE)

    def test_comprension_de_video(self) -> None:
        intent = ir.route("analiza este vídeo y dime qué pasa")
        self.assertEqual(intent.kind, ir.KIND_UNDERSTANDING)
        self.assertEqual(intent.action, ir.ACTION_ANALYZE)

    def test_resumen_de_video(self) -> None:
        intent = ir.route("resume el vídeo")
        self.assertEqual(intent.kind, ir.KIND_UNDERSTANDING)
        self.assertEqual(intent.action, ir.ACTION_SUMMARY)

    def test_audio_transcripcion(self) -> None:
        intent = ir.route("transcribe el audio a texto")
        self.assertEqual(intent.kind, ir.KIND_AUDIO)
        self.assertEqual(intent.action, ir.ACTION_TRANSCRIBE)

    def test_audio_voz(self) -> None:
        intent = ir.route("ponle voz a este texto con narración")
        self.assertEqual(intent.kind, ir.KIND_AUDIO)
        self.assertEqual(intent.action, ir.ACTION_GENERATE)

    def test_ensamblador(self) -> None:
        intent = ir.route("monta un vídeo con estos clips")
        self.assertEqual(intent.kind, ir.KIND_ASSEMBLER)
        self.assertEqual(intent.action, ir.ACTION_ASSEMBLE)

    # ---- Propiedades de Intent -------------------------------------

    def test_confianza_acotada(self) -> None:
        intent = ir.route("resume este pdf")
        self.assertGreaterEqual(intent.confidence, 0.0)
        self.assertLessEqual(intent.confidence, 1.0)

    def test_matched_incluye_terminos(self) -> None:
        intent = ir.route("resume este pdf")
        self.assertIn("pdf", intent.matched)

    def test_describe(self) -> None:
        self.assertIn("chat", ir.route("hola").describe().lower())
        self.assertIn("Documentos", ir.route("resume este pdf").describe())

    def test_etiquetas(self) -> None:
        self.assertEqual(ir.TASK_LABELS[ir.KIND_UNDERSTANDING], "Comprensión")
        self.assertEqual(ir.ACTION_LABELS[ir.ACTION_SUMMARY], "resumen")


class OrchestratorTest(unittest.TestCase):
    def test_sin_senal_devuelve_none(self) -> None:
        orch = TaskOrchestrator(available=lambda: ("chat", "document"))
        self.assertIsNone(orch.dispatch("hola, ¿qué tal?"))

    def test_tarea_no_registrada_devuelve_none(self) -> None:
        orch = TaskOrchestrator(available=lambda: ("chat",))
        self.assertIsNone(orch.dispatch("resume este pdf"))

    def test_despacha_a_tarea_disponible(self) -> None:
        orch = TaskOrchestrator(available=lambda: ("chat", "document"))
        dispatch = orch.dispatch("resume este pdf")
        self.assertIsNotNone(dispatch)
        assert dispatch is not None
        self.assertEqual(dispatch.task_id, "document")
        self.assertEqual(dispatch.action, "summary")

    def test_sugerencia_legible(self) -> None:
        orch = TaskOrchestrator(available=lambda: ("document",))
        dispatch = orch.dispatch("resume este pdf")
        assert dispatch is not None
        texto = dispatch.suggestion()
        self.assertIn("Documentos", texto)
        self.assertIn("resumen", texto)

    def test_available_tasks_tolera_fallos(self) -> None:
        def boom():
            raise RuntimeError("no disponible")

        orch = TaskOrchestrator(available=boom)
        self.assertEqual(orch.available_tasks(), [])
        self.assertIsNone(orch.dispatch("resume este pdf"))

    def test_sin_callable_no_hay_tareas(self) -> None:
        orch = TaskOrchestrator()
        self.assertEqual(orch.available_tasks(), [])

    def test_plan_explica_la_decision(self) -> None:
        orch = TaskOrchestrator(available=lambda: ("chat", "document"))
        self.assertIn("chat", orch.plan("hola").lower())
        orch_sin = TaskOrchestrator(available=lambda: ("chat",))
        self.assertIn("no está registrada", orch_sin.plan("resume este pdf"))


class _FakeOrchestrator:
    """Orquestador de mentira que devuelve siempre el mismo dispatch."""

    def __init__(self, dispatch) -> None:
        self._dispatch = dispatch
        self.seen: list[str] = []

    def dispatch(self, text: str):
        self.seen.append(text)
        return self._dispatch


class ChatRoutingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from PyQt6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def _widget(self, orchestrator=None):
        from ui import chat_widget as cw
        from ui.chat_widget import ChatWidget

        with mock.patch.object(cw, "run_pool_task"):
            widget = ChatWidget(engine=self._FakeEngine(), orchestrator=orchestrator)
        return widget

    class _FakeEngine:
        def list_model_details(self) -> list:
            return []

    def test_emite_routed_al_detectar_otra_tarea(self) -> None:
        from ui import chat_widget as cw

        dispatch = Dispatch(task_id="document", action="summary", confidence=0.6)
        orch = _FakeOrchestrator(dispatch)
        widget = self._widget(orch)
        try:
            recibidos: list = []
            widget.routed.connect(recibidos.append)
            widget._input.setText("resume este pdf")
            with mock.patch.object(cw.ChatWorker, "start"):
                widget._send_message()
            self.assertEqual(recibidos, [dispatch])
            self.assertEqual(orch.seen, ["resume este pdf"])
        finally:
            widget.close()

    def test_no_emite_si_no_hay_dispatch(self) -> None:
        from ui import chat_widget as cw

        widget = self._widget(_FakeOrchestrator(None))
        try:
            recibidos: list = []
            widget.routed.connect(recibidos.append)
            widget._input.setText("hola")
            with mock.patch.object(cw.ChatWorker, "start"):
                widget._send_message()
            self.assertEqual(recibidos, [])
        finally:
            widget.close()

    def test_sin_orquestador_no_emite_ni_falla(self) -> None:
        from ui import chat_widget as cw

        widget = self._widget(None)
        try:
            recibidos: list = []
            widget.routed.connect(recibidos.append)
            widget._input.setText("resume este pdf")
            with mock.patch.object(cw.ChatWorker, "start"):
                widget._send_message()
            self.assertEqual(recibidos, [])
        finally:
            widget.close()


if __name__ == "__main__":
    unittest.main()
