"""Tests de la comprensión de vídeo.

Cubre las tres piezas de la Fase 4:
- ``services.video_understanding``: muestreo, prompts y montaje del informe.
- ``core.frame_extractor``: construcción del comando de ffmpeg y errores.
- ``ui.understanding_widget.UnderstandingWorker``: pipeline completo con dobles
  de los motores (no se llama a YOLO/CLIP/OCR/Whisper/Ollama de verdad).
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from core import frame_extractor as fe  # noqa: E402
from engines.base_engine import CANCEL_MESSAGE, EngineError, Perception  # noqa: E402
from services import video_understanding as vu  # noqa: E402


class SamplingTest(unittest.TestCase):
    def test_instantes_en_el_centro_de_cada_intervalo(self) -> None:
        # 12 s cada 5 s → 2.5 y 7.5 (12.5 se sale).
        self.assertEqual(vu.sample_timestamps(12, 5, 24), [2.5, 7.5])

    def test_video_mas_corto_que_un_paso(self) -> None:
        self.assertEqual(vu.sample_timestamps(3, 5, 24), [1.5])

    def test_duracion_invalida(self) -> None:
        self.assertEqual(vu.sample_timestamps(0, 5, 24), [])
        self.assertEqual(vu.sample_timestamps("x", 5, 24), [])
        self.assertEqual(vu.sample_timestamps(-4, 5, 24), [])

    def test_respeta_el_maximo_de_fotogramas(self) -> None:
        instantes = vu.sample_timestamps(100, 1, 24)
        self.assertEqual(len(instantes), 24)
        self.assertEqual(instantes[0], 0.5)
        self.assertEqual(instantes[-1], 99.5)

    def test_un_solo_fotograma_toma_el_centro(self) -> None:
        self.assertEqual(vu.sample_timestamps(100, 1, 1), [50.5])

    def test_every_se_acota(self) -> None:
        self.assertEqual(vu.clamp_every(0.1), vu.MIN_EVERY_S)
        self.assertEqual(vu.clamp_every("malo"), vu.DEFAULT_EVERY_S)

    def test_max_frames_se_acota(self) -> None:
        self.assertEqual(vu.clamp_max_frames(0), 1)
        self.assertEqual(vu.clamp_max_frames(9999), vu.MAX_MAX_FRAMES)
        self.assertEqual(vu.clamp_max_frames("x"), vu.DEFAULT_MAX_FRAMES)

    def test_formato_de_tiempo(self) -> None:
        self.assertEqual(vu.format_timestamp(0), "00:00")
        self.assertEqual(vu.format_timestamp(95.2), "01:35")
        self.assertEqual(vu.format_timestamp(3725), "1:02:05")

    def test_srt_a_texto(self) -> None:
        srt = "1\n00:00:00,000 --> 00:00:02,000\nHola\n\n2\n00:00:02,000 --> 00:00:04,000\nmundo\n"
        self.assertEqual(vu.srt_to_text(srt), "Hola\nmundo")

    def test_srt_vacio(self) -> None:
        self.assertEqual(vu.srt_to_text(""), "")


class ClipLabelsTest(unittest.TestCase):
    """El vocabulario de CLIP que se sugiere en la pestaña no es genérico."""

    def test_el_vocabulario_de_deporte_es_del_contenido(self) -> None:
        texto = vu.clip_labels_text()
        for clave in ("balón", "portería", "grada", "lanzamiento", "celebración"):
            self.assertIn(clave, texto)
        # Nada de términos genéricos que devolvían ruido («calle, coche, animal»).
        self.assertNotIn("coche", texto)
        self.assertNotIn("animal", texto)


class VideoFilterTest(unittest.TestCase):
    """El filtro de los diálogos debe cubrir lo que ffmpeg lee de verdad."""

    def test_el_filtro_cubre_las_extensiones(self) -> None:
        for ext in vu.VIDEO_EXTENSIONS:
            self.assertEqual(ext, ext.lower())
            self.assertIn(f"*{ext}", vu.VIDEO_FILTER)
        # Los formatos de cámara son los que más se olvidan (y los que más se graban).
        self.assertIn("*.mts", vu.VIDEO_FILTER)
        self.assertIn("*.m2ts", vu.VIDEO_FILTER)
        self.assertIn("Todos", vu.VIDEO_FILTER_ALL)


class BuildSrtTest(unittest.TestCase):
    """El *escritor* de SRT (rótulos y crónica ya saben sus propios tiempos)."""

    def test_marca_de_tiempo(self) -> None:
        self.assertEqual(vu.srt_timestamp(0), "00:00:00,000")
        self.assertEqual(vu.srt_timestamp(95.2), "00:01:35,200")
        self.assertEqual(vu.srt_timestamp(-5), "00:00:00,000")
        self.assertEqual(vu.srt_timestamp(3661.5), "01:01:01,500")

    def test_build_srt(self) -> None:
        texto = vu.build_srt([(0.0, 2.5, "Hola"), (12.0, 14.5, "Gol  de  Álex")])
        self.assertEqual(
            texto,
            "1\n00:00:00,000 --> 00:00:02,500\nHola\n\n"
            "2\n00:00:12,000 --> 00:00:14,500\nGol de Álex\n",
        )

    def test_los_textos_vacios_no_ocupan_bloque(self) -> None:
        texto = vu.build_srt([(0.0, 1.0, "  "), (1.0, 2.0, "útil")])
        self.assertEqual(texto.count("-->"), 1)
        self.assertTrue(texto.startswith("1\n"))

    def test_sin_entradas(self) -> None:
        self.assertEqual(vu.build_srt([]), "")

    def test_ida_y_vuelta_con_srt_to_text(self) -> None:
        texto = vu.build_srt([(0.0, 2.0, "Primera"), (2.0, 4.0, "Segunda")])
        self.assertEqual(vu.srt_to_text(texto), "Primera\nSegunda")


class PlanTest(unittest.TestCase):
    def test_wants_perception(self) -> None:
        self.assertTrue(vu.UnderstandingPlan().wants_perception)
        self.assertFalse(
            vu.UnderstandingPlan(
                do_objects=False, do_labels=False, do_ocr=False
            ).wants_perception
        )

    def test_normalized_acota(self) -> None:
        plan = vu.UnderstandingPlan(every_s=0.01, max_frames=0).normalized()
        self.assertEqual(plan.every_s, vu.MIN_EVERY_S)
        self.assertEqual(plan.max_frames, 1)

    def test_describe(self) -> None:
        plan = vu.UnderstandingPlan(
            every_s=3, max_frames=10, do_labels=True, do_ocr=False
        ).normalized()
        texto = plan.describe()
        self.assertIn("cada 3 s", texto)
        self.assertIn("máx. 10", texto)
        self.assertIn("etiquetas", texto)
        self.assertIn("transcripción", texto)


class AggregatePromptTest(unittest.TestCase):
    def test_prompt_con_idioma(self) -> None:
        plan = vu.UnderstandingPlan(language="euskera")
        prompt = vu.aggregate_prompt(plan, 5, True)
        self.assertIn("euskera", prompt)
        self.assertNotIn("No hay transcripción", prompt)

    def test_prompt_avisa_sin_transcripcion(self) -> None:
        prompt = vu.aggregate_prompt(vu.UnderstandingPlan(), 0, False)
        self.assertIn("No hay transcripción de audio", prompt)
        self.assertIn("No hay anotaciones", prompt)

    def test_prompt_sin_idioma_pide_el_del_video(self) -> None:
        prompt = vu.aggregate_prompt(vu.UnderstandingPlan(language=None), 1, True)
        self.assertIn("mismo idioma", prompt)

    def test_prompt_con_marcador_real(self) -> None:
        # El marcador del OCR es dato duro: evita «un gol por fotograma».
        prompt = vu.aggregate_prompt(
            vu.UnderstandingPlan(), 3, True, "", "[00:26] 0-1\n[00:53] 0-2"
        )
        self.assertIn("MARCADOR REAL", prompt)
        self.assertIn("0-1", prompt)
        self.assertIn("No cuentes un gol por fotograma", prompt)

    def test_prompt_con_contexto_del_partido(self) -> None:
        # El contexto (fecha, lugar, equipos, alineación) va como verdad al resumen.
        prompt = vu.aggregate_prompt(
            vu.UnderstandingPlan(),
            3,
            True,
            "Fecha: 9/10/2026 20:30\nEquipos: A vs B",
        )
        self.assertIn(vu.CONTEXT_LABEL, prompt)
        self.assertIn("Fecha: 9/10/2026 20:30", prompt)


class BuildReportTest(unittest.TestCase):
    def test_informe_completo(self) -> None:
        frames = [vu.FrameReport(timestamp=2.5, summary="Objetos: person")]
        informe = vu.build_report(
            vu.UnderstandingPlan(), frames, "hola mundo", "resumen del llm"
        )
        self.assertIn(vu.SUMMARY_HEADING, informe)
        self.assertIn("resumen del llm", informe)
        self.assertIn("t=00:02 → Objetos: person", informe)
        self.assertIn("hola mundo", informe)
        self.assertIn("Plan:", informe)

    def test_avisos_cuando_falta_todo(self) -> None:
        informe = vu.build_report(vu.UnderstandingPlan(), [], "", "")
        self.assertIn(vu.NO_PERCEPTION_NOTICE, informe)
        self.assertIn(vu.NO_TRANSCRIPT_NOTICE, informe)
        self.assertIn(vu.NO_SUMMARY_NOTICE, informe)

    def test_fotograma_sin_hallazgos(self) -> None:
        informe = vu.build_report(
            vu.UnderstandingPlan(), [vu.FrameReport(timestamp=1.5)], "", ""
        )
        self.assertIn("(sin hallazgos)", informe)

    def test_el_informe_incluye_el_contexto(self) -> None:
        # El texto del informe lleva los datos del partido: la Crónica los recupera.
        informe = vu.build_report(
            vu.UnderstandingPlan(), [], "", "", (), context="Fecha: hoy"
        )
        self.assertIn(vu.CONTEXT_LABEL, informe)
        self.assertIn("Fecha: hoy", informe)


class _FakeFfmpegModule:
    @staticmethod
    def get_ffmpeg_exe() -> str:
        return "ffmpeg"


class FrameExtractorTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.video = Path(self._tmp.name) / "v.mp4"
        self.video.write_bytes(b"video")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_extract_frame_construye_el_comando(self) -> None:
        salida = Path(self._tmp.name) / "f.png"
        comandos: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            comandos.append(list(cmd))
            salida.write_bytes(b"png")
            return subprocess.CompletedProcess(cmd, 0)

        with mock.patch.object(fe, "imageio_ffmpeg", _FakeFfmpegModule()), mock.patch.object(
            fe.subprocess, "run", side_effect=fake_run
        ):
            resultado = fe.FrameExtractor().extract_frame(self.video, 3.5, salida)

        self.assertEqual(resultado, salida)
        comando = comandos[0]
        self.assertIn("-ss", comando)
        self.assertIn("3.500", comando)
        self.assertIn("-frames:v", comando)

    def test_extract_frame_sin_salida_es_error(self) -> None:
        salida = Path(self._tmp.name) / "no.png"

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 0)

        with mock.patch.object(fe, "imageio_ffmpeg", _FakeFfmpegModule()), mock.patch.object(
            fe.subprocess, "run", side_effect=fake_run
        ):
            with self.assertRaises(EngineError):
                fe.FrameExtractor().extract_frame(self.video, 1.0, salida)

    def test_timeout_es_error(self) -> None:
        def fake_run(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd, 60)

        with mock.patch.object(fe, "imageio_ffmpeg", _FakeFfmpegModule()), mock.patch.object(
            fe.subprocess, "run", side_effect=fake_run
        ):
            with self.assertRaises(EngineError) as ctx:
                fe.FrameExtractor().extract_frame(self.video, 1.0, "x.png")
        self.assertIn("no terminó", str(ctx.exception))

    def test_sin_imageio_no_hay_ffmpeg(self) -> None:
        with mock.patch.object(fe, "imageio_ffmpeg", None):
            with self.assertRaises(EngineError):
                fe.FrameExtractor().extract_frame(self.video, 1.0, "x.png")

    def test_video_inexistente(self) -> None:
        with self.assertRaises(EngineError):
            fe.FrameExtractor().extract_frame("C:/no/existe.mp4", 1.0, "x.png")

    def test_duration_delega_en_el_ensamblador(self) -> None:
        class _FakeAssembler:
            def probe(self, _path):
                return {"duration": 10.5, "width": None, "height": None}

        with mock.patch.object(fe, "VideoAssembler", _FakeAssembler):
            self.assertEqual(fe.FrameExtractor().duration(self.video), 10.5)

    def test_duration_sin_dato(self) -> None:
        class _FakeAssembler:
            def probe(self, _path):
                return {"duration": None}

        with mock.patch.object(fe, "VideoAssembler", _FakeAssembler):
            self.assertIsNone(fe.FrameExtractor().duration(self.video))


# ---- Dobles de los motores -----------------------------------------


class _FakeExtractor:
    def __init__(self, duration=12.0) -> None:
        self._duration = duration
        self.frames: list[float] = []
        self.audio_calls = 0

    def duration(self, _video):
        return self._duration

    def extract_frame(self, _video, timestamp, output_path):
        destino = Path(output_path)
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_bytes(b"png")
        self.frames.append(timestamp)
        return destino

    def cached_frame(self, _video, timestamp, suffix=".png"):
        # Doble de la caché: escribe en un temporal y registra el instante.
        destino = Path(tempfile.mkdtemp()) / f"frame_{timestamp}{suffix}"
        destino.write_bytes(b"png")
        self.frames.append(timestamp)
        return destino

    def extract_audio(self, _video, output_path):
        destino = Path(output_path)
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_bytes(b"wav")
        self.audio_calls += 1
        return destino


class _FakePerceptionEngine:
    id = "yolo"

    def __init__(self, etiqueta="person") -> None:
        self._etiqueta = etiqueta
        self.llamadas = 0

    def is_available(self) -> bool:
        return True

    def perceive(self, _path, prompt=""):
        self.llamadas += 1
        return Perception(objects=({"label": self._etiqueta},))


class _FakeTranscriber:
    id = "whisper"

    def __init__(self, srt="1\n00:00:00,000 --> 00:00:02,000\nhola mundo\n") -> None:
        self._srt = srt

    def is_available(self) -> bool:
        return True

    def transcribe(self, audio_path, language=None, **_kwargs):
        destino = Path(audio_path).with_suffix(".srt")
        destino.write_text(self._srt, encoding="utf-8")
        return destino


class _FakeChat:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def is_available(self) -> bool:
        return True

    def chat(self, messages, model=None):
        self.prompts.append(messages[-1]["content"])
        return "resumen del modelo"


class UnderstandingWorkerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from PyQt6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def _worker(self, **kwargs):
        from ui.understanding_widget import UnderstandingWorker

        parametros = {
            "extractor": _FakeExtractor(),
            "objects_engine": _FakePerceptionEngine(),
            "transcription_engine": _FakeTranscriber(),
            "chat_engine": _FakeChat(),
        }
        parametros.update(kwargs)
        plan = parametros.pop("plan", vu.UnderstandingPlan())
        return UnderstandingWorker("C:/fake/video.mp4", plan, **parametros)

    def test_informe_completo(self) -> None:
        # Plan explícito (un fotograma cada 5 s): el de por defecto apunta a máxima
        # calidad (cada 2 s) y aquí se comprueba el formato del informe.
        worker = self._worker(plan=vu.UnderstandingPlan(every_s=5.0, max_frames=24))
        informe = worker.work()
        self.assertIn("resumen del modelo", informe)
        self.assertIn("t=00:02", informe)
        self.assertIn("person", informe)
        self.assertIn("hola mundo", informe)

    def test_el_chat_recibe_las_anotaciones_y_la_transcripcion(self) -> None:
        chat = _FakeChat()
        worker = self._worker(chat_engine=chat)
        worker.work()
        self.assertEqual(len(chat.prompts), 1)
        self.assertIn("TRANSCRIPCIÓN DEL AUDIO", chat.prompts[0])
        self.assertIn("hola mundo", chat.prompts[0])
        self.assertIn("ANOTACIONES DE FOTOGRAMAS", chat.prompts[0])

    def test_el_prompt_del_resumen_se_acota(self) -> None:
        """Una transcripción enorme no debe llenar la ventana (se recorta)."""

        class _TranscriberLargo:
            def is_available(self) -> bool:
                return True

            def transcribe(self, audio_path, language=None, **_kwargs):
                destino = Path(audio_path).with_suffix(".srt")
                destino.write_text("linea de audio " * 5000, encoding="utf-8")
                return destino

        chat = _FakeChat()
        worker = self._worker(
            chat_engine=chat, transcription_engine=_TranscriberLargo()
        )
        worker.work()
        mensaje = chat.prompts[0]
        self.assertLess(len(mensaje), 20000)
        self.assertIn("[…]", mensaje)

    def test_duracion_ilegible(self) -> None:
        worker = self._worker(extractor=_FakeExtractor(duration=None))
        with self.assertRaises(EngineError):
            worker.work()

    def test_sin_motores_avisa_y_no_rompe(self) -> None:
        worker = self._worker(
            objects_engine=None,
            transcription_engine=None,
            chat_engine=None,
        )
        informe = worker.work()
        self.assertIn(vu.NO_PERCEPTION_NOTICE, informe)
        self.assertIn(vu.NO_TRANSCRIPT_NOTICE, informe)
        self.assertIn(vu.NO_SUMMARY_NOTICE, informe)

    def test_solo_transcripcion(self) -> None:
        plan = vu.UnderstandingPlan(
            do_objects=False, do_labels=False, do_ocr=False, transcribe=True
        )
        worker = self._worker(plan=plan)
        informe = worker.work()
        self.assertIn("hola mundo", informe)
        self.assertIn(vu.NO_PERCEPTION_NOTICE, informe)

    def test_no_transcribe_si_se_desactiva(self) -> None:
        plan = vu.UnderstandingPlan(transcribe=False)
        extractor = _FakeExtractor()
        worker = self._worker(plan=plan, extractor=extractor)
        worker.work()
        self.assertEqual(extractor.audio_calls, 0)

    def test_analiza_un_fotograma_por_instante(self) -> None:
        extractor = _FakeExtractor(duration=12.0)
        motor = _FakePerceptionEngine()
        worker = self._worker(
            extractor=extractor,
            objects_engine=motor,
            plan=vu.UnderstandingPlan(every_s=5.0, max_frames=24),
        )
        worker.work()
        self.assertEqual(extractor.frames, [2.5, 7.5])
        self.assertEqual(motor.llamadas, 2)

    def test_cancelacion(self) -> None:
        worker = self._worker()
        worker.isInterruptionRequested = lambda: True
        with self.assertRaises(EngineError) as ctx:
            worker.work()
        self.assertEqual(str(ctx.exception), CANCEL_MESSAGE)

    def test_usa_el_planificador_de_gpu(self) -> None:
        from core.gpu_scheduler import GpuScheduler, GpuSnapshot, SchedulerConfig
        from ui import understanding_widget as uw

        class _SinHueco:
            def snapshot(self):
                return GpuSnapshot(available=True, total_mb=1000, used_mb=1000)

        scheduler = GpuScheduler(
            probe=_SinHueco(), config=SchedulerConfig(vram_headroom_mb=1)
        )
        motor = _FakePerceptionEngine()
        worker = self._worker(objects_engine=motor, scheduler=scheduler)
        # Sin espera real: el fotograma se rinde en el acto (en la app espera
        # PERCEPTION_WAIT_S segundos a que la GPU tenga hueco).
        with mock.patch.object(uw, "PERCEPTION_WAIT_S", 0.0):
            informe = worker.work()
        self.assertEqual(motor.llamadas, 0, "sin hueco no se llama al motor")
        self.assertIn("(sin hallazgos)", informe)

    def test_planificador_libera_la_reserva(self) -> None:
        from core.gpu_scheduler import GpuScheduler, GpuSnapshot

        class _ConHueco:
            def snapshot(self):
                return GpuSnapshot(available=True, total_mb=16000, used_mb=1000)

        scheduler = GpuScheduler(probe=_ConHueco())
        worker = self._worker(scheduler=scheduler)
        worker.work()
        self.assertEqual(scheduler.active_jobs(), [])


class StructuredReportTest(unittest.TestCase):
    """La percepción estructurada que Comprensión deja para el Cuaderno."""

    def test_el_contexto_se_guarda_en_el_json(self):
        data = vu.understanding_to_dict(video_name="x", context="Fecha: hoy")
        self.assertEqual(data["contexto"], "Fecha: hoy")

    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = vu.understanding_to_dict(
                video_path="C:/v/peli.mts",
                duration_s=120.0,
                plan=vu.UnderstandingPlan(do_objects=True, do_labels=True),
                frames=[
                    vu.FrameReport(2.5, "JUGANDO: balón en juego"),
                    vu.FrameReport(60.0, "GOL del 7 azul"),
                ],
                transcript="gol, gol",
                summary="Partido igualado",
                notices=("Aviso: sin OCR",),
                context="Equipos: A vs B",
            )
            path = vu.save_understanding(Path(tmp) / vu.UNDERSTANDING_FILENAME, data)
            leido = vu.load_understanding(path)
        self.assertEqual(leido["video_name"], "peli.mts")
        self.assertEqual(leido["contexto"], "Equipos: A vs B")
        self.assertEqual(
            vu.understanding_frames(leido),
            [(2.5, "JUGANDO: balón en juego"), (60.0, "GOL del 7 azul")],
        )
        self.assertEqual(leido["transcripcion"], "gol, gol")

    def test_fichero_invalido(self):
        with tempfile.TemporaryDirectory() as tmp:
            malo = Path(tmp) / "x.json"
            malo.write_text('{"otra": 1}', encoding="utf-8")
            with self.assertRaises(vu.UnderstandingError):
                vu.load_understanding(malo)

    def test_anotacion_mas_cercana(self):
        percepcion = [(2.0, "A"), (10.0, "B")]
        self.assertEqual(vu.nearest_annotation(2.5, percepcion), "A")
        self.assertEqual(vu.nearest_annotation(9.0, percepcion), "B")
        # Fuera de la tolerancia: hueco, no la anotación de otro momento.
        self.assertEqual(vu.nearest_annotation(40.0, percepcion), "")
        self.assertEqual(vu.nearest_annotation(2.0, []), "")

    def test_informe_vacio_es_tolerante(self):
        self.assertEqual(vu.understanding_frames({}), [])
        self.assertEqual(
            vu.understanding_frames({"fotogramas": [1, {}, "x"]}), [(0.0, "")]
        )


class AggregatePromptDisciplinaTest(unittest.TestCase):
    """El resumen de Comprensión lleva las reglas de la disciplina."""

    def test_con_disciplina_incluye_reglas(self):
        prompt = vu.aggregate_prompt(vu.UnderstandingPlan(sport="balonmano"), 3, True)
        self.assertIn("Balonmano", prompt)
        self.assertIn("7 metros", prompt)

    def test_sin_disciplina_no_incluye_reglas(self):
        prompt = vu.aggregate_prompt(vu.UnderstandingPlan(), 3, True)
        self.assertNotIn("7 metros", prompt)

    def test_normalized_conserva_la_disciplina(self):
        plan = vu.UnderstandingPlan(sport="balonmano").normalized()
        self.assertEqual(plan.sport, "balonmano")


if __name__ == "__main__":
    unittest.main()
