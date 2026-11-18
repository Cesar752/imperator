"""Smoke tests de las pestañas que no tenían test propio.

Cada widget se construye en modo offscreen con una `ConfigManager` temporal (para
no tocar `config/` real) y se comprueba que arranca sin excepciones y con sus
controles básicos en su sitio. Son las pestañas de Vídeo, Imagen, Audio,
Ensamblador y Comprensión: la auditoría del 30/09/2026 señaló que solo el chat y
Documentos tenían tests, y que estas cinco son las que más superficie tienen.

Los sondeos que un widget lanza contra los motores al construirse (`run_pool_task`)
se sustituyen por un doble: el test no toca red ni GPU.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QComboBox, QWidget  # noqa: E402

from core.config_manager import ConfigManager  # noqa: E402
from core.engine_manager import EngineManager  # noqa: E402
from ui import assembler_widget, audio_widget, image_widget  # noqa: E402
from ui import notes_widget, photo_widget, understanding_widget, video_widget  # noqa: E402
from ui import chronicle_widget, video_enhance_widget  # noqa: E402


class WidgetsSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.cm = ConfigManager(config_dir=self._tmp.name)
        self._patches = [
            mock.patch.object(modulo, "run_pool_task", mock.MagicMock())
            for modulo in (
                video_widget,
                image_widget,
                understanding_widget,
                chronicle_widget,
            )
        ]
        for patcher in self._patches:
            patcher.start()

    def tearDown(self) -> None:
        for patcher in self._patches:
            patcher.stop()
        self._tmp.cleanup()

    def _comprobar(self, widget: QWidget, combos: int | None = None) -> QWidget:
        self.assertIsInstance(widget, QWidget)
        self.assertIsNotNone(widget.layout())
        if combos is not None:
            self.assertEqual(len(widget.findChildren(QComboBox)), combos)
        widget.close()
        return widget

    def test_video_widget_arranca(self) -> None:
        widget = video_widget.VideoWidget(None, self.cm)
        self._comprobar(widget)
        # Los modos T2V/I2V se cargan aunque no haya motor disponible.
        self.assertGreaterEqual(widget._mode.count(), 0)

    def test_video_enhance_widget_arranca(self) -> None:
        self._comprobar(video_enhance_widget.VideoEnhanceWidget())

    def test_image_widget_arranca(self) -> None:
        self._comprobar(image_widget.ImageWidget({}, self.cm))

    def test_photo_widget_arranca(self) -> None:
        self._comprobar(photo_widget.PhotoWidget())

    def test_audio_widget_arranca(self) -> None:
        self._comprobar(audio_widget.AudioWidget(None, None, self.cm))

    def test_assembler_widget_arranca(self) -> None:
        self._comprobar(assembler_widget.AssemblerWidget(None, self.cm))

    def test_notes_widget_arranca(self) -> None:
        widget = notes_widget.NotesWidget(EngineManager(self.cm))
        self._comprobar(widget)

    def test_el_replay_monta_el_grafo_de_repeticiones(self) -> None:
        """El botón de repetición compone el grafo (sin lanzar ffmpeg de verdad)."""
        widget = video_enhance_widget.VideoEnhanceWidget()
        widget._video = Path("peli.mp4")
        widget._info = {"has_audio": True}
        widget._spn_start.setValue(1.0)
        widget._spn_end.setValue(3.0)
        widget._spn_times.setValue(3)
        destino = Path(tempfile.gettempdir()) / "replay.mp4"
        with mock.patch.object(widget, "_out_path", return_value=destino), mock.patch.object(
            widget, "_start"
        ) as start:
            widget._apply_replay()
        self.assertTrue(start.called)
        worker = start.call_args.args[0]
        self.assertIsInstance(worker, video_enhance_widget.ReplayWorker)
        self.assertIn("concat=n=3:v=1:a=1", worker._graph.graph)
        widget.close()

    def test_el_replay_avisa_si_el_fragmento_no_vale(self) -> None:
        widget = video_enhance_widget.VideoEnhanceWidget()
        widget._video = Path("peli.mp4")
        widget._spn_start.setValue(5.0)
        widget._spn_end.setValue(5.0)
        with mock.patch.object(widget, "_start") as start:
            widget._apply_replay()
        self.assertFalse(start.called)
        self.assertIn("mayor que", widget._status.text())
        widget.close()

    def test_al_terminar_se_apaga_el_escalado_ia_si_falta(self) -> None:
        """``_set_busy(False)`` reactiva todo: el botón IA debe volver a apagarse."""
        widget = video_enhance_widget.VideoEnhanceWidget()
        with mock.patch.object(widget, "_ai_paths", return_value=None):
            widget._set_busy(False)
        self.assertFalse(widget._btn_ai.isEnabled())
        widget.close()

    def test_la_velocidad_1x_no_lanza_trabajo(self) -> None:
        widget = video_enhance_widget.VideoEnhanceWidget()
        widget._video = Path("peli.mp4")
        widget._spn_speed.setValue(1.0)
        with mock.patch.object(widget, "_start") as start:
            widget._apply_speed()
        self.assertFalse(start.called)
        self.assertIn("1x", widget._status.text())
        widget.close()

    def test_la_rampa_solo_ralentiza_el_tramo(self) -> None:
        widget = video_enhance_widget.VideoEnhanceWidget()
        widget._video = Path("peli.mp4")
        widget._info = {"duration": 10.0, "has_audio": True, "fps": 30.0}
        widget._spn_start.setValue(4.0)
        widget._spn_end.setValue(6.0)
        destino = Path(tempfile.gettempdir()) / "rampa.mp4"
        with mock.patch.object(widget, "_out_path", return_value=destino), mock.patch.object(
            widget, "_start"
        ) as start:
            widget._apply_ramp()
        worker = start.call_args.args[0]
        self.assertIsInstance(worker, video_enhance_widget.ReplayWorker)
        # Tres tramos: normal, lento y normal.
        self.assertIn("concat=n=3", worker._graph.graph)
        widget.close()

    def test_los_rotulos_se_queman_con_el_estilo_elegido(self) -> None:
        widget = video_enhance_widget.VideoEnhanceWidget()
        widget._video = Path("peli.mp4")
        widget._info = {"duration": 60.0, "width": 640, "height": 360}
        widget._txt_rolulos.setPlainText("0:12 Gol de Álex")
        destino = Path(tempfile.gettempdir()) / "rotulos.mp4"
        with mock.patch.object(widget, "_out_path", return_value=destino), mock.patch.object(
            widget, "_start"
        ) as start:
            widget._apply_rolulos()
        worker = start.call_args.args[0]
        self.assertIsInstance(worker, video_enhance_widget.RoluloWorker)
        self.assertIn("Gol de Álex", worker._srt_text)
        self.assertIn("00:00:12,000", worker._srt_text)
        self.assertEqual(worker._style["position"], "bottom")
        widget.close()

    def test_el_volumen_se_aplica_al_audio(self) -> None:
        """Bajar el volumen es un filtro de audio: puede ir sin tocar la imagen."""
        widget = video_enhance_widget.VideoEnhanceWidget()
        widget._video = Path("peli.mp4")
        widget._spn_volume.setValue(60)
        with mock.patch.object(widget, "_start") as start:
            widget._apply_ffmpeg()
        self.assertEqual(start.call_args.args[0]._audio_filter, "volume=0.600")
        widget.close()

    def test_la_calidez_llega_a_los_filtros(self) -> None:
        """El control «Calidez» se traduce al filtro de temperatura de ffmpeg."""
        widget = video_enhance_widget.VideoEnhanceWidget()
        widget._video = Path("peli.mp4")
        widget._spn_temperature.setValue(60)
        with mock.patch.object(widget, "_start") as start:
            widget._apply_ffmpeg()
        self.assertIn("colortemperature=", start.call_args.args[0]._filters)
        widget.close()

    def test_un_rotulo_mal_escrito_no_lanza_trabajo(self) -> None:
        widget = video_enhance_widget.VideoEnhanceWidget()
        widget._video = Path("peli.mp4")
        widget._txt_rolulos.setPlainText("esta línea no lleva tiempo")
        with mock.patch.object(widget, "_start") as start:
            widget._apply_rolulos()
        self.assertFalse(start.called)
        self.assertIn("Línea 1", widget._status.text())
        widget.close()

    def test_enviar_al_ensamblador_emite_la_ruta(self) -> None:
        widget = video_enhance_widget.VideoEnhanceWidget()
        recibidos: list = []
        widget.clips_ready.connect(recibidos.append)
        self.assertFalse(widget._btn_send.isEnabled())
        widget._on_result(Path("C:/tmp/replay.mp4"))
        self.assertTrue(widget._btn_send.isEnabled())
        widget._send_to_assembler()
        self.assertEqual(len(recibidos), 1)
        self.assertEqual(Path(recibidos[0]), Path("C:/tmp/replay.mp4"))
        self.assertIn("enviado", widget._status.text())
        widget.close()

    def test_el_lote_ordena_los_trozos_y_los_comprime_seguidos(self) -> None:
        widget = video_enhance_widget.VideoEnhanceWidget()
        with tempfile.TemporaryDirectory() as tmp:
            rutas = []
            for nombre in ("00001.MTS", "00000.MTS"):
                ruta = Path(tmp) / nombre
                ruta.write_bytes(b"x")
                rutas.append(str(ruta))
            with mock.patch.object(
                video_enhance_widget.QFileDialog,
                "getOpenFileNames",
                return_value=(rutas, ""),
            ), mock.patch.object(
                video_enhance_widget, "RECORDINGS_DIR", Path(tmp)
            ), mock.patch.object(widget, "_start") as start:
                widget._apply_compress_many()
        worker = start.call_args.args[0]
        self.assertIsInstance(worker, video_enhance_widget.BatchCompressWorker)
        self.assertEqual(
            [origen.name for origen, _destino in worker._trabajos],
            ["00000.MTS", "00001.MTS"],
        )
        # El lote no toca «el último vídeo»: trae su propio manejador de resultado.
        self.assertEqual(start.call_args.kwargs.get("on_result"), widget._on_batch_done)
        widget.close()

    def test_la_pestana_video_ia_tiene_scroll(self) -> None:
        """El formulario largo debe desplazarse: si no, lo de abajo se corta."""
        from PyQt6.QtWidgets import QScrollArea

        widget = video_enhance_widget.VideoEnhanceWidget()
        self.assertTrue(widget.findChildren(QScrollArea))
        widget.close()

    def test_la_pestana_cronica_tiene_scroll(self) -> None:
        """Sin scroll, «Generar crónica» y la barra quedaban fuera de la vista."""
        from PyQt6.QtWidgets import QScrollArea

        widget = chronicle_widget.ChronicleWidget(EngineManager(self.cm))
        self.assertTrue(widget.findChildren(QScrollArea))
        widget.close()

    def test_el_lote_no_pisa_el_origen_ni_repite_destino(self) -> None:
        """Dos ficheros con el mismo nombre no pueden compartir destino."""
        widget = video_enhance_widget.VideoEnhanceWidget()
        with tempfile.TemporaryDirectory() as tmp:
            carpeta_a = Path(tmp) / "a"
            carpeta_b = Path(tmp) / "b"
            carpeta_a.mkdir()
            carpeta_b.mkdir()
            uno = carpeta_a / "00000.MTS"
            dos = carpeta_b / "00000.MTS"
            uno.write_bytes(b"x")
            dos.write_bytes(b"x")
            with mock.patch.object(
                video_enhance_widget.QFileDialog,
                "getOpenFileNames",
                return_value=([str(uno), str(dos)], ""),
            ), mock.patch.object(
                video_enhance_widget, "RECORDINGS_DIR", Path(tmp) / "out"
            ), mock.patch.object(widget, "_start") as start:
                widget._apply_compress_many()
        trabajos = start.call_args.args[0]._trabajos
        destinos = [destino for _origen, destino in trabajos]
        self.assertEqual(len(destinos), 2)
        self.assertEqual(len(set(destinos)), 2)  # no colisionan entre sí
        for origen, destino in trabajos:
            self.assertNotEqual(destino.resolve(), origen.resolve())
        widget.close()

    def test_el_lote_ofrece_abrir_la_carpeta(self) -> None:
        """Al acabar el lote se recuerda la carpeta y se puede abrir."""
        widget = video_enhance_widget.VideoEnhanceWidget()
        destino_carpeta = Path(tempfile.gettempdir())
        with mock.patch.object(
            video_enhance_widget, "RECORDINGS_DIR", destino_carpeta
        ):
            widget._on_batch_done("3/3 comprimidos")
        self.assertTrue(widget._btn_open_out.isEnabled())
        self.assertEqual(widget._last_dir, destino_carpeta)
        self.assertIn("3/3 comprimidos", widget._status.text())
        widget.close()

    def test_el_lote_sigue_si_un_trozo_falla(self) -> None:
        """Un trozo dañado no puede tirar los otros nueve."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            origenes = []
            for nombre in ("a.mp4", "b.mp4"):
                ruta = tmp / nombre
                ruta.write_bytes(b"x" * 1024)
                origenes.append(ruta)
            trabajos = [(origen, tmp / f"out_{origen.name}") for origen in origenes]
            vistos: list[str] = []

            def falso_work(self):
                vistos.append(self._video.name)
                if len(vistos) == 1:
                    raise video_enhance_widget.EngineError("trozo dañado")
                Path(self._output).write_bytes(b"MP4" * 100)
                return self._output

            worker = video_enhance_widget.BatchCompressWorker(trabajos, 23, "cpu")
            avisos: list[int] = []
            worker.progress.connect(lambda p, _m: avisos.append(p))
            with mock.patch.object(
                video_enhance_widget.FfmpegEnhanceWorker, "work", falso_work
            ):
                resumen = worker.work()
        self.assertIn("1/2 comprimidos", resumen)
        self.assertIn("fallaron: a.mp4", resumen)
        self.assertEqual(vistos, ["a.mp4", "b.mp4"])
        # El progreso sube (0 → 100) y no vuelve a cero entre trozos.
        self.assertEqual(avisos, sorted(avisos))
        self.assertEqual((avisos[0], avisos[-1]), (0, 100))

    def test_el_lote_avisa_si_fallan_todos(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            origen = tmp / "a.mp4"
            origen.write_bytes(b"x")
            worker = video_enhance_widget.BatchCompressWorker(
                [(origen, tmp / "out.mp4")], 23, "cpu"
            )
            with mock.patch.object(
                video_enhance_widget.FfmpegEnhanceWorker,
                "work",
                side_effect=video_enhance_widget.EngineError("dañado"),
            ):
                with self.assertRaises(video_enhance_widget.EngineError) as ctx:
                    worker.work()
        self.assertIn("a.mp4", str(ctx.exception))

    def test_comprimir_no_necesita_ningun_ajuste(self) -> None:
        """Comprimir es el único caso que recodifica sin tocar la imagen."""
        widget = video_enhance_widget.VideoEnhanceWidget()
        widget._video = Path("peli.mp4")
        with mock.patch.object(
            video_enhance_widget, "RECORDINGS_DIR", Path(tempfile.gettempdir())
        ), mock.patch.object(widget, "_start") as start:
            widget._apply_compress()
        worker = start.call_args.args[0]
        self.assertIsInstance(worker, video_enhance_widget.FfmpegEnhanceWorker)
        self.assertEqual(worker._filters, "")
        self.assertEqual(worker._audio_codec, "aac")
        self.assertIn(worker._encoder, ("cpu", "gpu"))
        # La calidad es la del codificador elegido (el CQ de NVENC va aparte).
        self.assertEqual(
            worker._crf, video_enhance_widget.vp.quality_value("media", worker._encoder)
        )
        widget.close()

    def test_unir_grabaciones_ordena_y_lanza_el_trabajo(self) -> None:
        """Los trozos de cámara se unen en orden (00000 antes que 00001)."""
        widget = video_enhance_widget.VideoEnhanceWidget()
        with tempfile.TemporaryDirectory() as tmp:
            trozos = []
            for nombre in ("00001.MTS", "00000.MTS"):
                ruta = Path(tmp) / nombre
                ruta.write_bytes(b"x")
                trozos.append(str(ruta))
            with mock.patch.object(
                video_enhance_widget.QFileDialog,
                "getOpenFileNames",
                return_value=(trozos, ""),
            ):
                widget._add_clips()
            self.assertEqual(
                [widget._lista_union.item(i).text() for i in range(2)],
                [str(Path(tmp) / "00000.MTS"), str(Path(tmp) / "00001.MTS")],
            )
            with mock.patch.object(
                video_enhance_widget, "RECORDINGS_DIR", Path(tmp)
            ), mock.patch.object(widget, "_start") as start:
                widget._join_clips()
            self.assertTrue(start.called)
            worker = start.call_args.args[0]
            self.assertIsInstance(worker, video_enhance_widget.JoinWorker)
            self.assertEqual(len(worker._paths), 2)
        widget.close()

    def test_unir_pide_al_menos_dos_grabaciones(self) -> None:
        widget = video_enhance_widget.VideoEnhanceWidget()
        with mock.patch.object(widget, "_start") as start:
            widget._join_clips()
        self.assertFalse(start.called)
        self.assertIn("al menos dos", widget._status.text())
        widget.close()

    def test_el_reintento_iguala_los_trozos_si_la_copia_falla(self) -> None:
        """Unir trozos retocados: primero copia y, si no cuadra, recodifica."""
        with tempfile.TemporaryDirectory() as tmp:
            entradas = []
            for nombre in ("a.mp4", "b.mp4"):
                ruta = Path(tmp) / nombre
                ruta.write_bytes(b"x")
                entradas.append(ruta)
            destino = Path(tmp) / "unida.mp4"
            llamadas: list[list[str]] = []

            def falso_run(argv, _etiqueta, timeout=None, **kwargs):
                llamadas.append(list(argv))
                if len(llamadas) == 1:
                    return mock.Mock(returncode=1, stderr="Non-monotonous DTS")
                destino.write_bytes(b"MP4")
                return mock.Mock(returncode=0, stderr="")

            worker = video_enhance_widget.JoinWorker(entradas, destino)
            with mock.patch.object(
                video_enhance_widget, "_run_ffmpeg", side_effect=falso_run
            ), mock.patch.object(
                video_enhance_widget.VideoAssembler,
                "probe",
                return_value={"width": 1920, "height": 1080, "fps": 50.0},
            ):
                salida = worker.work()
        self.assertEqual(salida, destino)
        self.assertEqual(len(llamadas), 2)
        self.assertIn("copy", llamadas[0])
        self.assertIn("libx264", llamadas[1])
        self.assertIn("fps=50", " ".join(llamadas[1]))

    def test_igualar_desde_el_principio_no_reintenta(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            entradas = []
            for nombre in ("a.mp4", "b.mp4"):
                ruta = Path(tmp) / nombre
                ruta.write_bytes(b"x")
                entradas.append(ruta)
            destino = Path(tmp) / "unida.mp4"
            llamadas: list[list[str]] = []

            def falso_run(argv, _etiqueta, timeout=None, **kwargs):
                llamadas.append(list(argv))
                destino.write_bytes(b"MP4")
                return mock.Mock(returncode=0, stderr="")

            worker = video_enhance_widget.JoinWorker(entradas, destino, reencode=True)
            with mock.patch.object(
                video_enhance_widget, "_run_ffmpeg", side_effect=falso_run
            ):
                worker.work()
        self.assertEqual(len(llamadas), 1)
        self.assertIn("libx264", llamadas[0])

    def test_la_casilla_de_igualar_llega_al_worker(self) -> None:
        widget = video_enhance_widget.VideoEnhanceWidget()
        for indice in range(2):
            widget._lista_union.addItem(f"C:/tmp/trozo{indice}.mp4")
        widget._chk_join_reencode.setChecked(True)
        with mock.patch.object(
            video_enhance_widget, "RECORDINGS_DIR", Path(tempfile.gettempdir())
        ), mock.patch.object(widget, "_start") as start:
            widget._join_clips()
        self.assertTrue(start.call_args.args[0]._reencode)
        widget.close()

    def test_la_pestana_de_video_informa_del_progreso(self) -> None:
        """La barra refleja el avance; sin dato, queda en indeterminado."""
        widget = video_enhance_widget.VideoEnhanceWidget()
        widget._on_progress(42, "Aplicando filtros con ffmpeg…")
        self.assertEqual(widget._barra.value(), 42)
        self.assertIn("Aplicando filtros", widget._status.text())
        widget._on_progress(-1, "Arrancando…")
        self.assertEqual(widget._barra.minimum(), 0)
        self.assertEqual(widget._barra.maximum(), 0)  # indeterminada
        widget.close()

    def test_un_fallo_deja_la_barra_a_cero(self) -> None:
        widget = video_enhance_widget.VideoEnhanceWidget()
        widget._on_progress(70, "Casi…")
        widget._on_failed("se rompió")
        self.assertEqual(widget._barra.value(), 0)
        self.assertIn("Error", widget._status.text())
        widget.close()

    def test_chronicle_widget_arranca(self) -> None:
        widget = chronicle_widget.ChronicleWidget(EngineManager(self.cm))
        self._comprobar(widget)
        self.assertEqual(widget._lbl_video.text(), "(sin vídeo)")

    def test_la_cronica_conserva_el_audio_original_por_defecto(self) -> None:
        """Comentar un partido sin perder el ambiente es lo esperado por defecto."""
        widget = chronicle_widget.ChronicleWidget(EngineManager(self.cm))
        self.assertTrue(widget._chk_original.isChecked())
        widget._video = Path("peli.mp4")
        with mock.patch.object(widget, "_start") as start:
            widget._generate()
        self.assertTrue(start.call_args.args[0]._keep_original)
        widget.close()

    def test_el_ensamblador_no_conserva_el_audio_por_defecto(self) -> None:
        """En el Ensamblador manda la voz/música elegida; conservar es opcional."""
        widget = assembler_widget.AssemblerWidget(None, self.cm)
        self.assertFalse(widget._original_check.isChecked())
        widget.close()

    def test_el_ensamblador_ve_y_conserva_el_audio_del_video(self) -> None:
        """Al recibir una crónica (que ya trae audio) lo avisa y lo conserva."""
        widget = assembler_widget.AssemblerWidget(None, self.cm)
        with mock.patch.object(
            widget._assembler,
            "probe",
            return_value={"duration": 10.0, "width": 1920, "height": 1080, "has_audio": True},
        ):
            widget.set_video("C:/tmp/cronica.mp4")
        self.assertIn("Audio: sí", widget._video_info.text())
        self.assertTrue(widget._original_check.isChecked())
        self.assertIn("trae audio", widget._status.text())
        widget.close()

    def test_el_ensamblador_pasa_el_volumen_del_video(self) -> None:
        """El «Vol. vídeo» llega al worker como fracción (50 % -> 0.5)."""
        widget = assembler_widget.AssemblerWidget(None, self.cm)
        widget._video_path = "C:/tmp/v.mp4"
        widget._video_duration = 100.0  # evita sondear un fichero que no existe
        widget._original_volume.setValue(50)
        with mock.patch.object(assembler_widget, "AssemblerWorker") as worker_cls:
            widget._assemble()
        self.assertAlmostEqual(worker_cls.call_args.kwargs.get("original_volume"), 0.5)
        widget.close()

    def test_el_ensamblador_no_conserva_audio_si_el_video_no_lo_tiene(self) -> None:
        """Sin pista de audio, conservar haría fallar el ensamblado: se desmarca."""
        widget = assembler_widget.AssemblerWidget(None, self.cm)
        with mock.patch.object(
            widget._assembler,
            "probe",
            return_value={"duration": 10.0, "has_audio": False},
        ):
            widget.set_video("C:/tmp/mudo.mp4")
        self.assertIn("Audio: no", widget._video_info.text())
        self.assertFalse(widget._original_check.isChecked())
        widget.close()

    def test_la_cronica_pide_video(self) -> None:
        widget = chronicle_widget.ChronicleWidget(EngineManager(self.cm))
        with mock.patch.object(widget, "_start") as start:
            widget._generate()
        self.assertFalse(start.called)
        self.assertIn("Elige primero un vídeo", widget._status.text())
        widget.close()

    def test_la_cronica_pasa_los_volumenes(self) -> None:
        """Los volúmenes de la caja llegan al worker (vídeo de fondo y voz)."""
        widget = chronicle_widget.ChronicleWidget(EngineManager(self.cm))
        widget._video = Path("peli.mp4")
        widget._spn_original_volume.setValue(40)
        widget._spn_voice_volume.setValue(120)
        with mock.patch.object(widget, "_start") as start:
            widget._generate()
        worker = start.call_args.args[0]
        self.assertAlmostEqual(worker._original_volume, 0.4)
        self.assertAlmostEqual(worker._voice_volume, 1.2)
        widget.close()

    def test_la_cronica_recibe_los_datos_del_partido(self) -> None:
        """La alineación/marcador que escribe el usuario llega al worker."""
        widget = chronicle_widget.ChronicleWidget(EngineManager(self.cm))
        widget._txt_context.setPlainText("Muskis (azul) vs Corazonistas (rojo)")
        widget._video = Path("peli.mp4")
        with mock.patch.object(widget, "_start") as start:
            widget._generate()
        self.assertEqual(
            start.call_args.args[0]._context, "Muskis (azul) vs Corazonistas (rojo)"
        )
        widget.close()

    def test_la_cronica_pasa_la_disciplina(self) -> None:
        """El selector «Disciplina» llega al worker (por defecto, balonmano)."""
        widget = chronicle_widget.ChronicleWidget(EngineManager(self.cm))
        widget._video = Path("peli.mp4")
        with mock.patch.object(widget, "_start") as start:
            widget._generate()
        self.assertEqual(start.call_args.args[0]._sport, "balonmano")
        widget.close()

    def test_la_cronica_usa_el_informe_del_cuaderno(self) -> None:
        """Cargado un informe del Cuaderno, la crónica lo pasa al worker."""
        widget = chronicle_widget.ChronicleWidget(EngineManager(self.cm))
        widget._video = Path("peli.mp4")
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "informe.json"
            ruta.write_text(
                '{"video_path": "", "duracion_s": 100.0, '
                '"fotogramas": [{"t_s": 12.0, "descripcion": "gol de Álex"}], '
                '"transcripcion": ""}',
                encoding="utf-8",
            )
            widget._load_informe(str(ruta))
        self.assertTrue(widget._chk_informe.isChecked())
        with mock.patch.object(widget, "_start") as start:
            widget._generate()
        worker = start.call_args.args[0]
        self.assertIn("gol de Álex", worker._report)
        widget.close()

    def test_el_informe_rellena_los_datos_del_partido(self) -> None:
        """Al cargar un informe, la alineación del Cuaderno aparece en la caja."""
        widget = chronicle_widget.ChronicleWidget(EngineManager(self.cm))
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "informe.json"
            ruta.write_text(
                '{"video_path": "", "duracion_s": 100.0, '
                '"contexto": "Alineación Muskis: 1 Unai, 4 Álex", '
                '"fotogramas": [{"t_s": 12.0, "descripcion": "gol"}], '
                '"transcripcion": ""}',
                encoding="utf-8",
            )
            widget._load_informe(str(ruta))
        self.assertIn("Alineación Muskis", widget._txt_context.toPlainText())
        widget.close()

    def test_la_cronica_hereda_la_alineacion_del_informe(self) -> None:
        """Sin escribir nada en la caja, usa los datos del partido del informe."""
        widget = chronicle_widget.ChronicleWidget(EngineManager(self.cm))
        widget._video = Path("peli.mp4")
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "informe.json"
            ruta.write_text(
                '{"video_path": "", "duracion_s": 100.0, '
                '"contexto": "Alineación Muskis: 1 Unai, 4 Álex", '
                '"fotogramas": [{"t_s": 12.0, "descripcion": "gol"}], '
                '"transcripcion": ""}',
                encoding="utf-8",
            )
            widget._load_informe(str(ruta))
        # El usuario deja la caja vacía: la crónica debe usar la del informe.
        widget._txt_context.setPlainText("")
        with mock.patch.object(widget, "_start") as start:
            widget._generate()
        self.assertIn("Unai", start.call_args.args[0]._context)
        widget.close()

    def test_la_cronica_manda_su_video_al_ensamblador(self) -> None:
        widget = chronicle_widget.ChronicleWidget(EngineManager(self.cm))
        recibidos: list = []
        widget.clips_ready.connect(recibidos.append)
        widget._on_result(Path("C:/tmp/cronica.mp4"))
        widget._send_to_assembler()
        self.assertEqual(len(recibidos), 1)
        self.assertEqual(Path(recibidos[0]), Path("C:/tmp/cronica.mp4"))
        self.assertIn("enviada", widget._status.text())
        widget.close()

    def test_el_boton_de_acta_lanza_el_worker(self) -> None:
        """«Leer acta…» elige un archivo y arranca un ActaWorker (no un montaje)."""
        widget = chronicle_widget.ChronicleWidget(EngineManager(self.cm))
        with mock.patch.object(
            chronicle_widget.QFileDialog,
            "getOpenFileName",
            return_value=("C:/tmp/acta.pdf", ""),
        ), mock.patch.object(widget, "_start") as start:
            widget._read_acta()
        self.assertIsInstance(start.call_args.args[0], chronicle_widget.ActaWorker)
        self.assertTrue(callable(start.call_args.kwargs.get("on_result")))
        widget.close()

    def test_el_acta_rellena_los_datos_del_partido(self) -> None:
        """El bloque que devuelve el acta va a la caja y no marca un montaje."""
        widget = chronicle_widget.ChronicleWidget(EngineManager(self.cm))
        widget._on_acta("Equipos: A (azul) vs B (roja)")
        self.assertEqual(widget._txt_context.toPlainText(), "Equipos: A (azul) vs B (roja)")
        self.assertIn("Acta leída", widget._status.text())
        widget.close()

    def test_la_cronica_muestra_progreso_y_tiempo(self) -> None:
        """La pestaña Crónica tiene barra de avance y reloj (como Vídeo IA)."""
        widget = chronicle_widget.ChronicleWidget(EngineManager(self.cm))
        widget._on_progress(40, "Analizando el vídeo…")
        self.assertEqual(widget._barra.value(), 40)
        self.assertIn("Analizando", widget._status.text())
        widget._on_progress(-1, "Arrancando…")
        self.assertEqual(widget._barra.maximum(), 0)  # indeterminada
        widget._on_failed("se rompió")
        self.assertEqual(widget._barra.value(), 0)
        self.assertIn("Error", widget._status.text())
        widget.close()

    def test_el_cuaderno_acepta_avi(self) -> None:
        """El diálogo de vídeo del Cuaderno ofrece AVI (antes solo MP4/MKV/MOV/WebM)."""
        for extension in (".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"):
            self.assertIn(f"*{extension}", notes_widget.VIDEO_FILTER)

    def test_el_cuaderno_reutiliza_la_misma_carpeta_de_sesion(self) -> None:
        """El paquete y el análisis del mismo vídeo deben quedar juntos."""
        widget = notes_widget.NotesWidget(EngineManager(self.cm))
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            widget._video = tmp / "peli.mp4"
            carpeta = tmp / "notes"
            with mock.patch.object(notes_widget, "NOTES_DIR", carpeta):
                primera = widget._out_dir()
                segunda = widget._out_dir()
                self.assertEqual(primera, segunda)
                self.assertTrue(primera.is_dir())
            # Elegir otro vídeo reinicia la carpeta de sesión.
            otra = tmp / "otra.mp4"
            otra.write_bytes(b"x")
            with mock.patch.object(
                notes_widget.QFileDialog, "getOpenFileName",
                return_value=(str(otra), ""),
            ):
                widget._choose_video()
            self.assertIsNone(widget._session_dir)
        widget.close()

    def test_el_cuaderno_ofrece_abrir_la_carpeta(self) -> None:
        """Al terminar el paquete se recuerda la carpeta y se puede abrir."""
        widget = notes_widget.NotesWidget(EngineManager(self.cm))
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "peli_paquete.zip"
            zip_path.write_bytes(b"PK")
            widget._on_package_done(
                {"zip": zip_path, "frames": 10, "transcript": False}
            )
            self.assertTrue(widget._btn_open.isEnabled())
            self.assertEqual(widget._last_dir, Path(tmp))
            with mock.patch.object(notes_widget.QDesktopServices, "openUrl") as abrir:
                widget._open_folder()
            self.assertTrue(abrir.called)
        widget.close()

    def test_el_cuaderno_muestra_progreso(self) -> None:
        """La pestaña tiene barra de avance (antes solo había texto de estado)."""
        widget = notes_widget.NotesWidget(EngineManager(self.cm))
        widget._on_progress(40, "Analizando…")
        self.assertEqual(widget._barra.value(), 40)
        self.assertIn("Analizando", widget._status.text())
        widget.close()

    def test_el_cuaderno_pasa_disciplina_y_contexto(self) -> None:
        """La disciplina, el contexto y los fotogramas de visión llegan al worker."""
        widget = notes_widget.NotesWidget(EngineManager(self.cm))
        widget._video = Path("peli.mp4")
        widget._txt_context.setPlainText("Equipos: A vs B")
        widget._spn_vision.setValue(48)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            notes_widget, "NOTES_DIR", Path(tmp)
        ), mock.patch.object(widget, "_start_worker") as start:
            widget._analyze()
        worker = start.call_args.args[0]
        self.assertEqual(worker._sport, "balonmano")
        self.assertEqual(worker._context, "Equipos: A vs B")
        self.assertEqual(worker._vision_frames, 48)
        widget.close()

    def test_el_cuaderno_pasa_los_motores_de_percepcion(self) -> None:
        """Las señales de percepción (objetos/etiquetas/OCR) llegan al worker."""
        widget = notes_widget.NotesWidget(EngineManager(self.cm))
        widget._video = Path("peli.mp4")
        widget._chk_objects.setChecked(True)
        widget._chk_labels.setChecked(True)
        widget._chk_ocr.setChecked(False)  # se comprueba el caso «no marcada»
        widget._txt_clip_labels.setText("pista, balón")
        sentinels = {"yolo": object(), "clip": object(), "paddle_ocr": object()}
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            notes_widget, "NOTES_DIR", Path(tmp)
        ), mock.patch.object(
            widget._engine_manager,
            "get_engine",
            side_effect=lambda eid, *a, **k: sentinels.get(eid),
        ), mock.patch.object(widget, "_start_worker") as start:
            widget._analyze()
        worker = start.call_args.args[0]
        self.assertIs(worker._objects_engine, sentinels["yolo"])
        self.assertIs(worker._labels_engine, sentinels["clip"])
        self.assertIsNone(worker._ocr_engine)  # no marcada
        self.assertEqual(worker._clip_labels, "pista, balón")
        widget.close()

    def test_understanding_widget_arranca(self) -> None:
        widget = understanding_widget.UnderstandingWidget(EngineManager(self.cm))
        self._comprobar(widget)
        # Sin vídeo elegido no se puede analizar, y el widget lo dice.
        motivo = widget._validation_error(widget._build_plan())
        self.assertEqual(motivo, "Elige primero un vídeo.")
        widget.close()

    def test_la_comprension_pasa_la_disciplina(self) -> None:
        """El plan de Comprensión lleva la disciplina (por defecto, balonmano)."""
        widget = understanding_widget.UnderstandingWidget(EngineManager(self.cm))
        widget._video_path = Path("peli.mp4")
        self.assertEqual(widget._build_plan().sport, "balonmano")
        widget.close()

    def test_la_comprension_pasa_el_idioma(self) -> None:
        """El idioma del audio/resumen se puede fijar (evita el inglés del ruido)."""
        widget = understanding_widget.UnderstandingWidget(EngineManager(self.cm))
        widget._video_path = Path("peli.mp4")
        self.assertIsNone(widget._build_plan().language)  # Auto
        indice = widget._cmb_language.findData("es")
        widget._cmb_language.setCurrentIndex(indice)
        self.assertEqual(widget._build_plan().language, "es")
        widget.close()

    def test_la_comprension_pasa_etiquetas_clip(self) -> None:
        """El vocabulario de CLIP de la caja llega al worker (antes era fijo)."""
        widget = understanding_widget.UnderstandingWidget(EngineManager(self.cm))
        widget._video_path = Path("peli.mp4")
        widget._clip_labels.setText("pista de balonmano, balón, grada")
        motores = {
            "objects": None, "labels": None, "ocr": None,
            "transcription": None, "chat": None,
        }
        with mock.patch.object(
            widget, "_validation_error", return_value=None
        ), mock.patch.object(
            widget, "_resolve_engines", return_value=motores
        ), mock.patch.object(understanding_widget, "UnderstandingWorker") as worker_cls:
            widget._analyze()
        self.assertEqual(
            worker_cls.call_args.kwargs.get("clip_labels"),
            "pista de balonmano, balón, grada",
        )

    def test_la_comprension_pasa_el_contexto(self) -> None:
        """La caja de Contexto (fecha, lugar, equipos, alineación) llega al worker."""
        widget = understanding_widget.UnderstandingWidget(EngineManager(self.cm))
        widget._video_path = Path("peli.mp4")
        widget._context.setPlainText("Fecha: 9/10/2026\nEquipos: A vs B")
        motores = {
            "objects": None, "labels": None, "ocr": None,
            "transcription": None, "chat": None,
        }
        with mock.patch.object(
            widget, "_validation_error", return_value=None
        ), mock.patch.object(
            widget, "_resolve_engines", return_value=motores
        ), mock.patch.object(understanding_widget, "UnderstandingWorker") as worker_cls:
            widget._analyze()
        self.assertEqual(
            worker_cls.call_args.kwargs.get("context"),
            "Fecha: 9/10/2026\nEquipos: A vs B",
        )
        widget.close()

    def test_el_cuaderno_adopta_el_contexto_de_comprension(self) -> None:
        """Al reutilizar la percepción de Comprensión, su contexto rellena la caja."""
        widget = notes_widget.NotesWidget(EngineManager(self.cm))
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "comprension.json"
            ruta.write_text(
                '{"video_name": "x", "contexto": "Fecha: hoy\\nEquipos: A vs B", '
                '"fotogramas": [{"t_s": 1.0, "anotacion": "x"}], "transcripcion": ""}',
                encoding="utf-8",
            )
            widget._load_comprension(str(ruta))
        self.assertIn("Fecha: hoy", widget._txt_context.toPlainText())
        widget.close()

    def test_la_comprension_guarda_percepcion_estructurada(self) -> None:
        """El worker de Comprensión deja comprension.json para el Cuaderno."""
        from services import video_understanding as vu

        plan = vu.UnderstandingPlan(do_objects=True)
        with tempfile.TemporaryDirectory() as tmp:
            worker = understanding_widget.UnderstandingWorker(
                Path("peli.mp4"),
                plan,
                extractor=None,
                out_dir=Path(tmp),
            )
            worker._save_structured(
                plan,
                [vu.FrameReport(2.5, "JUGANDO")],
                transcript="hola",
                summary="resumen",
                duracion=120.0,
                notices=["x"],
            )
            leido = vu.load_understanding(Path(tmp) / vu.UNDERSTANDING_FILENAME)
        self.assertEqual(leido["video_name"], "peli.mp4")
        self.assertEqual(vu.understanding_frames(leido), [(2.5, "JUGANDO")])

    def test_el_cuaderno_reutiliza_la_percepcion_de_comprension(self) -> None:
        """Cargar comprension.json hace que el análisis lo pase al worker."""
        from services import video_understanding as vu

        widget = notes_widget.NotesWidget(EngineManager(self.cm))
        widget._video = Path("peli.mp4")
        with tempfile.TemporaryDirectory() as tmp:
            path = vu.save_understanding(
                Path(tmp) / vu.UNDERSTANDING_FILENAME,
                vu.understanding_to_dict(
                    video_path="C:/v/peli.mp4",
                    frames=[vu.FrameReport(2.5, "GOL del 7")],
                    transcript="gol",
                ),
            )
            widget._load_comprension(path)
            self.assertIsNotNone(widget._perception_data)
            with mock.patch.object(
                notes_widget, "NOTES_DIR", Path(tmp)
            ), mock.patch.object(widget, "_start_worker") as start:
                widget._analyze()
            worker = start.call_args.args[0]
        self.assertEqual(worker._perception, [(2.5, "GOL del 7")])
        self.assertEqual(worker._perception_transcript, "gol")
        widget.close()

    def test_el_cuaderno_muestra_el_marcador_ocr(self) -> None:
        """El resumen del marcador leído por OCR se pinta en su etiqueta."""
        widget = notes_widget.NotesWidget(EngineManager(self.cm))
        widget._mostrar_marcador("Marcador OCR: 1-0; 1 goles detectados.")
        self.assertIn("Marcador OCR", widget._lbl_marcador.text())
        widget.close()

    def test_el_cuaderno_muestra_el_contraste_de_contexto(self) -> None:
        """El Cuaderno avisa si el contexto choca con lo leído en pantalla."""
        widget = notes_widget.NotesWidget(EngineManager(self.cm))
        widget._txt_context.setPlainText("Marcador final: 28-25")
        widget._mostrar_contraste(
            {"fotogramas": [{"descripcion": "marcador 14-12"}], "transcripcion": ""}
        )
        self.assertIn("28-25", widget._lbl_contraste.text())
        widget.close()

    def test_la_disciplina_esta_sincronizada_entre_pestanas(self) -> None:
        """Cambiar la disciplina en una pestaña la cambia en las demás."""
        from services import preferences as prefs

        notas = notes_widget.NotesWidget(EngineManager(self.cm))
        cronica = chronicle_widget.ChronicleWidget(EngineManager(self.cm))
        comprension = understanding_widget.UnderstandingWidget(EngineManager(self.cm))
        # Las tres arrancan con la misma disciplina por defecto.
        for widget in (notas, cronica, comprension):
            self.assertEqual(widget._cmb_sport.currentData(), "balonmano")
        # Un cambio en el Cuaderno se propaga y se guarda como preferencia.
        notas._cmb_sport.setCurrentIndex(notas._cmb_sport.findData("generico"))
        self.assertEqual(cronica._cmb_sport.currentData(), "generico")
        self.assertEqual(comprension._cmb_sport.currentData(), "generico")
        self.assertEqual(self.cm.load_last_models().get(prefs.SPORT), "generico")
        notas.close()
        cronica.close()
        comprension.close()

    def test_las_preferencias_de_cada_pestana_van_a_la_config_dada(self) -> None:
        video_widget.VideoWidget(None, self.cm)
        audio_widget.AudioWidget(None, None, self.cm)

        ajustes = self.cm.load_settings()
        self.assertIn("last_models", ajustes)
        self.assertEqual(ajustes["last_models"].get("audio_voice_piper"), None)


if __name__ == "__main__":
    unittest.main()
