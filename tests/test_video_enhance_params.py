"""Tests de los filtros de retoque de vídeo (services/video_enhance_params.py)."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services import video_enhance_params as vp  # noqa: E402


class ColorAdjustTest(unittest.TestCase):
    def test_identidad(self):
        self.assertTrue(vp.ColorAdjust().is_identity)
        self.assertIsNone(vp.ColorAdjust().eq_filter())

    def test_eq_con_valores(self):
        f = vp.ColorAdjust(brightness=0.2, contrast=1.5, saturation=0.5, gamma=1.2).eq_filter()
        self.assertIn("eq=", f)
        self.assertIn("brightness=0.2", f)
        self.assertIn("contrast=1.5", f)
        self.assertIn("saturation=0.5", f)
        self.assertIn("gamma=1.2", f)

    def test_calidez_neutra_no_es_filtro(self):
        self.assertIsNone(vp.ColorAdjust().temperature_filter())
        self.assertTrue(vp.ColorAdjust().is_identity)

    def test_calidez_a_kelvin(self):
        self.assertEqual(vp.temperature_kelvin(0), 6500)
        self.assertEqual(vp.temperature_kelvin(100), vp.TEMPERATURE_WARM_K)
        self.assertEqual(vp.temperature_kelvin(-100), vp.TEMPERATURE_COOL_K)
        # 6500 - 2700·0,5 = 5150 (a medio camino hacia el cálido máximo).
        self.assertEqual(vp.temperature_kelvin(50), 5150)

    def test_calidez_se_acota_y_tolera_basura(self):
        self.assertEqual(vp.temperature_kelvin(999), vp.TEMPERATURE_WARM_K)
        self.assertEqual(vp.temperature_kelvin(-999), vp.TEMPERATURE_COOL_K)
        self.assertEqual(vp.temperature_kelvin("basura"), vp.TEMPERATURE_NEUTRAL_K)

    def test_el_filtro_de_calidez_no_toca_el_eq(self):
        ajuste = vp.ColorAdjust(temperature=40)
        self.assertEqual(
            ajuste.temperature_filter(),
            f"colortemperature=temperature={vp.temperature_kelvin(40)}",
        )
        self.assertFalse(ajuste.is_identity)
        # Sólo cambia la calidez: el ``eq`` sigue neutro y no se emite.
        self.assertIsNone(ajuste.eq_filter())
        self.assertIn("colortemperature=", vp.build_filters(adjust=ajuste))


class FiltersTest(unittest.TestCase):
    def test_denoise_niveles(self):
        self.assertIsNone(vp.denoise_filter("off"))
        self.assertIn("hqdn3d", vp.denoise_filter("medium"))

    def test_sharpen(self):
        self.assertIsNone(vp.sharpen_filter(0))
        self.assertIn("unsharp=", vp.sharpen_filter(1.0))

    def test_fps_minterpolate(self):
        self.assertIsNone(vp.fps_filter(None))
        self.assertIn("minterpolate=fps=60", vp.fps_filter(60))

    def test_build_filters_combina(self):
        cadena = vp.build_filters(
            adjust=vp.ColorAdjust(brightness=0.1), denoise="light", sharpen=0.5, fps=60
        )
        self.assertIn("eq=", cadena)
        self.assertIn("hqdn3d", cadena)
        self.assertIn("unsharp=", cadena)
        self.assertIn("minterpolate", cadena)
        self.assertEqual(cadena.count(","), 3)

    def test_build_filters_vacio_si_identidad(self):
        self.assertEqual(vp.build_filters(adjust=vp.ColorAdjust()), "")


class SpeedTest(unittest.TestCase):
    """Velocidad: cámara lenta/rápida con el audio pegado al vídeo."""

    def test_acota_valores(self):
        self.assertEqual(vp.clamp_speed("basura"), vp.SPEED_DEFAULT)
        # 0 (o negativo) daría una división de PTS imposible.
        self.assertEqual(vp.clamp_speed(0), vp.SPEED_DEFAULT)
        self.assertEqual(vp.clamp_speed(-4), vp.SPEED_DEFAULT)
        self.assertEqual(vp.clamp_speed(0.1), vp.SPEED_MIN)
        self.assertEqual(vp.clamp_speed(9), vp.SPEED_MAX)
        self.assertEqual(vp.clamp_speed(0.75), 0.75)

    def test_atempo_se_encadena_fuera_de_rango(self):
        self.assertEqual(vp.atempo_chain(1.0), ["atempo=1"])
        self.assertEqual(vp.atempo_chain(0.5), ["atempo=0.5"])
        self.assertEqual(vp.atempo_chain(2.0), ["atempo=2"])
        # ffmpeg sólo admite 0.5–2 por paso: 0.25x son dos pasos.
        self.assertEqual(vp.atempo_chain(0.25), ["atempo=0.5", "atempo=0.5"])

    def test_sin_cambio_no_hay_filtro_de_audio(self):
        self.assertIsNone(vp.audio_speed_filter(1.0))
        self.assertEqual(vp.audio_speed_filter(0.5), "atempo=0.5")
        self.assertEqual(vp.audio_speed_filter(0.25), "atempo=0.5,atempo=0.5")

    def test_fps_interpolado_al_suavizar(self):
        # 0.5x de 30 fps -> 60; 0.25x -> 120 (tope).
        self.assertEqual(vp.smooth_slow_fps(0.5, 30), 60)
        self.assertEqual(vp.smooth_slow_fps(0.25, 30), vp.SMOOTH_MAX_FPS)
        self.assertEqual(vp.smooth_slow_fps(0.5, 50), 100)
        self.assertEqual(vp.smooth_slow_fps(0.5, None), 60)
        self.assertEqual(vp.smooth_slow_fps(0.5, 0), 60)
        self.assertEqual(vp.smooth_slow_fps(0.5, "basura"), 60)
        # Un vídeo de 200 fps no dispara la interpolación por encima del tope.
        self.assertEqual(vp.smooth_slow_fps(0.5, 200), vp.SMOOTH_MAX_FPS)

    def test_speed_filter(self):
        self.assertIsNone(vp.speed_filter(1.0))
        self.assertEqual(vp.speed_filter(0.5), "setpts=PTS/0.5")
        self.assertEqual(vp.speed_filter(2.0), "setpts=PTS/2")

    def test_suavizado_interpola_antes_de_estirar(self):
        cadena = vp.speed_filter(0.5, smooth=True, source_fps=30)
        self.assertIn("minterpolate=fps=60", cadena)
        # El orden importa: se inventan los fotogramas y luego se estira el tiempo.
        self.assertLess(cadena.index("minterpolate"), cadena.index("setpts"))
        # En velocidad rápida no se interpola: no hay fotogramas que inventar.
        self.assertNotIn("minterpolate", vp.speed_filter(2.0, smooth=True, source_fps=30))

    def test_build_filters_con_velocidad(self):
        self.assertEqual(vp.build_filters(speed=0.5), "setpts=PTS/0.5")
        # Sin velocidad, la cadena es la de siempre (no se cuela ningún filtro).
        self.assertEqual(vp.build_filters(fps=60).count(","), 0)
        cadena = vp.build_filters(fps=60, speed=0.5)
        self.assertLess(cadena.index("minterpolate"), cadena.index("setpts"))


class ReplayTest(unittest.TestCase):
    """Repeticiones de un momento: N vueltas y la última en cámara lenta."""

    def test_repite_el_fragmento(self):
        grafo = vp.replay_filter_complex(1, 3, 3)
        self.assertEqual(grafo.graph.count("[0:v]trim=start=1:end=3"), 3)
        self.assertEqual(grafo.graph.count("[0:a]atrim=start=1:end=3"), 3)
        self.assertIn("concat=n=3:v=1:a=1", grafo.graph)
        self.assertEqual(grafo.maps, ("[v]", "[a]"))
        self.assertEqual(grafo.times, 3)
        self.assertTrue(grafo.slow_last)

    def test_solo_la_ultima_va_lenta(self):
        grafo = vp.replay_filter_complex(1, 3, 3)
        self.assertEqual(grafo.graph.count("atempo"), 1)
        # Las dos primeras vueltas a velocidad normal y la última a la mitad.
        self.assertEqual(grafo.graph.count("setpts=(PTS-STARTPTS)/1"), 2)
        self.assertIn(f"setpts=(PTS-STARTPTS)/{vp.REPLAY_SLOW_SPEED:g}", grafo.graph)

    def test_sin_lenta_no_hay_atempo(self):
        grafo = vp.replay_filter_complex(1, 3, 2, slow_last=False)
        self.assertNotIn("atempo", grafo.graph)
        self.assertFalse(grafo.slow_last)

    def test_velocidad_lenta_se_encadena(self):
        grafo = vp.replay_filter_complex(0, 2, 2, slow_speed=0.25)
        self.assertEqual(grafo.graph.count("atempo=0.5"), 2)

    def test_video_mudo_no_mapea_audio(self):
        grafo = vp.replay_filter_complex(1, 3, 3, has_audio=False)
        self.assertNotIn("[0:a]", grafo.graph)
        self.assertIn("concat=n=3:v=1:a=0[vv]", grafo.graph)
        # El vídeo sale siempre por ``[v]``: sin zoom se pasa por ``null``, que es
        # gratis y evita que el worker tenga que mirar qué etiqueta salió.
        self.assertTrue(grafo.graph.endswith("[vv]null[v]"))
        self.assertIsNone(grafo.audio_label)
        self.assertEqual(grafo.maps, ("[v]",))

    def test_acota_las_repeticiones(self):
        self.assertIn("concat=n=6:", vp.replay_filter_complex(1, 3, 99).graph)
        self.assertIn("concat=n=1:", vp.replay_filter_complex(1, 3, 0).graph)
        self.assertEqual(vp.clamp_replay_times("basura"), vp.REPLAY_TIMES_DEFAULT)
        self.assertEqual(vp.clamp_replay_times(-3), vp.REPLAY_MIN_TIMES)

    def test_fragmento_invalido(self):
        with self.assertRaises(ValueError):
            vp.replay_filter_complex(3, 1)
        with self.assertRaises(ValueError):
            vp.replay_filter_complex(2, 2)
        with self.assertRaises(ValueError):
            vp.replay_filter_complex("a", "b")

    def test_zoom_de_la_repeticion(self):
        grafo = vp.replay_filter_complex(1, 3, 2, zoom=2.0)
        self.assertIn("crop=trunc(iw/2/2)*2:trunc(ih/2/2)*2", grafo.graph)
        self.assertTrue(grafo.graph.endswith("[v]"))
        # Sin zoom, el vídeo sale igualmente por [v] (pasando por null).
        self.assertTrue(vp.replay_filter_complex(1, 3, 2).graph.endswith("[vv]null[v]"))


class ZoomTest(unittest.TestCase):
    def test_sin_zoom_no_hay_filtro(self):
        self.assertIsNone(vp.zoom_filter(vp.ZOOM_DEFAULT))

    def test_recorta_el_centro_y_reescala(self):
        cadena = vp.zoom_filter(1.5)
        self.assertIn("crop=trunc(iw/1.5/2)*2:trunc(ih/1.5/2)*2", cadena)
        # Redondeo a par: H.264 con yuv420p rechaza dimensiones impares.
        self.assertIn("trunc(iw*1.5/2)*2", cadena)

    def test_con_tamano_la_salida_no_cambia_de_resolucion(self):
        """Sin el tamaño, 1.5x sobre 640x360 daba 638x360 (medido con ffmpeg).

        Eso rompería un ``concat`` posterior en el Ensamblador, que exige que todos
        los clips compartan parámetros.
        """
        cadena = vp.zoom_filter(1.5, (640, 360))
        self.assertTrue(cadena.endswith("scale=640:360"))
        self.assertNotIn("trunc", cadena.split(",")[1])

    def test_acota(self):
        self.assertEqual(vp.clamp_zoom("basura"), vp.ZOOM_DEFAULT)
        self.assertEqual(vp.clamp_zoom(0), vp.ZOOM_DEFAULT)
        self.assertEqual(vp.clamp_zoom(9), vp.ZOOM_MAX)

    def test_build_filters_aplica_zoom_primero(self):
        cadena = vp.build_filters(adjust=vp.ColorAdjust(saturation=1.2), zoom=2.0)
        self.assertTrue(cadena.startswith("crop="), cadena)
        self.assertIn("eq=", cadena)


class RampTest(unittest.TestCase):
    """Rampa: el vídeo entero con un tramo ralentizado."""

    def test_tramo_central(self):
        grafo = vp.speed_ramp_filter_complex(10, 4, 6, slow_speed=0.5)
        self.assertEqual(grafo.times, 3)
        self.assertTrue(grafo.slow_last is False)
        # Normal antes, lento en medio y normal después.
        self.assertIn("[0:v]trim=start=0:end=4", grafo.graph)
        self.assertIn("[0:v]trim=start=4:end=6", grafo.graph)
        self.assertIn("[0:v]trim=start=6:end=10", grafo.graph)
        self.assertEqual(grafo.graph.count("atempo=0.5"), 1)
        self.assertIn("concat=n=3:v=1:a=1", grafo.graph)

    def test_el_tramo_lento_es_el_del_medio(self):
        grafo = vp.speed_ramp_filter_complex(10, 4, 6, slow_speed=0.5)
        velocidades = [tramo.speed for tramo in grafo.segments]
        self.assertEqual(velocidades, [1.0, 0.5, 1.0])
        self.assertTrue(grafo.segments[1].is_slow)

    def test_sin_tramo_previo(self):
        grafo = vp.speed_ramp_filter_complex(10, 0, 4, slow_speed=0.5)
        self.assertEqual(grafo.times, 2)

    def test_sin_tramo_posterior(self):
        grafo = vp.speed_ramp_filter_complex(10, 6, 10, slow_speed=0.5)
        self.assertEqual(grafo.times, 2)

    def test_rampa_invalida(self):
        with self.assertRaises(ValueError):
            vp.speed_ramp_filter_complex(10, 6, 4)
        with self.assertRaises(ValueError):
            vp.speed_ramp_filter_complex(0, 1, 2)
        with self.assertRaises(ValueError):
            vp.speed_ramp_filter_complex(10, -1, 4)
        with self.assertRaises(ValueError):
            # Sin cambio de velocidad no hay rampa que montar.
            vp.speed_ramp_filter_complex(10, 2, 4, slow_speed=1.0)

    def test_rampa_suavizada_interpola_solo_el_tramo(self):
        grafo = vp.speed_ramp_filter_complex(
            10, 4, 6, slow_speed=0.5, smooth=True, source_fps=30
        )
        self.assertEqual(grafo.graph.count("minterpolate=fps=60"), 1)


class RoluloTest(unittest.TestCase):
    """Rótulos: «minuto:segundo frase» → SRT que quema el Ensamblador."""

    def test_marcas_de_tiempo(self):
        self.assertEqual(vp.parse_timestamp("0:12"), 12)
        self.assertEqual(vp.parse_timestamp("1:02:03"), 3723)
        self.assertEqual(vp.parse_timestamp("95.5"), 95.5)
        self.assertEqual(vp.parse_timestamp("1:05"), 65)
        for malo in ("", "   ", "a:b", "1:2:3:4"):
            with self.assertRaises(ValueError):
                vp.parse_timestamp(malo)

    def test_parsea_y_ordena(self):
        rotulos = vp.parse_rolulos(
            "# cabecera\n1:05 Parada de Unai\n\n0:12 Gol de Álex\n0:00 Jornada 5\n"
        )
        self.assertEqual([r.texto for r in rotulos], [
            "Jornada 5", "Gol de Álex", "Parada de Unai",
        ])
        self.assertEqual(rotulos[1].t_s, 12)

    def test_lineas_con_error(self):
        with self.assertRaises(ValueError) as ctx:
            vp.parse_rolulos("0:12\n")
        self.assertIn("Línea 1", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            vp.parse_rolulos("\n\nhola sin tiempo\n")
        self.assertIn("Línea 3", str(ctx.exception))
        with self.assertRaises(ValueError):
            vp.parse_rolulos("  \n# sólo comentarios\n")

    def test_no_admite_rotulos_fuera_del_video(self):
        with self.assertRaises(ValueError) as ctx:
            vp.parse_rolulos("2:00 muy tarde", duration=60)
        self.assertIn("dura", str(ctx.exception))
        # Dentro del vídeo, sin problema.
        self.assertEqual(len(vp.parse_rolulos("0:10 vale", duration=60)), 1)

    def test_srt_resultante(self):
        rotulos = vp.parse_rolulos("0:12 Gol de Álex", default_seconds=2.0)
        srt = vp.rolulos_srt(rotulos, default_seconds=2.0)
        self.assertIn("00:00:12,000 --> 00:00:14,000", srt)
        self.assertIn("Gol de Álex", srt)


class CompressTest(unittest.TestCase):
    """Comprimir sin filtros: solo calidad, con CPU o con GPU."""

    def test_valores_de_calidad(self):
        self.assertEqual(vp.quality_value("media"), 23)
        self.assertEqual(vp.quality_value("minima"), 30)
        # Clave desconocida -> el valor por defecto, nunca una excepción.
        self.assertEqual(vp.quality_value("basura"), 23)

    def test_la_gpu_usa_su_propia_escala(self):
        """Medido: CRF 23 ≈ CQ 30 en tamaño; el CQ va ~6 puntos por encima."""
        self.assertEqual(vp.quality_value("media", vp.ENCODER_GPU), 29)
        self.assertEqual(vp.quality_value("minima", vp.ENCODER_GPU), 36)
        for clave, _etiqueta, crf, cq in vp.COMPRESSION_QUALITIES:
            self.assertGreater(cq, crf, clave)
            self.assertEqual(vp.quality_value(clave), crf)
            self.assertEqual(vp.quality_value(clave, vp.ENCODER_GPU), cq)

    def test_detecta_el_codificador_por_gpu(self):
        self.assertTrue(vp.has_nvenc(" V....D h264_nvenc  NVIDIA NVENC H.264 encoder"))
        self.assertFalse(vp.has_nvenc(" V..... libx264              H.264 / AVC"))
        self.assertFalse(vp.has_nvenc(""))

    def test_argumentos_cpu(self):
        args = vp.encoder_args(vp.ENCODER_CPU, 26)
        self.assertEqual(args[args.index("-c:v") + 1], "libx264")
        self.assertEqual(args[args.index("-crf") + 1], "26")
        self.assertIn("-preset", args)

    def test_progreso_del_lote(self):
        """Un lote de 10 trozos no puede dar diez «100 %» y volver a cero."""
        self.assertEqual(vp.batch_progress(0, 0, 10), 0)
        self.assertEqual(vp.batch_progress(0, 50, 10), 5)
        self.assertEqual(vp.batch_progress(4, 100, 10), 50)
        self.assertEqual(vp.batch_progress(9, 100, 10), 100)
        # Un solo fichero es un lote de uno; y un total absurdo no divide por cero.
        self.assertEqual(vp.batch_progress(0, 50, 1), 50)
        self.assertEqual(vp.batch_progress(0, 50, 0), 50)

    def test_argumentos_gpu(self):
        args = vp.encoder_args(vp.ENCODER_GPU, 26)
        self.assertEqual(args[args.index("-c:v") + 1], "h264_nvenc")
        # Calidad constante: -cq con bitrate objetivo a 0 (lo anula).
        self.assertEqual(args[args.index("-cq") + 1], "26")
        self.assertEqual(args[args.index("-b:v") + 1], "0")
        self.assertNotIn("-crf", args)


class JoinTest(unittest.TestCase):
    """Unir grabaciones troceadas (los ``.MTS`` de una cámara)."""

    def _trozos(self, nombres):
        carpeta = tempfile.TemporaryDirectory()
        self.addCleanup(carpeta.cleanup)
        rutas = []
        for nombre in nombres:
            ruta = Path(carpeta.name) / nombre
            ruta.write_bytes(b"x")
            rutas.append(ruta)
        return rutas

    def test_ordena_por_nombre_y_quita_repetidos(self):
        trozos = self._trozos(["00002.MTS", "00000.MTS", "00001.MTS"])
        ordenadas = vp.join_paths_ordered(
            [trozos[0], trozos[2], trozos[1], trozos[0]]
        )
        self.assertEqual(
            [ruta.name for ruta in ordenadas],
            ["00000.MTS", "00001.MTS", "00002.MTS"],
        )

    def test_descarta_lo_que_no_existe(self):
        trozos = self._trozos(["00000.MTS"])
        ordenadas = vp.join_paths_ordered(
            [trozos[0], "C:/no/existe/00001.MTS", "", None]
        )
        self.assertEqual(len(ordenadas), 1)

    def test_comando_unión(self):
        argv = vp.join_command("ffmpeg", "lista.txt", "unida.mp4")
        self.assertIn("concat", argv)
        self.assertEqual(argv[argv.index("-safe") + 1], "0")
        # El vídeo se copia (sin pérdida) y el audio AC-3 pasa a AAC.
        self.assertEqual(argv[argv.index("-c:v") + 1], "copy")
        self.assertEqual(argv[argv.index("-c:a") + 1], "aac")
        self.assertIn("+faststart", argv)
        self.assertEqual(argv[-1], "unida.mp4")

    def test_grafo_que_iguala_los_trozos(self):
        """Cada trozo se normaliza antes de pegarlo (no vale el demuxer)."""
        grafo = vp.join_filter_complex(3, width=1920, height=1080, fps=50)
        self.assertIn("[0:v]scale=1920:1080", grafo)
        self.assertIn("[2:v]scale=1920:1080", grafo)
        self.assertIn("pad=1920:1080", grafo)
        self.assertIn("setsar=1", grafo)
        self.assertIn("fps=50", grafo)
        self.assertIn("[0:a]aresample=48000", grafo)
        self.assertIn("channel_layouts=stereo", grafo)
        self.assertIn("concat=n=3:v=1:a=1[v][a]", grafo)
        # ``setpts`` antes de ``fps``: al revés, ffmpeg 7.x deja la tasa en 1/0.
        self.assertLess(grafo.index("setpts"), grafo.index("fps="))

    def test_comando_que_iguala_los_trozos(self):
        argv = vp.join_normalize_command(
            "ffmpeg", ["a.mp4", "b.mp4"], "unida.mp4",
            width=1920, height=1080, fps=50,
        )
        # Una entrada por trozo: así da igual el códec de audio de cada uno.
        self.assertEqual(argv.count("-i"), 2)
        self.assertIn("-filter_complex", argv)
        self.assertEqual(argv[argv.index("-c:v") + 1], "libx264")
        self.assertEqual(argv[argv.index("-c:a") + 1], "aac")
        self.assertNotIn("copy", argv)

    def test_unión_igualando_sin_sonda(self):
        """Sin datos del primer trozo se usan 1920x1080 y 50 fps (lo habitual)."""
        grafo = vp.join_filter_complex(2)
        self.assertIn("scale=1920:1080", grafo)
        self.assertIn(f"fps={vp.JOIN_FALLBACK_FPS:g}", grafo)

    def test_ajuste_del_ritmo_al_estándar(self):
        self.assertEqual(vp.snap_fps(23.98), 24)
        self.assertEqual(vp.snap_fps(29.97), 30)
        self.assertEqual(vp.snap_fps(59.94), 60)
        self.assertIsNone(vp.snap_fps("basura"))
        self.assertIsNone(vp.snap_fps(0))

    def test_el_ritmo_lo_manda_la_cámara(self):
        """Medido con ffmpeg: una rampa deja el fichero en 38,67 fps nominales.

        Igualar a ese valor sería absurdo; con el trozo crudo de 50 fps al lado, el
        destino es 50 (el estándar más alto de todos).
        """
        self.assertEqual(vp.join_target_fps([38.67, 50.0]), 50.0)
        self.assertEqual(vp.join_target_fps([23.98, 25.0]), 25.0)
        self.assertEqual(vp.join_target_fps([]), vp.JOIN_FALLBACK_FPS)
        self.assertEqual(vp.join_target_fps([None, "x"]), vp.JOIN_FALLBACK_FPS)


class VolumenTest(unittest.TestCase):
    """Bajar (o subir) el volumen del audio de un vídeo."""

    def test_cien_no_es_filtro(self):
        self.assertIsNone(vp.volume_filter(100))
        self.assertIsNone(vp.volume_filter("basura"))

    def test_baja_y_sube(self):
        self.assertEqual(vp.volume_filter(60), "volume=0.600")
        self.assertEqual(vp.volume_filter(150), "volume=1.500")

    def test_se_acota(self):
        self.assertEqual(vp.volume_filter(-10), "volume=0.000")
        self.assertEqual(vp.volume_filter(999), "volume=2.000")


class ProgresoTest(unittest.TestCase):
    """Avance real de ffmpeg para la barra de la pestaña Vídeo IA."""

    def test_lee_los_microsegundos(self):
        self.assertAlmostEqual(vp.parse_out_time("out_time_us=2500000\n"), 2.5)
        # ffmpeg llama «ms» a lo que en realidad son microsegundos.
        self.assertAlmostEqual(vp.parse_out_time("out_time_ms=1000000\n"), 1.0)

    def test_ignora_otras_lineas(self):
        self.assertIsNone(vp.parse_out_time("progress=continue\n"))
        self.assertIsNone(vp.parse_out_time("out_time_us=basura\n"))
        self.assertIsNone(vp.parse_out_time(""))

    def test_porcentaje_proporcional(self):
        self.assertAlmostEqual(vp.progress_percent(0, 100), 0.0)
        self.assertAlmostEqual(vp.progress_percent(50, 100), 50.0)
        self.assertAlmostEqual(vp.progress_percent(100, 100), 100.0)

    def test_no_pasa_del_cien_por_el_ultimo_fotograma(self):
        self.assertAlmostEqual(vp.progress_percent(150, 100), 100.0)

    def test_se_traslada_al_tramo_pedido(self):
        self.assertAlmostEqual(vp.progress_percent(0, 100, stage=(10.0, 90.0)), 10.0)
        self.assertAlmostEqual(vp.progress_percent(50, 100, stage=(10.0, 90.0)), 50.0)
        self.assertAlmostEqual(vp.progress_percent(100, 100, stage=(10.0, 90.0)), 90.0)

    def test_sin_duracion_no_inventa_porcentaje(self):
        self.assertAlmostEqual(vp.progress_percent(5, 0, stage=(5.0, 99.0)), 5.0)
        self.assertAlmostEqual(vp.progress_percent(5, None, stage=(5.0, 99.0)), 5.0)


if __name__ == "__main__":
    unittest.main()
