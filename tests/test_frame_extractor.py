"""Tests de `core.frame_extractor` con dobles (sin ffmpeg ni vídeos reales).

Es la pieza que llama a ffmpeg para Comprensión: se comprueba el comando que
construye (`-ss` antes de `-i`, audio mono a 16 kHz), el `timeout`, que un ffmpeg
que no escribe nada se trata como error y el mensaje cuando falta la dependencia.
Añadidos en la auditoría del 30/09/2026.
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import frame_extractor as fe  # noqa: E402
from engines.base_engine import EngineError  # noqa: E402


class _FakeImageioFfmpeg:
    @staticmethod
    def get_ffmpeg_exe() -> str:
        return "ffmpeg-falso"


class FrameExtractorTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.raiz = Path(self._tmp.name)
        self.video = self.raiz / "clip.mp4"
        self.video.write_bytes(b"MP4")
        self.comandos: list[list[str]] = []
        self.kwargs: dict = {}

        def run_falso(comando, **kwargs):
            self.comandos.append(list(comando))
            self.kwargs = kwargs
            # ffmpeg "escribe" el último argumento (el destino):
            destino = Path(comando[-1])
            destino.parent.mkdir(parents=True, exist_ok=True)
            destino.write_bytes(b"salida")
            return subprocess.CompletedProcess(comando, 0)

        self._patches = [
            mock.patch.object(fe, "imageio_ffmpeg", _FakeImageioFfmpeg),
            mock.patch.object(fe.subprocess, "run", side_effect=run_falso),
        ]
        for patcher in self._patches:
            patcher.start()
        self.extractor = fe.FrameExtractor(work_dir=self.raiz)

    def tearDown(self) -> None:
        for patcher in self._patches:
            patcher.stop()
        self._tmp.cleanup()

    # ---- Fotogramas ------------------------------------------------

    def test_extrae_el_fotograma_con_ss_antes_de_i(self) -> None:
        destino = self.raiz / "sub" / "frame.png"
        resultado = self.extractor.extract_frame(self.video, 12.3456, destino)

        self.assertEqual(resultado, destino)
        self.assertTrue(destino.is_file())
        comando = self.comandos[0]
        self.assertEqual(comando[0], "ffmpeg-falso")
        self.assertLess(comando.index("-ss"), comando.index("-i"))
        self.assertIn("12.346", comando)
        self.assertIn("-frames:v", comando)
        self.assertEqual(self.kwargs["timeout"], fe.FRAME_TIMEOUT)

    def test_un_timestamp_negativo_se_recorta_a_cero(self) -> None:
        self.extractor.extract_frame(self.video, -5, self.raiz / "f.png")
        self.assertIn("0.000", self.comandos[0])

    def test_si_ffmpeg_no_escribe_nada_es_un_error(self) -> None:
        with mock.patch.object(
            fe.subprocess, "run", return_value=subprocess.CompletedProcess(["ffmpeg"], 0)
        ):
            with self.assertRaises(EngineError) as ctx:
                self.extractor.extract_frame(self.video, 1, self.raiz / "f.png")
        self.assertIn("no generó el fotograma", str(ctx.exception))

    # ---- Audio -----------------------------------------------------

    def test_extrae_audio_mono_a_16k(self) -> None:
        destino = self.raiz / "voz.wav"
        self.extractor.extract_audio(self.video, destino)

        comando = self.comandos[0]
        self.assertIn("-vn", comando)
        self.assertEqual(comando[comando.index("-ac") + 1], "1")
        self.assertEqual(comando[comando.index("-ar") + 1], str(fe.AUDIO_SAMPLE_RATE))
        self.assertEqual(self.kwargs["timeout"], fe.AUDIO_TIMEOUT)

    def test_si_ffmpeg_no_escribe_el_audio_avisa_de_la_pista(self) -> None:
        with mock.patch.object(
            fe.subprocess, "run", return_value=subprocess.CompletedProcess(["ffmpeg"], 0)
        ):
            with self.assertRaises(EngineError) as ctx:
                self.extractor.extract_audio(self.video, self.raiz / "v.wav")
        self.assertIn("pista de sonido", str(ctx.exception))

    # ---- Errores ---------------------------------------------------

    def test_video_inexistente_avisa(self) -> None:
        with self.assertRaises(EngineError) as ctx:
            self.extractor.extract_frame(self.raiz / "no.mp4", 0, self.raiz / "f.png")
        self.assertIn("no existe", str(ctx.exception))

    def test_timeout_se_traduce_a_engine_error(self) -> None:
        expirado = subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=fe.FRAME_TIMEOUT)
        with mock.patch.object(fe.subprocess, "run", side_effect=expirado):
            with self.assertRaises(EngineError) as ctx:
                self.extractor.extract_frame(self.video, 0, self.raiz / "f.png")
        self.assertIn(f"{fe.FRAME_TIMEOUT} s", str(ctx.exception))

    def test_ffmpeg_que_no_se_puede_ejecutar_avisa(self) -> None:
        with mock.patch.object(fe.subprocess, "run", side_effect=OSError("no existe")):
            with self.assertRaises(EngineError) as ctx:
                self.extractor.extract_audio(self.video, self.raiz / "v.wav")
        self.assertIn("No se pudo ejecutar ffmpeg", str(ctx.exception))

    def test_sin_imageio_ffmpeg_avisa(self) -> None:
        with mock.patch.object(fe, "imageio_ffmpeg", None):
            with self.assertRaises(EngineError) as ctx:
                self.extractor.extract_frame(self.video, 0, self.raiz / "f.png")
        self.assertIn("imageio-ffmpeg no está instalado", str(ctx.exception))

    # ---- Consultas -------------------------------------------------

    def test_work_dir_se_expone(self) -> None:
        self.assertEqual(self.extractor.work_dir, self.raiz)
        self.assertIsNone(fe.FrameExtractor().work_dir)

    def test_duration_delega_en_el_ensamblador(self) -> None:
        with mock.patch.object(
            fe.VideoAssembler, "probe", return_value={"duration": 12.5}
        ):
            self.assertAlmostEqual(self.extractor.duration(self.video), 12.5)

    def test_duration_es_none_si_no_se_detecta(self) -> None:
        for devuelto in ({"duration": None}, {"duration": "raro"}):
            with mock.patch.object(fe.VideoAssembler, "probe", return_value=devuelto):
                self.assertIsNone(self.extractor.duration(self.video))


if __name__ == "__main__":
    unittest.main()
