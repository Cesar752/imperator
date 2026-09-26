"""Tests de core.video_assembler (lógica pura: SRT->ASS y helpers)."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import video_assembler as vc  # noqa: E402
from core.video_assembler import VideoAssembler  # noqa: E402
from engines.base_engine import EngineError  # noqa: E402

_SRT = """1
00:00:00,000 --> 00:00:01,500
Hola mundo

2
00:00:01,500 --> 00:00:03,000
Segunda linea
con salto
"""


class TimeHelpersTest(unittest.TestCase):
    def test_time_to_seconds(self) -> None:
        self.assertEqual(
            VideoAssembler._time_to_seconds("00", "01", "02", "500"), 62.5
        )

    def test_ass_time_format(self) -> None:
        self.assertEqual(VideoAssembler._ass_time(62.5), "0:01:02.50")
        self.assertEqual(VideoAssembler._ass_time(0), "0:00:00.00")

    def test_ass_time_negative_clamped(self) -> None:
        self.assertEqual(VideoAssembler._ass_time(-5), "0:00:00.00")

    def test_concat_entry_plain(self) -> None:
        entry = VideoAssembler._concat_entry(Path("C:/a/b.mp4"))
        self.assertEqual(entry, "file 'C:/a/b.mp4'")

    def test_concat_entry_escapes_quotes(self) -> None:
        entry = VideoAssembler._concat_entry(Path("C:/a/b'c.mp4"))
        # La comilla simple se escapa como '\'' para el demuxer concat.
        self.assertEqual(entry, "file 'C:/a/b'\\''c.mp4'")

    def test_filter_path_escapes(self) -> None:
        result = VideoAssembler._filter_path(Path("C:/a/b.ass"))
        self.assertTrue(result.endswith("a/b.ass"))
        self.assertIn("\\:", result)


class SrtParsingTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_parse_srt(self) -> None:
        srt = self.dir / "sub.srt"
        srt.write_text(_SRT, encoding="utf-8")
        events = VideoAssembler._parse_srt(srt)
        self.assertEqual(len(events), 2)
        start, end, text = events[0]
        self.assertAlmostEqual(start, 0.0)
        self.assertAlmostEqual(end, 1.5)
        self.assertEqual(text, "Hola mundo")
        # Varias líneas se unen con \N (salto de línea ASS).
        self.assertEqual(events[1][2], "Segunda linea\\Ncon salto")

    def test_build_ass_uses_video_resolution(self) -> None:
        srt = self.dir / "sub.srt"
        srt.write_text(_SRT, encoding="utf-8")
        assembler = VideoAssembler(output_dir=self.dir)
        ass = assembler._build_ass(
            srt,
            {"fontsize": 36, "position": "top", "color": "yellow"},
            video_width=1280,
            video_height=720,
        )
        content = ass.read_text(encoding="utf-8")
        self.assertIn("PlayResX: 1280", content)
        self.assertIn("PlayResY: 720", content)
        self.assertIn("Dialogue:", content)
        self.assertIn("Arial,36,", content)
        ass.unlink(missing_ok=True)


class FfmpegTimeoutTest(unittest.TestCase):
    """El ensamblaje no debe quedarse colgado si ffmpeg se atasca."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.assembler = VideoAssembler(output_dir=self.dir)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_run_pasa_el_timeout_a_subprocess(self) -> None:
        fake = mock.Mock(returncode=0, stderr="")
        with mock.patch.object(vc.subprocess, "run", return_value=fake) as run:
            VideoAssembler._run(["ffmpeg"], timeout=7)
        self.assertEqual(run.call_args.kwargs["timeout"], 7)

    def test_run_convierte_timeout_en_engine_error(self) -> None:
        expired = subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=7)
        with mock.patch.object(vc.subprocess, "run", side_effect=expired):
            with self.assertRaises(EngineError) as ctx:
                VideoAssembler._run(["ffmpeg"], timeout=7)
        self.assertIn("tiempo máximo", str(ctx.exception))

    def test_run_ok_no_lanza(self) -> None:
        fake = mock.Mock(returncode=0, stderr="")
        with mock.patch.object(vc.subprocess, "run", return_value=fake):
            VideoAssembler._run(["ffmpeg"], timeout=1)

    def test_run_error_incluye_ultimas_lineas(self) -> None:
        fake = mock.Mock(returncode=1, stderr="linea1\nlinea2")
        with mock.patch.object(vc.subprocess, "run", return_value=fake):
            with self.assertRaises(EngineError) as ctx:
                VideoAssembler._run(["ffmpeg"], timeout=1)
        self.assertIn("linea2", str(ctx.exception))

    def test_encode_timeout_escala_con_la_duracion(self) -> None:
        minimo = float(vc.ENCODE_TIMEOUT_MIN)
        casos = ((10.0, minimo), (600.0, 12000.0), (None, minimo))
        for duration, esperado in casos:
            with mock.patch.object(
                VideoAssembler, "probe", return_value={"duration": duration}
            ):
                self.assertEqual(
                    self.assembler._encode_timeout(Path("v.mp4")), esperado
                )

    def test_probe_usa_timeout(self) -> None:
        fake = mock.Mock(returncode=0, stderr="Duration: 00:00:10.00")
        with mock.patch.object(vc, "PROBE_TIMEOUT", 11), mock.patch.object(
            VideoAssembler, "_ffmpeg", return_value="ffmpeg"
        ), mock.patch.object(vc.subprocess, "run", return_value=fake) as run:
            self.assembler.probe("v.mp4")
        self.assertEqual(run.call_args.kwargs["timeout"], 11)


if __name__ == "__main__":
    unittest.main()
