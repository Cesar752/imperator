"""Tests de core.video_assembler (lógica pura: SRT->ASS y helpers)."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.video_assembler import VideoAssembler  # noqa: E402

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


if __name__ == "__main__":
    unittest.main()
