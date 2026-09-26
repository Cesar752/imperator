"""Tests de services.preferences (PrefsStore y dataclasses)."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config_manager import ConfigManager  # noqa: E402
from services import preferences as prefs  # noqa: E402


class PrefsStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.cm = ConfigManager(config_dir=self._tmp.name)
        self.store = prefs.PrefsStore(self.cm)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_defaults_when_empty(self) -> None:
        self.assertEqual(self.store.get_int("nada", 42), 42)
        self.assertEqual(self.store.get_float("nada", 1.5), 1.5)
        self.assertTrue(self.store.get_bool("nada", True))
        self.assertIsNone(self.store.get_str("nada"))

    def test_roundtrip_types(self) -> None:
        self.store.set("a", 1234)
        self.store.set("b", 2.5)
        self.store.set_bool("c", True)
        self.store.set("d", "texto")
        self.assertEqual(self.store.get_int("a", 0), 1234)
        self.assertEqual(self.store.get_float("b", 0.0), 2.5)
        self.assertTrue(self.store.get_bool("c", False))
        self.assertEqual(self.store.get_str("d"), "texto")

    def test_invalid_values_fall_back_to_default(self) -> None:
        self.cm.save_last_model("num", "no-es-numero")
        self.assertEqual(self.store.get_int("num", 7), 7)
        self.assertEqual(self.store.get_float("num", 7.5), 7.5)

    def test_set_ignores_none(self) -> None:
        self.store.set("x", None)
        self.assertIsNone(self.store.get_str("x"))

    def test_store_without_config(self) -> None:
        empty = prefs.PrefsStore(None)
        self.assertEqual(empty.as_map(), {})
        self.assertEqual(empty.get_int("k", 1), 1)
        empty.set("k", 5)  # no debe lanzar


class DataclassesTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.cm = ConfigManager(config_dir=self._tmp.name)
        self.store = prefs.PrefsStore(self.cm)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_video_prefs_defaults(self) -> None:
        vp = prefs.VideoPrefs.from_store(self.store)
        self.assertEqual(vp.mode, "t2v")
        self.assertEqual(vp.timeout, 1800)
        self.assertIsNone(vp.last_image)

    def test_video_prefs_values(self) -> None:
        self.cm.save_last_model(prefs.VIDEO, "i2v")
        self.cm.save_last_model(prefs.VIDEO_TIMEOUT, "900")
        self.cm.save_last_model(prefs.VIDEO_LAST_IMAGE, "a.png")
        vp = prefs.VideoPrefs.from_store(self.store)
        self.assertEqual(vp.mode, "i2v")
        self.assertEqual(vp.timeout, 900)
        self.assertEqual(vp.last_image, "a.png")

    def test_audio_prefs_defaults(self) -> None:
        ap = prefs.AudioPrefs.from_store(self.store)
        self.assertEqual(ap.engine, "piper")
        self.assertIsNone(ap.voice_piper)

    def test_assembler_prefs_defaults(self) -> None:
        asp = prefs.AssemblerPrefs.from_store(self.store)
        self.assertEqual(asp.fontsize, 24)
        self.assertEqual(asp.music_volume, 30)
        self.assertEqual(asp.voice_volume, 100)
        self.assertEqual(asp.voice_offset, 0.0)
        self.assertTrue(asp.ducking)

    def test_assembler_prefs_values(self) -> None:
        self.cm.save_last_model(prefs.ASSEMBLER_FONTSIZE, "40")
        self.cm.save_last_model(prefs.ASSEMBLER_VOICE_OFFSET, "1.5")
        self.cm.save_last_model(prefs.ASSEMBLER_DUCKING, "0")
        asp = prefs.AssemblerPrefs.from_store(self.store)
        self.assertEqual(asp.fontsize, 40)
        self.assertAlmostEqual(asp.voice_offset, 1.5)
        self.assertFalse(asp.ducking)


if __name__ == "__main__":
    unittest.main()
