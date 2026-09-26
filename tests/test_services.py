"""Tests de los módulos puros de services/ (sin Qt)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services import assembler_params as ap  # noqa: E402
from services import image_params as ip  # noqa: E402
from services import video_params as vp  # noqa: E402


class VideoParamsTest(unittest.TestCase):
    def test_sequence_length(self) -> None:
        # 4 s a 16 fps -> 64 frames + 1.
        self.assertEqual(vp.sequence_length(4, 16), 65)
        self.assertEqual(vp.sequence_length(2, 16), 33)

    def test_build_params_t2v(self) -> None:
        params = vp.build_generate_params(
            mode="t2v",
            negative_prompt="n",
            width=832,
            height=480,
            duration=2,
            fps=16,
            prompt="un gato",
        )
        self.assertEqual(params["prompt"], "un gato")
        self.assertIsNone(params["image_path"])
        self.assertEqual(params["length"], 33)
        self.assertNotIn("seed", params)

    def test_build_params_i2v_single(self) -> None:
        params = vp.build_generate_params(
            mode="i2v",
            negative_prompt="",
            width=480,
            height=832,
            duration=3,
            fps=16,
            photos=[{"path": "a.png", "prompt": "p1"}],
            seed=9,
        )
        self.assertEqual(params["image_path"], "a.png")
        self.assertEqual(params["prompt"], "p1")
        self.assertNotIn("image_paths", params)
        self.assertEqual(params["seed"], 9)

    def test_build_params_i2v_multiple(self) -> None:
        params = vp.build_generate_params(
            mode="i2v",
            negative_prompt="",
            width=480,
            height=832,
            duration=3,
            fps=16,
            photos=[{"path": "a", "prompt": "p1"}, {"path": "b", "prompt": "p2"}],
        )
        self.assertEqual(params["image_paths"], ["a", "b"])
        self.assertEqual(params["prompts"], ["p1", "p2"])

    def test_mode_labels_present(self) -> None:
        self.assertIn("t2v", vp.MODE_LABELS)
        self.assertIn("i2v", vp.MODE_LABELS)


class ImageParamsTest(unittest.TestCase):
    def test_display_model_bfl(self) -> None:
        self.assertEqual(
            ip.display_model("flux-kontext-pro", "bfl"),
            "FLUX.1 Kontext Pro · PAGO",
        )

    def test_display_model_local(self) -> None:
        self.assertEqual(ip.display_model("flux_kontext", "comfyui"), "Flux Kontext")

    def test_reference_helpers(self) -> None:
        self.assertTrue(ip.requires_reference("flux-kontext-max"))
        self.assertTrue(ip.optional_reference("flux-2-pro"))
        self.assertFalse(ip.requires_reference("flux-2-pro"))

    def test_build_generate_params(self) -> None:
        params = ip.build_generate_params(
            prompt="p",
            negative_prompt="n",
            width=1024,
            height=1024,
            steps=20,
            guidance=3.5,
            model="flux_dev",
            ref_path="ref.png",
            seed=1,
        )
        self.assertEqual(params["image_path"], "ref.png")
        self.assertEqual(params["seed"], 1)

    def test_build_generate_params_without_ref(self) -> None:
        params = ip.build_generate_params(
            prompt="p",
            negative_prompt="",
            width=1,
            height=1,
            steps=1,
            guidance=1.0,
            model="m",
        )
        self.assertNotIn("image_path", params)
        self.assertNotIn("seed", params)


class AssemblerParamsTest(unittest.TestCase):
    def test_build_subtitle_style(self) -> None:
        style = ap.build_subtitle_style(36, "top", "yellow")
        self.assertEqual(style["fontsize"], 36)
        self.assertEqual(style["position"], "top")
        self.assertEqual(style["color"], "yellow")
        self.assertEqual(style["outline"], ap.DEFAULT_OUTLINE)

    def test_duration_gap_none_when_similar(self) -> None:
        self.assertIsNone(ap.duration_gap_message(10.0, 10.5))

    def test_duration_gap_message(self) -> None:
        msg = ap.duration_gap_message(20.0, 5.0)
        self.assertIsNotNone(msg)
        self.assertIn("maestro", msg)

    def test_duration_gap_ignores_missing(self) -> None:
        self.assertIsNone(ap.duration_gap_message(None, 5.0))
        self.assertIsNone(ap.duration_gap_message(5.0, None))

    def test_offset_too_late(self) -> None:
        self.assertTrue(ap.offset_too_late(9.8, 10.0))
        self.assertFalse(ap.offset_too_late(2.0, 10.0))


if __name__ == "__main__":
    unittest.main()
