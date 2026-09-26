"""Tests de los módulos puros de services/ (sin Qt)."""

import sys
import unittest
from pathlib import Path
from unittest import mock

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services import assembler_params as ap  # noqa: E402
from services import audio_params as audio  # noqa: E402
from services import chat_params as cp  # noqa: E402
from services import document_params as dp  # noqa: E402
from services import image_params as ip  # noqa: E402
from services import preferences as prefs  # noqa: E402
from services import system_stats as ss  # noqa: E402
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


class AudioParamsTest(unittest.TestCase):
    def test_speed_range_is_sane(self) -> None:
        self.assertLess(audio.SPEED_MIN, audio.SPEED_DEFAULT)
        self.assertLess(audio.SPEED_DEFAULT, audio.SPEED_MAX)
        self.assertGreater(audio.SPEED_STEP, 0)

    def test_motors_are_unique_and_include_default(self) -> None:
        ids = [motor_id for _label, motor_id in audio.MOTORS]
        self.assertIn(audio.DEFAULT_MOTOR, ids)
        self.assertEqual(len(ids), len(set(ids)))

    def test_default_voices_cover_motors(self) -> None:
        for _label, motor_id in audio.MOTORS:
            self.assertIn(motor_id, audio.DEFAULT_VOICES)

    def test_voice_keys_match_preferences(self) -> None:
        self.assertEqual(audio.voice_key("piper"), prefs.AUDIO_VOICE_PIPER)
        self.assertEqual(audio.voice_key("kokoro"), prefs.AUDIO_VOICE_KOKORO)
        self.assertIsNone(audio.voice_key(None))
        self.assertIsNone(audio.voice_key("desconocido"))

    def test_resolve_voice_prefers_saved(self) -> None:
        self.assertEqual(
            audio.resolve_voice("piper", "guardada", ["guardada", "otra"]),
            "guardada",
        )

    def test_resolve_voice_falls_back_to_default(self) -> None:
        default = audio.DEFAULT_VOICES["piper"]
        self.assertEqual(
            audio.resolve_voice("piper", "no-existe", ["x", default]), default
        )

    def test_resolve_voice_falls_back_to_first(self) -> None:
        self.assertEqual(
            audio.resolve_voice("piper", "no-existe", ["primera"]), "primera"
        )

    def test_resolve_voice_none_without_voices(self) -> None:
        self.assertIsNone(audio.resolve_voice("piper", None, []))
        self.assertIsNone(audio.resolve_voice(None, None, []))


class DocumentParamsTest(unittest.TestCase):
    def test_prompt_for_known_modes(self) -> None:
        for mode_id in dp.MODE_LABELS:
            self.assertIn(mode_id, dp.PROMPTS)
            self.assertEqual(dp.prompt_for(mode_id), dp.PROMPTS[mode_id])

    def test_prompt_for_unknown_mode_falls_back_to_ocr(self) -> None:
        self.assertEqual(dp.prompt_for("desconocido"), dp.PROMPTS["ocr"])
        self.assertEqual(dp.prompt_for(None), dp.PROMPTS["ocr"])

    def test_los_prompts_de_imagen_difieren_por_modo(self) -> None:
        prompts = {mode: dp.prompt_for(mode) for mode in dp.MODE_LABELS}
        self.assertEqual(len(set(prompts.values())), len(prompts))

    def test_resumen_y_tablas_analizan_el_texto_nativo(self) -> None:
        for mode in ("resumen", "tablas"):
            self.assertTrue(dp.needs_text_analysis(mode))
            prompt = dp.text_prompt_for(mode)
            self.assertIsNotNone(prompt)
            self.assertNotEqual(prompt, dp.prompt_for(mode))
        self.assertNotEqual(
            dp.text_prompt_for("resumen"), dp.text_prompt_for("tablas")
        )

    def test_ocr_no_necesita_analisis_de_texto(self) -> None:
        # En OCR la capa de texto del PDF ya es la transcripción.
        self.assertIsNone(dp.text_prompt_for("ocr"))
        self.assertFalse(dp.needs_text_analysis("ocr"))
        self.assertFalse(dp.needs_text_analysis(None))
        self.assertFalse(dp.needs_text_analysis("desconocido"))

    def test_file_kind(self) -> None:
        self.assertEqual(dp.file_kind(".PDF"), "PDF")
        self.assertEqual(dp.file_kind(".png"), "Imagen")
        self.assertEqual(dp.file_kind(".docx"), "DOCX")
        self.assertEqual(dp.file_kind(""), "Archivo")

    def test_human_size(self) -> None:
        self.assertEqual(dp.human_size(512), "512 B")
        self.assertEqual(dp.human_size(2048), "2.0 KB")
        self.assertEqual(dp.human_size(5 * 1024**2), "5.0 MB")

    def test_describe_file(self) -> None:
        self.assertEqual(
            dp.describe_file("informe.pdf", ".pdf", 2048),
            "informe.pdf · PDF · 2.0 KB",
        )

    def test_has_native_text_respects_threshold(self) -> None:
        self.assertTrue(dp.has_native_text("x" * (dp.TEXT_THRESHOLD + 1)))
        self.assertFalse(dp.has_native_text("x" * dp.TEXT_THRESHOLD))
        self.assertFalse(dp.has_native_text(""))

    def test_is_pdf_ignores_case(self) -> None:
        self.assertTrue(dp.is_pdf(".pdf"))
        self.assertTrue(dp.is_pdf(".PDF"))
        self.assertFalse(dp.is_pdf(".png"))

    def test_file_filter_covers_extensions(self) -> None:
        self.assertIn("*.pdf", dp.FILE_FILTER)
        for ext in dp.IMAGE_EXTENSIONS:
            self.assertTrue(ext.startswith("."))
            self.assertEqual(ext, ext.lower())
            self.assertIn(f"*{ext}", dp.FILE_FILTER)


class ChatParamsTest(unittest.TestCase):
    def _history(self, count: int) -> list[dict]:
        return [{"role": "user", "content": str(i)} for i in range(count)]

    def test_limite_por_defecto_es_positivo(self) -> None:
        self.assertGreater(cp.MAX_HISTORY_MESSAGES, 0)

    def test_sin_recorte_si_cabe_entero(self) -> None:
        history = self._history(3)
        self.assertEqual(cp.trim_history(history, 5), history)

    def test_conserva_los_ultimos(self) -> None:
        history = self._history(10)
        esperado = [{"role": "user", "content": c} for c in ("7", "8", "9")]
        self.assertEqual(cp.trim_history(history, 3), esperado)

    def test_no_muta_el_original(self) -> None:
        history = self._history(5)
        cp.trim_history(history, 2)
        self.assertEqual(len(history), 5)

    def test_limite_cero_o_negativo_no_recorta(self) -> None:
        history = self._history(5)
        self.assertEqual(cp.trim_history(history, 0), history)
        self.assertEqual(cp.trim_history(history, -1), history)


class SystemStatsTest(unittest.TestCase):
    def _response(self, payload: dict) -> mock.Mock:
        resp = mock.Mock()
        resp.json.return_value = payload
        return resp

    def test_url_por_defecto_es_la_de_comfyui(self) -> None:
        self.assertIn("8188", ss.DEFAULT_BASE_URL)

    def test_lee_ram_free_y_pasa_timeout(self) -> None:
        resp = self._response({"system": {"ram_free": 1234}})
        with mock.patch.object(ss.requests, "get", return_value=resp) as get:
            self.assertEqual(ss.read_ram_free("http://x:1"), 1234)
        self.assertEqual(get.call_args.args[0], "http://x:1/system_stats")
        self.assertEqual(get.call_args.kwargs["timeout"], ss.STATS_TIMEOUT)

    def test_usa_el_endpoint_por_defecto_sin_base_url(self) -> None:
        resp = self._response({"system": {"ram_free": 1}})
        with mock.patch.object(ss.requests, "get", return_value=resp) as get:
            ss.read_ram_free(None)
        self.assertEqual(get.call_args.args[0], f"{ss.DEFAULT_BASE_URL}/system_stats")

    def test_devuelve_none_si_falla_la_red(self) -> None:
        with mock.patch.object(
            ss.requests, "get", side_effect=requests.RequestException("boom")
        ):
            self.assertIsNone(ss.read_ram_free("http://x:1"))

    def test_devuelve_none_si_el_json_es_invalido(self) -> None:
        resp = mock.Mock()
        resp.json.side_effect = ValueError("no json")
        with mock.patch.object(ss.requests, "get", return_value=resp):
            self.assertIsNone(ss.read_ram_free("http://x:1"))

    def test_devuelve_none_si_el_campo_falta_o_no_es_numerico(self) -> None:
        for payload in ({}, {"system": {}}, {"system": {"ram_free": "mucho"}}):
            with mock.patch.object(
                ss.requests, "get", return_value=self._response(payload)
            ):
                self.assertIsNone(ss.read_ram_free("http://x:1"))

    def test_is_fresh(self) -> None:
        self.assertTrue(ss.is_fresh(100.0, 100.0 + ss.STALE_AFTER - 1))
        self.assertFalse(ss.is_fresh(100.0, 100.0 + ss.STALE_AFTER))
        self.assertFalse(ss.is_fresh(None, 100.0))


if __name__ == "__main__":
    unittest.main()
