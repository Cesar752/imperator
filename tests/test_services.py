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

    def test_sequence_length_es_siempre_cuatro_n_mas_uno(self) -> None:
        # Los nodos de latent de Wan 2.2 exigen length = 4n+1: con 30 fps y 3 s
        # la cuenta directa daría 91 (rechazado por ComfyUI); se ajusta a 93.
        self.assertEqual(vp.sequence_length(3, 30), 93)
        for duration, fps in ((1, 30), (2, 30), (3, 24), (5, 25), (1, 60)):
            length = vp.sequence_length(duration, fps)
            self.assertEqual((length - 1) % 4, 0, (duration, fps, length))

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

    def test_build_params_i2v_sin_fotos_no_lleva_claves_de_secuencia(self) -> None:
        # El encadenado entre escenas pide i2v **sin** fotos (el image_path se
        # pone después, con el último fotograma del clip anterior). Antes se
        # colaban ``image_paths``/``prompts`` vacíos y el motor moría con
        # «unexpected keyword argument 'image_paths'».
        params = vp.build_generate_params(
            mode="i2v",
            negative_prompt="",
            width=480,
            height=832,
            duration=2,
            fps=16,
        )
        self.assertNotIn("image_paths", params)
        self.assertNotIn("prompts", params)
        self.assertIsNone(params["image_path"])
        self.assertEqual(params["prompt"], "")

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

    def test_voice_locale(self) -> None:
        self.assertEqual(audio.voice_locale("es_ES-davefx-medium"), "es_ES")
        self.assertEqual(audio.voice_locale("es_AR-daniela-high"), "es_AR")
        self.assertEqual(audio.voice_locale("ef_dora"), "")

    def test_voice_label_dice_de_donde_es(self) -> None:
        self.assertEqual(
            audio.voice_label("es_ES-davefx-medium"), "es_ES-davefx-medium · castellano (España)"
        )
        self.assertIn("Argentina", audio.voice_label("es_AR-daniela-high"))
        # Kokoro no lleva el idioma en el nombre: se anuncia aparte.
        self.assertIn("Kokoro", audio.voice_label("ef_dora", "kokoro"))

    def test_sort_voices_pone_castellano_primero(self) -> None:
        voces = ["es_AR-daniela-high", "eu_ES-antton-medium", "es_ES-davefx-medium"]
        self.assertEqual(
            audio.sort_voices(voces),
            ["es_ES-davefx-medium", "es_AR-daniela-high", "eu_ES-antton-medium"],
        )


class DocumentParamsTest(unittest.TestCase):
    def test_prompt_for_known_modes(self) -> None:
        for mode_id in dp.MODE_LABELS:
            self.assertIn(mode_id, dp.PROMPTS)
            # El prompt del modo empieza por su cuerpo y añade lo que corresponda
            # (idioma de salida y regla de no comentar).
            self.assertTrue(
                dp.prompt_for(mode_id).startswith(dp.PROMPTS[mode_id]), mode_id
            )

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

    def test_los_prompts_piden_conservar_el_idioma(self) -> None:
        # Los modelos locales (qwen3-vl, qwen3.x) cubren euskera: el OCR no debe
        # traducir, el resumen debe responder en el idioma del documento y las
        # tablas deben conservar el texto original en las celdas.
        ocr = dp.prompt_for("ocr")
        self.assertIn("euskera", ocr)
        self.assertIn("Do NOT translate", ocr)

        for prompt in (dp.prompt_for("resumen"), dp.text_prompt_for("resumen")):
            self.assertIn("euskera", prompt)
            self.assertIn("SAME language", prompt)

        for prompt in (dp.prompt_for("tablas"), dp.text_prompt_for("tablas")):
            self.assertIn("original language", prompt)

    def test_detecta_los_modelos_que_cubren_idiomas_minoritarios(self) -> None:
        for modelo in ("qwen3-vl:30b-a3b", "qwen2.5vl:7b", "gemma3:12b"):
            self.assertTrue(dp.supports_minority_languages(modelo), modelo)
        for modelo in ("llava:13b", "moondream", "minicpm-v"):
            self.assertFalse(dp.supports_minority_languages(modelo), modelo)
        self.assertFalse(dp.supports_minority_languages(None))

    def test_aviso_de_idioma_solo_en_los_modos_que_entienden_el_texto(self) -> None:
        # En OCR basta transcribir; resumir o extraer tablas exige comprender.
        self.assertIsNone(dp.language_warning("llava:13b", "ocr"))
        self.assertIsNone(dp.language_warning("qwen3-vl:8b", "resumen"))
        self.assertEqual(
            dp.language_warning("llava:13b", "resumen"), dp.LANGUAGE_WARNING
        )
        self.assertEqual(
            dp.language_warning("llava:13b", "tablas"), dp.LANGUAGE_WARNING
        )

    def test_el_idioma_de_salida_se_puede_forzar(self) -> None:
        fuerzas = (
            ("es", "Spanish"),
            ("eu", "Basque (euskera)"),
            ("en", "English"),
        )
        for language, esperado in fuerzas:
            for prompt in (
                dp.prompt_for("resumen", language),
                dp.text_prompt_for("resumen", language),
                dp.prompt_for("tablas", language),
                dp.text_prompt_for("tablas", language),
            ):
                self.assertIn(esperado, prompt, language)
                self.assertNotIn("SAME language", prompt, language)

    def test_el_ocr_ignora_el_idioma_elegido(self) -> None:
        # Una transcripción nunca se traduce.
        self.assertEqual(dp.prompt_for("ocr", "es"), dp.prompt_for("ocr"))
        self.assertEqual(dp.prompt_for("ocr", "eu"), dp.PROMPTS["ocr"])

    def test_los_prompts_de_analisis_prohiben_los_comentarios(self) -> None:
        # Al modelo se le escapaba un "(3 sentences)" copiado del prompt.
        for prompt in (
            dp.prompt_for("resumen"),
            dp.prompt_for("resumen", "eu"),
            dp.text_prompt_for("tablas", "es"),
        ):
            self.assertIn("Do not add commentary", prompt)

    def test_la_instruccion_de_idioma_cae_en_auto(self) -> None:
        auto = dp.language_instruction("auto")
        self.assertEqual(dp.language_instruction(None), auto)
        self.assertEqual(dp.language_instruction("desconocido"), auto)
        self.assertIn("SAME language", auto)

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

    # ---- Reglas de estilo del resumen ------------------------------

    def test_los_resumenes_llevan_reglas_de_estilo(self) -> None:
        # Revisando un resumen real: mezclaba apartados en un párrafo, alternaba
        # frases largas con telegráficas y repetía la misma cifra dos veces.
        # Detalle «por apartado»: el ejecutivo es el de por defecto y tiene sus
        # propias reglas (ver ResumenEjecutivoTest).
        for prompt in (
            dp.prompt_for("resumen", detail=dp.STYLE_DETAILED),
            dp.text_prompt_for("resumen", detail=dp.STYLE_DETAILED),
            dp.global_prompt_for("resumen", detail=dp.STYLE_DETAILED),
        ):
            self.assertIn(dp.SUMMARY_STYLE, prompt)

    def test_el_detalle_por_defecto_es_ejecutivo_mas_apartados(self) -> None:
        # Medido: el detallado devolvía el 89 % y el ejecutivo solo, el 54 %
        # (no se puede bajar sin perder datos). Por defecto van los dos.
        self.assertEqual(dp.DEFAULT_DETAIL, dp.STYLE_TWO_PARTS)
        self.assertIn(dp.STYLE_TWO_PARTS, dp.DETAIL_LABELS)
        prompt = dp.prompt_for("resumen")
        self.assertIn("executive summary", prompt)
        self.assertNotIn("One section, one block", prompt)
        self.assertIn("around a fifth of the document", prompt)

    def test_el_detalle_ejecutivo_se_puede_elegir_solo(self) -> None:
        prompt = dp.prompt_for("resumen", detail=dp.STYLE_EXECUTIVE)
        self.assertIn("executive summary", prompt)
        self.assertNotIn("One section, one block", prompt)

    def test_el_tope_de_palabras_se_calcula_del_documento(self) -> None:
        # 400 palabras al 25 % → 100; y un documento muy corto no baja del suelo.
        self.assertEqual(dp.max_words_for("palabra " * 400), 100)
        self.assertEqual(dp.max_words_for("dos palabras"), dp.EXECUTIVE_MIN_WORDS)
        self.assertIn(
            f"at most {dp.max_words_for('palabra ' * 400)} words",
            dp.prompt_for("resumen", max_words=dp.max_words_for("palabra " * 400)),
        )

    def test_se_comprime_solo_si_se_pasa_del_tope(self) -> None:
        self.assertFalse(dp.excede_palabras("diez palabras cortas de prueba aqui", 10))
        self.assertTrue(dp.excede_palabras("palabra " * 30, 10))
        # El margen evita comprimir por una palabra de más (el modelo no mide).
        self.assertFalse(dp.excede_palabras("palabra " * 11, 10))

    def test_el_prompt_de_compresion_pide_el_tope(self) -> None:
        prompt = dp.compress_prompt_for("resumen", max_words=120)
        self.assertIn("still too long", prompt)
        self.assertIn("at most 120 words", prompt)
        self.assertIsNone(dp.compress_prompt_for("ocr"))

    def test_el_detalle_detallado_se_puede_elegir(self) -> None:
        prompt = dp.prompt_for("resumen", detail=dp.STYLE_DETAILED)
        self.assertIn(dp.SUMMARY_STYLE, prompt)
        self.assertNotIn("about a fifth of the document", prompt)

    def test_normalize_detail_acepta_los_tres_y_cae_en_dos_partes(self) -> None:
        self.assertEqual(dp.normalize_detail("detailed"), dp.STYLE_DETAILED)
        self.assertEqual(dp.normalize_detail("executive"), dp.STYLE_EXECUTIVE)
        self.assertEqual(dp.normalize_detail("two"), dp.STYLE_TWO_PARTS)
        for heredado in (None, "vete a saber"):
            with self.subTest(valor=heredado):
                self.assertEqual(dp.normalize_detail(heredado), dp.STYLE_TWO_PARTS)

    def test_las_reglas_cubren_separar_apartados_y_no_repetir(self) -> None:
        # Primera revisión: apartados mezclados, frases muy desiguales y la misma
        # cifra repetida en dos apartados.
        estilo = dp.SUMMARY_STYLE
        self.assertIn("one per distinct topic", estilo)
        self.assertIn("similar length", estilo)
        self.assertIn("do not repeat", estilo)
        self.assertIn("every sentence adds information", estilo)

    def test_las_reglas_piden_un_titulo_por_bloque(self) -> None:
        # Tercera revisión: sin títulos, el resumen se lee como un texto corrido
        # y cuesta localizar cada apartado. Se piden en texto plano porque el
        # área de resultado de Documentos no pinta markdown.
        estilo = dp.SUMMARY_STYLE
        self.assertIn("start every block with a one-line title", estilo)
        self.assertIn("document's own headings", estilo)
        self.assertIn("plain text", estilo)
        self.assertIn("do not invent titles", estilo)

    def test_las_reglas_exigen_fidelidad_al_texto(self) -> None:
        # Segunda revisión: pedir un "cierre" con "→" invitaba a interpretar y el
        # modelo llegó a inventar datos (habló de "equipamiento" donde el
        # documento decía acumular la hora de lactancia). Nada de interpretar ni
        # de añadir datos que no estén escritos.
        estilo = dp.SUMMARY_STYLE
        self.assertNotIn("→", estilo)
        self.assertIn("only what the document states", estilo)
        self.assertIn("Do not interpret", estilo)
        self.assertIn("do not invent", estilo)

        # El cierre (y su flecha) pasó a ser opcional: lo que se pide por defecto
        # es prohibirlo, y eso vive en su propia regla.
        self.assertIn("arrows", dp.NO_CLOSING_STYLE)
        for prompt in (
            dp.prompt_for("resumen"),
            dp.text_prompt_for("resumen"),
            dp.global_prompt_for("resumen"),
        ):
            self.assertNotIn("→", prompt)
            self.assertIn(dp.NO_CLOSING_STYLE, prompt)

    # ---- Frase de cierre por apartado (opcional) --------------------

    def test_la_frase_de_cierre_es_opcional(self) -> None:
        sin_cierre = dp.prompt_for("resumen")
        con_cierre = dp.prompt_for("resumen", closing=True)
        self.assertIn(dp.NO_CLOSING_STYLE, sin_cierre)
        self.assertNotIn("→", sin_cierre)
        self.assertIn(dp.CLOSING_STYLE, con_cierre)
        self.assertIn("→", con_cierre)
        self.assertNotIn(dp.NO_CLOSING_STYLE, con_cierre)
        # Las reglas de fidelidad siguen puestas con la casilla encendida.
        for prompt in (sin_cierre, con_cierre):
            self.assertIn("only what the document states", prompt)

    def test_el_cierre_se_pide_extractivo(self) -> None:
        # El problema no era la flecha, era que el modelo *inventaba* para
        # "concluir": ahora se le prohíbe añadir nada que no esté en el bloque.
        estilo = dp.CLOSING_STYLE
        self.assertIn("already written in it", estilo)
        self.assertIn("never add", estilo)

    def test_el_cierre_llega_a_los_tres_prompts_de_resumen(self) -> None:
        for prompt in (
            dp.prompt_for("resumen", closing=True),
            dp.text_prompt_for("resumen", closing=True),
            dp.global_prompt_for("resumen", closing=True),
        ):
            self.assertIn(dp.CLOSING_STYLE, prompt)

    def test_el_cierre_no_toca_ocr_ni_tablas(self) -> None:
        prompts = (
            dp.prompt_for("ocr", closing=True),
            dp.prompt_for("tablas", closing=True),
            dp.text_prompt_for("tablas", closing=True),
            dp.global_prompt_for("tablas", closing=True) or "",
        )
        for prompt in prompts:
            self.assertNotIn(dp.CLOSING_STYLE, prompt)
            self.assertNotIn(dp.NO_CLOSING_STYLE, prompt)

    def test_las_reglas_de_estilo_no_tocan_los_demas_modos(self) -> None:
        # El OCR transcribe y las tablas devuelven celdas: no son prosa.
        for prompt in (
            dp.prompt_for("ocr"),
            dp.prompt_for("ocr", "eu"),
            dp.prompt_for("tablas"),
            dp.text_prompt_for("tablas"),
        ):
            self.assertNotIn(dp.SUMMARY_STYLE, prompt)

    def test_las_reglas_de_estilo_conviven_con_el_idioma(self) -> None:
        prompt = dp.prompt_for("resumen", "eu", detail=dp.STYLE_DETAILED)
        self.assertIn(dp.SUMMARY_STYLE, prompt)
        self.assertIn("Basque (euskera)", prompt)

        ejecutivo = dp.prompt_for("resumen", "eu")
        self.assertIn("executive summary", ejecutivo)
        self.assertIn("Basque (euskera)", ejecutivo)


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

    def test_recorta_tambien_por_tamano(self) -> None:
        # Contar mensajes no basta: dos respuestas largas llenan la ventana de
        # contexto del motor y la generación se trunca.
        history = [
            {"role": "assistant", "content": "a" * 5000},
            {"role": "assistant", "content": "b" * 5000},
            {"role": "user", "content": "corta"},
        ]
        recortado = cp.trim_history(history, max_messages=20, max_chars=6000)
        self.assertEqual(len(recortado), 2)
        self.assertTrue(recortado[0]["content"].startswith("b"))
        self.assertEqual(recortado[-1]["content"], "corta")

    def test_conserva_el_ultimo_aunque_supere_el_presupuesto(self) -> None:
        history = [{"role": "user", "content": "x" * 20000}]
        self.assertEqual(
            cp.trim_history(history, max_messages=20, max_chars=100), history
        )

    def test_presupuesto_cero_desactiva_el_recorte_por_tamano(self) -> None:
        history = [
            {"role": "assistant", "content": "a" * 5000},
            {"role": "user", "content": "b" * 5000},
        ]
        self.assertEqual(len(cp.trim_history(history, max_chars=100)), 1)
        self.assertEqual(cp.trim_history(history, max_chars=0), history)

    def test_mensajes_sin_contenido_no_rompen_el_recorte(self) -> None:
        history = [{"role": "user"}, {"role": "assistant", "content": None}]
        self.assertEqual(cp.trim_history(history, max_chars=10), history)


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
