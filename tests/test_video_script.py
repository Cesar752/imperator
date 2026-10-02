"""Tests del guion de vídeo (``services.video_script``) y del prompt de sistema.

Son dobles puros: no necesitan GPU, red ni los motores. Validan el modelo
(``Script``/``Scene``/``Character``), la serialización JSON, el *parsing*
tolerante de la respuesta del modelo y la construcción de prompts que garantiza
que los personajes se describen igual en todas las escenas.
"""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from services import chat_params as cp  # noqa: E402
from services import video_script as vs  # noqa: E402


def _guion_valido() -> dict:
    return {
        "title": "El faro",
        "logline": "Una farera descubre una luz que responde.",
        "language": "es",
        "style": "cine mudo en blanco y negro, luz dura, 35mm",
        "aspect": "16:9",
        "characters": [
            {
                "id": "ana",
                "name": "Ana",
                "appearance": "melena negra, gabardina ocre, 40 años",
                "description": "farera solitaria",
            }
        ],
        "scenes": [
            {
                "heading": "EXT. FARO - NOCHE",
                "action": "Ana sube las escaleras del faro",
                "visual_prompt": "Ana sube la escalera de caracol iluminada por la lámpara",
                "narration": "Cada noche encendía la luz.",
                "characters": ["ana"],
                "duration_s": 4,
                "transition": "fade",
            }
        ],
    }


class ModeloTest(unittest.TestCase):
    def test_from_dict_valido(self) -> None:
        script = vs.Script.from_dict(_guion_valido())
        self.assertEqual(script.title, "El faro")
        self.assertEqual(len(script.scenes), 1)
        self.assertEqual(script.characters[0].id, "ana")
        self.assertEqual(script.total_duration(), 4)

    def test_personaje_sin_apariencia_lanza(self) -> None:
        data = _guion_valido()
        data["characters"][0]["appearance"] = ""
        with self.assertRaises(vs.ScriptError):
            vs.Script.from_dict(data)

    def test_escena_con_personaje_desconocido_lanza(self) -> None:
        data = _guion_valido()
        data["scenes"][0]["characters"] = ["fantasma"]
        with self.assertRaises(vs.ScriptError):
            vs.Script.from_dict(data)

    def test_duracion_se_acota_al_rango_del_motor(self) -> None:
        data = _guion_valido()
        data["scenes"][0]["duration_s"] = 99
        self.assertEqual(vs.Script.from_dict(data).scenes[0].duration_s, vs.CLIP_MAX_S)
        data["scenes"][0]["duration_s"] = 0
        self.assertEqual(vs.Script.from_dict(data).scenes[0].duration_s, vs.CLIP_MIN_S)

    def test_clamp_duration_tolera_tipos_raros(self) -> None:
        self.assertEqual(vs.clamp_duration(True), 4)
        self.assertEqual(vs.clamp_duration(None), 4)
        self.assertEqual(vs.clamp_duration("3"), 3)

    def test_roundtrip_json(self) -> None:
        script = vs.Script.from_dict(_guion_valido())
        again = vs.Script.from_json(script.to_json())
        self.assertEqual(again.to_dict(), script.to_dict())

    def test_sin_escenas_lanza(self) -> None:
        data = _guion_valido()
        data["scenes"] = []
        with self.assertRaises(vs.ScriptError):
            vs.Script.from_dict(data)

    def test_formato_desconocido_lanza(self) -> None:
        data = _guion_valido()
        data["aspect"] = "4:3"
        with self.assertRaises(vs.ScriptError):
            vs.Script.from_dict(data)

    def test_config_roundtrip(self) -> None:
        data = _guion_valido()
        data["config"] = {"fps": 12, "transitions": False, "transition": "wipe_left"}
        script = vs.Script.from_dict(data)
        self.assertEqual(script.config["fps"], 12)
        self.assertEqual(vs.Script.from_json(script.to_json()).config, script.config)

    def test_config_no_dict_se_ignora(self) -> None:
        data = _guion_valido()
        data["config"] = "no-es-un-dict"
        self.assertEqual(vs.Script.from_dict(data).config, {})

    def test_transiciones_ampliadas(self) -> None:
        for key in ("slide_left", "wipe_right", "circle", "zoom"):
            self.assertIn(key, vs.TRANSITIONS)
            self.assertIn(key, vs.TRANSITION_LABELS)


class CoherenciaTest(unittest.TestCase):
    def test_prompt_de_escena_repite_estilo_y_personaje(self) -> None:
        script = vs.Script.from_dict(_guion_valido())
        prompt = vs.scene_visual_prompt(script, script.scenes[0])
        self.assertIn(script.style, prompt)
        self.assertIn("melena negra, gabardina ocre, 40 años", prompt)
        self.assertIn("escalera de caracol", prompt)

    def test_dos_escenas_repiten_la_misma_apariencia(self) -> None:
        """La apariencia literal se repite: es lo que mantiene al personaje igual."""
        data = _guion_valido()
        segunda = dict(data["scenes"][0])
        segunda["visual_prompt"] = "Ana mira el horizonte desde la barandilla"
        data["scenes"].append(segunda)
        script = vs.Script.from_dict(data)
        prompts = vs.scene_prompts(script)
        apariencia = "melena negra, gabardina ocre, 40 años"
        self.assertTrue(all(apariencia in p for p in prompts))

    def test_narracion_une_escenas(self) -> None:
        script = vs.Script.from_dict(_guion_valido())
        self.assertEqual(script.narration(), "Cada noche encendía la luz.")

    def test_transition_sequence(self) -> None:
        data = _guion_valido()
        segunda = dict(data["scenes"][0])
        segunda["transition"] = "dissolve"
        data["scenes"].append(segunda)
        script = vs.Script.from_dict(data)
        self.assertEqual(vs.transition_sequence(script), ["dissolve"])
        simple = vs.Script.from_dict(_guion_valido())
        self.assertEqual(vs.transition_sequence(simple), [])


class EdicionTest(unittest.TestCase):
    def _script(self) -> vs.Script:
        data = _guion_valido()
        for i in range(2, 4):
            escena = dict(data["scenes"][0])
            escena["heading"] = f"ESCENA {i}"
            data["scenes"].append(escena)
        return vs.Script.from_dict(data)

    def test_mover_escena(self) -> None:
        script = self._script()
        movido = script.move_scene(0, 2)
        self.assertEqual(movido.scenes[2].heading, "EXT. FARO - NOCHE")
        self.assertEqual(script.scenes[0].heading, "EXT. FARO - NOCHE")  # inmutable

    def test_quitar_y_añadir(self) -> None:
        script = self._script()
        sin_primera = script.remove_scene(0)
        self.assertEqual(len(sin_primera.scenes), 2)
        nueva = vs.Scene(heading="NUEVA", action="algo", visual_prompt="se ve algo")
        con_nueva = sin_primera.add_scene(nueva)
        self.assertEqual(con_nueva.scenes[-1].heading, "NUEVA")

    def test_mover_fuera_de_rango_lanza(self) -> None:
        with self.assertRaises(vs.ScriptError):
            self._script().move_scene(0, 9)


class ParsingTest(unittest.TestCase):
    def test_parse_script_con_vallas_y_texto_alrededor(self) -> None:
        respuesta = (
            "Claro, aquí tienes el guion:\n```json\n"
            + vs.Script.from_dict(_guion_valido()).to_json()
            + "\n```\n¡Espero que te guste!"
        )
        script = vs.parse_script(respuesta)
        self.assertEqual(script.title, "El faro")

    def test_try_parse_script_devuelve_error_sin_lanzar(self) -> None:
        script, error = vs.try_parse_script("lo siento, no puedo")
        self.assertIsNone(script)
        self.assertTrue(error)

    def test_parse_script_vacio_lanza(self) -> None:
        with self.assertRaises(vs.ScriptError):
            vs.parse_script("")

    def test_parse_scene_count(self) -> None:
        self.assertEqual(vs.parse_scene_count("haz un guion de 4 escenas"), 4)
        self.assertEqual(vs.parse_scene_count("a 3 scenes film"), 3)
        self.assertIsNone(vs.parse_scene_count("un guion sobre el mar"))
        self.assertIsNone(vs.parse_scene_count("de 99 escenas"))


class PromptTest(unittest.TestCase):
    def test_mensajes_incluyen_sistema_y_esquema(self) -> None:
        messages = vs.script_generation_messages("una farera y una luz", scenes=5)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("guionista", messages[0]["content"].lower())
        self.assertIn("5", messages[1]["content"])
        self.assertIn('"scenes"', messages[1]["content"])
        self.assertIn(vs.DEFAULT_LANGUAGE, messages[1]["content"])

    def test_with_system_prompt_antepone_y_reemplaza(self) -> None:
        base = [{"role": "user", "content": "hola"}]
        con_sistema = cp.with_system_prompt(base, "eres un guionista")
        self.assertEqual(con_sistema[0]["role"], "system")
        self.assertEqual(base[0]["role"], "user")  # no muta

        ya_con_sistema = [{"role": "system", "content": "viejo"}, *base]
        reemplazado = cp.with_system_prompt(ya_con_sistema, "nuevo")
        self.assertEqual(len(reemplazado), 2)
        self.assertEqual(reemplazado[0]["content"], "nuevo")

    def test_with_system_prompt_vacio_no_cambia(self) -> None:
        base = [{"role": "user", "content": "hola"}]
        self.assertEqual(cp.with_system_prompt(base, ""), base)


if __name__ == "__main__":
    unittest.main()
