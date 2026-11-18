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

    def test_el_prompt_obliga_a_seguir_la_idea_y_encadenar_las_escenas(self) -> None:
        """Fija el fallo real: el modelo se inventaba cosas y las escenas no se
        relacionaban entre sí (no seguían la historia)."""
        messages = vs.script_generation_messages("una farera y una luz", scenes=4)
        sistema = messages[0]["content"].lower()
        usuario = messages[1]["content"].lower()
        # La idea manda y no se inventa lo que no está en ella.
        self.assertIn("idea del usuario es el contrato", sistema)
        self.assertIn("no lo inventes", sistema)
        # Cadena causa → efecto entre escenas (en el sistema y en el repaso previo).
        self.assertIn("consecuencia directa de la anterior", sistema)
        self.assertIn("consecuencia de la anterior", usuario)
        # Lo que se narra es lo que se ve.
        self.assertIn("mismo instante", sistema)
        self.assertIn("mismo instante", usuario)
        # Y el esquema pide la relación causal también en el campo «action».
        self.assertIn("consecuencia de la anterior", messages[1]["content"])

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


class AmpliacionTest(unittest.TestCase):
    """Ampliar un guion: escenas nuevas que continúan lo ya escrito."""

    def _guion(self) -> vs.Script:
        return vs.Script.from_dict(
            {
                "title": "El faro",
                "language": "es",
                "style": "cine mudo en blanco y negro",
                "characters": [
                    {"id": "ana", "name": "Ana", "appearance": "melena negra"}
                ],
                "scenes": [
                    {
                        "heading": "ESC 1",
                        "action": "Ana llega al faro",
                        "visual_prompt": "se ve el faro",
                        "narration": "Ana sube al faro.",
                        "duration_s": 4,
                    }
                ],
            }
        )

    def test_los_mensajes_llevan_el_guion_y_piden_las_nuevas(self) -> None:
        original = self._guion()
        messages = vs.extension_messages(original, 3)
        self.assertIn("escenas nuevas", messages[0]["content"].lower())
        usuario = messages[1]["content"]
        # El modelo recibe el guion ya escrito (título, reparto y escenas).
        self.assertIn(original.title, usuario)
        self.assertIn("melena negra", usuario)
        self.assertIn("ESC 1", usuario)
        # Y se le piden 3 escenas más, solo las nuevas.
        self.assertIn("3 escena", usuario)
        self.assertIn("NUEVAS", usuario)

    def test_parse_extension_devuelve_las_escenas_nuevas(self) -> None:
        respuesta = (
            '{"scenes": [{"heading": "ESC 2", "action": "algo", '
            '"visual_prompt": "se ve algo", "narration": "texto", "duration_s": 4}]}'
        )
        scenes = vs.parse_extension(respuesta)
        self.assertEqual(len(scenes), 1)
        self.assertEqual(scenes[0].heading, "ESC 2")

    def test_parse_extension_sin_escenas_lanza(self) -> None:
        with self.assertRaises(vs.ScriptError):
            vs.parse_extension('{"scenes": []}')

    def test_extend_script_anade_al_final_sin_mutar_el_original(self) -> None:
        original = self._guion()
        nueva = vs.Scene(heading="ESC 2", action="x", visual_prompt="se ve")
        ampliado = vs.extend_script(original, (nueva,))
        self.assertEqual(len(ampliado.scenes), 2)
        self.assertEqual(ampliado.scenes[-1].heading, "ESC 2")
        self.assertEqual(len(original.scenes), 1)  # el modelo es inmutable


class AvisosTest(unittest.TestCase):
    """Validador ligero: avisos **deterministas** sobre el guion (sin modelo)."""

    def _guion(self, scenes, characters=None):
        return vs.Script.from_dict(
            {
                "title": "El faro",
                "language": "es",
                "style": "cine mudo",
                "characters": characters
                if characters is not None
                else [{"id": "ana", "name": "Ana", "appearance": "melena negra"}],
                "scenes": scenes,
            }
        )

    def _escena(self, **extra):
        base = {
            "heading": "EXT. FARO - NOCHE",
            "action": "Ana sube la escalera",
            "visual_prompt": "se ve la escalera",
            "narration": "Ana sube.",
            "characters": ["ana"],
            "duration_s": 4,
        }
        base.update(extra)
        return base

    def test_guion_limpio_no_da_avisos(self) -> None:
        script = self._guion(
            [
                self._escena(
                    heading="EXT. FARO - NOCHE",
                    action="Ana llega al faro",
                    narration="Ana llega al faro.",
                ),
                self._escena(
                    heading="INT. FARO - NOCHE",
                    action="Ana enciende la linterna",
                    narration="Enciende la linterna.",
                ),
            ]
        )
        self.assertEqual(vs.script_warnings(script), ())

    def test_detecta_lugar_y_accion_repetidos(self) -> None:
        script = self._guion(
            [self._escena(narration="Uno."), self._escena(narration="Dos.")]
        )
        codes = {w.code for w in vs.script_warnings(script)}
        self.assertIn("lugar_repetido", codes)
        self.assertIn("accion_repetida", codes)

    def test_detecta_narracion_vacia_y_repetida(self) -> None:
        script = self._guion(
            [
                self._escena(heading="A", action="ana sube", narration=""),
                self._escena(heading="B", action="ana corre", narration="Misma frase."),
                self._escena(heading="C", action="ana salta", narration="misma frase."),
            ]
        )
        codes = [w.code for w in vs.script_warnings(script)]
        self.assertIn("narracion_vacia", codes)
        self.assertIn("narracion_repetida", codes)

    def test_detecta_personaje_que_aparece_tarde(self) -> None:
        script = self._guion(
            [
                self._escena(narration="Ana entra."),
                self._escena(
                    heading="INT. FARO - NOCHE",
                    action="Luis grita en la puerta",
                    narration="Luis grita.",
                    characters=["luis"],
                ),
            ],
            characters=[
                {"id": "ana", "name": "Ana", "appearance": "melena negra"},
                {"id": "luis", "name": "Luis", "appearance": "gorra azul"},
            ],
        )
        avisos = [w for w in vs.script_warnings(script) if w.code == "personaje_nuevo"]
        self.assertEqual(len(avisos), 1)
        self.assertEqual(avisos[0].scene, 2)

    def test_detecta_personaje_del_reparto_sin_escena(self) -> None:
        script = self._guion(
            [self._escena(narration="Solo Ana.")],
            characters=[
                {"id": "ana", "name": "Ana", "appearance": "melena negra"},
                {"id": "luis", "name": "Luis", "appearance": "gorra azul"},
            ],
        )
        avisos = [
            w for w in vs.script_warnings(script) if w.code == "personaje_sin_escena"
        ]
        self.assertEqual(len(avisos), 1)
        self.assertIsNone(avisos[0].scene)

    def test_detecta_salto_de_manana_a_noche(self) -> None:
        script = self._guion(
            [
                self._escena(
                    heading="EXT. FARO - MAÑANA",
                    action="Ana llega al acantilado",
                    narration="Llega de día.",
                ),
                self._escena(
                    heading="INT. FARO - NOCHE",
                    action="Ana enciende la linterna",
                    narration="Ya es de noche.",
                ),
            ]
        )
        codes = {w.code for w in vs.script_warnings(script)}
        self.assertIn("salto_dia", codes)


if __name__ == "__main__":
    unittest.main()
