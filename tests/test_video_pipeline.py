"""Tests del proyecto persistente y de la continuidad entre escenas.

Dobles puros: no necesitan GPU, red ni ffmpeg. Cubren la persistencia
(``services.video_project``) y la traducción guion→parámetros con continuidad
(``services.video_params``: semilla compartida y encadenado i2v).
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from services import video_params as vpar  # noqa: E402
from services import video_project as vp  # noqa: E402
from services import video_script as vs  # noqa: E402


def _script(a_escenas: int = 2) -> vs.Script:
    escenas = []
    for i in range(a_escenas):
        escenas.append(
            {
                "heading": f"ESC {i}",
                "action": f"Ana hace algo {i}",
                "visual_prompt": f"plano {i} con Ana en el faro",
                "narration": f"Narración {i}.",
                "characters": ["ana"],
                "duration_s": 4,
            }
        )
    return vs.Script.from_dict(
        {
            "title": "El faro",
            "language": "es",
            "style": "cine mudo en blanco y negro",
            "characters": [
                {"id": "ana", "name": "Ana", "appearance": "melena negra, gabardina ocre"}
            ],
            "scenes": escenas,
        }
    )


class ProjectTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_create_and_load(self) -> None:
        project = vp.create_project("El Faro", _script(), root=self.root)
        self.assertTrue(project.script_path.is_file())
        self.assertTrue(project.manifest_path.is_file())
        again = vp.load_project(project.directory)
        self.assertEqual(again.script.to_dict(), project.script.to_dict())

    def test_carpeta_fuera_de_output(self) -> None:
        self.assertTrue(str(vp.projects_root(self.root)).endswith("projects"))

    def test_create_duplicado_lanza(self) -> None:
        vp.create_project("El Faro", _script(), root=self.root)
        with self.assertRaises(FileExistsError):
            vp.create_project("El Faro", _script(), root=self.root)
        # Con overwrite sí.
        vp.create_project("El Faro", _script(), root=self.root, overwrite=True)

    def test_rutas_de_artefactos(self) -> None:
        project = vp.create_project("faro", _script(), root=self.root)
        self.assertEqual(project.clip_path(0).name, "escena_00.mp4")
        self.assertEqual(project.clip_path(1).name, "escena_01.mp4")
        self.assertEqual(project.narration_path(0).name, "narracion_00.wav")

    def test_faltantes_y_guardados(self) -> None:
        project = vp.create_project("faro", _script(), root=self.root)
        self.assertEqual(project.missing_scenes(), [1, 2])
        self.assertEqual(project.saved_clips(), [])
        project.clip_path(0).write_bytes(b"x")
        self.assertEqual(project.missing_scenes(), [2])
        self.assertEqual(project.saved_clips(), [project.clip_path(0)])

    def test_list_projects(self) -> None:
        vp.create_project("uno", _script(), root=self.root)
        vp.create_project("dos", _script(), root=self.root)
        self.assertEqual(len(vp.list_projects(self.root)), 2)

    def test_load_sin_guion_lanza(self) -> None:
        with self.assertRaises(vs.ScriptError):
            vp.load_project(self.root / "inexistente")

    def test_delete_project(self) -> None:
        project = vp.create_project("borrar", _script(), root=self.root)
        self.assertTrue(vp.delete_project(project.directory))
        self.assertFalse(project.directory.exists())
        self.assertFalse(vp.delete_project(self.root / "no-existe"))


class ContinuidadTest(unittest.TestCase):
    def test_semilla_determinista_y_distinta(self) -> None:
        self.assertEqual(vpar.scene_seed(1000, 0), vpar.scene_seed(1000, 0))
        self.assertNotEqual(vpar.scene_seed(1000, 0), vpar.scene_seed(1000, 1))

    def test_preset_por_formato(self) -> None:
        self.assertEqual(vpar.default_preset_for(vs.ASPECT_VERTICAL), "tiktok_calidad")
        self.assertEqual(vpar.default_preset_for(vs.ASPECT_HORIZONTAL), "equilibrado")

    def test_params_de_escena_llevan_prompt_y_semilla(self) -> None:
        script = _script()
        params = vpar.scene_generate_params(script, 0, base_seed=777, fps=16)
        self.assertIn("cine mudo en blanco y negro", params["prompt"])
        self.assertIn("melena negra, gabardina ocre", params["prompt"])
        self.assertEqual(params["seed"], vpar.scene_seed(777, 0))
        self.assertEqual(params["length"], vpar.sequence_length(4, 16))
        self.assertEqual(params["width"], vpar.PRESETS["equilibrado"][1])
        self.assertIsNone(params["image_path"])

    def test_params_reproducibles_con_la_misma_base(self) -> None:
        script = _script()
        a = vpar.scene_generate_params(script, 1, base_seed=42)
        b = vpar.scene_generate_params(script, 1, base_seed=42)
        self.assertEqual(a, b)

    def test_encadenado_i2v_usa_fotograma_previo(self) -> None:
        script = _script()
        params = vpar.scene_generate_params(
            script, 1, base_seed=1, mode="i2v", previous_frame="C:/tmp/last.png"
        )
        self.assertEqual(params["model"], "i2v")
        self.assertEqual(params["image_path"], "C:/tmp/last.png")


if __name__ == "__main__":
    unittest.main()
