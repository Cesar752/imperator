"""Tests del worker de secuencia y de la tabla de guion de la pestaña Vídeo.

Dobles: el motor de vídeo es falso y la concatenación se parchea, así que no se
necesita GPU, ffmpeg ni los modelos reales. Se comprueba que las escenas se
generan con **semillas distintas** y prompt coherente, que se guardan en la
carpeta del proyecto y que la tabla editable vuelca de vuelta al guion.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from services import video_project as vproject  # noqa: E402
from services import video_script as vs  # noqa: E402


def _script(escenas: int = 2) -> vs.Script:
    filas = []
    for i in range(escenas):
        filas.append(
            {
                "heading": f"ESC {i}",
                "action": f"Ana actúa {i}",
                "visual_prompt": f"plano {i} de Ana en el faro",
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
                {"id": "ana", "name": "Ana", "appearance": "melena negra, gabardina"}
            ],
            "scenes": filas,
        }
    )


class _FakeEngine:
    """Motor de vídeo falso: devuelve siempre el mismo fichero y guarda los kwargs."""

    def __init__(self, clip: Path) -> None:
        self._clip = clip
        self.calls: list[dict] = []

    def list_models(self) -> list:
        return []

    def generate_video(self, **kwargs):
        self.calls.append(kwargs)
        return Path(self._clip)


class _FakeChat:
    """Motor de chat falso: devuelve una respuesta fija y guarda los mensajes."""

    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.messages = None

    def chat(self, messages, model=None):
        self.messages = messages
        return self._reply


class SequenceWorkerTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.clip = self.dir / "clip.mp4"
        self.clip.write_bytes(b"x")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_genera_clips_con_semillas_distintas_y_concatena(self) -> None:
        from ui import video_widget as vw

        script = _script(2)
        project = vproject.create_project("faro", script, root=self.dir)
        fake = _FakeEngine(self.clip)
        sequence = self.dir / "seq.mp4"
        worker = vw.SequenceWorker(fake, script, project, base_seed=5, chain=False)
        with mock.patch.object(
            vw.VideoAssembler, "concat_clips", return_value=sequence
        ):
            result = worker.work()

        self.assertEqual(len(result["clips"]), 2)
        self.assertTrue(project.clip_path(0).is_file())
        self.assertTrue(project.clip_path(1).is_file())
        self.assertEqual(result["sequence"], str(sequence))
        self.assertEqual(len(fake.calls), 2)
        self.assertNotEqual(fake.calls[0]["seed"], fake.calls[1]["seed"])
        self.assertIn("cine mudo en blanco y negro", fake.calls[0]["prompt"])

    def test_con_transiciones_usa_xfade(self) -> None:
        from ui import video_widget as vw

        script = _script(2)
        project = vproject.create_project("xfade", script, root=self.dir)
        sequence = self.dir / "seq.mp4"
        worker = vw.SequenceWorker(
            _FakeEngine(self.clip),
            script,
            project,
            base_seed=1,
            chain=False,
            transitions=True,
        )
        with mock.patch.object(
            vw.VideoAssembler, "concat_clips_with_transitions", return_value=sequence
        ) as xfade, mock.patch.object(vw.VideoAssembler, "concat_clips") as plain:
            result = worker.work()
        xfade.assert_called_once()
        plain.assert_not_called()
        self.assertEqual(result["sequence"], str(sequence))

    def test_un_solo_clip_no_concatena(self) -> None:
        from ui import video_widget as vw

        script = _script(1)
        project = vproject.create_project("uno", script, root=self.dir)
        worker = vw.SequenceWorker(
            _FakeEngine(self.clip), script, project, base_seed=1, chain=False
        )
        result = worker.work()
        self.assertEqual(result["sequence"], result["clips"][0])


class ScriptWorkerTest(unittest.TestCase):
    def test_work_parsea_el_json_del_motor(self) -> None:
        from ui import video_widget as vw

        reply = _script(2).to_json()
        worker = vw.ScriptWorker(_FakeChat(reply), "una farera y una luz", scenes=2)
        script = worker.work()
        self.assertEqual(script.title, "El faro")
        self.assertEqual(len(script.scenes), 2)

    def test_work_falla_con_respuesta_invalida(self) -> None:
        from engines.base_engine import EngineError
        from ui import video_widget as vw

        worker = vw.ScriptWorker(_FakeChat("lo siento, no puedo"), "idea")
        with self.assertRaises(EngineError):
            worker.work()


class VideoScriptTableTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from PyQt6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def _widget(self):
        from ui import video_widget as vw

        with mock.patch.object(vw, "run_pool_task"):
            return vw.VideoWidget(engine=_FakeEngine(Path("x")), config_manager=None)

    def test_load_script_rellena_la_tabla(self) -> None:
        widget = self._widget()
        try:
            widget.load_script(_script(3))
            self.assertEqual(widget._scene_table.rowCount(), 3)
            self.assertTrue(widget._script_group.isVisibleTo(widget))
        finally:
            widget.close()

    def test_editar_tabla_vuelca_al_guion(self) -> None:
        widget = self._widget()
        try:
            widget.load_script(_script(1))
            widget._scene_table.item(0, 1).setText("Texto editado")
            updated = widget._script_from_table()
            self.assertIsNotNone(updated)
            self.assertEqual(updated.scenes[0].narration, "Texto editado")
        finally:
            widget.close()

    def test_enviar_al_ensamblador_emite_secuencia(self) -> None:
        widget = self._widget()
        try:
            widget.load_script(_script(1))
            widget._clips = ["C:/tmp/a.mp4"]
            widget._sequence_path = "C:/tmp/a.mp4"
            recibidos: list = []
            widget.clips_ready.connect(recibidos.append)
            widget._send_to_assembler()
            self.assertEqual(recibidos, ["C:/tmp/a.mp4"])
        finally:
            widget.close()

    def test_start_script_desde_el_chat(self) -> None:
        from ui import video_widget as vw

        chat = _FakeChat(_script(4).to_json())
        with mock.patch.object(vw, "run_pool_task"):
            widget = vw.VideoWidget(engine=_FakeEngine(Path("x")), chat_engine=chat)
        try:
            with mock.patch.object(vw.ScriptWorker, "start"):
                widget.start_script("un viajero en el bosque", 4)
            self.assertEqual(widget._idea.text(), "un viajero en el bosque")
            self.assertEqual(widget._scenes_spin.value(), 4)
            self.assertIsNotNone(widget._script_worker)
        finally:
            widget.close()

    def test_boton_generar_guion_carga_la_tabla(self) -> None:
        from ui import video_widget as vw

        chat = _FakeChat(_script(2).to_json())
        with mock.patch.object(vw, "run_pool_task"):
            widget = vw.VideoWidget(engine=_FakeEngine(Path("x")), chat_engine=chat)
        try:
            widget._idea.setText("una farera y una luz")
            with mock.patch.object(vw.ScriptWorker, "start"):
                widget._generate_script()
                script = widget._script_worker.work()
            widget._on_script_ready(script)
            self.assertEqual(widget._scene_table.rowCount(), 2)
            self.assertIsNotNone(chat.messages)
            self.assertEqual(chat.messages[0]["role"], "system")
        finally:
            widget.close()


class VideoExtrasTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from PyQt6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def _widget(self, chat=None):
        from ui import video_widget as vw

        with mock.patch.object(vw, "run_pool_task"):
            return vw.VideoWidget(engine=_FakeEngine(Path("x")), chat_engine=chat)

    def test_modelo_por_guion(self) -> None:
        chat = _FakeChat("")
        chat.list_models = lambda: ["modelo-a", "modelo-b"]
        chat.default_model = "modelo-a"
        widget = self._widget(chat)
        try:
            # «por defecto» + los dos modelos.
            self.assertEqual(widget._script_model.count(), 3)
            self.assertEqual(widget._script_model.currentData(), "modelo-a")
        finally:
            widget.close()

    def test_transicion_override_y_config(self) -> None:
        widget = self._widget()
        try:
            widget.load_script(_script(2))
            widget._transition_combo.setCurrentIndex(
                widget._transition_combo.findData(vs.TRANSITION_WIPE_LEFT)
            )
            widget._fps.setValue(12)
            script = widget._apply_transition_override(widget._script)
            script = widget._with_config(script)
            self.assertTrue(
                all(s.transition == vs.TRANSITION_WIPE_LEFT for s in script.scenes)
            )
            self.assertEqual(script.config["fps"], 12)
            # Al reabrir (restore) vuelve el fps guardado.
            widget._script = script
            widget._transition_combo.setCurrentIndex(0)
            widget._fps.setValue(16)
            widget._restore_config()
            self.assertEqual(widget._fps.value(), 12)
        finally:
            widget.close()

    def test_reordenar_reordena_el_guion(self) -> None:
        from PyQt6.QtCore import Qt

        widget = self._widget()
        try:
            widget.load_script(_script(2))
            # Simular el arrastre: las filas 0 y 1 intercambian su índice original.
            widget._scene_table.item(0, 0).setData(Qt.ItemDataRole.UserRole, 1)
            widget._scene_table.item(1, 0).setData(Qt.ItemDataRole.UserRole, 0)
            widget._on_scene_rows_moved(None, 0, 0, 2, 2)
            self.assertEqual(widget._script.scenes[0].heading, "ESC 1")
            self.assertEqual(widget._script.scenes[1].heading, "ESC 0")
            self.assertEqual(widget._scene_table.item(0, 0).text(), "ESC 1")
        finally:
            widget.close()

    def test_refresh_thumbnails_sin_proyecto(self) -> None:
        widget = self._widget()
        try:
            widget.load_script(_script(1))
            widget._project = None
            widget._refresh_thumbnails()  # no debe lanzar
        finally:
            widget.close()


class ProjectsDialogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from PyQt6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def test_lista_y_borra(self) -> None:
        from PyQt6.QtWidgets import QMessageBox
        from ui.projects_dialog import ProjectsDialog

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = vproject.create_project("faro", _script(2), root=root)
            dialog = ProjectsDialog(root=root)
            try:
                self.assertEqual(dialog._list.count(), 1)
                dialog._list.setCurrentRow(0)
                with mock.patch(
                    "ui.projects_dialog.QMessageBox.question",
                    return_value=QMessageBox.StandardButton.Yes,
                ):
                    dialog._delete_selected()
                self.assertEqual(dialog._list.count(), 0)
                self.assertFalse(project.directory.exists())
            finally:
                dialog.close()


if __name__ == "__main__":
    unittest.main()
