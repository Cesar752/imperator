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

from services import video_params as vpar  # noqa: E402
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

    def test_reanuda_desde_la_primera_escena_que_falta(self) -> None:
        from ui import video_widget as vw

        script = _script(3)
        project = vproject.create_project("faro", script, root=self.dir, base_seed=999)
        # La escena 0 ya está hecha: simula el corte tras la primera escena.
        project.clip_path(0).write_bytes(b"x")
        fake = _FakeEngine(self.clip)
        worker = vw.SequenceWorker(
            fake,
            script,
            project,
            base_seed=project.base_seed,
            chain=False,
            resume=True,
            on_release=lambda: None,
            trim=lambda: False,
        )
        with mock.patch.object(
            vw.VideoAssembler, "concat_clips", return_value=self.dir / "seq.mp4"
        ):
            result = worker.work()

        # Solo se regeneran las que faltan, con la semilla común del proyecto.
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(fake.calls[0]["seed"], vpar.scene_seed(999, 1))
        self.assertEqual(len(result["clips"]), 3)
        self.assertTrue(project.clip_path(2).is_file())

    def test_sin_reanudar_regenera_todo(self) -> None:
        from ui import video_widget as vw

        script = _script(2)
        project = vproject.create_project("faro", script, root=self.dir)
        project.clip_path(0).write_bytes(b"x")
        fake = _FakeEngine(self.clip)
        worker = vw.SequenceWorker(
            fake,
            script,
            project,
            base_seed=1,
            chain=False,
            resume=False,
            trim=lambda: False,
        )
        with mock.patch.object(
            vw.VideoAssembler, "concat_clips", return_value=self.dir / "seq.mp4"
        ):
            worker.work()
        self.assertEqual(len(fake.calls), 2)

    def test_libera_memoria_entre_escenas(self) -> None:
        from ui import video_widget as vw

        script = _script(3)
        project = vproject.create_project("faro", script, root=self.dir)
        liberaciones: list = []
        worker = vw.SequenceWorker(
            _FakeEngine(self.clip),
            script,
            project,
            base_seed=1,
            chain=False,
            release_between_scenes=True,
            on_release=lambda: liberaciones.append(True),
            trim=lambda: False,
        )
        with mock.patch.object(
            vw.VideoAssembler, "concat_clips", return_value=self.dir / "seq.mp4"
        ):
            worker.work()
        # Una liberación por hueco entre escenas: 3 escenas -> 2 huecos.
        self.assertEqual(len(liberaciones), 2)

    def test_sin_liberacion_no_llama_al_hook(self) -> None:
        from ui import video_widget as vw

        script = _script(2)
        project = vproject.create_project("faro", script, root=self.dir)
        liberaciones: list = []
        worker = vw.SequenceWorker(
            _FakeEngine(self.clip),
            script,
            project,
            base_seed=1,
            chain=False,
            release_between_scenes=False,
            on_release=lambda: liberaciones.append(True),
            trim=lambda: False,
        )
        with mock.patch.object(
            vw.VideoAssembler, "concat_clips", return_value=self.dir / "seq.mp4"
        ):
            worker.work()
        self.assertEqual(liberaciones, [])

    def test_tamano_del_widget_llega_al_motor(self) -> None:
        from ui import video_widget as vw

        script = _script(1)
        project = vproject.create_project("faro", script, root=self.dir)
        fake = _FakeEngine(self.clip)
        worker = vw.SequenceWorker(
            fake,
            script,
            project,
            base_seed=1,
            chain=False,
            width=1024,
            height=576,
        )
        worker.work()
        self.assertEqual(fake.calls[0]["width"], 1024)
        self.assertEqual(fake.calls[0]["height"], 576)

    def test_solo_regenera_la_escena_indicada(self) -> None:
        """``only`` rehace una única escena (tras cambiar su foto), no el resto."""
        from ui import video_widget as vw

        script = _script(3)
        project = vproject.create_project("faro", script, root=self.dir)
        # Las tres escenas ya estaban hechas (caso real al regenerar una).
        for i in range(3):
            project.clip_path(i).write_bytes(b"x")
        fake = _FakeEngine(self.clip)
        worker = vw.SequenceWorker(
            fake, script, project, base_seed=1, chain=False, resume=True, only=1
        )
        with mock.patch.object(
            vw.VideoAssembler, "concat_clips", return_value=self.dir / "seq.mp4"
        ):
            result = worker.work()
        # Solo se llama al motor una vez, para la escena 1 (y solo esa se rehace).
        self.assertEqual(len(fake.calls), 1)
        self.assertIn("plano 1", fake.calls[0]["prompt"])
        # El resto se conserva: la secuencia se recompone con las tres.
        self.assertEqual(len(result["clips"]), 3)
        self.assertEqual(result["sequence"], str(self.dir / "seq.mp4"))


class _FakeImageEngine:
    """Motor de imagen falso: guarda los kwargs y devuelve un PNG recién creado."""

    def __init__(self, directory: Path) -> None:
        self._dir = directory
        self.calls: list[dict] = []
        self._n = 0

    def list_models(self) -> list:
        return ["flux_dev"]

    def generate_image(self, **kwargs):
        self.calls.append(kwargs)
        self._n += 1
        path = self._dir / f"salida_{self._n}.png"
        path.write_bytes(b"png")
        return path


class KeyframesTest(unittest.TestCase):
    """«Fotos primero»: una foto por escena y el clip i2v desde esa foto."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.clip = self.dir / "clip.mp4"
        self.clip.write_bytes(b"x")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_foto_por_escena_y_clip_desde_la_foto(self) -> None:
        from ui import video_widget as vw

        script = _script(2)
        project = vproject.create_project("faro", script, root=self.dir)
        images = _FakeImageEngine(self.dir)
        video = _FakeEngine(self.clip)
        worker = vw.SequenceWorker(
            video,
            script,
            project,
            base_seed=7,
            keyframes=True,
            image_engine=images,
            image_model="flux_dev",
            on_release=lambda: None,
            trim=lambda: False,
        )
        with mock.patch.object(
            vw.VideoAssembler, "concat_clips", return_value=self.dir / "seq.mp4"
        ):
            worker.work()

        # Una foto por escena, con su prompt y su semilla, guardada en el proyecto.
        self.assertEqual(len(images.calls), 2)
        self.assertEqual(images.calls[0]["model"], "flux_dev")
        self.assertEqual(images.calls[0]["seed"], vpar.scene_seed(7, 0))
        self.assertIn("plano 0", images.calls[0]["prompt"])
        self.assertTrue(project.frame_path(0).is_file())
        self.assertTrue(project.frame_path(1).is_file())
        # El clip i2v arranca de la foto de su escena (sin encadenado).
        self.assertEqual(video.calls[0]["image_path"], str(project.frame_path(0)))
        self.assertEqual(video.calls[1]["image_path"], str(project.frame_path(1)))
        self.assertEqual([call["model"] for call in video.calls], ["i2v", "i2v"])
        for call in video.calls:
            self.assertNotIn("image_paths", call)
            self.assertNotIn("prompts", call)

    def test_reutiliza_la_foto_ya_generada(self) -> None:
        script = _script(1)
        project = vproject.create_project("faro", script, root=self.dir)
        project.frame_path(0).write_bytes(b"png")  # ya estaba (o la puso el usuario)
        from ui import video_widget as vw

        images = _FakeImageEngine(self.dir)
        video = _FakeEngine(self.clip)
        worker = vw.SequenceWorker(
            video,
            script,
            project,
            base_seed=7,
            keyframes=True,
            image_engine=images,
        )
        worker.work()
        self.assertEqual(images.calls, [], "no se gasta GPU en una foto que ya está")
        self.assertEqual(video.calls[0]["image_path"], str(project.frame_path(0)))

    def test_sin_motor_de_imagen_cae_al_flujo_normal(self) -> None:
        from ui import video_widget as vw

        script = _script(1)
        project = vproject.create_project("faro", script, root=self.dir)
        video = _FakeEngine(self.clip)
        worker = vw.SequenceWorker(
            video,
            script,
            project,
            base_seed=7,
            keyframes=True,
            image_engine=None,
        )
        worker.work()
        self.assertEqual(video.calls[0]["model"], "t2v")
        self.assertEqual(project.saved_frames(), [])


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


class ScriptExtensionWorkerTest(unittest.TestCase):
    def test_devuelve_solo_las_escenas_nuevas(self) -> None:
        from ui import video_widget as vw

        reply = (
            '{"scenes": [{"heading": "ESC 3", "action": "algo", '
            '"visual_prompt": "se ve algo", "narration": "texto", "duration_s": 4}]}'
        )
        worker = vw.ScriptExtensionWorker(_FakeChat(reply), _script(2), 1)
        scenes = worker.work()
        self.assertEqual(len(scenes), 1)
        self.assertEqual(scenes[0].heading, "ESC 3")


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

    def test_el_guion_generado_engancha_el_proyecto_existente(self) -> None:
        """Un guion recién generado no trae proyecto: si sus escenas ya están
        hechas, el widget debe «saberlo» (clips cargados y «Enviar al Ensamblador»
        disponible) sin volver a generar nada."""
        from ui import video_widget as vw

        chat = _FakeChat(_script(2).to_json())
        with mock.patch.object(vw, "run_pool_task"):
            widget = vw.VideoWidget(engine=_FakeEngine(Path("x")), chat_engine=chat)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                project = vproject.create_project(
                    "El faro", _script(2), root=Path(tmp)
                )
                project.clip_path(0).write_bytes(b"x")
                project.clip_path(1).write_bytes(b"x")
                with mock.patch.object(
                    vw.vproject, "find_project", return_value=project
                ), mock.patch.object(vw.VideoWidget, "_refresh_thumbnails"):
                    widget._on_script_ready(_script(2))
            self.assertEqual(len(widget._clips), 2)
            self.assertTrue(widget._send_button.isEnabled())
            self.assertIn("ya generadas", widget._status.text())
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

    def test_nuevo_limpia_la_pestana(self) -> None:
        """«Nuevo» deja la pestaña en blanco para empezar otro vídeo."""
        widget = self._widget()
        try:
            widget.load_script(_script(2))
            widget._clips = ["a.mp4", "b.mp4"]
            widget._sequence_path = "a.mp4"
            widget._send_button.setEnabled(True)
            widget._start_fresh(confirm=False)
            self.assertIsNone(widget._script)
            self.assertIsNone(widget._project)
            self.assertEqual(widget._clips, [])
            self.assertEqual(widget._scene_table.rowCount(), 0)
            self.assertFalse(widget._send_button.isEnabled())
            self.assertIn("Vídeo nuevo", widget._status.text())
        finally:
            widget.close()

    def test_borrar_proyecto_lo_elimina_del_disco(self) -> None:
        from ui import video_widget as vw

        widget = self._widget()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                project = vproject.create_project("faro", _script(1), root=Path(tmp))
                widget._project = project
                with mock.patch.object(
                    vw.QMessageBox,
                    "question",
                    return_value=vw.QMessageBox.StandardButton.Yes,
                ), mock.patch.object(vw.VideoWidget, "_refresh_thumbnails"):
                    widget._delete_current_project()
                self.assertFalse(project.directory.exists())
            self.assertIsNone(widget._project)
            self.assertIn("borrado", widget._status.text())
        finally:
            widget.close()

    def test_reemplazar_clip_de_escena(self) -> None:
        from ui import video_widget as vw

        widget = self._widget()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                project = vproject.create_project("faro", _script(2), root=Path(tmp))
                source = Path(tmp) / "otro.mp4"
                source.write_bytes(b"clip-nuevo")
                widget._project = project
                widget._script = _script(2)
                widget._sequence_path = "vieja.mp4"
                with mock.patch.object(
                    vw.QFileDialog,
                    "getOpenFileName",
                    return_value=(str(source), ""),
                ), mock.patch.object(vw.VideoWidget, "_refresh_thumbnails"):
                    widget._replace_scene_clip(1)
                self.assertEqual(project.clip_path(1).read_bytes(), b"clip-nuevo")
            # La secuencia guardada ya no corresponde: se deja de ofrecer.
            self.assertIsNone(widget._sequence_path)
            self.assertFalse(widget._send_button.isEnabled())
            self.assertIn("reemplazado", widget._status.text())
        finally:
            widget.close()

    def test_reemplazar_foto_de_escena(self) -> None:
        from ui import video_widget as vw

        widget = self._widget()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                project = vproject.create_project("faro", _script(1), root=Path(tmp))
                source = Path(tmp) / "otro.png"
                source.write_bytes(b"png-nuevo")
                widget._project = project
                widget._script = _script(1)
                with mock.patch.object(
                    vw.QFileDialog,
                    "getOpenFileName",
                    return_value=(str(source), ""),
                ):
                    widget._replace_scene_photo(0)
                self.assertEqual(project.frame_path(0).read_bytes(), b"png-nuevo")
            self.assertIn("Regenerar esta escena", widget._status.text())
        finally:
            widget.close()

    def test_anadir_escena_en_blanco(self) -> None:
        """«Añadir escena» mete una fila nueva al final, para escribirla a mano."""
        widget = self._widget()
        try:
            widget.load_script(_script(2))
            widget._add_blank_scene()
            self.assertEqual(widget._scene_table.rowCount(), 3)
            self.assertEqual(len(widget._script.scenes), 3)
            self.assertEqual(widget._scene_table.item(2, 0).text(), "ESC 3")
            self.assertIn("Escena 3 añadida", widget._status.text())
        finally:
            widget.close()

    def test_los_avisos_deterministas_se_muestran_sin_bloquear(self) -> None:
        """El validador ligero avisa en la etiqueta y marca la fila, sin bloquear."""
        widget = self._widget()
        try:
            widget.load_script(_script(2))
            self.assertIn("aviso", widget._warnings.text())
            # La escena 2 repite la acción de la 1: se marca con su tooltip.
            self.assertIn("casi lo mismo", widget._scene_table.item(1, 0).toolTip())
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


class VideoWorkerTest(unittest.TestCase):
    """El worker del editor convierte las claves de la UI en parámetros del motor.

    ``image_paths``/``prompts`` (varias fotos) **no** son parámetros del motor:
    si llegaban a ``generate_video`` salía «unexpected keyword argument
    'image_paths'». El worker debe traducirlas a un ``image_path`` por foto.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.clip = self.dir / "clip.mp4"
        self.clip.write_bytes(b"x")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    @staticmethod
    def _params(mode: str, photos=None, prompt: str = "") -> dict:
        return vpar.build_generate_params(
            mode=mode,
            negative_prompt="borroso",
            width=832,
            height=480,
            duration=4,
            fps=16,
            prompt=prompt,
            photos=photos,
            seed=7,
        )

    def test_varias_fotos_un_clip_por_foto_y_concatena(self) -> None:
        from ui import video_widget as vw

        photos = [
            {"path": "a.png", "prompt": "p1"},
            {"path": "b.png", "prompt": "p2"},
        ]
        fake = _FakeEngine(self.clip)
        sequence = self.dir / "seq.mp4"
        worker = vw.VideoWorker(fake, self._params("i2v", photos=photos))
        with mock.patch.object(
            vw.VideoAssembler, "concat_clips", return_value=sequence
        ):
            result = worker.work()
        self.assertEqual(result, str(sequence))
        self.assertEqual([c["image_path"] for c in fake.calls], ["a.png", "b.png"])
        self.assertEqual([c["prompt"] for c in fake.calls], ["p1", "p2"])
        for call in fake.calls:
            self.assertNotIn("image_paths", call)
            self.assertNotIn("prompts", call)
            self.assertEqual(call["model"], "i2v")

    def test_image_paths_de_una_sola_foto_no_llega_al_motor(self) -> None:
        from ui import video_widget as vw

        fake = _FakeEngine(self.clip)
        worker = vw.VideoWorker(
            fake, {"model": "i2v", "image_paths": ["a.png"], "prompts": ["p1"]}
        )
        result = worker.work()
        self.assertEqual(result, str(self.clip))
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(fake.calls[0]["image_path"], "a.png")
        self.assertEqual(fake.calls[0]["prompt"], "p1")
        self.assertNotIn("image_paths", fake.calls[0])

    def test_t2v_envia_prompt_y_no_claves_de_secuencia(self) -> None:
        from ui import video_widget as vw

        fake = _FakeEngine(self.clip)
        worker = vw.VideoWorker(fake, self._params("t2v", prompt="un gato"))
        worker.work()
        self.assertEqual(fake.calls[0]["prompt"], "un gato")
        self.assertIsNone(fake.calls[0]["image_path"])
        self.assertNotIn("image_paths", fake.calls[0])


if __name__ == "__main__":
    unittest.main()
