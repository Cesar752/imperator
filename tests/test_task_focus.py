"""El foco vuelve al campo de escritura al terminar cualquier tarea.

Al deshabilitar los controles mientras hay trabajo en curso, Qt retira el foco
y no lo devuelve al rehabilitarlos. Cada tarea declara su control principal en
``_focus_widget()`` y la base (:mod:`ui.task_widget`) lo enfoca al acabar.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from core.config_manager import ConfigManager  # noqa: E402
from ui import assembler_widget as asw  # noqa: E402
from ui import audio_widget as aw  # noqa: E402
from ui import chat_widget as cw  # noqa: E402
from ui import document_widget as docw  # noqa: E402
from ui import image_widget as iw  # noqa: E402
from ui import video_widget as vw  # noqa: E402

# Tareas con campo de escritura propio (chat: barra; audio: área de texto;
# imagen y vídeo: prompt). Documentos y Ensamblador empiezan eligiendo archivo.
_CON_CAMPO = ("chat", "audio", "image", "video")


class TaskFocusTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from PyQt6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.cm = ConfigManager(config_dir=self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _build(self, tarea: str):
        """Construye el widget sin motores y sin lanzar tareas de pool."""
        with mock.patch.object(cw, "run_pool_task"), mock.patch.object(
            docw, "run_pool_task"
        ), mock.patch.object(iw, "run_pool_task"), mock.patch.object(
            vw, "run_pool_task"
        ):
            if tarea == "chat":
                return cw.ChatWidget(None, self.cm)
            if tarea == "audio":
                return aw.AudioWidget(None, None, self.cm)
            if tarea == "image":
                return iw.ImageWidget({}, self.cm)
            if tarea == "video":
                return vw.VideoWidget(None, self.cm)
            if tarea == "document":
                return docw.DocumentWidget(None, self.cm)
            if tarea == "assembler":
                return asw.AssemblerWidget(None, self.cm)
            raise AssertionError(f"tarea desconocida: {tarea}")

    def test_el_chat_declara_la_barra_de_escritura(self) -> None:
        widget = self._build("chat")
        self.assertIs(widget._focus_widget(), widget._input)

    def test_el_audio_declara_su_area_de_texto(self) -> None:
        widget = self._build("audio")
        self.assertIs(widget._focus_widget(), widget._text_area)

    def test_imagen_y_video_declaran_el_prompt(self) -> None:
        imagen = self._build("image")
        video = self._build("video")
        self.assertIs(imagen._focus_widget(), imagen._prompt)
        self.assertIs(video._focus_widget(), video._prompt)

    def test_documentos_y_ensamblador_no_tienen_campo_principal(self) -> None:
        self.assertIsNone(self._build("document")._focus_widget())
        self.assertIsNone(self._build("assembler")._focus_widget())

    def test_al_terminar_la_tarea_el_campo_recupera_el_foco(self) -> None:
        for tarea in _CON_CAMPO:
            with self.subTest(tarea=tarea):
                widget = self._build(tarea)
                campo = widget._focus_widget()
                widget._set_busy(True)  # simula el trabajo en curso
                with mock.patch.object(campo, "setFocus") as set_focus:
                    widget._on_finished()
                set_focus.assert_called_once()
                self.assertTrue(campo.isEnabled(), "se enfoca ya rehabilitado")

    def test_no_enfoca_un_campo_deshabilitado(self) -> None:
        widget = self._build("chat")
        campo = widget._input
        campo.setEnabled(False)
        with mock.patch.object(campo, "setFocus") as set_focus:
            widget._restore_focus()
        set_focus.assert_not_called()

    def test_restore_focus_sin_campo_no_lanza(self) -> None:
        self._build("document")._restore_focus()

    def test_entrar_en_la_pestana_enfoca_el_campo(self) -> None:
        widget = self._build("video")
        campo = widget._focus_widget()
        with mock.patch.object(campo, "setFocus") as set_focus:
            widget.show()
        self.app.processEvents()
        set_focus.assert_called()


if __name__ == "__main__":
    unittest.main()
