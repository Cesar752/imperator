"""Tests de ChatWidget: foco de la barra y formato de los turnos.

Cubre dos arreglos de usabilidad:
- la barra recupera el foco al terminar el turno (bloquearla lo perdía);
- cada turno se ve distingue: autor resaltado y turnos separados, para que no
  se mezcle lo que dice el humano con lo que responde el motor.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtGui import QFont, QShowEvent  # noqa: E402

from engines.base_engine import CANCEL_MESSAGE  # noqa: E402
from ui import chat_widget as cw  # noqa: E402
from ui.chat_widget import ChatWidget  # noqa: E402


class _FakeEngine:
    """Motor de chat de mentira (no se llega a ejecutar la petición)."""

    def list_model_details(self) -> list:
        return []


class _FakeEngineConNombre(_FakeEngine):
    """Motor con nombre legible, para el caso en que no hay modelo elegido."""

    nombre = "Ollama"


#: Modelo de prueba que se añade al combo para poder etiquetar las respuestas.
_MODELO = "qwen3.6:35b"


class ChatFocusTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from PyQt6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        with mock.patch.object(cw, "run_pool_task"):
            self.widget = ChatWidget(engine=_FakeEngine())
        self.widget.show()

    def tearDown(self) -> None:
        self.widget.close()

    # ---- Foco ------------------------------------------------------

    def test_restaura_el_foco_al_recibir_la_respuesta(self) -> None:
        barra = self.widget._input
        self.widget._set_busy(True)  # el motor está pensando
        with mock.patch.object(barra, "setFocus") as set_focus:
            self.widget._on_reply("hola")
        set_focus.assert_called_once()
        self.assertTrue(barra.isEnabled())

    def test_restaura_el_foco_al_fallar(self) -> None:
        barra = self.widget._input
        self.widget._set_busy(True)
        with mock.patch.object(barra, "setFocus") as set_focus:
            self.widget._on_failed("se cayó Ollama")
        set_focus.assert_called_once()

    def test_restaura_el_foco_al_cancelar(self) -> None:
        barra = self.widget._input
        self.widget._set_busy(True)
        with mock.patch.object(barra, "setFocus") as set_focus:
            self.widget._on_cancelled(CANCEL_MESSAGE)
        set_focus.assert_called_once()

    def test_restaura_el_foco_con_la_barra_ya_habilitada(self) -> None:
        barra = self.widget._input
        estados: list = []
        with mock.patch.object(
            barra, "setFocus", side_effect=lambda: estados.append(barra.isEnabled())
        ):
            self.widget._set_busy(True)
            self.widget._on_failed("boom")
        self.assertEqual(estados, [True], "se enfoca después de rehabilitarla")

    def test_no_roba_el_foco_mientras_esta_bloqueada(self) -> None:
        barra = self.widget._input
        self.widget._set_busy(True)
        with mock.patch.object(barra, "setFocus") as set_focus:
            self.widget._restore_focus()
        set_focus.assert_not_called()

    def test_declara_la_barra_como_control_principal(self) -> None:
        self.assertIs(self.widget._focus_widget(), self.widget._input)

    def test_al_mostrar_la_pestana_prepara_la_barra(self) -> None:
        with mock.patch.object(self.widget, "_restore_focus") as focus:
            self.widget.showEvent(QShowEvent())
        focus.assert_called_once()

    # ---- Envío -----------------------------------------------------

    def test_al_enviar_limpia_la_barra_y_manda_el_historial(self) -> None:
        self.widget._input.setText("  hola  ")
        with mock.patch.object(cw.ChatWorker, "start") as start:
            self.widget._send_message()
        self.assertEqual(self.widget._input.text(), "")
        self.assertEqual(self.widget._messages, [{"role": "user", "content": "hola"}])
        start.assert_called_once()

    def test_no_envia_texto_vacio(self) -> None:
        self.widget._input.setText("   ")
        with mock.patch.object(cw.ChatWorker, "start") as start:
            self.widget._send_message()
        start.assert_not_called()
        self.assertEqual(self.widget._messages, [])

    def test_sin_motor_avisa_en_la_conversacion(self) -> None:
        with mock.patch.object(cw, "run_pool_task"):
            widget = ChatWidget(engine=None)
        widget._input.setText("hola")
        with mock.patch.object(cw.ChatWorker, "start") as start:
            widget._send_message()
        start.assert_not_called()
        self.assertIn("Error: no hay motor activo.", widget._conversation.toPlainText())
        widget.close()

    # ---- Quién dice qué --------------------------------------------

    def _formato_de(self, aguja: str):
        return self.widget._conversation.document().find(aguja).charFormat()

    def _selecciona_modelo(self, modelo: str) -> None:
        """Añade el modelo al combo y lo selecciona.

        Al construir el widget con el pool de tareas neutralizado, el combo se
        queda con el placeholder "Cargando modelos…" (``currentData()`` None).
        """
        self.widget._model.addItem(modelo, modelo)
        self.widget._model.setCurrentIndex(self.widget._model.count() - 1)

    def _dos_turnos(self) -> None:
        self._selecciona_modelo(_MODELO)
        self.widget._input.setText("hola")
        with mock.patch.object(cw.ChatWorker, "start"):
            self.widget._send_message()
        self.widget._on_reply("respuesta del motor")

    def test_cada_turno_lleva_su_autor(self) -> None:
        self._dos_turnos()
        texto = self.widget._conversation.toPlainText()
        self.assertIn("Tú: hola", texto)
        self.assertIn(f"{_MODELO}: respuesta del motor", texto)

    def test_la_etiqueta_es_el_modelo_que_respondio(self) -> None:
        self._dos_turnos()
        self.assertIn(f"{_MODELO}:", self.widget._conversation.toPlainText())

    def test_usa_el_modelo_enviado_aunque_cambie_el_combo(self) -> None:
        self._selecciona_modelo(_MODELO)
        self.widget._input.setText("hola")
        with mock.patch.object(cw.ChatWorker, "start"):
            self.widget._send_message()

        # El usuario cambia de modelo mientras el motor responde.
        with mock.patch.object(
            cw.QMessageBox, "question", return_value=cw.QMessageBox.StandardButton.No
        ):
            self.widget._model.addItem("llama3:8b", "llama3:8b")
            self.widget._model.setCurrentIndex(self.widget._model.count() - 1)
        self.widget._on_reply("respuesta")

        texto = self.widget._conversation.toPlainText()
        self.assertIn(f"{_MODELO}: respuesta", texto)
        self.assertNotIn("llama3:8b:", texto)

    def test_sin_modelo_elegido_usa_el_nombre_del_motor(self) -> None:
        with mock.patch.object(cw, "run_pool_task"):
            widget = ChatWidget(engine=_FakeEngineConNombre())
        widget._input.setText("hola")
        with mock.patch.object(cw.ChatWorker, "start"):
            widget._send_message()
        widget._on_reply("respuesta")
        self.assertIn("Ollama: respuesta", widget._conversation.toPlainText())
        widget.close()

    def test_el_autor_va_resaltado_con_su_color(self) -> None:
        self._dos_turnos()
        usuario = self._formato_de("Tú:")
        motor = self._formato_de(f"{_MODELO}:")
        self.assertEqual(usuario.foreground().color().name(), cw._COLOR_USUARIO)
        self.assertEqual(motor.foreground().color().name(), cw._COLOR_MOTOR)
        negrita = int(QFont.Weight.Bold)
        self.assertEqual(int(usuario.fontWeight()), negrita)
        self.assertEqual(int(motor.fontWeight()), negrita)

    def test_el_cuerpo_conserva_el_color_del_tema(self) -> None:
        self._dos_turnos()
        cuerpo = self._formato_de("respuesta del motor")
        self.assertEqual(cuerpo.foreground().style(), Qt.BrushStyle.NoBrush)

    def test_los_turnos_se_separan_con_linea_en_blanco(self) -> None:
        self._dos_turnos()
        self.assertIn(f"\n\n{_MODELO}:", self.widget._conversation.toPlainText())

    def test_el_error_tambien_se_etiqueta(self) -> None:
        self.widget._on_failed("se cayó Ollama")
        texto = self.widget._conversation.toPlainText()
        self.assertIn("Error: se cayó Ollama", texto)
        self.assertEqual(
            self._formato_de("Error:").foreground().color().name(), cw._COLOR_ERROR
        )

    def test_una_respuesta_vacia_avisa_y_no_ensucia_el_historial(self) -> None:
        self.widget._input.setText("hola")
        with mock.patch.object(cw.ChatWorker, "start"):
            self.widget._send_message()

        self.widget._on_reply("   ")

        texto = self.widget._conversation.toPlainText()
        self.assertIn("Aviso: el motor devolvió una respuesta vacía.", texto)
        self.assertNotIn(f"{_MODELO}:", texto)
        self.assertEqual(len(self.widget._messages), 1)
        self.assertEqual(self.widget._messages[0]["role"], "user")
        self.assertTrue(self.widget._input.isEnabled())

    # ---- Desplazamiento --------------------------------------------

    def _llenar_conversacion(self) -> None:
        self.widget.resize(240, 120)
        self.app.processEvents()
        for i in range(200):
            self.widget._append_turn("Tú", f"linea {i}", cw._COLOR_USUARIO)
        self.app.processEvents()

    def test_baja_al_final_aunque_el_usuario_haya_subido(self) -> None:
        self._llenar_conversacion()
        barra = self.widget._conversation.verticalScrollBar()
        self.assertGreater(barra.maximum(), 0, "la conversación debe desbordar")

        barra.setValue(0)  # el usuario se ha ido arriba a leer
        self.widget._append_turn("Tú", "mensaje nuevo", cw._COLOR_USUARIO)
        self.app.processEvents()

        self.assertEqual(barra.value(), barra.maximum())

    def test_el_mensaje_del_humano_queda_visible(self) -> None:
        self._llenar_conversacion()
        barra = self.widget._conversation.verticalScrollBar()
        barra.setValue(0)

        self.widget._input.setText("hola")
        with mock.patch.object(cw.ChatWorker, "start"):
            self.widget._send_message()
        self.app.processEvents()

        self.assertEqual(barra.value(), barra.maximum())
        self.assertIn("Tú: hola", self.widget._conversation.toPlainText())


if __name__ == "__main__":
    unittest.main()
