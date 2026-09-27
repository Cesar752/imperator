"""Tests del foco de ChatWidget: la barra debe quedar lista para escribir.

Al enviar, el widget deshabilita la barra mientras el motor piensa; Qt le quita
el foco y, al rehabilitarla, no lo devuelve. Estos tests fijan el arreglo:
el cursor vuelve solo cuando termina el turno (respuesta, error o cancelación).
"""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtGui import QShowEvent  # noqa: E402

from ui import chat_widget as cw  # noqa: E402
from ui.chat_widget import ChatWidget  # noqa: E402


class _FakeEngine:
    """Motor de chat de mentira (no se llega a ejecutar la petición)."""

    def list_model_details(self) -> list:
        return []


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

    def test_restaura_el_foco_al_recibir_la_respuesta(self) -> None:
        barra = self.widget._input
        self.widget._set_busy(True)  # el motor está pensando
        with mock.patch.object(barra, "setFocus") as set_focus:
            self.widget._on_reply("hola")
        set_focus.assert_called_once()
        self.assertTrue(barra.isEnabled())

    def test_restaura_el_foco_al_cerrar_el_turno_por_error(self) -> None:
        barra = self.widget._input
        self.widget._set_busy(True)
        with mock.patch.object(barra, "setFocus") as set_focus:
            self.widget._close_turn("Error: se cayó Ollama")
        set_focus.assert_called_once()

    def test_restaura_el_foco_con_la_barra_ya_habilitada(self) -> None:
        barra = self.widget._input
        estados: list = []
        with mock.patch.object(
            barra, "setFocus", side_effect=lambda: estados.append(barra.isEnabled())
        ):
            self.widget._set_busy(True)
            self.widget._close_turn("Cancelado por el usuario")
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

    def _llenar_conversacion(self) -> None:
        self.widget.resize(240, 120)
        self.app.processEvents()
        for i in range(200):
            self.widget._conversation.appendPlainText(f"linea {i}")
        self.app.processEvents()

    def test_baja_al_final_aunque_el_usuario_haya_subido(self) -> None:
        self._llenar_conversacion()
        barra = self.widget._conversation.verticalScrollBar()
        self.assertGreater(barra.maximum(), 0, "la conversación debe desbordar")

        barra.setValue(0)  # el usuario se ha ido arriba a leer
        self.widget._append_line("Tú: mensaje nuevo")
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
