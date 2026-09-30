"""Tests de la validación del formulario de motores (`ui.engines_dialog`).

`EngineFormDialog` es el sitio donde se teclean los datos de un motor: JSON de
parámetros incluido. Se cubre que no deje pasar un formulario incompleto ni
parámetros que no sean un objeto JSON, y que devuelva lo escrito tal cual.
Añadidos en la auditoría del 30/09/2026. No se construye `EnginesDialog` (lista):
al abrirse lanza sondeos de disponibilidad de los motores, que aquí no tocan red.
"""

import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QDialog  # noqa: E402

from ui import engines_dialog as ed  # noqa: E402


class EngineFormDialogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _dialog(self, **kwargs) -> ed.EngineFormDialog:
        dialog = ed.EngineFormDialog(**kwargs)
        self.addCleanup(dialog.close)
        return dialog

    def _aceptar(self, dialog) -> list[str]:
        """Llama a `_on_accept` sin abrir el QMessageBox modal."""
        avisos: list[str] = []
        with mock.patch.object(
            ed.QMessageBox,
            "warning",
            side_effect=lambda _p, _t, texto, *a, **k: avisos.append(texto),
        ):
            dialog._on_accept()
        return avisos

    # ---- Alta ------------------------------------------------------

    def test_al_abrir_se_rellena_la_url_del_tipo(self) -> None:
        dialog = self._dialog()
        tipo = dialog._engine_type()
        self.assertEqual(dialog._ruta_edit.text(), ed._TYPE_URLS[tipo])
        self.assertFalse(dialog._enabled_check.isChecked())

    def test_al_cambiar_de_tipo_se_refrescan_url_y_parametros(self) -> None:
        dialog = self._dialog()
        segundo = dialog._type_combo.itemText(1)
        dialog._type_combo.setCurrentIndex(1)

        self.assertEqual(dialog._ruta_edit.text(), ed._TYPE_URLS[segundo])
        self.assertEqual(
            json.loads(dialog._params_edit.toPlainText()),
            ed._TYPE_TEMPLATES[segundo],
        )

    def test_sin_nombre_no_acepta(self) -> None:
        dialog = self._dialog()
        dialog._nombre_edit.setText("   ")

        avisos = self._aceptar(dialog)
        self.assertEqual(avisos, ["El nombre no puede estar vacío."])
        self.assertNotEqual(dialog.result(), QDialog.DialogCode.Accepted)

    def test_sin_url_no_acepta(self) -> None:
        dialog = self._dialog()
        dialog._nombre_edit.setText("Mi motor")
        dialog._ruta_edit.setText("  ")

        avisos = self._aceptar(dialog)
        self.assertEqual(avisos, ["La URL (ruta) no puede estar vacía."])
        self.assertNotEqual(dialog.result(), QDialog.DialogCode.Accepted)

    def test_parametros_que_no_son_json_no_pasan(self) -> None:
        dialog = self._dialog()
        dialog._nombre_edit.setText("Mi motor")
        dialog._params_edit.setPlainText("{esto no es json")

        avisos = self._aceptar(dialog)
        self.assertEqual(len(avisos), 1)
        self.assertIn("no son JSON válido", avisos[0])
        self.assertNotEqual(dialog.result(), QDialog.DialogCode.Accepted)

    def test_parametros_que_no_son_objeto_no_pasan(self) -> None:
        dialog = self._dialog()
        dialog._nombre_edit.setText("Mi motor")
        dialog._params_edit.setPlainText("[1, 2]")

        avisos = self._aceptar(dialog)
        self.assertEqual(avisos, ["Los parámetros deben ser un objeto JSON ({...})."])

    def test_un_formulario_completo_se_acepta(self) -> None:
        dialog = self._dialog()
        dialog._nombre_edit.setText("  Mi motor  ")
        dialog._ruta_edit.setText(" http://127.0.0.1:1234 ")
        dialog._enabled_check.setChecked(True)
        dialog._params_edit.setPlainText('{"model": "qwen3.6:35b"}')

        self.assertEqual(self._aceptar(dialog), [])
        self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted)

        config = dialog.result_config()
        self.assertEqual(config["nombre"], "Mi motor")
        self.assertEqual(config["ruta"], "http://127.0.0.1:1234")
        self.assertTrue(config["enabled"])
        self.assertEqual(config["parametros"], {"model": "qwen3.6:35b"})
        self.assertEqual(config["id"], dialog._engine_type())

    def test_parametros_vacios_son_un_objeto_vacio(self) -> None:
        dialog = self._dialog()
        dialog._params_edit.setPlainText("   ")

        self.assertEqual(dialog._parsed_params(), {})
        self.assertEqual(dialog.result_config()["parametros"], {})

    # ---- Edición ---------------------------------------------------

    def test_en_edicion_prellena_y_fija_el_tipo(self) -> None:
        original = {
            "id": "ollama",
            "nombre": "Ollama",
            "ruta": "http://127.0.0.1:11434",
            "enabled": False,
            "parametros": {"num_ctx": 8192},
        }
        dialog = self._dialog(engine_config=original)

        self.assertTrue(dialog._is_edit)
        self.assertFalse(dialog._type_combo.isEnabled())
        self.assertEqual(dialog._nombre_edit.text(), "Ollama")
        self.assertEqual(dialog._parsed_params(), {"num_ctx": 8192})
        self.assertFalse(dialog._enabled_check.isChecked())
        self.assertEqual(dialog.result_config()["parametros"], {"num_ctx": 8192})

    def test_en_edicion_cambiar_de_tipo_no_pisa_los_parametros(self) -> None:
        dialog = self._dialog(
            engine_config={
                "id": "comfyui",
                "nombre": "ComfyUI",
                "ruta": "http://127.0.0.1:8188",
                "enabled": True,
                "parametros": {"default_model": "flux_dev"},
            }
        )
        dialog._on_type_changed()

        self.assertEqual(dialog._parsed_params(), {"default_model": "flux_dev"})
        self.assertEqual(dialog._ruta_edit.text(), "http://127.0.0.1:8188")


if __name__ == "__main__":
    unittest.main()
