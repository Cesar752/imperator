"""Tests del markdown del chat: detección, acabado del código y renderizado.

El motor devuelve markdown en crudo. El chat lo pinta con ``QTextDocument``
(``ui.markdown``), pero Qt deja el código sin fondo ni monoespaciado: eso es lo
que se comprueba aquí, además de que el texto llano sigue siendo texto llano.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtGui import QColor, QFont, QPalette, QTextDocument, QTextTable  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from ui import chat_widget as cw  # noqa: E402
from ui import markdown as md  # noqa: E402
from ui import theme  # noqa: E402
from ui.chat_widget import ChatWidget  # noqa: E402


class _FakeEngine:
    """Motor de mentira: el widget solo pide el listado de modelos."""

    def list_model_details(self) -> list:
        return []


class _QtTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])


class MarkdownTest(_QtTestCase):
    # ---- Detección -------------------------------------------------

    def test_el_texto_llano_no_se_trata_como_markdown(self) -> None:
        for texto in ("", "hola mundo", "Total: 3.5 euros", "a * b", "2 + 2 = 4"):
            with self.subTest(texto=texto):
                self.assertFalse(md.has_markdown(texto))

    def test_detecta_los_marcadores_habituales(self) -> None:
        casos = (
            "**negrita**",
            "| a | b |",
            "- punto",
            "* punto",
            "1. uno",
            "# Titulo",
            "> cita",
            "`codigo`",
            "un\nsalto",
            "[enlace](http://x)",
            "---",
        )
        for texto in casos:
            with self.subTest(texto=texto):
                self.assertTrue(md.has_markdown(texto))

    def test_is_dark_por_el_fondo_base(self) -> None:
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
        self.assertTrue(md.is_dark(palette))
        palette.setColor(QPalette.ColorRole.Base, QColor(255, 255, 255))
        self.assertFalse(md.is_dark(palette))

    # ---- Acabado del código ---------------------------------------

    def _documento(self, texto: str, dark: bool) -> QTextDocument:
        """Reconstruye el documento desde el HTML que inserta el chat."""
        documento = QTextDocument()
        documento.setHtml(md.build_html(texto, dark))
        return documento

    def test_bloque_de_codigo_lleva_fondo_y_fuente_monoespaciada(self) -> None:
        for dark in (True, False):
            with self.subTest(dark=dark):
                html = md.build_html("```python\nprint(1)\n```", dark)
                self.assertIn(md.CODE_FAMILIES[0], html)
                documento = self._documento("```python\nprint(1)\n```", dark)
                fmt = documento.find("print(1)").charFormat()
                esperado = md.CODE_COLORS[dark][0]
                self.assertEqual(
                    fmt.background().color().name().lower(), esperado.lower()
                )
                self.assertIn(md.CODE_FAMILIES[0], list(fmt.fontFamilies() or []))

    def test_codigo_en_linea_se_marca_monoespaciado(self) -> None:
        html = md.build_html("usa `pip install x`", dark=True)
        self.assertIn(md.CODE_FAMILIES[0], html)

        documento = self._documento("usa `pip install x`", dark=True)
        fmt = documento.find("pip install x").charFormat()
        self.assertIn(md.CODE_FAMILIES[0], list(fmt.fontFamilies() or []))
        self.assertEqual(
            fmt.background().color().name().lower(),
            md.CODE_COLORS[True][0].lower(),
        )

    def test_la_negrita_no_se_pinta_como_codigo(self) -> None:
        documento = self._documento("**solo negrita**", dark=True)
        fmt = documento.find("solo negrita").charFormat()
        self.assertEqual(int(fmt.fontWeight()), int(QFont.Weight.Bold))
        self.assertFalse(fmt.fontFixedPitch())

    def test_la_tabla_sigue_siendo_una_tabla(self) -> None:
        documento = self._documento("| a | b |\n| - | - |\n| 1 | 2 |", dark=True)
        tablas = [
            f for f in documento.rootFrame().childFrames() if isinstance(f, QTextTable)
        ]
        self.assertEqual(len(tablas), 1)


class ChatMarkdownTest(_QtTestCase):
    def setUp(self) -> None:
        with mock.patch.object(cw, "run_pool_task"):
            self.widget = ChatWidget(engine=_FakeEngine())
        self.widget.show()

    def tearDown(self) -> None:
        self.widget.close()

    def _texto(self) -> str:
        return self.widget._conversation.toPlainText()

    def _formato(self, aguja: str):
        return self.widget._conversation.document().find(aguja).charFormat()

    def test_la_respuesta_con_markdown_se_renderiza(self) -> None:
        self.widget._append_turn("Motor", "**muy** importante", cw._COLOR_MOTOR)
        self.assertIn("muy importante", self._texto())
        self.assertNotIn("**", self._texto())
        self.assertEqual(int(self._formato("muy").fontWeight()), int(QFont.Weight.Bold))

    def test_la_tabla_deja_de_verse_en_crudo(self) -> None:
        self.widget._append_turn(
            "Motor", "| a | b |\n| - | - |\n| 1 | 2 |", cw._COLOR_MOTOR
        )
        self.assertNotIn("|", self._texto())
        documento = self.widget._conversation.document()
        tablas = [
            f for f in documento.rootFrame().childFrames() if isinstance(f, QTextTable)
        ]
        self.assertEqual(len(tablas), 1)
        self.assertIn("1", self._texto())

    def test_el_codigo_usa_los_colores_del_tema_oscuro(self) -> None:
        palette = self.widget.palette()
        palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
        self.widget.setPalette(palette)

        self.widget._append_turn("Motor", "```\nprint(1)\n```", cw._COLOR_MOTOR)

        fondo = self._formato("print(1)").background().color().name()
        self.assertEqual(fondo.lower(), md.CODE_COLORS[True][0].lower())

    def test_el_texto_llano_conserva_el_color_del_tema(self) -> None:
        self.widget._append_turn("Motor", "respuesta plana", cw._COLOR_MOTOR)
        self.assertIn("Motor: respuesta plana", self._texto())
        self.assertEqual(
            self._formato("respuesta plana").foreground().style(),
            Qt.BrushStyle.NoBrush,
        )

    def test_la_etiqueta_del_autor_sigue_resaltada(self) -> None:
        self.widget._append_turn("Motor", "cuerpo", cw._COLOR_MOTOR)
        self.assertEqual(
            self._formato("Motor:").foreground().color().name(),
            theme.semantic(cw._COLOR_MOTOR).name(),
        )


if __name__ == "__main__":
    unittest.main()
