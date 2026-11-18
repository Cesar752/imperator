"""Tema claro legible: colores semánticos del tema y contraste WCAG AA.

Los colores de aviso/error/autor estaban fijos en el código: sobre el tema claro
quedaban casi ilegibles. Ahora salen de ``ui.theme.SEMANTIC``, uno por tema.
"""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtGui import QColor  # noqa: E402

from ui import theme  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

#: Colores fijos que había antes (si vuelven, el tema claro se rompe otra vez).
FIJOS_PROHIBIDOS = (
    "#4a9eff",
    "#69c07a",
    "#e06c75",
    "#d8a657",
    "#b58900",
    "#4e9a06",
    "#4d3a12",
)

#: Fondo sobre el que se pinta cada rol (Base del tema / fondo del aviso).
FONDO = {
    "dark": {"chat_user": "#191919", "chat_engine": "#191919", "chat_error": "#191919",
             "chat_warning": "#191919", "warning_fg": "#4d3a12", "ok_fg": "#191919"},
    "light": {"chat_user": "#ffffff", "chat_engine": "#ffffff", "chat_error": "#ffffff",
              "chat_warning": "#ffffff", "warning_fg": "#fff0c2", "ok_fg": "#ffffff"},
}

#: Mínimo de WCAG AA para texto normal.
MINIMO = 4.5


def _luminancia(color: str) -> float:
    """Luminancia relativa de un color sRGB (0..1), según WCAG."""
    c = QColor(color)
    canales = []
    for valor in (c.redF(), c.greenF(), c.blueF()):
        canales.append(valor / 12.92 if valor <= 0.03928 else ((valor + 0.055) / 1.055) ** 2.4)
    return 0.2126 * canales[0] + 0.7152 * canales[1] + 0.0722 * canales[2]


def _contraste(fg: str, bg: str) -> float:
    a, b = _luminancia(fg), _luminancia(bg)
    claro, oscuro = max(a, b), min(a, b)
    return (claro + 0.05) / (oscuro + 0.05)


class TemaTest(unittest.TestCase):
    def test_los_dos_temas_definen_los_mismos_roles(self) -> None:
        self.assertEqual(set(theme.SEMANTIC["dark"]), set(theme.SEMANTIC["light"]))

    def test_contraste_wcag_aa_en_ambos_temas(self) -> None:
        for nombre, colores in theme.SEMANTIC.items():
            for rol, color in colores.items():
                if rol == "warning_bg":  # es un fondo, no texto
                    continue
                fondo = FONDO[nombre][rol]
                ratio = _contraste(color, fondo)
                with self.subTest(tema=nombre, rol=rol):
                    self.assertGreaterEqual(
                        ratio,
                        MINIMO,
                        f"{color} sobre {fondo} da {ratio:.2f} (< {MINIMO})",
                    )

    def test_apply_theme_recuerda_el_tema(self) -> None:
        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])
        theme.apply_theme(app, "light")
        try:
            self.assertEqual(theme.current_theme(), "light")
            self.assertEqual(
                theme.semantic("warning_fg").name(), theme.SEMANTIC["light"]["warning_fg"]
            )
        finally:
            theme.apply_theme(app, "dark")
        self.assertEqual(theme.current_theme(), "dark")

    def test_no_quedan_colores_fijos_en_los_widgets_afectados(self) -> None:
        for nombre in ("chat_widget.py", "notes_widget.py", "video_widget.py"):
            fuente = (ROOT / "ui" / nombre).read_text(encoding="utf-8")
            for color in FIJOS_PROHIBIDOS:
                with self.subTest(archivo=nombre, color=color):
                    self.assertNotIn(
                        color, fuente, "el color debe salir de ui.theme.SEMANTIC"
                    )


if __name__ == "__main__":
    unittest.main()
