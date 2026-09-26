"""Tema oscuro de la aplicación (estilo Fusion + paleta oscura)."""

from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication


def apply_dark_theme(app: QApplication) -> None:
    """Aplica el tema oscuro global.

    Usa el estilo ``Fusion`` (el más consistente entre plataformas) y una
    paleta ``QPalette`` oscura, sin hojas de estilo hardcodeadas por widget.
    """
    app.setStyle("Fusion")

    window = QColor(53, 53, 53)
    base = QColor(25, 25, 25)
    alt_base = QColor(45, 45, 45)
    text = QColor(235, 235, 235)
    button = QColor(53, 53, 53)
    highlight = QColor(42, 130, 218)
    disabled_text = QColor(127, 127, 127)

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, window)
    palette.setColor(QPalette.ColorRole.WindowText, text)
    palette.setColor(QPalette.ColorRole.Base, base)
    palette.setColor(QPalette.ColorRole.AlternateBase, alt_base)
    palette.setColor(QPalette.ColorRole.Text, text)
    palette.setColor(QPalette.ColorRole.Button, button)
    palette.setColor(QPalette.ColorRole.ButtonText, text)
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Highlight, highlight)
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Link, highlight)
    palette.setColor(QPalette.ColorRole.ToolTipBase, alt_base)
    palette.setColor(QPalette.ColorRole.ToolTipText, text)
    palette.setColor(QPalette.ColorRole.PlaceholderText, disabled_text)
    palette.setColor(QPalette.ColorRole.Accent, highlight)

    disabled = QPalette.ColorGroup.Disabled
    palette.setColor(disabled, QPalette.ColorRole.WindowText, disabled_text)
    palette.setColor(disabled, QPalette.ColorRole.Text, disabled_text)
    palette.setColor(disabled, QPalette.ColorRole.ButtonText, disabled_text)
    palette.setColor(disabled, QPalette.ColorRole.Highlight, QColor(80, 80, 80))
    palette.setColor(disabled, QPalette.ColorRole.HighlightedText, disabled_text)

    app.setPalette(palette)


def apply_light_theme(app: QApplication) -> None:
    """Restaura el tema claro (Fusion + paleta estándar)."""
    app.setStyle("Fusion")
    app.setPalette(app.style().standardPalette())


def apply_theme(app: QApplication, theme: str) -> None:
    """Aplica el tema según el valor (``"dark"`` o ``"light"``)."""
    if theme == "light":
        apply_light_theme(app)
    else:
        apply_dark_theme(app)
