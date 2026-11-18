"""Tema oscuro de la aplicación (estilo Fusion + paleta oscura)."""

from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication

#: Tema vigente (lo fija :func:`apply_theme`).
_CURRENT = {"name": "dark"}

#: Colores con semántica por tema. La ``QPalette`` no los cubre (avisos, errores,
#: autor del chat) y fijos rompían el contraste al cambiar de tema: el tema claro
#: dejaba los avisos casi ilegibles sobre fondo claro.
SEMANTIC: dict[str, dict[str, str]] = {
    "dark": {
        "chat_user": "#4a9eff",
        "chat_engine": "#69c07a",
        "chat_error": "#e06c75",
        "chat_warning": "#d8a657",
        "warning_fg": "#d8a657",
        "warning_bg": "#4d3a12",
        "ok_fg": "#69c07a",
    },
    "light": {
        "chat_user": "#1a5fb4",
        "chat_engine": "#1c7146",
        "chat_error": "#a51d2d",
        "chat_warning": "#8a5a00",
        "warning_fg": "#8a5a00",
        "warning_bg": "#fff0c2",
        "ok_fg": "#1c7146",
    },
}


def current_theme() -> str:
    """Nombre del tema aplicado (``"dark"`` o ``"light"``)."""
    return _CURRENT["name"]


def semantic(role: str) -> QColor:
    """Color con semántica (``warning_fg``, ``chat_user``…) del tema vigente.

    Falla con ``KeyError`` si el rol no existe: es un error de programación, no
    algo que deba degradarse en silencio a un color cualquiera.
    """
    return QColor(SEMANTIC[_CURRENT["name"]][role])




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
        _CURRENT["name"] = "light"
    else:
        apply_dark_theme(app)
        _CURRENT["name"] = "dark"
