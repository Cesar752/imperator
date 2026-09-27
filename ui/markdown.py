"""Renderizado del markdown que devuelve el motor en el chat.

``QTextDocument.setMarkdown()`` (Qt 6) importa el markdown, pero deja el código
sin tratar: ni familia monoespaciada en los bloques ni fondo que lo separe del
resto del texto. Este módulo aplica ese acabado con los colores del tema
(claro/oscuro) y devuelve HTML listo para ``QTextCursor.insertHtml()``.

El texto sin marcadores de markdown no pasa por aquí: se inserta tal cual, que
es más rápido y conserva el color del tema sin depender del HTML.
"""

import re

from PyQt6.QtGui import QColor, QPalette, QTextCharFormat, QTextCursor, QTextDocument

#: Marcadores que justifican el renderizado enriquecido. Los saltos de línea
#: cuentan: varias líneas ya son markdown de facto (párrafos, listas).
_MARKDOWN_MARKERS = re.compile(
    r"\n|`|\*\*|__|~~|\[[^\]\n]*\]\(|^#{1,6} "
    r"|^ *([-*_])\1{2,} *$|^ *[-*+] |^ *\d+[.)] |^ *> |^ *\|",
    re.MULTILINE,
)

#: Familias para el código, en orden de preferencia (Qt usa la primera que
#: exista en el sistema).
CODE_FAMILIES = ["Consolas", "Cascadia Mono", "Courier New", "monospace"]

#: Colores del código por tema: ``{es_oscuro: (fondo, texto)}``. El resto del
#: markdown hereda el color del widget, así que no se fija aquí.
CODE_COLORS = {
    True: ("#2b2b2b", "#e6e6e6"),
    False: ("#f0f0f0", "#1f1f1f"),
}


def is_dark(palette: QPalette) -> bool:
    """``True`` si la paleta corresponde a un tema oscuro.

    Se mira el fondo base (el del área de texto): casi negro en el tema oscuro
    de la app y blanco en el claro.
    """
    return palette.color(QPalette.ColorRole.Base).lightness() < 128


def has_markdown(text: str) -> bool:
    """``True`` si ``text`` trae algo que merece renderizado enriquecido."""
    return _MARKDOWN_MARKERS.search(text) is not None


def insert_markdown(cursor: QTextCursor, text: str, dark: bool) -> None:
    """Inserta ``text`` en ``cursor``: markdown renderizado o texto llano.

    En el caso llano se conserva el formato del cursor (el color del tema), que
    es lo que ya hacía el chat. En el enriquecido, Qt fusiona el primer bloque
    con el párrafo en curso, así que la etiqueta del autor sigue en su línea.
    """
    if not text:
        return
    if not has_markdown(text):
        cursor.insertText(text)
        return
    cursor.insertHtml(build_html(text, dark))


def build_html(text: str, dark: bool) -> str:
    """Convierte ``text`` (markdown) en el HTML que consume ``insertHtml``."""
    document = QTextDocument()
    document.setMarkdown(text)
    _style_code(document, dark)
    return document.toHtml()


# ---- Acabado del código ------------------------------------------


def _style_code(document: QTextDocument, dark: bool) -> None:
    """Aplica fondo y monoespaciado a los bloques y al código en línea."""
    background = QColor(CODE_COLORS[dark][0])
    foreground = QColor(CODE_COLORS[dark][1])
    _style_code_blocks(document, background, foreground)
    _style_inline_code(document, background, foreground)


def _code_format(background: QColor, foreground: QColor) -> QTextCharFormat:
    fmt = QTextCharFormat()
    fmt.setFontFamilies(CODE_FAMILIES)
    fmt.setBackground(background)
    fmt.setForeground(foreground)
    return fmt


def _style_code_blocks(
    document: QTextDocument, background: QColor, foreground: QColor
) -> None:
    """Los bloques de código (````` ``` `````) los marca Qt con ``nonBreakableLines``."""
    fmt = _code_format(background, foreground)
    block = document.begin()
    while block.isValid():
        if block.blockFormat().nonBreakableLines():
            cursor = QTextCursor(block)
            cursor.movePosition(QTextCursor.MoveOperation.StartOfBlock)
            cursor.movePosition(
                QTextCursor.MoveOperation.EndOfBlock,
                QTextCursor.MoveMode.KeepAnchor,
            )
            cursor.mergeCharFormat(fmt)
        block = block.next()


def _style_inline_code(
    document: QTextDocument, background: QColor, foreground: QColor
) -> None:
    """El código en línea (`` `x` ``) lo marca Qt con ``fontFixedPitch``."""
    fmt = _code_format(background, foreground)
    last = document.characterCount() - 1
    cursor = QTextCursor(document)
    start: int | None = None
    for pos in range(last + 1):
        fixed = pos < last and _is_fixed_pitch(cursor, pos)
        if fixed and start is None:
            start = pos
        elif not fixed and start is not None:
            _merge_range(document, start, pos, fmt)
            start = None
    if start is not None:
        _merge_range(document, start, last, fmt)


def _is_fixed_pitch(cursor: QTextCursor, pos: int) -> bool:
    cursor.setPosition(pos)
    cursor.setPosition(pos + 1, QTextCursor.MoveMode.KeepAnchor)
    return cursor.charFormat().fontFixedPitch()


def _merge_range(
    document: QTextDocument, start: int, end: int, fmt: QTextCharFormat
) -> None:
    cursor = QTextCursor(document)
    cursor.setPosition(start)
    cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
    cursor.mergeCharFormat(fmt)
