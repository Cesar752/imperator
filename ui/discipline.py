"""Disciplina del vídeo compartida por Cuaderno, Crónica y Comprensión.

Una sola preferencia (en ``settings.json``) y un **bus de señal** para
sincronizar en vivo: cambiar la disciplina en una pestaña la cambia en las demás,
de modo que el flujo (Comprensión → Cuaderno → Crónica) habla siempre del mismo
deporte. Antes cada pestaña tenía su selector aislado y podían divergir.

Sin Qt en la lógica de preferencias: aquí sólo se envuelve ``PrefsStore`` y se
ofrecen ayudas para rellenar/aplicar el combo.
"""

from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QComboBox

from services import preferences as prefs
from services.sport_params import DEFAULT_SPORT, SPORT_PRESETS


class _DisciplineBus(QObject):
    """Señal compartida: emite la clave de disciplina elegida en cualquier pestaña."""

    changed = pyqtSignal(str)


#: Instancia única del bus (una por proceso).
BUS = _DisciplineBus()


def fill(combo: QComboBox) -> None:
    """Rellena el combo con las disciplinas disponibles (etiqueta + clave)."""
    for clave, (etiqueta, _primer) in SPORT_PRESETS.items():
        combo.addItem(etiqueta, clave)


def saved(prefs_store: prefs.PrefsStore) -> str:
    """Clave de disciplina guardada (o la de por defecto si no hay ninguna)."""
    return prefs_store.get_str(prefs.SPORT, DEFAULT_SPORT) or DEFAULT_SPORT


def apply(combo: QComboBox, clave: str) -> None:
    """Selecciona ``clave`` en el combo **sin reemitir** la señal de cambio."""
    indice = combo.findData(clave)
    if indice < 0 or indice == combo.currentIndex():
        return
    bloqueado = combo.blockSignals(True)
    combo.setCurrentIndex(indice)
    combo.blockSignals(bloqueado)
