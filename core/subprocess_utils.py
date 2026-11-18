"""Lanzar subprocesos **sin que Windows abra una consola**.

Imperator corre con ``pythonw.exe`` (sin consola). En Windows, cada
``subprocess.run`` desde un proceso sin consola crea una **ventana de consola**
que parpadea en pantalla: el usuario lo describía como «una pantalla negra que se
esconde rápidamente, como si estuviera dando un comando a una terminal».

El caso más visible era ``nvidia-smi``: el planificador de GPU
(:mod:`core.gpu_scheduler`) lo sondea en **cada** reserva de GPU, así que el
parpadeo aparecía con solo enviar un mensaje en el chat. Lo mismo con ``ffmpeg``
(ensamblador, extracción de fotogramas/audio y sonda de duración de un audio).

Aquí vive la única forma de lanzarlos: ``**no_window_kwargs()``.
"""

from __future__ import annotations

import subprocess
import sys

#: ``CREATE_NO_WINDOW`` (Win32). ``subprocess`` solo lo expone en Windows.
_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)

#: ``SW_HIDE`` (Win32): no mostrar la ventana del proceso hijo.
_SW_HIDE = 0


def no_window_kwargs() -> dict:
    """``kwargs`` para ``subprocess`` que evitan la consola del hijo en Windows.

    Fuera de Windows devuelve un dict vacío (no hay consola que ocultar). En
    Windows añade ``creationflags=CREATE_NO_WINDOW`` y un ``startupinfo`` que pide
    ocultar la ventana; el *flag* ya basta, y ``startupinfo`` cubre además a quien
    lo ignore.
    """
    if sys.platform != "win32":
        return {}
    kwargs: dict = {"creationflags": _CREATE_NO_WINDOW}
    try:
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = _SW_HIDE
        kwargs["startupinfo"] = startupinfo
    except AttributeError:  # pragma: no cover - STARTUPINFO solo existe en Windows
        pass
    return kwargs
