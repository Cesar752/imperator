"""Punto de entrada de Imperator.

Crea la QApplication, muestra el splash (si existe el logo) y abre la ventana
principal con su icono.
"""

import ctypes
import logging
import logging.handlers
import sys
import time
from copy import deepcopy
from pathlib import Path

from PyQt6.QtCore import QEventLoop, QTimer
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import QApplication

from core.config_manager import DEFAULT_SETTINGS, ConfigError, ConfigManager
from core.paths import ASSETS, PROJECT_ROOT
from ui.shimmer_splash import ShimmerSplash
from ui.theme import apply_theme

logger = logging.getLogger(__name__)

# Tiempo mínimo (ms) que el splash permanece visible.
SPLASH_MIN_MS = 1000


def _load_settings_safe() -> dict:
    """Carga settings.json; si está corrupto, avisa y usa los valores por defecto.

    Un settings.json inválido no debe impedir que la app arranque.
    """
    try:
        return ConfigManager().load_settings()
    except ConfigError as exc:
        # Aún no hay logging configurado; el mensaje se verá tras configurarlo.
        print(f"[Imperator] settings.json inválido ({exc}); se usan los defaults.")
        return deepcopy(DEFAULT_SETTINGS)


def _configure_logging(settings: dict) -> None:
    """Configura el logging: nivel desde settings y log a archivo si procede.

    ``settings.log_level`` fija el nivel y ``settings.rutas.logs`` (opcional)
    activa un log rotado a diario en ``<logs>/plataformaia.log``.
    """
    level_name = str(settings.get("log_level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = "%(asctime)s %(levelname)s [%(name)s] %(message)s"

    handlers: list[logging.Handler] = [logging.StreamHandler()]

    logs_dir = (settings.get("rutas") or {}).get("logs")
    if logs_dir:
        try:
            path = Path(logs_dir)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            path.mkdir(parents=True, exist_ok=True)
            handlers.append(
                logging.handlers.TimedRotatingFileHandler(
                    path / "plataformaia.log",
                    when="midnight",
                    backupCount=7,
                    encoding="utf-8",
                )
            )
        except OSError as exc:
            print(f"[Imperator] No se pudo abrir el log en '{logs_dir}': {exc}")

    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)


def _create_splash() -> ShimmerSplash | None:
    """Crea el splash con el logo 'imperator' y su efecto de brillo."""
    logo_path = ASSETS / "imperator.png"
    if not logo_path.exists():
        logger.warning("Falta el logo: %s", logo_path)
        return None
    pixmap = QPixmap(str(logo_path))
    if pixmap.isNull():
        logger.warning("El logo no se pudo cargar: %s", logo_path)
        return None
    return ShimmerSplash(pixmap)


def _force_taskbar_icon(window) -> None:
    """Fuerza el icono de la barra de tareas vía Win32 (``WM_SETICON``).

    Qt 6 en Windows a veces no propaga ``setWindowIcon`` al HWND nativo y la
    barra de tareas se queda con el icono del ejecutable (``pythonw.exe`` = el
    de Python). Aquí se envía ``WM_SETICON`` a mano usando ``assets/icono.ico``
    (ya regenerado en DIB, que es lo que pinta el shell).
    """
    if sys.platform != "win32":
        return
    icon_path = ASSETS / "icono.ico"
    if not icon_path.exists():
        return

    from ctypes import wintypes

    user32 = ctypes.windll.user32
    user32.LoadImageW.argtypes = [
        wintypes.HINSTANCE,
        wintypes.LPCWSTR,
        wintypes.UINT,
        ctypes.c_int,
        ctypes.c_int,
        wintypes.UINT,
    ]
    user32.LoadImageW.restype = wintypes.HANDLE
    user32.SendMessageW.argtypes = [
        wintypes.HWND,
        wintypes.UINT,
        wintypes.WPARAM,
        wintypes.LPARAM,
    ]
    user32.SendMessageW.restype = wintypes.LPARAM

    IMAGE_ICON = 1
    LR_LOADFROMFILE = 0x00000010
    ICON_SMALL = 0
    ICON_BIG = 1
    WM_SETICON = 0x0080

    hwnd = int(window.winId())
    for flag in (ICON_BIG, ICON_SMALL):
        hicon = user32.LoadImageW(
            None, str(icon_path), IMAGE_ICON, 0, 0, LR_LOADFROMFILE
        )
        if hicon:
            user32.SendMessageW(hwnd, WM_SETICON, flag, hicon)


def main() -> int:
    settings = _load_settings_safe()
    _configure_logging(settings)

    # Fijar un AppUserModelID estable para que Windows asocie la app (y su icono)
    # a una identidad propia en vez de la de pythonw.exe. Para que la barra de
    # tareas muestre el icono hace falta, además, un acceso directo (.lnk) que
    # declare este mismo AUMID (ver tools/create_shortcut.py).
    if sys.platform == "win32":
        try:
            set_app_id = ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID
            set_app_id.argtypes = [ctypes.c_wchar_p]
            set_app_id("imperator.plataformaia.1.0")
        except Exception as exc:  # noqa: BLE001 - el AUMID no debe romper el arranque
            logger.warning("No se pudo fijar el AppUserModelID: %s", exc)

    app = QApplication(sys.argv)
    app.setApplicationName("Imperator")
    apply_theme(app, settings.get("theme", "dark"))
    font = app.font()
    font.setPointSize(int(settings.get("font_size", 11)))
    app.setFont(font)

    icon_path = ASSETS / "icono.ico"
    if not icon_path.exists():
        logger.warning("Falta el icono: %s", icon_path)
    else:
        icon = QIcon(str(icon_path))
        # Un .ico con las entradas pequeñas en PNG carga en Qt (se ve en la
        # ventana) pero Windows lo pinta en blanco en la barra de tareas:
        # tools/make_icon.py las escribe en DIB.
        if icon.isNull() or not icon.availableSizes():
            logger.warning("El icono no tiene imágenes utilizables: %s", icon_path)
        app.setWindowIcon(icon)

    # Mostrar el splash ANTES del trabajo pesado. Construir MainWindow carga
    # motores y widgets (la parte lenta del arranque); si el splash se muestra
    # después, el logo "imperator" aparece justo al final, cuando ya está a
    # punto de abrirse la ventana.
    splash = _create_splash()
    start = time.monotonic()
    if splash is not None:
        splash.show()
        # Pintar el splash ya, antes de bloquear la UI con MainWindow().
        app.processEvents()

    # Importar MainWindow aquí (no arriba): su árbol de imports incluye los
    # motores con librerías pesadas (kokoro→torch, piper→onnxruntime,
    # faster-whisper→ctranslate2), que tardan varios segundos. Al importarlo
    # después de mostrar el splash, ese tiempo no retrasa la aparición del logo.
    from ui.main_window import MainWindow

    window = MainWindow()

    # Forzar el icono nativo de la barra de tareas (Qt a veces no lo propaga).
    _force_taskbar_icon(window)

    if splash is not None:
        # Garantizar el tiempo mínimo visible del splash (evita que parpadee si
        # la ventana se construye muy rápido).
        remaining_ms = SPLASH_MIN_MS - int((time.monotonic() - start) * 1000)
        if remaining_ms > 0:
            loop = QEventLoop()
            QTimer.singleShot(remaining_ms, loop.quit)
            loop.exec()
        splash.finish(window)

    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
