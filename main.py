"""Punto de entrada de PlataformaIA.

Crea la QApplication, muestra el splash (si existe el logo) y abre la ventana
principal con su icono.
"""

import ctypes
import logging
import sys
from pathlib import Path

from PyQt6.QtCore import QEventLoop, Qt, QTimer
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import QApplication, QSplashScreen

from ui.main_window import MainWindow

logger = logging.getLogger(__name__)

# Raíz del proyecto (C:\Imperator); los assets viven en assets/.
ASSETS = Path(__file__).resolve().parent / "assets"

# Tiempo mínimo (ms) que el splash permanece visible.
SPLASH_MIN_MS = 1000


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def main() -> int:
    _configure_logging()
    if sys.platform == "win32":
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "imperator.plataformaia.1.0"
            )
        except Exception as exc:
            logger.warning("No se pudo fijar el AppUserModelID: %s", exc)
    app = QApplication(sys.argv)
    app.setApplicationName("PlataformaIA")

    icon_path = ASSETS / "icono.ico"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    else:
        logger.warning("Falta el icono: %s", icon_path)

    window = MainWindow()

    logo_path = ASSETS / "imperator.png"
    splash: QSplashScreen | None = None
    if logo_path.exists():
        pixmap = QPixmap(str(logo_path))
        if pixmap.isNull():
            logger.warning("El logo no se pudo cargar: %s", logo_path)
        else:
            splash = QSplashScreen(
                pixmap,
                Qt.WindowType.FramelessWindowHint | Qt.WindowType.SplashScreen,
            )
            splash.show()
            # Forzar el primer pintado antes de construir la ventana principal.
            app.processEvents()
    else:
        logger.warning("Falta el logo: %s", logo_path)

    if splash is not None:
        # Esperar al menos SPLASH_MIN_MS antes de cerrar el splash.
        loop = QEventLoop()
        QTimer.singleShot(SPLASH_MIN_MS, loop.quit)
        loop.exec()
        splash.finish(window)

    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
