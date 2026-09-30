"""Splash con un destello que barre las letras del logo.

El logo (assets/imperator.png) tiene fondo transparente; el brillo se recorta a
su canal alfa, de modo que el destello solo se ve sobre las letras/el dibujo y
no sobre el fondo.
"""

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QColor, QImage, QLinearGradient, QPainter, QPixmap
from PyQt6.QtWidgets import QSplashScreen


class ShimmerSplash(QSplashScreen):
    """Splash que muestra el logo con un brillo diagonal en movimiento.

    El brillo es una banda semitransparente que barre de izquierda a derecha de
    forma continua. Parámetros:
      - ``interval_ms``: cada cuánto se avanza la animación (ms).
      - ``speed``: avance de fase por tick (menor = más lento).
    """

    def __init__(
        self,
        pixmap: QPixmap,
        interval_ms: int = 30,
        speed: float = 0.02,
    ) -> None:
        super().__init__(
            pixmap,
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint,
        )
        self._base = pixmap
        # Fondo transparente: el logo flota y el brillo solo se ve sobre él.
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._phase = 0.0
        self._speed = speed
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)
        self._timer.start(interval_ms)

    def _advance(self) -> None:
        self._phase = (self._phase + self._speed) % 1.0
        self.update()

    def finish(self, widget) -> None:
        """Detiene la animación antes de cerrar el splash."""
        self._timer.stop()
        super().finish(widget)

    def paintEvent(self, event) -> None:  # noqa: N802 - override de Qt
        w = self.width()
        h = self.height()

        # 1) Base: el logo con su canal alfa sobre fondo transparente.
        frame = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
        frame.fill(Qt.GlobalColor.transparent)
        base = QPainter(frame)
        base.setRenderHint(QPainter.RenderHint.Antialiasing)
        base.drawPixmap(0, 0, self._base)
        base.end()

        # Banda diagonal y suave: transparente → blanco → transparente.
        x = self._phase * (w * 1.5) - w * 0.25
        grad = QLinearGradient(x - h, 0.0, x, float(h))
        grad.setColorAt(0.0, QColor(255, 255, 255, 0))
        grad.setColorAt(0.35, QColor(255, 255, 255, 0))
        grad.setColorAt(0.5, QColor(255, 255, 255, 210))
        grad.setColorAt(0.65, QColor(255, 255, 255, 0))
        grad.setColorAt(1.0, QColor(255, 255, 255, 0))

        # 2) Brillo recortado a la forma del logo (DestinationIn: el logo actúa
        #    de máscara sobre la banda blanca).
        shine = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
        shine.fill(Qt.GlobalColor.transparent)
        mask = QPainter(shine)
        mask.fillRect(0, 0, w, h, grad)
        mask.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationIn)
        mask.drawPixmap(0, 0, self._base)
        mask.end()

        # 3) Componer el brillo encima del logo.
        over = QPainter(frame)
        over.drawImage(0, 0, shine)
        over.end()

        # 4) Pintar el fotograma resultante.
        painter = QPainter(self)
        painter.drawImage(0, 0, frame)
        painter.end()
