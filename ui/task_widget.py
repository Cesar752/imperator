"""Base común de los widgets de tarea (P4 de la auditoría).

Los seis widgets de la app (chat, documentos, imagen, vídeo, audio y
ensamblador) repetían el mismo esqueleto: ``_set_status``, ``_set_busy``,
``_on_error`` comparando el mensaje de cancelación y ``_on_finished`` liberando
el worker. Aquí se escribe una sola vez; cada widget declara qué controles se
bloquean y, si lo necesita, afina el resto con los hooks.
"""

import time

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QShowEvent
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from engines.base_engine import CANCEL_MESSAGE


class TaskWidget(QWidget):
    """Widget de tarea con estado y ciclo de vida del ``ThreadWorker``.

    Contrato para las subclases:

    - ``_busy_widgets()``: controles que se deshabilitan mientras hay trabajo.
    - ``_busy_only_widgets()``: controles que solo se habilitan durante el
      trabajo (botones de Cancelar).
    - ``_on_busy_changed(busy)``: reglas extra (p. ej. Guardar depende del
      resultado, no solo del estado ocupado).
    - ``_on_cancelled()`` / ``_on_failed(message)``: qué hacer al terminar mal.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._worker = None

    # ---- Estado ----------------------------------------------------

    def _set_status(self, text: str) -> None:
        """Muestra ``text`` en la etiqueta de estado (si la tarea tiene una).

        El chat no usa etiqueta de estado (escribe en la conversación), así que
        aquí se tolera su ausencia en lugar de obligar a cada widget a tener una.
        """
        status = getattr(self, "_status", None)
        if status is not None:
            status.setText(text)

    def _set_busy(self, busy: bool) -> None:
        """Bloquea o desbloquea los controles de la tarea."""
        for widget in self._busy_widgets():
            widget.setEnabled(not busy)
        for widget in self._busy_only_widgets():
            widget.setEnabled(busy)
        self._on_busy_changed(busy)

    def _busy_widgets(self) -> tuple:
        """Controles que se deshabilitan mientras hay trabajo en curso."""
        return ()

    def _busy_only_widgets(self) -> tuple:
        """Controles que solo están habilitados mientras hay trabajo en curso."""
        return ()

    def _on_busy_changed(self, busy: bool) -> None:
        """Hook para reglas de habilitación propias de cada tarea."""

    # ---- Avance ----------------------------------------------------

    def _montar_contenido_scroll(self) -> QVBoxLayout:
        """Instala un contenedor **desplazable** en ``self`` y devuelve su layout.

        Las pestañas con muchos controles (Vídeo IA, Crónica…) apilaban todo el
        formulario directamente sobre el widget de la pestaña. Cuando el alto
        superaba el de la ventana, Qt recortaba la parte de abajo **sin barra de
        desplazamiento**: los campos «Del… al… Repeticiones», la caja de rótulos,
        el botón «Generar crónica» y la propia barra de progreso quedaban fuera
        de la vista y no había forma de alcanzarlos. Con el scroll, todo el
        contenido es siempre accesible.

        Devuelve el ``QVBoxLayout`` sobre el que hay que construir los controles
        (en vez de hacerlo sobre ``self``).
        """
        contenedor = QWidget()
        interior = QVBoxLayout(contenedor)
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.Shape.NoFrame)
        area.setWidget(contenedor)
        exterior = QVBoxLayout(self)
        exterior.setContentsMargins(0, 0, 0, 0)
        exterior.addWidget(area)
        self._scroll = area
        return interior

    def _montar_progreso(self, layout) -> None:
        """Añade la barra de avance y la fila «estado · tiempo» al ``layout`` dado.

        Crea ``self._barra``, ``self._status`` (si la tarea no traía uno) y
        ``self._tiempo``. Las tareas que ya usan su propia ``_progress`` (Vídeo,
        Documentos, Ensamblador) no llaman a esto y se quedan como están.
        """
        self._barra = QProgressBar()
        self._barra.setRange(0, 0)      # indeterminada hasta que haya dato real
        self._barra.setFormat("%p%")
        layout.addWidget(self._barra)
        if getattr(self, "_status", None) is None:
            self._status = QLabel("")
        self._tiempo = QLabel("")
        fila = QHBoxLayout()
        fila.addWidget(self._status, 1)
        fila.addWidget(self._tiempo)
        layout.addLayout(fila)
        self._t0 = 0.0
        self._reloj = QTimer(self)
        self._reloj.setInterval(1000)
        self._reloj.timeout.connect(self._tic)

    def _empezar_progreso(self) -> None:
        """Deja la barra en indeterminado y arranca el reloj (al lanzar un trabajo)."""
        self._barra.setRange(0, 0)
        self._t0 = time.monotonic()
        self._tiempo.setText("Tiempo: 0:00")
        self._reloj.start()

    def _vaciar_barra(self) -> None:
        """Devuelve la barra a 0 tras un fallo o una cancelación (si la usa)."""
        barra = getattr(self, "_barra", None)
        if barra is not None:
            barra.setRange(0, 100)
            barra.setValue(0)

    def _on_progress(self, porcentaje: int, mensaje: str) -> None:
        """Refresca la barra y el mensaje con lo que informa el worker.

        Un porcentaje negativo es «aún no sé cuánto queda»: la barra se pone en modo
        indeterminado en vez de mentir con un 0 %.
        """
        valor = int(porcentaje)
        if valor < 0:
            self._barra.setRange(0, 0)
        else:
            self._barra.setRange(0, 100)
            self._barra.setValue(max(0, min(100, valor)))
        self._set_status(mensaje)

    def _tic(self) -> None:
        """Refresca el tiempo transcurrido, para ver que el trabajo sigue vivo."""
        if not self._t0:
            return
        segundos = int(time.monotonic() - self._t0)
        self._tiempo.setText(f"Tiempo: {segundos // 60}:{segundos % 60:02d}")

    # ---- Ciclo de vida del worker ----------------------------------

    def _on_error(self, message: str) -> None:
        """Distingue la cancelación (no es un fallo) del error real."""
        if message == CANCEL_MESSAGE:
            self._on_cancelled(message)
        else:
            self._on_failed(message)

    def _on_cancelled(self, message: str) -> None:
        """Hook: el usuario canceló la tarea."""
        self._vaciar_barra()
        self._set_status(message)

    def _on_failed(self, message: str) -> None:
        """Hook: la tarea falló por un error real."""
        self._vaciar_barra()
        self._set_status(f"Error: {message}")

    def _on_finished(self) -> None:
        """Libera el worker al terminar (éxito, error o cancelación)."""
        reloj = getattr(self, "_reloj", None)
        if reloj is not None:
            reloj.stop()
        self._set_busy(False)
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
        self._restore_focus()

    # ---- GPU -------------------------------------------------------

    def _gpu_scheduler(self):
        """Planificador de GPU de la ventana principal (``None`` si no lo expone).

        Se busca en la ventana en vez de inyectarlo para no acoplar los widgets a
        ``MainWindow``: quien no lo tenga (tests, usos mínimos) simplemente no
        reserva GPU y todo sigue funcionando.
        """
        return getattr(self.window(), "gpu_scheduler", None)

    def _gpu_engine_ids(self) -> tuple[str, ...]:
        """Motores que esta tarea puede estar usando (no se les pide soltar VRAM).

        Se deduce del motor propio del widget (``self._engine``, que los cuatro
        widgets con motor declararan con ese nombre): si al pedir GPU se liberara
        también el motor que vamos a usar, se tiraría la VRAM que queríamos
        aprovechar (p. ej. recargar Flux entero, ~25 s).
        """
        propio = getattr(self, "_engine", None)
        engine_id = getattr(propio, "id", None)
        return (engine_id,) if engine_id else ()

    def _gpu_liberar_otros(self) -> None:
        """Pide a los demás motores que suelten su VRAM (y solo a ellos)."""
        liberar = getattr(self.window(), "release_engines", None)
        if liberar is not None:
            liberar(except_ids=self._gpu_engine_ids())

    # ---- Foco ------------------------------------------------------

    def _focus_widget(self):
        """Control que debe recuperar el foco al acabar la tarea, o ``None``.

        Lo sobrescriben las tareas con un campo de escritura (chat, audio,
        imagen, vídeo). Documentos y Ensamblador empiezan eligiendo archivo, así
        que no tienen un control principal que enfocar.
        """
        return None

    def _restore_focus(self) -> None:
        """Devuelve el foco al control principal de la tarea.

        Mientras hay trabajo en curso los controles se deshabilitan y Qt retira
        el foco; al rehabilitarlos no lo devuelve, de modo que había que volver a
        pinchar para seguir escribiendo. Solo se enfoca si el control está
        habilitado (si no, Qt ignoraría la llamada).
        """
        widget = self._focus_widget()
        if widget is not None and widget.isEnabled():
            widget.setFocus()

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802 - override de Qt
        """Al entrar en la pestaña, deja el foco listo para escribir."""
        super().showEvent(event)
        self._restore_focus()
