"""Base común de los widgets de tarea (P4 de la auditoría).

Los seis widgets de la app (chat, documentos, imagen, vídeo, audio y
ensamblador) repetían el mismo esqueleto: ``_set_status``, ``_set_busy``,
``_on_error`` comparando el mensaje de cancelación y ``_on_finished`` liberando
el worker. Aquí se escribe una sola vez; cada widget declara qué controles se
bloquean y, si lo necesita, afina el resto con los hooks.
"""

from PyQt6.QtWidgets import QWidget

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

    # ---- Ciclo de vida del worker ----------------------------------

    def _on_error(self, message: str) -> None:
        """Distingue la cancelación (no es un fallo) del error real."""
        if message == CANCEL_MESSAGE:
            self._on_cancelled(message)
        else:
            self._on_failed(message)

    def _on_cancelled(self, message: str) -> None:
        """Hook: el usuario canceló la tarea."""
        self._set_status(message)

    def _on_failed(self, message: str) -> None:
        """Hook: la tarea falló por un error real."""
        self._set_status(f"Error: {message}")

    def _on_finished(self) -> None:
        """Libera el worker al terminar (éxito, error o cancelación)."""
        self._set_busy(False)
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
