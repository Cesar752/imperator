"""Workers reutilizables para no bloquear el hilo de la UI.

Dos patrones únicos (antes cada widget reinventaba el suyo):

- :class:`ThreadWorker` — ``QThread`` para tareas largas con cancelación
  cooperativa (chat, imagen, vídeo, audio, documentos, ensamblador). Las
  subclases implementan ``work()``; la base centraliza el ``try/except``, la
  comprobación de cancelación y las señales ``result``/``failed``/``progress``.
- :class:`PoolTask` — ``QRunnable`` para consultas cortas (listar modelos,
  créditos, stats) lanzadas con :func:`run_pool_task`.
"""

import logging
from typing import Callable

from PyQt6.QtCore import QObject, QRunnable, QThread, QThreadPool, pyqtSignal

from engines.base_engine import EngineError

logger = logging.getLogger(__name__)

#: Mensaje estándar de cancelación (los widgets lo comparan para no mostrarlo
#: como error).
CANCEL_MESSAGE = "Cancelado por el usuario"


class ThreadWorker(QThread):
    """Base de los workers de tarea larga.

    Señales:
    - ``progress(int, str)``: porcentaje y mensaje (opcional).
    - ``result(object)``: valor devuelto por :meth:`work`.
    - ``failed(str)``: mensaje de error (o ``CANCEL_MESSAGE``).
    """

    progress = pyqtSignal(int, str)
    result = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

    def work(self):
        """Implementado por la subclase; devuelve el resultado de la tarea."""
        raise NotImplementedError

    def run(self) -> None:
        try:
            if self.isInterruptionRequested():
                raise EngineError(CANCEL_MESSAGE)
            value = self.work()
        except EngineError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001 - red/IO/JSON inesperados
            logger.debug(
                "Error inesperado en %s: %r", type(self).__name__, exc, exc_info=True
            )
            self.failed.emit(f"Error inesperado: {exc}")
        else:
            # Si el usuario canceló durante la tarea, no publicar el resultado.
            if self.isInterruptionRequested():
                self.failed.emit(CANCEL_MESSAGE)
            else:
                self.result.emit(value)


class _PoolSignals(QObject):
    result = pyqtSignal(object)
    failed = pyqtSignal(str)


class PoolTask(QRunnable):
    """Ejecuta una función en el ``QThreadPool`` y emite su resultado."""

    def __init__(self, fn: Callable, *args, **kwargs) -> None:
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self.signals = _PoolSignals()

    def run(self) -> None:
        try:
            value = self._fn(*self._args, **self._kwargs)
        except Exception as exc:  # noqa: BLE001 - red/timeout/JSON inesperados
            logger.debug("Error en tarea de pool: %r", exc, exc_info=True)
            self.signals.failed.emit(str(exc))
            return
        self.signals.result.emit(value)


def run_pool_task(
    fn: Callable,
    *args,
    on_result: Callable | None = None,
    on_error: Callable[[str], None] | None = None,
    **kwargs,
) -> PoolTask:
    """Lanza ``fn(*args, **kwargs)`` en el pool global y conecta las señales."""
    task = PoolTask(fn, *args, **kwargs)
    if on_result is not None:
        task.signals.result.connect(on_result)
    if on_error is not None:
        task.signals.failed.connect(on_error)
    QThreadPool.globalInstance().start(task)
    return task
