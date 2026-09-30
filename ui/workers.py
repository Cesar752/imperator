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
from contextlib import contextmanager
from typing import Callable, Iterator

from PyQt6.QtCore import QObject, QRunnable, QThread, QThreadPool, pyqtSignal

from core.gpu_scheduler import GpuScheduler, priority_for
from engines.base_engine import CANCEL_MESSAGE, EngineError

logger = logging.getLogger(__name__)

# ``CANCEL_MESSAGE`` se reexporta desde ``engines.base_engine`` para que motores
# y widgets compartan el mismo mensaje sin invertir el sentido de las
# dependencias (la UI depende de los motores, nunca al contrario).


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


@contextmanager
def gpu_slot(
    scheduler: GpuScheduler | None,
    name: str,
    *,
    liberar_otros: Callable[[], None] | None = None,
) -> Iterator[None]:
    """Deja la GPU lista para ``name`` mientras dure el bloque.

    Es el punto único donde las tareas pesadas piden permiso al planificador
    ([`core.gpu_scheduler`]), para que no se peleen por la VRAM. Medido con la GPU
    real: tras una imagen ComfyUI puede dejar **15 438 MB** ocupados y el chat
    necesita **13 116 MB** de 16 303, así que el segundo motor moría por OOM.

    Secuencia:

    1. Sin planificador (tests, usos mínimos, ventana sin él) no hace nada.
    2. Si ahora no se puede empezar, se pide a los demás motores que suelten su
       VRAM (``liberar_otros``) y se reintenta una vez: lo que sobra suele ser la
       VRAM que retiene otro motor (el mismo remedio que *Liberar VRAM ahora*,
       pero automático).
    3. Si sigue sin poder, lanza :class:`EngineError` con el motivo del
       planificador: decirlo es mejor que morir por OOM a mitad de trabajo.
    4. Al salir (éxito o error) devuelve la reserva.

    No se declara el consumo real de cada modelo (``vram_mb`` queda en 0) a
    propósito: Ollama y ComfyUI reparten capas entre CPU y GPU, así que una
    estimación alta bloquearía trabajos que hoy funcionan. Lo que se evita es
    **empezar con la GPU casi llena o con otro lote en marcha**, que es lo que
    rompía de verdad (el margen lo pone ``vram_headroom_mb``).
    """
    if scheduler is None:
        yield
        return

    ok, motivo = scheduler.can_start(priority_for(name))
    if not ok and liberar_otros is not None:
        logger.info(
            "GPU: '%s' no puede empezar (%s); se libera la VRAM de los otros "
            "motores y se reintenta.",
            name,
            motivo,
        )
        liberar_otros()
        ok, motivo = scheduler.can_start(priority_for(name))
    if not ok:
        raise EngineError(
            f"La GPU no está disponible para esta tarea ahora mismo ({motivo}). "
            "Espera a que termine el otro trabajo o usa "
            "«Motores → Liberar VRAM ahora»."
        )
    if not scheduler.acquire(name):
        raise EngineError(
            f"Otro trabajo ha ocupado la GPU antes de empezar ({motivo})."
        )
    try:
        yield
    finally:
        scheduler.release(name)


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
