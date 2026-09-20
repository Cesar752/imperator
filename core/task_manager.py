"""Registro y ciclo de vida de las tareas (plugins) de la plataforma."""

import logging

from core.engine_manager import EngineManager
from tasks.base_task import Task
from tasks.chat import ChatTask
from tasks.document import DocumentTask
from tasks.audio import AudioTask
from tasks.image import ImageTask
from tasks.video import VideoTask

logger = logging.getLogger(__name__)

# Mapa id de tarea -> clase Task. Registro explícito: añadir aquí cada nueva
# tarea (image, video, audio, docs) sin tocar el núcleo.
#
# TODO(fase 6): este dict con ``cls(engine_manager)`` asume que todas las
# tareas reciben el mismo único argumento. Cuando haya 3+ tareas y algunas
# necesiten otras dependencias (p. ej. ApiManager), migrar a factory functions
# o pasar las dependencias como kwargs para no romper la construcción uniforme.
TASK_FACTORY: dict[str, type[Task]] = {
    "chat": ChatTask,
    "document": DocumentTask,
    "image": ImageTask,
    "video": VideoTask,
    "audio": AudioTask,
}


class TaskManager:
    """Descubre (registro explícito) y registra las tareas de la plataforma.

    Mantiene las instancias de tarea construidas en :attr:`_tasks`. Cada tarea
    expone su widget mediante :meth:`Task.build_widget`, que es quien lo añade
    a la UI (ver :class:`ui.main_window.MainWindow`).
    """

    def __init__(self, engine_manager: EngineManager) -> None:
        self._engine_manager = engine_manager
        self._tasks: dict[str, Task] = {}
        self._register_all()

    # ---- Registro ------------------------------------------------

    def _register_all(self) -> None:
        for task_id, task_cls in TASK_FACTORY.items():
            self._tasks[task_id] = task_cls(self._engine_manager)

    # ---- Consultas ----------------------------------------------

    def list_tasks(self) -> list[Task]:
        """Tareas registradas, en orden de registro."""
        return list(self._tasks.values())

    def get_task(self, task_id: str) -> Task | None:
        """Devuelve la tarea registrada con ``task_id``, o ``None`` si no existe."""
        return self._tasks.get(task_id)
