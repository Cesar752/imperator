"""Registro y ciclo de vida de las tareas (plugins) de la plataforma."""

import logging
from collections.abc import Callable
from dataclasses import dataclass

from core.api_manager import ApiManager
from core.engine_manager import EngineManager
from tasks.base_task import Task
from tasks.assembler import AssemblerTask
from tasks.chat import ChatTask
from tasks.document import DocumentTask
from tasks.audio import AudioTask
from tasks.image import ImageTask
from tasks.video import VideoTask

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskDeps:
    """Dependencias compartidas que :class:`TaskManager` inyecta en las tareas.

    Antes cada tarea se construía con ``cls(engine_manager)``: un único
    argumento asumido por todas. Con este contenedor, una tarea que necesite
    otra dependencia (p. ej. ``ApiManager``) la recibe de forma explícita, sin
    romper el bucle uniforme de registro.
    """

    engine_manager: EngineManager
    api_manager: ApiManager


# Una factory recibe las dependencias ya resueltas y devuelve la tarea.
TaskFactory = Callable[[TaskDeps], Task]


def _engine_only(task_cls: type[Task]) -> TaskFactory:
    """Adapta una tarea cuyo constructor solo pide ``engine_manager``.

    Evita repetir la misma lambda en cada entrada del registro y deja el punto
    de extensión para las tareas que necesiten más campos de :class:`TaskDeps`.
    """

    def factory(deps: TaskDeps) -> Task:
        return task_cls(deps.engine_manager)

    return factory


# Mapa id de tarea -> factory. Registro explícito: añadir aquí cada nueva
# tarea (image, video, audio, docs) sin tocar el núcleo. Cada factory decide
# qué dependencias de TaskDeps inyectar: ``_engine_only`` para las que solo
# necesitan el motor; una factory propia para las que pidan algo más.
TASK_FACTORY: dict[str, TaskFactory] = {
    "chat": _engine_only(ChatTask),
    "document": _engine_only(DocumentTask),
    "image": _engine_only(ImageTask),
    "video": _engine_only(VideoTask),
    "audio": _engine_only(AudioTask),
    "assembler": _engine_only(AssemblerTask),
}


class TaskManager:
    """Descubre (registro explícito) y registra las tareas de la plataforma.

    Mantiene las instancias de tarea construidas en :attr:`_tasks`, creadas por
    las factories de :data:`TASK_FACTORY` a partir de las dependencias de
    :class:`TaskDeps`. Cada tarea expone su widget mediante
    :meth:`Task.build_widget`, que es quien lo añade a la UI
    (ver :class:`ui.main_window.MainWindow`).
    """

    def __init__(
        self,
        engine_manager: EngineManager,
        api_manager: ApiManager | None = None,
    ) -> None:
        self._engine_manager = engine_manager
        # Sin inyección explícita (tests, usos mínimos) se crea uno ligado al
        # mismo ConfigManager del EngineManager para no duplicar la config.
        self._api_manager = (
            api_manager
            if api_manager is not None
            else ApiManager(engine_manager.config_manager)
        )
        self._deps = TaskDeps(
            engine_manager=self._engine_manager,
            api_manager=self._api_manager,
        )
        self._tasks: dict[str, Task] = {}
        self._register_all()

    # ---- Registro ------------------------------------------------

    def _register_all(self) -> None:
        for task_id, factory in TASK_FACTORY.items():
            self._tasks[task_id] = factory(self._deps)

    # ---- Consultas ----------------------------------------------

    @property
    def api_manager(self) -> ApiManager:
        """ApiManager compartido con las tareas que necesiten consultarlo."""
        return self._api_manager

    def list_tasks(self) -> list[Task]:
        """Tareas registradas, en orden de registro."""
        return list(self._tasks.values())

    def get_task(self, task_id: str) -> Task | None:
        """Devuelve la tarea registrada con ``task_id``, o ``None`` si no existe."""
        return self._tasks.get(task_id)
