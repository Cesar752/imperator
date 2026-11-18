"""Registro y ciclo de vida de las tareas (plugins) de la plataforma."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING

from core.api_manager import ApiManager
from core.engine_manager import EngineManager
from services.task_orchestrator import TaskOrchestrator

if TYPE_CHECKING:
    from tasks.base_task import Task

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
    #: Orquestador que permite al chat desviar un mensaje a otra tarea. Es
    #: opcional (valor por defecto ``None``) para no romper a quien construya
    #: ``TaskDeps`` a mano en tests o en usos mínimos.
    orchestrator: TaskOrchestrator | None = None


# Una factory recibe las dependencias ya resueltas y devuelve la tarea.
TaskFactory = Callable[[TaskDeps], "Task"]


def _engine_only(module: str, class_name: str) -> TaskFactory:
    """Adapta una tarea cuyo constructor solo pide ``engine_manager``.

    Evita repetir la misma lambda en cada entrada del registro y deja el punto
    de extensión para las tareas que necesiten más campos de :class:`TaskDeps`.
    """

    def factory(deps: TaskDeps) -> Task:
        # Import perezoso: cada tarea importa su widget de ``ui`` (y con él
        # PyQt6), así que en la cabecera haría que ``import core.task_manager``
        # cargase la GUI.
        cls = getattr(import_module(module), class_name)
        return cls(deps.engine_manager)

    return factory


def _chat_factory(deps: TaskDeps) -> Task:
    """El chat además recibe el orquestador (para desviar mensajes a otras tareas)."""
    from tasks.chat import ChatTask

    return ChatTask(deps.engine_manager, deps.orchestrator)


# Mapa id de tarea -> factory. Registro explícito: añadir aquí cada nueva
# tarea (image, video, audio, docs) sin tocar el núcleo. Cada factory decide
# qué dependencias de TaskDeps inyectar: ``_engine_only`` para las que solo
# necesitan el motor; una factory propia para las que pidan algo más.
TASK_FACTORY: dict[str, TaskFactory] = {
    "chat": _chat_factory,
    "document": _engine_only("tasks.document", "DocumentTask"),
    "image": _engine_only("tasks.image", "ImageTask"),
    "photo": _engine_only("tasks.photo", "PhotoTask"),
    "video": _engine_only("tasks.video", "VideoTask"),
    "video_enhance": _engine_only("tasks.video_enhance", "VideoEnhanceTask"),
    "understanding": _engine_only("tasks.understanding", "UnderstandingTask"),
    "notes": _engine_only("tasks.notes", "NotesTask"),
    "chronicle": _engine_only("tasks.chronicle", "ChronicleTask"),
    "audio": _engine_only("tasks.audio", "AudioTask"),
    "assembler": _engine_only("tasks.assembler", "AssemblerTask"),
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
        self._tasks: dict[str, Task] = {}
        # El orquestador consulta las tareas registradas de forma diferida (una
        # lambda sobre ``self._tasks``), así que se puede crear antes del
        # registro sin ciclo TaskManager -> orchestrator -> TaskManager.
        self._orchestrator = TaskOrchestrator(available=lambda: tuple(self._tasks))
        self._deps = TaskDeps(
            engine_manager=self._engine_manager,
            api_manager=self._api_manager,
            orchestrator=self._orchestrator,
        )
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

    @property
    def orchestrator(self) -> TaskOrchestrator:
        """Orquestador de intenciones (lo usan el chat y la ventana principal)."""
        return self._orchestrator

    def list_tasks(self) -> list[Task]:
        """Tareas registradas, en orden de registro."""
        return list(self._tasks.values())

    def get_task(self, task_id: str) -> Task | None:
        """Devuelve la tarea registrada con ``task_id``, o ``None`` si no existe."""
        return self._tasks.get(task_id)
