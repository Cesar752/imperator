"""Tests de core.task_manager: registro de tareas e inyección de dependencias.

El ``TODO(fase 6)`` era que ``TASK_FACTORY`` asumía ``cls(engine_manager)`` para
todas las tareas. Ahora cada entrada es una factory que recibe ``TaskDeps``, de
modo que una tarea puede pedir otras dependencias (p. ej. ``ApiManager``) sin
romper el bucle uniforme de registro.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import task_manager as tm  # noqa: E402
from core.api_manager import ApiManager  # noqa: E402
from core.config_manager import ConfigManager  # noqa: E402
from tasks.base_task import Task  # noqa: E402

# Orden de registro esperado (una pestaña por tarea).
IDS_ESPERADOS = [
    "chat",
    "document",
    "image",
    "video",
    "understanding",
    "audio",
    "assembler",
]
NOMBRES_ESPERADOS = [
    "Chat",
    "Documentos",
    "Imagen",
    "Vídeo",
    "Comprensión",
    "Audio",
    "Ensamblador",
]


class _StubEngineManager:
    """Sustituto mínimo: las tareas no tocan motores hasta ``build_widget``."""

    def __init__(self, config_manager: ConfigManager | None = None) -> None:
        self._config_manager = config_manager

    @property
    def config_manager(self) -> ConfigManager | None:
        return self._config_manager


class _ApiAwareTask(Task):
    """Tarea de prueba que, además del motor, exige el ``ApiManager``."""

    def __init__(self, engine_manager: object, api_manager: object) -> None:
        self.engine_manager = engine_manager
        self.api_manager = api_manager

    @property
    def id(self) -> str:
        return "con_deps"

    @property
    def nombre(self) -> str:
        return "Con deps"

    def build_widget(self, parent=None):  # pragma: no cover - no se instancia UI
        raise AssertionError("el registro no debe construir widgets")


class TaskManagerTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.cm = ConfigManager(config_dir=self._tmp.name)
        self.engines = _StubEngineManager(self.cm)
        self.apis = ApiManager(self.cm)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_registra_todas_las_tareas_en_orden(self) -> None:
        manager = tm.TaskManager(self.engines, self.apis)
        self.assertEqual([t.id for t in manager.list_tasks()], IDS_ESPERADOS)
        self.assertEqual([t.nombre for t in manager.list_tasks()], NOMBRES_ESPERADOS)

    def test_las_tareas_cumplen_el_contrato_Task(self) -> None:
        manager = tm.TaskManager(self.engines, self.apis)
        for tarea in manager.list_tasks():
            self.assertIsInstance(tarea, Task)

    def test_get_task(self) -> None:
        manager = tm.TaskManager(self.engines, self.apis)
        self.assertIsNotNone(manager.get_task("chat"))
        self.assertEqual(manager.get_task("chat").id, "chat")
        self.assertIsNone(manager.get_task("inexistente"))

    def test_crea_api_manager_si_no_se_inyecta(self) -> None:
        """Sin ``api_manager`` se crea uno con el ConfigManager del motor."""
        manager = tm.TaskManager(self.engines)
        self.assertIsInstance(manager.api_manager, ApiManager)

    def test_usa_el_api_manager_inyectado(self) -> None:
        manager = tm.TaskManager(self.engines, self.apis)
        self.assertIs(manager.api_manager, self.apis)

    def test_factory_puede_pedir_dependencias_extra(self) -> None:
        """Punto de extensión: la factory recibe el ``TaskDeps`` completo."""
        manager = tm.TaskManager(self.engines, self.apis)
        factory = lambda deps: _ApiAwareTask(  # noqa: E731 - factory de prueba
            deps.engine_manager, deps.api_manager
        )
        with mock.patch.dict(tm.TASK_FACTORY, {"con_deps": factory}):
            manager._register_all()
            tarea = manager.get_task("con_deps")

        self.assertIsInstance(tarea, _ApiAwareTask)
        self.assertIs(tarea.engine_manager, self.engines)
        self.assertIs(tarea.api_manager, self.apis)

    def test_tasks_deps_es_inmutable(self) -> None:
        deps = tm.TaskDeps(engine_manager=self.engines, api_manager=self.apis)
        with self.assertRaises(Exception):
            deps.api_manager = None  # type: ignore[misc] - frozen dataclass


if __name__ == "__main__":
    unittest.main()
