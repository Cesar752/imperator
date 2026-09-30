"""Orquestador de tareas: traduce la intención del chat en una acción de la UI.

Es la capa que faltaba **por encima** de :class:`core.task_manager.TaskManager`:
el chat puede decidir a qué pestaña apunta un mensaje, pero el chat no debe
abrir pestañas ni conocer widgets. El orquestador resuelve la intención contra
las tareas realmente registradas y devuelve un :class:`Dispatch` que la ventana
principal interpreta (ofrecer abrir la pestaña, etc.).

Lógica pura: no importa Qt. Las tareas disponibles se inyectan como un
``Callable`` porque el registro vive en el ``TaskManager`` y se consulta de forma
diferida (evita el ciclo ``TaskManager → orchestrator → TaskManager``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from services import intent_router as ir


@dataclass(frozen=True)
class Dispatch:
    """Orden interna: «esto es una tarea de ``task_id`` con esta acción»."""

    task_id: str
    action: str
    confidence: float
    matched: tuple[str, ...] = ()
    text: str = ""

    @property
    def label(self) -> str:
        """Nombre legible de la tarea destino."""
        return ir.TASK_LABELS.get(self.task_id, self.task_id)

    @property
    def action_label(self) -> str:
        """Nombre legible de la acción (resumen, OCR, …)."""
        return ir.ACTION_LABELS.get(self.action, self.action)

    def suggestion(self) -> str:
        """Frase que se le ofrece al usuario."""
        return (
            f"Tu mensaje parece una tarea de la pestaña «{self.label}» "
            f"({self.action_label})."
        )


class TaskOrchestrator:
    """Decide si un mensaje del chat debe desviarse a otra tarea."""

    def __init__(
        self, available: Callable[[], Sequence[str]] | None = None
    ) -> None:
        #: Callable diferido: devuelve los ids de tarea registrados.
        self._available = available if available is not None else (lambda: ())

    def available_tasks(self) -> list[str]:
        """Ids de las tareas registradas ahora mismo."""
        try:
            return [str(task_id) for task_id in self._available()]
        except Exception:  # noqa: BLE001 - la consulta no debe romper el envío
            return []

    def is_available(self, task_id: str) -> bool:
        return task_id in self.available_tasks()

    def dispatch(self, text: str) -> Dispatch | None:
        """Devuelve el :class:`Dispatch` de ``text``, o ``None`` si no procede.

        No procede cuando el mensaje se queda en el chat (sin señal suficiente)
        o cuando apunta a una tarea que no está registrada (no habría nada que
        abrir).
        """
        intent = ir.route(text)
        if not intent.is_routed:
            return None
        if not self.is_available(intent.kind):
            return None
        return Dispatch(
            task_id=intent.kind,
            action=intent.action,
            confidence=intent.confidence,
            matched=intent.matched,
            text=text,
        )

    def plan(self, text: str) -> str:
        """Explicación legible de lo que haría el orquestador (para logs/tests)."""
        intent = ir.route(text)
        if not intent.is_routed:
            return intent.describe()
        if not self.is_available(intent.kind):
            return f"{intent.describe()} Pero esa pestaña no está registrada."
        return intent.describe()
