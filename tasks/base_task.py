"""Interfaz común para las tareas (plugins) de la plataforma."""

from abc import ABC, abstractmethod

from PyQt6.QtWidgets import QWidget


class Task(ABC):
    """Contrato que debe cumplir toda tarea registrable en la plataforma.

    Las tareas son la unidad de trabajo que ve el usuario (chat, imagen,
    vídeo, audio, documentos). Cada una construye su propio widget de UI y
    se registra en :class:`core.task_manager.TaskManager`.
    """

    # Id del motor cuyo modelo se descarga al salir de esta pestaña (``None``
    # para tareas sin motor único, p. ej. Audio con Piper+Kokoro). El gestor de
    # ventana deriva su mapa pestaña->motor de este atributo, no de una lista
    # duplicada.
    engine_id: str | None = None

    @property
    @abstractmethod
    def id(self) -> str:
        """Identificador único de la tarea (ej.: ``chat``)."""

    @property
    @abstractmethod
    def nombre(self) -> str:
        """Nombre legible de la tarea (ej.: ``Chat``)."""

    @abstractmethod
    def build_widget(self, parent: QWidget | None = None) -> QWidget:
        """Construye y devuelve el widget de la tarea."""
