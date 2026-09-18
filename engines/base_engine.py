"""Interfaz común para motores locales de la plataforma."""

from abc import ABC, abstractmethod


class EngineError(Exception):
    """Error producido por un motor local (conexión, timeout, respuesta inválida)."""


class Engine(ABC):
    """Contrato que debe cumplir todo motor local.

    La generación es bloqueante por ahora; el threading es responsabilidad
    de la capa de UI (ver ``ui/chat_widget.py``).
    """

    @property
    @abstractmethod
    def id(self) -> str:
        """Identificador único del motor (ej.: ``ollama``)."""

    @property
    @abstractmethod
    def nombre(self) -> str:
        """Nombre legible del motor (ej.: ``Ollama``)."""

    @abstractmethod
    def is_available(self) -> bool:
        """Devuelve ``True`` si el motor responde en su endpoint."""

    @abstractmethod
    def list_models(self) -> list[str]:
        """Devuelve los nombres de los modelos expuestos por el motor."""

    @abstractmethod
    def chat(self, messages: list[dict]) -> str:
        """Ejecuta una conversación y devuelve la respuesta completa."""
