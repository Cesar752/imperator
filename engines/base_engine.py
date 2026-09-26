"""Interfaces comunes para motores de la plataforma.

Antes existía un único ABC ``Engine`` que obligaba a TODOS los motores (imagen,
vídeo, audio...) a implementar ``chat()`` (los no-textuales solo podían lanzar
una excepción). Ahora ``Engine`` define únicamente el contrato común (identidad,
disponibilidad y catálogo de modelos) y cada familia de motor añade su método de
generación. Así cada motor declara exactamente lo que sabe hacer y el sistema de
tipos lo refleja.
"""

from abc import ABC, abstractmethod
from pathlib import Path


class EngineError(Exception):
    """Error producido por un motor (conexión, timeout, respuesta inválida)."""


#: Mensaje estándar de cancelación. Vive en la capa de motores (y no en la UI)
#: porque los motores deben poder lanzarlo sin depender de la presentación; los
#: widgets lo importan para no mostrarlo como error. Ver ``ui.workers``.
CANCEL_MESSAGE = "Cancelado por el usuario"


class Engine(ABC):
    """Contrato común a todo motor: identidad, disponibilidad y modelos."""

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


class ChatEngine(Engine):
    """Motor de conversación (LLM de texto)."""

    @abstractmethod
    def chat(self, messages: list[dict], model: str | None = None) -> str:
        """Ejecuta una conversación y devuelve la respuesta completa.

        ``model`` permite elegir el modelo sin reconstruir el motor.
        """


class VisionEngine(Engine):
    """Motor multimodal de visión (analiza una imagen y devuelve texto)."""

    @abstractmethod
    def analyze_image(
        self, image_path: Path, prompt: str, model: str | None = None
    ) -> str:
        """Analiza ``image_path`` según ``prompt`` y devuelve el texto resultante."""

    def analyze_text(self, text: str, prompt: str, model: str | None = None) -> str:
        """Analiza ``text`` según ``prompt`` (sin imagen) y devuelve el resultado.

        Lo usan los documentos con capa de texto propia (PDF nativo): resumir o
        extraer tablas no requiere OCR, pero sí pasar por el modelo. Por defecto
        no está soportado; los motores que puedan hacerlo lo sobrescriben.
        """
        raise EngineError(f"{self.nombre}: no soporta análisis de texto sin imagen.")


class ImageEngine(Engine):
    """Motor de generación/edición de imagen."""

    @abstractmethod
    def generate_image(self, prompt: str, **kwargs) -> Path:
        """Genera una imagen y devuelve la ruta del fichero creado."""


class VideoEngine(Engine):
    """Motor de generación de vídeo."""

    @abstractmethod
    def generate_video(self, prompt: str, **kwargs) -> Path:
        """Genera un vídeo y devuelve la ruta del fichero creado."""


class AudioEngine(Engine):
    """Motor de texto a voz (TTS)."""

    @abstractmethod
    def synthesize(self, text: str, voice: str | None = None, **kwargs) -> Path:
        """Sintetiza ``text`` y devuelve la ruta del audio generado."""


class TranscriptionEngine(Engine):
    """Motor de transcripción de audio a subtítulos (STT)."""

    @abstractmethod
    def transcribe(self, audio_path: str | Path, **kwargs) -> Path:
        """Transcribe ``audio_path`` y devuelve la ruta del SRT generado."""
