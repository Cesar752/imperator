"""Interfaces comunes para motores de la plataforma.

Antes existía un único ABC ``Engine`` que obligaba a TODOS los motores (imagen,
vídeo, audio...) a implementar ``chat()`` (los no-textuales solo podían lanzar
una excepción). Ahora ``Engine`` define únicamente el contrato común (identidad,
disponibilidad y catálogo de modelos) y cada familia de motor añade su método de
generación. Así cada motor declara exactamente lo que sabe hacer y el sistema de
tipos lo refleja.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


class EngineError(Exception):
    """Error producido por un motor (conexión, timeout, respuesta inválida)."""


#: Mensaje estándar de cancelación. Vive en la capa de motores (y no en la UI)
#: porque los motores deben poder lanzarlo sin depender de la presentación; los
#: widgets lo importan para no mostrarlo como error. Ver ``ui.workers``.
CANCEL_MESSAGE = "Cancelado por el usuario"

#: Tope por defecto de las descargas de resultados (512 MiB). Los motores bajan
#: binarios de un backend externo (imagen/vídeo de ComfyUI o de la API de BFL) sin
#: ningún límite: un workflow mal configurado o un backend que devuelva otra cosa
#: llenarían el disco sin avisar.
MAX_DOWNLOAD_BYTES = 512 * 1024 * 1024


def _declared_size(resp) -> int | None:
    """Tamaño declarado en ``Content-Length`` (``None`` si no es legible)."""
    try:
        valor = resp.headers.get("Content-Length")
    except (AttributeError, TypeError):
        return None
    try:
        return int(valor)
    except (TypeError, ValueError):
        return None


def check_download_size(
    resp, content: bytes, label: str, max_bytes: int = MAX_DOWNLOAD_BYTES
) -> None:
    """Comprueba que una descarga no excede ``max_bytes``.

    Se revisa lo declarado en ``Content-Length`` (si el backend lo envía) y el
    cuerpo ya recibido. Lanza :class:`EngineError` con el motivo, para que la UI
    lo muestre como error de motor y no como un fallo inesperado.
    """
    declarado = _declared_size(resp)
    if declarado is not None and declarado > max_bytes:
        raise EngineError(
            f"{label}: el resultado declara {declarado} bytes y el tope es "
            f"{max_bytes} bytes; se descarta."
        )
    if len(content) > max_bytes:
        raise EngineError(
            f"{label}: el resultado ocupa {len(content)} bytes y el tope es "
            f"{max_bytes} bytes; se descarta."
        )


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

    def chat_stream(
        self,
        messages: list[dict],
        model: str | None = None,
        *,
        on_token: Callable[[str, int], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> str:
        """Como :meth:`chat`, pero informando del avance y pudiendo cancelar.

        ``on_token(delta, total_chars)`` recibe cada fragmento según llega y
        ``cancel_check()`` se consulta mientras se espera: al devolver ``True`` se
        corta la lectura (y el servidor detiene la generación).

        Por defecto **no** hay streaming: se delega en :meth:`chat`, así que un
        motor que no lo implemente —o un doble de test— sigue funcionando igual.
        """
        return self.chat(messages, model=model)


def chat_engine_reply(
    engine,
    messages: list[dict],
    *,
    model: str | None = None,
    on_token: Callable[[str, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> str:
    """Respuesta de un motor de chat, con **streaming si el motor lo soporta**.

    Es el punto único que usan los workers de chat y de guion: si el motor
    implementa ``chat_stream`` (Ollama) se aprovechan el avance real y la
    cancelación inmediata; si no (dobles de test, motores antiguos) se cae a
    ``chat`` sin streaming, sin romper nada.
    """
    stream = getattr(engine, "chat_stream", None)
    if callable(stream):
        return stream(
            messages, model=model, on_token=on_token, cancel_check=cancel_check
        )
    return engine.chat(messages, model=model)


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


class MusicEngine(Engine):
    """Motor de generación de música a partir de una descripción de texto."""

    @abstractmethod
    def generate_music(self, prompt: str, **kwargs) -> Path:
        """Genera música según ``prompt`` y devuelve la ruta del audio creado."""


class TranscriptionEngine(Engine):
    """Motor de transcripción de audio a subtítulos (STT)."""

    @abstractmethod
    def transcribe(self, audio_path: str | Path, **kwargs) -> Path:
        """Transcribe ``audio_path`` y devuelve la ruta del SRT generado."""


@dataclass(frozen=True)
class Perception:
    """Resultado **estructurado** de un motor de percepción de imagen.

    A diferencia de :class:`VisionEngine` (que devuelve la respuesta en lenguaje
    natural de un modelo multimodal), aquí los datos vienen ya separados para
    que el resto del sistema pueda agregarlos sin volver a parsear texto:
    objetos detectados (YOLO), etiquetas (CLIP) y texto leído (OCR).

    Es inmutable; para combinar resultados de varios motores se usa
    :meth:`merge`.
    """

    #: Objetos: ``[{"label": str, "score": float, "box": (x1, y1, x2, y2)}]``.
    objects: tuple[dict, ...] = field(default_factory=tuple)
    #: Etiquetas de imagen: ``[{"label": str, "score": float}]``.
    labels: tuple[dict, ...] = field(default_factory=tuple)
    #: Texto leído en la imagen (OCR).
    text: str = ""
    #: Descripción breve, si el motor la produce.
    caption: str = ""

    @property
    def is_empty(self) -> bool:
        return not (self.objects or self.labels or self.text.strip() or self.caption.strip())

    def merge(self, other: "Perception") -> "Perception":
        """Combina dos percepciones (p. ej. detección + OCR)."""
        textos = [t for t in (self.text.strip(), other.text.strip()) if t]
        caption = self.caption.strip() or other.caption.strip()
        return Perception(
            objects=self.objects + other.objects,
            labels=self.labels + other.labels,
            text="\n".join(textos),
            caption=caption,
        )

    def summary(self) -> str:
        """Resumen en texto plano, pensado para agregar y pasar al LLM."""
        lineas: list[str] = []
        objetos = _format_labels(self.objects)
        if objetos:
            lineas.append(f"Objetos: {objetos}")
        etiquetas = _format_labels(self.labels)
        if etiquetas:
            lineas.append(f"Etiquetas: {etiquetas}")
        if self.text.strip():
            lineas.append(f"Texto: {self.text.strip()}")
        if self.caption.strip():
            lineas.append(f"Descripción: {self.caption.strip()}")
        return "\n".join(lineas)

    def to_dict(self) -> dict:
        """Serializable (para logs, JSON o pasar entre hilos/UI)."""
        return {
            "objects": [dict(o) for o in self.objects],
            "labels": [dict(etiqueta) for etiqueta in self.labels],
            "text": self.text,
            "caption": self.caption,
        }


def _format_labels(items: tuple[dict, ...]) -> str:
    """``«person (x3), car»``: cuenta por etiqueta y ordena por frecuencia."""
    conteo: dict[str, int] = {}
    for item in items:
        etiqueta = str(item.get("label", "")).strip()
        if etiqueta:
            conteo[etiqueta] = conteo.get(etiqueta, 0) + 1
    if not conteo:
        return ""
    ordenadas = sorted(conteo.items(), key=lambda par: (-par[1], par[0]))
    return ", ".join(
        etiqueta if n == 1 else f"{etiqueta} (x{n})" for etiqueta, n in ordenadas
    )


class PerceptionEngine(Engine):
    """Motor de percepción de imagen que devuelve datos estructurados.

    Lo implementan los motores ligeros y deterministas (YOLO, CLIP, OCR) cuyos
    resultados se pueden agregar sin lenguaje natural. Los modelos multimodales
    siguen siendo :class:`VisionEngine`.
    """

    @abstractmethod
    def perceive(self, image_path: str | Path, prompt: str = "") -> Perception:
        """Analiza ``image_path`` y devuelve lo que sepa extraer.

        ``prompt`` es opcional y solo lo usan los motores que lo admiten (CLIP lo
        interpreta como lista de etiquetas candidatas separadas por comas).
        """
