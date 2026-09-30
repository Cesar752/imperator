"""Motores de percepción de imagen: YOLO (detección), CLIP (etiquetas) y OCR.

Los tres devuelven un :class:`engines.base_engine.Perception` **estructurado**
(objetos, etiquetas, texto) en vez de lenguaje natural, que es lo que aporta
sobre el VLM de Ollama: datos que se pueden agregar sin volver a interpretarlos.

Convenciones (las mismas que ``musicgen``):

- **Dependencia pesada, opcional y perezosa.** ``ultralytics``,
  ``transformers``+``torch`` y ``paddleocr`` se importan **solo al usar el
  motor**, después de comprobar con ``importlib.util.find_spec`` que están
  instalados: sin ellos la app arranca igual y el motor se marca como no
  disponible (``is_available()`` devuelve ``False``). Importarlos al cargar el
  módulo costaba **8,72 s** de arranque (auditoría del 30/09/2026).
- **Modelo cacheado.** Se carga en la primera llamada y se reutiliza.
- **Runner inyectable.** El cálculo se puede sustituir por un doble en tests sin
  instalar las dependencias pesadas (mismo patrón que ``GpuProbe``).

Sobre TensorRT: ``ultralytics`` carga directamente un motor compilado
(``yolov8n.engine`` / ``.plan``) pasándole esa ruta, y ``torch`` puede usar
``torch.compile``. Aquí no se compila nada: se acepta la ruta del motor y se
deja el *export* como tarea aparte, porque requiere TensorRT y una GPU concreta.

Aviso de idioma: ``paddleocr`` cubre ``latin`` (castellano, euskera aproximado)
y varios alfabetos, pero **no** tiene modelo propio de euskera. CLIP está
entrenado mayoritariamente en inglés; sus etiquetas se dan como vocabulario
cerrado y configurable, no como descripción libre.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

from engines.base_engine import EngineError, Perception, PerceptionEngine

logger = logging.getLogger(__name__)


def _instalado(*modulos: str) -> bool:
    """¿Están instalados esos módulos? Sin importarlos (``find_spec`` es ~0 ms).

    Antes se importaban al cargar el módulo y ``torch`` + ``transformers``
    costaban **8,72 s** de arranque (medidos), para motores que además vienen
    deshabilitados por defecto.
    """
    for modulo in modulos:
        try:
            if importlib.util.find_spec(modulo) is None:
                return False
        except (ImportError, ValueError):
            return False
    return True


def _cargar_yolo():
    """Importa ``ultralytics`` la primera vez que se usa YOLO."""
    from ultralytics import YOLO  # noqa: PLC0415 - import perezoso a propósito

    return YOLO


def _cargar_clip():
    """Importa ``torch`` + ``transformers`` la primera vez que se usa CLIP."""
    import torch  # noqa: PLC0415 - import perezoso a propósito
    from transformers import CLIPModel, CLIPProcessor  # noqa: PLC0415

    return torch, CLIPModel, CLIPProcessor


def _cargar_paddle():
    """Importa ``paddleocr`` la primera vez que se usa el OCR local."""
    from paddleocr import PaddleOCR  # noqa: PLC0415 - import perezoso a propósito

    return PaddleOCR

#: Tamaños publicados de YOLO (Ultralytics). Se acepta también una ruta a un
#: motor ya exportado (``.engine``/``.plan``).
YOLO_MODELS = (
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
    "yolov8x.pt",
)

#: Idiomas que PaddleOCR trae de serie y que aquí tienen sentido (el castellano
#: usa ``latin``; no existe modelo de euskera).
OCR_LANGS = ("latin", "en", "ch", "arabic", "cyrillic")

#: Vocabulario por defecto de CLIP cuando el prompt no trae etiquetas. Es
#: deliberadamente corto y en castellano/inglés: CLIP puntúa mejor listas
#: concretas que descripciones abiertas.
DEFAULT_CLIP_LABELS = (
    "persona",
    "coche",
    "calle",
    "edificio",
    "paisaje",
    "animal",
    "comida",
    "interior",
    "exterior",
    "texto",
    "gráfico",
    "naturaleza",
)


class YoloEngine(PerceptionEngine):
    """Detección de objetos con YOLO (Ultralytics)."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.25,
        model=None,
    ) -> None:
        self._model_path = model_path or YOLO_MODELS[0]
        try:
            self._confidence = min(max(float(confidence), 0.0), 1.0)
        except (TypeError, ValueError):
            self._confidence = 0.25
        # Modelo inyectado (tests) o cacheado tras la primera carga.
        self._model = model

    @property
    def id(self) -> str:
        return "yolo"

    @property
    def nombre(self) -> str:
        return "YOLO (detección)"

    def is_available(self) -> bool:
        return self._model is not None or _instalado("ultralytics")

    def list_models(self) -> list[str]:
        return list(YOLO_MODELS)

    def perceive(self, image_path: str | Path, prompt: str = "") -> Perception:
        """Detecta objetos en ``image_path`` (el prompt se ignora)."""
        model = self._get_model()
        path = self._require_image(image_path)
        logger.info("YOLO: detectando objetos en '%s'...", path.name)
        try:
            results = model.predict(str(path), conf=self._confidence, verbose=False)
        except EngineError:
            raise
        except Exception as exc:  # noqa: BLE001 - backend de inferencia
            raise EngineError(f"YOLO: fallo al detectar objetos: {exc}") from exc
        return Perception(objects=tuple(_objects_from(results)))

    # ---- Internos --------------------------------------------------

    def _get_model(self):
        if self._model is not None:
            return self._model
        if not _instalado("ultralytics"):
            raise EngineError(
                "ultralytics no está instalado. Instálalo con: pip install ultralytics"
            )
        try:
            self._model = _cargar_yolo()(self._model_path)
        except Exception as exc:  # noqa: BLE001 - descarga/carga del peso
            raise EngineError(
                f"YOLO: no se pudo cargar '{self._model_path}': {exc}"
            ) from exc
        return self._model

    @staticmethod
    def _require_image(image_path: str | Path) -> Path:
        path = Path(image_path)
        if not path.is_file():
            raise EngineError(f"La imagen no existe: {path}")
        return path


class ClipEngine(PerceptionEngine):
    """Etiquetado zero-shot con CLIP (transformers).

    CLIP no describe imágenes: puntúa una **lista cerrada** de etiquetas. El
    ``prompt`` de :meth:`perceive` se interpreta como esa lista separada por
    comas; si viene vacío se usa :data:`DEFAULT_CLIP_LABELS`.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        labels: tuple[str, ...] | list[str] | None = None,
        top_k: int = 4,
        runner=None,
    ) -> None:
        self._model_name = model_name
        self._labels = tuple(labels) if labels else DEFAULT_CLIP_LABELS
        self._top_k = max(1, int(top_k))
        #: ``runner(image_path, labels) -> list[float]`` con una probabilidad por
        #: etiqueta. Es lo que se sustituye en los tests.
        self._runner = runner
        self._model = None
        self._processor = None

    @property
    def id(self) -> str:
        return "clip"

    @property
    def nombre(self) -> str:
        return "CLIP (etiquetado)"

    def is_available(self) -> bool:
        return self._runner is not None or _instalado("torch", "transformers")

    def list_models(self) -> list[str]:
        return ["openai/clip-vit-base-patch32", "openai/clip-vit-large-patch14"]

    def perceive(self, image_path: str | Path, prompt: str = "") -> Perception:
        """Puntúa las etiquetas del prompt (o las del motor) sobre la imagen."""
        path = Path(image_path)
        if not path.is_file():
            raise EngineError(f"La imagen no existe: {path}")

        labels = self._labels_from_prompt(prompt)
        if not labels:
            raise EngineError("CLIP: no hay etiquetas que puntuar.")

        if self._runner is not None:
            probs = self._runner(str(path), labels)
        else:
            probs = self._run_clip(path, labels)

        scored = sorted(
            ((label, float(p)) for label, p in zip(labels, probs)),
            key=lambda par: par[1],
            reverse=True,
        )[: self._top_k]
        return Perception(
            labels=tuple(
                {"label": label, "score": round(prob, 4)}
                for label, prob in scored
                if prob > 0
            )
        )

    # ---- Internos --------------------------------------------------

    def _labels_from_prompt(self, prompt: str) -> tuple[str, ...]:
        """El prompt es una lista separada por comas; si no, el vocabulario."""
        texto = (prompt or "").strip()
        if not texto:
            return self._labels
        partes = tuple(p.strip() for p in texto.split(",") if p.strip())
        return partes or self._labels

    def _run_clip(self, path: Path, labels: tuple[str, ...]) -> list[float]:
        model, processor = self._get_model()
        torch, _CLIPModel, _CLIPProcessor = _cargar_clip()
        try:
            inputs = processor(
                text=list(labels), images=str(path), return_tensors="pt", padding=True
            )
            with torch.no_grad():
                outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0]
            return [float(p) for p in probs]
        except EngineError:
            raise
        except Exception as exc:  # noqa: BLE001 - torch/transformers
            raise EngineError(f"CLIP: fallo al puntuar la imagen: {exc}") from exc

    def _get_model(self):
        if self._model is not None and self._processor is not None:
            return self._model, self._processor
        if not _instalado("torch", "transformers"):
            raise EngineError(
                "transformers/torch no están instalados. Instálalos con: "
                "pip install transformers torch pillow"
            )
        try:
            _torch, CLIPModel, CLIPProcessor = _cargar_clip()
            self._model = CLIPModel.from_pretrained(self._model_name)
            self._processor = CLIPProcessor.from_pretrained(self._model_name)
        except Exception as exc:  # noqa: BLE001 - descarga/carga del modelo
            raise EngineError(
                f"CLIP: no se pudo cargar '{self._model_name}': {exc}"
            ) from exc
        return self._model, self._processor


class PaddleOcrEngine(PerceptionEngine):
    """OCR local con PaddleOCR (DBNet + CRNN), pensado para lotes grandes.

    Es la alternativa rápida al OCR por VLM de ``ollama_vision``: transcribe
    mucho más rápido y sin razonamiento, a cambio de no entender nada. El
    ``ocr`` es inyectable para los tests.
    """

    def __init__(self, lang: str = "latin", use_gpu: bool = False, ocr=None) -> None:
        self._lang = lang if lang in OCR_LANGS else "latin"
        self._use_gpu = bool(use_gpu)
        self._ocr = ocr

    @property
    def id(self) -> str:
        return "paddle_ocr"

    @property
    def nombre(self) -> str:
        return "OCR local (PaddleOCR)"

    def is_available(self) -> bool:
        return self._ocr is not None or _instalado("paddleocr")

    def list_models(self) -> list[str]:
        return list(OCR_LANGS)

    def perceive(self, image_path: str | Path, prompt: str = "") -> Perception:
        """Extrae el texto de ``image_path`` (el prompt se ignora)."""
        ocr = self._get_ocr()
        path = Path(image_path)
        if not path.is_file():
            raise EngineError(f"La imagen no existe: {path}")
        logger.info("OCR local: transcribiendo '%s'...", path.name)
        try:
            raw = _run_ocr(ocr, str(path))
        except EngineError:
            raise
        except Exception as exc:  # noqa: BLE001 - paddle/opencv
            raise EngineError(f"OCR local: fallo al transcribir: {exc}") from exc
        return Perception(text=parse_ocr_output(raw))

    # ---- Internos --------------------------------------------------

    def _get_ocr(self):
        if self._ocr is not None:
            return self._ocr
        if not _instalado("paddleocr"):
            raise EngineError(
                "paddleocr no está instalado. Instálalo con: pip install paddleocr"
            )
        try:
            self._ocr = _cargar_paddle()(lang=self._lang, use_angle_cls=True)
        except Exception as exc:  # noqa: BLE001 - variantes de la API
            raise EngineError(f"OCR local: no se pudo inicializar: {exc}") from exc
        return self._ocr


def _run_ocr(ocr, image_path: str):
    """Llama a PaddleOCR tolerando las dos API (2.x ``ocr`` / 3.x ``predict``)."""
    if hasattr(ocr, "predict"):
        return ocr.predict(image_path)
    return ocr.ocr(image_path, cls=True)


def parse_ocr_output(raw) -> str:
    """Extrae el texto de la salida de PaddleOCR (2.x y 3.x).

    - 2.x: ``[[[caja, (texto, score)], ...]]`` — listas anidadas por página.
    - 3.x: ``[resultado]`` con ``resultado["rec_texts"]`` (dict o atributo).

    El recorrido de la forma 2.x es **recursivo** a propósito: el número de
    niveles cambia entre versiones (y con varias imágenes) y buscar el patrón
    ``[caja, (texto, score)]`` en cualquier profundidad evita depender de la
    forma exacta.
    """
    if not raw:
        return ""
    textos = _rec_texts_from(raw)
    if textos:
        return "\n".join(textos)
    recogidos: list[str] = []
    _collect_pairs(raw, recogidos)
    return "\n".join(recogidos)


def _rec_texts_from(raw) -> list[str]:
    """Forma 3.x: recoge ``rec_texts`` de dicts u objetos."""
    salida: list[str] = []
    items = raw if isinstance(raw, (list, tuple)) else [raw]
    for item in items:
        rec_texts = None
        if isinstance(item, dict):
            rec_texts = item.get("rec_texts")
        elif hasattr(item, "get"):
            try:
                rec_texts = item.get("rec_texts")
            except Exception:  # noqa: BLE001 - get con firma inesperada
                rec_texts = None
        if rec_texts is None:
            rec_texts = getattr(item, "rec_texts", None)
        if rec_texts:
            salida.extend(str(t) for t in rec_texts if str(t).strip())
    return salida


def _collect_pairs(node, out: list[str]) -> None:
    """Busca pares ``[caja, (texto, score)]`` a cualquier profundidad."""
    if not isinstance(node, (list, tuple)):
        return
    if (
        len(node) == 2
        and isinstance(node[1], (list, tuple))
        and len(node[1]) >= 1
        and isinstance(node[1][0], str)
    ):
        texto = node[1][0].strip()
        if texto:
            out.append(texto)
        return
    for child in node:
        _collect_pairs(child, out)


def _first(value):
    """Primer elemento de un tensor/lista; el propio valor si no es indexable."""
    try:
        return value[0]
    except (TypeError, IndexError, KeyError):
        return value


def _objects_from(results) -> list[dict]:
    """Convierte los resultados de Ultralytics en objetos serializables.

    Se accede por atributos (``boxes``, ``cls``, ``conf``, ``xyxy``) y con
    ``getattr`` defensivo: el formato interno de Ultralytics cambia entre
    versiones y aquí no debe reventar por eso.
    """
    objects: list[dict] = []
    for result in results or ():
        names = getattr(result, "names", None) or {}
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            cls_value = getattr(box, "cls", None)
            conf_value = getattr(box, "conf", None)
            xyxy = getattr(box, "xyxy", None)
            if cls_value is None:
                continue
            index = int(_first(cls_value))
            label = names.get(index, str(index)) if isinstance(names, dict) else str(index)
            score = float(_first(conf_value)) if conf_value is not None else 0.0
            caja = tuple(float(v) for v in _first(xyxy)) if xyxy is not None else ()
            objects.append(
                {"label": label, "score": round(score, 4), "box": caja}
            )
    return objects
