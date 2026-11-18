"""Motor de **clonado de voz** local con XTTS-v2 (Coqui TTS).

Clonado *zero-shot*: con unos segundos de audio de referencia, sintetiza texto en
la voz clonada. La muestra de referencia vive en ``voices/clones/`` (o se pasa la
ruta de un fichero).

**Dos backends**, porque Coqui aún no soporta Python 3.12 (el de la app):

- ``subprocess`` (recomendado): la síntesis la hace ``tools/xtts_worker.py``
  ejecutado con el **intérprete del venv de coqui-tts** (Python 3.11). La app no
  importa ``TTS`` ni ``torch``: sólo lanza un proceso y recoge el WAV. Así el
  motor funciona aunque el intérprete de Imperator no pueda cargar Coqui.
- ``inprocess``: importa ``TTS`` en el propio proceso (útil si algún día la app
  corre en un intérprete compatible).

El backend se elige en ``engines.json`` (``parametros.backend``). Si el intérprete
o el worker no existen, ``is_available()`` devuelve ``False`` y el motor aparece
como no disponible, sin romper el arranque.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import threading
from datetime import datetime
from pathlib import Path

from core.subprocess_utils import no_window_kwargs
from engines.base_engine import AudioEngine, EngineError

logger = logging.getLogger(__name__)

# Coqui (TTS) se importa **de forma perezosa**: cargarlo trae torch+transformers
# (~10 s) y, si está instalado en el venv de la app con otra versión de
# transformers, rompería el arranque ligero. Sólo se paga si de verdad se usa el
# backend ``inprocess``.
_TTS_CLASS = None
_TTS_TRIED = False


def _load_tts():
    """Devuelve la clase ``TTS`` de coqui-tts, o ``None`` si no está disponible."""
    global _TTS_CLASS, _TTS_TRIED
    if not _TTS_TRIED:
        _TTS_TRIED = True
        try:
            from TTS.api import TTS as _TTS

            _TTS_CLASS = _TTS
        except Exception:  # noqa: BLE001 - ImportError o choque de versiones
            _TTS_CLASS = None
    return _TTS_CLASS

#: Raíz del proyecto (C:\Imperator); voices/, tools/ y output/ cuelgan de aquí.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Modelo XTTS-v2 multilingüe (incluye español).
DEFAULT_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

#: Idioma por defecto de la síntesis (código ISO 639-1).
DEFAULT_LANGUAGE = "es"

#: Carpeta por defecto de muestras de voz de referencia.
DEFAULT_VOICES_DIR = "voices/clones"

#: Backend por defecto al construir el motor a mano (tests, usos mínimos).
DEFAULT_BACKEND = "inprocess"

#: Intérprete por defecto del venv de coqui-tts (Python 3.11) y worker del backend
#: por subprocess (rutas relativas a la raíz del proyecto).
DEFAULT_PYTHON = "venv-tts/Scripts/python.exe"
DEFAULT_WORKER = "tools/xtts_worker.py"

#: Timeout de una síntesis por subprocess (segundos): la primera carga del modelo
#: y su descarga pueden tardar; finito para que un worker atascado no cuelgue.
DEFAULT_TIMEOUT = 900

#: Extensiones de audio admitidas como muestra de referencia.
REFERENCE_SUFFIXES = (".wav", ".mp3", ".flac", ".m4a", ".ogg")


class VoiceCloneEngine(AudioEngine):
    """Sintetiza texto con una voz clonada a partir de una muestra de referencia."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        language: str = DEFAULT_LANGUAGE,
        device: str = "cuda",
        voices_dir: str | Path = DEFAULT_VOICES_DIR,
        output_dir: str | Path | None = None,
        backend: str = DEFAULT_BACKEND,
        python: str | Path | None = None,
        worker: str | Path | None = None,
        timeout: float | None = None,
    ) -> None:
        self._model_name = model_name
        self._language = (language or DEFAULT_LANGUAGE).lower()
        self._device = device or "cuda"
        self._backend = (backend or DEFAULT_BACKEND).lower()
        self._voices_dir = self._resolve(voices_dir)
        self._python = python
        self._worker = worker
        self._timeout = float(timeout) if timeout is not None else DEFAULT_TIMEOUT
        self._output_dir = (
            Path(output_dir) if output_dir is not None else PROJECT_ROOT / "output"
        )
        # Modelo en proceso (sólo backend ``inprocess``): se carga al primer uso y
        # se reutiliza; la instancia es única, así que va bajo lock.
        self._model = None
        self._lock = threading.Lock()

    @staticmethod
    def _resolve(path: str | Path) -> Path:
        p = Path(path)
        return p if p.is_absolute() else PROJECT_ROOT / p

    # ---- Contrato Engine -------------------------------------------

    @property
    def id(self) -> str:
        return "voice_clone"

    @property
    def nombre(self) -> str:
        return "Clonar voz (XTTS-v2)"

    @property
    def backend(self) -> str:
        return self._backend

    def is_available(self) -> bool:
        """``True`` si el backend elegido puede funcionar en esta máquina."""
        if self._backend == "subprocess":
            return self._python_path() is not None and self._worker_path() is not None
        return _load_tts() is not None

    def list_models(self) -> list[str]:
        return [self._model_name]

    def list_voices(self) -> list[str]:
        """Nombres de las muestras de referencia disponibles (``voices/clones``)."""
        return sorted(p.stem for p in self.list_reference_samples())

    def list_reference_samples(self) -> list[Path]:
        if not self._voices_dir.is_dir():
            return []
        muestras = [
            p
            for p in self._voices_dir.iterdir()
            if p.is_file() and p.suffix.lower() in REFERENCE_SUFFIXES
        ]
        return sorted(muestras)

    # ---- Síntesis --------------------------------------------------

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float | None = None,
        *,
        language: str | None = None,
        output_path: str | Path | None = None,
    ) -> Path:
        """Clona ``voice`` (muestra o ruta) para decir ``text`` y devuelve el WAV."""
        texto = (text or "").strip()
        if not texto:
            raise EngineError("El texto a sintetizar está vacío.")

        referencia = self._resolve_reference(voice)
        idioma = (language or self._language or DEFAULT_LANGUAGE).lower()

        if output_path is None:
            output_path = self._default_output_path(referencia.stem)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self._backend == "subprocess":
            return self._synthesize_subprocess(texto, referencia, idioma, speed, output_path)
        return self._synthesize_inprocess(texto, referencia, idioma, speed, output_path)

    # ---- Backend: subprocess ---------------------------------------

    def _python_path(self) -> Path | None:
        """Ruta del intérprete del venv de coqui-tts, o ``None`` si no se encuentra."""
        if not self._python:
            return None
        candidato = self._resolve(self._python)
        if candidato.is_file():
            return candidato
        encontrado = shutil.which(str(self._python))
        return Path(encontrado) if encontrado else None

    def _worker_path(self) -> Path | None:
        ruta = self._resolve(self._worker or DEFAULT_WORKER)
        return ruta if ruta.is_file() else None

    def _synthesize_subprocess(
        self, texto: str, referencia: Path, idioma: str, speed, output_path: Path
    ) -> Path:
        python = self._python_path()
        worker = self._worker_path()
        if python is None:
            raise EngineError(
                "XTTS (subprocess): no se encuentra el intérprete del venv de "
                f"coqui-tts ({self._python or DEFAULT_PYTHON}). Créalo con Python "
                "3.11 e instala 'coqui-tts' + 'torch', o ajusta 'python' en engines.json."
            )
        if worker is None:
            raise EngineError(
                f"XTTS (subprocess): no se encuentra el worker {DEFAULT_WORKER}."
            )
        argv = [
            str(python), str(worker),
            "--text", texto,
            "--speaker", str(referencia),
            "--out", str(output_path),
            "--language", idioma,
            "--model", self._model_name,
            "--device", self._device,
        ]
        if speed is not None:
            try:
                argv += ["--speed", str(float(speed))]
            except (TypeError, ValueError):
                pass

        logger.info("XTTS: sintetizando por subprocess (%s, idioma=%s)...", python, idioma)
        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self._timeout,
                **no_window_kwargs(),
            )
        except subprocess.TimeoutExpired as exc:
            raise EngineError(
                f"XTTS: la síntesis no terminó en {self._timeout:g}s."
            ) from exc
        except OSError as exc:
            raise EngineError(f"XTTS: no se pudo ejecutar el worker: {exc}") from exc

        detalle = self._worker_result(proc.stdout)
        if proc.returncode != 0 or not detalle.get("ok", False):
            motivo = detalle.get("error") or (proc.stderr or proc.stdout or "").strip()[-500:]
            raise EngineError(f"XTTS: la síntesis falló. {motivo}")
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise EngineError("XTTS no generó el audio de salida.")
        logger.info("XTTS: audio guardado en '%s'.", output_path)
        return output_path

    @staticmethod
    def _worker_result(stdout: str) -> dict:
        """Última línea JSON del worker (o ``{}`` si no hay JSON)."""
        for linea in reversed((stdout or "").splitlines()):
            linea = linea.strip()
            if linea.startswith("{"):
                try:
                    data = json.loads(linea)
                except ValueError:
                    continue
                if isinstance(data, dict):
                    return data
        return {}

    # ---- Backend: in-process ---------------------------------------

    def _synthesize_inprocess(
        self, texto: str, referencia: Path, idioma: str, speed, output_path: Path
    ) -> Path:
        tts_cls = _load_tts()
        if tts_cls is None:
            raise EngineError(
                "XTTS no está instalado en este intérprete. Usa el backend "
                "'subprocess' con un venv aparte (engines.json), o instala "
                "'coqui-tts' + 'torch' aquí."
            )
        model = self._get_model()
        kwargs: dict = {
            "text": texto,
            "speaker_wav": str(referencia),
            "language": idioma,
            "file_path": str(output_path),
        }
        if speed is not None:
            try:
                kwargs["speed"] = float(speed)
            except (TypeError, ValueError):
                pass
        logger.info("XTTS: sintetizando (%s, idioma=%s)...", self._model_name, idioma)
        try:
            model.tts_to_file(**kwargs)
        except Exception as exc:  # noqa: BLE001 - backend torch/IO
            raise EngineError(f"XTTS: no se pudo sintetizar: {exc}") from exc
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise EngineError("XTTS no generó el audio de salida.")
        logger.info("XTTS: audio guardado en '%s'.", output_path)
        return output_path

    # ---- Comunes ---------------------------------------------------

    def _resolve_reference(self, voice: str | None) -> Path:
        """Resuelve la muestra de referencia (ruta directa o nombre en voices/clones)."""
        if voice:
            directa = Path(voice)
            if directa.is_file():
                return directa
            for ext in REFERENCE_SUFFIXES:
                candidata = self._voices_dir / f"{voice}{ext}"
                if candidata.is_file():
                    return candidata
            raise EngineError(
                f"XTTS: muestra de voz no encontrada: {voice!r} "
                f"(se busca en {self._voices_dir})."
            )
        muestras = self.list_reference_samples()
        if not muestras:
            raise EngineError(
                "XTTS: añade al menos una muestra de voz de referencia en "
                f"{self._voices_dir} (unos segundos de audio)."
            )
        if len(muestras) == 1:
            return muestras[0]
        raise EngineError(
            "XTTS: hay varias muestras de referencia; elige una en el desplegable."
        )

    def _default_output_path(self, referencia: str) -> Path:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9_-]", "_", referencia)
        marca = f"{datetime.now():%Y%m%d_%H%M%S}"
        return self._output_dir / f"plataformaia_clon_{marca}_{safe}.wav"

    def _get_model(self):
        """Devuelve el modelo XTTS cargado o lo carga (sólo backend ``inprocess``)."""
        with self._lock:
            if self._model is not None:
                return self._model
            # Evita que Coqui se quede pidiendo aceptar la licencia por consola.
            os.environ.setdefault("COQUI_TOS_AGREED", "1")
            tts_cls = _load_tts()
            if tts_cls is None:
                raise EngineError(
                    "XTTS no está instalado en este intérprete. Usa el backend "
                    "'subprocess' (engines.json)."
                )
            logger.info(
                "XTTS: cargando '%s' (device=%s)...", self._model_name, self._device
            )
            try:
                modelo = tts_cls(self._model_name, progress_bar=False)
                if hasattr(modelo, "to"):
                    modelo = modelo.to(self._device)
            except Exception as exc:  # noqa: BLE001 - torch/OOM/descarga
                raise EngineError(
                    f"XTTS: no se pudo cargar el modelo {self._model_name!r}: {exc}"
                ) from exc
            self._model = modelo
            return modelo
