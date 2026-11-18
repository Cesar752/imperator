"""Ejecución de herramientas de línea de comandos **externas**.

Permite integrar en Imperator programas que no exponen API ni son bibliotecas
Python: el motor les pasa la entrada mediante una **plantilla de argumentos**
(``{prompt}``, ``{input}``, ``{output}``, ``{output_dir}``, ``{model}``) y recoge
el artefacto que generan.

Convenciones heredadas de :mod:`core.video_assembler` y
:mod:`core.frame_extractor`:

- **Binario resuelto por PATH o ruta absoluta.** No se instala nada por su
  cuenta: si el ejecutable no está, el motor aparece como no disponible.
- **``timeout`` y kill.** ``subprocess.run`` mata el proceso al expirar; una CLI
  externa atascada no puede colgar el hilo del worker.
- **Consola oculta.** Imperator corre con ``pythonw.exe``: se usa
  :func:`core.subprocess_utils.no_window_kwargs` para que no parpadee una ventana
  de consola en cada invocación.
- **Se comprueba la salida.** Un código de retorno distinto de cero o un
  artefacto ausente se tratan como error de motor.

Es lógica pura, sin dependencias de PyQt6, para poder testarla sin GUI.
"""

from __future__ import annotations

import logging
import shlex
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from core.subprocess_utils import no_window_kwargs
from engines.base_engine import EngineError

logger = logging.getLogger(__name__)

#: Raíz del proyecto (C:\Imperator); ``output/`` cuelga de aquí.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Carpetas del propio proyecto donde buscar ejecutables instalados "en local"
#: (todo en el equipo, sin depender del PATH ni de rutas absolutas). Si existe
#: el mismo nombre con extensión de Windows (``.exe``/``.bat``/...), se resuelve
#: igual: la búsqueda delega en :func:`shutil.which`, que aplica ``PATHEXT``.
LOCAL_BIN_DIRS: tuple[str, ...] = ("tools", "bin")

#: Timeout por defecto (segundos) de una invocación externa.
DEFAULT_TIMEOUT = 600

#: Extensiones que se consideran artefacto al resolver la salida por glob.
_ARTIFACT_SUFFIXES = (
    ".mp4", ".mkv", ".mov", ".webm", ".gif",
    ".png", ".jpg", ".jpeg", ".webp",
    ".wav", ".mp3", ".flac", ".ogg",
    ".srt", ".zip", ".md", ".json", ".txt",
)


def resolve_executable(
    executable: str | None,
    *,
    search_dirs: "list[str] | tuple[str, ...] | None" = None,
) -> str | None:
    """Ruta real del ejecutable, o ``None`` si no se encuentra.

    Busca, en este orden: la ruta absoluta/relativa dada, las carpetas locales
    del proyecto (:data:`LOCAL_BIN_DIRS`, para instalaciones "todo en local") y,
    por último, el ``PATH``. Las carpetas relativas se resuelven contra
    :data:`PROJECT_ROOT`.
    """
    if not executable:
        return None
    candidato = Path(executable)
    if candidato.is_file():
        return str(candidato)
    for carpeta in (LOCAL_BIN_DIRS if search_dirs is None else search_dirs):
        base = Path(carpeta)
        if not base.is_absolute():
            base = PROJECT_ROOT / base
        encontrado = shutil.which(executable, path=str(base))
        if encontrado:
            return encontrado
    return shutil.which(executable)


def _strip_quotes(token: str) -> str:
    """Quita las comillas envolventes de un token (``"{prompt}"`` -> ``{prompt}``)."""
    if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'":
        return token[1:-1]
    return token


def split_args(template: str | list[str] | tuple[str, ...] | None) -> list[str]:
    """Normaliza la plantilla de argumentos a una lista de tokens.

    Acepta una lista ya troceada (lo habitual, y lo que permite el JSON) o una
    cadena que se parte con :func:`shlex.split`. Se usa ``posix=False`` para **no**
    interpretar las barras invertidas como escapes (rutas de Windows como
    ``C:\\ruta``) y luego se quitan las comillas envolventes de cada token.
    """
    if template is None:
        return []
    if isinstance(template, str):
        return [_strip_quotes(t) for t in shlex.split(template, posix=False)]
    return [str(t) for t in template]


@dataclass(frozen=True)
class CliResult:
    """Resultado de una invocación externa."""

    returncode: int
    stdout: str
    stderr: str


class CliRunner:
    """Invoca un ejecutable externo y resuelve el artefacto que produce."""

    def __init__(
        self,
        executable: str | None,
        *,
        label: str = "CLI",
        args: str | list[str] | tuple[str, ...] | None = None,
        output_dir: str | Path | None = None,
        output_ext: str | None = None,
        output_glob: str | None = None,
        timeout: float | None = None,
        cwd: str | Path | None = None,
    ) -> None:
        self._executable = executable
        self._label = label
        self._args = split_args(args)
        self._output_dir = Path(output_dir) if output_dir is not None else None
        self._output_ext = output_ext
        self._output_glob = output_glob
        self._timeout = timeout if timeout is not None else DEFAULT_TIMEOUT
        self._cwd = str(cwd) if cwd is not None else None

    # ---- Consultas -------------------------------------------------

    @property
    def label(self) -> str:
        return self._label

    @property
    def executable(self) -> str | None:
        return self._executable

    @property
    def has_template(self) -> bool:
        return bool(self._args)

    def executable_path(self) -> str | None:
        """Ruta real del ejecutable, o ``None`` si no está instalado."""
        return resolve_executable(self._executable)

    def is_available(self) -> bool:
        return self.executable_path() is not None

    # ---- Invocación ------------------------------------------------

    def _default_output_dir(self) -> Path:
        return self._output_dir if self._output_dir is not None else PROJECT_ROOT / "output"

    def prepare(self, values: dict) -> dict:
        """Completa ``values`` con ``output_dir`` y ``output`` ya resueltos.

        El artefacto se nombra con un identificador único para no pisar corridas
        anteriores; el ejecutable externo escribe en ``{output}`` si su CLI lo
        admite (o en ``{output_dir}`` y se localiza con ``output_glob``).
        """
        values = dict(values)
        out_dir = Path(values.get("output_dir") or self._default_output_dir())
        out_dir.mkdir(parents=True, exist_ok=True)
        values["output_dir"] = str(out_dir)
        if not values.get("output"):
            ext = self._output_ext or ".out"
            slug = "".join(c if c.isalnum() else "_" for c in self._label).strip("_") or "cli"
            values["output"] = str(out_dir / f"{slug}_{uuid.uuid4().hex[:8]}{ext}")
        return values

    def build_args(self, values: dict) -> list[str]:
        """Formatea la plantilla con ``values`` (``KeyError`` descriptivo si falta)."""
        resultado: list[str] = []
        for token in self._args:
            try:
                resultado.append(str(token).format(**values))
            except KeyError as exc:
                raise EngineError(
                    f"La plantilla de argumentos de '{self._label}' usa un hueco "
                    f"desconocido {exc}. Huecos válidos: {', '.join(sorted(values))}."
                ) from exc
        return resultado

    def run(self, values: dict, *, timeout: float | None = None) -> CliResult:
        """Ejecuta el binario con los argumentos formateados y devuelve su salida."""
        exe = self.executable_path()
        if exe is None:
            raise EngineError(
                f"'{self._label}': el ejecutable '{self._executable}' no está "
                "instalado o no se encuentra en el PATH."
            )
        args = self.build_args(values)
        if not args:
            raise EngineError(
                f"'{self._label}': no hay plantilla de argumentos configurada. "
                "Añade 'args' en sus 'parametros' (engines.json)."
            )
        cmd = [exe, *args]
        logger.debug("Ejecutando '%s': %s", self._label, cmd)
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout if timeout is not None else self._timeout,
                cwd=self._cwd,
                **no_window_kwargs(),
            )
        except subprocess.TimeoutExpired as exc:
            raise EngineError(
                f"'{self._label}' no terminó en {timeout or self._timeout}s."
            ) from exc
        except OSError as exc:
            raise EngineError(f"No se pudo ejecutar '{self._label}': {exc}") from exc

        if proc.returncode != 0:
            detalle = (proc.stderr or proc.stdout or "").strip()
            cola = f" {detalle[-800:]}" if detalle else ""
            raise EngineError(
                f"'{self._label}' terminó con código {proc.returncode}.{cola}"
            )
        return CliResult(proc.returncode, proc.stdout or "", proc.stderr or "")

    def resolve_output(self, values: dict, *, since: float | None = None) -> Path:
        """Localiza el artefacto generado (por ``output_glob`` o por ``{output}``)."""
        base = Path(values.get("output_dir") or self._default_output_dir())
        if self._output_glob:
            candidatos = sorted(
                (
                    p
                    for p in base.glob(self._output_glob)
                    if p.is_file() and (since is None or p.stat().st_mtime >= since)
                ),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if candidatos:
                return candidatos[0]
            raise EngineError(
                f"'{self._label}' no generó ningún fichero que case con "
                f"'{self._output_glob}' en '{base}'."
            )
        salida = values.get("output")
        if salida and Path(salida).is_file() and Path(salida).stat().st_size > 0:
            return Path(salida)
        raise EngineError(
            f"'{self._label}' no generó el fichero esperado '{salida}'."
        )

    def invoke(self, values: dict, *, timeout: float | None = None) -> Path:
        """Ejecuta el binario y devuelve la ruta del artefacto producido.

        Es el punto de entrada que usan los motores: prepara ``values`` (añade
        ``output_dir``/``output``), ejecuta y resuelve la salida.
        """
        preparado = self.prepare(values)
        inicio = time.time()
        self.run(preparado, timeout=timeout)
        return self.resolve_output(preparado, since=inicio)
