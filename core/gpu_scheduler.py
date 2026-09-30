"""Planificador de GPU: cola con prioridad y muestreo real de VRAM.

Sustituye las heurísticas de VRAM (``unload_on_tab_change``, ``POST /free``) por
datos: pregunta a ``nvidia-smi`` cuánta memoria hay libre y decide si un trabajo
puede empezar ahora o debe esperar, con tres niveles de prioridad.

Prioridades (menor número = antes):

- :attr:`JobPriority.PERCEPTION`: trabajos cortos que alimentan al usuario en el
  momento (OCR de una página, detección de objetos, transcripción de un clip).
- :attr:`JobPriority.INTERACTIVE`: lo que el usuario está pidiendo ahora (chat,
  resumen de un documento).
- :attr:`JobPriority.BATCH`: trabajos largos que pueden esperar (generación de
  vídeo, comprensión de un vídeo completo).

Decisiones de diseño:

- **Lógica pura y thread-safe.** El sondeo real de ``nvidia-smi`` vive detrás de
  :class:`GpuProbe`, que es inyectable: los tests no necesitan GPU.
- **Fallo abierto (fail-open).** Sin ``nvidia-smi`` (o desactivado) el
  planificador **nunca** bloquea: devuelve que sí se puede empezar. Un
  planificador que rompiera la app por no poder medir sería peor que no tenerlo.
- **No ejecuta trabajos.** Aquí solo se decide *si* pueden empezar; quien los
  lanza (workers de la UI, motores) pide permiso con :meth:`GpuScheduler.acquire`
  y lo devuelve con :meth:`GpuScheduler.release` (o usa el contexto
  :meth:`GpuScheduler.job`).
"""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, Iterator

logger = logging.getLogger(__name__)

#: Tiempo máximo (s) que se espera a ``nvidia-smi``. Si no responde, se asume
#: GPU desconocida y el planificador falla abierto.
NVIDIA_SMI_TIMEOUT = 5

#: Campos que se le piden a ``nvidia-smi`` (CSV sin cabecera, en MiB).
NVIDIA_SMI_QUERY = "name,memory.total,memory.used,utilization.gpu"


class JobPriority(IntEnum):
    """Prioridad de un trabajo de GPU. Menor número = antes."""

    PERCEPTION = 0
    INTERACTIVE = 1
    BATCH = 2

    @property
    def label(self) -> str:
        return {
            JobPriority.PERCEPTION: "percepción",
            JobPriority.INTERACTIVE: "interactivo",
            JobPriority.BATCH: "lote",
        }[self]


#: Prioridad por defecto de cada tarea de la plataforma. El chat y los
#: documentos son interactivos; imagen, vídeo y comprensión son lotes que pueden
#: esperar a que la GPU esté menos ocupada.
TASK_PRIORITY: dict[str, JobPriority] = {
    "chat": JobPriority.INTERACTIVE,
    "document": JobPriority.INTERACTIVE,
    "image": JobPriority.BATCH,
    "video": JobPriority.BATCH,
    "understanding": JobPriority.BATCH,
    "audio": JobPriority.INTERACTIVE,
    "assembler": JobPriority.BATCH,
}


def priority_for(task_id: str) -> JobPriority:
    """Prioridad por defecto de ``task_id`` (interactiva si no está en el mapa)."""
    return TASK_PRIORITY.get(task_id, JobPriority.INTERACTIVE)


@dataclass(frozen=True)
class GpuSnapshot:
    """Estado puntual de la GPU."""

    available: bool = False
    name: str = ""
    total_mb: int = 0
    used_mb: int = 0
    util_percent: int = 0

    @property
    def free_mb(self) -> int:
        return max(0, self.total_mb - self.used_mb)

    @property
    def used_ratio(self) -> float:
        if self.total_mb <= 0:
            return 0.0
        return self.used_mb / self.total_mb

    def describe(self) -> str:
        """Frase legible para la barra de estado o un diálogo."""
        if not self.available:
            return "GPU: sin datos (nvidia-smi no disponible)"
        return (
            f"GPU: {self.name or 'desconocida'} · "
            f"{self.free_mb} MB libres de {self.total_mb} MB · "
            f"uso {self.util_percent}%"
        )


class GpuProbe:
    """Lee el estado de la GPU con ``nvidia-smi``.

    ``runner`` y ``timeout`` son inyectables para poder testear el parseo sin
    ejecutar nada. El runner debe devolver un objeto con ``stdout``/``returncode``
    (como ``subprocess.CompletedProcess``).
    """

    def __init__(
        self,
        runner: Callable[..., object] | None = None,
        timeout: int = NVIDIA_SMI_TIMEOUT,
    ) -> None:
        self._runner = runner if runner is not None else self._run
        self._timeout = int(timeout)

    def _run(self, cmd: list[str], timeout: int) -> object:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

    @staticmethod
    def _csv_command() -> list[str]:
        return [
            "nvidia-smi",
            f"--query-gpu={NVIDIA_SMI_QUERY}",
            "--format=csv,noheader,nounits",
        ]

    def snapshot(self) -> GpuSnapshot:
        """Devuelve el estado de la primera GPU; ``available=False`` si no se pudo."""
        try:
            proc = self._runner(self._csv_command(), self._timeout)
        except (OSError, subprocess.SubprocessError) as exc:
            logger.debug("No se pudo ejecutar nvidia-smi: %s", exc)
            return GpuSnapshot()

        output = str(getattr(proc, "stdout", "") or "")
        parsed = self.parse(output)
        if parsed is None:
            logger.debug("nvidia-smi no devolvió datos utilizables: %r", output[:200])
            return GpuSnapshot()
        return parsed

    @staticmethod
    def parse(csv_text: str) -> GpuSnapshot | None:
        """Parsea la salida CSV de ``nvidia-smi``; ``None`` si no es utilizable.

        Formato esperado (una GPU por línea, MiB):
        ``NVIDIA GeForce RTX 5080, 16303, 4096, 12``
        """
        line = next((ln for ln in (csv_text or "").splitlines() if ln.strip()), "")
        if not line:
            return None
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            return None

        def as_int(value: str) -> int:
            try:
                return int(float(value))
            except (TypeError, ValueError):
                return 0

        total = as_int(parts[1])
        if total <= 0:
            return None
        return GpuSnapshot(
            available=True,
            name=parts[0],
            total_mb=total,
            used_mb=as_int(parts[2]),
            util_percent=as_int(parts[3]),
        )


@dataclass(frozen=True)
class SchedulerConfig:
    """Ajustes del planificador (leídos de ``settings.json``)."""

    enabled: bool = True
    max_concurrent: int = 2
    vram_headroom_mb: int = 1024
    defer_batch_above_util: int = 70
    defer_batch_below_free_mb: int = 2048

    @classmethod
    def from_settings(cls, settings: dict | None) -> "SchedulerConfig":
        """Construye la config desde ``settings.gpu_scheduler`` (tolerante a fallos)."""
        raw = (settings or {}).get("gpu_scheduler")
        if not isinstance(raw, dict):
            return cls()

        def as_int(key: str, default: int) -> int:
            try:
                return int(raw.get(key, default))
            except (TypeError, ValueError):
                return default

        return cls(
            enabled=bool(raw.get("enabled", True)),
            max_concurrent=max(1, as_int("max_concurrent", 2)),
            vram_headroom_mb=max(0, as_int("vram_headroom_mb", 1024)),
            defer_batch_above_util=as_int("defer_batch_above_util", 70),
            defer_batch_below_free_mb=max(
                0, as_int("defer_batch_below_free_mb", 2048)
            ),
        )

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "max_concurrent": self.max_concurrent,
            "vram_headroom_mb": self.vram_headroom_mb,
            "defer_batch_above_util": self.defer_batch_above_util,
            "defer_batch_below_free_mb": self.defer_batch_below_free_mb,
        }


@dataclass(frozen=True)
class ActiveJob:
    """Trabajo que tiene una reserva de GPU en este momento."""

    name: str
    priority: JobPriority
    vram_mb: int = 0


class GpuScheduler:
    """Decide qué trabajos pueden usar la GPU ahora.

    Es thread-safe (``threading.Lock``): los workers de la UI piden permiso desde
    hilos distintos.
    """

    def __init__(
        self,
        probe: GpuProbe | None = None,
        config: SchedulerConfig | None = None,
    ) -> None:
        self._probe = probe if probe is not None else GpuProbe()
        self._config = config if config is not None else SchedulerConfig()
        self._lock = threading.Lock()
        self._active: dict[str, ActiveJob] = {}

    @classmethod
    def from_settings(
        cls, settings: dict | None, probe: GpuProbe | None = None
    ) -> "GpuScheduler":
        return cls(probe=probe, config=SchedulerConfig.from_settings(settings))

    @property
    def config(self) -> SchedulerConfig:
        return self._config

    # ---- Estado ----------------------------------------------------

    def snapshot(self) -> GpuSnapshot:
        """Estado actual de la GPU (nunca lanza: falla abierto)."""
        try:
            return self._probe.snapshot()
        except Exception as exc:  # noqa: BLE001 - medir no debe romper nada
            logger.debug("Fallo al sondear la GPU: %s", exc)
            return GpuSnapshot()

    def active_jobs(self) -> list[ActiveJob]:
        """Trabajos con reserva, ordenados por prioridad y nombre."""
        with self._lock:
            return sorted(
                self._active.values(), key=lambda job: (int(job.priority), job.name)
            )

    def is_active(self, name: str) -> bool:
        with self._lock:
            return name in self._active

    # ---- Decisión --------------------------------------------------

    def can_start(
        self, priority: JobPriority, vram_mb: int = 0
    ) -> tuple[bool, str]:
        """¿Puede empezar un trabajo? Devuelve ``(sí/no, motivo)``.

        El motivo se usa tal cual en logs y avisos, así que es una frase. El
        sondeo de la GPU se hace **fuera** del lock: ``nvidia-smi`` puede tardar
        segundos y bloquearía al resto de workers que piden permiso.
        """
        with self._lock:
            active_count = len(self._active)
        return self._decide(active_count, self.snapshot(), priority, vram_mb)

    def _decide(
        self,
        active_count: int,
        snapshot: GpuSnapshot,
        priority: JobPriority,
        vram_mb: int,
    ) -> tuple[bool, str]:
        """Regla de decisión pura: sin lock y sin I/O, todo se pasa por parámetro."""
        if not self._config.enabled:
            return True, "planificador desactivado"

        if active_count >= self._config.max_concurrent:
            return False, (
                f"ya hay {active_count} trabajos en la GPU "
                f"(máximo {self._config.max_concurrent})"
            )

        if not snapshot.available:
            # Sin medida fiable no se bloquea: fallo abierto.
            return True, "sin datos de GPU (se permite)"

        needed = max(0, int(vram_mb)) + self._config.vram_headroom_mb
        if snapshot.free_mb < needed:
            return False, (
                f"quedan {snapshot.free_mb} MB libres y se necesitan "
                f"{needed} MB (reserva de {self._config.vram_headroom_mb} MB)"
            )

        if priority == JobPriority.BATCH and self._gpu_looks_busy(snapshot):
            return False, (
                f"la GPU está ocupada (uso {snapshot.util_percent}%, "
                f"{snapshot.free_mb} MB libres); los trabajos de lote esperan"
            )

        return True, "hay hueco"

    def _gpu_looks_busy(self, snapshot: GpuSnapshot) -> bool:
        """¿Está la GPU lo bastante ocupada como para aplazar un lote?"""
        return (
            snapshot.util_percent >= self._config.defer_batch_above_util
            or snapshot.free_mb <= self._config.defer_batch_below_free_mb
        )

    def should_defer(self, priority: JobPriority) -> bool:
        """Atajo: ``True`` si un trabajo de esa prioridad debería esperar."""
        ok, _reason = self.can_start(priority)
        return not ok

    # ---- Reservas --------------------------------------------------

    def acquire(
        self,
        name: str,
        priority: JobPriority | None = None,
        vram_mb: int = 0,
    ) -> bool:
        """Reserva GPU para ``name``. ``False`` si aún no puede empezar.

        Si ``name`` ya tiene una reserva, devuelve ``True`` sin duplicarla
        (idempotente): evita fugas cuando un widget reintenta.
        """
        priority = priority if priority is not None else priority_for(name)
        with self._lock:
            if name in self._active:
                return True
            active_count = len(self._active)
        # El sondeo de la GPU se hace fuera del lock (puede tardar segundos).
        ok, reason = self._decide(active_count, self.snapshot(), priority, vram_mb)
        if not ok:
            logger.info("GPU: '%s' aplazado — %s", name, reason)
            return False
        with self._lock:
            if name in self._active:
                return True
            if len(self._active) >= self._config.max_concurrent:
                # Otro hilo reservó mientras se sondeaba la GPU.
                return False
            self._active[name] = ActiveJob(
                name=name, priority=priority, vram_mb=max(0, int(vram_mb))
            )
        logger.debug("GPU: '%s' (%s) empieza", name, priority.label)
        return True

    def release(self, name: str) -> None:
        """Devuelve la reserva de ``name`` (no falla si no la tenía)."""
        with self._lock:
            job = self._active.pop(name, None)
        if job is not None:
            logger.debug("GPU: '%s' termina", name)

    def reserve(
        self,
        name: str,
        priority: JobPriority | None = None,
        vram_mb: int = 0,
        timeout: float = 0.0,
        poll: float = 0.5,
    ) -> bool:
        """Reserva, esperando hasta ``timeout`` segundos a que haya hueco.

        ``timeout=0`` es una sola tentativa (comportamiento de :meth:`acquire`).
        """
        deadline = time.monotonic() + max(0.0, float(timeout))
        while True:
            if self.acquire(name, priority, vram_mb):
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(max(0.05, float(poll)))

    @contextmanager
    def job(
        self,
        name: str,
        priority: JobPriority | None = None,
        vram_mb: int = 0,
        wait: float = 0.0,
    ) -> Iterator[bool]:
        """Contexto que reserva y libera GPU, y dice si se pudo empezar.

        Uso::

            with scheduler.job("document", vram_mb=6000) as ok:
                if not ok:
                    return   # se puede reintentar más tarde
                ...trabajo pesado...

        Libera la reserva aunque el cuerpo lance una excepción.
        """
        started = self.reserve(name, priority, vram_mb, timeout=wait)
        try:
            yield started
        finally:
            if started:
                self.release(name)

    # ---- Diagnóstico ----------------------------------------------

    def status_line(self) -> str:
        """Línea corta para la barra de estado."""
        snapshot = self.snapshot()
        if not snapshot.available:
            return "GPU: sin datos"
        active = len(self.active_jobs())
        suffix = f" · {active} trabajo(s)" if active else ""
        return (
            f"GPU {snapshot.free_mb} MB libres ({snapshot.util_percent}%){suffix}"
        )

    def describe(self) -> str:
        """Informe multilínea del estado y los trabajos activos."""
        lines = [self.snapshot().describe()]
        lines.append(
            f"Planificador: {'activo' if self._config.enabled else 'desactivado'} "
            f"· máximo {self._config.max_concurrent} trabajos · "
            f"reserva {self._config.vram_headroom_mb} MB"
        )
        jobs = self.active_jobs()
        if jobs:
            lines.append("Trabajos en curso:")
            lines.extend(
                f"  - {job.name} ({job.priority.label}"
                + (f", ~{job.vram_mb} MB" if job.vram_mb else "")
                + ")"
                for job in jobs
            )
        else:
            lines.append("Trabajos en curso: ninguno")
        return "\n".join(lines)
