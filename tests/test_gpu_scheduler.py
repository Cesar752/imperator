"""Tests de core.gpu_scheduler: parseo de nvidia-smi y reglas de decisión.

No tocan ninguna GPU: el sondeo real va detrás de ``GpuProbe``, que aquí se
sustituye por dobles. Es la parte que decide *si* un trabajo puede empezar, así
que las reglas (prioridad, VRAM libre, concurrencia, fallo abierto) se prueban
sin hardware.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import gpu_scheduler as gs  # noqa: E402


class _FakeRunner:
    """Runner de ``nvidia-smi`` de mentira: devuelve un stdout fijo o lanza."""

    class _Proc:
        def __init__(self, stdout: str, returncode: int = 0) -> None:
            self.stdout = stdout
            self.returncode = returncode

    def __init__(self, stdout: str = "", raises: Exception | None = None) -> None:
        self._stdout = stdout
        self._raises = raises
        self.calls: list[list[str]] = []

    def __call__(self, cmd, timeout):
        self.calls.append(list(cmd))
        if self._raises is not None:
            raise self._raises
        return self._Proc(self._stdout)


class _FakeProbe:
    """Probe de mentira: devuelve siempre el mismo snapshot."""

    def __init__(self, snapshot: gs.GpuSnapshot) -> None:
        self._snapshot = snapshot

    def snapshot(self) -> gs.GpuSnapshot:
        return self._snapshot


def _healthy(free_mb: int = 14000, util: int = 5) -> gs.GpuSnapshot:
    total = 16303
    return gs.GpuSnapshot(
        available=True,
        name="NVIDIA GeForce RTX 5080",
        total_mb=total,
        used_mb=total - free_mb,
        util_percent=util,
    )


class GpuSnapshotTest(unittest.TestCase):
    def test_free_mb_no_es_negativo(self) -> None:
        snap = gs.GpuSnapshot(available=True, total_mb=100, used_mb=250)
        self.assertEqual(snap.free_mb, 0)

    def test_used_ratio(self) -> None:
        snap = gs.GpuSnapshot(available=True, total_mb=1000, used_mb=250)
        self.assertAlmostEqual(snap.used_ratio, 0.25)

    def test_used_ratio_sin_total(self) -> None:
        self.assertEqual(gs.GpuSnapshot().used_ratio, 0.0)

    def test_describe_sin_datos(self) -> None:
        self.assertIn("sin datos", gs.GpuSnapshot().describe())

    def test_describe_con_datos(self) -> None:
        texto = _healthy(free_mb=1000, util=40).describe()
        self.assertIn("1000 MB libres", texto)
        self.assertIn("40%", texto)


class GpuProbeParseTest(unittest.TestCase):
    def test_parsea_linea_valida(self) -> None:
        snap = gs.GpuProbe.parse("NVIDIA GeForce RTX 5080, 16303, 4096, 12")
        self.assertIsNotNone(snap)
        assert snap is not None  # para el type checker
        self.assertTrue(snap.available)
        self.assertEqual(snap.name, "NVIDIA GeForce RTX 5080")
        self.assertEqual(snap.total_mb, 16303)
        self.assertEqual(snap.used_mb, 4096)
        self.assertEqual(snap.util_percent, 12)
        self.assertEqual(snap.free_mb, 12207)

    def test_ignora_lineas_vacias_previas(self) -> None:
        snap = gs.GpuProbe.parse("\n\nNVIDIA RTX, 100, 20, 0\n")
        self.assertIsNotNone(snap)

    def test_linea_incompleta(self) -> None:
        self.assertIsNone(gs.GpuProbe.parse("NVIDIA RTX, 100"))

    def test_total_cero_no_es_utilizable(self) -> None:
        self.assertIsNone(gs.GpuProbe.parse("NVIDIA RTX, 0, 0, 0"))

    def test_texto_vacio(self) -> None:
        self.assertIsNone(gs.GpuProbe.parse(""))

    def test_valores_no_numericos_caen_a_cero(self) -> None:
        snap = gs.GpuProbe.parse("NVIDIA RTX, 16000, N/A, N/A")
        self.assertIsNotNone(snap)
        assert snap is not None
        self.assertEqual(snap.used_mb, 0)
        self.assertEqual(snap.util_percent, 0)


class GpuProbeSnapshotTest(unittest.TestCase):
    def test_snapshot_ok(self) -> None:
        probe = gs.GpuProbe(runner=_FakeRunner("NVIDIA RTX, 16000, 2000, 30"))
        snap = probe.snapshot()
        self.assertTrue(snap.available)
        self.assertEqual(snap.free_mb, 14000)

    def test_comando_incluye_la_consulta(self) -> None:
        runner = _FakeRunner("NVIDIA RTX, 16000, 2000, 30")
        gs.GpuProbe(runner=runner).snapshot()
        self.assertIn("nvidia-smi", runner.calls[0][0])
        self.assertIn(f"--query-gpu={gs.NVIDIA_SMI_QUERY}", runner.calls[0])

    def test_sin_nvidia_smi_falla_abierto(self) -> None:
        probe = gs.GpuProbe(runner=_FakeRunner(raises=OSError("no existe")))
        self.assertFalse(probe.snapshot().available)

    def test_salida_inservible(self) -> None:
        probe = gs.GpuProbe(runner=_FakeRunner(""))
        self.assertFalse(probe.snapshot().available)


class SchedulerConfigTest(unittest.TestCase):
    def test_defaults_sin_settings(self) -> None:
        cfg = gs.SchedulerConfig.from_settings(None)
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.max_concurrent, 2)
        self.assertEqual(cfg.vram_headroom_mb, 1024)

    def test_settings_no_dict(self) -> None:
        self.assertEqual(
            gs.SchedulerConfig.from_settings({"gpu_scheduler": "malo"}),
            gs.SchedulerConfig(),
        )

    def test_lee_valores(self) -> None:
        cfg = gs.SchedulerConfig.from_settings(
            {"gpu_scheduler": {"enabled": False, "max_concurrent": 4}}
        )
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.max_concurrent, 4)

    def test_valores_invalidos_caen_a_default(self) -> None:
        cfg = gs.SchedulerConfig.from_settings(
            {"gpu_scheduler": {"max_concurrent": "muchos", "vram_headroom_mb": None}}
        )
        self.assertEqual(cfg.max_concurrent, 2)
        self.assertEqual(cfg.vram_headroom_mb, 1024)

    def test_max_concurrent_se_acota_a_uno(self) -> None:
        cfg = gs.SchedulerConfig.from_settings(
            {"gpu_scheduler": {"max_concurrent": 0}}
        )
        self.assertEqual(cfg.max_concurrent, 1)

    def test_round_trip_dict(self) -> None:
        cfg = gs.SchedulerConfig(enabled=False, max_concurrent=3)
        self.assertEqual(gs.SchedulerConfig.from_settings({"gpu_scheduler": cfg.to_dict()}), cfg)


class PriorityTest(unittest.TestCase):
    def test_orden_de_prioridades(self) -> None:
        self.assertLess(gs.JobPriority.PERCEPTION, gs.JobPriority.INTERACTIVE)
        self.assertLess(gs.JobPriority.INTERACTIVE, gs.JobPriority.BATCH)

    def test_prioridad_por_tarea(self) -> None:
        self.assertEqual(gs.priority_for("chat"), gs.JobPriority.INTERACTIVE)
        self.assertEqual(gs.priority_for("video"), gs.JobPriority.BATCH)

    def test_tarea_desconocida_es_interactiva(self) -> None:
        self.assertEqual(gs.priority_for("lo-que-sea"), gs.JobPriority.INTERACTIVE)

    def test_label(self) -> None:
        self.assertEqual(gs.JobPriority.BATCH.label, "lote")


class GpuSchedulerDecisionTest(unittest.TestCase):
    def test_desactivado_permite_siempre(self) -> None:
        sch = gs.GpuScheduler(
            probe=_FakeProbe(gs.GpuSnapshot()),
            config=gs.SchedulerConfig(enabled=False),
        )
        ok, motivo = sch.can_start(gs.JobPriority.BATCH)
        self.assertTrue(ok)
        self.assertIn("desactivado", motivo)

    def test_sin_datos_falla_abierto(self) -> None:
        sch = gs.GpuScheduler(probe=_FakeProbe(gs.GpuSnapshot()))
        ok, motivo = sch.can_start(gs.JobPriority.BATCH, vram_mb=8000)
        self.assertTrue(ok)
        self.assertIn("sin datos", motivo)

    def test_hay_hueco(self) -> None:
        sch = gs.GpuScheduler(probe=_FakeProbe(_healthy()))
        ok, motivo = sch.can_start(gs.JobPriority.INTERACTIVE)
        self.assertTrue(ok)
        self.assertEqual(motivo, "hay hueco")

    def test_lote_espera_si_la_gpu_esta_ocupada(self) -> None:
        sch = gs.GpuScheduler(probe=_FakeProbe(_healthy(free_mb=3000, util=90)))
        ok_batch, motivo = sch.can_start(gs.JobPriority.BATCH)
        self.assertFalse(ok_batch)
        self.assertIn("lote", motivo)
        ok_inter, _ = sch.can_start(gs.JobPriority.INTERACTIVE)
        self.assertTrue(ok_inter, "lo interactivo no se aplaza por uso alto")

    def test_lote_espera_si_queda_poca_vram(self) -> None:
        sch = gs.GpuScheduler(probe=_FakeProbe(_healthy(free_mb=1500, util=5)))
        ok, motivo = sch.can_start(gs.JobPriority.BATCH)
        self.assertFalse(ok)
        self.assertIn("libres", motivo)

    def test_no_cabe_por_vram_estimada(self) -> None:
        sch = gs.GpuScheduler(probe=_FakeProbe(_healthy(free_mb=3000, util=5)))
        ok, motivo = sch.can_start(gs.JobPriority.INTERACTIVE, vram_mb=4000)
        self.assertFalse(ok)
        self.assertIn("se necesitan", motivo)

    def test_limite_de_concurrencia(self) -> None:
        sch = gs.GpuScheduler(
            probe=_FakeProbe(_healthy()),
            config=gs.SchedulerConfig(max_concurrent=1),
        )
        self.assertTrue(sch.acquire("a"))
        ok, motivo = sch.can_start(gs.JobPriority.PERCEPTION)
        self.assertFalse(ok)
        self.assertIn("máximo 1", motivo)

    def test_modelo_ya_cargado_no_se_bloquea_por_vram(self) -> None:
        # Caso real: el chat fallaba con «quedan 557 MB libres y se necesitan
        # 1024 MB» mientras el modelo que iba a usar era el que llenaba la GPU.
        # Si ya está cargado, la tarea no reserva memoria nueva.
        sch = gs.GpuScheduler(probe=_FakeProbe(_healthy(free_mb=557, util=95)))
        ok_sin, _ = sch.can_start(gs.JobPriority.INTERACTIVE)
        self.assertFalse(ok_sin, "sin el modelo cargado sí se bloquea")
        ok, motivo = sch.can_start(gs.JobPriority.INTERACTIVE, already_loaded=True)
        self.assertTrue(ok)
        self.assertIn("ya está cargado", motivo)

    def test_acquire_propaga_el_modelo_ya_cargado(self) -> None:
        # Sin propagarlo, ``can_start`` pasaba y ``acquire`` volvía a bloquear
        # con el mismo motivo de VRAM y la tarea moría igual.
        sch = gs.GpuScheduler(probe=_FakeProbe(_healthy(free_mb=557, util=95)))
        self.assertTrue(sch.acquire("chat", already_loaded=True))
        self.assertTrue(sch.is_active("chat"))

    def test_modelo_ya_cargado_respeta_la_concurrencia(self) -> None:
        # El atajo del modelo cargado no salta el techo de trabajos a la vez: no
        # es una puerta abierta para meter dos modelos en la misma GPU.
        sch = gs.GpuScheduler(
            probe=_FakeProbe(_healthy(free_mb=557, util=95)),
            config=gs.SchedulerConfig(max_concurrent=1),
        )
        self.assertTrue(sch.acquire("a", already_loaded=True))
        ok, motivo = sch.can_start(gs.JobPriority.INTERACTIVE, already_loaded=True)
        self.assertFalse(ok)
        self.assertIn("máximo 1", motivo)

    def test_should_defer(self) -> None:
        sch = gs.GpuScheduler(probe=_FakeProbe(_healthy(free_mb=3000, util=90)))
        self.assertTrue(sch.should_defer(gs.JobPriority.BATCH))
        self.assertFalse(sch.should_defer(gs.JobPriority.INTERACTIVE))


class GpuSchedulerReserveTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sch = gs.GpuScheduler(
            probe=_FakeProbe(_healthy()),
            config=gs.SchedulerConfig(max_concurrent=2),
        )

    def test_acquire_y_release(self) -> None:
        self.assertTrue(self.sch.acquire("document"))
        self.assertTrue(self.sch.is_active("document"))
        self.sch.release("document")
        self.assertFalse(self.sch.is_active("document"))

    def test_acquire_es_idempotente(self) -> None:
        self.assertTrue(self.sch.acquire("document"))
        self.assertTrue(self.sch.acquire("document"))
        self.assertEqual(len(self.sch.active_jobs()), 1)

    def test_release_sin_reserva_no_falla(self) -> None:
        self.sch.release("inexistente")

    def test_busy_refleja_las_reservas(self) -> None:
        # Lo usa ``gpu_slot`` para no pedir a los demás motores que suelten su
        # VRAM mientras hay un trabajo en marcha (le rompería el modelo).
        self.assertFalse(self.sch.busy())
        self.sch.acquire("document")
        self.assertTrue(self.sch.busy())
        self.sch.release("document")
        self.assertFalse(self.sch.busy())

    def test_respeta_el_maximo(self) -> None:
        self.assertTrue(self.sch.acquire("a"))
        self.assertTrue(self.sch.acquire("b"))
        self.assertFalse(self.sch.acquire("c"))

    def test_active_jobs_ordenados_por_prioridad(self) -> None:
        self.sch.acquire("video", gs.JobPriority.BATCH)
        self.sch.acquire("ocr", gs.JobPriority.PERCEPTION)
        self.assertEqual(
            [job.name for job in self.sch.active_jobs()], ["ocr", "video"]
        )

    def test_reserve_con_timeout_cero_no_espera(self) -> None:
        self.sch.acquire("a", gs.JobPriority.BATCH)
        self.sch.acquire("b", gs.JobPriority.BATCH)
        self.assertFalse(self.sch.reserve("c", timeout=0.0))

    def test_job_contexto_libera_al_salir(self) -> None:
        with self.sch.job("document") as ok:
            self.assertTrue(ok)
            self.assertTrue(self.sch.is_active("document"))
        self.assertFalse(self.sch.is_active("document"))

    def test_job_contexto_libera_al_fallar(self) -> None:
        with self.assertRaises(RuntimeError):
            with self.sch.job("document") as ok:
                self.assertTrue(ok)
                raise RuntimeError("boom")
        self.assertFalse(self.sch.is_active("document"))

    def test_job_contexto_sin_hueco_no_reserva(self) -> None:
        self.sch.acquire("a")
        self.sch.acquire("b")
        with self.sch.job("c") as ok:
            self.assertFalse(ok)
        self.assertFalse(self.sch.is_active("c"))


class GpuSchedulerStatusTest(unittest.TestCase):
    def test_status_line_sin_datos(self) -> None:
        sch = gs.GpuScheduler(probe=_FakeProbe(gs.GpuSnapshot()))
        self.assertIn("sin datos", sch.status_line())

    def test_status_line_con_datos_y_trabajos(self) -> None:
        sch = gs.GpuScheduler(probe=_FakeProbe(_healthy(free_mb=9000, util=10)))
        sch.acquire("document")
        linea = sch.status_line()
        self.assertIn("9000 MB libres", linea)
        self.assertIn("1 trabajo", linea)

    def test_describe_lista_los_trabajos(self) -> None:
        sch = gs.GpuScheduler(probe=_FakeProbe(_healthy()))
        sch.acquire("video", gs.JobPriority.BATCH, vram_mb=8000)
        texto = sch.describe()
        self.assertIn("video", texto)
        self.assertIn("~8000 MB", texto)

    def test_describe_sin_trabajos(self) -> None:
        sch = gs.GpuScheduler(probe=_FakeProbe(_healthy()))
        self.assertIn("ninguno", sch.describe())


if __name__ == "__main__":
    unittest.main()
