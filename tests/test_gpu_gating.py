"""Reserva de GPU de las tareas pesadas (``ui.workers.gpu_slot``).

La auditoría midió que dos motores no caben a la vez en la GPU (Flux deja hasta
15 438 MB de 16 303 y el chat necesita 13 116), así que las tareas pesadas piden
permiso al planificador antes de empezar. Estos tests cubren el contrato del
contexto sin GPU real: no reservar, reservar y liberar, abortar con motivo y
liberar la VRAM de los demás motores antes de rendirse.
"""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from core.gpu_scheduler import JobPriority  # noqa: E402
from engines.base_engine import EngineError  # noqa: E402
from ui.image_widget import ImageWorker  # noqa: E402
from ui.workers import gpu_slot  # noqa: E402


class SchedulerFalso:
    """Doble del planificador: responde lo que diga la lista y apunta las llamadas."""

    def __init__(self, respuestas=((True, "hay hueco"),)) -> None:
        self.respuestas = list(respuestas)
        self.preguntas: list[JobPriority] = []
        self.adquiridos: list[str] = []
        self.liberados: list[str] = []

    def can_start(self, priority, vram_mb=0):
        self.preguntas.append(priority)
        if len(self.respuestas) > 1:
            return self.respuestas.pop(0)
        return self.respuestas[0]

    def acquire(self, name, priority=None, vram_mb=0) -> bool:
        self.adquiridos.append(name)
        return True

    def release(self, name) -> None:
        self.liberados.append(name)

    def is_active(self, name) -> bool:
        return name in self.adquiridos and name not in self.liberados


class GpuSlotTest(unittest.TestCase):
    def test_sin_planificador_no_reserva_nada(self) -> None:
        with gpu_slot(None, "image"):
            paso = True
        self.assertTrue(paso)

    def test_reserva_y_libera_al_salir(self) -> None:
        scheduler = SchedulerFalso()
        with gpu_slot(scheduler, "image"):
            self.assertEqual(scheduler.adquiridos, ["image"])
        self.assertEqual(scheduler.liberados, ["image"])

    def test_libera_aunque_el_cuerpo_falle(self) -> None:
        scheduler = SchedulerFalso()
        with self.assertRaises(RuntimeError):
            with gpu_slot(scheduler, "video"):
                raise RuntimeError("se rompió")
        self.assertEqual(scheduler.liberados, ["video"])

    def test_si_no_cabe_lo_dice_con_el_motivo(self) -> None:
        scheduler = SchedulerFalso(
            respuestas=((False, "quedan 645 MB libres y se necesitan 1024 MB"),)
        )
        with self.assertRaises(EngineError) as ctx:
            with gpu_slot(scheduler, "chat"):
                self.fail("no debería entrar en el bloque")
        self.assertIn("645 MB", str(ctx.exception))
        self.assertEqual(scheduler.adquiridos, [])

    def test_primero_libera_los_otros_motores_y_reintenta(self) -> None:
        scheduler = SchedulerFalso(
            respuestas=[(False, "sin hueco"), (True, "hay hueco")]
        )
        liberados: list[bool] = []

        with gpu_slot(scheduler, "image", liberar_otros=lambda: liberados.append(True)):
            self.assertEqual(scheduler.adquiridos, ["image"])

        self.assertEqual(liberados, [True])
        self.assertEqual(len(scheduler.preguntas), 2)

    def test_solo_libera_una_vez(self) -> None:
        """Si liberar no arregla el problema, no se insiste en bucle."""
        scheduler = SchedulerFalso(respuestas=((False, "sigue lleno"),))
        liberados: list[bool] = []
        with self.assertRaises(EngineError):
            with gpu_slot(scheduler, "image", liberar_otros=lambda: liberados.append(True)):
                self.fail("no debería entrar en el bloque")
        self.assertEqual(liberados, [True])

    def test_usa_la_prioridad_de_cada_tarea(self) -> None:
        lote = SchedulerFalso()
        with gpu_slot(lote, "video"):
            pass
        interactivo = SchedulerFalso()
        with gpu_slot(interactivo, "chat"):
            pass
        self.assertEqual(lote.preguntas[0], JobPriority.BATCH)
        self.assertEqual(interactivo.preguntas[0], JobPriority.INTERACTIVE)

    def test_una_tarea_desconocida_es_interactiva(self) -> None:
        scheduler = SchedulerFalso()
        with gpu_slot(scheduler, "noexiste"):
            pass
        self.assertEqual(scheduler.preguntas[0], JobPriority.INTERACTIVE)


class _MotorFalso:
    def __init__(self) -> None:
        self.llamadas = 0

    def generate_image(self, **_kwargs) -> str:
        self.llamadas += 1
        return "output/plataformaia_x.png"


class ImageWorkerTest(unittest.TestCase):
    """El worker real pasa por el mismo camino que la app."""

    def test_bloqueado_no_llama_al_motor(self) -> None:
        motor = _MotorFalso()
        scheduler = SchedulerFalso(respuestas=((False, "GPU ocupada"),))
        worker = ImageWorker(motor, {"prompt": "x"}, scheduler=scheduler)
        with self.assertRaises(EngineError):
            worker.work()
        self.assertEqual(motor.llamadas, 0)

    def test_con_hueco_genera_y_devuelve_la_ruta(self) -> None:
        motor = _MotorFalso()
        scheduler = SchedulerFalso()
        worker = ImageWorker(motor, {"prompt": "x"}, scheduler=scheduler)
        self.assertTrue(worker.work().endswith(".png"))
        self.assertEqual(motor.llamadas, 1)
        self.assertEqual(scheduler.adquiridos, ["image"])
        self.assertEqual(scheduler.liberados, ["image"])

    def test_sin_scheduler_funciona_como_antes(self) -> None:
        motor = _MotorFalso()
        worker = ImageWorker(motor, {"prompt": "x"})
        self.assertTrue(worker.work().endswith(".png"))
        self.assertEqual(motor.llamadas, 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
