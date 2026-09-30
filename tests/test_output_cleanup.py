"""Retención de las salidas generadas (``services.output_cleanup``).

La auditoría encontró 180 ficheros (110,9 MB) en ``output/`` sin ninguna política
de limpieza. Estos tests fijan el contrato: no se toca nada reciente ni nada que
no sea un producto de la app, y los límites (edad, número, tamaño) actúan del
fichero más antiguo al más nuevo.
"""

import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config_manager import DEFAULT_SETTINGS  # noqa: E402
from services import output_cleanup as oc  # noqa: E402


def _crear(carpeta: Path, nombre: str, *, horas: float, bytes_: int = 10) -> Path:
    """Crea ``nombre`` con la antigüedad pedida (en horas)."""
    ruta = carpeta / nombre
    ruta.write_bytes(b"x" * bytes_)
    marca = (datetime.now() - timedelta(hours=horas)).timestamp()
    os.utime(ruta, (marca, marca))
    return ruta


class PlanCleanupTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_no_borra_lo_recien_generado(self) -> None:
        _crear(self.dir, "nuevo.png", horas=1)
        plan = oc.plan_cleanup(self.dir.iterdir(), max_files=0, max_age_days=0)
        self.assertEqual(plan.count, 0)
        self.assertEqual(plan.ignored, 1)

    def test_borra_por_edad(self) -> None:
        viejo = _crear(self.dir, "viejo.png", horas=24 * 100)
        nuevo = _crear(self.dir, "nuevo.png", horas=24 * 2)
        plan = oc.plan_cleanup(self.dir.iterdir(), max_age_days=90)
        self.assertEqual(plan.to_delete, (viejo,))
        self.assertEqual(plan.kept, 1)

    def test_borra_por_numero_desde_el_mas_antiguo(self) -> None:
        a = _crear(self.dir, "a.png", horas=24 * 10)
        b = _crear(self.dir, "b.png", horas=24 * 8)
        c = _crear(self.dir, "c.png", horas=24 * 6)
        plan = oc.plan_cleanup(self.dir.iterdir(), max_files=1, max_age_days=1000)
        self.assertEqual(plan.to_delete, (a, b))
        self.assertEqual(plan.kept, 1)
        self.assertNotIn(c, plan.to_delete)

    def test_borra_por_tamano(self) -> None:
        a = _crear(self.dir, "a.mp4", horas=24 * 10, bytes_=600 * 1024)
        b = _crear(self.dir, "b.mp4", horas=24 * 8, bytes_=600 * 1024)
        c = _crear(self.dir, "c.mp4", horas=24 * 6, bytes_=600 * 1024)
        plan = oc.plan_cleanup(
            self.dir.iterdir(), max_mb=1, max_age_days=1000, max_files=100
        )
        # 1,8 MB con tope de 1 MB: se van los dos más antiguos (0,6 MB quedan).
        self.assertEqual(plan.to_delete, (a, b))
        self.assertEqual(plan.kept, 1)
        self.assertNotIn(c, plan.to_delete)

    def test_ignora_extensiones_ajenas(self) -> None:
        _crear(self.dir, "apuntes.txt", horas=24 * 100)
        _crear(self.dir, "foto.png", horas=24 * 100)
        plan = oc.plan_cleanup(self.dir.iterdir(), max_age_days=1)
        self.assertEqual(plan.count, 1)
        self.assertEqual(plan.ignored, 1)

    def test_valores_invalidos_caen_a_los_defaults(self) -> None:
        _crear(self.dir, "a.png", horas=24 * 30)
        plan = oc.plan_cleanup(
            self.dir.iterdir(), max_files="pocos", max_age_days=None, max_mb="muchos"
        )
        # Con los defaults (500 ficheros, 90 días, 8 GB) 30 días no se borra.
        self.assertEqual(plan.count, 0)
        self.assertEqual(plan.kept, 1)

    def test_los_bytes_liberados_se_cuentan(self) -> None:
        _crear(self.dir, "a.png", horas=24 * 100, bytes_=2048)
        plan = oc.plan_cleanup(self.dir.iterdir(), max_age_days=90)
        self.assertEqual(plan.bytes_to_free, 2048)


class CleanupOutputTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_borra_el_antiguo_y_respeta_el_resto(self) -> None:
        viejo = _crear(self.dir, "viejo.png", horas=24 * 100)
        nuevo = _crear(self.dir, "nuevo.png", horas=24 * 2)
        ajeno = _crear(self.dir, "apuntes.txt", horas=24 * 100)

        plan = oc.cleanup_output(self.dir, max_age_days=90)

        self.assertEqual(plan.count, 1)
        self.assertFalse(viejo.exists())
        self.assertTrue(nuevo.exists())
        self.assertTrue(ajeno.exists())

    def test_carpeta_inexistente_no_es_un_error(self) -> None:
        plan = oc.cleanup_output(self.dir / "no_existe")
        self.assertEqual(plan.count, 0)
        self.assertIn("nada que borrar", plan.describe())

    def test_describe_dice_lo_que_va_a_borrar(self) -> None:
        _crear(self.dir, "viejo.png", horas=24 * 100)
        plan = oc.cleanup_output(self.dir, max_age_days=90)
        self.assertIn("1 ficheros", plan.describe())


class DefaultsCoherentesTest(unittest.TestCase):
    def test_los_defaults_del_settings_y_del_servicio_coinciden(self) -> None:
        """Las dos copias (config y servicio) no pueden desincronizarse."""
        ajustes = DEFAULT_SETTINGS["output_limpieza"]
        self.assertTrue(ajustes["enabled"])
        self.assertEqual(ajustes["max_files"], oc.DEFAULT_MAX_FILES)
        self.assertEqual(ajustes["max_age_days"], oc.DEFAULT_MAX_AGE_DAYS)
        self.assertEqual(ajustes["max_mb"], oc.DEFAULT_MAX_MB)
        self.assertEqual(ajustes["keep_recent_hours"], oc.DEFAULT_KEEP_RECENT_HOURS)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
