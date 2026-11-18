"""Rutas centralizadas: no dependen del CWD (lo que rompe un .exe).

Las rutas se construían a mano con ``Path(__file__).parent.parent / "output"`` en
varios widgets, que funcionaba desde el código fuente pero no desde un ejecutable
empaquetado ni si el CWD cambia.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import paths  # noqa: E402

#: Carpetas de datos de usuario que deben colgar de la raíz del proyecto.
CARPETAS = {
    "OUTPUT": "output",
    "MUSIC": "music",
    "CACHE": "cache",
    "ASSETS": "assets",
}


class PathsTest(unittest.TestCase):
    def test_son_absolutas_y_cuelgan_de_la_raiz(self) -> None:
        self.assertEqual(paths.PROJECT_ROOT, ROOT)
        for nombre, carpeta in CARPETAS.items():
            ruta = getattr(paths, nombre)
            with self.subTest(ruta=nombre):
                self.assertTrue(ruta.is_absolute(), f"{nombre} debe ser absoluta")
                self.assertEqual(ruta, paths.PROJECT_ROOT / carpeta)

    def test_no_dependen_del_cwd(self) -> None:
        antes = {n: getattr(paths, n) for n in CARPETAS}
        cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                # Se releen desde el módulo ya importado: no dependen del CWD.
                for nombre, ruta in antes.items():
                    self.assertEqual(getattr(paths, nombre), ruta)
            finally:
                os.chdir(cwd)

    def test_los_widgets_no_construyen_rutas_a_mano(self) -> None:
        for nombre in (
            "assembler_widget.py",
            "document_widget.py",
            "video_widget.py",
            "understanding_widget.py",
        ):
            fuente = (ROOT / "ui" / nombre).read_text(encoding="utf-8")
            with self.subTest(archivo=nombre):
                self.assertNotIn(
                    'Path(__file__).resolve().parent.parent / "output"', fuente
                )
                self.assertNotIn(
                    'Path(__file__).resolve().parent.parent / "music"', fuente
                )

    def test_el_nucleo_tampoco(self) -> None:
        for nombre in ("cli_runner.py", "video_assembler.py", "frame_cache.py"):
            fuente = (ROOT / "core" / nombre).read_text(encoding="utf-8")
            with self.subTest(archivo=nombre):
                self.assertNotIn('PROJECT_ROOT / "output"', fuente)
                self.assertNotIn('PROJECT_ROOT / "cache"', fuente)

    def test_la_cache_de_fotogramas_cuelga_de_la_raiz(self) -> None:
        from core.frame_cache import DEFAULT_CACHE_DIR

        self.assertEqual(DEFAULT_CACHE_DIR, paths.CACHE / "frames")


if __name__ == "__main__":
    unittest.main()
