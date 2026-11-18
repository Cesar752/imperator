"""Poda de ``cache/frames``: comportamiento y cableado en el arranque.

El hallazgo medio de la auditoría decía que ``FrameCache.prune`` no se llamaba
nunca. Al comprobarlo, sí se llama: ``MainWindow.__init__`` → ``_cleanup_output``
→ ``run_pool_task(FrameCache().prune, ...)``. Lo que faltaba era el test que lo
demuestra, que es lo que deja que una afirmación así sobreviva.
"""

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from core.frame_cache import FrameCache  # noqa: E402
from ui import main_window as mw  # noqa: E402




def _png(carpeta: Path, nombre: str, edad_s: float) -> Path:
    """Crea un PNG falso con ``mtime`` de hace ``edad_s`` segundos."""
    ruta = carpeta / nombre
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_bytes(b"\x89PNG falso")
    momento = time.time() - edad_s
    os.utime(ruta, (momento, momento))
    return ruta


class PruneTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.raiz = Path(self._tmp.name) / "frames"
        self.raiz.mkdir()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_borra_los_mas_antiguos(self) -> None:
        viejos = [_png(self.raiz, f"v{i}.png", 1000 + i) for i in range(3)]
        nuevos = [_png(self.raiz, f"n{i}.png", 10 + i) for i in range(2)]

        borrados = FrameCache(self.raiz).prune(max_files=2)

        self.assertEqual(borrados, 3)
        self.assertFalse(any(p.exists() for p in viejos))
        self.assertTrue(all(p.exists() for p in nuevos))

    def test_por_debajo_del_tope_no_borra_nada(self) -> None:
        _png(self.raiz, "a.png", 100)
        self.assertEqual(FrameCache(self.raiz).prune(max_files=10), 0)

    def test_sin_carpeta_no_falla(self) -> None:
        self.assertEqual(FrameCache(self.raiz / "no_existe").prune(), 0)

    def test_el_arranque_lanza_la_poda(self) -> None:
        """``_cleanup_output`` (que corre en ``MainWindow.__init__``) lanza prune."""
        ventana = mw.MainWindow.__new__(mw.MainWindow)
        with mock.patch.object(
            mw.MainWindow, "_load_settings_merged", return_value={}
        ), mock.patch.object(mw, "run_pool_task") as pool:
            ventana._cleanup_output()

        lanzadas = [llamada.args[0] for llamada in pool.call_args_list if llamada.args]
        self.assertTrue(
            any(
                isinstance(getattr(fn, "__self__", None), FrameCache)
                for fn in lanzadas
            ),
            "el arranque debe lanzar FrameCache().prune",
        )

    def test_el_init_de_la_ventana_llama_a_la_limpieza(self) -> None:
        """El otro eslabón: ``MainWindow.__init__`` llama a ``_cleanup_output``.

        Se comprueba sobre el código fuente en vez de construir la ventana real:
        construirla lanza tareas de pool asíncronas y, al destruirla en el test,
        alguna puede emitir contra un ``QObject`` ya muerto (aborta el proceso).
        """
        import inspect

        fuente = inspect.getsource(mw.MainWindow.__init__)
        self.assertIn("_cleanup_output()", fuente)


if __name__ == "__main__":
    unittest.main()
