"""Tests de la higiene de memoria entre escenas (``core.memory``).

Solo se comprueba el contrato puro: ``release_memory`` recolecta ciclos, invoca
el recorte inyectado una vez y **nunca lanza**. El recorte real del *working set*
de Windows no se ejerce en la suite salvo en una comprobación de tipo, para no
tocar la memoria del proceso de test.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import memory  # noqa: E402


class ReleaseMemoryTest(unittest.TestCase):
    def test_llama_al_trim_inyectado_una_vez(self) -> None:
        llamadas: list = []

        def trim() -> bool:
            llamadas.append(True)
            return True

        collected = memory.release_memory(trim=trim)
        self.assertIsInstance(collected, int)
        self.assertEqual(llamadas, [True])

    def test_un_trim_que_falla_no_rompe(self) -> None:
        def boom() -> bool:
            raise RuntimeError("no se puede tocar el working set")

        # No debe propagar la excepción: liberar es best-effort.
        self.assertIsInstance(memory.release_memory(trim=boom), int)

    def test_trim_working_set_devuelve_bool(self) -> None:
        self.assertIsInstance(memory.trim_working_set(), bool)


if __name__ == "__main__":
    unittest.main()
