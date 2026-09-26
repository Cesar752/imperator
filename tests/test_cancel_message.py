"""Guardas de la constante compartida de cancelación.

``CANCEL_MESSAGE`` vive en ``engines.base_engine`` porque tanto los motores
(que lo lanzan) como los widgets (que lo comparan para no mostrarlo como error)
deben compartir el mismo texto sin invertir las dependencias. Este test evita
que el literal vuelva a dispersarse por el código.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engines.base_engine import CANCEL_MESSAGE  # noqa: E402
from ui import workers  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent

# Capas donde el literal no debe aparecer (solo puede estar en su definición).
_SCANNED_DIRS = ("engines", "ui", "core", "services", "tasks")
_DEFINITION = "engines/base_engine.py"


class CancelMessageTest(unittest.TestCase):
    def test_value(self) -> None:
        self.assertEqual(CANCEL_MESSAGE, "Cancelado por el usuario")

    def test_workers_reexporta_la_misma_constante(self) -> None:
        self.assertIs(workers.CANCEL_MESSAGE, CANCEL_MESSAGE)

    def test_literal_solo_en_su_definicion(self) -> None:
        ofensores: list[str] = []
        for folder in _SCANNED_DIRS:
            for path in sorted((_ROOT / folder).rglob("*.py")):
                relative = path.relative_to(_ROOT).as_posix()
                if relative == _DEFINITION:
                    continue
                if "Cancelado por el usuario" in path.read_text(encoding="utf-8"):
                    ofensores.append(relative)
        self.assertEqual(
            ofensores, [], f"estos ficheros deben usar CANCEL_MESSAGE: {ofensores}"
        )


if __name__ == "__main__":
    unittest.main()
