"""Capas: ``import core.task_manager`` no debe arrastrar PyQt6 ni las tareas.

``core`` importaba las 12 tareas en la cabecera y cada tarea importa su widget
de ``ui``: importar el núcleo cargaba la GUI entera. Ahora el registro guarda
factories que importan a demanda.
"""

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import task_manager as tm  # noqa: E402

#: Las 11 tareas registradas (chat + 10 con solo engine_manager).
TAREAS = {
    "chat",
    "document",
    "image",
    "photo",
    "video",
    "video_enhance",
    "understanding",
    "notes",
    "chronicle",
    "audio",
    "assembler",
}


class CapasTest(unittest.TestCase):
    def _importar_en_proceso_limpio(self) -> dict:
        codigo = (
            "import json, sys;"
            f"sys.path.insert(0, r'{ROOT}');"
            "import core.task_manager as tm;"
            "print(json.dumps({"
            "'pyqt6': sorted(m for m in sys.modules if m.startswith('PyQt6')),"
            "'tasks': sorted(m for m in sys.modules if m.startswith('tasks')),"
            "'tareas': sorted(tm.TASK_FACTORY)}))"
        )
        proceso = subprocess.run(
            [sys.executable, "-c", codigo],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(proceso.returncode, 0, proceso.stderr[-800:])
        return json.loads(proceso.stdout.strip().splitlines()[-1])

    def test_importar_el_nucleo_no_carga_pyqt6(self) -> None:
        datos = self._importar_en_proceso_limpio()
        self.assertEqual(
            datos["pyqt6"], [], "core no puede depender de la GUI para importarse"
        )

    def test_importar_el_nucleo_no_importa_las_tareas(self) -> None:
        datos = self._importar_en_proceso_limpio()
        self.assertEqual(datos["tasks"], [], "el registro debe importar a demanda")

    def test_el_registro_sigue_completo(self) -> None:
        self.assertEqual(set(tm.TASK_FACTORY), TAREAS)


if __name__ == "__main__":
    unittest.main()
