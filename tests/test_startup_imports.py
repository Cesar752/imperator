"""El arranque no debe pagar los imports de los motores pesados.

La auditoría del 30/09/2026 midió el import de ``ui.main_window`` en **9,85 s**:
9,53 s eran ``kokoro`` (que arrastra ``torch`` + ``transformers``) y 8,72 s
``perception`` (``torch`` + ``transformers``), importados al cargar los módulos
aunque sus motores ni se fuesen a usar. Ahora se importan bajo demanda.

El test se ejecuta en **un proceso limpio** a propósito: en el proceso de los
tests cualquier módulo puede haberse cargado ya y la comprobación no valdría.
"""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from core.paths import PROJECT_ROOT  # noqa: E402

#: Módulos que no deben cargarse al arrancar la app (torch/transformers son los
#: caros; los demás —ctranslate2, onnxruntime, PyMuPDF— son baratos y sí se usan).
PESADOS_PROHIBIDOS = ("torch", "transformers", "kokoro")
VIGILADOS = PESADOS_PROHIBIDOS + ("ctranslate2", "onnxruntime", "pymupdf")


class ArranqueLigeroTest(unittest.TestCase):
    def _importar_en_proceso_limpio(self) -> dict:
        codigo = (
            "import json, sys, time;"
            f"sys.path.insert(0, r'{PROJECT_ROOT}');"
            "t0 = time.perf_counter();"
            "import ui.main_window;"
            "print(json.dumps({'s': round(time.perf_counter() - t0, 2),"
            "'pesadas': sorted(m for m in "
            f"{list(VIGILADOS)!r}"
            " if m in sys.modules)}))"
        )
        proceso = subprocess.run(
            [sys.executable, "-c", codigo],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(proceso.returncode, 0, proceso.stderr[-800:])
        return json.loads(proceso.stdout.strip().splitlines()[-1])

    def test_importar_la_app_no_carga_torch_ni_transformers(self) -> None:
        datos = self._importar_en_proceso_limpio()
        for modulo in PESADOS_PROHIBIDOS:
            self.assertNotIn(
                modulo,
                datos["pesadas"],
                f"'{modulo}' se carga al importar ui.main_window: el arranque vuelve "
                "a pagar los ~9,5 s que quitó el import perezoso.",
            )

    def test_el_import_de_la_app_es_rapido(self) -> None:
        datos = self._importar_en_proceso_limpio()
        # Antes eran 9,85 s; con los motores pesados perezosos baja a ~1-2 s.
        self.assertLess(datos["s"], 5.0, f"import de ui.main_window: {datos['s']} s")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
