"""El modelo por defecto del chat debe ser el mismo en todas sus copias.

El valor vive en cuatro sitios (``_OLLAMA_DEFAULT_MODEL`` al construir el motor,
``DEFAULT_ENGINES`` que se escribe en ``config/engines.json``, la plantilla del
diálogo de motores y ``config/engines.example.json``). Este test evita que se
desincronicen: si uno cambia y otro no, el chat acabaría avisando de que el
modelo por defecto no está instalado.
"""

import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from core.config_manager import DEFAULT_ENGINES  # noqa: E402
from core.engine_manager import _OLLAMA_DEFAULT_MODEL  # noqa: E402
from core.paths import PROJECT_ROOT  # noqa: E402


class OllamaDefaultModelTest(unittest.TestCase):
    def test_la_configuracion_y_la_ui_usan_el_mismo_default(self) -> None:
        from ui import engines_dialog as ed

        configurado = next(
            entrada for entrada in DEFAULT_ENGINES if entrada["id"] == "ollama"
        )["parametros"]["model"]
        self.assertEqual(configurado, _OLLAMA_DEFAULT_MODEL)
        self.assertEqual(
            ed._TYPE_TEMPLATES["ollama"]["model"], _OLLAMA_DEFAULT_MODEL
        )

    def test_el_ejemplo_versionado_usa_el_mismo_default(self) -> None:
        ejemplo = json.loads(
            (PROJECT_ROOT / "config" / "engines.example.json").read_text(
                encoding="utf-8"
            )
        )
        modelo = next(
            entrada for entrada in ejemplo if entrada["id"] == "ollama"
        )["parametros"]["model"]
        self.assertEqual(modelo, _OLLAMA_DEFAULT_MODEL)


if __name__ == "__main__":
    unittest.main()
