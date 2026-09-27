"""Valores por defecto de Ollama que deben ser coherentes entre sitios.

El modelo del chat vive en cuatro sitios (``_OLLAMA_DEFAULT_MODEL`` al construir
el motor, ``DEFAULT_ENGINES`` que se escribe en ``config/engines.json``, la
plantilla del diálogo de motores y ``config/engines.example.json``); la ventana
de contexto la comparten el chat y la visión
(``ollama_api.DEFAULT_NUM_CTX``). Estos tests evitan que se desincronicen: si uno
cambia y otro no, el chat avisa de que el modelo no está instalado o el documento
deja de caber en el contexto.
"""

import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from core.config_manager import DEFAULT_ENGINES  # noqa: E402
from core.engine_manager import _OLLAMA_DEFAULT_MODEL  # noqa: E402
from core.paths import PROJECT_ROOT  # noqa: E402
from engines import ollama as ollama_mod  # noqa: E402
from engines.ollama import OllamaEngine  # noqa: E402
from engines.ollama_api import DEFAULT_NUM_CTX  # noqa: E402
from engines.ollama_vision import OllamaVisionEngine  # noqa: E402


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


class OllamaContextTest(unittest.TestCase):
    """La ventana de contexto es la misma en el chat y en la visión."""

    def test_los_dos_motores_comparten_el_contexto(self) -> None:
        self.assertEqual(OllamaEngine._NUM_CTX, DEFAULT_NUM_CTX)
        self.assertEqual(OllamaVisionEngine._NUM_CTX, DEFAULT_NUM_CTX)

    def test_la_configuracion_puede_fijar_el_contexto(self) -> None:
        # Sin este camino configurable, subir el contexto exigía tocar el código.
        from core.engine_manager import EngineManager

        engine = EngineManager._build(
            "ollama", {"parametros": {"model": "m", "num_ctx": 32768}}
        )
        self.assertIsInstance(engine, OllamaEngine)
        resp = mock.Mock()
        resp.json.return_value = {"message": {"content": "ok"}}
        with mock.patch.object(ollama_mod.requests, "post", return_value=resp) as post:
            engine.chat([{"role": "user", "content": "hola"}])
        self.assertEqual(
            post.call_args.kwargs["json"]["options"]["num_ctx"], 32768
        )

    def test_la_vision_tambien_admite_contexto_configurado(self) -> None:
        from core.engine_manager import EngineManager

        engine = EngineManager._build(
            "ollama_vision", {"parametros": {"model": "v", "num_ctx": 16384}}
        )
        self.assertIsInstance(engine, OllamaVisionEngine)
        self.assertEqual(engine._num_ctx, 16384)


if __name__ == "__main__":
    unittest.main()
