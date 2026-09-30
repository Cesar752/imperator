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

from core import engine_manager as em  # noqa: E402
from core.config_manager import DEFAULT_ENGINES  # noqa: E402
from core.engine_manager import _OLLAMA_DEFAULT_MODEL  # noqa: E402
from core.paths import PROJECT_ROOT  # noqa: E402
from engines import ollama as ollama_mod  # noqa: E402
from engines import ollama_vision as vision_mod  # noqa: E402
from engines.ollama import OllamaEngine  # noqa: E402
from engines.ollama_api import DEFAULT_NUM_CTX, DEFAULT_VISION_NUM_CTX  # noqa: E402
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


class URLSinLocalhostTest(unittest.TestCase):
    """``localhost`` cuesta ~2 s por petición en Windows (fallback de IPv6).

    Medido en la auditoría: ``/api/version`` tardaba 2 041 ms con ``localhost`` y
    15 ms con ``127.0.0.1``; lo mismo en ComfyUI. Estos tests impiden que el
    ``localhost`` vuelva a colarse por las plantillas o los valores por defecto.
    """

    def test_los_defaults_de_la_configuracion_usan_la_ip(self) -> None:
        for entrada in DEFAULT_ENGINES:
            if entrada["id"] in ("ollama", "ollama_vision"):
                self.assertNotIn("localhost", entrada["ruta"])
                self.assertIn("127.0.0.1", entrada["ruta"])

    def test_los_motores_y_el_gestor_coinciden(self) -> None:
        self.assertEqual(ollama_mod._DEFAULT_BASE_URL, em._OLLAMA_DEFAULT_BASE_URL)
        self.assertEqual(ollama_mod._DEFAULT_MODEL, em._OLLAMA_DEFAULT_MODEL)
        self.assertEqual(
            vision_mod._DEFAULT_BASE_URL, em._OLLAMA_VISION_DEFAULT_BASE_URL
        )
        self.assertEqual(vision_mod._DEFAULT_MODEL, em._OLLAMA_VISION_DEFAULT_MODEL)
        for url in (
            ollama_mod._DEFAULT_BASE_URL,
            em._OLLAMA_DEFAULT_BASE_URL,
            vision_mod._DEFAULT_BASE_URL,
            em._OLLAMA_VISION_DEFAULT_BASE_URL,
        ):
            self.assertNotIn("localhost", url)

    def test_construir_el_motor_sin_argumentos_usa_los_mismos_defaults(self) -> None:
        # El motor decía ``qwen3.5:9b`` mientras el proyecto usaba ``qwen3.6:35b``.
        self.assertEqual(OllamaEngine().default_model, em._OLLAMA_DEFAULT_MODEL)
        self.assertEqual(
            OllamaVisionEngine()._default_model, em._OLLAMA_VISION_DEFAULT_MODEL
        )

    def test_la_ui_y_el_ejemplo_versionado_no_registran_localhost(self) -> None:
        from ui import engines_dialog as ed

        self.assertNotIn("localhost", ed._TYPE_URLS["ollama"])
        self.assertNotIn("localhost", ed._TYPE_URLS["ollama_vision"])
        ejemplo = json.loads(
            (PROJECT_ROOT / "config" / "engines.example.json").read_text(
                encoding="utf-8"
            )
        )
        for entrada in ejemplo:
            self.assertNotIn("localhost", entrada["ruta"])


class ParalelismoDePaginasTest(unittest.TestCase):
    """El procesado de páginas por lotes se descartó **por medición**."""

    def test_las_paginas_van_de_una_en_una(self) -> None:
        from ui.document_widget import MAX_PARALLEL_PAGES

        # Medido con el worker real y 2 páginas: 1,02x y 0,62x (nunca mejor que
        # en serie), y con lotes el progreso sólo avisaba del final del lote.
        self.assertEqual(MAX_PARALLEL_PAGES, 1)

    def test_el_contexto_de_vision_no_esta_doblado(self) -> None:
        # El doble (32768) sólo servía para repartir 16384 entre dos slots.
        self.assertEqual(DEFAULT_VISION_NUM_CTX, 16_384)


class OllamaContextTest(unittest.TestCase):
    """Cada motor pide su contexto, y la visión pide más que el chat."""

    def test_cada_motor_usa_su_contexto(self) -> None:
        self.assertEqual(OllamaEngine._NUM_CTX, DEFAULT_NUM_CTX)
        self.assertEqual(OllamaVisionEngine._NUM_CTX, DEFAULT_VISION_NUM_CTX)

    def test_la_vision_pide_mas_contexto_que_el_chat(self) -> None:
        # No es por el prompt, sino porque los modelos de visión razonan: el
        # "thinking" se come la ventana antes de escribir la respuesta (medido:
        # 4143 tokens generados para 390 caracteres de respuesta).
        self.assertGreater(DEFAULT_VISION_NUM_CTX, DEFAULT_NUM_CTX)

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

    def test_la_vision_admite_un_timeout_configurado(self) -> None:
        from core.engine_manager import EngineManager

        engine = EngineManager._build(
            "ollama_vision", {"parametros": {"timeout": 1800}}
        )
        self.assertEqual(engine._timeout, 1800)
        self.assertEqual(
            EngineManager._build("ollama_vision", {"parametros": {}})._timeout,
            OllamaVisionEngine._ANALYZE_TIMEOUT,
        )


if __name__ == "__main__":
    unittest.main()
