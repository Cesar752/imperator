"""Guardas del gestor de motores (hallazgos de la auditoría).

Cubre tres cosas que fallaban en silencio:

- añadir un motor con un ``id`` ya existente (antes el error genérico del esquema
  no decía qué motor era);
- los ``parametros`` de un motor del camino genérico que no encajan con su
  constructor (antes salía un ``TypeError`` crudo como «error inesperado»);
- una ``ruta`` con ``localhost``, que en Windows añade ~2 s a cada petición.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from core import engine_manager as em  # noqa: E402
from core.config_manager import ConfigError, ConfigManager  # noqa: E402
from core.engine_manager import EngineManager  # noqa: E402


class EngineManagerGuardasTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.cm = ConfigManager(self._tmp.name)
        self.gestor = EngineManager(self.cm)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_no_admite_un_id_repetido(self) -> None:
        existente = dict(self.gestor.get_config("ollama") or {})
        self.assertTrue(existente)
        with self.assertRaises(ConfigError) as ctx:
            self.gestor.add_engine(existente)
        self.assertIn("ollama", str(ctx.exception))
        ids = [config["id"] for config in self.gestor.list_configs()]
        self.assertEqual(ids.count("ollama"), 1)

    def test_anadir_un_motor_nuevo_sigue_funcionando(self) -> None:
        # Se parte de una configuración con un solo motor.
        self.cm.save_engines(
            [
                {
                    "id": "ollama",
                    "nombre": "Ollama",
                    "ruta": "http://127.0.0.1:11434",
                    "enabled": True,
                    "parametros": {"model": "m"},
                }
            ]
        )
        self.gestor.reload()
        self.assertIsNone(self.gestor.get_config("musicgen"))

        self.gestor.add_engine(
            {
                "id": "musicgen",
                "nombre": "MusicGen (música local)",
                "ruta": "local",
                "enabled": True,
                "parametros": {"model": "small", "duration": 8},
            }
        )
        self.assertIsNotNone(self.gestor.get_config("musicgen"))
        self.assertIn("musicgen", [c["id"] for c in self.gestor.list_configs()])

    def test_parametros_que_no_encajan_dan_config_error(self) -> None:
        class MotorFuturo:
            def __init__(self, obligatorio: int = 0) -> None:
                self.obligatorio = obligatorio

        with mock.patch.dict(em.ENGINE_FACTORY, {"futuro": MotorFuturo}):
            with self.assertRaises(ConfigError) as ctx:
                EngineManager._build("futuro", {"parametros": {"otra_cosa": 1}})
        self.assertIn("futuro", str(ctx.exception))
        self.assertIn("parametros", str(ctx.exception))

    def test_avisa_si_el_motor_usa_localhost(self) -> None:
        with self.assertLogs("core.engine_manager", level="WARNING") as registro:
            EngineManager._build(
                "ollama",
                {"ruta": "http://localhost:11434", "parametros": {"model": "m"}},
            )
        self.assertIn("localhost", "\n".join(registro.output))

    def test_no_avisa_con_la_ip(self) -> None:
        with self.assertNoLogs("core.engine_manager", level="WARNING"):
            EngineManager._build(
                "ollama",
                {"ruta": "http://127.0.0.1:11434", "parametros": {"model": "m"}},
            )

    def test_los_defaults_no_necesitan_url_en_la_configuracion(self) -> None:
        motor = EngineManager._build("ollama", {"parametros": {}})
        self.assertEqual(motor._base_url, em._OLLAMA_DEFAULT_BASE_URL)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
