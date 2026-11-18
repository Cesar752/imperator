"""A8: el mapa id→builder construye todos los motores y el desconocido falla.

Es un test de estructura: si un motor nuevo entra en ``ENGINE_FACTORY`` sin
builder, el mapa y el factory se separan y la construcción cambia de camino sin
que nadie lo note.
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
from core.config_manager import ConfigManager  # noqa: E402
from engines.base_engine import EngineError  # noqa: E402


#: Motores que exigen algo más que los defaults (ya lo exigían antes de A8).
_EXTRAS: dict[str, dict] = {
    "comfyui": {"workflows": {"t2v": "workflow.json"}},
    "wanvideo": {"workflows": {"t2v": "workflow.json"}},
}


def _config(engine_id: str) -> dict:
    """Entrada mínima válida de engines.json (sin ``ruta``: se usan defaults)."""
    return {
        "id": engine_id,
        "nombre": engine_id,
        "enabled": True,
        "parametros": dict(_EXTRAS.get(engine_id, {})),
    }


class BuildersTest(unittest.TestCase):
    def test_hay_un_builder_para_cada_motor_del_factory(self) -> None:
        self.assertEqual(
            set(em._BUILDERS),
            set(em.ENGINE_FACTORY),
            "un motor sin builder caería en el genérico sin avisar",
        )

    def test_todos_los_motores_se_construyen(self) -> None:
        for engine_id in sorted(em.ENGINE_FACTORY):
            with self.subTest(engine_id=engine_id):
                motor = em.EngineManager._build(engine_id, _config(engine_id))
                self.assertIsInstance(motor, em.ENGINE_FACTORY[engine_id])

    def test_un_id_desconocido_da_engine_error(self) -> None:
        with self.assertRaises(EngineError) as ctx:
            em.EngineManager._build("no_existe", {"parametros": {}})
        self.assertIn("no_existe", str(ctx.exception))

    def test_el_generico_sigue_dando_config_error(self) -> None:
        class MotorFuturo:
            def __init__(self, obligatorio: int = 0) -> None:
                self.obligatorio = obligatorio

        with mock.patch.dict(em.ENGINE_FACTORY, {"futuro": MotorFuturo}):
            with self.assertRaises(em.ConfigError):
                em.EngineManager._build("futuro", {"parametros": {"otra_cosa": 1}})

    def test_reload_sigue_funcionando(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gestor = em.EngineManager(ConfigManager(config_dir=tmp))

            self.assertIn("ollama", gestor.supported_types())
            self.assertIsNotNone(gestor.get_engine("ollama"))

            gestor.reload()

            self.assertIsNotNone(gestor.get_engine("ollama"))


if __name__ == "__main__":
    unittest.main()
