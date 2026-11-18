"""Preferencias: una lectura de ``settings.json``, no una por valor.

``PrefsStore`` leía el fichero entero (parseo + validación de esquema) por cada
preferencia: construir una tarea costaba 11 lecturas. Ahora el mapa
``last_models`` se cachea hasta que cambie el ``mtime`` del fichero.
"""

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config_manager import ConfigManager  # noqa: E402
from services import preferences as prefs  # noqa: E402


class PrefsCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.cm = ConfigManager(config_dir=str(self.dir))
        self.cm.save_last_model("assembler_a", "1")
        self.cm.save_last_model("assembler_b", "2")
        # ConfigManager nuevo: caché vacía, como al arrancar la app.
        self.cm = ConfigManager(config_dir=str(self.dir))
        self.store = prefs.PrefsStore(self.cm)
        self._lecturas = 0
        original = ConfigManager._load_json

        def contando(cm_self, *args, **kwargs):
            self._lecturas += 1
            return original(cm_self, *args, **kwargs)

        self._patch = mock.patch.object(ConfigManager, "_load_json", contando)
        self._patch.start()

    def tearDown(self) -> None:
        self._patch.stop()
        self._tmp.cleanup()

    def test_construir_las_prefs_cuesta_una_lectura(self) -> None:
        prefs.AssemblerPrefs.from_store(self.store)
        self.assertEqual(self._lecturas, 1, "no una por cada valor")

    def test_un_set_no_relee_para_leer(self) -> None:
        self.assertEqual(self.store.get_int("assembler_a", 0), 1)
        self._lecturas = 0

        with mock.patch.object(ConfigManager, "_save_json") as guardado:
            self.store.set("assembler_c", "3")

        self.assertEqual(guardado.call_count, 1, "una sola escritura")
        self.assertLessEqual(self._lecturas, 1, "solo el merge bajo lock")
        self._lecturas = 0
        self.assertEqual(self.store.get_int("assembler_c", 0), 3)
        self.assertEqual(self._lecturas, 0, "la lectura posterior sale de la caché")

    def test_una_edicion_externa_invalida_la_cache(self) -> None:
        self.assertEqual(self.store.get_int("assembler_a", 0), 1)

        ruta = self.dir / "settings.json"
        datos = json.loads(ruta.read_text(encoding="utf-8"))
        datos["last_models"]["assembler_a"] = "9"
        time.sleep(0.01)  # que cambie el mtime
        ruta.write_text(json.dumps(datos, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        self.assertEqual(self.store.get_int("assembler_a", 0), 9, "se relee si cambia el mtime")

    def test_el_mapa_devuelto_es_una_copia(self) -> None:
        mapa = self.store.as_map()
        mapa["assembler_a"] = "999"
        self.assertEqual(self.store.get_int("assembler_a", 0), 1)


if __name__ == "__main__":
    unittest.main()
