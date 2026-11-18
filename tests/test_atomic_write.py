"""A9: escritura atómica. Un fallo a mitad no deja el fichero truncado."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.atomic_io import atomic_write_json, atomic_write_text  # noqa: E402


class AtomicWriteTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _temporales(self) -> list[Path]:
        return list(self.dir.glob("*.tmp"))

    def test_json_escribe_y_termina_en_salto(self) -> None:
        destino = self.dir / "informe.json"

        atomic_write_json(destino, {"a": 1, "b": "á"})

        texto = destino.read_text(encoding="utf-8")
        self.assertTrue(texto.endswith("\n"))
        self.assertEqual(json.loads(texto), {"a": 1, "b": "á"})
        self.assertEqual(self._temporales(), [])

    def test_crea_las_carpetas_que_falten(self) -> None:
        destino = self.dir / "sub" / "carpeta" / "guion.json"

        atomic_write_json(destino, {"ok": True})

        self.assertTrue(destino.is_file())

    def test_sobrescribe_lo_anterior(self) -> None:
        destino = self.dir / "datos.json"
        destino.write_text('{"viejo": true}', encoding="utf-8")

        atomic_write_json(destino, {"nuevo": True})

        self.assertEqual(json.loads(destino.read_text(encoding="utf-8")), {"nuevo": True})

    def test_si_falla_el_replace_el_original_queda_intacto(self) -> None:
        destino = self.dir / "informe.json"
        original = '{"version": "buena"}'
        destino.write_text(original, encoding="utf-8")

        with mock.patch.object(os, "replace", side_effect=OSError("disco lleno")):
            with self.assertRaises(OSError):
                atomic_write_json(destino, {"version": "nueva"})

        self.assertEqual(destino.read_text(encoding="utf-8"), original)
        self.assertEqual(self._temporales(), [], "el temporal se borra al fallar")

    def test_si_falla_no_crea_el_destino_si_no_existia(self) -> None:
        destino = self.dir / "nuevo.json"

        with mock.patch.object(os, "replace", side_effect=OSError("disco lleno")):
            with self.assertRaises(OSError):
                atomic_write_json(destino, {"a": 1})

        self.assertFalse(destino.exists(), "nunca un fichero a medias")
        self.assertEqual(self._temporales(), [])

    def test_texto_tambien_es_atomico(self) -> None:
        destino = self.dir / "guion.json"
        destino.write_text("viejo", encoding="utf-8")

        with mock.patch.object(os, "replace", side_effect=OSError("disco lleno")):
            with self.assertRaises(OSError):
                atomic_write_text(destino, "nuevo")

        self.assertEqual(destino.read_text(encoding="utf-8"), "viejo")


if __name__ == "__main__":
    unittest.main()
