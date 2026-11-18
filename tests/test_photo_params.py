"""Tests del retoque de fotos: ajustes/conversión (services) y worker (subprocess)."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image  # noqa: E402

from services import photo_params as pp  # noqa: E402


class AdjustmentsTest(unittest.TestCase):
    def test_identidad(self):
        self.assertTrue(pp.Adjustments().is_identity)
        self.assertFalse(pp.Adjustments(brightness=1.5).is_identity)

    def test_brillo_cambia_los_pixeles(self):
        img = Image.new("RGB", (4, 4), (100, 100, 100))
        oscura = pp.Adjustments(brightness=0.5).apply(img)
        clara = pp.Adjustments(brightness=2.0).apply(img)
        self.assertNotEqual(oscura.getpixel((0, 0)), img.getpixel((0, 0)))
        self.assertGreater(clara.getpixel((0, 0))[0], img.getpixel((0, 0))[0])

    def test_clamp(self):
        self.assertEqual(pp.clamp_adjust("x"), pp.ADJUST_DEFAULT)
        self.assertEqual(pp.clamp_adjust(99), pp.ADJUST_MAX)
        self.assertEqual(pp.clamp_adjust(-1), pp.ADJUST_MIN)


class ConvertTest(unittest.TestCase):
    def test_convierte_de_jpeg_a_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            origen = tmp / "foto.jpg"
            Image.new("RGB", (20, 10), (10, 20, 30)).save(origen, format="JPEG")
            destino = pp.convert(origen, tmp / "foto.png", "PNG")
            self.assertTrue(destino.is_file())
            self.assertEqual(Image.open(destino).format, "PNG")
            self.assertEqual(Image.open(destino).size, (20, 10))

    def test_rgba_a_jpeg_aplana(self):
        with tempfile.TemporaryDirectory() as tmp:
            img = Image.new("RGBA", (8, 8), (255, 0, 0, 128))
            destino = pp.save_image(img, Path(tmp) / "x.jpg", "JPEG", 90)
            self.assertEqual(Image.open(destino).mode, "RGB")

    def test_aplica_ajustes_al_convertir(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            origen = tmp / "gris.png"
            Image.new("RGB", (4, 4), (120, 120, 120)).save(origen)
            destino = pp.convert(
                origen, tmp / "out.png", "PNG",
                adjustments=pp.Adjustments(brightness=2.0),
            )
            self.assertGreater(Image.open(destino).getpixel((0, 0))[0], 120)


class PhotoWorkerTest(unittest.TestCase):
    _STUB = """
import argparse, json, shutil
p = argparse.ArgumentParser()
p.add_argument("--input"); p.add_argument("--output")
p.add_argument("--scale"); p.add_argument("--face", action="store_true")
p.add_argument("--model-dir")
a = p.parse_args()
shutil.copyfile(a.input, a.output)
print(json.dumps({"ok": True, "out": a.output}))
"""

    @classmethod
    def setUpClass(cls):
        from PyQt6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def test_ejecuta_el_worker_y_devuelve_la_salida(self):
        from ui.photo_widget import PhotoWorker

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            stub = tmp / "worker.py"
            stub.write_text(self._STUB, encoding="utf-8")
            entrada = tmp / "in.png"
            Image.new("RGB", (6, 6), (1, 2, 3)).save(entrada)
            salida = tmp / "out.png"
            worker = PhotoWorker(
                python=Path(sys.executable),
                worker=stub,
                input_path=entrada,
                output_path=salida,
                scale=2,
                face=False,
                model_dir=tmp,
            )
            resultado = worker.work()
            self.assertEqual(resultado, salida)
            self.assertTrue(salida.is_file())


if __name__ == "__main__":
    unittest.main()
