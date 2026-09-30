"""Tests de engines.perception y de la familia Perception.

Las dependencias pesadas (ultralytics, transformers/torch, paddleocr) no están
instaladas en CI, así que los motores se prueban con dobles inyectados: el
modelo de YOLO, el runner de CLIP y el objeto de PaddleOCR. También se cubre el
contrato ``Perception`` (agregación y resumen) y la construcción desde el
``EngineManager``.
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config_manager import ConfigManager  # noqa: E402
from core.engine_manager import ENGINE_FACTORY, EngineManager  # noqa: E402
from engines.base_engine import EngineError, Perception, PerceptionEngine  # noqa: E402
from engines.perception import (  # noqa: E402
    ClipEngine,
    PaddleOcrEngine,
    YoloEngine,
    parse_ocr_output,
)


class PerceptionContractTest(unittest.TestCase):
    def test_vacia(self) -> None:
        self.assertTrue(Perception().is_empty)
        self.assertFalse(Perception(text="algo").is_empty)

    def test_resumen_cuenta_etiquetas(self) -> None:
        percepcion = Perception(
            objects=({"label": "person"}, {"label": "person"}, {"label": "car"}),
            text="hola",
        )
        resumen = percepcion.summary()
        self.assertIn("Objetos: person (x2), car", resumen)
        self.assertIn("Texto: hola", resumen)

    def test_resumen_omite_vacias(self) -> None:
        self.assertEqual(Perception().summary(), "")

    def test_merge_une_todo(self) -> None:
        a = Perception(objects=({"label": "car"},), text="uno")
        b = Perception(labels=({"label": "calle"},), text="dos", caption="una calle")
        unido = a.merge(b)
        self.assertEqual(len(unido.objects), 1)
        self.assertEqual(len(unido.labels), 1)
        self.assertEqual(unido.text, "uno\ndos")
        self.assertEqual(unido.caption, "una calle")

    def test_to_dict(self) -> None:
        data = Perception(objects=({"label": "car"},), text="x").to_dict()
        self.assertEqual(data["objects"][0]["label"], "car")
        self.assertEqual(set(data), {"objects", "labels", "text", "caption"})

    def test_es_una_familia_de_motor(self) -> None:
        self.assertTrue(issubclass(PerceptionEngine, object))
        self.assertTrue(hasattr(PerceptionEngine, "perceive"))


class _Box:
    def __init__(self, cls, conf, xyxy) -> None:
        self.cls = [cls]
        self.conf = [conf]
        self.xyxy = [xyxy]


class _Result:
    def __init__(self, names, boxes) -> None:
        self.names = names
        self.boxes = boxes


class _FakeYolo:
    def __init__(self, results) -> None:
        self._results = results
        self.calls: list = []

    def predict(self, path, conf=None, verbose=False):
        self.calls.append((path, conf))
        return self._results


class YoloEngineTest(unittest.TestCase):
    def _imagen(self, tmp: str) -> Path:
        path = Path(tmp) / "foto.png"
        path.write_bytes(b"\x89PNG\r\n")
        return path

    def test_detecta_objetos(self) -> None:
        modelo = _FakeYolo(
            [_Result({0: "person", 2: "car"}, [_Box(0, 0.9, [1, 2, 3, 4]), _Box(2, 0.5, [5, 6, 7, 8])])]
        )
        with tempfile.TemporaryDirectory() as tmp:
            engine = YoloEngine(model=modelo, confidence=0.4)
            percepcion = engine.perceive(self._imagen(tmp))
        self.assertEqual(
            [o["label"] for o in percepcion.objects], ["person", "car"]
        )
        self.assertEqual(percepcion.objects[0]["score"], 0.9)
        self.assertEqual(modelo.calls[0][1], 0.4, "propaga la confianza")

    def test_imagen_inexistente(self) -> None:
        engine = YoloEngine(model=_FakeYolo([]))
        with self.assertRaises(EngineError):
            engine.perceive("C:/no/existe.png")

    def test_sin_ultralytics_no_disponible(self) -> None:
        engine = YoloEngine()
        if engine.is_available():  # ultralytics instalado en este entorno
            self.skipTest("ultralytics está instalado")
        with self.assertRaises(EngineError):
            engine.perceive("x.png")

    def test_metadatos(self) -> None:
        engine = YoloEngine(model=_FakeYolo([]))
        self.assertEqual(engine.id, "yolo")
        self.assertTrue(engine.is_available())
        self.assertIn("yolov8n.pt", engine.list_models())


class ClipEngineTest(unittest.TestCase):
    def _imagen(self, tmp: str) -> Path:
        path = Path(tmp) / "foto.png"
        path.write_bytes(b"\x89PNG\r\n")
        return path

    def test_puntua_y_ordena(self) -> None:
        engine = ClipEngine(
            labels=("gato", "perro", "coche"),
            top_k=2,
            runner=lambda _path, _labels: [0.1, 0.8, 0.1],
        )
        with tempfile.TemporaryDirectory() as tmp:
            percepcion = engine.perceive(self._imagen(tmp))
        self.assertEqual(
            [etiqueta["label"] for etiqueta in percepcion.labels], ["perro", "gato"]
        )
        self.assertEqual(percepcion.labels[0]["score"], 0.8)

    def test_etiquetas_desde_el_prompt(self) -> None:
        capturado: dict = {}

        def runner(_path, labels):
            capturado["labels"] = labels
            return [0.5] * len(labels)

        engine = ClipEngine(runner=runner)
        with tempfile.TemporaryDirectory() as tmp:
            engine.perceive(self._imagen(tmp), prompt="sol, luna")
        self.assertEqual(capturado["labels"], ("sol", "luna"))

    def test_descarta_probabilidad_cero(self) -> None:
        engine = ClipEngine(labels=("a", "b"), runner=lambda _p, _l: [0.0, 1.0])
        with tempfile.TemporaryDirectory() as tmp:
            percepcion = engine.perceive(self._imagen(tmp))
        self.assertEqual([etiqueta["label"] for etiqueta in percepcion.labels], ["b"])

    def test_imagen_inexistente(self) -> None:
        engine = ClipEngine(runner=lambda _p, _l: [1.0])
        with self.assertRaises(EngineError):
            engine.perceive("C:/no/existe.png")

    def test_sin_dependencias_no_disponible(self) -> None:
        engine = ClipEngine()
        if engine.is_available():
            self.skipTest("transformers/torch instalados")
        with self.assertRaises(EngineError):
            engine.perceive("x.png")


class ParseOcrOutputTest(unittest.TestCase):
    def test_vacio(self) -> None:
        self.assertEqual(parse_ocr_output(None), "")
        self.assertEqual(parse_ocr_output([]), "")

    def test_forma_2x_anidada_por_pagina(self) -> None:
        raw = [
            [
                [[[0, 0], [1, 0], [1, 1], [0, 1]], ("hola", 0.99)],
                [[[0, 2], [1, 2], [1, 3], [0, 3]], ("mundo", 0.9)],
            ]
        ]
        self.assertEqual(parse_ocr_output(raw), "hola\nmundo")

    def test_forma_3x_dict(self) -> None:
        raw = [{"rec_texts": ["uno", "dos"]}]
        self.assertEqual(parse_ocr_output(raw), "uno\ndos")

    def test_forma_3x_objeto(self) -> None:
        class _Resultado:
            rec_texts = ["tres"]

        self.assertEqual(parse_ocr_output([_Resultado()]), "tres")

    def test_estructura_desconocida(self) -> None:
        self.assertEqual(parse_ocr_output([[["x"]]]), "")


class _FakeOcrPredict:
    def __init__(self, raw) -> None:
        self._raw = raw
        self.calls: list = []

    def predict(self, path):
        self.calls.append(path)
        return self._raw


class _FakeOcrClassic:
    def __init__(self, raw) -> None:
        self.raw = raw

    def ocr(self, path, cls=True):
        return self.raw


class PaddleOcrEngineTest(unittest.TestCase):
    def _imagen(self, tmp: str) -> Path:
        path = Path(tmp) / "doc.png"
        path.write_bytes(b"\x89PNG\r\n")
        return path

    def test_usa_predict_si_existe(self) -> None:
        ocr = _FakeOcrPredict([{"rec_texts": ["hola"]}])
        engine = PaddleOcrEngine(ocr=ocr)
        with tempfile.TemporaryDirectory() as tmp:
            percepcion = engine.perceive(self._imagen(tmp))
        self.assertEqual(percepcion.text, "hola")
        self.assertEqual(len(ocr.calls), 1)

    def test_usa_ocr_clasico_si_no_hay_predict(self) -> None:
        ocr = _FakeOcrClassic([[[[[0, 0]], ("texto", 0.9)]]])
        engine = PaddleOcrEngine(ocr=ocr)
        with tempfile.TemporaryDirectory() as tmp:
            percepcion = engine.perceive(self._imagen(tmp))
        self.assertEqual(percepcion.text, "texto")

    def test_imagen_inexistente(self) -> None:
        engine = PaddleOcrEngine(ocr=_FakeOcrPredict([]))
        with self.assertRaises(EngineError):
            engine.perceive("C:/no/existe.png")

    def test_idioma_desconocido_cae_a_latin(self) -> None:
        engine = PaddleOcrEngine(lang="klingon", ocr=_FakeOcrPredict([]))
        self.assertTrue(engine.is_available())

    def test_metadatos(self) -> None:
        engine = PaddleOcrEngine(ocr=_FakeOcrPredict([]))
        self.assertEqual(engine.id, "paddle_ocr")
        self.assertIn("latin", engine.list_models())


class EngineManagerPerceptionTest(unittest.TestCase):
    def test_registrados_en_el_factory(self) -> None:
        self.assertIn("yolo", ENGINE_FACTORY)
        self.assertIn("clip", ENGINE_FACTORY)
        self.assertIn("paddle_ocr", ENGINE_FACTORY)

    def test_construidos_desde_default_engines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = EngineManager(ConfigManager(config_dir=tmp))
            self.assertIsInstance(manager.get_instance("yolo"), YoloEngine)
            self.assertIsInstance(manager.get_instance("clip"), ClipEngine)
            self.assertIsInstance(
                manager.get_instance("paddle_ocr"), PaddleOcrEngine
            )
            # Deshabilitados por defecto: existen pero no se ofrecen a las tareas.
            self.assertIsNone(manager.get_engine("yolo"))
            self.assertIn("yolo", manager.supported_types())


if __name__ == "__main__":
    unittest.main()
