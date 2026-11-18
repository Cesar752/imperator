"""Tests del backend por subprocess del clonador de voz.

No requieren coqui-tts ni torch: usan un worker de prueba ejecutado con el propio
intérprete de Python (``sys.executable``).
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engines import voice_clone as vc  # noqa: E402
from engines.base_engine import EngineError  # noqa: E402

# Worker de prueba: escribe en --out "idioma|speaker|texto" y responde JSON ok.
_OK_WORKER = """
import argparse, json
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("--text"); p.add_argument("--speaker"); p.add_argument("--out")
p.add_argument("--language", default="es"); p.add_argument("--model", default="")
p.add_argument("--device", default=""); p.add_argument("--speed", type=float, default=None)
a = p.parse_args()
Path(a.out).write_text(f"{a.language}|{a.speaker}|{a.text}", encoding="utf-8")
print(json.dumps({"ok": True, "out": a.out}))
"""

# Worker que falla: JSON de error + código 1.
_FAIL_WORKER = """
import json, sys
print(json.dumps({"ok": False, "error": "boom"}))
sys.exit(1)
"""


def _write(tmp: Path, name: str, content: str) -> Path:
    ruta = tmp / name
    ruta.write_text(content, encoding="utf-8")
    return ruta


def _engine(tmp: Path, worker: Path, **kwargs) -> vc.VoiceCloneEngine:
    return vc.VoiceCloneEngine(
        backend="subprocess",
        python=sys.executable,
        worker=str(worker),
        voices_dir=tmp,
        output_dir=tmp,
        **kwargs,
    )


class SubprocessBackendTest(unittest.TestCase):
    def test_disponible_con_python_y_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            worker = _write(tmp, "worker.py", _OK_WORKER)
            self.assertTrue(_engine(tmp, worker).is_available())

    def test_no_disponible_sin_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            ausente = tmp / "no_existe.py"
            engine = _engine(tmp, ausente)
            self.assertFalse(engine.is_available())
            (tmp / "a.wav").write_bytes(b"x")
            with self.assertRaises(EngineError):
                engine.synthesize("hola", voice="a")

    def test_sintetiza_y_pasa_los_argumentos(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            worker = _write(tmp, "worker.py", _OK_WORKER)
            (tmp / "ana.wav").write_bytes(b"x")
            engine = _engine(tmp, worker, language="eu")
            salida = engine.synthesize("Hola mundo", voice="ana")
            self.assertTrue(salida.is_file())
            self.assertEqual(salida.parent, tmp)
            contenido = salida.read_text(encoding="utf-8")
            self.assertEqual(contenido, f"eu|{tmp / 'ana.wav'}|Hola mundo")

    def test_worker_con_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            worker = _write(tmp, "worker.py", _FAIL_WORKER)
            (tmp / "a.wav").write_bytes(b"x")
            engine = _engine(tmp, worker)
            with self.assertRaises(EngineError) as ctx:
                engine.synthesize("hola", voice="a")
            self.assertIn("boom", str(ctx.exception))

    def test_backend_por_defecto_es_inprocess(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = vc.VoiceCloneEngine(voices_dir=tmp, output_dir=tmp)
            self.assertEqual(engine.backend, "inprocess")


if __name__ == "__main__":
    unittest.main()
