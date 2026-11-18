"""Tests del motor de clonado de voz (engines/voice_clone.py).

No requieren coqui-tts ni torch: se sustituye ``_load_tts`` (el import perezoso de
Coqui) por un doble que registra los argumentos y escribe el fichero de salida.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engines import voice_clone as vc  # noqa: E402
from engines.base_engine import EngineError  # noqa: E402


class _FakeModel:
    last_kwargs: dict | None = None

    def to(self, device):
        self.device = device
        return self

    def tts_to_file(self, **kwargs):
        _FakeModel.last_kwargs = kwargs
        Path(kwargs["file_path"]).write_bytes(b"RIFF")


class _FakeTTS:
    last_model_name: str | None = None

    def __call__(self, model_name, progress_bar=False):
        _FakeTTS.last_model_name = model_name
        return _FakeModel()


def _patch_tts(fake):
    """Sustituye el import perezoso de Coqui por ``fake`` (o ``None``)."""
    return mock.patch.object(vc, "_load_tts", return_value=fake)


def _engine(tmp, **kwargs):
    return vc.VoiceCloneEngine(
        backend="inprocess",
        device="cpu",
        voices_dir=tmp,
        output_dir=tmp,
        **kwargs,
    )


class AvailabilityTest(unittest.TestCase):
    def test_disponible_si_tts_instalado(self):
        with tempfile.TemporaryDirectory() as tmp:
            with _patch_tts(_FakeTTS()):
                self.assertTrue(_engine(tmp).is_available())

    def test_no_disponible_sin_tts(self):
        with tempfile.TemporaryDirectory() as tmp:
            with _patch_tts(None):
                engine = _engine(tmp)
                self.assertFalse(engine.is_available())
                with self.assertRaises(EngineError):
                    engine.synthesize("hola", voice="ref")


class IdentityTest(unittest.TestCase):
    def test_id_nombre_modelos(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = _engine(tmp)
            self.assertEqual(engine.id, "voice_clone")
            self.assertIn("XTTS", engine.nombre)
            self.assertEqual(len(engine.list_models()), 1)


class ReferenceSamplesTest(unittest.TestCase):
    def test_lista_muestras(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "ana.wav").write_bytes(b"x")
            (Path(tmp) / "nota.txt").write_text("no es audio", encoding="utf-8")
            engine = _engine(tmp)
            self.assertEqual(engine.list_voices(), ["ana"])


class SynthesizeTest(unittest.TestCase):
    def test_clona_con_muestra_por_nombre(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "ana.wav").write_bytes(b"x")
            with _patch_tts(_FakeTTS()):
                engine = _engine(tmp)
                salida = engine.synthesize("Hola mundo", voice="ana")
            self.assertTrue(salida.is_file())
            self.assertEqual(salida.parent, tmp_path)
            self.assertEqual(_FakeModel.last_kwargs["language"], "es")
            self.assertTrue(str(_FakeModel.last_kwargs["speaker_wav"]).endswith("ana.wav"))
            self.assertEqual(_FakeModel.last_kwargs["text"], "Hola mundo")

    def test_clona_con_ruta_directa(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref = Path(tmp) / "muestra.wav"
            ref.write_bytes(b"x")
            with _patch_tts(_FakeTTS()):
                engine = _engine(tmp)
                engine.synthesize("Hola", voice=str(ref), language="eu")
            self.assertEqual(_FakeModel.last_kwargs["language"], "eu")
            self.assertEqual(_FakeModel.last_kwargs["speaker_wav"], str(ref))

    def test_usa_la_unica_muestra_si_no_se_indica_voz(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "unica.mp3").write_bytes(b"x")
            with _patch_tts(_FakeTTS()):
                engine = _engine(tmp)
                salida = engine.synthesize("Hola")
            self.assertTrue(salida.is_file())

    def test_varias_muestras_exige_elegir(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "a.wav").write_bytes(b"x")
            (Path(tmp) / "b.wav").write_bytes(b"x")
            with _patch_tts(_FakeTTS()):
                engine = _engine(tmp)
                with self.assertRaises(EngineError):
                    engine.synthesize("Hola")

    def test_sin_muestras_avisa(self):
        with tempfile.TemporaryDirectory() as tmp:
            with _patch_tts(_FakeTTS()):
                engine = _engine(tmp)
                with self.assertRaises(EngineError):
                    engine.synthesize("Hola")

    def test_texto_vacio(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "a.wav").write_bytes(b"x")
            with _patch_tts(_FakeTTS()):
                engine = _engine(tmp)
                with self.assertRaises(EngineError):
                    engine.synthesize("   ")


if __name__ == "__main__":
    unittest.main()
