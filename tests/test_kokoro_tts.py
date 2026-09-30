"""Tests de `engines.kokoro_tts` con dobles (sin kokoro/torch ni descargas).

Kokoro se importa de forma perezosa (importa torch) y el pipeline se carga en la
primera síntesis: aquí se inyecta un pipeline falso, se comprueba el WAV, la
inversión de la velocidad y que el modelo se carga una sola vez. Añadidos en la
auditoría del 30/09/2026.
"""

import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from engines import kokoro_tts as kt  # noqa: E402
from engines.base_engine import EngineError  # noqa: E402


class _Audio:
    def __init__(self, valores) -> None:
        self._valores = np.array(valores, dtype=np.float32)

    def numpy(self):
        return self._valores


class _Result:
    def __init__(self, valores) -> None:
        self.audio = _Audio(valores)


class _Pipeline:
    """Doble del KPipeline: recuerda los argumentos de cada llamada."""

    builds: list[dict] = []
    calls: list[dict] = []

    def __init__(self, **kwargs) -> None:
        _Pipeline.builds.append(kwargs)
        self.kwargs = kwargs

    def __call__(self, text, voice=None, speed=1.0):
        _Pipeline.calls.append({"text": text, "voice": voice, "speed": speed})
        # 1.5 queda fuera de [-1, 1] a propósito: la escritura debe recortarlo.
        yield _Result([0.0, 0.5, -0.5, 1.5])

    @staticmethod
    def cargador():
        return _Pipeline


class KokoroTTSEngineTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.output = Path(self._tmp.name) / "output"
        _Pipeline.builds = []
        _Pipeline.calls = []
        self._patches = [
            mock.patch.object(kt, "_instalado", lambda modulo: True),
            mock.patch.object(kt, "_cargar_pipeline", _Pipeline.cargador),
        ]
        for patcher in self._patches:
            patcher.start()
        self.engine = kt.KokoroTTSEngine(output_dir=self.output)

    def tearDown(self) -> None:
        for patcher in self._patches:
            patcher.stop()
        self._tmp.cleanup()

    def test_lista_las_voces_en_espanol(self) -> None:
        self.assertEqual(self.engine.list_voices(), kt.KOKORO_VOICES)
        self.assertEqual(self.engine.list_models(), kt.KOKORO_VOICES)

    def test_sintetiza_un_wav_mono_a_24k(self) -> None:
        destino = self.engine.synthesize("hola")

        self.assertTrue(destino.is_file())
        self.assertTrue(destino.name.startswith("kokoro_"))
        with wave.open(str(destino), "rb") as wav:
            self.assertEqual(wav.getnchannels(), 1)
            self.assertEqual(wav.getsampwidth(), 2)
            self.assertEqual(wav.getframerate(), kt._SAMPLE_RATE)
            self.assertEqual(wav.getnframes(), 4)

    def test_recorta_las_muestras_fuera_de_rango(self) -> None:
        destino = self.engine.synthesize("hola")

        with wave.open(str(destino), "rb") as wav:
            muestras = np.frombuffer(wav.readframes(4), dtype=np.int16)
        self.assertEqual(int(muestras[3]), 32767)

    def test_invierte_la_velocidad_respecto_a_piper(self) -> None:
        """La UI usa 1.5 = más lento; Kokoro lo entiende al revés."""
        self.engine.synthesize("hola", speed=2.0)
        self.assertAlmostEqual(_Pipeline.calls[0]["speed"], 0.5)

    def test_una_sola_carga_del_modelo(self) -> None:
        self.engine.synthesize("uno")
        self.engine.synthesize("dos")

        self.assertEqual(len(_Pipeline.builds), 1)
        self.assertEqual(_Pipeline.builds[0]["lang_code"], "e")

    def test_texto_vacio_avisa(self) -> None:
        with self.assertRaises(EngineError):
            self.engine.synthesize("  ")

    def test_voz_desconocida_avisa(self) -> None:
        with self.assertRaises(EngineError) as ctx:
            self.engine.synthesize("hola", voice="em_inexistente")
        self.assertIn("no existe", str(ctx.exception))

    def test_sin_dependencias_avisa(self) -> None:
        with mock.patch.object(kt, "_instalado", lambda modulo: False):
            self.assertFalse(self.engine.is_available())
            with self.assertRaises(EngineError) as ctx:
                self.engine.synthesize("hola")
        self.assertIn("kokoro no está instalado", str(ctx.exception))

    def test_is_available_consulta_los_modulos(self) -> None:
        consultados: list[str] = []

        def espia(modulo: str) -> bool:
            consultados.append(modulo)
            return True

        with mock.patch.object(kt, "_instalado", espia):
            self.assertTrue(kt.KokoroTTSEngine().is_available())
        self.assertEqual(consultados, ["kokoro", "torch"])


if __name__ == "__main__":
    unittest.main()
