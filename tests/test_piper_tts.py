"""Tests de `engines.piper_tts` con dobles (sin Piper ni voces reales).

El motor escribe WAV y cachea voces ONNX, y su nombre de voz llega de la UI o de
`settings.json` para construir una ruta: aquí se cubren las dos cosas sin cargar
un modelo de verdad. Añadidos en la auditoría del 30/09/2026 (el motor no tenía
test propio).
"""

import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engines import piper_tts as pt  # noqa: E402
from engines.base_engine import EngineError  # noqa: E402


class _Chunk:
    """Doble de un `AudioChunk` de Piper."""

    sample_rate = 22050
    sample_width = 2
    sample_channels = 1
    audio_int16_bytes = b"\x00\x01" * 32


class _FakeVoice:
    """Doble de `PiperVoice`: cuenta cargas y sintetiza un chunk fijo."""

    loads: list[Path] = []

    def __init__(self, model_path) -> None:
        self.model_path = Path(model_path)

    @classmethod
    def load(cls, model_path):
        cls.loads.append(Path(model_path))
        return cls(model_path)

    def synthesize(self, text, syn_config):
        yield _Chunk()


class _FakeSynthesisConfig:
    def __init__(self, length_scale: float = 1.0) -> None:
        self.length_scale = length_scale


class PiperTTSEngineTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        raiz = Path(self._tmp.name)
        self.voices = raiz / "voices"
        self.voices.mkdir()
        (self.voices / "es_ES-davefx-medium.onnx").write_bytes(b"modelo")
        (self.voices / "eu_ES-antton-medium.onnx").write_bytes(b"modelo")
        self.output = raiz / "output"

        _FakeVoice.loads = []
        self._patches = [
            mock.patch.object(pt, "PiperVoice", _FakeVoice),
            mock.patch.object(pt, "SynthesisConfig", _FakeSynthesisConfig),
        ]
        for patcher in self._patches:
            patcher.start()

        self.engine = pt.PiperTTSEngine(
            voices_dir=self.voices,
            default_voice="es_ES-davefx-medium",
            output_dir=self.output,
        )

    def tearDown(self) -> None:
        for patcher in self._patches:
            patcher.stop()
        self._tmp.cleanup()

    # ---- Catálogo --------------------------------------------------

    def test_lista_las_voces_ordenadas(self) -> None:
        self.assertEqual(
            self.engine.list_voices(),
            ["es_ES-davefx-medium", "eu_ES-antton-medium"],
        )
        self.assertEqual(self.engine.list_models(), self.engine.list_voices())

    def test_estan_disponible_con_voces_y_libreria(self) -> None:
        self.assertTrue(self.engine.is_available())

    def test_no_esta_disponible_sin_voces(self) -> None:
        vacio = Path(self._tmp.name) / "vacio"
        vacio.mkdir()
        self.assertFalse(
            pt.PiperTTSEngine(voices_dir=vacio, output_dir=self.output).is_available()
        )

    # ---- Síntesis --------------------------------------------------

    def test_sintetiza_un_wav_valido(self) -> None:
        destino = self.engine.synthesize("hola mundo", output_path=None)

        self.assertTrue(destino.is_file())
        with wave.open(str(destino), "rb") as wav:
            self.assertEqual(wav.getnchannels(), 1)
            self.assertEqual(wav.getsampwidth(), 2)
            self.assertEqual(wav.getframerate(), 22050)
            self.assertEqual(wav.getnframes(), 32)

    def test_salida_por_defecto_lleva_la_voz_en_el_nombre(self) -> None:
        destino = self.engine.synthesize("hola")
        self.assertTrue(destino.name.startswith("plataformaia_"))
        self.assertTrue(destino.name.endswith("_es_ES-davefx-medium.wav"))
        self.assertEqual(destino.parent, self.output)

    def test_reutiliza_la_voz_cacheada(self) -> None:
        self.engine.synthesize("uno")
        self.engine.synthesize("dos")

        self.assertEqual(len(_FakeVoice.loads), 1)
        self.assertEqual(_FakeVoice.loads[0].name, "es_ES-davefx-medium.onnx")

    def test_carga_cada_voz_una_vez(self) -> None:
        self.engine.synthesize("uno", voice="es_ES-davefx-medium")
        self.engine.synthesize("dos", voice="eu_ES-antton-medium")
        self.engine.synthesize("tres", voice="es_ES-davefx-medium")

        self.assertEqual([p.name for p in _FakeVoice.loads], [
            "es_ES-davefx-medium.onnx",
            "eu_ES-antton-medium.onnx",
        ])

    # ---- Entradas inválidas ----------------------------------------

    def test_texto_vacio_no_sintetiza(self) -> None:
        with self.assertRaises(EngineError):
            self.engine.synthesize("   ")

    def test_voz_inexistente_no_carga_nada(self) -> None:
        with self.assertRaises(EngineError) as ctx:
            self.engine.synthesize("hola", voice="no-existe")

        self.assertIn("no-existe", str(ctx.exception))
        self.assertEqual(_FakeVoice.loads, [])

    def test_una_voz_con_ruta_no_sale_del_directorio_de_voces(self) -> None:
        """El nombre llega de la UI o de settings.json: no puede ser una ruta."""
        for malicioso in ("../secreto", "..\\..\\windows\\system32\\x", "/etc/passwd"):
            with self.assertRaises(EngineError):
                self.engine.synthesize("hola", voice=malicioso)
        self.assertEqual(_FakeVoice.loads, [])

    def test_el_nombre_del_fichero_de_salida_va_saneado(self) -> None:
        destino = self.engine._default_output_path("../../otra/voz rara")
        self.assertEqual(destino.parent, self.output)
        self.assertNotIn("/", destino.name)
        self.assertNotIn("\\", destino.name)
        self.assertIn("otra_voz_rara", destino.name)

    def test_sin_piper_instalado_avisa(self) -> None:
        with mock.patch.object(pt, "PiperVoice", None):
            with self.assertRaises(EngineError) as ctx:
                self.engine.synthesize("hola")
        self.assertIn("Piper no está instalado", str(ctx.exception))

    def test_sin_voces_no_puede_elegir_voz(self) -> None:
        vacio = Path(self._tmp.name) / "vacio"
        vacio.mkdir()
        engine = pt.PiperTTSEngine(voices_dir=vacio, output_dir=self.output)
        with self.assertRaises(EngineError) as ctx:
            engine.synthesize("hola")
        self.assertIn("No hay voces", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
