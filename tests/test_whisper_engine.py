"""Tests de `engines.whisper_engine` con dobles (sin faster-whisper ni modelos).

El motor carga CTranslate2 y escribe SRT: aquí se inyecta un modelo falso y se
comprueba el formato del SRT, el cacheo del modelo por tamaño, el progreso y los
errores de entrada. Añadidos en la auditoría del 30/09/2026.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engines import whisper_engine as we  # noqa: E402
from engines.base_engine import EngineError  # noqa: E402


class _Info(SimpleNamespace):
    def __init__(self, duration: float = 0.0) -> None:
        super().__init__(duration=duration, language="es")


class _FakeWhisperModel:
    """Doble de `WhisperModel`: cuenta las cargas y devuelve segmentos fijos."""

    builds: list[str] = []
    segments: list = []
    duration = 0.0

    def __init__(self, model_size, device=None, compute_type=None) -> None:
        _FakeWhisperModel.builds.append(model_size)
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type

    def transcribe(self, path, language=None):
        self.ultimo_idioma = language
        return iter(list(_FakeWhisperModel.segments)), _Info(_FakeWhisperModel.duration)


def _segmento(start, end, text):
    return SimpleNamespace(start=start, end=end, text=text)


class WhisperEngineTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.raiz = Path(self._tmp.name)
        self.audio = self.raiz / "voz.wav"
        self.audio.write_bytes(b"RIFF")

        _FakeWhisperModel.builds = []
        _FakeWhisperModel.segments = [
            _segmento(0.0, 1.5, "  hola  "),
            _segmento(1.5, 3.0, ""),          # vacío: se ignora
            _segmento(3.0, 4.25, "adiós"),
        ]
        _FakeWhisperModel.duration = 10.0

        self._patcher = mock.patch.object(we, "WhisperModel", _FakeWhisperModel)
        self._patcher.start()
        self.engine = we.WhisperEngine(model_size="small")

    def tearDown(self) -> None:
        self._patcher.stop()
        self._tmp.cleanup()

    # ---- Catálogo --------------------------------------------------

    def test_lista_los_tamanos_disponibles(self) -> None:
        self.assertEqual(self.engine.list_models(), we.WHISPER_MODELS)
        self.assertTrue(self.engine.is_available())

    def test_sin_faster_whisper_avisa(self) -> None:
        with mock.patch.object(we, "WhisperModel", None):
            self.assertFalse(self.engine.is_available())
            with self.assertRaises(EngineError) as ctx:
                self.engine.transcribe(self.audio)
        self.assertIn("faster-whisper no está instalado", str(ctx.exception))

    # ---- Transcripción ---------------------------------------------

    def test_escribe_el_srt_junto_al_audio(self) -> None:
        srt = self.engine.transcribe(self.audio)

        self.assertEqual(srt, self.audio.with_suffix(".srt"))
        texto = srt.read_text(encoding="utf-8")
        self.assertIn("1\n00:00:00,000 --> 00:00:01,500\nhola\n", texto)
        self.assertIn("2\n00:00:03,000 --> 00:00:04,250\nadiós\n", texto)
        self.assertNotIn("3\n", texto)  # el segmento vacío no cuenta

    def test_respeta_el_destino_y_el_idioma(self) -> None:
        destino = self.raiz / "sub" / "salida.srt"
        srt = self.engine.transcribe(self.audio, language="eu", output_srt=destino)

        self.assertEqual(srt, destino)
        self.assertTrue(srt.is_file())
        self.assertEqual(self.engine._model.ultimo_idioma, "eu")

    def test_emite_progreso(self) -> None:
        vistos: list[tuple[int, str]] = []
        self.engine.transcribe(self.audio, progress_callback=lambda p, m: vistos.append((p, m)))

        self.assertTrue(vistos)
        self.assertTrue(all(0 <= p <= 99 for p, _ in vistos))

    def test_audio_inexistente_avisa(self) -> None:
        with self.assertRaises(EngineError) as ctx:
            self.engine.transcribe(self.raiz / "no-existe.wav")
        self.assertIn("no existe", str(ctx.exception))

    def test_sin_segmentos_no_deja_srt_vacio(self) -> None:
        _FakeWhisperModel.segments = []
        with self.assertRaises(EngineError) as ctx:
            self.engine.transcribe(self.audio)

        self.assertIn("no produjo subtítulos", str(ctx.exception))
        self.assertFalse(self.audio.with_suffix(".srt").exists())

    def test_un_fallo_del_modelo_se_traduce_a_engine_error(self) -> None:
        with mock.patch.object(
            _FakeWhisperModel, "transcribe", side_effect=RuntimeError("CTranslate2 roto")
        ):
            with self.assertRaises(EngineError) as ctx:
                self.engine.transcribe(self.audio)
        self.assertIn("no se pudo transcribir", str(ctx.exception))

    # ---- Cacheo del modelo -----------------------------------------

    def test_reutiliza_el_modelo_entre_transcripciones(self) -> None:
        self.engine.transcribe(self.audio)
        self.engine.transcribe(self.audio)

        self.assertEqual(_FakeWhisperModel.builds, ["small"])

    def test_recarga_si_cambia_el_tamano(self) -> None:
        self.engine.transcribe(self.audio)
        self.engine.transcribe(self.audio, model_size="medium")

        self.assertEqual(_FakeWhisperModel.builds, ["small", "medium"])
        self.assertEqual(self.engine._model_size, "medium")

    def test_una_carga_fallida_no_cambia_el_estado(self) -> None:
        def explota(model_size, device=None, compute_type=None):
            raise RuntimeError("sin disco")

        with mock.patch.object(we, "WhisperModel", explota):
            with self.assertRaises(EngineError):
                self.engine.transcribe(self.audio, model_size="large-v3")

        self.assertEqual(self.engine._model_size, "small")

    # ---- Formato ---------------------------------------------------

    def test_formatea_timestamps_en_formato_srt(self) -> None:
        self.assertEqual(we.WhisperEngine._format_timestamp(0), "00:00:00,000")
        self.assertEqual(we.WhisperEngine._format_timestamp(3661.5), "01:01:01,500")
        self.assertEqual(we.WhisperEngine._format_timestamp(-3), "00:00:00,000")


if __name__ == "__main__":
    unittest.main()
