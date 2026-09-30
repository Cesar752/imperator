"""Tests del motor MusicGen (música local, motor opcional).

``audiocraft`` (y por debajo torch y ffmpeg) es una dependencia pesada que no
está en la suite de tests, así que aquí se sustituyen el modelo y la escritura
del WAV por dobles. Se comprueba la lógica del motor —validación, tamaño y
duración, caché, destino del fichero y errores—, no la generación real.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engines import musicgen as mg  # noqa: E402
from engines.base_engine import EngineError, MusicEngine  # noqa: E402

_PROMPT = "rock épico con guitarras y batería"


class _FakeTensor:
    def cpu(self):
        return self


class _FakeModel:
    """Doble del modelo de MusicGen: registra parámetros y prompts recibidos."""

    def __init__(self, sample_rate: int = 32000) -> None:
        self.sample_rate = sample_rate
        self.params: dict | None = None
        self.prompts: list[str] | None = None

    def set_generation_params(self, **kwargs) -> None:
        self.params = kwargs

    def generate(self, prompts):
        self.prompts = list(prompts)
        return [_FakeTensor()]


class MusicGenEngineTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.output = Path(self._tmp.name)
        self.audio_calls: list[tuple] = []
        self.repos: list[str] = []
        self.model = _FakeModel()

        fake_musicgen = mock.Mock()
        fake_musicgen.get_pretrained.side_effect = self._get_pretrained
        patcher = mock.patch.multiple(
            mg,
            MusicGen=fake_musicgen,
            audio_write=mock.Mock(side_effect=self._audio_write),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

        self.engine = mg.MusicGenEngine(output_dir=self.output)

    def _get_pretrained(self, repo_id: str) -> _FakeModel:
        self.repos.append(repo_id)
        return self.model

    def _audio_write(self, path, wav, sample_rate, strategy=None) -> None:
        self.audio_calls.append((path, wav, sample_rate, strategy))
        Path(f"{path}.wav").write_bytes(b"RIFF....WAVE")

    # ---- Contrato --------------------------------------------------

    def test_cumple_el_contrato_de_motor_de_musica(self) -> None:
        self.assertIsInstance(self.engine, MusicEngine)
        self.assertEqual(self.engine.id, "musicgen")
        self.assertEqual(self.engine.nombre, "MusicGen")
        self.assertEqual(self.engine.list_models(), list(mg.MODEL_SIZES))
        self.assertTrue(self.engine.is_available())

    # ---- Generación ------------------------------------------------

    def test_genera_un_wav_a_partir_del_prompt(self) -> None:
        target = self.engine.generate_music(_PROMPT)
        self.assertTrue(target.exists())
        self.assertEqual(target.suffix, ".wav")
        self.assertEqual(target.parent, self.output)
        self.assertEqual(self.model.prompts, [_PROMPT])
        self.assertEqual(self.repos, [mg.REPO_IDS[mg.DEFAULT_MODEL]])

    def test_escribe_el_audio_con_la_frecuencia_del_modelo(self) -> None:
        self.engine.generate_music(_PROMPT)
        path, _, sample_rate, strategy = self.audio_calls[0]
        self.assertEqual(sample_rate, self.model.sample_rate)
        self.assertEqual(strategy, "loudness")
        self.assertFalse(path.endswith(".wav"), "audio_write añade la extensión")

    def test_acota_la_duracion_al_rango_de_musicgen(self) -> None:
        self.engine.generate_music(_PROMPT, duration=999)
        self.assertEqual(self.model.params["duration"], mg.MAX_DURATION)

        self.engine.generate_music(_PROMPT, duration=0)
        self.assertEqual(self.model.params["duration"], mg.MIN_DURATION)

        self.engine.generate_music(_PROMPT, duration="no-es-un-numero")
        self.assertEqual(self.model.params["duration"], mg.DEFAULT_DURATION)

    def test_elige_el_tamano_del_modelo(self) -> None:
        self.engine.generate_music(_PROMPT, model="medium")
        self.assertEqual(self.repos[-1], mg.REPO_IDS["medium"])

        self.engine.generate_music(_PROMPT, model="gigante")
        self.assertEqual(self.repos[-1], mg.REPO_IDS[mg.DEFAULT_MODEL])

    def test_cachea_el_modelo_entre_generaciones(self) -> None:
        self.engine.generate_music("una cosa")
        self.engine.generate_music("otra cosa")
        self.assertEqual(len(self.repos), 1, "el modelo se carga una sola vez")

    def test_normaliza_el_destino_a_wav(self) -> None:
        destino = self.output / "mi_musica"
        target = self.engine.generate_music(_PROMPT, output_path=destino)
        self.assertEqual(target, destino.with_suffix(".wav"))
        self.assertTrue(target.exists())

    # ---- Errores ---------------------------------------------------

    def test_prompt_vacio(self) -> None:
        with self.assertRaises(EngineError):
            self.engine.generate_music("   ")

    def test_sin_audiocraft_avisa_y_no_esta_disponible(self) -> None:
        with mock.patch.object(mg, "MusicGen", None):
            self.assertFalse(self.engine.is_available())
            with self.assertRaises(EngineError) as ctx:
                self.engine.generate_music(_PROMPT)
        self.assertIn("audiocraft", str(ctx.exception))

    def test_un_fallo_de_generacion_no_deja_fichero(self) -> None:
        self.model.generate = mock.Mock(side_effect=RuntimeError("sin memoria"))
        destino = self.output / "fallo.wav"
        with self.assertRaises(EngineError) as ctx:
            self.engine.generate_music(_PROMPT, output_path=destino)
        self.assertIn("sin memoria", str(ctx.exception))
        self.assertFalse(destino.exists())

    def test_avisa_si_no_se_produjo_audio(self) -> None:
        with mock.patch.object(mg, "audio_write", mock.Mock()):
            with self.assertRaises(EngineError) as ctx:
                self.engine.generate_music(
                    _PROMPT, output_path=self.output / "vacio.wav"
                )
        self.assertIn("no produjo audio", str(ctx.exception))

    # ---- Integración con el registro -------------------------------

    def test_la_configuracion_del_motor_llega_al_constructor(self) -> None:
        from core.engine_manager import EngineManager

        engine = EngineManager._build(
            "musicgen",
            {
                "id": "musicgen",
                "ruta": "local",
                "parametros": {"model": "medium", "duration": 12},
            },
        )
        self.assertIsInstance(engine, mg.MusicGenEngine)
        engine.generate_music("sin indicar nada")
        self.assertEqual(self.repos[-1], mg.REPO_IDS["medium"])
        self.assertEqual(self.model.params["duration"], 12.0)


class MusicGenRegistrationTest(unittest.TestCase):
    """El motor debe ser descubrible: factoría, configuración y UI de motores."""

    def test_esta_en_la_factoria_de_motores(self) -> None:
        from core.engine_manager import ENGINE_FACTORY

        self.assertIs(ENGINE_FACTORY["musicgen"], mg.MusicGenEngine)

    def test_esta_en_la_configuracion_por_defecto(self) -> None:
        from core.config_manager import DEFAULT_ENGINES

        ids = [entrada["id"] for entrada in DEFAULT_ENGINES]
        self.assertIn("musicgen", ids)

    def test_la_ui_de_motores_cubre_todos_los_tipos(self) -> None:
        from core.engine_manager import ENGINE_FACTORY
        from ui import engines_dialog as ed

        self.assertEqual(set(ed._TYPE_TEMPLATES), set(ENGINE_FACTORY))
        self.assertEqual(set(ed._TYPE_URLS), set(ENGINE_FACTORY))


if __name__ == "__main__":
    unittest.main()
