"""Tests de la integración de CLIs externas (core/cli_runner.py + engines/cli_tool.py).

No dependen de ningún programa instalado: usan el propio intérprete de Python
(``sys.executable``) ejecutando un pequeño script de prueba escrito en un
directorio temporal, como haría cualquier herramienta externa.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

from core.cli_runner import CliRunner, resolve_executable, split_args
from engines.base_engine import EngineError
from engines.cli_tool import CliVideoEngine

# Escribe el contenido recibido en la ruta exacta pasada como primer argumento.
_WRITE_TO_PATH = (
    "import pathlib, sys;"
    "p = pathlib.Path(sys.argv[1]);"
    "p.parent.mkdir(parents=True, exist_ok=True);"
    "p.write_bytes(b'ok')"
)

# Escribe un artefacto con nombre fijo dentro del directorio pasado.
_WRITE_ARTIFACT = (
    "import pathlib, sys;"
    "d = pathlib.Path(sys.argv[1]);"
    "d.mkdir(parents=True, exist_ok=True);"
    "(d / 'artifact.mp4').write_bytes(b'ok')"
)

# Falla con código distinto de cero y escribe en stderr.
_FAIL = "import sys; sys.stderr.write('boom'); sys.exit(3)"

# Duerme más de lo que dura el timeout de prueba.
_SLEEP = "import time; time.sleep(30)"


class ResolveExecutableTest(unittest.TestCase):
    def test_ruta_absoluta_existente(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "tool.py"
            script.write_text("", encoding="utf-8")
            self.assertEqual(resolve_executable(str(script)), str(script))

    def test_vacio_y_ausente(self):
        self.assertIsNone(resolve_executable(None))
        self.assertIsNone(resolve_executable(""))
        self.assertIsNone(resolve_executable("no-existe-xyz-123"))

    def test_encuentra_en_carpeta_local_del_proyecto(self):
        # Instalación "todo en local": un ejecutable dentro de una carpeta del
        # proyecto se resuelve sin depender del PATH. En Windows un binario
        # resoluble lleva extensión (.exe/.bat/...), que es lo que aplica
        # shutil.which vía PATHEXT.
        ext = ".bat" if os.name == "nt" else ""
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / f"herramienta{ext}").write_text("", encoding="utf-8")
            encontrado = resolve_executable("herramienta", search_dirs=[tmp])
            self.assertIsNotNone(encontrado)
            # En Windows PATHEXT puede devolver la extensión en mayúsculas
            # (.BAT); el sistema de ficheros es insensible a mayúsculas.
            self.assertEqual(Path(encontrado).name.lower(), f"herramienta{ext}".lower())


class SplitArgsTest(unittest.TestCase):
    def test_lista_y_cadena(self):
        self.assertEqual(split_args(["--a", "{input}"]), ["--a", "{input}"])
        self.assertEqual(split_args(None), [])
        # La cadena se trocea respetando las comillas.
        self.assertIn("{prompt}", split_args('--prompt "{prompt}"'))


class CliRunnerTest(unittest.TestCase):
    def _runner(self, args, **kwargs) -> CliRunner:
        return CliRunner(sys.executable, label="Stub", args=args, **kwargs)

    def test_no_disponible_sin_ejecutable(self):
        runner = CliRunner("no-existe-xyz-123", label="Stub", args=["{output}"])
        self.assertFalse(runner.is_available())
        with self.assertRaises(EngineError):
            runner.run({"output_dir": "."})

    def test_hueco_desconocido_da_error_claro(self):
        runner = self._runner(["{desconocido}"])
        with self.assertRaises(EngineError):
            runner.build_args({"output_dir": "."})

    def test_sin_plantilla_da_error(self):
        runner = self._runner([])
        with self.assertRaises(EngineError):
            runner.run({"output_dir": "."})

    def test_invoke_escribe_en_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = self._runner(["-c", _WRITE_TO_PATH, "{output}"], output_dir=tmp, output_ext=".mp4")
            salida = runner.invoke({"prompt": "hola"})
            self.assertTrue(salida.is_file())
            self.assertEqual(salida.suffix, ".mp4")
            self.assertEqual(salida.parent, Path(tmp))

    def test_invoke_resuelve_por_glob(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = self._runner(["-c", _WRITE_ARTIFACT, "{output_dir}"], output_dir=tmp, output_glob="*.mp4")
            salida = runner.invoke({})
            self.assertEqual(salida.name, "artifact.mp4")

    def test_codigo_de_error_es_EngineError(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = self._runner(["-c", _FAIL, "{output}"], output_dir=tmp)
            with self.assertRaises(EngineError) as ctx:
                runner.invoke({})
            self.assertIn("boom", str(ctx.exception))

    def test_timeout_es_EngineError(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = self._runner(["-c", _SLEEP, "{output}"], output_dir=tmp)
            with self.assertRaises(EngineError):
                runner.invoke({}, timeout=1)

    def test_ausente_el_artefacto_es_EngineError(self):
        # El script no crea el fichero de salida declarado en {output}.
        with tempfile.TemporaryDirectory() as tmp:
            runner = self._runner(["-c", "pass", "{output}"], output_dir=tmp)
            with self.assertRaises(EngineError):
                runner.invoke({})


class CliVideoEngineTest(unittest.TestCase):
    def _engine(self, tmp, **kwargs) -> CliVideoEngine:
        return CliVideoEngine(
            engine_id="bkingfilm",
            nombre="BkingFilm",
            executable=sys.executable,
            args=["-c", _WRITE_TO_PATH, "{output}"],
            output_dir=tmp,
            output_ext=".mp4",
            **kwargs,
        )

    def test_identidad_y_modelos(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self._engine(tmp, models=["m1", "m2"])
            self.assertEqual(engine.id, "bkingfilm")
            self.assertEqual(engine.nombre, "BkingFilm")
            self.assertEqual(engine.list_models(), ["m1", "m2"])

    def test_disponibilidad(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(self._engine(tmp).is_available())
            ausente = CliVideoEngine(
                engine_id="lapian_notes",
                nombre="Lapian Notes",
                executable="no-existe-xyz-123",
                args=["{output}"],
            )
            self.assertFalse(ausente.is_available())

    def test_generate_video_devuelve_artefacto(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self._engine(tmp)
            salida = engine.generate_video("un viajero", input_path="pelicula.mp4")
            self.assertTrue(salida.is_file())
            self.assertEqual(salida.parent, Path(tmp))


if __name__ == "__main__":
    unittest.main()
