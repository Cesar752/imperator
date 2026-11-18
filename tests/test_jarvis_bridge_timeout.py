"""jarvis_bridge: un ffmpeg colgado no puede bloquear a Jarvis.

Cinco llamadas a ffmpeg no tenían timeout; un proceso colgado dejaba el puente
esperando para siempre.
"""

import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import jarvis_bridge as jb  # noqa: E402


class TimeoutTest(unittest.TestCase):
    def test_el_helper_devuelve_none_al_agotarse(self) -> None:
        agotado = subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=1)
        with mock.patch.object(jb.subprocess, "run", side_effect=agotado):
            with self.assertLogs("tools.jarvis_bridge", level="ERROR"):
                self.assertIsNone(jb._run_ffmpeg(["ffmpeg"], 1, "prueba"))

    def test_el_helper_pasa_el_timeout_a_subprocess(self) -> None:
        falso = mock.Mock(returncode=0, stdout="", stderr="")
        with mock.patch.object(jb.subprocess, "run", return_value=falso) as ejecutado:
            jb._run_ffmpeg(["ffmpeg", "-i", "x"], 7, "prueba")

        self.assertEqual(ejecutado.call_args.kwargs["timeout"], 7)

    def test_la_sonda_de_fps_tolera_el_timeout(self) -> None:
        with mock.patch.object(jb, "_run_ffmpeg", return_value=None):
            self.assertEqual(jb._probe_fps(Path("v.mp4")), 30.0)

    def test_la_sonda_de_stream_tolera_el_timeout(self) -> None:
        with mock.patch.object(jb, "_run_ffmpeg", return_value=None):
            self.assertEqual(jb._probe_stream(Path("v.mp4")), {})

    def test_enhance_devuelve_error_claro_si_se_cuelga(self) -> None:
        with mock.patch.object(jb, "_ffmpeg_exe", return_value="ffmpeg"), mock.patch.object(
            jb, "_run_ffmpeg", return_value=None
        ):
            resultado = jb.run_enhance(Path("entrada.mp4"), Path("salida.mp4"), {})

        self.assertEqual(resultado["exit_code"], -1)
        self.assertIsNone(resultado["output"])
        self.assertIn("tiempo", resultado["log"])

    def test_upscale_devuelve_error_claro_si_se_cuelga(self) -> None:
        with mock.patch.object(jb, "_ffmpeg_exe", return_value="ffmpeg"), mock.patch.object(
            jb, "_probe_stream", return_value={"duration": 1.0}
        ), mock.patch.object(jb, "_probe_fps", return_value=30.0), mock.patch.object(
            jb, "_run_ffmpeg", return_value=None
        ):
            resultado = jb.run_upscale(Path("entrada.mp4"), Path("salida.mp4"), {})

        self.assertEqual(resultado["exit_code"], -1)
        self.assertIsNone(resultado["output"])


if __name__ == "__main__":
    unittest.main()
