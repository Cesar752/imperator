"""C5.4: la conversión a MP3 se hace en el pool, no en el hilo de UI.

La conversión es un ``subprocess.run`` que puede tardar minutos; el slot del
botón debe volver en milisegundos, deshabilitar el botón y avisar al terminar.
(No hay debounce: entre pulsación y pulsación hay un ``QFileDialog`` modal, así
que no existen «N pulsaciones rápidas».)
"""

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from core.config_manager import ConfigManager  # noqa: E402
from ui import audio_widget as aw  # noqa: E402

#: Criterio de la ronda: el slot debe volver en <100 ms.
LIMITE_MS = 100


class Mp3Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.cm = ConfigManager(config_dir=self.dir)
        self.wav = self.dir / "voz.wav"
        self.wav.write_bytes(b"RIFF falso")
        self.mp3 = self.dir / "voz.mp3"
        with mock.patch.object(aw, "run_pool_task"):
            self.widget = aw.AudioWidget(None, None, self.cm)
        self.widget._last_path = str(self.wav)
        self.widget._save_button.setEnabled(True)

    def tearDown(self) -> None:
        self.widget.close()
        self._tmp.cleanup()

    def _guardar_como(self, destino: Path):
        with mock.patch.object(
            aw.QFileDialog, "getSaveFileName", return_value=(str(destino), "")
        ):
            return self.widget._save_as()

    def test_mp3_no_bloquea_y_deshabilita_el_boton(self) -> None:
        with mock.patch.object(aw, "run_pool_task") as pool:
            inicio = time.monotonic()
            self._guardar_como(self.mp3)
            transcurrido_ms = (time.monotonic() - inicio) * 1000

        self.assertLess(transcurrido_ms, LIMITE_MS)
        self.assertEqual(pool.call_count, 1)
        self.assertFalse(self.widget._save_button.isEnabled())
        self.assertIn("Convirtiendo", self.widget._status.text())
        # El destino viaja como contexto del receptor, no capturado en un cierre.
        self.assertIsNotNone(self.widget._mp3_rx)

    def test_wav_sigue_siendo_copia_directa(self) -> None:
        destino = self.dir / "copia.wav"
        with mock.patch.object(aw, "run_pool_task") as pool:
            self._guardar_como(destino)

        self.assertFalse(pool.called, "copiar un WAV es instantáneo: no va al pool")
        self.assertTrue(destino.is_file())
        self.assertIn("Guardado en", self.widget._status.text())

    def test_al_terminar_se_avisa_y_se_rehabilita(self) -> None:
        with mock.patch.object(aw, "run_pool_task"):
            self._guardar_como(self.mp3)

        self.widget._on_mp3_done(str(self.mp3), None)

        self.assertTrue(self.widget._save_button.isEnabled())
        self.assertIn(str(self.mp3), self.widget._status.text())

    def test_error_de_conversion_avisa_y_rehabilita(self) -> None:
        with mock.patch.object(aw, "run_pool_task"):
            self._guardar_como(self.mp3)

        with mock.patch.object(aw.QMessageBox, "critical") as critical:
            self.widget._on_mp3_error(str(self.mp3), "sin ffmpeg")

        self.assertTrue(self.widget._save_button.isEnabled())
        self.assertTrue(critical.called)


if __name__ == "__main__":
    unittest.main()
