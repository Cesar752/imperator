"""Tests de la caché de fotogramas (core/frame_cache.py) y de su uso en el extractor."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.frame_cache import FrameCache  # noqa: E402
from core import frame_extractor as fe  # noqa: E402
from engines.base_engine import EngineError  # noqa: E402


def _fake_video(tmp: Path, name: str = "peli.mp4", size: int = 1024) -> Path:
    video = tmp / name
    video.write_bytes(b"x" * size)
    return video


class FrameCacheTest(unittest.TestCase):
    def test_layout_y_estabilidad(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = _fake_video(root)
            cache = FrameCache(root / "cache")
            p1 = cache.path_for(video, 1.0)
            p2 = cache.path_for(video, 1.0)
            self.assertEqual(p1, p2)
            self.assertEqual(p1.parent, cache.dir_for(video))
            # El nombre codifica milisegundos.
            self.assertEqual(p1.name, "t0000001000.png")
            self.assertEqual(cache.video_key(video), cache.video_key(video))

    def test_clave_cambia_si_cambia_el_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = _fake_video(root, size=1024)
            cache = FrameCache(root / "cache")
            k1 = cache.video_key(video)
            video.write_bytes(b"y" * 2048)  # cambia el tamaño
            k2 = cache.video_key(video)
            self.assertNotEqual(k1, k2)

    def test_get_miss_e_hit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = _fake_video(root)
            cache = FrameCache(root / "cache")
            self.assertIsNone(cache.get(video, 2.0))
            cache.ensure_dir(video)
            destino = cache.path_for(video, 2.0)
            destino.write_bytes(b"PNG")
            self.assertEqual(cache.get(video, 2.0), destino)
            # Un fichero vacío no cuenta como acierto.
            destino.write_bytes(b"")
            self.assertIsNone(cache.get(video, 2.0))

    def test_prune_borra_los_mas_antiguos(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = FrameCache(root / "cache")
            video = _fake_video(root)
            cache.ensure_dir(video)
            for i in range(5):
                p = cache.path_for(video, float(i))
                p.write_bytes(b"PNG")
            borrados = cache.prune(max_files=3)
            self.assertEqual(borrados, 2)
            self.assertEqual(cache.stats()["files"], 3)
            # Por debajo del tope no borra nada.
            self.assertEqual(cache.prune(max_files=10), 0)


class CachedFrameTest(unittest.TestCase):
    def test_reutiliza_sin_re_extraer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = _fake_video(root)
            extractor = fe.FrameExtractor(cache=FrameCache(root / "cache"))
            llamadas = []

            def falso_extract(_video, timestamp, output_path):
                llamadas.append(timestamp)
                destino = Path(output_path)
                destino.parent.mkdir(parents=True, exist_ok=True)
                destino.write_bytes(b"PNG")
                return destino

            with mock.patch.object(extractor, "extract_frame", side_effect=falso_extract):
                primera = extractor.cached_frame(video, 3.0)
                segunda = extractor.cached_frame(video, 3.0)

            self.assertEqual(primera, segunda)
            self.assertEqual(len(llamadas), 1)
            self.assertTrue(primera.is_file())

    def test_sin_cache_avisa(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = _fake_video(Path(tmp))
            with self.assertRaises(EngineError):
                fe.FrameExtractor().cached_frame(video, 1.0)


if __name__ == "__main__":
    unittest.main()
