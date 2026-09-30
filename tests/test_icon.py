"""Tests del icono de la barra de tareas (``assets/icono.ico``).

Regresión del fallo "la barra de tareas muestra el icono en blanco": Pillow
escribe *todas* las entradas del .ico como PNG y el shell de Windows solo admite
PNG en la entrada de 256×256. Con las entradas pequeñas en PNG, Qt sí pinta el
icono en la ventana, pero Windows lo deja en blanco en la barra de tareas.
"""

import struct
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.paths import ASSETS  # noqa: E402
from tools.make_icon import (  # noqa: E402
    PNG_SIZES,
    SIZES,
    build_ico,
    inspect_ico,
)

try:
    from PIL import Image
except ImportError:  # Pillow solo hace falta para regenerar el icono
    Image = None

ICONO = ASSETS / "icono.ico"
FUENTE = ASSETS / "icono_fuente.png"

# Cabecera de un DIB: tamaño, ancho, alto doblado, planos, bpp y compresión.
_BITMAPINFOHEADER = 40


def _leer_dib(data: bytes) -> dict:
    """Decodifica una entrada DIB de 32 bpp (cabecera + píxeles + máscara AND)."""
    size, width, height2, planes, bpp, compression = struct.unpack("<IiiHHI", data[:20])
    height = height2 // 2
    filas = width * 4 * height
    mask_row = ((width + 31) // 32) * 4
    return {
        "size": size,
        "width": width,
        "height": height,
        "planes": planes,
        "bpp": bpp,
        "compression": compression,
        "pixels": data[_BITMAPINFOHEADER : _BITMAPINFOHEADER + filas],
        "mask": data[_BITMAPINFOHEADER + filas :],
        "mask_esperada": mask_row * height,
    }


class IconoTest(unittest.TestCase):
    """El .ico versionado debe ser apto para la barra de tareas de Windows."""

    def setUp(self) -> None:
        if not ICONO.is_file():
            self.skipTest(f"falta el icono: {ICONO}")
        self.entries = inspect_ico(ICONO)

    def test_cubre_las_resoluciones_de_windows(self) -> None:
        self.assertEqual([entry["size"] for entry in self.entries], SIZES)

    def test_solo_el_256_va_comprimido_en_png(self) -> None:
        for entry in self.entries:
            esperado = "png" if entry["size"] in PNG_SIZES else "dib"
            with self.subTest(size=entry["size"]):
                self.assertEqual(
                    entry["format"],
                    esperado,
                    "las entradas pequeñas en PNG dejan la barra de tareas en blanco",
                )

    def test_entradas_dib_son_32_bpp_sin_comprimir(self) -> None:
        data = ICONO.read_bytes()
        for entry in self.entries:
            if entry["format"] != "dib":
                continue
            dib = _leer_dib(data[entry["offset"] : entry["offset"] + entry["bytes"]])
            with self.subTest(size=entry["size"]):
                self.assertEqual(dib["size"], _BITMAPINFOHEADER)
                self.assertEqual((dib["width"], dib["height"]), entry["size"])
                self.assertEqual(dib["planes"], 1)
                self.assertEqual(dib["bpp"], 32)
                self.assertEqual(dib["compression"], 0)  # BI_RGB
                # El tamaño declarado cubre cabecera + píxeles + máscara AND.
                self.assertEqual(
                    entry["bytes"],
                    _BITMAPINFOHEADER + len(dib["pixels"]) + len(dib["mask"]),
                )
                self.assertEqual(len(dib["mask"]), dib["mask_esperada"])

    def test_falta_la_fuente_del_icono(self) -> None:
        self.assertTrue(FUENTE.is_file(), f"falta el arte del icono: {FUENTE}")


@unittest.skipIf(Image is None, "Pillow solo hace falta para regenerar el icono")
class GeneradorTest(unittest.TestCase):
    """``build_ico`` debe producir DIB (no PNG) en los tamaños pequeños."""

    def setUp(self) -> None:
        if not FUENTE.is_file():
            self.skipTest(f"falta el arte del icono: {FUENTE}")
        assert Image is not None  # garantizado por skipIf
        self.src = Image.open(FUENTE).convert("RGBA")

    def test_genera_dib_en_los_tamanos_pequenos(self) -> None:
        entries = inspect_ico(build_ico(self.src))
        self.assertEqual([entry["size"] for entry in entries], SIZES)
        for entry in entries:
            esperado = "png" if entry["size"] in PNG_SIZES else "dib"
            with self.subTest(size=entry["size"]):
                self.assertEqual(entry["format"], esperado)

    def test_los_pixeles_del_dib_coinciden_con_la_fuente(self) -> None:
        # Ida y vuelta: la entrada de 48x48 se decodifica igual que la fuente
        # reescalada (comprueba que el volcado BGRA de abajo arriba es correcto).
        assert Image is not None  # garantizado por skipIf
        data = build_ico(self.src)
        entry = next(e for e in inspect_ico(data) if e["size"] == (48, 48))
        dib = _leer_dib(data[entry["offset"] : entry["offset"] + entry["bytes"]])

        esperado = self.src.resize((48, 48), Image.Resampling.LANCZOS).tobytes(
            "raw", "BGRA"
        )
        row = 48 * 4
        filas_esperadas = [
            esperado[y * row : (y + 1) * row] for y in range(48 - 1, -1, -1)
        ]
        self.assertEqual(dib["pixels"], b"".join(filas_esperadas))
        self.assertEqual(set(dib["mask"]), {0})  # el alfa manda: máscara en cero

    def test_la_mascara_alinea_a_cuatro_bytes(self) -> None:
        # 17 px de ancho: (17 + 31) // 32 * 4 = 4 bytes por fila de máscara.
        dib = _leer_dib(_entrada_dib(bytes(17 * 4 * 1), 17, 1))
        self.assertEqual(len(dib["mask"]), 4)


def _entrada_dib(pixels: bytes, width: int, height: int) -> bytes:
    """Envuelve píxeles sueltos en un DIB para probar el decodificador."""
    mask_row = ((width + 31) // 32) * 4
    return (
        struct.pack("<IiiHHIIiiII", 40, width, height * 2, 1, 32, 0, len(pixels), 0, 0, 0, 0)
        + pixels
        + b"\x00" * (mask_row * height)
    )


if __name__ == "__main__":
    unittest.main()
