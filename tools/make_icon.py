"""Regenera assets/icono.ico a partir del arte cuadrado del icono.

Uso:
    python tools/make_icon.py                    # usa assets/icono_fuente.png
    python tools/make_icon.py "C:\\ruta\\logo.png"

Requiere Pillow (no es dependencia de runtime): ``pip install Pillow>=10``.
Aplica un recorte central (cuadrado) —por si la fuente es apaisada, como
``assets/imperator.png``— y genera un ``.ico`` multi-resolución, apto para la
barra de tareas de Windows. La fuente por defecto es ``icono_fuente.png`` (el
arte cuadrado del icono); ``imperator.png`` es el logo apaisado del splash.

**Por qué no se usa ``Image.save(dst, sizes=...)``:** Pillow escribe *todas* las
entradas del .ico como PNG, y el shell de Windows solo admite PNG en la entrada
de 256×256 («Vista icons»). Con los tamaños pequeños en PNG, la barra de tareas
pinta el icono en blanco (Qt sí lo lee, por eso el icono se veía en la ventana
pero no en la barra). Aquí se escriben a mano: DIB sin comprimir (BGRA + máscara
AND) para 16–128 y PNG solo para 256.
"""

import struct
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # solo para las anotaciones; Pillow es opcional en runtime
    from PIL import Image as PILImage

try:
    from PIL import Image
except ImportError:  # pragma: no cover - dependencia opcional
    # No se aborta al importar: el módulo también lo leen los tests (y el CI no
    # instala Pillow). Solo hace falta para regenerar el icono.
    Image = None  # type: ignore[assignment]

_FALTA_PILLOW = (
    "Falta Pillow. Instálalo con: pip install Pillow>=10\n"
    "(solo hace falta para regenerar el icono, no para ejecutar la app)."
)

# Destino dentro del proyecto (C:\Imperator\assets\icono.ico).
DST = Path(__file__).resolve().parent.parent / "assets" / "icono.ico"

# Arte por defecto si no se pasa ruta (cuadrado; el logo del splash es apaisado).
DEFAULT_SRC = Path(__file__).resolve().parent.parent / "assets" / "icono_fuente.png"

# Resoluciones que Windows necesita en la barra de tareas.
SIZES = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]

# Único tamaño que Windows acepta comprimido en PNG dentro de un .ico.
PNG_SIZES = {(256, 256)}

# BITMAPINFOHEADER: tamaño, ancho, alto (doblado), planos y bits por píxel.
_BITMAPINFOHEADER = 40

# Firmas para reconocer el formato de cada entrada al inspeccionar el .ico.
_BMP_MAGIC = struct.pack("<I", _BITMAPINFOHEADER)
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _cuadrado_central(img: "PILImage.Image") -> "PILImage.Image":
    """Recorta el centro de la imagen para obtener un cuadrado."""
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def _a_dib(img: "PILImage.Image") -> bytes:
    """Codifica la imagen como DIB de 32 bpp (BITMAPINFOHEADER + píxeles + AND).

    El formato que espera el shell de Windows para las entradas menores de
    256×256: cabecera de 40 bytes, píxeles BGRA de abajo arriba y máscara AND
    de 1 bpp con cada fila alineada a 4 bytes (a cero: el alfa manda).
    """
    width, height = img.size
    header = struct.pack(
        "<IiiHHIIiiII",
        _BITMAPINFOHEADER,
        width,
        height * 2,  # biHeight doblado: XOR (color) + AND (máscara)
        1,  # biPlanes
        32,  # biBitCount
        0,  # biCompression = BI_RGB
        width * height * 4,  # biSizeImage
        0,  # biXPelsPerMeter
        0,  # biYPelsPerMeter
        0,  # biClrUsed
        0,  # biClrImportant
    )

    raw = img.tobytes("raw", "BGRA")  # de arriba abajo
    row_bytes = width * 4
    pixels = b"".join(
        raw[y * row_bytes : (y + 1) * row_bytes] for y in range(height - 1, -1, -1)
    )

    mask_row_bytes = ((width + 31) // 32) * 4  # 1 bpp alineado a 4 bytes
    mask = b"\x00" * (mask_row_bytes * height)
    return header + pixels + mask


def _a_png(img: "PILImage.Image") -> bytes:
    """Codifica la imagen como PNG (solo para la entrada de 256×256)."""
    import io

    buffer = io.BytesIO()
    img.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def build_ico(
    img: "PILImage.Image", sizes: list[tuple[int, int]] | None = None
) -> bytes:
    """Devuelve los bytes de un ``.ico`` con las entradas en el formato correcto."""
    if Image is None:  # pragma: no cover - entorno sin Pillow
        raise RuntimeError(_FALTA_PILLOW)
    sizes = sizes or SIZES
    payloads: list[tuple[int, int, bytes]] = []
    for width, height in sizes:
        frame = _cuadrado_central(img).resize(
            (width, height), Image.Resampling.LANCZOS
        )
        data = (
            _a_png(frame) if (width, height) in PNG_SIZES else _a_dib(frame)
        )
        payloads.append((width, height, data))

    header = struct.pack("<HHH", 0, 1, len(payloads))  # reservado, tipo=icono, nº
    offset = len(header) + 16 * len(payloads)
    entries = b""
    for width, height, data in payloads:
        entries += struct.pack(
            "<BBBBHHII",
            width if width < 256 else 0,  # 256 se codifica como 0
            height if height < 256 else 0,
            0,  # colores de paleta
            0,  # reservado
            1,  # planos
            32,  # bits por píxel
            len(data),
            offset,
        )
        offset += len(data)
    return header + entries + b"".join(data for _, _, data in payloads)


def inspect_ico(source: "Path | str | bytes") -> list[dict]:
    """Lee el directorio del ``.ico`` y describe el formato de cada entrada.

    ``source`` puede ser una ruta o los propios bytes (cómodo para los tests).
    Devuelve una lista de ``{"size", "bpp", "bytes", "format", "offset"}``, donde
    ``format`` es ``"png"`` o ``"dib"``. Lo usan el propio script (para avisar si
    el fichero vuelve a quedar todo en PNG) y los tests de regresión.
    """
    data = source if isinstance(source, bytes) else Path(source).read_bytes()
    if len(data) < 6 or data[:4] != b"\x00\x00\x01\x00":
        raise ValueError("no es un fichero .ico válido")
    count = struct.unpack("<H", data[4:6])[0]
    entries: list[dict] = []
    for i in range(count):
        base = 6 + 16 * i
        width, height = data[base], data[base + 1]
        bpp = struct.unpack("<H", data[base + 6 : base + 8])[0]
        size, offset = struct.unpack("<II", data[base + 8 : base + 16])
        magic = data[offset : offset + 8]
        if magic.startswith(_PNG_MAGIC):
            fmt = "png"
        elif magic.startswith(_BMP_MAGIC):
            fmt = "dib"
        else:
            fmt = "desconocido"
        entries.append(
            {
                "size": (width or 256, height or 256),
                "bpp": bpp,
                "bytes": size,
                "format": fmt,
                "offset": offset,
            }
        )
    return entries


def _avisar_formatos(path: Path) -> bool:
    """Imprime el formato de cada entrada y avisa si alguna rompe la barra."""
    wrong = []
    for entry in inspect_ico(path):
        width, height = entry["size"]
        print(f"  {width:>3}x{height:<3} {entry['bpp']:>2} bpp {entry['format']:>4}")
        esperado = "png" if (width, height) in PNG_SIZES else "dib"
        if entry["format"] != esperado:
            wrong.append((width, height, entry["format"]))
    if wrong:
        print(
            "AVISO: estas entradas deberían ir en DIB (Windows solo admite PNG en "
            f"256x256) y quedaron en PNG: {wrong}.\n"
            "Con el icono así, la barra de tareas lo muestra en blanco."
        )
        return False
    return True


def main(argv: list[str]) -> int:
    if Image is None:
        print(_FALTA_PILLOW)
        return 2
    if len(argv) > 2:
        print("Uso: python tools/make_icon.py [ruta_del_logo.png]")
        return 1
    src = Path(argv[1]) if len(argv) == 2 else DEFAULT_SRC
    if not src.is_file():
        print(f"No existe el logo fuente: {src}")
        return 1

    img = Image.open(src).convert("RGBA")
    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_bytes(build_ico(img))
    print(f"OK: {DST} -> {len(SIZES)} resoluciones {SIZES} (fuente: {src.name})")
    return 0 if _avisar_formatos(DST) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
