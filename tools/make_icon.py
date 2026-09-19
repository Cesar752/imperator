"""Regenera assets/icono.ico a partir del logo fuente.

Uso:
    python tools/make_icon.py

Aplica un recorte central (cuadrado) sobre el logo apaisado y genera un
``.ico`` multi-resolución, apto para la barra de tareas de Windows.
"""

from pathlib import Path

from PIL import Image

# Ruta del logo fuente (RGB 1536x1024, fondo sólido).
SRC = r"C:\Users\Cesar Esti\Downloads\icono.png"

# Destino dentro del proyecto (C:\Imperator\assets\icono.ico).
DST = Path(__file__).resolve().parent.parent / "assets" / "icono.ico"

# Resoluciones que Windows necesita en la barra de tareas.
SIZES = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]


def _cuadrado_central(img: Image.Image) -> Image.Image:
    """Recorta el centro de la imagen para obtener un cuadrado."""
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def main() -> int:
    img = Image.open(SRC).convert("RGBA")
    img = _cuadrado_central(img)
    DST.parent.mkdir(parents=True, exist_ok=True)
    img.save(DST, sizes=SIZES)
    print(f"OK: {DST} -> {len(SIZES)} resoluciones {SIZES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
