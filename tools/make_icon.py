"""Regenera assets/icono.ico a partir de un logo fuente.

Uso:
    python tools/make_icon.py "C:\\ruta\\a\\logo.png"

Requiere Pillow (no es dependencia de runtime): ``pip install Pillow>=10``.
Aplica un recorte central (cuadrado) sobre el logo apaisado y genera un
``.ico`` multi-resolución, apto para la barra de tareas de Windows.
"""

import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:  # pragma: no cover - utilidad opcional
    print(
        "Falta Pillow. Instálalo con: pip install Pillow>=10\n"
        "(solo hace falta para regenerar el icono, no para ejecutar la app)."
    )
    raise SystemExit(2)

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


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("Uso: python tools/make_icon.py <ruta_del_logo.png>")
        return 1
    src = Path(argv[1])
    if not src.is_file():
        print(f"No existe el logo fuente: {src}")
        return 1

    img = Image.open(src).convert("RGBA")
    img = _cuadrado_central(img)
    DST.parent.mkdir(parents=True, exist_ok=True)
    img.save(DST, sizes=SIZES)
    print(f"OK: {DST} -> {len(SIZES)} resoluciones {SIZES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
