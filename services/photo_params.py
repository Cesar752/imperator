"""Retoque de fotos: ajustes de color y conversión de formato (lógica pura).

Sin Qt. Los ajustes clásicos (brillo, contraste, color y nitidez) se aplican con
Pillow —funcionan siempre, sin instalar nada— y la conversión de **cualquier
formato** de entrada/salida (PNG, JPEG, WEBP, TIFF, BMP…) también. La mejora con
IA (Real-ESRGAN/GFPGAN) vive aparte, en el worker por subprocess.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps

#: Formatos de salida ofrecidos en la UI.
OUTPUT_FORMATS: tuple[str, ...] = ("PNG", "JPEG", "WEBP", "TIFF", "BMP")
DEFAULT_FORMAT = "PNG"

#: Extensiones de entrada admitidas (Pillow lee muchas más; estas son las típicas).
INPUT_SUFFIXES: tuple[str, ...] = (
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff",
    ".gif", ".heic", ".heif", ".avif", ".ico", ".ppm", ".pgm",
)

#: Límites de los deslizadores de ajuste (1.0 = sin cambio).
ADJUST_MIN = 0.0
ADJUST_MAX = 3.0
ADJUST_DEFAULT = 1.0

#: Escalas de la mejora con IA.
SCALES: tuple[int, ...] = (2, 3, 4)
DEFAULT_SCALE = 4


@dataclass(frozen=True)
class Adjustments:
    """Ajustes clásicos de color/nitidez (1.0 = sin cambio)."""

    brightness: float = ADJUST_DEFAULT
    contrast: float = ADJUST_DEFAULT
    color: float = ADJUST_DEFAULT
    sharpness: float = ADJUST_DEFAULT

    @property
    def is_identity(self) -> bool:
        return all(
            abs(v - ADJUST_DEFAULT) < 1e-6
            for v in (self.brightness, self.contrast, self.color, self.sharpness)
        )

    def apply(self, image: Image.Image) -> Image.Image:
        """Devuelve una copia con los ajustes aplicados (no muta la original)."""
        resultado = image
        if abs(self.brightness - ADJUST_DEFAULT) > 1e-6:
            resultado = ImageEnhance.Brightness(resultado).enhance(self.brightness)
        if abs(self.contrast - ADJUST_DEFAULT) > 1e-6:
            resultado = ImageEnhance.Contrast(resultado).enhance(self.contrast)
        if abs(self.color - ADJUST_DEFAULT) > 1e-6:
            resultado = ImageEnhance.Color(resultado).enhance(self.color)
        if abs(self.sharpness - ADJUST_DEFAULT) > 1e-6:
            resultado = ImageEnhance.Sharpness(resultado).enhance(self.sharpness)
        return resultado


def clamp_adjust(value) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ADJUST_DEFAULT
    return min(max(v, ADJUST_MIN), ADJUST_MAX)


def load_image(path: str | Path) -> Image.Image:
    """Abre una imagen en **cualquier formato** y la normaliza a RGB/RGBA."""
    img = Image.open(path)
    img.load()
    # Respeta la orientación EXIF (fotos de móvil).
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:  # noqa: BLE001 - EXIF corrupto no debe romper
        pass
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        return img.convert("RGBA")
    return img.convert("RGB")


def save_image(
    image: Image.Image, path: str | Path, fmt: str = DEFAULT_FORMAT, quality: int = 95
) -> Path:
    """Guarda ``image`` en ``path`` con el formato pedido (convierte si hace falta)."""
    destino = Path(path)
    destino.parent.mkdir(parents=True, exist_ok=True)
    fmt = (fmt or DEFAULT_FORMAT).upper()
    img = image
    if fmt in ("JPEG", "BMP") and img.mode == "RGBA":
        # Sin canal alfa: se aplana sobre blanco.
        fondo = Image.new("RGB", img.size, (255, 255, 255))
        fondo.paste(img, mask=img.split()[-1])
        img = fondo
    kwargs: dict = {}
    if fmt == "JPEG":
        kwargs["quality"] = int(quality)
    elif fmt == "WEBP":
        kwargs["quality"] = int(quality)
    img.save(destino, format=fmt, **kwargs)
    return destino


def convert(
    src: str | Path,
    dst: str | Path,
    fmt: str = DEFAULT_FORMAT,
    *,
    quality: int = 95,
    adjustments: Adjustments | None = None,
) -> Path:
    """Convierte ``src`` (cualquier formato) a ``dst`` con los ajustes aplicados."""
    img = load_image(src)
    if adjustments is not None and not adjustments.is_identity:
        img = adjustments.apply(img)
    return save_image(img, dst, fmt, quality)
