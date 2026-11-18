"""Worker de mejora de fotos con IA (Real-ESRGAN + GFPGAN).

Se ejecuta **dentro de `venv-photo`** (torch CPU/GPU + realesrgan + gfpgan), no en
el intérprete de Imperator: así el motor no arrastra esas dependencias.

Uso (lo lanza ``ui/photo_widget.py`` por subprocess):

    python photo_worker.py --input foto.png --output salida.png \
        --scale 4 [--face] [--model-dir tools/models] [--tile 0]

Imprime en la última línea un JSON: ``{"ok": true, "out": <ruta>}`` o
``{"ok": false, "error": <motivo>}``. La primera vez descarga los pesos
(Real-ESRGAN ~64 MB; GFPGAN ~330 MB con --face) a ``--model-dir``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REALESRGAN_X4_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
)
_GFPGAN_V14_URL = (
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Mejora de fotos (Real-ESRGAN/GFPGAN)")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--scale", type=float, default=4)
    p.add_argument("--face", action="store_true", help="restaurar caras con GFPGAN")
    p.add_argument("--model-dir", default="tools/models")
    p.add_argument("--tile", type=int, default=0, help="0 = sin teselado (usa más RAM)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        import cv2
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from basicsr.utils.download_util import load_file_from_url
        from realesrgan import RealESRGANer
    except Exception as exc:  # noqa: BLE001 - depende del venv del usuario
        print(json.dumps({"ok": False, "error": f"dependencias no disponibles: {exc}"}))
        return 2

    try:
        modelo = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4
        )
        pesos = load_file_from_url(
            url=_REALESRGAN_X4_URL, model_dir=str(model_dir), progress=False, file_name=None
        )
        upsampler = RealESRGANer(
            scale=4, model_path=pesos, model=modelo,
            tile=int(args.tile), tile_pad=10, pre_pad=0, half=False,
        )

        imagen = cv2.imread(args.input, cv2.IMREAD_COLOR)
        if imagen is None:
            raise ValueError(f"no se pudo leer la imagen: {args.input}")

        salida = upsampler.enhance(imagen, outscale=float(args.scale))[0]

        if args.face:
            from gfpgan import GFPGANer

            pesos_cara = load_file_from_url(
                url=_GFPGAN_V14_URL, model_dir=str(model_dir), progress=False, file_name=None
            )
            restaurador = GFPGANer(
                model_path=pesos_cara, upscale=2, arch="clean",
                channel_multiplier=2, bg_upsampler=upsampler,
            )
            _, _, salida = restaurador.enhance(
                salida, has_aligned=False, only_center_face=False, paste_back=True
            )

        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(args.output, salida):
            raise ValueError("no se pudo escribir la imagen de salida")
    except Exception as exc:  # noqa: BLE001 - torch/CUDA/IO/descarga
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 1

    print(json.dumps({"ok": True, "out": args.output}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
