"""Worker de escalado de vídeo con IA: reescala una carpeta de fotogramas.

Se ejecuta en ``venv-photo`` (Real-ESRGAN). Carga el modelo **una sola vez** y
recorre los fotogramas en orden; así no se paga la carga del modelo por cada uno.

Uso (lo lanza ``ui/video_enhance_widget.py``):

    python video_upscale_worker.py --frames-in dir_in --frames-out dir_out \
        --scale 2 --model-dir tools/models

Imprime una línea JSON por fotograma (``{"progress": i, "total": n}``) y, al
final, ``{"ok": true, "count": n}`` (o ``{"ok": false, "error": ...}``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REALESRGAN_X4_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Escalado IA de fotogramas (Real-ESRGAN)")
    p.add_argument("--frames-in", required=True)
    p.add_argument("--frames-out", required=True)
    p.add_argument("--scale", type=float, default=2)
    p.add_argument("--model-dir", default="tools/models")
    p.add_argument("--tile", type=int, default=0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    entrada = Path(args.frames_in)
    salida = Path(args.frames_out)
    salida.mkdir(parents=True, exist_ok=True)
    modelos = Path(args.model_dir)
    modelos.mkdir(parents=True, exist_ok=True)

    try:
        import cv2
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from basicsr.utils.download_util import load_file_from_url
        from realesrgan import RealESRGANer
    except Exception as exc:  # noqa: BLE001 - depende del venv del usuario
        print(json.dumps({"ok": False, "error": f"dependencias no disponibles: {exc}"}))
        return 2

    fotogramas = sorted(entrada.glob("*.png"))
    if not fotogramas:
        print(json.dumps({"ok": False, "error": "no hay fotogramas de entrada"}))
        return 1

    try:
        modelo = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4
        )
        pesos = load_file_from_url(
            url=_REALESRGAN_X4_URL, model_dir=str(modelos), progress=False, file_name=None
        )
        upsampler = RealESRGANer(
            scale=4, model_path=pesos, model=modelo,
            tile=int(args.tile), tile_pad=10, pre_pad=0, half=False,
        )
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 1

    total = len(fotogramas)
    for i, ruta in enumerate(fotogramas, start=1):
        try:
            imagen = cv2.imread(str(ruta), cv2.IMREAD_COLOR)
            if imagen is None:
                raise ValueError(f"no se pudo leer {ruta.name}")
            resultado = upsampler.enhance(imagen, outscale=float(args.scale))[0]
            cv2.imwrite(str(salida / ruta.name), resultado)
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"ok": False, "error": f"{ruta.name}: {exc}"}))
            return 1
        print(json.dumps({"progress": i, "total": total}), flush=True)

    print(json.dumps({"ok": True, "count": total}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
