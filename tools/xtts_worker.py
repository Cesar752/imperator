"""Worker de clonado de voz XTTS-v2 (Coqui).

Se ejecuta **dentro del venv de coqui-tts** (Python 3.11), no en el intérprete de
Imperator: así el motor ``voice_clone`` funciona aunque la app use Python 3.12.

Uso (lo lanza ``engines/voice_clone.py`` por subprocess):

    python xtts_worker.py --text "hola" --speaker ref.wav --out salida.wav \
        --language es --model tts_models/multilingual/multi-dataset/xtts_v2 --device cuda

Imprime en la **última línea** un JSON: ``{"ok": true, "out": <ruta>}`` o
``{"ok": false, "error": <motivo>}``. El motor lee ese JSON para decidir.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clonado de voz XTTS-v2")
    parser.add_argument("--text", required=True)
    parser.add_argument("--speaker", required=True, help="WAV de referencia")
    parser.add_argument("--out", required=True, help="WAV de salida")
    parser.add_argument("--language", default="es")
    parser.add_argument(
        "--model", default="tts_models/multilingual/multi-dataset/xtts_v2"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--speed", type=float, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # La licencia de pesos de Coqui se acepta aquí (una vez) para que la descarga
    # no se quede pidiendo confirmación por consola.
    os.environ.setdefault("COQUI_TOS_AGREED", "1")

    try:
        from TTS.api import TTS
    except ImportError as exc:  # pragma: no cover - depende del venv del usuario
        print(json.dumps({"ok": False, "error": f"coqui-tts no instalado: {exc}"}))
        return 2

    try:
        model = TTS(args.model, progress_bar=False)
        if hasattr(model, "to"):
            model = model.to(args.device)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        kwargs: dict = {
            "text": args.text,
            "speaker_wav": args.speaker,
            "language": args.language,
            "file_path": str(out),
        }
        if args.speed is not None:
            kwargs["speed"] = args.speed
        model.tts_to_file(**kwargs)
    except Exception as exc:  # noqa: BLE001 - torch/OOM/descarga/IO
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 1

    if not out.is_file() or out.stat().st_size == 0:
        print(json.dumps({"ok": False, "error": "no se generó el audio de salida"}))
        return 1
    print(json.dumps({"ok": True, "out": str(out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
