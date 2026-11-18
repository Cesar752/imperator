"""Puente headless de Imperator para Jarvis.

Recibe una operación en JSON por stdin y ejecuta el pipeline de vídeo de
Imperator SIN abrir la GUI. Reutiliza ``services.video_enhance_params`` (lógica
pura de filtros ffmpeg) e ``imageio_ffmpeg`` (el binario). Devuelve JSON a stdout.

Operaciones:
- ``enhance``: limpieza (denoise), nitidez, color, interpolación de FPS,
  velocidad y desentrelazado opcional (yadif), en un solo pase de ffmpeg.

Uso:
  echo '{"input": "...", "operation": "enhance", "params": {...}}' \
    | python tools/jarvis_bridge.py
"""
import json
import logging
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# Permitir `from services import ...` ejecutando el script desde tools/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

import imageio_ffmpeg  # noqa: E402 - va tras fijar PROJECT_ROOT en sys.path

from core.paths import MODELS  # noqa: E402 - va tras fijar sys.path
from services import video_enhance_params as vp  # noqa: E402 - idem

OUTPUT_DIR = PROJECT_ROOT / "output"
#: Carpeta de grabaciones (fuera de ``output/``, que se limpia por antigüedad).
RECORDINGS_DIR = PROJECT_ROOT / "projects" / "grabaciones"

#: Extensiones de vídeo que se aceptan dentro de una carpeta.
VIDEO_EXTS = (
    ".mts", ".m2ts", ".m2t", ".mod", ".tod",
    ".mp4", ".mov", ".mkv", ".avi", ".m4v", ".ts", ".mpg", ".mpeg", ".wmv",
)
#: Timeout de la unión (copia de streams de una grabación larga).
JOIN_TIMEOUT = 2 * 3600
JOIN_NORMALIZE_TIMEOUT = 6 * 3600

#: Topes de los pasos que antes no tenían ninguno: un ffmpeg colgado dejaba a
#: Jarvis esperando para siempre (el ``excepthook`` solo cubre el hilo principal).
PROBE_TIMEOUT = 60
ENHANCE_TIMEOUT = 2 * 3600


# Reescalado IA (Real-ESRGAN en venv-photo).
VENV_PHOTO_PYTHON = PROJECT_ROOT / "venv-photo" / "Scripts" / "python.exe"
UPSCALE_WORKER = PROJECT_ROOT / "tools" / "video_upscale_worker.py"
MODELS_DIR = MODELS
UPSCALE_TIMEOUT = 4 * 3600  # 4 h: reescalar vídeo fotograma a fotograma es lento.


def _ffmpeg_exe() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def _run_ffmpeg(argv: list[str], timeout: float, etiqueta: str):
    """Ejecuta ffmpeg con tope de tiempo.

    Devuelve ``None`` si se agota: quien llama decide qué hacer (una sonda
    devuelve valores por defecto; un trabajo devuelve error claro).
    """
    try:
        return subprocess.run(
            argv,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        logger.error("jarvis_bridge: %s agotó el tiempo (%s s)", etiqueta, timeout)
        return None



def _videos_in(folder: Path) -> tuple[Path, ...]:
    """Vídeos de una carpeta, ordenados por nombre (00000.MTS, 00001.MTS…)."""
    candidatos = (
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )
    return vp.join_paths_ordered(candidatos)


def _probe_stream(path: Path) -> dict:
    """Duración, tamaño y fps leídos del stderr de ``ffmpeg -i``."""
    proc = _run_ffmpeg(
        [_ffmpeg_exe(), "-hide_banner", "-i", str(path)],
        PROBE_TIMEOUT,
        f"sonda de '{path.name}'",
    )
    if proc is None:
        return {}
    texto = proc.stderr or ""
    datos: dict = {}
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", texto)
    if m:
        datos["duration"] = (
            int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
        )
    m = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", texto)
    if m:
        datos["width"], datos["height"] = int(m.group(1)), int(m.group(2))
    m = re.search(r"(\d+(?:\.\d+)?)\s*fps", texto)
    if m:
        datos["fps"] = float(m.group(1))
    return datos


def _concat_list_text(paths) -> str:
    """Contenido del fichero de lista del demuxer ``concat`` (rutas escapadas)."""
    lineas = []
    for ruta in paths:
        escapada = str(ruta).replace("'", "'\\''")
        lineas.append(f"file '{escapada}'")
    return "\n".join(lineas) + "\n"


def _join_clips(paths, salida: Path) -> dict:
    """Une trozos en un MP4: primero copiando streams, si no igualando tamaño y fps."""
    ffmpeg = _ffmpeg_exe()
    salida.parent.mkdir(parents=True, exist_ok=True)
    info = [_probe_stream(p) for p in paths]
    duracion = sum(float(d.get("duration") or 0.0) for d in info) or None

    def _run(argv, timeout):
        return subprocess.run(
            argv, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=timeout,
        )

    with tempfile.TemporaryDirectory(prefix="jarvis_unir_") as tmp:
        lista = Path(tmp) / "lista.txt"
        lista.write_text(_concat_list_text(paths), encoding="utf-8")
        proc = _run(vp.join_command(ffmpeg, lista, salida), JOIN_TIMEOUT)
        if proc.returncode != 0 or not salida.is_file():
            # Los trozos no coinciden (audio o tamaño distinto): se igualan.
            salida.unlink(missing_ok=True)
            primero = info[0] if info else {}
            proc = _run(
                vp.join_normalize_command(
                    ffmpeg, paths, salida,
                    width=primero.get("width"), height=primero.get("height"),
                    fps=vp.join_target_fps([d.get("fps") for d in info]),
                ),
                JOIN_NORMALIZE_TIMEOUT,
            )

    if proc.returncode != 0 or not salida.is_file():
        raise RuntimeError(
            "No se pudieron unir las grabaciones. " + (proc.stderr or "").strip()[-600:]
        )
    return {"unido": str(salida), "clips": len(paths), "segundos": round(duracion or 0, 1)}


def _resolve_input(entrada: Path) -> tuple[Path, dict]:
    """Ruta de un vídeo listo para procesar, aceptando también una **carpeta**.

    Una cámara trocea una grabación en ``00000.MTS``, ``00001.MTS``…: si lo que
    llega es la carpeta, se unen los trozos (sin recodificar) y se devuelve ese
    MP4. La unión se reutiliza si ya existe y es más nueva que todos los trozos.
    """
    if entrada.is_file():
        return entrada, {}
    if not entrada.exists():
        raise FileNotFoundError(f"No existe la ruta: {entrada}")
    if not entrada.is_dir():
        raise ValueError(f"No es un vídeo ni una carpeta: {entrada}")

    clips = _videos_in(entrada)
    if not clips:
        raise ValueError(f"La carpeta no contiene vídeos compatibles: {entrada}")
    if len(clips) == 1:
        return clips[0], {"clips": 1}

    salida = RECORDINGS_DIR / f"{entrada.name}_unido.mp4"
    if salida.is_file() and salida.stat().st_size > 0:
        ultimo = max(c.stat().st_mtime for c in clips)
        if salida.stat().st_mtime >= ultimo:
            return salida, {"unido": str(salida), "clips": len(clips), "reutilizado": True}
    return salida, _join_clips(clips, salida)


def _build_filters(params: dict) -> str:
    """Cadena ``-vf`` a partir de los parámetros de limpieza/color/velocidad."""
    adjust = vp.ColorAdjust(
        brightness=float(params.get("brightness", vp.BRIGHTNESS_DEFAULT)),
        contrast=float(params.get("contrast", vp.CONTRAST_DEFAULT)),
        saturation=float(params.get("saturation", vp.SATURATION_DEFAULT)),
        gamma=float(params.get("gamma", vp.GAMMA_DEFAULT)),
        temperature=float(params.get("temperature", vp.TEMPERATURE_DEFAULT)),
    )
    fps = params.get("fps")
    return vp.build_filters(
        adjust=adjust,
        denoise=params.get("denoise", "off"),
        sharpen=float(params.get("sharpen", vp.SHARPEN_DEFAULT)),
        fps=int(fps) if fps else None,
        speed=float(params.get("speed", vp.SPEED_DEFAULT)),
    )


def run_enhance(entrada: Path, salida: Path, params: dict) -> dict:
    """Aplica filtros (y desentrelazado opcional) y recodifica el vídeo."""
    filters = _build_filters(params)
    deinterlace = bool(params.get("deinterlace", False))
    encoder = params.get("encoder", vp.ENCODER_CPU)
    crf = int(params.get("crf", vp.CRF_DEFAULT))

    ffmpeg = _ffmpeg_exe()
    argv = [ffmpeg, "-y", "-hide_banner", "-i", str(entrada)]
    if deinterlace:
        vf = "yadif" + (f",{filters}" if filters else "")
        argv += ["-vf", vf]
    elif filters:
        argv += ["-vf", filters]
    argv += vp.encoder_args(encoder, crf)
    argv += ["-c:a", "copy"]
    argv += ["-movflags", "+faststart", str(salida)]

    proc = _run_ffmpeg(argv, ENHANCE_TIMEOUT, "enhance")
    if proc is None:
        return {
            "exit_code": -1,
            "output": None,
            "log": f"ffmpeg superó el tiempo máximo ({ENHANCE_TIMEOUT} s)",
        }
    return {
        "exit_code": proc.returncode,
        "output": str(salida) if proc.returncode == 0 and salida.is_file() else None,
        "log": (proc.stderr or "")[-2000:],
    }


def _last_json(stdout: str) -> dict:
    """Último objeto JSON impreso por un worker (línea a línea)."""
    for linea in reversed((stdout or "").splitlines()):
        linea = linea.strip()
        if linea.startswith("{"):
            try:
                data = json.loads(linea)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return data
    return {}


def _probe_fps(entrada: Path) -> float:
    """fps del vídeo leyendo el stderr de ``ffmpeg -i`` (30 si no se lee)."""
    proc = _run_ffmpeg(
        [_ffmpeg_exe(), "-i", str(entrada)], PROBE_TIMEOUT, "sonda de fps"
    )
    if proc is None:
        return 30.0
    m = re.search(r"(\d+(?:\.\d+)?)\s*fps", proc.stderr or "")
    return float(m.group(1)) if m else 30.0


def run_upscale(entrada: Path, salida: Path, params: dict) -> dict:
    """Reescala con Real-ESRGAN: extrae fotogramas, reescala y remonta con audio."""
    scale = int(params.get("scale", 2))
    if scale not in (2, 4):
        scale = 2
    fps = _probe_fps(entrada)
    ffmpeg = _ffmpeg_exe()

    with tempfile.TemporaryDirectory(prefix="jarvis_upscale_") as tmp:
        frames_in = Path(tmp) / "in"
        frames_out = Path(tmp) / "out"
        frames_in.mkdir()
        frames_out.mkdir()

        # 1) Extraer fotogramas a PNG.
        extract = [
            ffmpeg, "-y", "-hide_banner", "-i", str(entrada),
            "-vf", f"fps={fps:g}", "-start_number", "0",
            str(frames_in / "%05d.png"),
        ]
        proc = _run_ffmpeg(extract, UPSCALE_TIMEOUT, "extraer fotogramas")
        if proc is None:
            return {
                "exit_code": -1,
                "output": None,
                "log": f"ffmpeg superó el tiempo máximo ({UPSCALE_TIMEOUT} s)",
            }
        if proc.returncode != 0:
            return {"exit_code": proc.returncode, "output": None, "log": (proc.stderr or "")[-2000:]}

        # 2) Reescalar con Real-ESRGAN (worker en venv-photo).
        worker = [
            str(VENV_PHOTO_PYTHON), str(UPSCALE_WORKER),
            "--frames-in", str(frames_in), "--frames-out", str(frames_out),
            "--scale", str(scale), "--model-dir", str(MODELS_DIR),
        ]
        proc = subprocess.run(
            worker, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=UPSCALE_TIMEOUT,
        )
        detalle = _last_json(proc.stdout)
        if proc.returncode != 0 or not detalle.get("ok"):
            motivo = detalle.get("error") or (proc.stderr or "").strip()[-600:]
            return {"exit_code": 1, "output": None, "log": f"Escalado IA fallido: {motivo}"}

        # 3) Remontar el vídeo reescalado con el audio original.
        assemble = [
            ffmpeg, "-y", "-hide_banner",
            "-framerate", f"{fps:g}", "-start_number", "0",
            "-i", str(frames_out / "%05d.png"),
            "-i", str(entrada),
            "-map", "0:v", "-map", "1:a?",
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "copy", "-shortest", str(salida),
        ]
        proc = _run_ffmpeg(assemble, UPSCALE_TIMEOUT, "remontar vídeo")
        if proc is None:
            return {
                "exit_code": -1,
                "output": None,
                "log": f"ffmpeg superó el tiempo máximo ({UPSCALE_TIMEOUT} s)",
            }
        if proc.returncode != 0 or not salida.is_file():
            return {"exit_code": proc.returncode, "output": None, "log": (proc.stderr or "")[-2000:]}

    return {"exit_code": 0, "output": str(salida), "log": f"Reescalado x{scale} completado"}


OPERATIONS = {
    "enhance": run_enhance,
    "upscale": run_upscale,
}


def main() -> int:
    raw = sys.stdin.read()
    try:
        req = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(json.dumps({"error": f"JSON inválido: {exc}"}))
        return 2

    op = req.get("operation")
    if op not in OPERATIONS:
        print(json.dumps({"error": f"Operación desconocida: {op}"}))
        return 2

    entrada = Path(req.get("input") or "")
    try:
        entrada, union = _resolve_input(entrada)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 2

    salida = (
        Path(req["output"])
        if req.get("output")
        else OUTPUT_DIR / f"{entrada.stem}_{op}.mp4"
    )
    salida.parent.mkdir(parents=True, exist_ok=True)

    try:
        resultado = OPERATIONS[op](entrada, salida, req.get("params", {}))
    except Exception as exc:  # noqa: BLE001 - el error debe llegar a Jarvis
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 1

    if union:
        resultado["entrada"] = union
    print(json.dumps(resultado, ensure_ascii=False))
    return 0 if resultado.get("exit_code") == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
