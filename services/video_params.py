"""Presets y construcción de parámetros de la tarea de vídeo (sin Qt).

Incluye la **continuidad entre escenas**: un guion (``services.video_script``) se
traduce a los parámetros de cada clip con una semilla derivada de una base común
(así todos los planos comparten "ADN" y no cambian de aspecto) y, opcionalmente,
con el **encadenado i2v** (el último fotograma de una escena entra como imagen de
la siguiente), que es lo que da continuidad visual real.
"""

import random

from services import video_script as vscript

# Preset: id -> (label, width, height, duración_s, tooltip).
PRESETS: dict[str, tuple[str, int, int, int, str]] = {
    "rapido": ("Rápido", 832, 480, 2, "832×480 · 2 s (~69 s)"),
    "equilibrado": ("Equilibrado", 832, 480, 4, "832×480 · 4 s (~140 s)"),
    "calidad": ("Calidad", 1024, 576, 5, "1024×576 · 5 s (~250 s)"),
    "tiktok_rapido": ("TikTok Rápido", 480, 832, 2, "480×832 · 2 s (~55 s)"),
    "tiktok_calidad": ("TikTok Calidad", 720, 1280, 3, "720×1280 · 3 s (~120 s)"),
    "tiktok_alto": ("TikTok Alto", 896, 1600, 4, "896×1600 · 4 s (~200 s, mucha VRAM)"),
}

# Orden de visualización de cada grupo de presets.
HORIZONTAL_KEYS = ("rapido", "equilibrado", "calidad")
VERTICAL_KEYS = ("tiktok_rapido", "tiktok_calidad", "tiktok_alto")

# Etiquetas de los modos (clave del workflow -> texto legible).
MODE_LABELS: dict[str, str] = {
    "t2v": "Texto a vídeo",
    "i2v": "Imagen a vídeo",
}


def sequence_length(duration_s: int | float, fps: int) -> int:
    """Frames totales del clip, redondeados a la forma ``4n+1`` de Wan 2.2.

    ``duración * fps + 1``, ajustado al ``4n+1`` más cercano: los nodos de
    latent de Wan 2.2 (``EmptyHunyuanLatentVideo`` y ``WanImageToVideo``)
    rechazan longitudes que no sean ``4n+1`` (p. ej. 91 frames con 30 fps y 3 s,
    que ComfyUI devolvía como error de validación del workflow).
    """
    frames = round(duration_s * fps) + 1
    return max(1, ((frames - 1) + 2) // 4 * 4 + 1)


def build_generate_params(
    *,
    mode: str,
    negative_prompt: str,
    width: int,
    height: int,
    duration: int,
    fps: int,
    prompt: str = "",
    photos: list[dict] | None = None,
    seed: int | None = None,
) -> dict:
    """Construye los ``kwargs`` de ``generate_video`` según el modo (t2v/i2v).

    En ``i2v`` con una sola foto se usa ``image_path``/``prompt``; con varias se
    pasan ``image_paths``/``prompts`` (una por foto, para generar y concatenar).

    **Sin fotos** (``i2v`` del encadenado, donde el ``image_path`` se pone
    después) no se añade ninguna de esas dos claves. Antes se ponían vacías y
    llegaban al motor como parámetros que no entiende: era el «unexpected
    keyword argument 'image_paths'» que tumbaba el encadenado entre escenas.
    """
    params: dict = {
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "length": sequence_length(duration, fps),
        "fps": fps,
        "model": mode,
    }
    if mode == "i2v":
        photos = photos or []
        if len(photos) == 1:
            params["image_path"] = photos[0]["path"]
            params["prompt"] = photos[0]["prompt"]
        elif photos:
            params["image_paths"] = [p["path"] for p in photos]
            params["prompts"] = [p["prompt"] for p in photos]
        else:
            params["image_path"] = None
            params["prompt"] = prompt
    else:
        params["image_path"] = None
        params["prompt"] = prompt
    if seed is not None:
        params["seed"] = seed
    return params


def parse_seed(text: str) -> int:
    """Convierte el texto del campo seed a entero (lanza ``ValueError``)."""
    return int(text.strip())


# ---- Continuidad entre escenas ----------------------------------------------

#: Máximo de una semilla de 31 bits (el rango que aceptan los nodos de ComfyUI).
MAX_SEED = 2**31 - 1


def new_base_seed() -> int:
    """Semilla base aleatoria para una secuencia de escenas."""
    return random.randint(0, MAX_SEED)


def scene_seed(base_seed: int, index: int) -> int:
    """Semilla de la escena ``index`` derivada de ``base_seed``.

    Deterministica y distinta por escena: la misma base reproduce la secuencia
    entera, y cada escena no es un clon del mismo ruido (que daría planos
    repetidos). El primo ``9973`` desacopla semillas consecutivas.
    """
    return (int(base_seed) + (index + 1) * 9973) % MAX_SEED


def default_preset_for(aspect: str) -> str:
    """Preset por defecto según el formato del guion (vertical ⇒ TikTok)."""
    if aspect == vscript.ASPECT_VERTICAL:
        return "tiktok_calidad"
    return "equilibrado"


def scene_generate_params(
    script: vscript.Script,
    index: int,
    *,
    base_seed: int,
    preset_key: str | None = None,
    fps: int = 16,
    mode: str = "t2v",
    negative_prompt: str = "",
    previous_frame: str | None = None,
    width: int | None = None,
    height: int | None = None,
) -> dict:
    """Parámetros de ``generate_video`` para la escena ``index`` del guion.

    - El prompt combina estilo + apariencia de los personajes + plano
      (:func:`services.video_script.scene_visual_prompt`), así el clip es
      coherente con el resto.
    - La semilla sale de ``base_seed`` (misma secuencia reproducible).
    - Con ``mode="i2v"`` y ``previous_frame``, ese fotograma entra como imagen de
      partida: es el encadenado que mantiene la continuidad entre escenas.
    - ``width``/``height`` permiten respetar el preset elegido en la pestaña; si
      no llegan, se usa el preset por defecto del formato del guion. Sin esto la
      generación por escenas ignoraba el tamaño configurado en la UI.
    """
    scene = script.scenes[index]
    preset_key = preset_key or default_preset_for(script.aspect)
    _label, preset_w, preset_h, _duration, _tip = PRESETS.get(
        preset_key, PRESETS["equilibrado"]
    )
    prompt = vscript.scene_visual_prompt(script, scene)
    params = build_generate_params(
        mode=mode,
        negative_prompt=negative_prompt,
        width=int(width) if width else preset_w,
        height=int(height) if height else preset_h,
        duration=scene.duration_s,
        fps=fps,
        prompt=prompt,
        seed=scene_seed(base_seed, index),
    )
    if mode == "i2v" and previous_frame:
        params["image_path"] = str(previous_frame)
        params["prompt"] = prompt
    return params


# ---- Fotos de escena (imagen → vídeo) ---------------------------------------

#: Steps y guidance por defecto de las fotos de escena (los mismos de Imagen).
IMAGE_STEPS = 20
IMAGE_GUIDANCE = 3.5


def default_size(
    script: vscript.Script, width: int | None = None, height: int | None = None
) -> tuple[int, int]:
    """Tamaño ``(ancho, alto)`` de la escena: el explícito o el preset del formato."""
    _label, preset_w, preset_h, _duration, _tip = PRESETS.get(
        default_preset_for(script.aspect), PRESETS["equilibrado"]
    )
    return (
        int(width) if width else preset_w,
        int(height) if height else preset_h,
    )


def scene_image_params(
    script: vscript.Script,
    index: int,
    *,
    model: str | None = None,
    width: int | None = None,
    height: int | None = None,
    negative_prompt: str = "",
    seed: int | None = None,
    steps: int = IMAGE_STEPS,
    guidance: float = IMAGE_GUIDANCE,
) -> dict:
    """Parámetros de ``generate_image`` para la **foto** de la escena ``index``.

    Es lo que hace que cada escena tenga foto y prompt **a la vez**: la imagen se
    genera con el mismo prompt visual que usará el clip
    (:func:`services.video_script.scene_visual_prompt` = estilo + personajes +
    plano) y con la semilla de la escena, así que la foto es reproducible y
    coherente con el resto de la secuencia. El tamaño es el del vídeo para que
    el encuadre no se deforme al entrar como primer fotograma (i2v).
    """
    scene = script.scenes[index]
    scene_w, scene_h = default_size(script, width, height)
    params: dict = {
        "prompt": vscript.scene_visual_prompt(script, scene),
        "negative_prompt": negative_prompt,
        "width": scene_w,
        "height": scene_h,
        "steps": int(steps),
        "guidance": float(guidance),
    }
    if model:
        params["model"] = model
    if seed is not None:
        params["seed"] = seed
    return params
