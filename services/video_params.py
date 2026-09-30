"""Presets y construcción de parámetros de la tarea de vídeo (sin Qt)."""

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
        else:
            params["image_paths"] = [p["path"] for p in photos]
            params["prompts"] = [p["prompt"] for p in photos]
    else:
        params["image_path"] = None
        params["prompt"] = prompt
    if seed is not None:
        params["seed"] = seed
    return params


def parse_seed(text: str) -> int:
    """Convierte el texto del campo seed a entero (lanza ``ValueError``)."""
    return int(text.strip())
