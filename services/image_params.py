"""Presets y construcción de parámetros de la tarea de imagen (sin Qt)."""

# Preset: clave -> (label, width, height, tooltip).
PRESETS: dict[str, tuple[str, int, int, str]] = {
    "vertical": ("Vertical 9:16", 768, 1344, "768×1344 · vertical para TikTok/Reels"),
    "cuadrado": ("Cuadrado 1:1", 1024, 1024, "1024×1024 · default"),
    "horizontal": ("Horizontal 16:9", 1344, 768, "1344×768 · panorámico"),
}

# Modelos que REQUIEREN una imagen de referencia (solo edición de imagen).
REQUIRED_REFERENCE_MODELS = {
    "flux_kontext",        # ComfyUI local (gratis)
    "flux-kontext-pro",    # BFL (pago)
    "flux-kontext-max",    # BFL (pago)
}

# Modelos con referencia OPCIONAL: con foto editan, sin foto crean desde cero.
OPTIONAL_REFERENCE_MODELS = {
    "flux-2-pro",          # BFL (pago)
}

# Etiquetas legibles para los modelos de la API de pago.
BFL_LABELS: dict[str, str] = {
    "flux-kontext-pro": "FLUX.1 Kontext Pro",
    "flux-kontext-max": "FLUX.1 Kontext Max",
    "flux-2-pro": "FLUX.2 Pro",
}


def display_model(model_id: str, engine_id: str = "") -> str:
    """Nombre legible de un modelo (añade la marca ``PAGO`` para BFL)."""
    if engine_id == "bfl":
        return f"{BFL_LABELS.get(model_id, model_id)} · PAGO"
    return model_id.replace("_", " ").title()


def requires_reference(model: str) -> bool:
    return model in REQUIRED_REFERENCE_MODELS


def optional_reference(model: str) -> bool:
    return model in OPTIONAL_REFERENCE_MODELS


def parse_seed(text: str) -> int:
    """Convierte el texto del campo seed a entero (lanza ``ValueError``)."""
    return int(text.strip())


def build_generate_params(
    *,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    model: str,
    ref_path: str = "",
    seed: int | None = None,
) -> dict:
    """Construye los ``kwargs`` de ``generate_image``."""
    params: dict = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "steps": steps,
        "guidance": guidance,
        "model": model,
    }
    if ref_path:
        params["image_path"] = ref_path
    if seed is not None:
        params["seed"] = seed
    return params
