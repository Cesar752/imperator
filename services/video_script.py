"""Guion cinematográfico de vídeo: modelo puro, validación y prompts (sin Qt).

Es la pieza que faltaba para que el chat deje de producir texto suelto y genere
un **guion estructurado** que la pestaña Vídeo pueda mostrar como escenas
editables y que el motor Wan pueda consumir escena a escena con coherencia.

Tres ideas sostienen el diseño:

- **Un guion es una historia, no una lista de prompts.** El modelo pide un
  ``logline``, un estilo único (la "biblia visual") y un reparto con apariencia
  **bloqueada**; cada escena hereda ese estilo y repite literalmente la
  apariencia de los personajes que aparecen. Así los clips no "derivan": los
  mismos personajes se describen igual en todas las escenas.
- **Los límites son del motor, no del gusto.** Los clips de Wan duran 2-5 s
  (ver :mod:`services.video_params`), así que la duración por escena se acota
  aquí y el modelo recibe esa restricción explícita. Un guion de escenas de 40 s
  sería irrealizable.
- **Lógica pura y testeable.** JSON de entrada/salida, validación con errores
  legibles y *parsing* tolerante (el modelo puede envolver el JSON en vallas de
  código o añadir texto). Nada de red ni de Qt aquí: eso vive en los workers.

El prompt de sistema (:data:`SCRIPT_SYSTEM_PROMPT`) convierte al motor de chat
en un **guionista**; :func:`script_generation_messages` construye la petición
completa (sistema + idea del usuario + esquema JSON) y :func:`parse_script`
convierte la respuesta en un :class:`Script` validado.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field, replace

# ---- Límites y valores por defecto -------------------------------------------

#: Duración mínima/máxima de un clip de Wan 2.2 (alineada con los presets de
#: ``services.video_params``). Fuera de este rango el motor no genera bien.
CLIP_MIN_S = 2
CLIP_MAX_S = 5

#: Número de escenas por defecto cuando el usuario no lo dice.
DEFAULT_SCENES = 6

#: Idiomas que el modelo sabe escribir para la narración.
DEFAULT_LANGUAGE = "es"

# ---- Transiciones -----------------------------------------------------------

TRANSITION_CUT = "cut"
TRANSITION_FADE = "fade"
TRANSITION_DISSOLVE = "dissolve"
TRANSITION_SLIDE_LEFT = "slide_left"
TRANSITION_SLIDE_RIGHT = "slide_right"
TRANSITION_WIPE_LEFT = "wipe_left"
TRANSITION_WIPE_RIGHT = "wipe_right"
TRANSITION_CIRCLE = "circle"
TRANSITION_ZOOM = "zoom"

#: Transiciones admitidas. Se traducirán a filtros ``xfade`` en el ensamblador
#: (ver ``core.video_assembler.XFADE_NAMES``, que debe cubrir todas).
TRANSITIONS: tuple[str, ...] = (
    TRANSITION_CUT,
    TRANSITION_FADE,
    TRANSITION_DISSOLVE,
    TRANSITION_SLIDE_LEFT,
    TRANSITION_SLIDE_RIGHT,
    TRANSITION_WIPE_LEFT,
    TRANSITION_WIPE_RIGHT,
    TRANSITION_CIRCLE,
    TRANSITION_ZOOM,
)
TRANSITION_LABELS: dict[str, str] = {
    TRANSITION_CUT: "Corte",
    TRANSITION_FADE: "Fundido",
    TRANSITION_DISSOLVE: "Disolvencia",
    TRANSITION_SLIDE_LEFT: "Deslizar ←",
    TRANSITION_SLIDE_RIGHT: "Deslizar →",
    TRANSITION_WIPE_LEFT: "Barrido ←",
    TRANSITION_WIPE_RIGHT: "Barrido →",
    TRANSITION_CIRCLE: "Círculo",
    TRANSITION_ZOOM: "Zoom",
}

# ---- Formatos ---------------------------------------------------------------

ASPECT_HORIZONTAL = "16:9"
ASPECT_VERTICAL = "9:16"
ASPECTS: tuple[str, ...] = (ASPECT_HORIZONTAL, ASPECT_VERTICAL)
ASPECT_LABELS: dict[str, str] = {
    ASPECT_HORIZONTAL: "Horizontal (YouTube)",
    ASPECT_VERTICAL: "Vertical (TikTok)",
}


class ScriptError(ValueError):
    """El guion no es válido (campo ausente, referencia rota, duración fuera de rango)."""


# ---- Utilidades -------------------------------------------------------------


def _slug(text: str) -> str:
    """Identificador estable en minúsculas, sin acentos y con guiones."""
    decomposed = unicodedata.normalize("NFKD", str(text or ""))
    ascii_text = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    ascii_text = ascii_text.lower().strip()
    return re.sub(r"[^a-z0-9]+", "-", ascii_text).strip("-")


def clamp_duration(value: object, default: int = 4) -> int:
    """Duración entera dentro de ``[CLIP_MIN_S, CLIP_MAX_S]``; ``default`` si no es número."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        seconds = int(default)
    elif isinstance(value, str):
        try:
            seconds = int(round(float(value.strip())))
        except ValueError:
            seconds = int(default)
    else:
        seconds = int(round(float(value)))
    return max(CLIP_MIN_S, min(CLIP_MAX_S, seconds))


def _as_str(value: object, default: str = "") -> str:
    """Texto limpio (colapsa espacios); ``default`` si no es cadena."""
    if not isinstance(value, str):
        return default
    return re.sub(r"\s+", " ", value).strip() or default


def _as_str_list(value: object) -> tuple[str, ...]:
    """Lista de cadenas limpias a partir de una lista/tupla/valor suelto."""
    if value is None:
        return ()
    items = value if isinstance(value, (list, tuple)) else [value]
    result: list[str] = []
    for item in items:
        text = _as_str(item)
        if text and text not in result:
            result.append(text)
    return tuple(result)


#: Número máximo de escenas que se acepta pedir de una vez.
MAX_REQUESTED_SCENES = 20

_SCENE_COUNT_RE = re.compile(
    r"\b(\d{1,2})\s*(?:escenas?|scenes?|planos?|gidoi)", re.IGNORECASE
)


def parse_scene_count(text: str) -> int | None:
    """Número de escenas pedido en el texto («guion de 4 escenas»).

    Devuelve ``None`` si no se pide ninguno o si el número está fuera de rango
    (1-:data:`MAX_REQUESTED_SCENES`). Es lo que permite que el chat arranque el
    guion respetando el «de 4 escenas» del usuario sin obligarle a repetirlo.
    """
    match = _SCENE_COUNT_RE.search(str(text or ""))
    if not match:
        return None
    value = int(match.group(1))
    return value if 1 <= value <= MAX_REQUESTED_SCENES else None


# ---- Modelo -----------------------------------------------------------------


@dataclass(frozen=True)
class Character:
    """Un personaje del reparto con apariencia **bloqueada**.

    ``appearance`` es la descripción que se repite palabra por palabra en el
    prompt de cada escena donde aparece: es lo que mantiene al personaje igual
    entre clips (``rubia, coleta alta, chaqueta verde caqui`` da el mismo
    resultado en todas las escenas; ``una mujer alta`` no).
    """

    id: str
    name: str
    appearance: str
    description: str = ""
    voice: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "appearance": self.appearance,
            "description": self.description,
            "voice": self.voice,
        }

    @classmethod
    def from_dict(cls, data: object) -> "Character":
        if not isinstance(data, dict):
            raise ScriptError("Cada personaje debe ser un objeto JSON.")
        name = _as_str(data.get("name"))
        appearance = _as_str(data.get("appearance"))
        if not name:
            raise ScriptError("Un personaje no tiene nombre.")
        if not appearance:
            raise ScriptError(f"El personaje «{name}» no tiene apariencia bloqueada.")
        cid = _slug(data.get("id") or name) or _slug(name)
        if not cid:
            raise ScriptError(f"No se pudo generar un id para el personaje «{name}».")
        return cls(
            id=cid,
            name=name,
            appearance=appearance,
            description=_as_str(data.get("description")),
            voice=_as_str(data.get("voice")),
        )

    def descriptor(self) -> str:
        """Bloque listo para pegar en un prompt: ``Nombre (apariencia)``."""
        return f"{self.name} ({self.appearance})"


@dataclass(frozen=True)
class Scene:
    """Una escena: un clip de Wan más su narración e información de continuidad."""

    heading: str
    action: str
    visual_prompt: str
    narration: str = ""
    characters: tuple[str, ...] = ()
    duration_s: int = 4
    transition: str = TRANSITION_CUT

    def to_dict(self) -> dict:
        return {
            "heading": self.heading,
            "action": self.action,
            "visual_prompt": self.visual_prompt,
            "narration": self.narration,
            "characters": list(self.characters),
            "duration_s": self.duration_s,
            "transition": self.transition,
        }

    @classmethod
    def from_dict(cls, data: object) -> "Scene":
        if not isinstance(data, dict):
            raise ScriptError("Cada escena debe ser un objeto JSON.")
        visual = _as_str(data.get("visual_prompt"))
        action = _as_str(data.get("action"))
        if not visual and not action:
            raise ScriptError("Una escena no tiene descripción visual ni acción.")
        transition = _as_str(data.get("transition"), TRANSITION_CUT).lower()
        if transition not in TRANSITIONS:
            transition = TRANSITION_CUT
        return cls(
            heading=_as_str(data.get("heading")),
            action=action,
            visual_prompt=visual or action,
            narration=_as_str(data.get("narration")),
            characters=_as_str_list(data.get("characters")),
            duration_s=clamp_duration(data.get("duration_s")),
            transition=transition,
        )


@dataclass(frozen=True)
class Script:
    """Guion completo: reparto, estilo único y escenas ordenadas."""

    title: str
    logline: str = ""
    language: str = DEFAULT_LANGUAGE
    style: str = ""
    aspect: str = ASPECT_HORIZONTAL
    characters: tuple[Character, ...] = ()
    scenes: tuple[Scene, ...] = ()
    #: Ajustes propios del proyecto (preset, fps, transiciones…). Se persisten
    #: con el guion para que reabrirlo restaure cómo se generó.
    config: dict = field(default_factory=dict)

    # ---- Accesos --------------------------------------------------

    def character(self, character_id: str) -> Character | None:
        """Personaje por id (los ids se normalizan con :func:`_slug`)."""
        wanted = _slug(character_id)
        return next((c for c in self.characters if c.id == wanted), None)

    def total_duration(self) -> int:
        """Duración estimada del vídeo (suma de las escenas, en segundos)."""
        return sum(scene.duration_s for scene in self.scenes)

    def narration(self) -> str:
        """Narración completa, escena a escena, lista para el TTS."""
        return "\n".join(s.narration for s in self.scenes if s.narration).strip()

    # ---- Validación ----------------------------------------------

    def validate(self) -> None:
        """Lanza :class:`ScriptError` con el primer problema encontrado."""
        if not _as_str(self.title):
            raise ScriptError("El guion no tiene título.")
        if not self.scenes:
            raise ScriptError("El guion no tiene ninguna escena.")
        if self.aspect not in ASPECTS:
            raise ScriptError(
                f"Formato desconocido «{self.aspect}» (usa {' o '.join(ASPECTS)})."
            )
        if not _as_str(self.language, DEFAULT_LANGUAGE):
            raise ScriptError("El guion no indica idioma.")
        if not isinstance(self.config, dict):
            raise ScriptError("El campo «config» del guion debe ser un objeto.")

        ids = [c.id for c in self.characters]
        if len(ids) != len(set(ids)):
            raise ScriptError("Hay personajes con el mismo id.")

        for position, scene in enumerate(self.scenes, start=1):
            if not (_as_str(scene.visual_prompt) or _as_str(scene.action)):
                raise ScriptError(f"La escena {position} no describe nada visualmente.")
            if not (CLIP_MIN_S <= scene.duration_s <= CLIP_MAX_S):
                raise ScriptError(
                    f"La escena {position} dura {scene.duration_s} s; el motor admite "
                    f"entre {CLIP_MIN_S} y {CLIP_MAX_S} s."
                )
            for cid in scene.characters:
                if _slug(cid) not in ids:
                    raise ScriptError(
                        f"La escena {position} referencia al personaje «{cid}», "
                        "que no está en el reparto."
                    )

    # ---- Serialización -------------------------------------------

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "logline": self.logline,
            "language": self.language,
            "style": self.style,
            "aspect": self.aspect,
            "config": dict(self.config),
            "characters": [c.to_dict() for c in self.characters],
            "scenes": [s.to_dict() for s in self.scenes],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, data: object) -> "Script":
        """Construye y valida un guion a partir de un dict (tolerante a faltantes)."""
        if not isinstance(data, dict):
            raise ScriptError("El guion debe ser un objeto JSON.")

        raw_chars = data.get("characters") or []
        if not isinstance(raw_chars, list):
            raise ScriptError("«characters» debe ser una lista.")
        characters = tuple(Character.from_dict(item) for item in raw_chars)

        raw_scenes = data.get("scenes") or []
        if not isinstance(raw_scenes, list):
            raise ScriptError("«scenes» debe ser una lista.")
        scenes = tuple(Scene.from_dict(item) for item in raw_scenes)

        aspect = _as_str(data.get("aspect"), ASPECT_HORIZONTAL)
        raw_config = data.get("config")
        config = dict(raw_config) if isinstance(raw_config, dict) else {}
        script = cls(
            title=_as_str(data.get("title")),
            logline=_as_str(data.get("logline")),
            language=_as_str(data.get("language"), DEFAULT_LANGUAGE),
            style=_as_str(data.get("style")),
            aspect=aspect,
            characters=characters,
            scenes=scenes,
            config=config,
        )
        script.validate()
        return script

    @classmethod
    def from_json(cls, text: str) -> "Script":
        try:
            data = json.loads(text)
        except ValueError as exc:
            raise ScriptError(f"El JSON del guion no se pudo parsear: {exc}") from exc
        return cls.from_dict(data)

    # ---- Edición (devuelven un guion nuevo; el modelo es inmutable) ----

    def with_scene(self, index: int, scene: Scene) -> "Script":
        """Reemplaza la escena ``index`` (0-based)."""
        scenes = list(self.scenes)
        if not 0 <= index < len(scenes):
            raise ScriptError(f"No existe la escena {index + 1}.")
        scenes[index] = scene
        return replace(self, scenes=tuple(scenes))

    def add_scene(self, scene: Scene, at: int | None = None) -> "Script":
        """Añade una escena al final o en la posición ``at``."""
        scenes = list(self.scenes)
        if at is None or at >= len(scenes):
            scenes.append(scene)
        else:
            scenes.insert(max(0, at), scene)
        return replace(self, scenes=tuple(scenes))

    def remove_scene(self, index: int) -> "Script":
        scenes = list(self.scenes)
        if not 0 <= index < len(scenes):
            raise ScriptError(f"No existe la escena {index + 1}.")
        del scenes[index]
        return replace(self, scenes=tuple(scenes))

    def move_scene(self, source: int, target: int) -> "Script":
        """Reordena una escena (para la tabla editable de la pestaña Vídeo)."""
        scenes = list(self.scenes)
        if not 0 <= source < len(scenes) or not 0 <= target < len(scenes):
            raise ScriptError("Movimiento de escena fuera de rango.")
        scene = scenes.pop(source)
        scenes.insert(target, scene)
        return replace(self, scenes=tuple(scenes))


# ---- Prompts ----------------------------------------------------------------


#: Prompt de sistema: convierte el motor de chat en un guionista cinematográfico.
SCRIPT_SYSTEM_PROMPT = """Eres un guionista cinematográfico y director de arte.
Escribes guiones cortos para vídeos generados por IA (clips de 2 a 5 segundos).

Reglas que cumples siempre:
1. Cuentas UNA historia coherente con inicio, desarrollo y cierre; nada de escenas sueltas.
2. Mantienes a los mismos personajes de principio a fin, con la MISMA apariencia
   física descrita de forma idéntica en todas las escenas (color de pelo, ropa,
   edad, rasgos). Nunca cambias su aspecto ni su nombre.
3. Defines un estilo visual único (época, luz, paleta, cámara) que se aplica a
   todas las escenas, para que el vídeo parezca del mismo rodaje.
4. Cada escena se puede filmar en un solo plano de 2 a 5 segundos: describe lo
   que se ve en un instante, no acciones largas.
5. La narración es breve (una o dos frases por escena) y avanza la historia.
6. Respetas el idioma, el número de escenas y el formato (horizontal o vertical)
   que se te pidan.
7. Respondes ÚNICAMENTE con un objeto JSON válido, sin explicaciones ni markdown
   alrededor. Usa exactamente el esquema indicado."""


#: Esquema JSON que se le pide al modelo (texto para el prompt, no codificación).
SCRIPT_JSON_SCHEMA = """{
  "title": "título corto del vídeo",
  "logline": "una frase que resume la historia",
  "language": "es",
  "style": "biblia visual única que se aplica a TODAS las escenas",
  "characters": [
    {
      "id": "identificador-corto",
      "name": "Nombre",
      "appearance": "apariencia física fija y detallada (se repite igual en cada escena)",
      "description": "rasgo de carácter o papel en la historia",
      "voice": "tono de voz para la narración, si procede"
    }
  ],
  "scenes": [
    {
      "heading": "INT./EXT. LUGAR - MOMENTO",
      "action": "qué ocurre en la escena en una frase",
      "visual_prompt": "qué se ve exactamente en el plano (para el generador de vídeo)",
      "narration": "texto que se narra en esta escena",
      "characters": ["id-de-cada-personaje-que-aparece"],
      "duration_s": 4,
      "transition": "cut|fade|dissolve|slide_left|slide_right|wipe_left|wipe_right|circle|zoom"
    }
  ]
}"""


def script_generation_messages(
    idea: str,
    *,
    scenes: int = DEFAULT_SCENES,
    language: str = DEFAULT_LANGUAGE,
    aspect: str = ASPECT_HORIZONTAL,
    style: str = "",
) -> list[dict]:
    """Mensajes (sistema + usuario) para pedirle un guion al motor de chat.

    Es la forma correcta de "hablar" con el motor cuando el usuario pide un
    vídeo: en lugar de una conversación abierta, se le da un rol y un esquema.
    """
    scene_count = max(1, int(scenes))
    total = scene_count * CLIP_MIN_S
    aspect_label = ASPECT_LABELS.get(aspect, aspect)
    style_hint = f"\nEstilo visual pedido por el usuario: {style}." if style else ""

    user = (
        f"Idea del vídeo: {_as_str(idea)}\n\n"
        f"Escribe un guion de {scene_count} escenas en idioma «{language}», "
        f"en formato {aspect_label}.{style_hint}\n\n"
        f"Cada escena dura entre {CLIP_MIN_S} y {CLIP_MAX_S} segundos "
        f"(el vídeo será de unos {total}-{scene_count * CLIP_MAX_S} s en total).\n"
        "Mantén los mismos personajes con la misma apariencia en todas las escenas "
        "y un único estilo visual coherente.\n\n"
        "Devuelve SOLO el JSON con este esquema exacto:\n"
        f"{SCRIPT_JSON_SCHEMA}"
    )
    return [
        {"role": "system", "content": SCRIPT_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def character_block(script: Script, character_ids: tuple[str, ...]) -> str:
    """Bloque de apariencia de los personajes presentes, para pegar en el prompt.

    Repetir la apariencia **literal** es lo que mantiene al personaje igual entre
    escenas; por eso se toma tal cual del reparto y no se reformula.
    """
    descriptors: list[str] = []
    for cid in character_ids:
        character = script.character(cid)
        if character is not None:
            descriptors.append(character.descriptor())
    return ", ".join(descriptors)


def scene_visual_prompt(script: Script, scene: Scene) -> str:
    """Prompt final de Wan para una escena: estilo + personajes + plano.

    El orden importa: primero la biblia visual y los personajes (constantes en
    todo el vídeo), después la acción concreta del plano.
    """
    parts: list[str] = []
    if script.style:
        parts.append(script.style)
    people = character_block(script, scene.characters)
    if people:
        parts.append(people)
    if scene.visual_prompt:
        parts.append(scene.visual_prompt)
    elif scene.action:
        parts.append(scene.action)
    return ". ".join(p.strip().rstrip(".") for p in parts if p and p.strip())


def scene_prompts(script: Script) -> list[str]:
    """Prompt de cada escena, en orden (lo que consumirá el generador de vídeo)."""
    return [scene_visual_prompt(script, scene) for scene in script.scenes]


def transition_sequence(script: Script) -> list[str]:
    """Transición de cada cruce entre escenas consecutivas (``n - 1`` valores).

    La transición de la escena *i* describe el paso de la escena *i-1* a la *i*;
    la de la primera se ignora porque no hay cruce antes. Es lo que consume
    ``VideoAssembler.concat_clips_with_transitions``.
    """
    return [script.scenes[i].transition for i in range(1, len(script.scenes))]


# ---- Parsing tolerante ------------------------------------------------------


def _extract_json(text: str) -> object:
    """Extrae el objeto JSON de una respuesta que puede traer texto alrededor."""
    raw = str(text or "").strip()
    if not raw:
        raise ScriptError("La respuesta del modelo está vacía.")
    fence = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL | re.IGNORECASE)
    if fence:
        raw = fence.group(1).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end <= start:
        raise ScriptError("La respuesta del modelo no contiene un JSON de guion.")
    candidate = raw[start : end + 1]
    try:
        return json.loads(candidate)
    except ValueError as exc:
        raise ScriptError(f"El JSON del guion no se pudo parsear: {exc}") from exc


def parse_script(text: str) -> Script:
    """Convierte la respuesta del modelo en un :class:`Script` validado."""
    return Script.from_dict(_extract_json(text))


def try_parse_script(text: str) -> tuple[Script | None, str]:
    """Como :func:`parse_script` pero sin lanzar: ``(guion | None, error)``."""
    try:
        return parse_script(text), ""
    except ScriptError as exc:
        return None, str(exc)


#: Nombres de campo obligatorios, para mensajes de ayuda en la UI.
REQUIRED_SCENE_FIELDS: tuple[str, ...] = ("visual_prompt", "narration")
