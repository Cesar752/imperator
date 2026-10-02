"""Proyecto de vídeo persistente: carpeta con el guion, los clips y el audio.

La generación por escenas produce **muchos** artefactos intermedios (un MP4 por
escena, un WAV de narración por escena, el SRT). Guardarlos "sueltos" en
``output/`` tenía dos problemas: la retención de ``output/`` los puede borrar
(ver :mod:`services.output_cleanup`) y no había forma de reabrir un trabajo a
medias. Un **proyecto** es una carpeta autocontenida:

```
projects/
└── el-faro/
    ├── proyecto.json     # manifiesto (nombre, fechas, nº de escenas)
    ├── guion.json        # el Script completo (services.video_script)
    ├── escena_00.mp4     # clip de cada escena
    ├── escena_01.mp4
    ├── narracion_00.wav  # narración de cada escena (opcional)
    ├── subtitulos.srt    # subtítulos (opcional)
    └── video_final.mp4   # montaje final (opcional)
```

Vive en ``projects/`` (no en ``output/``), así que **la retención nunca lo toca**
y el usuario puede seguir editando a mano (pestañas Audio y Ensamblador) sin
perder los intermedios.

Lógica pura: solo sistema de archivos y JSON, sin Qt ni motores.
"""

from __future__ import annotations

import json
import re
import shutil
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from services import video_script as vscript

#: Carpeta raíz de los proyectos (colgando de la raíz del repo, no de output/).
PROJECTS_DIRNAME = "projects"

SCRIPT_FILENAME = "guion.json"
MANIFEST_FILENAME = "proyecto.json"
SUBTITLES_FILENAME = "subtitulos.srt"
FINAL_FILENAME = "video_final.mp4"

#: Plantillas de nombre de los artefactos por escena.
CLIP_TEMPLATE = "escena_{index:02d}.mp4"
NARRATION_TEMPLATE = "narracion_{index:02d}.wav"

_REPO_ROOT = Path(__file__).resolve().parent.parent


def projects_root(root: str | Path | None = None) -> Path:
    """Carpeta raíz de proyectos (``projects/`` bajo la raíz si no se indica otra)."""
    base = Path(root) if root is not None else _REPO_ROOT
    return base / PROJECTS_DIRNAME


def slugify(name: str) -> str:
    """Nombre de carpeta seguro y estable (sin acentos, espacios ni símbolos)."""
    decomposed = unicodedata.normalize("NFKD", str(name or ""))
    ascii_text = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")


@dataclass(frozen=True)
class Project:
    """Un proyecto de vídeo: su carpeta y el guion que lo describe."""

    directory: Path
    script: vscript.Script

    @property
    def name(self) -> str:
        return self.directory.name

    @property
    def script_path(self) -> Path:
        return self.directory / SCRIPT_FILENAME

    @property
    def manifest_path(self) -> Path:
        return self.directory / MANIFEST_FILENAME

    @property
    def subtitles_path(self) -> Path:
        return self.directory / SUBTITLES_FILENAME

    @property
    def final_path(self) -> Path:
        return self.directory / FINAL_FILENAME

    def clip_path(self, index: int) -> Path:
        """Ruta del clip de la escena ``index`` (0-based)."""
        return self.directory / CLIP_TEMPLATE.format(index=index)

    def narration_path(self, index: int) -> Path:
        """Ruta del WAV de narración de la escena ``index`` (0-based)."""
        return self.directory / NARRATION_TEMPLATE.format(index=index)

    def saved_clips(self) -> list[Path]:
        """Clips ya generados, en orden de escena (para el Ensamblador)."""
        return [
            path
            for i in range(len(self.script.scenes))
            if (path := self.clip_path(i)).is_file()
        ]

    def missing_scenes(self) -> list[int]:
        """Índices (1-based) de las escenas cuyo clip aún no existe."""
        return [
            i + 1
            for i in range(len(self.script.scenes))
            if not self.clip_path(i).is_file()
        ]


def create_project(
    name: str,
    script: vscript.Script,
    root: str | Path | None = None,
    *,
    overwrite: bool = False,
) -> Project:
    """Crea la carpeta del proyecto y guarda el guion; devuelve el proyecto."""
    script.validate()
    base = projects_root(root)
    slug = slugify(name) or slugify(script.title) or "proyecto"
    folder = base / slug
    if folder.exists() and not overwrite:
        raise FileExistsError(f"El proyecto «{folder.name}» ya existe.")
    folder.mkdir(parents=True, exist_ok=True)
    project = Project(directory=folder, script=script)
    save_script(project, script)
    return project


def save_script(project: Project, script: vscript.Script) -> Path:
    """Escribe ``guion.json`` y actualiza el manifiesto; devuelve la ruta del guion."""
    project.directory.mkdir(parents=True, exist_ok=True)
    project.script_path.write_text(script.to_json(), encoding="utf-8")
    manifest = {
        "name": project.name,
        "title": script.title,
        "language": script.language,
        "aspect": script.aspect,
        "scene_count": len(script.scenes),
        "updated": datetime.now().isoformat(timespec="seconds"),
    }
    project.manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return project.script_path


def load_project(directory: str | Path) -> Project:
    """Abre un proyecto existente a partir de su carpeta (valida el guion)."""
    folder = Path(directory)
    script_path = folder / SCRIPT_FILENAME
    if not script_path.is_file():
        raise vscript.ScriptError(f"No hay guion.json en «{folder}».")
    script = vscript.Script.from_json(script_path.read_text(encoding="utf-8"))
    return Project(directory=folder, script=script)


def list_projects(root: str | Path | None = None) -> list[Path]:
    """Carpetas de proyecto con ``guion.json``, ordenadas por fecha (reciente primero)."""
    base = projects_root(root)
    if not base.is_dir():
        return []
    folders = [p for p in base.iterdir() if (p / SCRIPT_FILENAME).is_file()]
    return sorted(folders, key=lambda p: (p / SCRIPT_FILENAME).stat().st_mtime, reverse=True)


def delete_project(directory: str | Path) -> bool:
    """Borra la carpeta de un proyecto (clips incluidos). ``False`` si no existe.

    ``projects/`` vive fuera de la retención de ``output/`` a propósito (para no
    perder intermedios), así que el borrado es **explícito**: el usuario decide
    cuándo liberar el disco. Solo actúa sobre un directorio que parece un
    proyecto (con ``guion.json``) para no borrar una carpeta cualquiera.
    """
    folder = Path(directory)
    if not (folder / SCRIPT_FILENAME).is_file():
        return False
    shutil.rmtree(folder, ignore_errors=True)
    return not folder.exists()


#: Extensiones que el proyecto conserva (para saber qué no es un artefacto suyo).
PROJECT_ASSET_SUFFIXES: tuple[str, ...] = (".mp4", ".wav", ".srt", ".json")
