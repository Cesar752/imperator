"""Rutas compartidas del proyecto Imperator.

Centraliza la raíz del proyecto y las carpetas de datos para no repetir
``Path(__file__).resolve().parent...`` en cada módulo.

Desde la ronda 9, **todo lo que la aplicación escribe** vive en
``%APPDATA%\\Imperator``: en un .exe la carpeta de la aplicación puede ser de
solo lectura (Program Files) y allí no se pueden guardar modelos ni salidas.
"""

import os
from pathlib import Path

# Raíz del proyecto (C:\Imperator). En un .exe, la carpeta del ejecutable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directorio de assets versionados (icono, logo, splash).
ASSETS = PROJECT_ROOT / "assets"

# Carpeta de datos del usuario: lo único que la app necesita poder escribir.
USER_DATA_DIR = Path(os.environ.get("APPDATA") or Path.home()) / "Imperator"

# Carpeta donde los motores dejan lo que generan (imágenes, vídeos, audios, SRT).
# Tiene política de retención: ver services/output_cleanup.py.
OUTPUT = USER_DATA_DIR / "output"

#: Música de fondo del Ensamblador (MP3 del usuario).
MUSIC = USER_DATA_DIR / "music"

#: Caché de fotogramas extraídos de vídeo (core/frame_cache.py).
CACHE = USER_DATA_DIR / "cache"

#: Pesos de los modelos de IA (Real-ESRGAN, GFPGAN…): se descargan solos.
MODELS = USER_DATA_DIR / "models"

#: Voces de Piper y muestras de clonado de voz que añade el usuario.
VOICES = USER_DATA_DIR / "voices"

#: Carpetas que la aplicación crea al arrancar (ver :func:`ensure_dirs`).
DIRECTORIOS = (OUTPUT, MUSIC, CACHE, MODELS, VOICES)


def ensure_dirs() -> None:
    """Crea las carpetas de datos de usuario (idempotente)."""
    for carpeta in DIRECTORIOS:
        carpeta.mkdir(parents=True, exist_ok=True)
