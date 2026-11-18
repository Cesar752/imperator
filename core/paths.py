"""Rutas compartidas del proyecto Imperator.

Centraliza la raíz del proyecto y el directorio de assets para no repetir
``Path(__file__).resolve().parent...`` en cada módulo.
"""

from pathlib import Path

# Raíz del proyecto (C:\Imperator).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directorio de assets versionados (icono, logo, splash).
ASSETS = PROJECT_ROOT / "assets"

# Carpeta donde los motores dejan lo que generan (imágenes, vídeos, audios, SRT).
# Tiene política de retención: ver services/output_cleanup.py.
OUTPUT = PROJECT_ROOT / "output"

#: Música de fondo del Ensamblador (MP3 del usuario; gitignored).
MUSIC = PROJECT_ROOT / "music"

#: Caché de fotogramas extraídos de vídeo (core/frame_cache.py).
CACHE = PROJECT_ROOT / "cache"
