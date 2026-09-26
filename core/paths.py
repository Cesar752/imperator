"""Rutas compartidas del proyecto Imperator.

Centraliza la raíz del proyecto y el directorio de assets para no repetir
``Path(__file__).resolve().parent...`` en cada módulo.
"""

from pathlib import Path

# Raíz del proyecto (C:\Imperator).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directorio de assets versionados (icono, logo, splash).
ASSETS = PROJECT_ROOT / "assets"
