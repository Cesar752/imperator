# -*- mode: python ; coding: utf-8 -*-
"""Spec de PyInstaller para Imperator (onedir).

Ronda 8: **preparado, no construido**. Falta ejecutar
``pyinstaller --clean --noconfirm imperator.spec`` con un Python 3.12 que tenga
``pyinstaller`` instalado (en esta máquina no lo hay: el único pyinstaller del
sistema es de Python 3.14 y no puede analizar las extensiones cp312 del venv).

Pendiente de decidir antes de construir (ver informe de la ronda 8):

* **ffmpeg**: ``imageio-ffmpeg`` trae el binario dentro del paquete
  (``binaries/ffmpeg-win-x86_64-v7.1.exe``, 87,6 MB) y la app lo localiza con
  ``imageio_ffmpeg.get_ffmpeg_exe()``. Aquí se empaqueta; la alternativa es
  esperar un ffmpeg en PATH (bundle ~88 MB más pequeño pero dependencia externa).
* **Modelos y voces** (``tools/models``, ``voices``): se descargan o los pone el
  usuario. No se empaquetan: en un .exe ``core.paths.PROJECT_ROOT`` apuntaría a
  una carpeta de solo lectura, así que hay que decidir una ruta de usuario.
* **Workers de ``tools/``**: se lanzan como subproceso (no se importan), así que
  van como datos y deben quedar al lado del ejecutable.
"""

from PyInstaller.utils.hooks import collect_data_files

#: Recursos versionados que la app lee en caliente.
datas = [
    ("assets", "assets"),
]

#: ffmpeg viaja dentro de imageio-ffmpeg y es lo que usa la app (no el PATH).
datas += collect_data_files("imageio_ffmpeg")

#: Los workers se ejecutan como script suelto, no se importan.
datas += [
    ("tools/photo_worker.py", "tools"),
    ("tools/video_upscale_worker.py", "tools"),
    ("tools/xtts_worker.py", "tools"),
]

#: ``core.task_manager`` importa las tareas con ``importlib`` (ronda 6), así que
#: el análisis estático de PyInstaller no las ve. Sin esto, el .exe arrancaría
#: sin pestañas.
hiddenimports = [
    "core.task_manager",
    "tasks.assembler",
    "tasks.audio",
    "tasks.chat",
    "tasks.chronicle",
    "tasks.document",
    "tasks.image",
    "tasks.notes",
    "tasks.photo",
    "tasks.understanding",
    "tasks.video",
    "tasks.video_enhance",
]

#: Motores opcionales: ``engines/perception.py`` y ``voice_clone`` los importan
#: dentro de funciones y hoy no están instalados. Si se instalan, PyInstaller
#: los recogerá por el análisis normal (son imports estáticos dentro del método).
excludes = [
    ".git",
    "venv",
    "venv-photo",
    "venv-tts",
    "tests",
    "tkinter",
    "pytest",
    "ruff",
]

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Imperator",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon="assets/icono.ico",
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="Imperator",
)
