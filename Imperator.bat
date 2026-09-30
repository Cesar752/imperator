@echo off
rem Lanza Imperator sin ventana de consola (pythonw), de modo que en la barra de
rem tareas solo quede el boton de la aplicacion, con su propio icono.
rem
rem Para dejarlo fijo: abre el programa con este .bat, haz clic derecho sobre su
rem icono en la barra de tareas y elige "Anclar a la barra de tareas".
setlocal
cd /d "%~dp0"

if not exist "venv\Scripts\pythonw.exe" (
    echo No se encuentra venv\Scripts\pythonw.exe
    echo Crea el entorno con:
    echo     py -m venv venv
    echo     venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)

start "" "venv\Scripts\pythonw.exe" main.py
