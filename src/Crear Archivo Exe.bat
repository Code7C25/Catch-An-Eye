@echo off
REM =================================================================
REM  SCRIPT DE COMPILACIÓN CON PYINSTALLER
REM  Edita las variables en la sección "CONFIGURACIÓN" para tu proyecto.
REM  Asegúrate de ejecutar este archivo desde la carpeta raíz de tu proyecto.
REM =================================================================

echo Iniciando la compilacion con PyInstaller...
echo.

REM --- CONFIGURACIÓN (MODIFICA ESTAS LÍNEAS) ---

REM El nombre de tu archivo principal de Python.
set SCRIPT_NAME=eye_tracker_final.py

REM El nombre que tendrá tu archivo .exe final.
set APP_NAME=Catch-An-Eye

REM La ruta a tu archivo de icono (.ico). Usa "NONE" si no tienes uno.
set ICON_PATH=Catch-An-Eye.ico


REM --- COMANDO DE PYINSTALLER ---

pyinstaller ^
    --onefile ^
    --windowed ^
    --name "%APP_NAME%" ^
    --icon="%ICON_PATH%" ^
    --clean ^
    "%SCRIPT_NAME%"


echo.
echo ==========================================================
echo Compilacion finalizada!
echo El archivo ejecutable se encuentra en la carpeta 'dist'.
echo ==========================================================
echo.
pause