@echo off
chcp 65001 >nul
cd /d "%~dp0"
where py >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    py -3 reecriture_ollama.py
) else (
    python reecriture_ollama.py
)
if ERRORLEVEL 1 (
    echo.
    echo Le programme s'est arrete avec une erreur.
    echo Verifie que Python 3 et Ollama sont installes.
    pause
)
