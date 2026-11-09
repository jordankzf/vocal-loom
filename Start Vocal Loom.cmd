@echo off
setlocal
cd /d "%~dp0"
if exist "%~dp0.venv\Scripts\pythonw.exe" (
    start "VocalLoom" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0pipeline_ui.py"
    exit /b
)
if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" "%~dp0pipeline_ui.py"
    exit /b
)
echo The project environment is missing.
echo Follow the Windows quickstart in README.md to install the audio libraries.
pause
