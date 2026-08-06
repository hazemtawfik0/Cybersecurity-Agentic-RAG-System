@echo off
setlocal
cd /d "%~dp0"

title Cybersecurity Agentic RAG

if not exist ".venv\Scripts\python.exe" (
    call setup_and_run.bat
    exit /b
)

call ".venv\Scripts\activate.bat"
python app.py
pause
