@echo off
setlocal
cd /d "%~dp0"

title Cybersecurity Agentic RAG - Setup

echo.
echo ============================================================
echo   Cybersecurity Agentic RAG - Setup and Launch
echo ============================================================
echo.

where py >nul 2>nul
if %errorlevel%==0 (
    set "PYTHON_CMD=py -3"
) else (
    where python >nul 2>nul
    if errorlevel 1 (
        echo Python was not found.
        echo Install Python 3.10 or 3.11 and enable Add Python to PATH.
        pause
        exit /b 1
    )
    set "PYTHON_CMD=python"
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Creating virtual environment...
    %PYTHON_CMD% -m venv .venv
    if errorlevel 1 (
        echo Failed to create the virtual environment.
        pause
        exit /b 1
    )
)

call ".venv\Scripts\activate.bat"

echo [2/3] Installing or updating packages...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo Package installation failed. Check your internet connection.
    pause
    exit /b 1
)

echo [3/3] Starting the app...
echo.
python app.py
pause
