@echo off
setlocal EnableExtensions
cd /d "C:\Users\Tawfik\Downloads\trae\cyber"

title Upload Cybersecurity Agentic RAG to GitHub

echo.
echo ============================================================
echo   Upload Cybersecurity Agentic RAG to GitHub
echo ============================================================
echo.

where git >nul 2>nul
if errorlevel 1 (
    echo ERROR: Git is not installed or is not available in PATH.
    echo Install Git for Windows, then run this file again.
    echo https://git-scm.com/download/win
    pause
    exit /b 1
)

if not exist ".gitignore" (
    echo ERROR: .gitignore is missing from this folder.
    echo Copy the provided .gitignore file here first.
    pause
    exit /b 1
)

echo [1/7] Initializing the local repository...
if not exist ".git" (
    git init
    if errorlevel 1 goto :failed
)

echo [2/7] Setting the branch name to main...
git branch -M main
if errorlevel 1 goto :failed

echo [3/7] Configuring the GitHub repository...
git remote get-url origin >nul 2>nul
if errorlevel 1 (
    git remote add origin https://github.com/hazemtawfik0/Cybersecurity-Agentic-RAG-System.git
) else (
    git remote set-url origin https://github.com/hazemtawfik0/Cybersecurity-Agentic-RAG-System.git
)
if errorlevel 1 goto :failed

echo [4/7] Checking for files larger than 95 MB...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$files = Get-ChildItem -LiteralPath . -Recurse -File -Force | Where-Object { $_.FullName -notmatch '\\.git\\|\\.venv\\|\\.hf_cache\\|\\__pycache__\\' -and $_.Length -gt 95MB }; if ($files) { Write-Host ''; Write-Host 'WARNING: Large files found:' -ForegroundColor Yellow; $files | Select-Object FullName,@{Name='SizeMB';Expression={[math]::Round($_.Length/1MB,2)}} | Format-Table -AutoSize; exit 2 }"
if errorlevel 2 (
    echo.
    echo Upload stopped because one or more non-ignored files exceed 95 MB.
    echo Remove them, add them to .gitignore, or use Git LFS.
    pause
    exit /b 2
)

echo [5/7] Adding project files...
git add .
if errorlevel 1 goto :failed

echo.
echo Files prepared for commit:
git status --short
echo.

git diff --cached --quiet
if not errorlevel 1 (
    echo No new changes were found.
    echo The project may already be committed.
    goto :push
)

echo [6/7] Creating the commit...
git commit -m "Add Cybersecurity Agentic RAG Gradio application"
if errorlevel 1 (
    echo.
    echo Git may require your name and email.
    echo Run these commands, then start this file again:
    echo.
    echo git config --global user.name "Your Name"
    echo git config --global user.email "your-email@example.com"
    echo.
    pause
    exit /b 1
)

:push
echo [7/7] Pushing to GitHub...
git push -u origin main
if errorlevel 1 (
    echo.
    echo Push failed.
    echo.
    echo If the GitHub repository already contains a README or another commit,
    echo run these commands in this folder:
    echo.
    echo git pull origin main --allow-unrelated-histories
    echo git push -u origin main
    echo.
    echo Resolve any merge conflict before running git push again.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Upload completed successfully.
echo ============================================================
echo.
echo Repository:
echo https://github.com/hazemtawfik0/Cybersecurity-Agentic-RAG-System
echo.
pause
exit /b 0

:failed
echo.
echo A Git command failed. Review the message above.
pause
exit /b 1
