@echo off
setlocal
cd /d "C:\Users\Tawfik\Downloads\trae\cyber"

title Check GitHub Upload Files

echo.
echo Checking ignored folders...
echo.

if exist ".venv" (
    echo [OK] .venv exists locally and should be ignored.
) else (
    echo [INFO] .venv was not found.
)

if exist ".hf_cache" (
    echo [OK] .hf_cache exists locally and should be ignored.
) else (
    echo [INFO] .hf_cache was not found.
)

echo.
echo Checking files larger than 95 MB outside ignored folders...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$files = Get-ChildItem -LiteralPath . -Recurse -File -Force | Where-Object { $_.FullName -notmatch '\\.git\\|\\.venv\\|\\.hf_cache\\|\\__pycache__\\' -and $_.Length -gt 95MB }; if ($files) { $files | Select-Object FullName,@{Name='SizeMB';Expression={[math]::Round($_.Length/1MB,2)}} | Format-Table -AutoSize } else { Write-Host 'No non-ignored files larger than 95 MB were found.' -ForegroundColor Green }"

echo.
echo After copying .gitignore into this folder, run:
echo upload_to_github.bat
echo.
pause
