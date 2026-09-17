@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\pythonw.exe" (
  echo The application is not installed. Run install.bat first.
  pause
  exit /b 1
)

start "Borehole Journal OCR" ".venv\Scripts\pythonw.exe" "main.py"
exit /b 0
