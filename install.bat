@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
  echo Python Launcher was not found.
  echo Install Python 3.11 x64 from python.org and enable the Python Launcher.
  pause
  exit /b 1
)

py -3.11 -m venv .venv
if errorlevel 1 goto error
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
if errorlevel 1 goto error
python -m pip install -r requirements-base.txt
if errorlevel 1 goto error

echo.
echo Base installation completed successfully.
echo Start the application with run.bat.
pause
exit /b 0

:error
echo.
echo Installation failed. Copy the error text or take a screenshot.
pause
exit /b 1
