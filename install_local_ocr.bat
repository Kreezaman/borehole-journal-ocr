@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Run install.bat first.
  pause
  exit /b 1
)

call ".venv\Scripts\activate.bat"
python -m pip install -r requirements-local.txt
if errorlevel 1 goto error
python -m app.ocr.prepare_models
if errorlevel 1 goto error

echo.
echo Local OCR was installed successfully.
echo The Russian OCR models are ready.
pause
exit /b 0

:error
echo.
echo Local OCR installation failed. Copy the error text or take a screenshot.
pause
exit /b 1
