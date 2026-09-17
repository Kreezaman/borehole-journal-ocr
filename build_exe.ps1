$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

if (-not (Test-Path -LiteralPath ".venv\Scripts\python.exe")) {
    throw "Run install.bat first. The .venv directory was not found."
}

& ".venv\Scripts\python.exe" -m pip install --upgrade pyinstaller
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller installation failed with exit code $LASTEXITCODE."
}

$arguments = @(
    "--noconfirm",
    "--clean",
    "--windowed",
    "--onedir",
    "--name", "BoreholeJournalOCR",
    "--add-data", "app/assets;app/assets",
    "--collect-all", "pymupdf",
    "--collect-all", "google.genai",
    "--collect-submodules", "keyring",
    "--exclude-module", "paddleocr",
    "--exclude-module", "paddlex",
    "--exclude-module", "paddle",
    "main.py"
)

& ".venv\Scripts\pyinstaller.exe" @arguments
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE."
}

Write-Host "Build completed successfully." -ForegroundColor Green
Write-Host "EXE: dist\BoreholeJournalOCR\BoreholeJournalOCR.exe" -ForegroundColor Green
