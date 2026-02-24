@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

echo [Microscopy Thumbnail Viewer] Starting installation...

if not exist "requirements.txt" (
    echo [Microscopy Thumbnail Viewer] ERROR: requirements.txt not found in project root.
    exit /b 1
)

set "PYTHON_CMD="
for %%V in (3.12 3.11 3.10 3.13) do (
    py -%%V --version >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_CMD=py -%%V"
        goto :python_selected
    )
)

where py >nul 2>nul
if not errorlevel 1 set "PYTHON_CMD=py -3"

if not defined PYTHON_CMD (
    where python >nul 2>nul
    if not errorlevel 1 set "PYTHON_CMD=python"
)

:python_selected
if not defined PYTHON_CMD (
    echo [Microscopy Thumbnail Viewer] ERROR: Python 3.10+ is required and was not found.
    exit /b 1
)

set "PY_VER="
for /f "tokens=2 delims= " %%V in ('%PYTHON_CMD% --version 2^>nul') do set "PY_VER=%%V"
for /f "tokens=1,2 delims=." %%A in ("%PY_VER%") do set "PY_MM=%%A.%%B"

echo [Microscopy Thumbnail Viewer] Using %PYTHON_CMD% (%PY_VER%)

if exist ".venv\Scripts\python.exe" (
    set "VENV_VER="
    for /f "tokens=2 delims= " %%V in ('".venv\Scripts\python.exe" --version 2^>nul') do set "VENV_VER=%%V"
    set "VENV_MM="
    for /f "tokens=1,2 delims=." %%A in ("!VENV_VER!") do set "VENV_MM=%%A.%%B"
    if not "!VENV_MM!"=="%PY_MM%" (
        echo [Microscopy Thumbnail Viewer] Existing .venv uses Python !VENV_VER!, recreating...
        rmdir /s /q ".venv"
        if errorlevel 1 (
            echo [Microscopy Thumbnail Viewer] ERROR: Could not remove existing .venv
            exit /b 1
        )
    )
)

if not exist ".venv\Scripts\python.exe" (
    echo [Microscopy Thumbnail Viewer] Creating virtual environment...
    %PYTHON_CMD% -m venv .venv
    if errorlevel 1 (
        echo [Microscopy Thumbnail Viewer] ERROR: Failed to create virtual environment.
        exit /b 1
    )
)

echo [Microscopy Thumbnail Viewer] Upgrading pip tooling...
".venv\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo [Microscopy Thumbnail Viewer] ERROR: Failed to upgrade pip tooling.
    exit /b 1
)

echo [Microscopy Thumbnail Viewer] Installing dependencies from requirements.txt...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [Microscopy Thumbnail Viewer] ERROR: Dependency installation failed.
    exit /b 1
)

echo [Microscopy Thumbnail Viewer] Verifying required imports...
".venv\Scripts\python.exe" -c "import PyQt6, qdarktheme, bioio, numpy, PIL, diskcache"
if errorlevel 1 (
    echo [Microscopy Thumbnail Viewer] ERROR: Verification failed for required packages.
    exit /b 1
)

echo [Microscopy Thumbnail Viewer] Verifying microscopy plugins...
".venv\Scripts\python.exe" -c "import bioio_czi, bioio_nd2, bioio_ome_tiff" >nul 2>nul
if errorlevel 1 (
    echo [Microscopy Thumbnail Viewer] WARNING: One or more bioio plugins failed to import.
    echo [Microscopy Thumbnail Viewer] The app can still run, but some proprietary formats may be unavailable.
) else (
    echo [Microscopy Thumbnail Viewer] BioIO plugins available.
)

echo [Microscopy Thumbnail Viewer] Installation complete.
exit /b 0
