@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

echo [BioArena] Installing dependencies...

set "PYTHON_CMD="
for %%V in (3.12 3.11 3.10) do (
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
    echo [BioArena] ERROR: Python not found in PATH.
    echo [BioArena] Install Python 3.10+ and rerun install.bat.
    exit /b 1
)

set "PY_VER="
for /f "tokens=2 delims= " %%V in ('%PYTHON_CMD% --version 2^>nul') do set "PY_VER=%%V"
for /f "tokens=1,2 delims=." %%A in ("%PY_VER%") do set "PY_MM=%%A.%%B"

echo [BioArena] Using %PYTHON_CMD% (%PY_VER%)

if exist ".venv\Scripts\python.exe" (
    set "VENV_PY_VER="
    for /f "tokens=2 delims= " %%V in ('".venv\Scripts\python.exe" --version 2^>nul') do set "VENV_PY_VER=%%V"
    set "VENV_MM="
    for /f "tokens=1,2 delims=." %%A in ("!VENV_PY_VER!") do set "VENV_MM=%%A.%%B"
    if not "!VENV_MM!"=="%PY_MM%" (
        echo [BioArena] Existing .venv uses Python !VENV_PY_VER!, recreating with %PY_MM%...
        rmdir /s /q ".venv"
        if errorlevel 1 (
            echo [BioArena] ERROR: Failed to remove existing .venv
            exit /b 1
        )
    )
)

if not exist ".venv\Scripts\python.exe" (
    echo [BioArena] Creating virtual environment...
    %PYTHON_CMD% -m venv .venv
    if errorlevel 1 (
        echo [BioArena] ERROR: Failed to create .venv
        exit /b 1
    )
)

echo [BioArena] Upgrading pip tooling...
".venv\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo [BioArena] ERROR: Failed to upgrade pip tooling.
    exit /b 1
)

set "AICS_OPTIONAL=0"
if "%PY_MM%"=="3.13" set "AICS_OPTIONAL=1"
if "%PY_MM%"=="3.14" set "AICS_OPTIONAL=1"

if "%AICS_OPTIONAL%"=="1" (
    echo [BioArena] Python %PY_MM% detected. Installing core dependencies first...
    ".venv\Scripts\python.exe" -m pip install PyQt6 "numpy<2" h5py zarr diskcache "numcodecs<0.16" "readlif>=0.6.4" bioformats_jar
    if errorlevel 1 (
        echo [BioArena] ERROR: Core dependency installation failed.
        exit /b 1
    )

    echo [BioArena] Installing optional aicsimageio for Python %PY_MM%...
    ".venv\Scripts\python.exe" -m pip install aicsimageio
    if errorlevel 1 (
        echo [BioArena] WARNING: aicsimageio installation failed on Python %PY_MM%.
        echo [BioArena] TIFF/CZI/LIF support will be unavailable with this interpreter.
    )
) else (
    echo [BioArena] Installing requirements...
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [BioArena] ERROR: Dependency installation failed.
        exit /b 1
    )
)

echo [BioArena] Verifying core imports...
".venv\Scripts\python.exe" -c "import numpy, PyQt6, h5py, zarr, diskcache"
if errorlevel 1 (
    echo [BioArena] ERROR: Verification failed. Core packages are missing.
    exit /b 1
)

".venv\Scripts\python.exe" -c "import aicsimageio" >nul 2>nul
if errorlevel 1 (
    echo [BioArena] WARNING: aicsimageio is unavailable.
    echo [BioArena] TIFF/CZI/LIF readers will be disabled.
) else (
    echo [BioArena] aicsimageio available.
)

".venv\Scripts\python.exe" -c "import readlif" >nul 2>nul
if errorlevel 1 (
    echo [BioArena] WARNING: readlif is unavailable.
    echo [BioArena] Native LIF reader support is disabled.
) else (
    echo [BioArena] readlif available.
)

".venv\Scripts\python.exe" -c "import bioformats_jar, jpype" >nul 2>nul
if errorlevel 1 (
    echo [BioArena] WARNING: bioformats_jar is unavailable.
    echo [BioArena] Bio-Formats fallback for LIF and CZI is disabled.
) else (
    echo [BioArena] bioformats_jar available.
    where java >nul 2>nul
    if errorlevel 1 (
        echo [BioArena] WARNING: java executable not found in PATH.
        echo [BioArena] Bio-Formats runtime may fail without Java.
    )
    where mvn >nul 2>nul
    if errorlevel 1 (
        echo [BioArena] WARNING: maven executable not found in PATH.
        echo [BioArena] Bio-Formats runtime may fail without Maven.
    )
)

echo [BioArena] Install complete.
exit /b 0
