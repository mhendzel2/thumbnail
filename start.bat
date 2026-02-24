@echo off
setlocal EnableExtensions

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [BioArena] Virtual environment not found.
    echo [BioArena] Running install.bat first...
    call install.bat
    if errorlevel 1 (
        echo [BioArena] ERROR: Installation failed. Cannot start GUI.
        exit /b 1
    )
)

".venv\Scripts\python.exe" -c "import numpy as np, PyQt6, h5py, zarr, diskcache, readlif, bioformats_jar, sys; sys.exit(0 if int(np.__version__.split('.')[0]) < 2 else 1)" >nul 2>nul
if errorlevel 1 (
    echo [BioArena] Required packages are missing or incompatible.
    echo [BioArena] Running install.bat to repair environment...
    call install.bat
    if errorlevel 1 (
        echo [BioArena] ERROR: Installation failed. Cannot start GUI.
        exit /b 1
    )
)

echo [BioArena] Starting GUI...
".venv\Scripts\python.exe" main.py
set "EXIT_CODE=%errorlevel%"

if not "%EXIT_CODE%"=="0" (
    echo [BioArena] GUI exited with code %EXIT_CODE%.
)

exit /b %EXIT_CODE%
