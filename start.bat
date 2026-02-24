@echo off
setlocal EnableExtensions

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [Microscopy Thumbnail Viewer] Virtual environment not found.
    echo [Microscopy Thumbnail Viewer] Running install.bat...
    call install.bat
    if errorlevel 1 (
        echo [Microscopy Thumbnail Viewer] ERROR: Installation failed. Cannot start.
        exit /b 1
    )
)

".venv\Scripts\python.exe" -c "import PyQt6, qdarktheme, bioio, numpy, PIL, diskcache" >nul 2>nul
if errorlevel 1 (
    echo [Microscopy Thumbnail Viewer] Required packages are missing or broken.
    echo [Microscopy Thumbnail Viewer] Running install.bat to repair...
    call install.bat
    if errorlevel 1 (
        echo [Microscopy Thumbnail Viewer] ERROR: Installation failed. Cannot start.
        exit /b 1
    )
)

echo [Microscopy Thumbnail Viewer] Launching application...
".venv\Scripts\python.exe" main.py
set "EXIT_CODE=%errorlevel%"

if not "%EXIT_CODE%"=="0" (
    echo [Microscopy Thumbnail Viewer] Application exited with code %EXIT_CODE%.
)

exit /b %EXIT_CODE%
