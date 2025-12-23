@echo off
echo.
echo ==================================================
echo       Starting local OSRM server (Volga Federal District)
echo ==================================================
echo.
echo Server will be available at: http://127.0.0.1:5000
echo Press Ctrl+C to stop the server
echo.

cd /d "%~dp0"

set DATA_PREFIX=volga-fed-district-251222.osrm

echo Looking for OSRM MLD data...
echo Current directory: %cd%
echo.

REM Проверяем ключевые файлы MLD
if not exist "%DATA_PREFIX%.partition" (
    echo ERROR: MLD partition file not found: %DATA_PREFIX%.partition
    echo Data is not prepared or path is incorrect.
    pause
    exit /b 1
)

if not exist "%DATA_PREFIX%.cells" (
    echo ERROR: MLD cells file not found: %DATA_PREFIX%.cells
    echo Data is not prepared or path is incorrect.
    pause
    exit /b 1
)

echo OSRM MLD data found.
echo.
echo Starting OSRM server...
echo.

docker run -t -i ^
  -p 5000:5000 ^
  -v "%cd%":/data ^
  ghcr.io/project-osrm/osrm-backend ^
  osrm-routed --algorithm mld /data/%DATA_PREFIX%

echo.
echo Server stopped.
pause
