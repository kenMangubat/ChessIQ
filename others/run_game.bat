@echo off
setlocal EnableExtensions

cd /d "%~dp0.."

set "PY_CMD="
where py >nul 2>&1
if %errorlevel%==0 (
    py -3 --version >nul 2>&1
    if %errorlevel%==0 set "PY_CMD=py -3"
)

if not defined PY_CMD (
    where python >nul 2>&1
    if %errorlevel%==0 (
        python --version >nul 2>&1
        if %errorlevel%==0 set "PY_CMD=python"
    )
)

if not defined PY_CMD (
    echo [ERROR] Python 3 was not found on this computer.
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    echo Make sure "Add Python to PATH" is checked during installation.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment in .venv ...
    %PY_CMD% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create .venv.
        pause
        exit /b 1
    )
)

echo Installing required packages from others\requirements.txt ...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r others\requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies from others\requirements.txt.
    echo Check your internet connection and try running others\run_game.bat again.
    pause
    exit /b 1
)

if not exist "engine\stockfish.exe" (
    echo Stockfish engine not found. Running automatic Stockfish setup ...
    ".venv\Scripts\python.exe" -m others.setup_stockfish
    if errorlevel 1 (
        echo [WARNING] Stockfish could not be installed automatically.
        echo Follow the manual / USB instructions above, or press any key to launch anyway.
        pause
    )
)

echo Launching Chess Ken ...
".venv\Scripts\python.exe" main.py
if errorlevel 1 (
    echo [ERROR] The game exited with an error.
    pause
    exit /b 1
)

endlocal
