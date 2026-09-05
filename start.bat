@echo off
echo =======================================================
echo RETURNGUARD: Return-Risk Scorer Launch
echo =======================================================
echo.

echo [1/3] Checking Python Virtual Environment...
if not exist "venv\Scripts\activate.bat" (
    echo Virtual environment not found. Creating 'venv'...
    python -m venv venv

    echo Activating venv and installing dependencies...
    call venv\Scripts\activate.bat
    pip install -r requirements.txt
) else (
    echo Virtual environment found. Skipping dependency installation for speed.
)

echo [2/3] Booting FastAPI Backend Engine...
start cmd /k "title ReturnGuard API && call venv\Scripts\activate.bat && uvicorn serve:app --reload --port 8000"

echo Waiting for backend to finish starting up...

:: Health-check poll against serve.py's /health endpoint, instead of a
:: fixed sleep -- actually knows when uvicorn is ready rather than guessing.
where curl >nul 2>&1
if errorlevel 1 (
    echo curl not found -- falling back to a fixed 5 second wait instead of health-checking.
    timeout /t 5 /nobreak >nul
    goto API_READY
)

set /a ATTEMPTS=0
:WAIT_FOR_API
set /a ATTEMPTS+=1
curl -s -o nul -w "%%{http_code}" http://127.0.0.1:8000/health > "%TEMP%\rg_status.txt" 2>nul
set /p STATUS=<"%TEMP%\rg_status.txt"
del "%TEMP%\rg_status.txt" >nul 2>&1
if "%STATUS%"=="200" (
    echo Backend is up! ^(after %ATTEMPTS% check^(s^)^)
    goto API_READY
)
if %ATTEMPTS% GEQ 30 (
    echo WARNING: Backend did not respond after 30 seconds.
    echo Launching frontend anyway -- check the "ReturnGuard API" window for errors.
    goto API_READY
)
timeout /t 1 /nobreak >nul
goto WAIT_FOR_API

:API_READY
echo [3/3] Booting Streamlit Demo UI...
start cmd /k "title ReturnGuard Demo && call venv\Scripts\activate.bat && streamlit run app.py"

echo.
echo System Boot Sequence Complete.
echo Backend API running on: http://127.0.0.1:8000
echo Streamlit Demo launching in your browser...
pause