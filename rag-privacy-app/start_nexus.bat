@echo off
setlocal

cd /d %~dp0
echo [Nexus] Project dir: %cd%

set "VENV_PYTHON=%~dp0..\.venv\Scripts\python.exe"

if not exist "%VENV_PYTHON%" (
	echo [Nexus] .venv not found, creating virtual environment...
	py -3 -m venv "%~dp0..\.venv"
)

if not exist "%VENV_PYTHON%" (
	echo [Nexus] Failed to locate venv python at: "%VENV_PYTHON%"
	exit /b 1
)

if /I "%1"=="--install-deps" (
	echo [Nexus] Installing dependencies...
	"%VENV_PYTHON%" -m pip install -r requirements.txt
)

for /f "tokens=5" %%p in ('netstat -ano ^| findstr /R /C:":5000 .*LISTENING"') do (
	echo [Nexus] Port 5000 occupied by PID %%p, stopping it...
	taskkill /PID %%p /F >nul 2>&1
)

set "PYTHONPATH=."
set "PYTHONUNBUFFERED=1"
set "HF_ENDPOINT=https://hf-mirror.com"
set "LLM_URL=http://100.92.149.102:8080/v1/chat/completions"

echo [Nexus] Starting server on http://127.0.0.1:5000
"%VENV_PYTHON%" -c "from app.app import create_app; app=create_app(); app.run(debug=True, port=5000, use_reloader=False)"

endlocal
