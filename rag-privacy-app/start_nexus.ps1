param(
    [switch]$InstallDeps
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "[Nexus] Project dir: $scriptDir" -ForegroundColor Cyan

$venvPython = Join-Path $scriptDir "..\..\.venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "[Nexus] .venv not found, creating virtual environment..." -ForegroundColor Yellow
    py -3 -m venv "..\..\.venv"
}

if (-not (Test-Path $venvPython)) {
    throw "[Nexus] Failed to locate venv python at: $venvPython"
}

if ($InstallDeps) {
    Write-Host "[Nexus] Installing dependencies..." -ForegroundColor Yellow
    & $venvPython -m pip install -r "requirements.txt"
}

# Free port 5000 if occupied
$conn = Get-NetTCPConnection -LocalPort 5000 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($conn) {
    Write-Host "[Nexus] Port 5000 occupied by PID $($conn.OwningProcess), stopping it..." -ForegroundColor Yellow
    Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
}

$env:PYTHONPATH = "."
$env:PYTHONUNBUFFERED = "1"

Write-Host "[Nexus] Starting server on http://127.0.0.1:5000" -ForegroundColor Green
& $venvPython -c "from app.app import create_app; app=create_app(); app.run(debug=True, port=5000, use_reloader=False)"
