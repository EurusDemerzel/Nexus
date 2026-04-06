param(
    [switch]$InstallDeps
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "[Nexus] start_nexus.ps1 is deprecated. Use start_nexus.bat as the single startup entry." -ForegroundColor Yellow

$batPath = Join-Path $scriptDir "start_nexus.bat"
if (-not (Test-Path $batPath)) {
    throw "[Nexus] Missing startup script: $batPath"
}

if ($InstallDeps) {
    & $batPath "--install-deps"
} else {
    & $batPath
}
