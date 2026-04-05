@echo off
setlocal
cd /d %~dp0
powershell -ExecutionPolicy Bypass -File "%~dp0start_nexus.ps1"
endlocal
