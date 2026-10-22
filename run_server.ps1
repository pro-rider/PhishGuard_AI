# =====================================================================
# PhishGuard_AI - local dev server launcher (insecure dev mode)
#
# The application is SECURE (fail-closed) by default: open mode is refused
# unless DEV_INSECURE=true. This launcher explicitly opts into insecure
# local development so the operator dashboard works with no API key.
# In production: leave DEV_INSECURE unset, configure AUTH_USERS/API_KEYS
# and a real SECRET_KEY, and never run with --reload.
#
# Usage:  .\run_server.ps1            (default port 8000)
#         .\run_server.ps1 -Port 9000
# =====================================================================
param([int]$Port = 8000)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $root

# Explicitly opt into insecure local development (localhost-only, open mode).
$env:DEV_INSECURE = 'true'
# Clear any leftover API key so the server starts in open mode.
Remove-Item Env:\API_KEYS -ErrorAction SilentlyContinue
$env:API_KEYS = $null

$venvPy = Join-Path $root 'venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPy)) {
    Write-Host "venv python not found: $venvPy" -ForegroundColor Red
    Pop-Location
    exit 1
}

Write-Host ""
Write-Host "  PhishGuard_AI API  (DEV_INSECURE open mode, localhost only)" -ForegroundColor Cyan
Write-Host "  Dashboard : http://127.0.0.1:$Port/" -ForegroundColor Green
Write-Host "  API docs  : http://127.0.0.1:$Port/docs" -ForegroundColor Green
Write-Host "  (Ctrl+C to stop)" -ForegroundColor DarkGray
Write-Host ""

& $venvPy -m uvicorn api_gateway.main:app --reload --host 127.0.0.1 --port $Port

Pop-Location
