# =====================================================================
# PhishGuard_AI - local dev server launcher
#
# Runs the API in OPEN mode (no API key required) for local development.
# A leftover API_KEYS from an earlier session is cleared first so the
# auth guard stays in open mode and the operator dashboard just works.
#
# Usage:  .\run_server.ps1            (default port 8000)
#         .\run_server.ps1 -Port 9000
# =====================================================================
param([int]$Port = 8000)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $root

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
Write-Host "  PhishGuard_AI API  (open mode, no API key required)" -ForegroundColor Cyan
Write-Host "  Dashboard : http://127.0.0.1:$Port/" -ForegroundColor Green
Write-Host "  API docs  : http://127.0.0.1:$Port/docs" -ForegroundColor Green
Write-Host "  (Ctrl+C to stop)" -ForegroundColor DarkGray
Write-Host ""

& $venvPy -m uvicorn api_gateway.main:app --reload --host 127.0.0.1 --port $Port

Pop-Location
