# Deploy: sobe o sistema com docker compose (bot + dashboard + ml_dashboard)
# Uso: .\scripts\deploy.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

if (-not (Test-Path .env)) {
    Write-Host "[ERRO] Arquivo .env nao encontrado." -ForegroundColor Red
    Write-Host "Copie .env.example para .env e preencha as chaves (DEEPSEEK, BINANCE, etc.)."
    exit 1
}

Write-Host "[DEPLOY] Build e subida dos servicos..." -ForegroundColor Cyan
docker compose build
docker compose down --remove-orphans 2>$null
docker compose up -d

Write-Host "[DEPLOY] Servicos no ar. Logs (Ctrl+C para sair):" -ForegroundColor Green
docker compose logs -f
