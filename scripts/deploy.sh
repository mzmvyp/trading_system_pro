#!/bin/bash
# Deploy: sobe o sistema com docker compose (bot + dashboard + ml_dashboard)
# Uso: ./scripts/deploy.sh   ou   bash scripts/deploy.sh

set -e
cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
  echo "[ERRO] Arquivo .env nao encontrado."
  echo "Copie .env.example para .env e preencha as chaves (DEEPSEEK, BINANCE, etc.)."
  exit 1
fi

echo "[DEPLOY] Build e subida dos servicos..."
docker compose build
docker compose down --remove-orphans 2>/dev/null || true
docker compose up -d

echo "[DEPLOY] Servicos no ar. Logs (Ctrl+C para sair):"
docker compose logs -f
