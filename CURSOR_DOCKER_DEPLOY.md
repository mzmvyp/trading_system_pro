# Prompt para o Cursor — Deploy Docker + Limpeza Git

> **Objetivo:** Atualizar Docker para a nova estrutura, testar build, e limpar Git.

---

## INSTRUÇÕES PARA O CURSOR

Preciso que você faça 3 coisas nesta ordem:

### PARTE 1: Atualizar Docker para nova estrutura

O projeto cresceu significativamente na Phase 6. A estrutura atual é:

```
trading_system_pro/
├── src/
│   ├── core/           # config, constants, logger, exceptions
│   ├── exchange/       # binance client, executor, utils
│   ├── analysis/       # 12 módulos (indicators, confluence, divergence, etc.)
│   ├── ml/             # 7 módulos (lstm, xgboost, feature_eng, etc.)
│   ├── trading/        # 8 módulos (agent, position_manager, risk, etc.)
│   ├── strategies/     # 4 estratégias (breakout, swing, trend, mean_rev)
│   ├── filters/        # 4 filtros (volatility, time, market_condition, fundamental)
│   ├── services/       # 2 serviços (notification, backup)
│   ├── backtesting/    # preparado para optimization_engine
│   ├── prompts/        # deepseek_prompt
│   └── dashboard/      # streamlit app + ml_dashboard
├── tests/
├── main.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

#### Atualizar Dockerfile:

1. **Verificar** se o Dockerfile atual funciona com as novas dependências:
   - `xgboost>=2.0.0` (precisa de compilação C++)
   - `scipy>=1.11.4`
   - `ta>=0.10.2`
2. **Adicionar** diretório `data/` no container (para models, backups, trade_history):
   ```
   RUN mkdir -p signals logs paper_trades portfolio deepseek_logs data/models data/backups
   ```
3. **Copiar** os novos pacotes (`src/strategies/`, `src/filters/`, `src/services/`, `src/backtesting/`)
4. **Garantir** que TA-Lib C library está corretamente instalada (multi-stage build já existe)

#### Atualizar docker-compose.yml:

1. **Adicionar volume** para `data/`:
   ```yaml
   volumes:
     - ./data:/app/data
   ```
2. **Adicionar serviço** ml-dashboard (opcional):
   ```yaml
   ml-dashboard:
     build: .
     container_name: trading-ml-dashboard
     command: streamlit run src/dashboard/ml_dashboard.py --server.port=8502 --server.address=0.0.0.0
     ports:
       - "8502:8502"
   ```
3. **Verificar** que `env_file: .env` está em todos os serviços
4. **Adicionar** variáveis de notificação no `.env.example`:
   ```
   # Notifications (optional)
   TELEGRAM_BOT_TOKEN=
   TELEGRAM_CHAT_ID=
   DISCORD_WEBHOOK_URL=
   SLACK_WEBHOOK_URL=
   ```

#### Testar o build:

```bash
docker compose build --no-cache
docker compose up -d
docker compose logs -f --tail=50
```

Se der erro:
- Se `TA-Lib` falhar: verificar se `ta-lib-0.4.0-src.tar.gz` download está funcionando
- Se `xgboost` falhar: pode precisar `gcc` e `g++` no builder stage
- Se imports falharem: verificar `PYTHONPATH=/app` está setado

---

### PARTE 2: Limpeza de arquivos desnecessários

Verificar e remover do repositório (se existirem):

1. **Arquivos de prompt/extração que não são mais necessários:**
   - `CONSOLIDATION_PROMPT.md` — pode deletar (já foi usado)
   - `CURSOR_DOCKER_DEPLOY.md` — deletar depois de executar
   - Qualquer `REPOS_EXTRACTION.md` se existir

2. **Arquivos legacy/duplicados no root:**
   - Verificar se existem `.py` avulsos no root que deveriam estar em `src/`
   - Verificar se existem pastas duplicadas (ex: `strategies/` no root E em `src/strategies/`)
   - Verificar se existem `__pycache__/` commitados

3. **Verificar `.gitignore`** inclui:
   ```
   # Data files
   data/models/*.pkl
   data/backups/*.zip
   data/trade_history.json

   # Prompt files (temporary)
   CONSOLIDATION_PROMPT.md
   CURSOR_DOCKER_DEPLOY.md
   REPOS_EXTRACTION.md
   ```

---

### PARTE 3: Limpeza do Git

#### Branches para manter:
- `main` (ou `master`) — branch principal
- `claude/trading-system-restructure-Wk3a6` — branch de desenvolvimento atual

#### Tarefas Git:

1. **Merge para main/master:**
   ```bash
   git checkout main
   git merge claude/trading-system-restructure-Wk3a6
   git push origin main
   ```

2. **Deletar branches antigas** (se existirem):
   ```bash
   # Listar todas as branches
   git branch -a

   # Deletar branches remotas que não são mais necessárias
   # EXCETO main e claude/trading-system-restructure-Wk3a6
   # Exemplo:
   # git push origin --delete claude/review-trading-bot-QeflN
   ```

3. **Verificar** que não há PRs abertos pendentes:
   ```bash
   gh pr list --state open
   ```

4. **Tag de versão** (após merge):
   ```bash
   git tag -a v2.0.0 -m "Phase 6: Full consolidation from 5 repos - 60 modules, strategies, ML, filters, services"
   git push origin v2.0.0
   ```

---

### PARTE 4: Validação final

Após tudo feito, verificar:

```bash
# 1. Estrutura correta
find src/ -name "*.py" | wc -l  # Deve ser ~60

# 2. Docker funciona
docker compose build
docker compose up -d
docker compose ps  # Todos running

# 3. Git limpo
git status  # Nada pendente
git log --oneline -5  # Commits recentes

# 4. Imports funcionam dentro do container
docker compose exec bot python -c "
from src.strategies.breakout_strategy import BreakoutStrategy
from src.analysis.confluence_analyzer import ConfluenceAnalyzer
from src.ml.xgboost_predictor import XGBOOST_AVAILABLE
from src.trading.position_manager import PositionManager
from src.services.notification_service import NotificationService
from src.filters.volatility_filter import VolatilityFilter
print('All imports OK!')
print(f'XGBoost available: {XGBOOST_AVAILABLE}')
"
```

Se algum import falhar, provavelmente é porque:
- Falta `PYTHONPATH=/app` no Dockerfile (adicionar `ENV PYTHONPATH=/app`)
- Ou falta `__init__.py` em algum pacote (todos já devem existir)

---

## RESUMO DE COMANDOS

```bash
# Build e test
docker compose build --no-cache
docker compose up -d
docker compose logs bot --tail=20

# Limpeza git
git checkout main
git merge claude/trading-system-restructure-Wk3a6
git push origin main
git tag -a v2.0.0 -m "v2.0.0: Full 5-repo consolidation"
git push origin v2.0.0

# Cleanup
rm CONSOLIDATION_PROMPT.md CURSOR_DOCKER_DEPLOY.md
git add -A && git commit -m "chore: cleanup temporary prompt files"
git push origin main
```
