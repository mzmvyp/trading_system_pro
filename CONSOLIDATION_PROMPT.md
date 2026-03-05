# Prompt para o Cursor — Extração de Código dos Repos para Consolidação

> **Objetivo:** Gerar um arquivo `REPOS_EXTRACTION.md` com TODO o código relevante
> dos 5 repositórios que serão mesclados no `trading_system_pro`.
> Este arquivo será usado pelo Claude Code para fazer o merge.

---

## INSTRUÇÕES PARA O CURSOR

Preciso que você analise os 5 repositórios abaixo e gere UM ÚNICO arquivo chamado
`REPOS_EXTRACTION.md` contendo todas as informações necessárias para consolidação.

**IMPORTANTE:**
- NÃO resuma o código — inclua o código COMPLETO de cada arquivo relevante
- NÃO inclua código que já existe no trading_system_pro (vou listar o que já temos)
- Para cada arquivo, indique EXATAMENTE onde ele deve ir na estrutura alvo
- Se um arquivo tem funcionalidade parcialmente duplicada, indique o que é NOVO vs DUPLICADO

---

## O QUE JÁ EXISTE NO trading_system_pro

O projeto destino já tem esta estrutura e estas funcionalidades:

```
trading_system_pro/
├── src/
│   ├── core/
│   │   ├── config.py          # Settings via pydantic-settings (.env)
│   │   ├── constants.py       # Trading constants (timeouts, limits, symbols)
│   │   ├── logger.py          # Centralized logging with rotation
│   │   └── exceptions.py      # Custom exception hierarchy
│   ├── exchange/
│   │   ├── client.py          # BinanceClient async (klines, ticker, orderbook, funding, OI)
│   │   ├── executor.py        # BinanceFuturesExecutor (market/limit/SL/TP orders)
│   │   └── utils.py           # RateLimiter, CircuitBreaker, exponential_backoff_retry
│   ├── analysis/
│   │   ├── indicators.py      # RSI, MACD, ADX, ATR, BB, SMA, EMA, OBV, Fibonacci, Volume Profile
│   │   ├── sentiment.py       # Market sentiment (funding rate, OI, price action based)
│   │   ├── market_data.py     # get_market_data (price, volume, funding, OI)
│   │   ├── order_flow.py      # Orderbook imbalance, CVD, buy/sell pressure
│   │   ├── multi_timeframe.py # 5m/15m/1h/4h/1d trend confluence
│   │   └── market_classifier.py # SCALP/DAY_TRADE/SWING/POSITION classifier
│   ├── ml/
│   │   ├── lstm_validator.py  # LSTM signal validator (TensorFlow/Keras)
│   │   ├── simple_validator.py # Ensemble: RandomForest + GradientBoosting + LogisticRegression
│   │   ├── online_learning.py # Continuous learning from trade results
│   │   └── dataset_generator.py # Generate training datasets from historical data
│   ├── trading/
│   │   ├── agent.py           # AgnoTradingAgent (AGNO + DeepSeek orchestration)
│   │   ├── signal_parser.py   # JSON extraction, regex price parsing from LLM responses
│   │   ├── risk_manager.py    # Risk validation, position sizing, circuit breakers
│   │   ├── paper_trading.py   # Full paper trading simulation with SL/TP monitoring
│   │   ├── portfolio.py       # Portfolio viewer/manager CLI
│   │   └── orphan_cleaner.py  # Clean orphan orders on Binance
│   ├── prompts/
│   │   └── deepseek_prompt.py # Prepare analysis for LLM, create prompts, call DeepSeek
│   └── dashboard/
│       ├── app.py             # Streamlit dashboard (positions, PnL, signals, Binance integration)
│       └── ml_dashboard.py    # ML model performance dashboard
├── tests/
├── main.py                    # Entry point (single, monitor, top5, top10 modes)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

### Indicadores que JÁ temos:
RSI, MACD (histogram + crossover), ADX, ATR, Bollinger Bands (position),
SMA (20/50/200), EMA (20/50/200), OBV (trend), Fibonacci levels,
Volume Profile (POC), Market Structure (HH/LL/trend)

### ML que JÁ temos:
- LSTM signal validator
- SimpleSignalValidator (RF + GB + LR ensemble)
- Online learning (incremental retrain)
- Dataset generator

### Funcionalidades que JÁ temos:
- Binance Futures execution (market, limit, SL, TP)
- Paper trading with position monitoring
- Risk management (confidence, drawdown, daily limits, position sizing)
- DeepSeek + AGNO agent integration
- Multi-timeframe analysis
- Market condition classification
- Streamlit dashboard

---

## REPOS PARA ANALISAR

### REPO 1: smart_trading_system
**Caminho:** c:\Users\Willian\python_projects\smart_trading_system
**O que extrair:**
- [ ] **TODAS as estratégias** em `strategies/` (trend_following, mean_reversion, breakout, swing)
- [ ] **TODOS os filtros** em `filters/` (volatility, time, market_condition, fundamental)
- [ ] **BaseStrategy** se existir (classe abstrata)
- [ ] **Performance analyzer** se for diferente do que já temos
- [ ] **Portfolio/risk manager** APENAS se tiver funcionalidade que o nosso não tem

**Para cada arquivo, responda:**
1. Nome do arquivo e caminho original
2. O código COMPLETO
3. Destino sugerido no trading_system_pro (ex: `src/strategies/trend_following.py`)
4. Dependências externas necessárias (pip packages)
5. O que é NOVO vs o que já temos

---

### REPO 2: sinais
**Caminho:** c:\Users\Willian\python_projects\sinais
**O que extrair:**
- [ ] **XGBoost predictor** + feature engineering pipeline
- [ ] **LLM sentiment analyzer** (se diferente do nosso DeepSeek)
- [ ] **News fetcher** (se existir)
- [ ] **Optimization engine**
- [ ] **Performance tracker** APENAS se melhor que o nosso

**Para cada arquivo, responda:**
1. Nome e caminho original
2. Código COMPLETO
3. Destino sugerido
4. Dependências externas
5. NOVO vs DUPLICADO — comparar especificamente com:
   - Nosso `src/ml/simple_validator.py` (usa RF+GB+LR)
   - Nosso `src/ml/dataset_generator.py`
   - Nosso `src/analysis/sentiment.py`

---

### REPO 3: trader_monitor
**Caminho:** c:\Users\Willian\python_projects\trader_monitor
**O que extrair:**
- [ ] **Estratégias** scalp, swing, day_trade (se diferentes das do smart_trading_system)
- [ ] **Dashboard streaming** com dados real-time Binance
- [ ] **Config manager + backup** (se diferente do nosso pydantic Settings)
- [ ] **Sistema de notificações** (se existir)
- [ ] **Multi-asset manager** (se diferente do nosso)

**Para cada arquivo, responda:**
1. Nome e caminho original
2. Código COMPLETO
3. Destino sugerido
4. Dependências externas
5. NOVO vs DUPLICADO — comparar com:
   - Nosso `src/dashboard/app.py`
   - Nosso `src/analysis/multi_timeframe.py`

---

### REPO 4: agente_trade_futuros
**Caminho:** c:\Users\Willian\python_projects\agente_trade_futuros
**O que extrair:**
- [ ] **technical_indicators.py** — comparar indicador por indicador com nosso `src/analysis/indicators.py`
- [ ] **Regime optimizer** (se existir)
- [ ] **Position sizing** APENAS se melhor que nosso risk_manager
- [ ] **Backtest engine** APENAS se melhor que nosso backtest_strategy

**Para cada arquivo, responda:**
1. Nome e caminho original
2. Código COMPLETO
3. Destino sugerido
4. Dependências externas
5. NOVO vs DUPLICADO — comparar indicador a indicador:
   - Nossos indicadores: RSI, MACD, ADX, ATR, BB, SMA, EMA, OBV, Fibonacci, Volume Profile
   - Listar APENAS indicadores que eles têm e nós NÃO temos
   - Ex: Stochastic? Ichimoku? VWAP? Williams %R?

---

### REPO 5: trade_bot_new
**Caminho:** c:\Users\Willian\python_projects\trade_bot_new
**O que extrair:**
- [ ] **signal_reevaluator** — código COMPLETO
- [ ] **stop_adjuster** (trailing stop, break-even) — código COMPLETO
- [ ] Qualquer outra funcionalidade ÚNICA que não existe nos outros repos

**Para cada arquivo, responda:**
1. Nome e caminho original
2. Código COMPLETO
3. Destino sugerido
4. Dependências externas
5. O que faz que nosso sistema ainda não faz

---

## FORMATO DO ARQUIVO DE SAÍDA

Gere o arquivo `REPOS_EXTRACTION.md` neste formato EXATO:

```markdown
# REPOS_EXTRACTION — Código para Consolidação no trading_system_pro

## Sumário de Extração

| Repo | Arquivos Extraídos | Novos Indicadores | Novas Features |
|------|--------------------|-------------------|----------------|
| smart_trading_system | X arquivos | ... | ... |
| sinais | X arquivos | ... | ... |
| trader_monitor | X arquivos | ... | ... |
| agente_trade_futuros | X arquivos | ... | ... |
| trade_bot_new | X arquivos | ... | ... |

## Dependências Novas Necessárias
- package1>=version (usado por: repo X, arquivo Y)
- package2>=version (usado por: repo X, arquivo Y)

---

## REPO 1: smart_trading_system

### Arquivo: strategies/trend_following.py
**Destino:** src/strategies/trend_following.py
**Status:** NOVO (não existe equivalente)
**Dependências:** [listar]

\```python
[CÓDIGO COMPLETO AQUI]
\```

### Arquivo: strategies/mean_reversion.py
**Destino:** src/strategies/mean_reversion.py
...

[Repetir para cada arquivo de cada repo]

---

## REPO 2: sinais
...

## REPO 3: trader_monitor
...

## REPO 4: agente_trade_futuros

### Comparação de Indicadores

| Indicador | trading_system_pro | agente_trade_futuros | Ação |
|-----------|-------------------|---------------------|------|
| RSI | ✅ src/analysis/indicators.py | ✅ technical_indicators.py | JÁ TEMOS |
| MACD | ✅ | ✅ | JÁ TEMOS |
| Stochastic | ❌ | ✅ | IMPORTAR |
| VWAP | ❌ | ✅ | IMPORTAR |
| ... | ... | ... | ... |

### Arquivo: technical_indicators.py (APENAS funções NOVAS)
...

## REPO 5: trade_bot_new
...
```

---

## REGRAS CRÍTICAS

1. **NÃO RESUMA** — Inclua código COMPLETO de cada arquivo
2. **NÃO DUPLIQUE** — Se algo já existe no trading_system_pro, diga "JÁ TEMOS" e pule
3. **COMPARE ANTES** — Para indicadores e ML, compare função a função
4. **INDIQUE CONFLITOS** — Se dois repos têm versões diferentes da mesma coisa, inclua AMBAS e indique qual é melhor
5. **INCLUA IMPORTS** — Cada bloco de código deve ter seus imports completos
6. **INDIQUE ADAPTAÇÕES** — Se o código precisa de mudanças para funcionar no trading_system_pro (ex: imports diferentes), indique EXATAMENTE o que mudar
7. O arquivo pode ser grande (50-100KB) — isso é esperado e necessário
8. Se um repo não tiver nada novo, diga "NADA A EXTRAIR" e explique por quê
