# FASE 5 — Indicadores, Dashboard e Signal Reevaluator

**Branch:** feature/consolidation

## TAREFA A — Indicadores (agente_trade_futuros)

- **trading_system_pro:** Indicadores estavam em `agno_tools.py` (talib) e `constants.py`. Não havia módulo centralizado.
- **agente_trade_futuros:** `technical_indicators.py` com classe única usando biblioteca `ta`.
- **Ação:** Criado `indicators/technical.py` centralizando indicadores com **talib** (já usado no projeto), alinhado a `constants.py` (RSI_PERIOD, MACD_*, BOLLINGER_*, ATR_PERIOD, SMA_*). Métodos: `calculate_rsi`, `calculate_macd`, `calculate_bollinger_bands`, `calculate_sma`, `calculate_ema`, `calculate_atr`, `calculate_all_indicators`, `get_latest_indicators`.
- **regime_optimizer:** `agno_regime_optimizer.py` em agente_trade_futuros não foi portado; pode ser adicionado depois como opcional (config/AGNO específicos).

## TAREFA B — Dashboard (trader_monitor)

- **trading_system_pro:** Um único arquivo `streamlit_dashboard.py` (layout wide, Binance API, posições, P&L, gráficos).
- **trader_monitor:** App multi-rota (Flask/FastAPI?), websocket, multi-timeframe, notificações.
- **Ação:** Estrutura de pastas criada para migração futura:
  - `dashboard/app.py` — entry que chama `streamlit_dashboard.py`.
  - `dashboard/pages/` — overview, backtesting (placeholders).
  - `dashboard/components/` — charts, metrics (placeholders).
- **Streaming Binance em tempo real / multi-timeframe / notificações:** Não integrados neste passo; o dashboard atual já usa requests à API REST. Para streaming e notificações, portar depois de `trader_monitor` (websocket, notification_system).

## TAREFA C — Signal Reevaluator e Stop Adjuster (trade_bot_new)

- **trading_system_pro:** Tinha apenas config de reavaliação (`reevaluation_enabled`, `reevaluation_interval_hours`, etc.); não tinha módulos `signal_reevaluator` nem `stop_adjuster`.
- **Ação:** Copiados para a raiz do projeto:
  - `signal_reevaluator.py` — reavalia posições abertas (paper e real), usa DeepSeek para sugestões (HOLD, CLOSE, MOVE_STOP_BREAKEVEN, etc.), integra com `agno_tools` para dados de mercado e com `stop_adjuster` quando disponível.
  - `stop_adjuster.py` — ajuste de stop loss após TP1, consulta DeepSeek para novo nível de stop.
- **Cliente DeepSeek:** Adicionado `deepseek_client.py` (usa `agno` + `DEEPSEEK_API_KEY`) para uso por `stop_adjuster` e qualquer fluxo que precise de chat com DeepSeek.
- **Dependências:** `position_manager` não existe no projeto; o código já trata com `try/except` e usa apenas paper/real positions e Binance executor.

## Testes

- Não foi executada suíte de testes (não há pasta `tests/`). Verificação manual recomendada: import de `indicators`, `signal_reevaluator`, `stop_adjuster`, `deepseek_client` após configurar `.env`.

## Resumo

| Item | Status |
|------|--------|
| indicators/technical.py (talib) | Criado e centralizado |
| regime_optimizer | Não portado (TODO opcional) |
| dashboard/pages + components | Estrutura criada (placeholders) |
| Streaming / notificações | Pendente (trader_monitor) |
| signal_reevaluator.py | Copiado e funcional |
| stop_adjuster.py | Copiado e funcional |
| deepseek_client.py | Criado para suporte ao stop_adjuster |
