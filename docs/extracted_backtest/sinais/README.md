# Código extraído: mzmvyp/sinais (backtesting)

**Origem:** repositório **privado** `mzmvyp/sinais`, obtido via `gh repo clone` (GitHub CLI com sua conta).

## Arquivos copiados

| Arquivo | Descrição |
|---------|-----------|
| **optimization_engine.py** | Engine de otimização: grid/random de RSI, MACD, Bollinger, confidence; score 30% win rate + 30% return + 20% Sharpe + 20% drawdown; otimização contínua em thread; simulação de trades e salvamento da melhor config. |
| **backtest_engine.py** | Motor de backtest do sinais. |
| **data_analyzer.py** | Análise de dados para backtest. |
| **__init__.py** | Init do pacote backtesting. |

## Dependências (projeto sinais)

- `core.data_reader.DataReader` (SQLite: `crypto_ohlc`)
- `indicators.technical.TechnicalAnalyzer`
- `ml.ml_integration.create_ml_enhancer`
- `llm.llm_integration.create_llm_enhancer`
- `config.settings.settings`

Para usar no **trading_system_pro**: adaptar esses imports para os módulos locais (data reader Binance, indicadores em `src/analysis` ou `src/ml`, config em `src.core.config`).

## Score composto (optimization_engine.py)

```python
score = (
    win_rate * 0.3 +
    (avg_return / 100) * 0.3 +
    sharpe_ratio * 0.2 +
    (1 - max_drawdown / 100) * 0.2
)
```

## Uso no sinais

- `OptimizationEngine().run_optimization(symbol, timeframe, max_configs, recent_days)`
- `run_optimization_with_progress(..., progress_callback=...)` para dashboard
- `start_continuous_optimization(symbol, timeframe, interval_hours)` para loop em background
- Melhor config salva em `backtesting/optimization/best_config_{symbol}_{timeframe}.json`
