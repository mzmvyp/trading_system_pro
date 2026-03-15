# Código extraído: Backtesting (mzmvyp/smart_trading_system)

Origem: **https://github.com/mzmvyp/smart_trading_system** (pasta `backtesting/`).

## Arquivos

- **backtest_engine.py** – Engine de backtesting (sinais históricos, execução com slippage/fees, métricas).
- **reports.py** – Geração de relatórios HTML com gráficos (equity, drawdown, retornos, trades).
- **__init__.py** – Init do pacote (referência).

## Dependências do repositório original

- `..core.signal_generator`: SignalGenerator, MasterSignal, SignalStatus
- `..core.risk_manager`: RiskManager
- `..database.models`: DatabaseManager, BacktestResult
- `..core.market_data`: MarketDataProvider
- `utils.logger`, `utils.helpers` (em reports.py)

Para usar no **trading_system_pro**, é necessário:

1. Adaptar imports para os módulos locais (ou criar stubs/adapters).
2. O `reports.py` espera um tipo `BacktestResult` com campos como `total_return`, `sharpe_ratio`, `equity_curve`, `trade_history`, `monthly_returns`, `strategy_performance`. O engine retorna `BacktestResults` (dataclass); pode ser criado um mapeamento ou um BacktestResult local que seja preenchido a partir de BacktestResults.
3. A classe `HistoricalDataProvider` está definida no final de `backtest_engine.py` (versão completa neste folder) para uso na geração de sinais por período.

## Uso sugerido

- Copiar/adaptar `BacktestConfig`, `BacktestPosition`, `BacktestResults` e a lógica de simulação (slippage, SL/TP, equity, Sharpe, Sortino, drawdown) para o `trading_system_pro`.
- Integrar com o data provider de histórico (ex.: Binance) e com o gerador de sinais existente no projeto.
