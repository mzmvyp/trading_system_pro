# Sistema completo de backtest contínuo – repositório **sinais** (mzmvyp/sinais)

Este é o sistema que você descreveu: **3 partes em um único repo** – coleta de candles, bot/analisador e backtest contínuo com ranking de setups.

---

## Os 3 sistemas (tudo no repo **sinais**)

### 1. Coleta de candles (grava local em SQLite)
- **Arquivo:** `binance_data_collector.py` (raiz do repo)
- **Função:** Coleta dados reais da Binance (WebSocket + API), grava em **SQLite** (`data/crypto_stream.db`).
- **Tabelas:** `crypto_ohlc` (OHLCV por symbol/timeframe), `kline_microstructure_1m` (1m).
- **Símbolos:** BTCUSDT, ETHUSDT, ADAUSDT, DOTUSDT, LINKUSDT, UNIUSDT, SOLUSDT, MATICUSDT.
- **Timeframes:** 1h, 4h, 1d.
- **Como rodar:** executar `binance_data_collector.py` (precisa rodar de forma contínua para encher o banco).

### 2. Bot / analisador (sinais em tempo real)
- **Arquivos:** `main.py`, `core/analyzer.py`, `core/signal_manager.py`, `core/signal_monitor.py`, `core/data_reader.py`, etc.
- **Função:** Lê dados do SQLite via `DataReader`, analisa multi-timeframe, emite sinais, modo `--continuous` com timing inteligente.
- **Como rodar:** `python main.py --continuous` (usa os candles já coletados no SQLite).

### 3. Backtest contínuo + ranking de setups
- **Arquivos:**
  - `backtesting/optimization_engine.py` – **motor principal**: testa **setups aleatórios** (RSI, MACD, Bollinger, confidence_threshold), usa dados do SQLite (`crypto_ohlc`), **ranqueia por score** (30% win rate + 30% return + 20% Sharpe + 20% drawdown), roda **otimização contínua** em thread (`start_continuous_optimization` a cada X horas).
  - `backtesting/backtest_engine.py` – executa backtest com dados reais (um ou vários testes, com progresso).
  - `backtesting/data_analyzer.py` – análise de dados e estatísticas para backtest.
- **Período de teste:** Por padrão usa **últimos 7 dias** (`test_period_days = 7`, `recent_days`). Para 3–6 meses, altere `test_period_days` / `recent_days` no código ou na chamada (ex.: `recent_days=90` ou `180`).
- **Dados:** Sempre lidos do **DataReader** → SQLite (`crypto_ohlc`). Ou seja, o sistema 1 precisa ter rodado antes (ou ter histórico importado) para ter 3–6 meses de dados.
- **Como rodar:**
  - Uma vez: `python -m backtesting.optimization_engine --symbol BTCUSDT --timeframe 1h --configs 50 --days 7`
  - Contínuo: `python -m backtesting.optimization_engine --continuous --interval 6`
  - Ou pelo **dashboard:** `streamlit run dashboard/streamlit_dashboard.py` → aba de backtest/otimização → “Iniciar Otimização Contínua”.

---

## Onde está cada coisa (para passar ao Claude Code / Cursor)

| Parte              | O que é                          | Arquivos principais |
|--------------------|-----------------------------------|----------------------|
| Coleta candles    | Grava Binance → SQLite local     | `binance_data_collector.py` |
| Bot               | Analisa e emite sinais            | `main.py`, `core/analyzer.py`, `core/data_reader.py`, `core/signal_*.py` |
| Backtest + rank   | Testa setups, rankeia, contínuo   | `backtesting/optimization_engine.py`, `backtesting/backtest_engine.py`, `backtesting/data_analyzer.py` |
| Dados             | Leitura do banco                 | `core/data_reader.py` (DataReader, tabela `crypto_ohlc`) |
| Config            | Settings e paths                  | `config/settings.py` |
| Dashboard         | UI backtest + otimização contínua| `dashboard/streamlit_dashboard.py` |

---

## Dependências

- `config.settings` (settings.py)
- `core.data_reader.DataReader` (lê `crypto_ohlc`)
- `indicators.technical.TechnicalAnalyzer`
- `ml.ml_integration`, `llm.llm_integration` (usados pelo backtest/analyzer)
- Python: ver `requirements.txt` (ex.: pandas, websockets, binance, streamlit, etc.)

---

## Resumo para o “Code”

- O **sistema completo de backtest que roda continuamente** e testa **sinais/setups aleatórios** com **ranking por acertividade** (score composto) está no repositório **sinais**.
- Os “3 sistemas” (coleta, bot, backtest) estão **no mesmo repo**, nesta pasta extraída: `docs/extracted_backtest/sistema_backtest_sinais_completo/`.
- Período: no código padrão está **7 dias**; para 3–6 meses, aumentar `recent_days` / `test_period_days` e garantir que o coletor (ou um import) já tenha esse histórico no SQLite.

Use esta pasta como referência única para o Claude Code analisar ou integrar o backtest contínuo no `trading_system_pro`.
