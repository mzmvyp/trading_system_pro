# Prompt para Claude Opus 4 / Cursor – Acessar informações de backtest e optimization

Use o texto abaixo no Opus 4.6 (Claude Code) ou no Cursor para que o modelo use o que foi gerado no projeto.

---

## Prompt (copiar e colar)

```
Contexto: No repositório trading_system_pro foram geradas e commitadas as seguintes informações. Use-as como fonte de verdade para implementar ou refinar backtesting e optimization.

1) RELATÓRIO DE EXTRAÇÃO
- Leia o arquivo: docs/BACKTEST_EXTRACTION_REPORT.md
- Contém: status de todos os repos GitHub do usuário mzmvyp (públicos e privados), quais foram acessíveis, o que foi extraído e onde está cada arquivo.

2) CÓDIGO EXTRAÍDO (referência e integração)
- Pasta: docs/extracted_backtest/
- docs/extracted_backtest/README.md: visão geral e como usar o código extraído.
- docs/extracted_backtest/sinais/:
  - optimization_engine.py: engine completo do repo privado mzmvyp/sinais (otimização contínua, score 30% win rate + 30% return + 20% Sharpe + 20% drawdown, simulação de trades, RSI/MACD/Bollinger/confidence).
  - backtest_engine.py, data_analyzer.py, __init__.py: backtest e análise do sinais.
  - README.md: dependências do projeto sinais (core.data_reader, indicators.technical, ml.ml_integration, llm.llm_integration, config.settings) e como adaptar ao trading_system_pro.
- docs/extracted_backtest/smart_trading_system/:
  - reports.py: geração de relatórios HTML (equity, drawdown, trades).
  - __init__.py: init do pacote.
- O backtest_engine completo do smart_trading_system está referenciado no relatório; conceitos (BacktestConfig, slippage, métricas) estão descritos em docs/extracted_backtest/smart_trading_system/ e no BACKTEST_EXTRACTION_REPORT.

3) ENGINE LOCAL (já no src)
- src/backtesting/optimization_engine.py: engine de otimização integrado ao trading_system_pro (OptimizationParams, BacktestMetrics, run_optimization com dados Binance, walk_forward_windows, score composto, apply_best_params).
- src/backtesting/__init__.py: exporta OptimizationParams, BacktestMetrics, run_optimization, apply_best_params.

Tarefa que quero que você faça agora:
[ DESCREVA AQUI A TAREFA: ex. integrar o optimization_engine do sinais ao trading_system_pro substituindo o data_reader por Binance; ou implementar walk-forward no optimization_engine local; ou gerar relatório HTML usando reports.py do smart_trading_system com os resultados do run_optimization; ou outro. ]

Ao implementar, priorize:
- Reutilizar a lógica e o score do docs/extracted_backtest/sinais/optimization_engine.py onde fizer sentido.
- Usar dados históricos via Binance (src/exchange/client.get_historical_klines ou equivalente).
- Manter o score composto 30% win rate + 30% return + 20% Sharpe + 20% drawdown (ou ajustar com justificativa).
- Documentar onde cada conceito veio (sinais vs smart_trading_system vs engine local).
```

---

## Versão curta (só localizar as informações)

```
No projeto trading_system_pro, onde estão as informações de backtesting e optimization extraídas dos repos mzmvyp?

Resposta esperada: 
- Relatório: docs/BACKTEST_EXTRACTION_REPORT.md
- Código extraído: docs/extracted_backtest/ (sinais/ e smart_trading_system/)
- Engine integrado: src/backtesting/optimization_engine.py e __init__.py
```

---

Depois de colar o prompt, substitua a linha `[ DESCREVA AQUI A TAREFA: ... ]` pela tarefa concreta que você quer que o Opus/Claude faça com essas informações.
