# Conexões: DeepSeek, Confluência, ML, LSTM e Optimizer

## Visão geral do fluxo (agent.py)

```
1. SINAL DEEPSEEK (opcional)
   └─ Se ACCEPT_DEEPSEEK_SIGNALS=true → get_deepseek_analysis() → sinal direto
   └─ Execução própria (trend filter, risk, paper/real)

2. SINAL AGNO (principal)
   └─ Coleta dados local (indicadores, MTF, order flow)
   └─ Uma chamada DeepSeek com prompt preparado → LLM devolve BUY/SELL/NO_SIGNAL
   └─ CONFLUÊNCIA (votos):
        • Técnico: RSI, MACD, EMA/trend, ADX, BB, volume, MTF, CVD
          (thresholds vêm do best_config do Optimizer quando existe)
        • LLM: 1 voto (peso 1 ou 0.5 conforme confidence)
        • Bi-LSTM: 1 voto a favor se prob > 0.6, 1 contra se prob < 0.4
   └─ Exige: total_for >= 4 e combined_score >= 0.55 senão vira NO_SIGNAL
   └─ Validação ML (SimpleSignalValidator): observador; só bloqueia se ML_VALIDATION_REQUIRED=true
   └─ Trend filter + risk → execução paper/real
```

## O que está conectado

| Componente | Onde | Função |
|-----------|------|--------|
| **DeepSeek** | `get_deepseek_analysis`, prompt único AGNO | Gera sinal LLM (BUY/SELL/NO_SIGNAL) e confidence. |
| **Optimizer (best_config)** | `_calculate_technical_confluence` → `load_best_config(symbol, "1h")` | Fornece RSI oversold/overbought, ADX mínimo, volume surge para os votos técnicos. |
| **Confluência técnica** | `_calculate_technical_confluence()` | RSI, MACD, trend, ADX, BB, volume, MTF, CVD → votes_for / votes_against. |
| **LSTM (Bi-LSTM)** | Bloco `lstm_sequence_validator` no fluxo AGNO | Busca candles 1h, roda `predict_from_candles()`; prob > 0.6 = +1 voto, < 0.4 = +1 contra. |
| **ML (SimpleSignalValidator)** | `_validate_with_ml_model(agno_signal)` | Prediz sucesso/falha e probabilidade; por padrão só observa (não bloqueia) a menos que `ml_required=True`. |

## “Simple signal”

- **SimpleSignalValidator** = validador ML (sklearn) que prevê se o sinal deve ser executado ou não. Não é um gerador de sinal “simples” sem DeepSeek.
- Não existe hoje um fluxo que gera sinal **só** com indicadores técnicos, sem chamar o DeepSeek. Os dois caminhos (sinal DeepSeek direto e sinal AGNO) usam o LLM.

## Configuração (.env)

- `ACCEPT_DEEPSEEK_SIGNALS` – habilita sinal direto DeepSeek.
- `ACCEPT_AGNO_SIGNALS` – habilita sinal AGNO (confluência + LLM + LSTM).
- `ML_VALIDATION_ENABLED` – usa o ML para registrar previsão.
- `ML_VALIDATION_REQUIRED` – se true, bloqueia sinal quando ML discorda (threshold em `ML_VALIDATION_THRESHOLD`).

## Correção feita

- Na confluência técnica, os thresholds do `best_config` (BacktestParams) usam `adx_min_strength` e `volume_surge_multiplier`; o agent estava referenciando `adx_threshold` e `volume_spike_threshold`. Ajustado para usar os atributos corretos (com fallback).
