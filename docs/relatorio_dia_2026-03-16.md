# Relatório do dia – 16/03/2026

**Fontes:** `change.log` (raiz do projeto) e `logs/src_trading_agent.log` (linhas do dia 2026-03-16). As estatísticas foram extraídas desses logs.

---

## 1. Visão geral

| Métrica | Valor |
|--------|--------|
| **Ciclos de análise (símbolos analisados)** | **250** |
| **Sinais executados (trade real)** | **1** (BTCUSDT BUY – 09:01:09) |
| **Sinais reprovados / não executados** | **76** (75 por ML + 1 por risco/retorno) |

---

## 2. DeepSeek (LLM) – Sinais diretos

| Tipo | Quantidade | % do total de respostas |
|------|------------|-------------------------|
| **NO_SIGNAL** (sinal fraco / sem direção) | **239** | **95,6%** |
| **BUY ou SELL** (sinal forte o suficiente para confluência) | **11** | **4,4%** |

- Ou seja: na grande maioria das análises o modelo **não** deu BUY/SELL; apenas 11 vezes passou para a etapa de confluência com sinal direto.
- As 11 ocorrências de BUY/SELL foram validadas por bias técnico (confiança ≥ 5) e entraram na confluência (indicadores + LSTM).

---

## 3. Confluência (indicadores + LSTM)

- **Bloqueios por confluência no dia:** **0**  
  (Não houve nenhum “CONFLUENCE BLOCK” ou “confluência insuficiente” em 16/03.)
- Todos os sinais BUY/SELL que chegaram à confluência **passaram** (score e votos suficientes).
- O **LSTM (Bi-LSTM)** participa como **voto na confluência** (FOR / NEUTRAL/AGAINST), não como etapa separada de “reprovado por LSTM”. Ou seja, não há contagem à parte de “reprovados só por LSTM”; ele influencia o score de confluência.
- **76 eventos** no dia tiveram “[CONFLUENCE] BUY/SELL … segue para ML/risco” (sinais que passaram da confluência e foram para ML e depois risco/retorno).

---

## 4. Reprovação por ML (modelo de classificação)

- **Total bloqueado por ML:** **75**
- **Condição:** `ml_required=True` e probabilidade de sucesso **abaixo do threshold** (65%).
- **Único sinal que passou no ML e foi executado:** BTCUSDT BUY (09:01), que teve prob ≥ 65% e seguiu para risco/retorno e execução.

### Estatísticas das probabilidades quando o ML bloqueou

| Estatística | Valor |
|-------------|--------|
| **Média da prob (quando bloqueou)** | **~43,5%** (aprox. dos 75 registros) |
| **Mínima** | **11,3%** (PAXGUSDT SELL) |
| **Máxima** | **64,7%** (XRPUSDT BUY – logo abaixo do 65%) |

### Distribuição por predição no bloqueio

- **predicao=FALHA:** maioria dos 75 bloqueios (modelo prevendo falha).
- **predicao=SUCESSO** mas **prob &lt; 65%:** vários casos (ex.: BTCUSDT 58.2%, 61.7%; DOGEUSDT 53.6%, 63.4%; DOTUSDT 57.5%, 54.3%; XRPUSDT 54.3%, 64.7%). Ou seja, o ML “concordava” com sucesso, mas a confiança ficou abaixo do mínimo e por isso bloqueou.

### Símbolos mais bloqueados por ML no dia (contagem aproximada)

| Símbolo | Direção | Ocorrências (aprox.) |
|---------|---------|----------------------|
| PAXGUSDT | SELL | 14 |
| DOTUSDT | BUY | 8 |
| SOLUSDT | BUY | 7 |
| XRPUSDT | BUY | 7 |
| ETHUSDT | BUY | 6 |
| BNBUSDT | BUY | 5 |
| AVAXUSDT | BUY | 6 |
| LINKUSDT | BUY | 5 |
| DOGEUSDT | BUY | 5 |
| BTCUSDT | BUY | 5 |
| ADAUSDT | BUY | 3 |

---

## 5. Reprovação por risco/retorno

- **Total reprovado por risco ou retorno:** **1**
- **Detalhe:**  
  **PAXGUSDT SELL** (00:04:38) – motivo: **Risk:Reward inadequado: 0.15:1 (mínimo 1.5:1). Risco=157.59, Reward(TP1)=23.90.**

Ou seja: um único sinal passou de confluência e ML mas foi barrado na etapa final por **risco/retorno** (R:R abaixo do mínimo configurado).

---

## 6. Resumo do funil do dia

```
250 análises (ciclos)
    │
    ├─ 239 → DeepSeek NO_SIGNAL (não gera BUY/SELL para confluência)
    │
    └─ 11 → DeepSeek BUY/SELL (+ outros que vêm de validação/confluência)
              │
              ▼
         76 sinais passaram da CONFLUÊNCIA (indicadores + LSTM)
              │
              ├─ 75 → BLOQUEADOS por ML (prob < 65%)
              │
              └─ 1  → PAXGUSDT reprovado por RISCO/RETORNO (R:R 0.15:1)
              │
              └─ 1  → BTCUSDT BUY EXECUTADO (09:01:09)
```

---

## 7. Conclusões rápidas

1. **Poucos sinais fortes do LLM:** 95,6% das respostas foram NO_SIGNAL; só 4,4% BUY/SELL.
2. **Confluência e LSTM:** Nenhum bloqueio por confluência no dia; LSTM atua dentro da confluência, não como filtro “reprovado só por LSTM”.
3. **ML foi o principal filtro:** 75 dos 76 sinais que passaram da confluência foram barrados pelo ML (prob < 65%).
4. **Risco/retorno:** Apenas 1 sinal (PAXGUSDT) foi barrado só por R:R inadequado.
5. **Execução:** Apenas **1 trade real** no dia (BTCUSDT BUY às 09:01).

Se quiser, posso sugerir um script que leia `src_trading_agent.log` e gere esse mesmo tipo de relatório (médias, contagens por etapa, por símbolo) para qualquer dia.
