# FASE 4 — ML e Sentiment (sinais)

**Branch:** feature/consolidation

## 1. Comparação ML existente vs sinais

| Módulo | trading_system_pro | sinais | Ação |
|--------|-------------------|--------|------|
| Modelos | simple_signal_validator (RF, GB, MLP, LogReg), lstm_signal_validator (LSTM) | XGBoost predictor, model_trainer | Mantido existente; XGBoost pode ser adicionado depois |
| Features | generate_dataset, ml_online_learning (indicadores no DataFrame) | feature_engineering (DataReader + TechnicalAnalyzer) | Mantido existente; feature_engineering do sinais depende de DataReader próprio |
| Optimization | Não existia | optimization_engine (grid search, SQLite) | Stub criado em ml/optimization/ para expansão futura |
| Validation | Validação em validators | walk-forward no backtest | Mantido existente |
| Sentiment LLM | Não existia | sentiment_analyzer (OpenAI GPT-4o-mini) | **Importado** em sentiment/llm_analyzer.py |
| News fetcher | Não existia | news_fetcher (CryptoCompare + mock) | **Importado** em sentiment/news_fetcher.py |

## 2. Estrutura criada

```
trading_system_pro/
├── ml/
│   ├── __init__.py
│   └── optimization/        # Stub para hyperparameter tuning
│       └── __init__.py
└── sentiment/
    ├── __init__.py
    ├── llm_analyzer.py      # LLMSentimentAnalyzer (OPENAI_API_KEY)
    ├── news_fetcher.py      # NewsFetcher (CRYPTOCOMPARE_API_KEY opcional)
    └── sentiment_signals.py # get_sentiment_feature() para uso como feature
```

## 3. Adaptações

- **Variáveis de ambiente:** OPENAI_API_KEY e CRYPTOCOMPARE_API_KEY adicionadas ao .env.example.
- **Logger:** Todos os módulos usam `from logger import get_logger`.
- **OpenAI:** Uso de `openai` 1.x (OpenAI(api_key=...) e client.chat.completions.create). Se o projeto usar outra versão, ajustar em llm_analyzer.py.
- **Sentiment como feature:** sentiment_signals.get_sentiment_feature(symbol, result) retorna valor normalizado para uso nos modelos existentes.

## 4. Testes

- Não há pasta tests/; verificação manual: `from sentiment import NewsFetcher, LLMSentimentAnalyzer` (LLMSentimentAnalyzer exige OPENAI_API_KEY e pip install openai).
- Modelos ML existentes (simple_signal_validator, lstm_signal_validator) não foram alterados.

## 5. Pendente

- Portar optimization_engine completo quando houver DataReader/indicadores unificados.
- Opcional: adicionar XGBoost ao pipeline de validação (sinais tem optimized_xgboost_predictor).
- Testes unitários para sentiment e integração com agno_tools.
