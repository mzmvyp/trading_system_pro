# 🚀 Sistema de Trading Automatizado

Sistema completo de trading com análise técnica, machine learning e sentiment analysis integrados.

## ⚡ Início Rápido

### 1. Instalar Dependências
```bash
pip install -r requirements.txt
```

### 2. Iniciar Sistema
```bash
# Opção 1: Inicialização automática
python start.py

# Opção 2: Manual
python start_data_collection.py
python main.py --continuous
streamlit run dashboard/streamlit_dashboard.py
```

## 📋 Scripts Principais

- `start.py` - Inicialização automática
- `start_data_collection.py` - Coleta dados em tempo real
- `main.py` - Análise de sinais contínua
- `dashboard/streamlit_dashboard.py` - Dashboard web
- `run_system.py` - Script helper com comandos

## 🏗️ Estrutura

```
sinais/
├── core/            # Sistema de análise
├── ml/              # Machine Learning
├── llm/             # Sentiment Analysis
├── trading/         # Paper Trading
├── dashboard/       # Interface web
├── backtesting/     # Backtest e otimização
├── indicators/      # Indicadores técnicos
├── config/          # Configurações
└── data/            # Bancos de dados
```

## 📊 Features

- **Stream de Dados**: BTC 1h com retenção de 4 anos
- **Análise Técnica**: RSI, MACD, padrões candlestick
- **Machine Learning**: XGBoost com 80+ features
- **Sentiment Analysis**: LLM com notícias em tempo real
- **Paper Trading**: Simulação realista
- **Backtesting**: 1000+ testes históricos
- **Dashboard Interativo**: Progress bars e métricas em tempo real

## 🔧 Configuração

Edite `config/settings.py` para ajustar:
- Timeframes ativos
- Pesos ML/LLM/Técnico
- Parâmetros de indicadores
- Configurações de trading

## 💡 Sistema de Pesos

```
Confiança Final = Técnico × 0.55 + ML × 0.25 + LLM × 0.20
```

## 🚀 Próximos Passos

1. Execute `start_data_collection.py`
2. Aguarde coleta de dados (1000+ candles)
3. Execute `main.py --continuous`
4. Abra dashboard: http://localhost:8501
5. Monitore sinais em tempo real

---

**Sistema pronto para produção! 🎯**