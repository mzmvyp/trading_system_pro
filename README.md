# 🤖 Sistema de Trading com AGNO + DeepSeek

Sistema de trading automatizado que usa **AGNO Agent** com **DeepSeek** para orquestrar análises de mercado, indicadores técnicos, sentimento e IA para gerar sinais de trading precisos.

## 🚀 Instalação e Configuração

### 1. Instalar Dependências
```bash
pip install -r requirements.txt
```

### 2. Configurar DeepSeek API Key
1. Obtenha sua API key em: https://platform.deepseek.com/
2. Crie arquivo `.env` com:
```env
DEEPSEEK_API_KEY=sua_chave_aqui
TRADING_SYMBOL=BTCUSDT
LOG_LEVEL=INFO
```

### 3. Testar Sistema
```bash
python test_agno.py
```

## 🎯 Como Usar

### Análise Única
```bash
python main.py --symbol BTCUSDT --mode single
```

### Monitoramento Contínuo
```bash
python main.py --symbol BTCUSDT --mode monitor --interval 300
```

### Top 10 Criptomoedas
```bash
python main.py --mode top10
```

## 🧠 Como Funciona

O **AGNO Agent** com **DeepSeek** orquestra todo o processo:

1. **Coleta de Dados** → `get_market_data()`
2. **Análise Técnica** → `analyze_technical_indicators()`
3. **Sentimento** → `analyze_market_sentiment()`
4. **IA Avançada** → `get_deepseek_analysis()`
5. **Validação de Risco** → `validate_risk_and_position()`
6. **Execução** → `execute_paper_trade()`

## 📁 Estrutura do Projeto

```
agent_trade/
├── agno_tools.py              # Ferramentas para o AGNO
├── trading_agent_agno.py      # Agent principal com DeepSeek
├── main.py                    # Script principal
├── test_agno.py              # Testes
├── signals/                   # Sinais gerados
├── paper_trades/             # Trades simulados
└── logs/                     # Logs do sistema
```

## 📊 Sinais Gerados

O sistema gera sinais estruturados em JSON:
```json
{
  "symbol": "BTCUSDT",
  "signal": "BUY",
  "confidence": 8,
  "entry_price": 45000.0,
  "stop_loss": 44000.0,
  "take_profit_1": 46000.0,
  "take_profit_2": 47000.0,
  "timestamp": "2025-01-24T10:00:00"
}
```

## ⚠️ Importante

- **Paper Trading**: Por padrão, o sistema apenas simula trades
- **DeepSeek API**: API key é obrigatória
- **Risco**: Máximo 2% por trade
- **Stop Loss**: Sempre definido automaticamente

## 🔍 Troubleshooting

### Erro: "DEEPSEEK_API_KEY not set"
```bash
# Configurar variável de ambiente
export DEEPSEEK_API_KEY=sua_chave_aqui

# OU no Windows PowerShell
$env:DEEPSEEK_API_KEY="sua_chave_aqui"
```

### Erro: "Insufficient Balance"
- Adicione créditos na conta DeepSeek: https://platform.deepseek.com/

## 📚 Referências

- [AGNO Documentation](https://docs-v1.agno.com/)
- [DeepSeek Models](https://docs-v1.agno.com/models/deepseek)
- [DeepSeek Platform](https://platform.deepseek.com/)

---

**Sistema totalmente refatorado para usar AGNO Agent com DeepSeek como orquestrador principal!** 🚀