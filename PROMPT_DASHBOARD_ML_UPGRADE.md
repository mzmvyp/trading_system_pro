# PROMPT: Upgrade do ML Dashboard com Backtest + LSTM Training Controls

## Contexto

O sistema `trading_system_pro` roda no Docker (docker-compose.yml). Tem 3 containers:
- `trading_bot` → `python main.py --mode monitor` (bot principal)
- `dashboard` → streamlit em `:8501` (trading dashboard)
- `ml_dashboard` → streamlit em `:8502` (ML dashboard)

Recentemente adicionamos:
1. `src/ml/backtest_dataset_generator.py` - gera dataset de treino para LSTM a partir de trades simulados do backtest
2. `src/ml/lstm_sequence_validator.py` - Bi-LSTM que valida sinais usando sequencias temporais de candles
3. Integração do Bi-LSTM como voto extra no sistema de confluência (`src/trading/agent.py`)

**Problema:** Esses novos componentes não têm controle no dashboard. O usuário precisa fazer tudo via terminal. Precisamos adicionar ao ML Dashboard (`:8502`) uma interface para:
- Gerar dataset do backtest
- Treinar o Bi-LSTM
- Ver status e métricas do Bi-LSTM
- Rodar backtests avulsos com visualização
- Ver resultados do continuous optimizer

## Arquivos existentes relevantes

### ML Dashboard atual: `src/dashboard/ml_dashboard.py`
- Já tem: status do modelo sklearn, online learning buffer, predições, feature importance
- Já tem botões: "Treinar Modelo do Zero", "Alimentar com Sinais", "Forçar Retreino"
- Usa CSS dark theme customizado
- Framework: Streamlit com Plotly

### Backtest Dataset Generator: `src/ml/backtest_dataset_generator.py`
```python
class BacktestDatasetGenerator:
    def __init__(self, symbols, interval="1h", sequence_length=60, days_back=180, n_param_variations=20):
    async def generate(self) -> Dict:  # retorna stats

# Output: ml_dataset/backtest/X_train_latest.npy, X_test_latest.npy, y_train_latest.npy, y_test_latest.npy
# Metadata: ml_dataset/backtest/dataset_info_latest.json
```

### LSTM Sequence Validator: `src/ml/lstm_sequence_validator.py`
```python
class LSTMSequenceValidator:
    def load_dataset(self) -> Tuple[X_train, X_test, y_train, y_test]
    def build_model(self)  # Bi-LSTM
    def train(self, epochs=100, batch_size=32) -> Dict  # retorna results
    def load_model(self) -> bool
    def predict_from_candles(self, candles_df) -> Dict  # {probability, prediction, confidence}

# Model: ml_models/bilstm_sequence_validator.h5
# Scaler: ml_models/bilstm_scaler.pkl
# Info: ml_models/bilstm_model_info.json
```

### Backtest Engine: `src/backtesting/backtest_engine.py`
```python
class BacktestEngine:
    async def fetch_data(self, symbol, interval, start_time, end_time) -> pd.DataFrame
    def calculate_indicators(self, df) -> pd.DataFrame
    def generate_signals(self, df) -> pd.DataFrame
    def simulate_trades(self, df, max_hold_bars=48) -> List[Trade]
    def calculate_metrics(self, trades) -> BacktestMetrics

@dataclass
class BacktestMetrics:
    total_trades, winning_trades, losing_trades, win_rate, total_return_pct,
    avg_return_pct, max_drawdown_pct, sharpe_ratio, sortino_ratio, profit_factor,
    avg_winner_pct, avg_loser_pct, max_consecutive_losses, trades, equity_curve

@dataclass
class Trade:
    entry_time, exit_time, direction, entry_price, exit_price, stop_loss,
    take_profit_1, take_profit_2, exit_reason, pnl_pct, is_winner
```

### Optimization Engine: `src/backtesting/optimization_engine.py`
```python
class OptimizationEngine:
    async def run_optimization(self, start_time, end_time, n_iterations=50) -> List[OptimizationResult]
    async def walk_forward(self, full_start, full_end, n_windows=4) -> List[WalkForwardWindow]
    def save_results(self, filepath=None) -> str

# Resultados salvos em: data/optimization/best_config_{symbol}_{interval}.json
```

### Continuous Optimizer: `src/backtesting/continuous_optimizer.py`
- Roda em background a cada 6h no modo monitor
- Salva best_config em `data/optimization/`

### Docker volumes (ml_dashboard tem acesso a):
```yaml
volumes:
  - ./ml_models:/app/ml_models
  - ./ml_dataset:/app/ml_dataset
  - ./data:/app/data
  - ./signals:/app/signals
```

## O que implementar

### 1. Nova aba/seção: "Bi-LSTM Sequence Validator"

Adicionar ao `ml_dashboard.py` uma seção para o Bi-LSTM (separada do sklearn validator existente):

**Status Card:**
- Se modelo existe (`ml_models/bilstm_model_info.json`):
  - Mostrar: tipo (Bi-LSTM), data do treino, accuracy treino/teste, F1 treino/teste, nº parâmetros, sequence_length, n_features
  - Mostrar: total de amostras treino/teste, win rate treino/teste
- Se não existe: mostrar alerta com botão para treinar

**Botão "Gerar Dataset do Backtest":**
```python
if st.button("Gerar Dataset do Backtest"):
    with st.spinner("Rodando backtests para gerar dados de treino... (pode levar 5-10 min)"):
        from src.ml.backtest_dataset_generator import BacktestDatasetGenerator
        generator = BacktestDatasetGenerator(
            symbols=symbols_selecionados,  # multiselect no sidebar
            interval=interval_selecionado,  # selectbox
            sequence_length=seq_length,     # slider 20-120
            days_back=days_back,            # slider 30-365
            n_param_variations=n_variations # slider 5-50
        )
        stats = asyncio.run(generator.generate())
    st.success(f"Dataset gerado: {stats['total_trades']} trades")
```

**Botão "Treinar Bi-LSTM":**
```python
if st.button("Treinar Bi-LSTM"):
    with st.spinner("Treinando Bi-LSTM..."):
        from src.ml.lstm_sequence_validator import LSTMSequenceValidator
        validator = LSTMSequenceValidator()
        results = validator.train(epochs=epochs, batch_size=batch_size)
    st.success(f"Treino concluído! Test acc: {results['test']['accuracy']:.1%}")
```

**Configurações no sidebar:**
- Símbolos: multiselect (default: BTCUSDT, ETHUSDT, SOLUSDT)
- Interval: selectbox (1h, 4h, 15m)
- Sequence Length: slider (20-120, default 60)
- Days Back: slider (30-365, default 180)
- Param Variations: slider (5-50, default 20)
- Epochs: slider (10-200, default 100)
- Batch Size: selectbox (16, 32, 64)

### 2. Nova aba/seção: "Backtest Explorer"

Interface para rodar backtests avulsos com visualização:

**Inputs:**
- Symbol: selectbox
- Interval: selectbox (1h, 4h, 15m, 5m)
- Período: date_input (start, end)
- Parâmetros: usar defaults ou customizar (expander com sliders para RSI, EMA, MACD, etc.)

**Botão "Rodar Backtest":**
```python
engine = BacktestEngine(params=params)
df = await engine.fetch_data(symbol, interval, start, end)
df = engine.calculate_indicators(df)
df = engine.generate_signals(df)
trades = engine.simulate_trades(df)
metrics = engine.calculate_metrics(trades)
```

**Visualizações:**
- Métricas cards: Win Rate, Return %, Sharpe, Max DD, Profit Factor, Total Trades
- Equity Curve (plotly line chart)
- Tabela de trades (entry_time, direction, entry_price, exit_price, pnl_pct, exit_reason, is_winner)
- Distribuição de PnL (histograma)
- Win/Loss pie chart
- Se tiver Bi-LSTM treinado: mostrar predição do LSTM para cada trade

### 3. Nova aba/seção: "Optimizer Results"

Mostrar resultados do continuous optimizer:

**Carregar resultados:**
```python
import glob
config_files = glob.glob("data/optimization/best_config_*.json")
# Para cada arquivo, mostrar: symbol, score, win_rate, return, sharpe, drawdown, params
```

**Botão "Rodar Otimização Agora":**
```python
from src.backtesting.optimization_engine import OptimizationEngine
engine = OptimizationEngine(symbol=symbol, interval=interval)
results = await engine.run_optimization(start_time, end_time, n_iterations=n_iter)
```

**Visualizações:**
- Tabela dos top 10 resultados (score, win_rate, return, sharpe, drawdown)
- Gráfico de convergência (score por iteração)
- Parâmetros do melhor resultado (JSON expandable)

### 4. Estrutura sugerida do dashboard

Usar `st.tabs()` para organizar:

```python
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 ML Validator (sklearn)",     # conteúdo atual do ml_dashboard
    "🧠 Bi-LSTM Sequence",          # novo
    "🔬 Backtest Explorer",          # novo
    "⚙️ Optimizer",                  # novo
])
```

### 5. Notas técnicas

- O dashboard roda em container separado, mas compartilha volumes com o bot
- Para rodar código async no Streamlit: `asyncio.run()` ou `loop.run_until_complete()`
- O Bi-LSTM usa TensorFlow que já está no requirements.txt
- Os modelos são salvos em `ml_models/` (volume compartilhado)
- Os datasets em `ml_dataset/` (volume compartilhado)
- Os resultados de otimização em `data/` (volume compartilhado)
- O ml_dashboard container precisa do volume `./data:/app/data` (JÁ TEM no docker-compose)
- Manter o CSS dark theme que já existe
- Manter tudo em português (interface do usuário)
- Usar plotly para gráficos (já importado)
- O backtest engine precisa chamar a Binance API para dados históricos, então o container precisa das env vars BINANCE_API_KEY (JÁ TEM via env_file)

### 6. Tratamento de erros

- Se TensorFlow não carregar, mostrar aviso mas não quebrar o dashboard
- Se dataset não existir, mostrar "Gere o dataset primeiro" com o botão
- Se Binance API falhar no backtest, mostrar erro amigável
- Progress bars para operações longas (backtest, treino)
- Usar `st.status()` para mostrar progresso multi-step

### 7. NÃO fazer

- NÃO criar novos arquivos de dashboard (tudo no ml_dashboard.py)
- NÃO alterar docker-compose.yml (volumes já estão corretos)
- NÃO alterar os módulos de ML/backtest (já funcionam)
- NÃO remover funcionalidade existente do ml_dashboard
- NÃO adicionar dependências novas ao requirements.txt
