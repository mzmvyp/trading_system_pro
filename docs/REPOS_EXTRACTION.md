# REPOS_EXTRACTION — Código para Consolidação no trading_system_pro

> **Gerado em:** 2026-03-04
> **Repos analisados:** 5 (em `_repos_compare\`)
> **Nota:** Código COMPLETO dos arquivos mais relevantes e NOVOS. Arquivos duplicados marcados como JÁ TEMOS.

---

## Sumário de Extração

| Repo | Arquivos Extraídos | Novos Indicadores | Novas Features |
|------|--------------------|-------------------|----------------|
| smart_trading_system | 18 arquivos | Confluence, Divergence, Leading Indicators, Trend Analyzer | Breakout/Swing Strategy (completas), Market Structure, Risk Manager avançado, Signal Generator, 4 Filters completos |
| sinais | 8 arquivos | VWAP, Candlestick Patterns (5) | XGBoost + Feature Engineering, ML Signal Enhancer, LLM Sentiment (GPT-4o-mini), Optimization Engine, Performance Tracker |
| trader_monitor | 6 arquivos | Elliott Waves, Double Bottom | Notification Service (multi-canal), Backup Service, Advanced Pattern Analyzer, Scalp/DayTrade strategies |
| agente_trade_futuros | 6 arquivos | Stochastic Oscillator | Market Regime Detector (Futures), Position Sizing, Champion Tactics (4 analyzers), AGNO Advanced, Regime Optimizer |
| trade_bot_new | 2 arquivos novos | — | Position Manager (trailing/time exit/cooldown), Market Regime Filter (BTC-based) |

## Dependências Novas Necessárias

```
xgboost>=2.0.0              # sinais: OptimizedXGBoostPredictor
scipy>=1.11.4               # smart_trading_system, agente_trade_futuros: signal processing, stats
openai>=1.0.0               # sinais: SentimentAnalyzer (GPT-4o-mini)
ta>=0.10.2                  # agente_trade_futuros: alternativa ao TA-Lib para Stochastic
schedule>=1.2.0             # trader_monitor: backup scheduler
```

---

## REPO 1: smart_trading_system

**Caminho:** `_repos_compare\smart_trading_system`

> **NOTA:** As versões de strategies/ e filters/ no ROOT do trading_system_pro são stubs.
> As versões abaixo do smart_trading_system são as implementações COMPLETAS (5-10x maiores).
> Para código completo, ler os arquivos originais nos caminhos indicados.

### Arquivo: strategies/breakout_strategy.py
**Destino:** `src/strategies/breakout_strategy.py`
**Status:** NOVO (trading_system_pro tem stub de ~22 linhas)
**Dependências:** numpy, pandas
**Linhas:** 682
**Caminho completo:** `_repos_compare\smart_trading_system\strategies\breakout_strategy.py`

**O que faz:**
- `ConsolidationPattern` dataclass: pattern_type (RANGE/TRIANGLE/FLAG), upper/lower bounds, duration, volume_profile
- `BreakoutSetup` dataclass: pattern, breakout_price, volume_surge, retest_opportunity
- `BreakoutSignal` dataclass: setup, direction, strength, entry/stop/targets, confluence_score
- `BreakoutStrategy`: detect_consolidation(), identify_breakout(), confirm_with_volume(), check_retest(), generate_signals()
- Volume surge threshold: 2x average
- False breakout filter via price action

**ADAPTAÇÃO:** Imports relativos `..core.market_structure`, `..indicators.trend_analyzer` → adaptar para imports absolutos.

### Arquivo: strategies/swing_strategy.py
**Destino:** `src/strategies/swing_strategy.py`
**Status:** NOVO (trading_system_pro tem stub de ~22 linhas)
**Dependências:** numpy, pandas
**Linhas:** 660
**Caminho completo:** `_repos_compare\smart_trading_system\strategies\swing_strategy.py`

**O que faz:**
- Multi-timeframe (4H/1D/1H)
- `SwingSetup` enum: STRUCTURE_BREAK, PULLBACK_ENTRY, SR_RETEST
- Confluência com pesos: structure(35%), trend(25%), leading(25%), S/R(15%)
- Signal expiration: 72 horas
- Market Structure: HH/HL/LH/LL detection

### Arquivo: strategies/trend_following.py
**Destino:** `src/strategies/trend_following.py` (SUBSTITUIR versão atual de 123 linhas)
**Status:** UPGRADE (1149 linhas vs 123 atuais)
**Caminho completo:** `_repos_compare\smart_trading_system\strategies\trend_following.py`

**Adições vs versão atual:**
- TrendSetup enum (5 tipos), TrendPhase enum (Early/Mature/Exhaustion)
- TrendFollowingConfig (20+ parâmetros configuráveis)
- 4 tipos de setup: Pullback, Breakout Follow, MA Cross, Momentum
- Confluence score, risk score, priority system (1-5)

### Arquivo: strategies/mean_reversion.py
**Destino:** `src/strategies/mean_reversion.py` (SUBSTITUIR versão atual de 100 linhas)
**Status:** UPGRADE (751 linhas vs 100 atuais)
**Caminho completo:** `_repos_compare\smart_trading_system\strategies\mean_reversion.py`

**Adições:** Stochastic, Williams %R, RSI divergence, Bollinger squeeze, S/R bounce

### Arquivo: indicators/confluence_analyzer.py
**Destino:** `src/analysis/confluence_analyzer.py`
**Status:** NOVO
**Linhas:** 872
**Caminho completo:** `_repos_compare\smart_trading_system\indicators\confluence_analyzer.py`

**O que faz:** Sistema central combinando 5 módulos com pesos:
1. Market Structure (25%)
2. Trend Analysis multi-TF (25%)
3. Leading Indicators (20%)
4. Strategy Signals (20%)
5. S/R Levels (10%)

### Arquivo: indicators/divergence_detector.py
**Destino:** `src/analysis/divergence_detector.py`
**Status:** NOVO
**Linhas:** 645
**Caminho completo:** `_repos_compare\smart_trading_system\indicators\divergence_detector.py`

**O que faz:** Divergências entre preço e indicadores (RSI, MACD, Stochastic, Momentum, Williams %R, CCI).
- Tipos: Bullish Regular, Bearish Regular, Bullish Hidden, Bearish Hidden
- Scoring: strength (WEAK/MODERATE/STRONG/VERY_STRONG), confidence, reliability

### Arquivo: indicators/leading_indicators.py
**Destino:** `src/analysis/leading_indicators.py`
**Status:** NOVO (complementa nosso order_flow.py)
**Linhas:** 675
**Caminho completo:** `_repos_compare\smart_trading_system\indicators\leading_indicators.py`

**O que faz:**
- `VolumeProfileAnalyzer`: POC, Value Areas, Volume Nodes (mais avançado que nosso)
- `OrderFlowAnalyzer`: Buy/Sell pressure, momentum score, liquidity
- `LiquidityAnalyzer`: Liquidity zones, sweep detection
- `LeadingIndicatorsSystem`: Combina os 3 analyzers com score agregado

### Arquivo: indicators/trend_analyzer.py
**Destino:** `src/analysis/trend_analyzer.py`
**Status:** NOVO (complementa nosso multi_timeframe.py)
**Linhas:** 877
**Caminho completo:** `_repos_compare\smart_trading_system\indicators\trend_analyzer.py`

**O que faz:** Multi-TF trend com hierarquia 1D(3x) > 4H(2x) > 1H(1x):
- TimeframeTrend: direction, strength, phase, MA alignment, momentum
- MultiTimeframeTrendAnalysis: alignment score, entry signals
- Reversal detection, breakout signals

### Arquivo: core/market_structure.py
**Destino:** `src/analysis/market_structure.py`
**Status:** NOVO
**Linhas:** 728
**Caminho completo:** `_repos_compare\smart_trading_system\core\market_structure.py`

**O que faz:**
- MarketPhase enum: Accumulation, Markup, Distribution, Markdown
- StructureType enum: HH, HL, LH, LL, EH, EL
- StructurePoint: significance scoring
- TrendAnalysis: regression line, R², confidence
- BreakoutAnalysis: volume confirmation, false breakout risk

### Arquivo: core/risk_manager.py
**Destino:** `src/trading/risk_manager_advanced.py`
**Status:** NOVO (mais avançado que nosso validate_risk_and_position)
**Linhas:** 769
**Caminho completo:** `_repos_compare\smart_trading_system\core\risk_manager.py`

### Arquivo: core/signal_generator.py
**Destino:** `src/trading/signal_generator_master.py`
**Status:** NOVO (orquestrador central)
**Linhas:** 1396
**Caminho completo:** `_repos_compare\smart_trading_system\core\signal_generator.py`

### Filtros (IMPLEMENTAÇÕES COMPLETAS — substituir stubs atuais)

| Arquivo | Destino | Linhas | O que faz |
|---------|---------|--------|-----------|
| `filters/volatility_filter.py` | `src/filters/volatility_filter.py` | 833 | 5 regimes, GARCH, compression/expansion/clustering |
| `filters/time_filter.py` | `src/filters/time_filter.py` | 973 | Sessions Asia/EU/US, liquidity, timing per strategy |
| `filters/market_condition.py` | `src/filters/market_condition.py` | 946 | 9 conditions, Fear/Greed, multi-TF |
| `filters/fundamental_filter.py` | `src/filters/fundamental_filter.py` | 996 | Economic calendar, news sentiment, black swan risk |

---

## REPO 2: sinais

**Caminho:** `_repos_compare\sinais`

### Comparação ML: sinais vs trading_system_pro

| Feature | trading_system_pro | sinais | Ação |
|---------|-------------------|--------|------|
| RF + GB + LR ensemble | ✅ | ❌ | JÁ TEMOS |
| LSTM validator | ✅ | ❌ | JÁ TEMOS |
| Online learning | ✅ | ❌ | JÁ TEMOS |
| XGBoost otimizado | ❌ | ✅ | **IMPORTAR** |
| Feature Engineering pipeline | ❌ | ✅ | **IMPORTAR** |
| ML Signal Enhancer | ❌ | ✅ | **IMPORTAR** |
| LLM Sentiment (GPT-4o-mini) | ✅ DeepSeek | ✅ OpenAI | **IMPORTAR** (complementa) |
| Candlestick Patterns | ❌ | ✅ | **IMPORTAR** |
| Optimization Engine | ❌ | ✅ | **IMPORTAR** |
| Performance Tracker | ❌ | ✅ | **IMPORTAR** |
| VWAP | ❌ | ✅ | **IMPORTAR** |

### Arquivo: ml/optimized_xgboost_predictor.py
**Destino:** `src/ml/xgboost_predictor.py`
**Status:** NOVO
**Dependências:** xgboost, scikit-learn
**Linhas:** 483

**ADAPTAÇÃO:** Trocar `from ml.feature_engineering import FeatureEngineer` → import local. Trocar `from config.settings import settings` → nosso config.

```python
#!/usr/bin/env python3
"""
XGBoost Predictor Otimizado - Com otimização de hiperparâmetros robusta
Versão melhorada com RandomizedSearch, Early Stopping e validação temporal
"""

import pickle
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

try:
    import xgboost as xgb
    from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.utils import class_weight
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from ml.feature_engineering import FeatureEngineer
from config.settings import settings


class OptimizedXGBoostPredictor:
    """
    Preditor XGBoost Otimizado com hiperparâmetros

    Features:
    - RandomizedSearch para otimização eficiente
    - Early stopping para evitar overfitting
    - Validação cruzada temporal robusta
    - Análise de importância de features
    - Retreino automático baseado em performance
    """

    def __init__(self, model_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)

        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost não instalado. Execute: pip install xgboost scikit-learn")

        self.feature_engineer = FeatureEngineer()

        self.base_params = {
            'objective': 'binary:logistic',
            'tree_method': 'hist',
            'eval_metric': 'logloss',
            'random_state': 42,
            'n_jobs': 1
        }

        self.param_distribution = {
            'max_depth': [3, 4, 5, 6, 7, 8],
            'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
            'n_estimators': [300, 500, 800, 1000, 1200],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5, 7, 10],
            'gamma': [0, 0.1, 0.2, 0.3, 0.5],
            'reg_alpha': [0, 0.1, 0.5, 1.0, 2.0],
            'reg_lambda': [0.5, 1.0, 1.5, 2.0, 3.0]
        }

        self.model_path = model_path or os.path.join('data', 'models', 'optimized_xgboost_model.pkl')
        self.model = None
        self.feature_names = []
        self.best_params = None
        self.optimization_results = {}

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.load_model()
        self.logger.info("Optimized XGBoost Predictor inicializado")

    def train_with_hyperparameter_optimization(
        self, symbols: List[str], timeframe: str = "1h", lookback: int = 500,
        prediction_horizon: int = 12, n_iter: int = 50, cv_folds: int = 3,
        early_stopping_rounds: int = 50, test_size: float = 0.2
    ) -> Dict:
        self.logger.info(f"Iniciando treinamento otimizado: {symbols}, TF={timeframe}")

        X, y, feature_names = self.feature_engineer.prepare_training_data(
            symbols, timeframe, lookback, prediction_horizon
        )

        if X is None or y is None:
            return {'error': 'Data preparation failed', 'optimization_results': {}}

        self.feature_names = feature_names
        tscv = TimeSeriesSplit(n_splits=cv_folds, gap=24)

        try:
            class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
            scale_pos_weight = class_weights[1] / class_weights[0] if len(class_weights) > 1 else 1.0
        except Exception:
            scale_pos_weight = 1.0

        xgb_params = self.base_params.copy()
        xgb_params['scale_pos_weight'] = scale_pos_weight
        xgb_model = xgb.XGBClassifier(**xgb_params)

        search = RandomizedSearchCV(
            xgb_model, self.param_distribution, n_iter=n_iter, cv=tscv,
            scoring='roc_auc', n_jobs=1, random_state=42, verbose=1
        )

        start_time = datetime.now()
        try:
            search.fit(X, y)
        except Exception as e:
            return {'error': f'Optimization failed: {e}', 'optimization_results': {}}

        optimization_time = (datetime.now() - start_time).total_seconds()

        self.best_params = search.best_params_
        self.model = search.best_estimator_
        self.save_model()

        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, y_pred_proba)
        }

        feature_importance = self._analyze_feature_importance()

        self.optimization_results = {
            'method': 'RandomizedSearch', 'cv_folds': cv_folds, 'n_iter': n_iter,
            'optimization_time': optimization_time, 'best_score': search.best_score_,
            'best_params': self.best_params, 'n_samples': len(X),
            'n_features': len(feature_names), 'metrics': metrics,
            'feature_importance': feature_importance,
            'training_date': datetime.now().isoformat()
        }

        return {'success': True, 'optimization_results': self.optimization_results}

    def _analyze_feature_importance(self) -> Dict:
        if self.model is None:
            return {}
        try:
            importance_scores = self.model.feature_importances_
            feature_importance = []
            for i, (feature, score) in enumerate(zip(self.feature_names, importance_scores)):
                feature_importance.append({
                    'rank': i + 1, 'feature': feature,
                    'importance': float(score),
                    'percentage': float(score / importance_scores.sum() * 100)
                })
            feature_importance.sort(key=lambda x: x['importance'], reverse=True)
            return {'all_features': feature_importance, 'top_10': feature_importance[:10],
                    'total_features': len(self.feature_names)}
        except Exception as e:
            self.logger.error(f"Erro ao analisar importância: {e}")
            return {}

    def predict(self, symbol: str, timeframe: str = "1h") -> Optional[Dict]:
        if self.model is None:
            return None
        try:
            df = self.feature_engineer.prepare_features(symbol, timeframe, 200)
            if df is None or len(df) < 50:
                return None
            X = df[self.feature_names].iloc[[-1]]
            prediction = self.model.predict(X)[0]
            prediction_proba = self.model.predict_proba(X)[0]
            confidence = prediction_proba[1] if prediction == 1 else prediction_proba[0]
            return {
                'symbol': symbol,
                'prediction': 'BULLISH' if prediction == 1 else 'BEARISH',
                'confidence': float(confidence),
                'probability_up': float(prediction_proba[1]),
                'probability_down': float(prediction_proba[0]),
                'timestamp': datetime.now().isoformat(),
                'model_info': {
                    'optimization_method': self.optimization_results.get('method', 'unknown'),
                    'best_score': self.optimization_results.get('best_score', 0),
                    'training_date': self.optimization_results.get('training_date', 'unknown')
                }
            }
        except Exception as e:
            self.logger.error(f"Erro na predição {symbol}: {e}")
            return None

    def get_model_info(self) -> Dict:
        if self.model is None:
            return {'status': 'not_trained'}
        return {
            'status': 'trained', 'optimization_results': self.optimization_results,
            'best_params': self.best_params, 'feature_count': len(self.feature_names)
        }

    def save_model(self):
        try:
            model_data = {
                'model': self.model, 'feature_names': self.feature_names,
                'best_params': self.best_params, 'optimization_results': self.optimization_results
            }
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
        except Exception as e:
            self.logger.error(f"Erro ao salvar modelo: {e}")

    def load_model(self):
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                self.model = model_data['model']
                self.feature_names = model_data['feature_names']
                self.best_params = model_data['best_params']
                self.optimization_results = model_data['optimization_results']
                return True
            return False
        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo: {e}")
            return False

    def should_retrain(self, days_threshold: int = 7) -> bool:
        if not self.optimization_results:
            return True
        training_date = self.optimization_results.get('training_date')
        if not training_date:
            return True
        try:
            last_training = datetime.fromisoformat(training_date)
            return (datetime.now() - last_training).days >= days_threshold
        except:
            return True
```

### Arquivo: ml/feature_engineering.py
**Destino:** `src/ml/feature_engineering.py`
**Status:** NOVO
**Linhas:** 328

**ADAPTAÇÃO:** `from core.data_reader import DataReader` → adaptar para usar nosso `BinanceClient`.

```python
"""
Feature Engineering - Preparação de features para ML
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from core.data_reader import DataReader
from indicators.technical import TechnicalAnalyzer


class FeatureEngineer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.data_reader = DataReader()
        self.technical_analyzer = TechnicalAnalyzer()

    def prepare_features(self, symbol: str, timeframe: str = "5m", lookback: int = 200) -> Optional[pd.DataFrame]:
        try:
            market_data = self.data_reader.get_latest_data(symbol, timeframe)
            if not market_data or not market_data.is_sufficient_data:
                return None
            df = market_data.data.copy()
            if len(df) < lookback:
                return None
            df = df.tail(lookback).copy()
            df = self._add_technical_indicators(df, timeframe)
            df = self._add_lag_features(df)
            df = self._add_rolling_features(df)
            df = self._add_volume_features(df)
            df = self._add_price_patterns(df)
            df = self._add_temporal_features(df)
            df = df.dropna()
            if len(df) < 50:
                return None
            return df
        except Exception as e:
            self.logger.error(f"Erro ao preparar features para {symbol}: {e}")
            return None

    def _add_technical_indicators(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        delta = df['close_price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))

        ema_fast = df['close_price'].ewm(span=12, adjust=False).mean()
        ema_slow = df['close_price'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        df['bb_middle'] = df['close_price'].rolling(window=20).mean()
        bb_std = df['close_price'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (2 * bb_std)
        df['bb_lower'] = df['bb_middle'] - (2 * bb_std)
        df['bb_position'] = (df['close_price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)

        df['prev_close'] = df['close_price'].shift(1)
        df['tr1'] = df['high_price'] - df['low_price']
        df['tr2'] = abs(df['high_price'] - df['prev_close'])
        df['tr3'] = abs(df['low_price'] - df['prev_close'])
        df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr'] = df['true_range'].ewm(span=14, adjust=False).mean()

        for period in [9, 21, 50]:
            df[f'ema_{period}'] = df['close_price'].ewm(span=period, adjust=False).mean()

        df.drop(['prev_close', 'tr1', 'tr2', 'tr3', 'true_range'], axis=1, inplace=True, errors='ignore')
        return df

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for lag in [1, 2, 3, 5, 10]:
            df[f'close_lag_{lag}'] = df['close_price'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        for lag in [1, 2, 3, 5]:
            df[f'return_lag_{lag}'] = df['close_price'].pct_change(lag).shift(1)
        return df

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for window in [5, 10, 20, 50]:
            df[f'close_mean_{window}'] = df['close_price'].rolling(window=window).mean()
            df[f'close_std_{window}'] = df['close_price'].rolling(window=window).std()
            df[f'close_min_{window}'] = df['close_price'].rolling(window=window).min()
            df[f'close_max_{window}'] = df['close_price'].rolling(window=window).max()
            df[f'close_position_{window}'] = (
                (df['close_price'] - df[f'close_min_{window}']) /
                (df[f'close_max_{window}'] - df[f'close_min_{window}'] + 1e-10)
            )
        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['volume_mean_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_mean_20'] + 1e-10)
        typical_price = (df['high_price'] + df['low_price'] + df['close_price']) / 3
        df['vwap'] = (typical_price * df['volume']).rolling(window=20).sum() / (df['volume'].rolling(window=20).sum() + 1e-10)
        df['vwap_distance'] = (df['close_price'] - df['vwap']) / (df['vwap'] + 1e-10)
        for window in [10, 20]:
            df[f'vwma_{window}'] = (df['close_price'] * df['volume']).rolling(window=window).sum() / (df['volume'].rolling(window=window).sum() + 1e-10)
        return df

    def _add_price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        df['body_size'] = abs(df['close_price'] - df['open_price'])
        df['body_size_pct'] = df['body_size'] / (df['open_price'] + 1e-10)
        df['upper_shadow'] = df['high_price'] - df[['open_price', 'close_price']].max(axis=1)
        df['lower_shadow'] = df[['open_price', 'close_price']].min(axis=1) - df['low_price']
        df['is_bullish'] = (df['close_price'] > df['open_price']).astype(int)
        df['candle_range'] = df['high_price'] - df['low_price']
        df['candle_range_pct'] = df['candle_range'] / (df['open_price'] + 1e-10)
        df['consecutive_up'] = 0
        df['consecutive_down'] = 0
        for i in range(1, len(df)):
            if df.iloc[i]['is_bullish'] == 1:
                df.iloc[i, df.columns.get_loc('consecutive_up')] = df.iloc[i-1]['consecutive_up'] + 1
            else:
                df.iloc[i, df.columns.get_loc('consecutive_down')] = df.iloc[i-1]['consecutive_down'] + 1
        return df

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        return df

    def create_target(self, df: pd.DataFrame, prediction_horizon: int = 12) -> pd.DataFrame:
        df['future_return'] = df['close_price'].shift(-prediction_horizon) / df['close_price'] - 1
        threshold = 0.01
        df['target'] = (df['future_return'] > threshold).astype(int)
        df = df[~df['target'].isna()].copy()
        df.drop('future_return', axis=1, inplace=True)
        return df

    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        exclude_cols = ['timestamp', 'symbol', 'timeframe', 'target',
                       'open_price', 'high_price', 'low_price', 'close_price', 'volume']
        return [col for col in df.columns if col not in exclude_cols]

    def prepare_training_data(self, symbols: List[str], timeframe: str = "5m",
                            lookback: int = 1000, prediction_horizon: int = 12
                            ) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], List[str]]:
        all_data = []
        for symbol in symbols:
            df = self.prepare_features(symbol, timeframe, lookback)
            if df is not None and len(df) > 0:
                df = self.create_target(df, prediction_horizon)
                if len(df) > 0:
                    df['symbol'] = symbol
                    all_data.append(df)
        if not all_data:
            return None, None, []
        combined_df = pd.concat(all_data, ignore_index=True)
        feature_names = self.get_feature_names(combined_df)
        X = combined_df[feature_names]
        y = combined_df['target']
        return X, y, feature_names
```

### Arquivo: ml/ml_integration.py
**Destino:** `src/ml/ml_signal_enhancer.py`
**Status:** NOVO
**Linhas:** 177

```python
"""
ML Integration - Adiciona score ML aos sinais existentes
"""
import logging
from typing import Dict, Optional

from ml.optimized_xgboost_predictor import OptimizedXGBoostPredictor, XGBOOST_AVAILABLE


class MLSignalEnhancer:
    def __init__(self, enabled: bool = True, ml_weight: float = 0.25):
        self.logger = logging.getLogger(__name__)
        self.enabled = enabled and XGBOOST_AVAILABLE
        self.ml_weight = ml_weight
        self.predictor: Optional[OptimizedXGBoostPredictor] = None

        if self.enabled:
            try:
                self.predictor = OptimizedXGBoostPredictor()
                if self.predictor.model is None:
                    self.enabled = False
            except Exception:
                self.enabled = False

    def enhance_signal(self, symbol: str, signal_type: str,
                      technical_confidence: float, timeframe: str = "5m") -> Dict:
        if not self.enabled:
            return {'final_confidence': technical_confidence, 'ml_enabled': False}

        try:
            ml_result = self.predictor.predict(symbol, timeframe)
            if ml_result is None:
                return {'final_confidence': technical_confidence, 'ml_enabled': True, 'ml_error': 'failed'}

            ml_bullish = ml_result['prediction'] == 'BULLISH'
            signal_bullish = 'BUY' in signal_type
            agreement = ml_bullish == signal_bullish

            if agreement:
                ml_contribution = ml_result['confidence'] * self.ml_weight
                final_confidence = technical_confidence * (1 - self.ml_weight) + ml_contribution
            else:
                final_confidence = technical_confidence * (1 - self.ml_weight) + (1 - ml_result['confidence']) * self.ml_weight

            final_confidence = max(0.0, min(1.0, final_confidence))

            return {
                'final_confidence': final_confidence, 'ml_enabled': True,
                'ml_prediction': ml_result['prediction'],
                'ml_confidence': ml_result['confidence'],
                'ml_agrees': agreement, 'technical_confidence': technical_confidence
            }
        except Exception as e:
            return {'final_confidence': technical_confidence, 'ml_error': str(e)}
```

### Arquivo: indicators/candlestick_patterns_detector.py
**Destino:** `src/analysis/candlestick_patterns.py`
**Status:** NOVO
**Linhas:** 640

**ADAPTAÇÃO:** Colunas `close_price`→`close`, `open_price`→`open`, `high_price`→`high`, `low_price`→`low`

**Código completo:** `_repos_compare\sinais\indicators\candlestick_patterns_detector.py`

**Resumo das classes:**
- `CandlestickPattern` dataclass: name, type, entry, stop, target1, target2, reliability, strength
- `SimplifiedCandlestickDetector`: 5 padrões efetivos
  - Bullish Engulfing (reliability: 0.85)
  - Bearish Engulfing (reliability: 0.90)
  - Hammer (reliability: 0.75)
  - Shooting Star (reliability: 0.75)
  - Doji (reliability: 0.60)
- Validações: max risk 2.5%, min R:R 1.2:1, max target 5%
- `generate_candlestick_signals()`: Função compatível com sistema existente

### Arquivo: indicators/technical.py (APENAS VWAPAnalyzer — o resto JÁ TEMOS)
**Destino:** Adicionar ao `src/analysis/indicators.py`
**Status:** PARCIAL

```python
class VWAPAnalyzer:
    """VWAP integrado ao sistema principal"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def analyze(self, df, timeframe: str):
        if len(df) < 20:
            return None

        typical_price = (df['high'] + df['low'] + df['close']) / 3
        volume = df['volume']
        window = min(50, len(typical_price))
        pv = typical_price * volume
        rolling_pv = pv.rolling(window=window).sum()
        rolling_volume = volume.rolling(window=window).sum()
        vwap = rolling_pv / (rolling_volume + 1e-10)

        latest_price = typical_price.iloc[-1]
        latest_vwap = vwap.iloc[-1]
        distance_pct = abs(latest_price - latest_vwap) / latest_vwap * 100

        signals = []
        if distance_pct < 0.5:
            if latest_price > latest_vwap:
                signals.append({'type': 'vwap_support_test', 'signal_type': 'BUY_LONG', 'confidence': 0.70})
            else:
                signals.append({'type': 'vwap_resistance_test', 'signal_type': 'SELL_SHORT', 'confidence': 0.70})

        return {
            'vwap': vwap, 'current_vwap': latest_vwap,
            'price_vs_vwap': 'above' if latest_price > latest_vwap else 'below',
            'distance_pct': distance_pct, 'signals': signals
        }
```

### Arquivo: backtesting/optimization_engine.py
**Destino:** `src/backtesting/optimization_engine.py`
**Status:** NOVO
**Linhas:** 707
**Caminho completo:** `_repos_compare\sinais\backtesting\optimization_engine.py`

**O que faz:** Testa configs aleatórias de RSI/MACD/BB/volume/confidence, simula trades, calcula score composto (30% win rate + 30% return + 20% Sharpe + 20% drawdown), background thread.

### Arquivo: trading/performance_tracker.py
**Destino:** `src/trading/performance_tracker.py`
**Status:** NOVO
**Linhas:** 386

**ADAPTAÇÃO:** `from config.settings import settings` → nosso config. `self.db_path` → nosso DB path.

```python
"""
Performance Tracker - Métricas avançadas de trading
"""
import sqlite3
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np

from config.settings import settings


@dataclass
class TradeStats:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    profit_factor: float
    avg_duration_minutes: float


class PerformanceTracker:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.db_path = settings.database.signals_db_path

    def _get_connection(self):
        return sqlite3.connect(self.db_path, timeout=10)

    def get_closed_trades(self, days: int = 7) -> List[Dict]:
        cutoff_date = datetime.now() - timedelta(days=days)
        query = """
        SELECT id, symbol, side, entry_price, exit_price, quantity,
               entry_time, exit_time, pnl, pnl_percentage, fees_paid,
               slippage_cost, exit_reason, duration_minutes, metadata
        FROM paper_trades WHERE datetime(created_at) >= ?
        ORDER BY exit_time DESC
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (cutoff_date.isoformat(),))
                rows = cursor.fetchall()
                return [{'id': r[0], 'symbol': r[1], 'side': r[2], 'entry_price': r[3],
                         'exit_price': r[4], 'quantity': r[5], 'entry_time': r[6],
                         'exit_time': r[7], 'pnl': r[8], 'pnl_percentage': r[9],
                         'fees_paid': r[10], 'slippage_cost': r[11], 'exit_reason': r[12],
                         'duration_minutes': r[13]} for r in rows]
        except Exception as e:
            self.logger.error(f"Erro ao buscar trades: {e}")
            return []

    def calculate_trade_stats(self, trades: List[Dict]) -> TradeStats:
        if not trades:
            return TradeStats(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] < 0]
        total = len(trades)
        win_rate = (len(wins) / total) * 100 if total > 0 else 0
        avg_win = sum(t['pnl'] for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t['pnl'] for t in losses) / len(losses) if losses else 0
        total_gains = sum(t['pnl'] for t in wins)
        total_losses = abs(sum(t['pnl'] for t in losses))
        profit_factor = total_gains / total_losses if total_losses > 0 else float('inf')
        return TradeStats(
            total, len(wins), len(losses), win_rate, avg_win, avg_loss,
            max((t['pnl'] for t in wins), default=0),
            min((t['pnl'] for t in losses), default=0),
            profit_factor, sum(t['duration_minutes'] for t in trades) / total
        )

    def calculate_sharpe_ratio(self, trades: List[Dict], risk_free_rate: float = 0.02) -> float:
        if not trades or len(trades) < 2:
            return 0.0
        returns = [t['pnl_percentage'] / 100 for t in trades]
        avg_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        if std_return == 0:
            return 0.0
        avg_duration_days = np.mean([t['duration_minutes'] / (60 * 24) for t in trades])
        period_risk_free = risk_free_rate * (avg_duration_days / 365)
        sharpe = (avg_return - period_risk_free) / std_return
        trades_per_year = 365 / avg_duration_days
        return sharpe * np.sqrt(trades_per_year)

    def calculate_sortino_ratio(self, trades: List[Dict], risk_free_rate: float = 0.02) -> float:
        if not trades or len(trades) < 2:
            return 0.0
        returns = [t['pnl_percentage'] / 100 for t in trades]
        avg_return = np.mean(returns)
        downside_returns = [r for r in returns if r < 0]
        if not downside_returns:
            return float('inf')
        downside_std = np.std(downside_returns, ddof=1)
        if downside_std == 0:
            return 0.0
        avg_duration_days = np.mean([t['duration_minutes'] / (60 * 24) for t in trades])
        period_risk_free = risk_free_rate * (avg_duration_days / 365)
        sortino = (avg_return - period_risk_free) / downside_std
        trades_per_year = 365 / avg_duration_days
        return sortino * np.sqrt(trades_per_year)

    def calculate_expectancy(self, trades: List[Dict]) -> float:
        if not trades:
            return 0.0
        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] < 0]
        total = len(trades)
        win_rate = len(wins) / total if total > 0 else 0
        avg_win = sum(t['pnl'] for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t['pnl'] for t in losses) / len(losses) if losses else 0
        return (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    def get_comprehensive_report(self, days: int = 7) -> Dict:
        trades = self.get_closed_trades(days)
        stats = self.calculate_trade_stats(trades)
        return {
            'period_days': days, 'timestamp': datetime.now().isoformat(),
            'basic_stats': {
                'total_trades': stats.total_trades, 'win_rate': stats.win_rate,
                'profit_factor': stats.profit_factor, 'avg_win': stats.avg_win,
                'avg_loss': stats.avg_loss
            },
            'advanced_metrics': {
                'sharpe_ratio': self.calculate_sharpe_ratio(trades),
                'sortino_ratio': self.calculate_sortino_ratio(trades),
                'expectancy': self.calculate_expectancy(trades)
            }
        }
```

---

## REPO 3: trader_monitor

**Caminho:** `_repos_compare\trader_monitor`

> **NOTA:** Sistema Flask massivo. Abaixo apenas funcionalidade NOVA e portável.

### Arquivo: services/advanced_pattern_analyzer.py
**Destino:** `src/analysis/advanced_patterns.py`
**Status:** NOVO
**Linhas:** ~1160
**Caminho completo:** `_repos_compare\trader_monitor\services\advanced_pattern_analyzer.py`

**O que faz (ÚNICO):**
1. **Elliott Wave Analysis**: 5 ondas impulso + 3 correção, validação por Fibonacci (wave2: 50-78.6%, wave3: 1-2.618x, wave4: 23.6-50%, wave5: 0.618-1.618x)
2. **Double Bottom**: max 2% diff entre fundos, 4-48h entre fundos, 3%+ pico, volume confirmation
3. **OCO/OCOI**: Order management patterns

### Arquivo: services/notification_service.py
**Destino:** `src/services/notification_service.py`
**Status:** NOVO
**Linhas:** ~730
**Caminho completo:** `_repos_compare\trader_monitor\services\notification_service.py`

**O que faz:** Multi-channel notifications:
- Email (SMTP), Slack (Webhook), Discord (Webhook), Telegram (Bot API), Webhook genérico
- SQLite para histórico e configurações
- Rate limiting, priority system (low/medium/high/critical)

### Arquivo: services/backup_service.py
**Destino:** `src/services/backup_service.py`
**Status:** NOVO
**Linhas:** ~730
**Caminho completo:** `_repos_compare\trader_monitor\services\backup_service.py`

**O que faz:** Backup automático agendado, ZIP+SHA256, rotação, restore com verificação.

### Estratégias (scalp_strategy.py contém 3 classes)
**Status:** PARCIALMENTE DUPLICADO com smart_trading_system

As do trader_monitor são mais simples. **RECOMENDAÇÃO:** Usar smart_trading_system (mais completas) e parametrizar timeframes para scalp (1m).

---

## REPO 4: agente_trade_futuros

**Caminho:** `_repos_compare\agente_trade_futuros`

### Comparação de Indicadores

| Indicador | trading_system_pro | agente_trade_futuros | Ação |
|-----------|-------------------|---------------------|------|
| RSI | ✅ TA-Lib | ✅ ta library | JÁ TEMOS |
| MACD | ✅ | ✅ | JÁ TEMOS |
| ADX | ✅ | ✅ | JÁ TEMOS |
| ATR | ✅ | ✅ | JÁ TEMOS |
| Bollinger Bands | ✅ | ✅ | JÁ TEMOS |
| SMA | ✅ 20/50/200 | ✅ 20/50 | JÁ TEMOS |
| EMA | ✅ 20/50/200 | ✅ 12/26 | JÁ TEMOS |
| OBV | ✅ | ❌ | JÁ TEMOS |
| Volume Profile | ✅ | ❌ | JÁ TEMOS |
| Fibonacci | ✅ | ✅ | JÁ TEMOS |
| **Stochastic** | ❌ | ✅ | **IMPORTAR** |
| VWAP | ❌ | ❌ | IMPORTAR do sinais |
| Ichimoku | ❌ | ❌ | NÃO EXISTE em nenhum repo |
| Williams %R | ❌ | ❌ | NÃO EXISTE |
| CCI | ❌ | ❌ | NÃO EXISTE |

### Arquivo: technical_indicators.py (APENAS Stochastic — resto JÁ TEMOS)
**Destino:** Adicionar ao `src/analysis/indicators.py`

```python
# Versão usando 'ta' library (como no agente_trade_futuros)
import ta

@staticmethod
def calculate_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    """Calcula o Oscilador Estocástico"""
    stoch = ta.momentum.StochasticOscillator(
        df['high'], df['low'], df['close'],
        window=k_period, smooth_window=d_period
    )
    return stoch.stoch(), stoch.stoch_signal()

# Versão TA-Lib (para nosso sistema que já usa TA-Lib):
import talib

def calculate_stochastic(close, high, low, k_period=14, d_period=3):
    """Stochastic Oscillator usando TA-Lib"""
    slowk, slowd = talib.STOCH(high, low, close,
                                fastk_period=k_period,
                                slowk_period=d_period,
                                slowd_period=d_period)
    return slowk, slowd
```

### Arquivo: market_regime_detector_futures.py
**Destino:** `src/analysis/market_regime_detector.py`
**Status:** NOVO
**Dependências:** pandas, numpy
**Linhas:** 297

```python
import pandas as pd
import numpy as np
from typing import Dict
import logging
from binance_client import BinanceClient

logger = logging.getLogger(__name__)

class MarketRegimeDetectorFutures:
    """
    Detector de Regime de Mercado Otimizado para Binance Futures.
    Analisa tendência, volatilidade, momentum e dados de futuros.
    """

    def __init__(self, binance_client: BinanceClient):
        self.binance_client = binance_client
        self.ema_periods = [9, 21, 50, 200]
        logger.info("MarketRegimeDetectorFutures inicializado.")

    def detect_regime(self, symbol: str = 'BTCUSDT', lookback: int = 200) -> Dict:
        try:
            df_15m = self.binance_client.get_klines(symbol, interval='15m', limit=lookback)
            df_1h = self.binance_client.get_klines(symbol, interval='1h', limit=lookback)
            df_4h = self.binance_client.get_klines(symbol, interval='4h', limit=100)

            if df_15m is None or df_15m.empty or df_1h is None or df_1h.empty:
                return self._get_default_regime()

            trend_analysis = self._analyze_trend(df_1h, df_4h)
            volatility_analysis = self._analyze_volatility(df_15m)
            momentum_analysis = self._analyze_momentum(df_15m, df_1h)
            funding_analysis = self._analyze_funding_rate(symbol)
            oi_analysis = self._analyze_open_interest(symbol)
            liquidation_analysis = self._analyze_liquidations(symbol)

            return self._combine_analyses(
                trend_analysis, volatility_analysis, momentum_analysis,
                funding_analysis, oi_analysis, liquidation_analysis
            )
        except Exception as e:
            logger.error(f"Erro na detecção de regime: {e}", exc_info=True)
            return self._get_default_regime()

    def _analyze_trend(self, df_1h, df_4h) -> Dict:
        try:
            for period in self.ema_periods:
                df_1h[f'ema_{period}'] = df_1h['close'].ewm(span=period, adjust=False).mean()
            current_price = df_1h['close'].iloc[-1]
            ema9, ema21, ema50, ema200 = [df_1h[f'ema_{p}'].iloc[-1] for p in self.ema_periods]

            score = 0
            if current_price > ema9 > ema21 > ema50 > ema200: score = 1.0
            elif current_price < ema9 < ema21 < ema50 < ema200: score = -1.0
            else:
                if current_price > ema50: score += 0.4
                if current_price < ema50: score -= 0.4
                if ema9 > ema21: score += 0.3
                if ema9 < ema21: score -= 0.3

            adx = self._calculate_adx(df_1h).iloc[-1]
            if adx < 25:
                trend = "SIDEWAYS"
                confidence = 1 - (adx / 25)
            else:
                if score > 0.5: trend = "BULL"
                elif score < -0.5: trend = "BEAR"
                else: trend = "SIDEWAYS"
                confidence = min(abs(score), adx / 50)
            return {'trend': trend, 'confidence': confidence, 'adx': adx}
        except Exception:
            return {'trend': 'SIDEWAYS', 'confidence': 0.5, 'adx': 25}

    def _analyze_volatility(self, df) -> Dict:
        try:
            atr = self._calculate_atr(df)
            natr = (atr / df['close']).iloc[-1]
            upper, _, lower = self._calculate_bollinger_bands(df)
            bbw = ((upper - lower) / df['close']).iloc[-1]
            natr_series = (atr / df['close'])
            vol_high = natr_series.rolling(100).quantile(0.80).iloc[-1]
            vol_low = natr_series.rolling(100).quantile(0.20).iloc[-1]
            if bbw < 0.025: vol_regime = "SQUEEZE"
            elif natr > vol_high: vol_regime = "HIGH"
            elif natr < vol_low: vol_regime = "LOW"
            else: vol_regime = "NORMAL"
            return {'volatility': vol_regime, 'value': natr, 'bbw': bbw}
        except Exception:
            return {'volatility': 'NORMAL', 'value': 0.03, 'bbw': 0.05}

    def _analyze_momentum(self, df_15m, df_1h) -> Dict:
        try:
            rsi_15m = self._calculate_rsi(df_15m).iloc[-1]
            rsi_1h = self._calculate_rsi(df_1h).iloc[-1]
            _, _, macd_hist = self._calculate_macd(df_15m)
            score = 0
            if rsi_15m > 55 and rsi_1h > 52: score += 0.5
            if rsi_15m < 45 and rsi_1h < 48: score -= 0.5
            if macd_hist.iloc[-1] > 0 and macd_hist.iloc[-1] > macd_hist.iloc[-2]: score += 0.5
            if macd_hist.iloc[-1] < 0 and macd_hist.iloc[-1] < macd_hist.iloc[-2]: score -= 0.5
            if score > 0.5: momentum = "BULLISH"
            elif score < -0.5: momentum = "BEARISH"
            else: momentum = "NEUTRAL"
            return {'momentum': momentum, 'score': score, 'rsi_15m': rsi_15m}
        except Exception:
            return {'momentum': 'NEUTRAL', 'score': 0, 'rsi_15m': 50}

    def _analyze_funding_rate(self, symbol) -> Dict:
        try:
            rate = self.binance_client.get_funding_rate(symbol)
            if rate is None: return {'bias': 'NEUTRAL', 'rate': 0.0, 'confidence': 0.0}
            if rate > 0.0002: bias = "OVERBOUGHT"
            elif rate < -0.0002: bias = "OVERSOLD"
            else: bias = "NEUTRAL"
            return {'bias': bias, 'rate': rate, 'confidence': min(abs(rate) * 1000, 1.0)}
        except Exception:
            return {'bias': 'NEUTRAL', 'rate': 0.0, 'confidence': 0.0}

    def _analyze_open_interest(self, symbol) -> Dict:
        try:
            oi_df = self.binance_client.get_open_interest_hist(symbol, period='1h', limit=24)
            if oi_df is None or oi_df.empty:
                return {'trend': 'NEUTRAL', 'change': 0.0, 'confidence': 0.0}
            if 'open_interest' not in oi_df.columns and 'sumOpenInterest' in oi_df.columns:
                oi_df['open_interest'] = pd.to_numeric(oi_df['sumOpenInterest'])
            current_oi = float(oi_df['open_interest'].iloc[-1])
            past_oi = float(oi_df['open_interest'].iloc[0])
            change = (current_oi - past_oi) / past_oi if past_oi != 0 else 0.0
            if change > 0.05: trend = "INCREASING"
            elif change < -0.05: trend = "DECREASING"
            else: trend = "STABLE"
            return {'trend': trend, 'change': change, 'confidence': min(abs(change) * 10, 1.0)}
        except Exception:
            return {'trend': 'NEUTRAL', 'change': 0.0, 'confidence': 0.0}

    def _analyze_liquidations(self, symbol) -> Dict:
        return {'bias': 'NEUTRAL', 'total': 0, 'confidence': 0.0}

    def _combine_analyses(self, trend, volatility, momentum, funding, oi, liq) -> Dict:
        bull_score, bear_score = 0, 0
        if trend['trend'] == 'BULL': bull_score += 1.0 * trend['confidence']
        if trend['trend'] == 'BEAR': bear_score += 1.0 * trend['confidence']
        if momentum['momentum'] == 'BULLISH': bull_score += 0.8 * abs(momentum['score'])
        if momentum['momentum'] == 'BEARISH': bear_score += 0.8 * abs(momentum['score'])
        if oi['trend'] == 'INCREASING' and trend['trend'] == 'BULL': bull_score += 0.5 * oi['confidence']
        if oi['trend'] == 'INCREASING' and trend['trend'] == 'BEAR': bear_score += 0.5 * oi['confidence']
        if funding['bias'] == 'OVERBOUGHT': bear_score += 0.3 * funding['confidence']
        if funding['bias'] == 'OVERSOLD': bull_score += 0.3 * funding['confidence']

        if bull_score > bear_score and trend['adx'] > 25: base_regime = 'BULL'
        elif bear_score > bull_score and trend['adx'] > 25: base_regime = 'BEAR'
        else: base_regime = 'SIDEWAYS'

        final_regime = f"{base_regime}_{volatility['volatility']}_VOL"
        if volatility['volatility'] == 'SQUEEZE': final_regime = "SQUEEZE_BREAKOUT_PENDING"
        if base_regime == 'BULL' and funding['bias'] == 'OVERBOUGHT' and momentum['momentum'] != 'BULLISH':
            final_regime = 'LONG_SQUEEZE'
        if base_regime == 'BEAR' and funding['bias'] == 'OVERSOLD' and momentum['momentum'] != 'BEARISH':
            final_regime = 'SHORT_SQUEEZE'

        final_confidence = (bull_score + bear_score) / 2 if (bull_score + bear_score) > 0 else 0.5
        return {
            'regime': final_regime, 'base_regime': base_regime,
            'confidence': min(final_confidence, 1.0),
            'details': {'trend': trend, 'volatility': volatility, 'momentum': momentum,
                        'funding': funding, 'open_interest': oi}
        }

    def _calculate_atr(self, df, period=14):
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()

    def _calculate_adx(self, df, period=14):
        df['up_move'] = df['high'].diff()
        df['down_move'] = -df['low'].diff()
        df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0.0)
        df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0.0)
        atr = self._calculate_atr(df, period)
        plus_di = 100 * (df['plus_dm'].ewm(alpha=1/period).mean() / atr)
        minus_di = 100 * (df['minus_dm'].ewm(alpha=1/period).mean() / atr)
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
        return dx.ewm(alpha=1/period).mean().fillna(25)

    def _calculate_rsi(self, df, period=14):
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/period, adjust=False).mean()
        loss = -delta.where(delta < 0, 0).ewm(alpha=1/period, adjust=False).mean()
        rs = gain / loss
        return (100 - (100 / (1 + rs))).fillna(50)

    def _calculate_macd(self, df, fast=12, slow=26, signal=9):
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line, signal_line, macd_line - signal_line

    def _calculate_bollinger_bands(self, df, period=20, std=2):
        sma = df['close'].rolling(window=period).mean()
        std_dev = df['close'].rolling(window=period).std()
        return sma + (std_dev * std), sma, sma - (std_dev * std)

    def _get_default_regime(self):
        return {'regime': 'SIDEWAYS_NORMAL_VOL', 'confidence': 0.5, 'base_regime': 'SIDEWAYS'}
```

### Arquivo: position_sizing.py
**Destino:** `src/trading/position_sizing.py`
**Status:** NOVO
**Linhas:** 254

```python
"""Position Sizing - Sistema de Gerenciamento de Risco"""
import logging
from typing import Dict, Optional, Tuple
import config

logger = logging.getLogger(__name__)

class PositionSizing:
    """Position Sizing baseado em risco"""

    def __init__(self):
        self.risk_percentage = config.RISK_PERCENTAGE_PER_TRADE
        self.max_risk_per_trade = config.MAX_RISK_PER_TRADE
        self.max_total_risk = config.MAX_TOTAL_RISK
        self.available_capital = 0.0
        self.total_capital = 0.0

    def set_capital(self, total_capital: float, available_capital: float = None):
        self.total_capital = total_capital
        self.available_capital = available_capital if available_capital is not None else total_capital

    def calculate_position_size(self, entry_price: float, stop_loss: float,
                              risk_percentage: float = None) -> Dict:
        if self.available_capital <= 0:
            return {'success': False, 'error': 'Capital não definido'}

        risk_pct = risk_percentage if risk_percentage is not None else self.risk_percentage

        if entry_price > stop_loss:
            risk_per_share = entry_price - stop_loss
        else:
            risk_per_share = stop_loss - entry_price

        risk_percentage_per_unit = (risk_per_share / entry_price) * 100
        risk_amount = self.available_capital * risk_pct

        if risk_percentage_per_unit > 0:
            position_size = risk_amount / risk_per_share
        else:
            return {'success': False, 'error': 'Stop muito próximo da entrada'}

        position_value = position_size * entry_price
        if position_value > self.available_capital:
            position_value = self.available_capital * 0.95
            position_size = position_value / entry_price
            risk_amount = position_size * risk_per_share

        actual_risk_pct = (risk_amount / self.available_capital) * 100
        return {
            'success': True, 'position_size': position_size,
            'position_value': position_value, 'risk_amount': risk_amount,
            'risk_percentage': actual_risk_pct, 'entry_price': entry_price,
            'stop_loss': stop_loss, 'risk_per_unit': risk_per_share
        }

    def validate_position(self, position_size, entry_price, stop_loss) -> Dict:
        position_value = position_size * entry_price
        risk_per_share = abs(entry_price - stop_loss)
        risk_amount = position_size * risk_per_share
        risk_pct = (risk_amount / self.available_capital) * 100
        validations = {
            'capital_available': position_value <= self.available_capital,
            'risk_per_trade': risk_pct <= (self.max_risk_per_trade * 100),
            'position_size_reasonable': position_size > 0
        }
        return {'valid': all(validations.values()), 'risk_percentage': risk_pct,
                'validations': validations}

    def calculate_multiple_positions(self, positions: list) -> Dict:
        total_risk = 0
        for pos in positions:
            risk_per_share = abs(pos['entry_price'] - pos['stop_loss'])
            total_risk += pos['position_size'] * risk_per_share
        total_risk_pct = (total_risk / self.available_capital) * 100
        return {'total_risk_percentage': total_risk_pct,
                'within_limits': total_risk_pct <= (self.max_total_risk * 100)}
```

### Arquivo: champion_tactics.py
**Destino:** `src/analysis/champion_tactics.py`
**Status:** NOVO
**Dependências:** scipy, talib
**Linhas:** 769
**Caminho completo:** `_repos_compare\agente_trade_futuros\champion_tactics.py`

**Classes (código completo no caminho acima):**
1. `OrderFlowAnalyzer`: Volume ponderado por direção, cumulative delta, delta trend
2. `VolumeProfileAnalyzer`: Distribuição de volume por preço, S/R por volume
3. `MarketMicrostructureAnalyzer`: Spread, liquidity assessment
4. `MomentumDivergenceAnalyzer`: RSI/MACD divergence com scipy.signal.find_peaks
5. `ChampionTacticsIntegrator`: Pesos OrderFlow(30%) + Volume(25%) + Micro(20%) + Divergence(25%)

### Arquivo: agno_advanced.py
**Destino:** `src/analysis/agno_advanced.py`
**Status:** NOVO
**Linhas:** 506
**Caminho completo:** `_repos_compare\agente_trade_futuros\agno_advanced.py`

**O que faz:** AGNO expandido com 6 análises ponderadas:
- Basic(30%) + Candlestick(15%) + Fibonacci(15%) + Structure(20%) + Divergence(10%) + Liquidity(10%)

### Arquivo: agno_regime_optimizer.py
**Destino:** `src/backtesting/regime_optimizer.py`
**Status:** NOVO
**Linhas:** 558
**Caminho completo:** `_repos_compare\agente_trade_futuros\agno_regime_optimizer.py`

**O que faz:** Otimizador evolutivo (population-based) com 4 regime presets.
**NOTA:** Interface com MarketRegimeDetectorFutures tem bug (chama com start_date/end_date mas método aceita lookback).

---

## REPO 5: trade_bot_new

**Caminho:** `_repos_compare\trade_bot_new`

> **NOTA:** signal_reevaluator.py, stop_adjuster.py, agno_tools.py, binance_client.py, config.py, constants.py, real_paper_trading.py, trading_agent_agno.py, streamlit_dashboard.py — TODOS JÁ EXISTEM no trading_system_pro.

### Arquivo: position_manager.py
**Destino:** `src/trading/position_manager.py`
**Status:** NOVO
**Linhas:** 532

**ADAPTAÇÃO:** `from logger import get_logger` → nosso logger. `from config import settings` → nosso config. Adicionar settings: `trailing_stop_enabled`, `trailing_stop_activation_pct`, `trailing_stop_atr_multiplier`, `trailing_stop_step_pct`, `time_based_exit_enabled`, `time_exit_scalp_candles`, `time_exit_day_trade_candles`, `time_exit_swing_candles`, `time_exit_action`, `post_profit_cooldown_enabled`, `post_profit_cooldown_hours`, `post_profit_cooldown_partial_hours`, `trade_limit_enabled`, `max_trades_per_pair_per_day`, `max_total_trades_per_day`, `exit_distribution`, `dynamic_confidence_enabled`, `confidence_after_profit`, `confidence_after_loss`, `recent_trade_hours`.

```python
"""
Position Manager - Gerenciamento Avançado de Posições
Funcionalidades: Trailing Stop, Time-Based Exit, Cooldown, Trade Limits,
Exit Distribution, Dynamic Confidence
"""
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from logger import get_logger
from config import settings

logger = get_logger(__name__)


class PositionManager:
    def __init__(self):
        self.state_file = Path("portfolio/state.json")
        self.trade_history_file = Path("portfolio/trade_history.json")
        self.trailing_stops = {}
        Path("portfolio").mkdir(exist_ok=True)
        self._load_trade_history()

    def _load_trade_history(self):
        self.trade_history = []
        try:
            if self.trade_history_file.exists():
                with open(self.trade_history_file, "r", encoding="utf-8") as f:
                    self.trade_history = json.load(f)
        except Exception:
            self.trade_history = []

    def _save_trade_history(self):
        try:
            with open(self.trade_history_file, "w", encoding="utf-8") as f:
                json.dump(self.trade_history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Erro ao salvar histórico: {e}")

    def record_trade(self, symbol, signal_type, result, pnl_percent, entry_price, exit_price, duration_hours):
        trade = {
            "timestamp": datetime.now().isoformat(), "symbol": symbol,
            "signal_type": signal_type, "result": result,
            "pnl_percent": pnl_percent, "entry_price": entry_price,
            "exit_price": exit_price, "duration_hours": duration_hours
        }
        self.trade_history.append(trade)
        self._save_trade_history()

    async def calculate_trailing_stop(self, position: Dict, current_price: float, atr: float) -> Optional[float]:
        if not settings.trailing_stop_enabled:
            return None
        entry_price = position.get("entry_price", 0)
        signal_type = position.get("signal", "BUY")
        current_stop = position.get("stop_loss", 0) or 0
        if entry_price <= 0 or current_price <= 0:
            return None

        if signal_type == "BUY":
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
        else:
            pnl_pct = ((entry_price - current_price) / entry_price) * 100

        if pnl_pct < settings.trailing_stop_activation_pct:
            return None

        trailing_distance = atr * settings.trailing_stop_atr_multiplier

        if signal_type == "BUY":
            new_stop = current_price - trailing_distance
            if current_stop and new_stop <= current_stop:
                return None
            if current_stop > 0:
                improvement_pct = ((new_stop - current_stop) / current_stop) * 100
                if improvement_pct < settings.trailing_stop_step_pct:
                    return None
        else:
            new_stop = current_price + trailing_distance
            if current_stop and new_stop >= current_stop:
                return None
            if current_stop > 0:
                improvement_pct = ((current_stop - new_stop) / current_stop) * 100
                if improvement_pct < settings.trailing_stop_step_pct:
                    return None
        return new_stop

    def check_time_based_exit(self, position: Dict) -> Optional[str]:
        if not settings.time_based_exit_enabled:
            return None
        operation_type = position.get("operation_type", "DAY_TRADE").upper()
        timestamp_str = position.get("timestamp")
        if not timestamp_str:
            return None
        try:
            open_time = datetime.fromisoformat(timestamp_str)
            hours_open = (datetime.now() - open_time).total_seconds() / 3600
            if "SCALP" in operation_type:
                max_hours = settings.time_exit_scalp_candles * (5 / 60)
            elif "DAY" in operation_type:
                max_hours = settings.time_exit_day_trade_candles * 1
            else:
                max_hours = settings.time_exit_swing_candles * 4
            if hours_open > max_hours:
                return settings.time_exit_action
            return None
        except Exception:
            return None

    def is_in_post_profit_cooldown(self, symbol: str) -> Tuple[bool, float]:
        if not settings.post_profit_cooldown_enabled:
            return False, 0
        now = datetime.now()
        for trade in reversed(self.trade_history):
            if trade.get("symbol") != symbol:
                continue
            try:
                trade_time = datetime.fromisoformat(trade.get("timestamp", ""))
                hours_since = (now - trade_time).total_seconds() / 3600
                if trade.get("result") == "WIN":
                    pnl = trade.get("pnl_percent", 0)
                    cooldown = settings.post_profit_cooldown_hours if pnl >= 3.0 else settings.post_profit_cooldown_partial_hours
                    if hours_since < cooldown:
                        return True, cooldown - hours_since
                break
            except Exception:
                continue
        return False, 0

    def can_open_trade(self, symbol: str) -> Tuple[bool, str]:
        if not settings.trade_limit_enabled:
            return True, "OK"
        now = datetime.now()
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        trades_today_symbol = 0
        trades_today_total = 0
        for trade in self.trade_history:
            try:
                trade_time = datetime.fromisoformat(trade.get("timestamp", ""))
                if trade_time >= day_start:
                    trades_today_total += 1
                    if trade.get("symbol") == symbol:
                        trades_today_symbol += 1
            except Exception:
                continue
        if trades_today_symbol >= settings.max_trades_per_pair_per_day:
            return False, f"Limite diário para {symbol} atingido"
        if trades_today_total >= settings.max_total_trades_per_day:
            return False, "Limite diário total atingido"
        return True, "OK"

    def get_exit_distribution(self) -> List[float]:
        dist = settings.exit_distribution
        if sum(dist) != 100:
            return [0.4, 0.4, 0.2]
        return [d / 100.0 for d in dist]

    def calculate_position_sizes(self, total_quantity: float) -> Dict[str, float]:
        distribution = self.get_exit_distribution()
        return {
            "tp1": total_quantity * distribution[0],
            "tp2": total_quantity * distribution[1],
            "runner": total_quantity * distribution[2]
        }

    def get_dynamic_confidence_threshold(self, symbol: str) -> int:
        if not settings.dynamic_confidence_enabled:
            return settings.min_confidence_0_10
        now = datetime.now()
        cutoff = now - timedelta(hours=settings.recent_trade_hours)
        for trade in reversed(self.trade_history):
            if trade.get("symbol") != symbol:
                continue
            try:
                trade_time = datetime.fromisoformat(trade.get("timestamp", ""))
                if trade_time >= cutoff:
                    if trade.get("result") == "WIN":
                        return settings.confidence_after_profit
                    elif trade.get("result") == "LOSS":
                        return settings.confidence_after_loss
                break
            except Exception:
                continue
        return settings.min_confidence_0_10

    def should_take_signal(self, symbol: str, signal_confidence: int) -> Tuple[bool, str]:
        can_trade, reason = self.can_open_trade(symbol)
        if not can_trade:
            return False, reason
        in_cooldown, remaining = self.is_in_post_profit_cooldown(symbol)
        if in_cooldown:
            return False, f"Cooldown pós-lucro ({remaining:.1f}h restantes)"
        threshold = self.get_dynamic_confidence_threshold(symbol)
        if signal_confidence < threshold:
            return False, f"Confiança {signal_confidence} < threshold {threshold}"
        return True, "OK"

position_manager = PositionManager()
```

### Arquivo: market_regime_filter.py
**Destino:** `src/analysis/market_regime_filter.py`
**Status:** NOVO
**Linhas:** 393

**ADAPTAÇÃO:** `from binance_client import BinanceClient` → nosso client async. `import talib` → já temos.

```python
"""
Filtro de Regime de Mercado baseado em BTC.
Bloqueia sinais contraditórios à tendência geral.
"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
from enum import Enum

from logger import get_logger

logger = get_logger(__name__)


class MarketRegime(Enum):
    STRONG_BULLISH = "STRONG_BULLISH"
    BULLISH = "BULLISH"
    NEUTRAL = "NEUTRAL"
    BEARISH = "BEARISH"
    STRONG_BEARISH = "STRONG_BEARISH"


class MarketRegimeFilter:
    """
    Regras:
    - STRONG_BULLISH: Bloquear SHORTs
    - STRONG_BEARISH: Bloquear LONGs
    - BULLISH/BEARISH: Aviso mas PERMITE
    - NEUTRAL: Permite ambos
    """

    def __init__(self):
        self.current_regime = MarketRegime.NEUTRAL
        self.last_analysis_time = None
        self.regime_confidence = 0.0
        self.btc_data = {}
        self.cache_duration_minutes = 15

    async def analyze_btc_regime(self, force=False) -> Dict:
        if not force and self._is_cache_valid():
            return {"regime": self.current_regime.value, "confidence": self.regime_confidence, "cached": True}

        try:
            from binance_client import BinanceClient
            import talib
            import numpy as np

            async with BinanceClient() as client:
                klines_1h = await client.get_klines("BTCUSDT", "1h", limit=100)
                klines_4h = await client.get_klines("BTCUSDT", "4h", limit=50)
                klines_1d = await client.get_klines("BTCUSDT", "1d", limit=30)
                ticker = await client.get_ticker_24hr("BTCUSDT")

            if klines_1h.empty or klines_4h.empty:
                return {"regime": "NEUTRAL", "confidence": 0}

            close_1h = klines_1h['close'].values.astype(float)
            close_4h = klines_4h['close'].values.astype(float)
            current_price = float(ticker['lastPrice'])
            price_change_24h = float(ticker['priceChangePercent'])

            ema_20_1h = talib.EMA(close_1h, timeperiod=20)[-1]
            ema_50_1h = talib.EMA(close_1h, timeperiod=50)[-1]
            ema_20_4h = talib.EMA(close_4h, timeperiod=20)[-1]
            ema_50_4h = talib.EMA(close_4h, timeperiod=50)[-1]

            high_1h = klines_1h['high'].values.astype(float)
            low_1h = klines_1h['low'].values.astype(float)
            adx_1h = talib.ADX(high_1h, low_1h, close_1h, timeperiod=14)[-1]
            rsi_1h = talib.RSI(close_1h, timeperiod=14)[-1]
            rsi_4h = talib.RSI(close_4h, timeperiod=14)[-1]
            macd, macd_signal, macd_hist = talib.MACD(close_1h)
            macd_bullish = macd_hist[-1] > 0

            bullish_score, bearish_score, total_weight = 0, 0, 0

            for check, weight in [
                (current_price > ema_20_1h, 3), (current_price > ema_50_1h, 3),
                (ema_20_1h > ema_50_1h, 4), (current_price > ema_20_4h, 3),
                (ema_20_4h > ema_50_4h, 4)
            ]:
                if check: bullish_score += weight
                else: bearish_score += weight
                total_weight += weight

            if price_change_24h > 2: bullish_score += 2
            elif price_change_24h < -2: bearish_score += 2
            total_weight += 2

            if macd_bullish: bullish_score += 2
            else: bearish_score += 2
            total_weight += 2

            avg_rsi = (rsi_1h + rsi_4h) / 2
            if avg_rsi > 55: bullish_score += 2
            elif avg_rsi < 45: bearish_score += 2
            total_weight += 2

            net_score = (bullish_score - bearish_score) / total_weight if total_weight > 0 else 0
            trend_strength = min(adx_1h / 50, 1.0)

            if net_score > 0.6: regime, confidence = MarketRegime.STRONG_BULLISH, min(0.9, 0.7 + net_score * 0.3)
            elif net_score > 0.3: regime, confidence = MarketRegime.BULLISH, min(0.8, 0.5 + net_score * 0.4)
            elif net_score < -0.6: regime, confidence = MarketRegime.STRONG_BEARISH, min(0.9, 0.7 + abs(net_score) * 0.3)
            elif net_score < -0.3: regime, confidence = MarketRegime.BEARISH, min(0.8, 0.5 + abs(net_score) * 0.4)
            else: regime, confidence = MarketRegime.NEUTRAL, 0.5

            if regime != MarketRegime.NEUTRAL:
                confidence *= (0.7 + 0.3 * trend_strength)

            self.current_regime = regime
            self.regime_confidence = confidence
            self.last_analysis_time = datetime.now()
            self.btc_data = {
                "current_price": current_price, "price_change_24h": price_change_24h,
                "net_score": net_score, "adx": adx_1h
            }

            return {"regime": regime.value, "confidence": confidence, "btc_data": self.btc_data}

        except Exception as e:
            return {"regime": "NEUTRAL", "confidence": 0, "error": str(e)}

    def _is_cache_valid(self):
        if not self.last_analysis_time:
            return False
        return (datetime.now() - self.last_analysis_time).total_seconds() < self.cache_duration_minutes * 60

    def should_allow_signal(self, signal_type: str) -> Tuple[bool, str]:
        regime = self.current_regime
        if regime == MarketRegime.STRONG_BULLISH and signal_type == "SELL":
            return False, "BLOQUEADO: STRONG_BULLISH - SHORTs não permitidos"
        if regime == MarketRegime.STRONG_BEARISH and signal_type == "BUY":
            return False, "BLOQUEADO: STRONG_BEARISH - LONGs não permitidos"
        if regime == MarketRegime.BULLISH and signal_type == "SELL":
            return True, "AVISO: BULLISH - SHORTs com risco elevado"
        if regime == MarketRegime.BEARISH and signal_type == "BUY":
            return True, "AVISO: BEARISH - LONGs com risco elevado"
        return True, "OK"

    def get_allowed_signals(self) -> list:
        if self.current_regime == MarketRegime.STRONG_BULLISH: return ["BUY"]
        if self.current_regime == MarketRegime.STRONG_BEARISH: return ["SELL"]
        return ["BUY", "SELL"]

market_regime_filter = MarketRegimeFilter()

async def check_market_regime_before_signal(signal_type: str) -> Tuple[bool, str, Dict]:
    regime_data = await market_regime_filter.analyze_btc_regime()
    allowed, reason = market_regime_filter.should_allow_signal(signal_type)
    return allowed, reason, regime_data
```

### Diferenças no binance_futures_executor.py (trade_bot_new vs trading_system_pro)

A versão do trade_bot_new pode ter atualizações para Algo Order API (`/fapi/v1/algoOrder`):
- `cancel_stop_loss_orders()`, `get_open_algo_orders()`, `cancel_algo_order()`
- Dynamic leverage (margin limit 10%, max 20x)
- `cleanup_orphan_orders()` — cancela ordens de symbols sem posição

**VERIFICAR:** Se a versão no trading_system_pro já tem essas funções. Se não, merge manual.

---

## Resumo de Prioridades para Merge

### ALTA (funcionalidade única):
1. `position_manager.py` (trade_bot_new) → Trailing/time/cooldown/limits
2. `market_regime_filter.py` (trade_bot_new) → BTC regime filter
3. `market_regime_detector_futures.py` (agente_trade_futuros) → Futures regime
4. `optimized_xgboost_predictor.py` + `feature_engineering.py` (sinais) → XGBoost ML
5. `champion_tactics.py` (agente_trade_futuros) → Institutional analysis
6. `performance_tracker.py` (sinais) → Sharpe/Sortino/Expectancy
7. `candlestick_patterns_detector.py` (sinais) → Pattern detection
8. Stochastic (agente_trade_futuros) + VWAP (sinais) → Novos indicadores
9. `position_sizing.py` (agente_trade_futuros) → Risk-based sizing

### MÉDIA (complementa existente):
10. `breakout_strategy.py` + `swing_strategy.py` (smart_trading_system) → Full strategies
11. `divergence_detector.py` (smart_trading_system) → Divergence detection
12. `leading_indicators.py` (smart_trading_system) → Volume/Flow/Liquidity
13. `market_structure.py` (smart_trading_system) → HH/HL/LH/LL + phases
14. `notification_service.py` (trader_monitor) → Multi-channel alerts
15. `advanced_pattern_analyzer.py` (trader_monitor) → Elliott Wave + Double Bottom

### BAIXA (nice to have):
16. Filters completos (smart_trading_system) → Substituir stubs
17. `backup_service.py` (trader_monitor) → Auto backup
18. `signal_generator.py` master (smart_trading_system) → Orquestrador
19. `trend_following.py` + `mean_reversion.py` upgrades (smart_trading_system)
20. `agno_advanced.py` + `agno_regime_optimizer.py` (agente_trade_futuros)
