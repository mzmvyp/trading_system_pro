"""
ML Training Pipeline - Treinamento Completo do Modelo de Validacao de Sinais
=============================================================================

Pipeline robusto que funciona em dois modos:

1. BOOTSTRAP MODE (sem sinais existentes):
   - Gera dados de treinamento a partir de klines historicos da Binance
   - Simula cenarios de BUY/SELL em pontos historicos reais
   - Calcula indicadores tecnicos (RSI, MACD, BB, ADX, ATR) reais
   - Valida se TP/SL seriam atingidos nos dados subsequentes
   - Produz dataset com as MESMAS 16 features usadas na predicao

2. SIGNAL MODE (com sinais avaliados):
   - Usa sinais existentes em signals/ avaliados pelo signal_tracker
   - Busca klines do momento do sinal para calcular indicadores reais
   - Enriquece com features tecnicas reais (nao regex)

CRITICO: As features devem ser IDENTICAS entre:
- train_from_signals.py (treinamento)
- agent.py _validate_with_ml_model() (predicao)
- online_learning.py add_signal_result() (retreinamento)
- simple_validator.py prepare_features() (inferencia)

Features (16):
    rsi, macd_histogram, adx, atr, bb_position,
    cvd, orderbook_imbalance, bullish_tf_count, bearish_tf_count,
    confidence, trend_encoded, sentiment_encoded, signal_encoded,
    risk_distance_pct, reward_distance_pct, risk_reward_ratio

Uso:
    python -m src.ml.train_from_signals
    python main.py --mode train_ml
"""

import json
import os
import pickle
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from src.core.logger import get_logger

logger = get_logger(__name__)

MODEL_DIR = "ml_models"
DATASET_DIR = "ml_dataset"

# As 16 features canonicas - DEVEM ser identicas em todos os modulos ML
FEATURE_COLUMNS = [
    'rsi', 'macd_histogram', 'adx', 'atr', 'bb_position',
    'cvd', 'orderbook_imbalance', 'bullish_tf_count', 'bearish_tf_count',
    'confidence', 'trend_encoded', 'sentiment_encoded', 'signal_encoded',
    'risk_distance_pct', 'reward_distance_pct', 'risk_reward_ratio'
]

# Binance public API
BINANCE_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"

# Simbolos para bootstrap
BOOTSTRAP_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
    "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT",
    "DOTUSDT", "LINKUSDT"
]


# ============================================================
# INDICADORES TECNICOS (calculados diretamente dos klines)
# ============================================================

def compute_rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    """Calcula RSI a partir de precos de fechamento."""
    delta = closes.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def compute_macd_histogram(closes: pd.Series) -> pd.Series:
    """Calcula MACD histogram."""
    ema12 = closes.ewm(span=12, adjust=False).mean()
    ema26 = closes.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd - signal


def compute_adx(highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int = 14) -> pd.Series:
    """Calcula ADX (Average Directional Index)."""
    tr1 = highs - lows
    tr2 = (highs - closes.shift(1)).abs()
    tr3 = (lows - closes.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()

    up_move = highs - highs.shift(1)
    down_move = lows.shift(1) - lows

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)

    plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / (atr + 1e-10)
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / (atr + 1e-10)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, adjust=False).mean()
    return adx


def compute_atr(highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int = 14) -> pd.Series:
    """Calcula ATR (Average True Range)."""
    tr1 = highs - lows
    tr2 = (highs - closes.shift(1)).abs()
    tr3 = (lows - closes.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def compute_bb_position(closes: pd.Series, period: int = 20) -> pd.Series:
    """Calcula posicao dentro das Bollinger Bands (0-1)."""
    middle = closes.rolling(period).mean()
    std = closes.rolling(period).std()
    upper = middle + 2 * std
    lower = middle - 2 * std
    return (closes - lower) / (upper - lower + 1e-10)


def compute_ema_trend(closes: pd.Series) -> pd.Series:
    """Calcula trend encoded baseado em EMAs (EMA9 vs EMA21 vs EMA50)."""
    ema9 = closes.ewm(span=9, adjust=False).mean()
    ema21 = closes.ewm(span=21, adjust=False).mean()
    ema50 = closes.ewm(span=50, adjust=False).mean()

    def classify(row_idx):
        e9 = ema9.iloc[row_idx]
        e21 = ema21.iloc[row_idx]
        e50 = ema50.iloc[row_idx]
        if e9 > e21 > e50:
            return 2  # strong_bullish
        elif e9 > e21:
            return 1  # bullish
        elif e9 < e21 < e50:
            return -2  # strong_bearish
        elif e9 < e21:
            return -1  # bearish
        return 0  # neutral

    return pd.Series([classify(i) for i in range(len(closes))], index=closes.index)


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula todos os indicadores tecnicos a partir de OHLCV."""
    df = df.copy()
    closes = df['close']
    highs = df['high']
    lows = df['low']

    df['rsi'] = compute_rsi(closes)
    df['macd_histogram'] = compute_macd_histogram(closes)
    df['adx'] = compute_adx(highs, lows, closes)
    df['atr'] = compute_atr(highs, lows, closes)
    df['bb_position'] = compute_bb_position(closes)
    df['trend_encoded'] = compute_ema_trend(closes)

    # CVD e orderbook_imbalance - proxies a partir de volume e price action
    # Em producao estes vem do orderbook real; aqui estimamos
    df['cvd'] = (df['close'] - df['open']) * df['volume']
    df['cvd'] = df['cvd'].rolling(20).sum()

    # Orderbook imbalance proxy: ratio de volume em candles bullish vs total
    is_bull = (df['close'] > df['open']).astype(float)
    bull_vol = (df['volume'] * is_bull).rolling(20).sum()
    total_vol = df['volume'].rolling(20).sum() + 1e-10
    df['orderbook_imbalance'] = bull_vol / total_vol

    # Multi-timeframe counts proxy
    df['bullish_tf_count'] = is_bull.rolling(10).sum()
    df['bearish_tf_count'] = 10 - df['bullish_tf_count']

    # Sentiment encoded: baseado em momentum geral
    momentum = closes.pct_change(10)
    df['sentiment_encoded'] = momentum.apply(
        lambda x: 1 if x > 0.02 else (-1 if x < -0.02 else 0)
    )

    return df


# ============================================================
# BUSCA DE DADOS DA BINANCE
# ============================================================

def fetch_klines(symbol: str, interval: str = "15m", limit: int = 1500) -> Optional[pd.DataFrame]:
    """Busca klines da Binance Futures (API publica)."""
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base_volume',
            'taker_buy_quote_volume', 'ignore'
        ])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    except Exception as e:
        logger.warning(f"Erro ao buscar klines de {symbol}: {e}")
        return None


def fetch_klines_range(symbol: str, interval: str = "15m",
                       start_ms: int = None, end_ms: int = None,
                       limit: int = 1000) -> Optional[pd.DataFrame]:
    """Busca klines em um range especifico."""
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_ms:
        params["startTime"] = start_ms
    if end_ms:
        params["endTime"] = end_ms
    try:
        resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return None

        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base_volume',
            'taker_buy_quote_volume', 'ignore'
        ])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    except Exception as e:
        logger.warning(f"Erro ao buscar klines range de {symbol}: {e}")
        return None


# ============================================================
# MODO BOOTSTRAP - Gera dataset a partir de dados historicos
# ============================================================

def generate_bootstrap_signals(symbol: str, df: pd.DataFrame,
                                lookforward: int = 48) -> List[Dict]:
    """
    Gera sinais sinteticos a partir de dados historicos.
    SL/TP baseados em ATR (realista).
    """
    signals = []

    df = compute_all_indicators(df)
    df = df.dropna().reset_index(drop=True)

    if len(df) < lookforward + 60:
        return signals

    step = 4

    for i in range(60, len(df) - lookforward, step):
        row = df.iloc[i]
        close = row['close']
        atr = row['atr']

        if atr <= 0 or close <= 0:
            continue

        rsi = row['rsi']
        macd_hist = row['macd_histogram']
        adx_val = row['adx']
        bb_pos = row['bb_position']
        trend = row['trend_encoded']

        signal_type = None

        # Gerar sinais APENAS com base técnica real (sem aleatórios)
        if rsi < 45 and (trend >= 0 or bb_pos < 0.3):
            signal_type = 'BUY'
        elif rsi > 55 and (trend <= 0 or bb_pos > 0.7):
            signal_type = 'SELL'

        if signal_type is None:
            continue

        sl_distance = 1.5 * atr
        tp1_distance = 2.0 * atr

        if signal_type == 'BUY':
            entry_price = close
            stop_loss = close - sl_distance
            tp1 = close + tp1_distance
        else:
            entry_price = close
            stop_loss = close + sl_distance
            tp1 = close - tp1_distance

        future = df.iloc[i+1:i+1+lookforward]
        outcome = _check_outcome(signal_type, entry_price, stop_loss, tp1, future)

        if outcome is None:
            continue

        # Features de mercado INDEPENDENTES do SL/TP (evitar tautologia com label)
        atr_pct = (atr / close * 100) if close > 0 else 0
        candle_body_pct = abs(row['close'] - row['open']) / close * 100 if close > 0 else 0
        vol_ma = df['volume'].iloc[max(0, i-20):i].mean()
        volume_ratio = (row['volume'] / vol_ma) if vol_ma > 0 else 1.0

        confidence = _estimate_confidence(rsi, adx_val, bb_pos, trend, signal_type)

        signals.append({
            'symbol': symbol,
            'timestamp': str(row['timestamp']),
            'rsi': float(rsi),
            'macd_histogram': float(macd_hist),
            'adx': float(adx_val),
            'atr': float(atr),
            'bb_position': float(bb_pos),
            'cvd': float(row.get('cvd', 0)),
            'orderbook_imbalance': float(row.get('orderbook_imbalance', 0.5)),
            'bullish_tf_count': float(row.get('bullish_tf_count', 5)),
            'bearish_tf_count': float(row.get('bearish_tf_count', 5)),
            'confidence': confidence,
            'trend_encoded': int(trend),
            'sentiment_encoded': int(row.get('sentiment_encoded', 0)),
            'signal_encoded': 1 if signal_type == 'BUY' else (-1 if signal_type == 'SELL' else 0),
            'risk_distance_pct': float(atr_pct),
            'reward_distance_pct': float(candle_body_pct),
            'risk_reward_ratio': float(volume_ratio),
            'target': outcome,
            'signal_type': signal_type,
        })

    return signals


def _check_outcome(signal_type: str, entry: float, sl: float,
                   tp1: float, future: pd.DataFrame) -> Optional[int]:
    """Verifica se TP ou SL seria atingido nos candles futuros."""
    for _, candle in future.iterrows():
        h = candle['high']
        low = candle['low']

        if signal_type == 'BUY':
            if low <= sl:
                return 0
            if h >= tp1:
                return 1
        else:
            if h >= sl:
                return 0
            if low <= tp1:
                return 1

    return 0  # Timeout = loss


def _estimate_confidence(rsi: float, adx: float, bb_pos: float,
                         trend: int, signal_type: str) -> int:
    """Estima confidence (1-10) baseado na qualidade do setup tecnico."""
    score = 5

    if signal_type == 'BUY':
        if rsi < 30:
            score += 2
        elif rsi < 40:
            score += 1
        elif rsi > 60:
            score -= 1
    else:
        if rsi > 70:
            score += 2
        elif rsi > 60:
            score += 1
        elif rsi < 40:
            score -= 1

    if adx > 30:
        score += 1
    elif adx < 15:
        score -= 1

    if signal_type == 'BUY' and trend > 0:
        score += 1
    elif signal_type == 'SELL' and trend < 0:
        score += 1
    elif signal_type == 'BUY' and trend < -1:
        score -= 1
    elif signal_type == 'SELL' and trend > 1:
        score -= 1

    if signal_type == 'BUY' and bb_pos < 0.2:
        score += 1
    elif signal_type == 'SELL' and bb_pos > 0.8:
        score += 1

    return max(1, min(10, score))


# ============================================================
# MODO SIGNAL - Treina a partir de sinais existentes
# ============================================================

def load_signal_files() -> List[Dict]:
    """Carrega sinais dos diretorios signals/ e deepseek_logs/."""
    signals = []

    for search_dir in ["signals", "deepseek_logs"]:
        if not os.path.exists(search_dir):
            continue
        for root, dirs, files in os.walk(search_dir):
            for f in files:
                if not f.endswith('.json'):
                    continue
                if '_last_analysis' in f or 'cache' in f:
                    continue
                filepath = os.path.join(root, f)
                try:
                    with open(filepath, 'r', encoding='utf-8') as fh:
                        data = json.load(fh)
                        if data.get('signal') in ('BUY', 'SELL') and data.get('entry_price'):
                            data['_filepath'] = filepath
                            signals.append(data)
                except (json.JSONDecodeError, IOError):
                    continue

    return signals


def load_evaluated_signals() -> List[Dict]:
    """Carrega sinais ja avaliados pelo signal_tracker."""
    cache_file = "signals/evaluations_cache.json"
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                evaluations = json.load(f)
            print(f"[OK] Carregados {len(evaluations)} sinais avaliados do cache")
            return evaluations
        except (json.JSONDecodeError, IOError):
            pass

    try:
        from src.trading.signal_tracker import evaluate_all_signals
        evaluations = evaluate_all_signals()
        print(f"[OK] Avaliados {len(evaluations)} sinais")
        return evaluations
    except Exception as e:
        logger.warning(f"Falha ao avaliar sinais: {e}")
        return []


def enrich_signal_with_klines(signal: Dict) -> Optional[Dict]:
    """
    Enriquece um sinal avaliado com indicadores tecnicos reais
    buscando klines do momento da emissao do sinal.
    """
    symbol = signal.get('symbol', '')
    timestamp_str = signal.get('timestamp', '')
    signal_type = signal.get('signal', '')
    entry_price = signal.get('entry_price', 0)
    outcome = signal.get('outcome', '')

    if not symbol or not entry_price or signal_type not in ('BUY', 'SELL'):
        return None

    if outcome not in ('SL_HIT', 'TP1_HIT', 'TP2_HIT'):
        return None

    try:
        if 'T' in timestamp_str:
            ts = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            if ts.tzinfo:
                ts = ts.replace(tzinfo=None)
        else:
            ts = datetime.strptime(timestamp_str[:19], '%Y-%m-%d %H:%M:%S')
        start_ms = int(ts.timestamp() * 1000)
    except (ValueError, TypeError):
        return None

    end_ms = start_ms
    klines_before_ms = start_ms - (300 * 15 * 60 * 1000)

    df = fetch_klines_range(symbol, "15m", start_ms=klines_before_ms,
                           end_ms=end_ms, limit=300)
    if df is None or len(df) < 60:
        return None

    df = compute_all_indicators(df)
    df = df.dropna()
    if len(df) == 0:
        return None

    last = df.iloc[-1]

    # Features de mercado independentes (não derivadas do SL/TP que define o label)
    close_price = float(last['close'])
    atr_val = float(last['atr'])
    atr_pct = (atr_val / close_price * 100) if close_price > 0 else 0
    candle_body_pct = abs(float(last['close']) - float(last['open'])) / close_price * 100 if close_price > 0 else 0
    vol_series = df['volume'].iloc[-21:-1]
    vol_ma = vol_series.mean() if len(vol_series) > 0 else 1.0
    volume_ratio = (float(last['volume']) / vol_ma) if vol_ma > 0 else 1.0

    target = 1 if outcome in ('TP1_HIT', 'TP2_HIT') else 0

    return {
        'rsi': float(last['rsi']),
        'macd_histogram': float(last['macd_histogram']),
        'adx': float(last['adx']),
        'atr': float(last['atr']),
        'bb_position': float(last['bb_position']),
        'cvd': float(last.get('cvd', 0)),
        'orderbook_imbalance': float(last.get('orderbook_imbalance', 0.5)),
        'bullish_tf_count': float(last.get('bullish_tf_count', 5)),
        'bearish_tf_count': float(last.get('bearish_tf_count', 5)),
        'confidence': signal.get('confidence', 5),
        'trend_encoded': int(last.get('trend_encoded', 0)),
        'sentiment_encoded': int(last.get('sentiment_encoded', 0)),
        'signal_encoded': 1 if signal_type == 'BUY' else (-1 if signal_type == 'SELL' else 0),
        'risk_distance_pct': float(atr_pct),
        'reward_distance_pct': float(candle_body_pct),
        'risk_reward_ratio': float(volume_ratio),
        'target': target,
    }


# ============================================================
# TREINAMENTO
# ============================================================

def build_dataset_from_bootstrap() -> Optional[pd.DataFrame]:
    """Gera dataset de treinamento a partir de dados historicos da Binance."""
    print("\n[BOOTSTRAP] Gerando dataset a partir de dados historicos...")
    print("[BOOTSTRAP] Buscando klines de 15m (max dados disponiveis) para cada simbolo...")

    all_signals = []

    # Buscar multiplos intervalos para mais dados e diversidade
    intervals_and_limits = [
        ("15m", 1500),  # ~15 dias de 15m
        ("1h", 1500),   # ~62 dias de 1h
        ("4h", 1000),   # ~166 dias de 4h
    ]

    for symbol in BOOTSTRAP_SYMBOLS:
        for interval, limit in intervals_and_limits:
            print(f"  [{symbol} {interval}] Buscando klines...", end='', flush=True)
            df = fetch_klines(symbol, interval=interval, limit=limit)

            if df is None or len(df) < 200:
                print(" ERRO (dados insuficientes)")
                continue

            # Ajustar lookforward baseado no intervalo
            lookforward_map = {"15m": 48, "1h": 24, "4h": 12}
            lf = lookforward_map.get(interval, 48)

            signals = generate_bootstrap_signals(symbol, df, lookforward=lf)
            all_signals.extend(signals)
            wins = sum(1 for s in signals if s['target'] == 1)
            print(f" {len(signals)} sinais ({wins} win, {len(signals)-wins} loss)")

            time.sleep(0.3)

    if not all_signals:
        print("[ERRO] Nenhum sinal gerado no bootstrap")
        return None

    df = pd.DataFrame(all_signals)
    print(f"\n[BOOTSTRAP] Dataset total: {len(df)} amostras")
    print(f"[BOOTSTRAP] Win rate: {df['target'].mean():.1%}")
    print(f"[BOOTSTRAP] Simbolos: {df['symbol'].nunique()}")

    return df


def build_dataset_from_signals() -> Optional[pd.DataFrame]:
    """Gera dataset a partir de sinais existentes avaliados."""
    print("\n[SIGNALS] Carregando sinais avaliados...")

    evaluations = load_evaluated_signals()
    finalized = [e for e in evaluations
                 if e.get('outcome') in ('SL_HIT', 'TP1_HIT', 'TP2_HIT')]

    if len(finalized) < 20:
        print(f"[SIGNALS] Apenas {len(finalized)} sinais finalizados (minimo: 20)")
        return None

    print(f"[SIGNALS] {len(finalized)} sinais finalizados encontrados")
    print("[SIGNALS] Enriquecendo com indicadores tecnicos reais...")

    enriched = []
    for i, ev in enumerate(finalized):
        print(f"\r  Processando {i+1}/{len(finalized)}...", end='', flush=True)
        result = enrich_signal_with_klines(ev)
        if result:
            enriched.append(result)
        time.sleep(0.2)

    print(f"\n[SIGNALS] {len(enriched)} sinais enriquecidos com sucesso")

    if len(enriched) < 20:
        return None

    df = pd.DataFrame(enriched)
    print(f"[SIGNALS] Win rate: {df['target'].mean():.1%}")
    return df


def train_models(df: pd.DataFrame) -> Dict:
    """Treina ensemble de modelos usando as 16 features canonicas."""
    available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    if len(available_features) < 10:
        print(f"[ERRO] Poucas features disponiveis: {len(available_features)}")
        return {}

    X = df[available_features].fillna(0).values
    y = df['target'].values

    print(f"\n[TRAIN] Dataset: {len(X)} amostras, {len(available_features)} features")
    print(f"[TRAIN] Distribuicao: {sum(y)} TP ({sum(y)/len(y):.1%}), {len(y)-sum(y)} SL ({1-sum(y)/len(y):.1%})")

    # Split TEMPORAL: treino = 80% mais antigo, teste = 20% mais recente
    # Dados já devem estar ordenados por timestamp para evitar data leakage
    if 'timestamp' in df.columns:
        df_sorted = df.sort_values('timestamp')
        X = df_sorted[available_features].fillna(0).values
        y = df_sorted['target'].values
        print("[TRAIN] Split temporal (treino=passado, teste=futuro)")
    else:
        print("[TRAIN] Sem timestamp, usando split sequencial")

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"[TRAIN] Treino: {len(X_train)} ({sum(y_train)} TP, {len(y_train)-sum(y_train)} SL)")
    print(f"[TRAIN] Teste:  {len(X_test)} ({sum(y_test)} TP, {len(y_test)-sum(y_test)} SL)")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Walk-forward CV para dados temporais
    n_splits = min(5, max(2, len(X_train) // 30))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # =========================================================
    # FASE 1: Hyperparameter Tuning via RandomizedSearchCV
    # =========================================================
    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING + TREINAMENTO")
    print("=" * 60)

    sample_weights = compute_sample_weight('balanced', y_train)

    # Definir modelos e espaços de busca
    model_configs = {
        "LogisticRegression": {
            "model": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
            "params": {
                "C": [0.01, 0.1, 0.5, 1.0, 5.0],
                "solver": ["lbfgs", "liblinear"],
            },
            "use_sample_weight": False,
        },
        "RandomForest": {
            "model": RandomForestClassifier(class_weight="balanced", random_state=42),
            "params": {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 5, 7, 10],
                "min_samples_leaf": [3, 5, 10, 20],
            },
            "use_sample_weight": False,
        },
        "GradientBoosting": {
            "model": GradientBoostingClassifier(random_state=42),
            "params": {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 4, 5, 6],
                "learning_rate": [0.01, 0.05, 0.1],
                "min_samples_leaf": [5, 10, 20],
                "subsample": [0.7, 0.8, 0.9, 1.0],
            },
            "use_sample_weight": True,
        },
    }

    # Adicionar XGBoost se disponível
    if XGBOOST_AVAILABLE:
        model_configs["XGBoost"] = {
            "model": xgb.XGBClassifier(
                objective="binary:logistic",
                tree_method="hist",
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            ),
            "params": {
                "n_estimators": [100, 200, 300, 500],
                "max_depth": [3, 4, 5, 6, 7],
                "learning_rate": [0.01, 0.05, 0.1, 0.15],
                "subsample": [0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
                "min_child_weight": [1, 3, 5],
                "reg_alpha": [0, 0.01, 0.1],
                "reg_lambda": [1, 1.5, 2],
            },
            "use_sample_weight": True,
        }
    else:
        print("[WARN] XGBoost não instalado — excluído do ensemble")

    results = {}
    best_score = 0
    best_name = None
    trained_models = {}

    for name, config in model_configs.items():
        try:
            print(f"\n  {name}: tuning {len(config['params'])} hiperparâmetros...")

            search = RandomizedSearchCV(
                config["model"],
                config["params"],
                n_iter=min(20, np.prod([len(v) for v in config["params"].values()])),
                cv=tscv,
                scoring="f1",
                random_state=42,
                n_jobs=-1,
                error_score=0,
            )

            if config["use_sample_weight"]:
                search.fit(X_train_scaled, y_train, sample_weight=sample_weights)
            else:
                search.fit(X_train_scaled, y_train)

            model = search.best_estimator_
            trained_models[name] = model

            # Avaliar no teste (out-of-time)
            y_pred = model.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            try:
                y_proba = model.predict_proba(X_test_scaled)[:, 1]
                auc = roc_auc_score(y_test, y_proba)
            except Exception:
                auc = 0.5

            results[name] = {
                "test_accuracy": float(acc),
                "test_f1": float(f1),
                "test_auc": float(auc),
                "cv_f1_mean": float(search.best_score_),
                "best_params": search.best_params_,
            }

            # Model selection uses ONLY CV score (never test metrics)
            # to keep test set as unbiased final evaluation
            combined_score = search.best_score_

            print(f"    Best params: {search.best_params_}")
            print(f"    Test Acc:  {acc:.4f}")
            print(f"    Test F1:   {f1:.4f}")
            print(f"    AUC-ROC:   {auc:.4f}")
            print(f"    CV F1:     {search.best_score_:.4f}")
            print(f"    Combined:  {combined_score:.4f}")

            if combined_score > best_score:
                best_score = combined_score
                best_name = name

        except Exception as e:
            print(f"  {name}: ERRO - {e}")

    if not best_name:
        print("[ERRO] Nenhum modelo treinado com sucesso")
        return {}

    # =========================================================
    # FASE 2: Calibração de probabilidades
    # =========================================================
    print(f"\n{'='*60}")
    print(f"MELHOR MODELO: {best_name} (score: {best_score:.4f})")
    print(f"{'='*60}")

    best_model = trained_models[best_name]

    # Calibrar probabilidades usando CV no TREINO (nunca no test set)
    try:
        print(f"\n  Calibrando probabilidades ({best_name})...")
        calibrated = CalibratedClassifierCV(best_model, method='isotonic', cv=3)
        calibrated.fit(X_train_scaled, y_train,
                       sample_weight=sample_weights if model_configs[best_name]["use_sample_weight"] else None)
        trained_models[best_name] = calibrated
        best_model = calibrated
        print("  Calibração aplicada com sucesso (cv=3 no treino)")
    except Exception as e:
        print(f"  Calibração falhou (usando modelo sem calibração): {e}")

    y_pred_best = best_model.predict(X_test_scaled)
    print(f"\nClassification Report ({best_name}):")
    print(classification_report(y_test, y_pred_best,
                               target_names=['Loss/SL', 'Win/TP'],
                               zero_division=0))

    importance = {}
    base_model = best_model
    if hasattr(best_model, 'estimator'):
        base_model = best_model.estimator
    if hasattr(base_model, "feature_importances_"):
        for feat, imp in zip(available_features, base_model.feature_importances_):
            importance[feat] = float(imp)
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        print("Feature Importance (top 8):")
        for feat, imp in sorted_imp[:8]:
            bar = "=" * int(imp * 100)
            print(f"  {feat:25s} {imp:.4f} {bar}")

    return {
        "models": trained_models,
        "scaler": scaler,
        "feature_columns": available_features,
        "best_model": best_name,
        "results": results,
        "importance": importance,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }


def save_models(train_result: Dict, df: pd.DataFrame):
    """Salva modelos e artefatos para uso em producao."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(DATASET_DIR, exist_ok=True)

    models_path = os.path.join(MODEL_DIR, "signal_validators.pkl")
    with open(models_path, "wb") as f:
        pickle.dump(train_result["models"], f)
    print(f"[SALVO] Modelos: {models_path}")

    scaler_path = os.path.join(MODEL_DIR, "scaler_simple.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(train_result["scaler"], f)
    print(f"[SALVO] Scaler: {scaler_path}")

    best_name = train_result["best_model"]
    best_results = train_result["results"].get(best_name, {})

    info = {
        "training_date": datetime.now().isoformat(),
        "feature_columns": train_result["feature_columns"],
        "n_features": len(train_result["feature_columns"]),
        "train_samples": train_result.get("train_size", 0),
        "test_samples": train_result.get("test_size", 0),
        "best_model": best_name,
        "best_accuracy": best_results.get("test_accuracy", 0),
        "best_f1": best_results.get("test_f1", 0),
        "best_auc": best_results.get("test_auc", 0),
        "results": train_result["results"],
        "feature_importance": train_result.get("importance", {}),
        "retrain_count": 0,
    }

    info_path = os.path.join(MODEL_DIR, "model_info_simple.json")
    if os.path.exists(info_path):
        try:
            with open(info_path, 'r') as f:
                old_info = json.load(f)
                info["retrain_count"] = old_info.get("retrain_count", 0)
        except (json.JSONDecodeError, IOError):
            pass

    with open(info_path, "w") as f:
        json.dump(info, f, indent=2, default=str)
    print(f"[SALVO] Info: {info_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    feature_cols_plus_target = train_result["feature_columns"] + ['target']
    save_cols = [c for c in feature_cols_plus_target if c in df.columns]
    train_df = df[save_cols]

    train_path = os.path.join(DATASET_DIR, f"dataset_train_{timestamp}.csv")
    train_df.to_csv(train_path, index=False)

    latest_path = os.path.join(DATASET_DIR, "dataset_train_latest.csv")
    try:
        import shutil
        if os.path.exists(latest_path):
            os.remove(latest_path)
        shutil.copy(train_path, latest_path)
    except Exception:
        pass

    print(f"[SALVO] Dataset: {train_path}")

    pred_log_path = os.path.join(MODEL_DIR, "prediction_log.json")
    if not os.path.exists(pred_log_path):
        with open(pred_log_path, 'w') as f:
            json.dump([], f)

    perf_path = os.path.join(MODEL_DIR, "model_performance.json")
    perf_history = []
    if os.path.exists(perf_path):
        try:
            with open(perf_path, 'r') as f:
                perf_history = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    perf_history.append({
        'timestamp': datetime.now().isoformat(),
        'accuracy': best_results.get("test_accuracy", 0),
        'f1_score': best_results.get("test_f1", 0),
        'n_samples': len(df),
        'model': best_name,
    })
    with open(perf_path, 'w') as f:
        json.dump(perf_history, f, indent=2, default=str)


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def run_training_pipeline():
    """Pipeline completo de treinamento ML."""
    print("=" * 60)
    print("ML TRAINING PIPELINE")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)

    df = None

    # Tentar modo SIGNAL primeiro (dados reais > dados sinteticos)
    load_signal_files()
    evaluations = load_evaluated_signals()
    finalized = [e for e in evaluations
                 if e.get('outcome') in ('SL_HIT', 'TP1_HIT', 'TP2_HIT')]

    if len(finalized) >= 20:
        print(f"\n[MODE] SIGNAL MODE - {len(finalized)} sinais avaliados encontrados")
        df = build_dataset_from_signals()

    # Se nao tem sinais suficientes, usar BOOTSTRAP
    if df is None or len(df) < 20:
        print("\n[MODE] BOOTSTRAP MODE - Gerando dados a partir de historico Binance")
        df = build_dataset_from_bootstrap()

    if df is None or len(df) < 30:
        print(f"[ERRO] Dataset insuficiente ({len(df) if df is not None else 0} amostras)")
        return False

    # Treinar modelos
    print(f"\n[TRAIN] Treinando com {len(df)} amostras...")
    result = train_models(df)

    if not result:
        print("[ERRO] Falha no treinamento")
        return False

    # Salvar
    print("\n[SAVE] Salvando modelos e artefatos...")
    save_models(result, df)

    print("\n" + "=" * 60)
    print("TREINAMENTO CONCLUIDO COM SUCESSO!")
    print(f"  Melhor modelo: {result['best_model']}")
    print(f"  Dataset: {len(df)} amostras")
    print(f"  Features: {len(result['feature_columns'])}")
    print(f"  Test F1: {result['results'][result['best_model']].get('test_f1', 0):.4f}")
    print(f"  Modelos salvos em: {MODEL_DIR}/")
    print("=" * 60)

    return True


def main():
    """Entry point."""
    success = run_training_pipeline()
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
