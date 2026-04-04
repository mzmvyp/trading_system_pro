"""
Filtro de Tendência Multi-Timeframe com Confluência.

Lógica:
- Busca klines de múltiplos timeframes (15m, 1h, 4h, 1d) da Binance PRODUCTION
- Calcula EMA 20 e EMA 50 em cada timeframe
- Determina tendência por TF: BULLISH (EMA20 > EMA50), BEARISH, NEUTRAL
- Usa pesos por timeframe para confluência:
    15m = 1.0, 1h = 1.5, 4h = 2.0, 1d = 1.0
  (4h tem peso máximo pois é o sweet spot para day trades;
   1d tem peso reduzido para não vetar sinais curtos em bear macro)
- Decisão final baseada em score ponderado:
    Score > 0 → tendência bullish, Score < 0 → tendência bearish
    |Score| < threshold → neutro (permite tudo)
- Bloqueio só acontece quando há forte confluência contra a direção do trade
- Cache de 15min por símbolo

NOTA: Substituiu o filtro single-timeframe (4h) que não cruzava tendências.
O usuário pediu cruzamento entre 1d, 4h, 30m, 15m para confluência.
"""
import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import pandas as pd

from src.core.logger import get_logger

logger = get_logger(__name__)

# Cache: {symbol: {"data": Dict, "timestamp": float}}
_trend_cache: Dict[str, Dict[str, Any]] = {}
CACHE_TTL_SECONDS = 900  # 15min — mais responsivo com múltiplos TFs

# Timeframes e seus pesos para day trading
# 4h é o sweet spot: captura tendência intraday sem ruído do 5m/15m
# 1d informa mas NÃO veta (peso reduzido) — evita bloquear BUY em bear macro
# 15m/1h capturam momentum de curto prazo
TIMEFRAME_CONFIG: List[Tuple[str, float, int]] = [
    # (interval, weight, min_candles_needed)
    ("15m", 1.0, 55),
    ("1h",  1.5, 55),
    ("4h",  2.0, 55),
    ("1d",  1.0, 55),
]

# Threshold para bloqueio — score ponderado precisa ser forte para bloquear
# Score máximo possível: 1.0+1.5+2.0+1.0 = 5.5 (todos TFs alinhados)
# Bloqueia apenas quando score >= 3.0 (>54% do máximo)
BLOCK_THRESHOLD = 3.0

# Zona neutra por TF: se EMAs estão dentro deste % uma da outra, TF é neutro
NEUTRAL_ZONE_PCT = {
    "15m": 0.15,   # 15m é mais ruidoso, zona neutra menor
    "1h":  0.3,
    "4h":  0.8,
    "1d":  1.2,    # 1d precisa de mais separação para ser significativo
}


async def _fetch_klines(symbol: str, interval: str, limit: int = 100) -> Optional[pd.DataFrame]:
    """Busca klines diretamente da Binance PRODUCTION."""
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    logger.warning(f"[TREND-MTF] Erro ao buscar klines {interval} de {symbol}: HTTP {resp.status}")
                    return None
                data = await resp.json()

        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base_volume',
            'taker_buy_quote_volume', 'ignore'
        ])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df

    except Exception as e:
        logger.warning(f"[TREND-MTF] Erro ao buscar klines {interval} de {symbol}: {e}")
        return None


def _calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Calcula EMA usando pandas."""
    return series.ewm(span=period, adjust=False).mean()


def _analyze_single_tf(df: pd.DataFrame, interval: str) -> Dict[str, Any]:
    """
    Analisa tendência de um único timeframe usando EMA20/50.

    Returns:
        {
            "trend": "BULLISH" | "BEARISH" | "NEUTRAL",
            "ema20": float,
            "ema50": float,
            "price": float,
            "distance_pct": float,  # distância entre EMAs em %
            "recent_cross": bool,
        }
    """
    close = df['close']
    current_price = float(close.iloc[-1])
    ema20 = _calculate_ema(close, 20)
    ema50 = _calculate_ema(close, 50)

    ema20_val = float(ema20.iloc[-1])
    ema50_val = float(ema50.iloc[-1])

    # Distância percentual entre EMAs
    if ema50_val > 0:
        distance_pct = abs(ema20_val - ema50_val) / ema50_val * 100
    else:
        distance_pct = 0

    # Cruzamento recente (últimas 3 candles)
    recent_cross = False
    for i in range(-3, 0):
        if i - 1 >= -len(ema20):
            prev_diff = float(ema20.iloc[i - 1]) - float(ema50.iloc[i - 1])
            curr_diff = float(ema20.iloc[i]) - float(ema50.iloc[i])
            if prev_diff * curr_diff < 0:
                recent_cross = True
                break

    # Determinar tendência do TF
    neutral_zone = NEUTRAL_ZONE_PCT.get(interval, 0.5)

    if distance_pct < neutral_zone or recent_cross:
        trend = "NEUTRAL"
    elif ema20_val > ema50_val:
        trend = "BULLISH"
    else:
        trend = "BEARISH"

    return {
        "trend": trend,
        "ema20": ema20_val,
        "ema50": ema50_val,
        "price": current_price,
        "distance_pct": distance_pct,
        "recent_cross": recent_cross,
    }


async def get_trend(symbol: str) -> Dict[str, Any]:
    """
    Retorna a tendência atual baseada em confluência multi-timeframe.

    Analisa 15m, 1h, 4h, 1d e calcula score ponderado.
    Só bloqueia trades quando há forte confluência contra a direção.

    Returns:
        {
            "trend": "BULLISH" | "BEARISH" | "NEUTRAL",
            "ema50": float,       # EMA rápida do 4h (compatibilidade)
            "ema200": float,      # EMA lenta do 4h (compatibilidade)
            "price": float,
            "strength": float,    # 0.0 a 1.0
            "description": str,
            "allow_long": bool,
            "allow_short": bool,
            "timeframes": dict,   # detalhes por TF
            "confluence_score": float,
        }
    """
    # Verificar cache
    now = time.time()
    if symbol in _trend_cache:
        cached = _trend_cache[symbol]
        if now - cached["timestamp"] < CACHE_TTL_SECONDS:
            return cached["data"]

    # Buscar todos os timeframes em paralelo
    fetch_tasks = []
    for interval, weight, min_candles in TIMEFRAME_CONFIG:
        fetch_tasks.append(_fetch_klines(symbol, interval, limit=100))

    klines_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

    # Analisar cada timeframe
    tf_analyses = {}
    bullish_score = 0.0
    bearish_score = 0.0
    total_weight = 0.0
    ema20_4h = 0.0
    ema50_4h = 0.0
    current_price = 0.0

    for idx, (interval, weight, min_candles) in enumerate(TIMEFRAME_CONFIG):
        df = klines_results[idx]

        if isinstance(df, Exception) or df is None or len(df) < min_candles:
            logger.debug(f"[TREND-MTF] {symbol} {interval}: dados insuficientes, ignorando")
            continue

        analysis = _analyze_single_tf(df, interval)
        tf_analyses[interval] = analysis

        if analysis["trend"] == "BULLISH":
            bullish_score += weight
        elif analysis["trend"] == "BEARISH":
            bearish_score += weight
        # NEUTRAL contribui 0 para ambos

        total_weight += weight
        current_price = analysis["price"]

        # Guardar EMAs do 4h para compatibilidade
        if interval == "4h":
            ema20_4h = analysis["ema20"]
            ema50_4h = analysis["ema50"]

    # Se não conseguiu dados de nenhum TF, permitir tudo
    if total_weight == 0:
        logger.warning(f"[TREND-MTF] Sem dados para {symbol}, permitindo todos os sinais")
        result = {
            "trend": "UNKNOWN",
            "ema50": 0,
            "ema200": 0,
            "price": 0,
            "strength": 0,
            "description": "Dados insuficientes para determinar tendência",
            "allow_long": True,
            "allow_short": True,
            "timeframes": {},
            "confluence_score": 0,
        }
        return result

    # Calcular score de confluência
    # Positivo = bullish, negativo = bearish
    confluence_score = bullish_score - bearish_score
    max_possible = sum(w for _, w, _ in TIMEFRAME_CONFIG)
    strength = abs(confluence_score) / max_possible  # 0.0 a 1.0

    # Determinar tendência geral e permissões
    if confluence_score >= BLOCK_THRESHOLD:
        trend = "BULLISH"
        allow_long = True
        allow_short = False
    elif confluence_score <= -BLOCK_THRESHOLD:
        trend = "BEARISH"
        allow_long = False
        allow_short = True
    else:
        # Confluência insuficiente para bloquear — permite tudo
        if confluence_score > 0:
            trend = "NEUTRAL_BULLISH"
        elif confluence_score < 0:
            trend = "NEUTRAL_BEARISH"
        else:
            trend = "NEUTRAL"
        allow_long = True
        allow_short = True

    # Montar descrição
    tf_summary = []
    for interval, _, _ in TIMEFRAME_CONFIG:
        if interval in tf_analyses:
            t = tf_analyses[interval]["trend"]
            d = tf_analyses[interval]["distance_pct"]
            emoji = {"BULLISH": "ALTA", "BEARISH": "BAIXA", "NEUTRAL": "NEUTRA"}[t]
            tf_summary.append(f"{interval}={emoji}({d:.1f}%)")

    description = f"MTF Confluência: {' | '.join(tf_summary)} → Score={confluence_score:+.1f}/{max_possible:.1f}"

    if not allow_long:
        description += " [BLOQUEIA LONG]"
    if not allow_short:
        description += " [BLOQUEIA SHORT]"

    result = {
        "trend": trend,
        "ema50": ema20_4h,    # EMA rápida 4h (compatibilidade)
        "ema200": ema50_4h,   # EMA lenta 4h (compatibilidade)
        "price": current_price,
        "strength": strength,
        "description": description,
        "allow_long": allow_long,
        "allow_short": allow_short,
        "timeframes": tf_analyses,
        "confluence_score": confluence_score,
    }

    # Salvar cache
    _trend_cache[symbol] = {
        "data": result,
        "timestamp": now
    }

    # Log detalhado
    tf_log = " | ".join(
        f"{tf}:{a['trend']}({a['distance_pct']:.1f}%)"
        for tf, a in tf_analyses.items()
    )
    logger.info(
        f"[TREND-MTF] {symbol}: {trend} (Score={confluence_score:+.1f}, "
        f"Força={strength:.2f}) [{tf_log}] → "
        f"Long={'SIM' if allow_long else 'NÃO'}, Short={'SIM' if allow_short else 'NÃO'}"
    )

    return result
