"""
Filtro de Tendência Dinâmico baseado em EMA 50/200 no timeframe 4h.

Lógica:
- Busca klines 4h da Binance PRODUCTION (fapi.binance.com)
- Calcula EMA 50 e EMA 200
- Determina tendência: BULLISH (EMA50 > EMA200), BEARISH (EMA50 < EMA200), NEUTRAL (cruzamento recente)
- Filtra sinais contra-tendência (ex: bloqueia BUY em tendência BEARISH)
- Zona neutra: quando EMAs estão muito próximas (<0.3%), permite ambas as direções
- Cache de 1h para não sobrecarregar a API

Adaptativo: quando o mercado virar de bearish para bullish, o filtro
automaticamente começa a permitir longs novamente.
"""
import time
from typing import Any, Dict, Optional

import aiohttp
import pandas as pd

from src.core.logger import get_logger

logger = get_logger(__name__)

# Cache: {symbol: {"trend": str, "ema50": float, "ema200": float, "timestamp": float, "strength": float}}
_trend_cache: Dict[str, Dict[str, Any]] = {}
CACHE_TTL_SECONDS = 3600  # 1 hora de cache


async def _fetch_klines_4h(symbol: str, limit: int = 250) -> Optional[pd.DataFrame]:
    """Busca klines 4h diretamente da Binance PRODUCTION (não testnet)."""
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {
        "symbol": symbol,
        "interval": "4h",
        "limit": limit
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    logger.warning(f"[TREND] Erro ao buscar klines 4h de {symbol}: HTTP {resp.status}")
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
        logger.warning(f"[TREND] Erro ao buscar klines 4h de {symbol}: {e}")
        return None


def _calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Calcula EMA usando pandas."""
    return series.ewm(span=period, adjust=False).mean()


async def get_trend(symbol: str) -> Dict[str, Any]:
    """
    Retorna a tendência atual do símbolo baseada em EMA 50/200 no 4h.

    Returns:
        {
            "trend": "BULLISH" | "BEARISH" | "NEUTRAL",
            "ema50": float,
            "ema200": float,
            "price": float,
            "strength": float,  # 0-1, quão forte é a tendência
            "description": str,
            "allow_long": bool,
            "allow_short": bool
        }
    """
    # Verificar cache
    now = time.time()
    if symbol in _trend_cache:
        cached = _trend_cache[symbol]
        if now - cached["timestamp"] < CACHE_TTL_SECONDS:
            return cached["data"]

    # Buscar dados
    df = await _fetch_klines_4h(symbol, limit=250)

    if df is None or len(df) < 200:
        # Sem dados suficientes: permitir tudo (fail-open)
        logger.warning(f"[TREND] Dados insuficientes para {symbol}, permitindo todos os sinais")
        result = {
            "trend": "UNKNOWN",
            "ema50": 0,
            "ema200": 0,
            "price": 0,
            "strength": 0,
            "description": "Dados insuficientes para determinar tendência",
            "allow_long": True,
            "allow_short": True
        }
        return result

    close = df['close']
    current_price = float(close.iloc[-1])
    ema50 = _calculate_ema(close, 50)
    ema200 = _calculate_ema(close, 200)

    ema50_val = float(ema50.iloc[-1])
    ema200_val = float(ema200.iloc[-1])

    # Calcular distância percentual entre EMAs
    ema_distance_pct = abs(ema50_val - ema200_val) / ema200_val * 100

    # Verificar se houve cruzamento recente (últimas 5 velas = 20h)
    recent_cross = False
    for i in range(-5, 0):
        if i - 1 >= -len(ema50):
            prev_diff = float(ema50.iloc[i - 1]) - float(ema200.iloc[i - 1])
            curr_diff = float(ema50.iloc[i]) - float(ema200.iloc[i])
            if prev_diff * curr_diff < 0:  # Mudou de sinal
                recent_cross = True
                break

    # Verificar posição do preço em relação às EMAs
    price_above_ema50 = current_price > ema50_val
    price_above_ema200 = current_price > ema200_val

    # Determinar tendência
    if ema_distance_pct < 0.3 or recent_cross:
        # EMAs muito próximas ou cruzamento recente = NEUTRO
        trend = "NEUTRAL"
        strength = ema_distance_pct / 0.3 if ema_distance_pct < 0.3 else 0.5
        allow_long = True
        allow_short = True
        description = f"Tendência NEUTRA - EMAs muito próximas ({ema_distance_pct:.2f}%)"
        if recent_cross:
            description += " - Cruzamento recente detectado"

    elif ema50_val > ema200_val:
        # EMA 50 acima da 200 = BULLISH
        trend = "BULLISH"
        strength = min(ema_distance_pct / 2.0, 1.0)  # Normalizar para 0-1

        if price_above_ema50 and price_above_ema200:
            # Preço acima de ambas EMAs = tendência forte
            allow_long = True
            allow_short = False
            description = f"Tendência ALTA forte - preço acima de EMA50 e EMA200 ({ema_distance_pct:.2f}%)"
        elif price_above_ema200:
            # Preço entre EMAs = pode estar corrigindo
            allow_long = True
            allow_short = True  # Permite short como correção
            description = f"Tendência ALTA com correção - preço entre EMAs ({ema_distance_pct:.2f}%)"
        else:
            # Preço abaixo de ambas mas EMAs bullish = possível reversão
            allow_long = True
            allow_short = True
            description = f"Tendência ALTA fraca - preço abaixo das EMAs ({ema_distance_pct:.2f}%)"

    else:
        # EMA 50 abaixo da 200 = BEARISH
        trend = "BEARISH"
        strength = min(ema_distance_pct / 2.0, 1.0)

        if not price_above_ema50 and not price_above_ema200:
            # Preço abaixo de ambas = tendência forte
            allow_long = False
            allow_short = True
            description = f"Tendência BAIXA forte - preço abaixo de EMA50 e EMA200 ({ema_distance_pct:.2f}%)"
        elif not price_above_ema200:
            # Preço entre EMAs = pode estar corrigindo
            allow_long = True  # Permite long como bounce
            allow_short = True
            description = f"Tendência BAIXA com correção - preço entre EMAs ({ema_distance_pct:.2f}%)"
        else:
            # Preço acima de ambas mas EMAs bearish = possível reversão
            allow_long = True
            allow_short = True
            description = f"Tendência BAIXA fraca - preço acima das EMAs ({ema_distance_pct:.2f}%)"

    result = {
        "trend": trend,
        "ema50": ema50_val,
        "ema200": ema200_val,
        "price": current_price,
        "strength": strength,
        "description": description,
        "allow_long": allow_long,
        "allow_short": allow_short
    }

    # Salvar cache
    _trend_cache[symbol] = {
        "data": result,
        "timestamp": now
    }

    logger.info(f"[TREND] {symbol}: {trend} (EMA50={ema50_val:.2f}, EMA200={ema200_val:.2f}, "
                f"Preço={current_price:.2f}, Força={strength:.2f}) -> "
                f"Long={'SIM' if allow_long else 'NÃO'}, Short={'SIM' if allow_short else 'NÃO'}")

    return result


def clear_cache(symbol: str = None):
    """Limpa o cache de tendência."""
    if symbol:
        _trend_cache.pop(symbol, None)
    else:
        _trend_cache.clear()
