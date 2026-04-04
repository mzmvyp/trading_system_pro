"""
Filtro de Tendência Dinâmico baseado em EMA 20/50 no timeframe 4h.

Lógica:
- Busca klines 4h da Binance PRODUCTION (fapi.binance.com)
- Calcula EMA 20 e EMA 50 (3.3 dias e 8.3 dias — adequado para day trades)
- Determina tendência: BULLISH (EMA20 > EMA50), BEARISH (EMA20 < EMA50), NEUTRAL
- Filtra sinais contra-tendência (ex: bloqueia BUY em tendência BEARISH forte)
- Zona neutra ampla: quando EMAs estão próximas (<1%) ou cruzamento recente → permite tudo
- Cache de 30min para acompanhar mudanças no 4h

NOTA: Usa 4h (não 1d) porque os trades têm alvos curtos (day trade / scalp).
O 1d bloqueava BUY em AVAX/DOT/LINK que estão em bear macro há meses,
mesmo quando no 4h havia tendência de alta clara com bons setups.
"""
import time
from typing import Any, Dict, Optional

import aiohttp
import pandas as pd

from src.core.logger import get_logger

logger = get_logger(__name__)

# Cache: {symbol: {"data": Dict, "timestamp": float}}
_trend_cache: Dict[str, Dict[str, Any]] = {}
CACHE_TTL_SECONDS = 1800  # 30min de cache (candle 4h = mais dinâmico que 1d)


async def _fetch_klines_4h(symbol: str, limit: int = 100) -> Optional[pd.DataFrame]:
    """Busca klines 4h diretamente da Binance PRODUCTION."""
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
    Retorna a tendência atual do símbolo baseada em EMA 20/50 no 4h.

    EMA20 = ~3.3 dias, EMA50 = ~8.3 dias.
    Adequado para day trades com alvos curtos.

    Returns:
        {
            "trend": "BULLISH" | "BEARISH" | "NEUTRAL",
            "ema50": float,
            "ema200": float,  # mantido para compatibilidade (agora é EMA50)
            "price": float,
            "strength": float,
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

    # Buscar dados 4h
    df = await _fetch_klines_4h(symbol, limit=100)

    if df is None or len(df) < 55:
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
    ema20 = _calculate_ema(close, 20)
    ema50 = _calculate_ema(close, 50)

    ema20_val = float(ema20.iloc[-1])
    ema50_val = float(ema50.iloc[-1])

    # Distância percentual entre EMAs
    ema_distance_pct = abs(ema20_val - ema50_val) / ema50_val * 100

    # Cruzamento recente (últimas 3 velas = 12h)
    recent_cross = False
    for i in range(-3, 0):
        if i - 1 >= -len(ema20):
            prev_diff = float(ema20.iloc[i - 1]) - float(ema50.iloc[i - 1])
            curr_diff = float(ema20.iloc[i]) - float(ema50.iloc[i])
            if prev_diff * curr_diff < 0:
                recent_cross = True
                break

    # Posição do preço em relação às EMAs
    price_above_ema20 = current_price > ema20_val
    price_above_ema50 = current_price > ema50_val

    # Zona neutra ampla (1%) — EMAs próximas ou cruzamento recente
    if ema_distance_pct < 1.0 or recent_cross:
        trend = "NEUTRAL"
        strength = ema_distance_pct / 1.0 if ema_distance_pct < 1.0 else 0.5
        allow_long = True
        allow_short = True
        description = f"Tendência NEUTRA 4h — EMAs próximas ({ema_distance_pct:.2f}%)"
        if recent_cross:
            description += " — cruzamento recente"

    elif ema20_val > ema50_val:
        trend = "BULLISH"
        strength = min(ema_distance_pct / 3.0, 1.0)
        allow_long = True
        allow_short = False

        if price_above_ema20 and price_above_ema50:
            description = f"Tendência ALTA 4h — preço acima de EMA20 e EMA50 ({ema_distance_pct:.2f}%)"
        elif price_above_ema50:
            description = f"Tendência ALTA 4h — preço entre EMAs ({ema_distance_pct:.2f}%)"
        else:
            description = f"Tendência ALTA 4h — preço testando suporte ({ema_distance_pct:.2f}%)"

    else:
        trend = "BEARISH"
        strength = min(ema_distance_pct / 3.0, 1.0)
        allow_long = False
        allow_short = True

        if not price_above_ema20 and not price_above_ema50:
            description = f"Tendência BAIXA 4h — preço abaixo de EMA20 e EMA50 ({ema_distance_pct:.2f}%)"
        elif not price_above_ema50:
            description = f"Tendência BAIXA 4h — preço entre EMAs ({ema_distance_pct:.2f}%)"
        else:
            description = f"Tendência BAIXA 4h — preço testando resistência ({ema_distance_pct:.2f}%)"

    result = {
        "trend": trend,
        "ema50": ema20_val,   # EMA rápida (era EMA50 no 1d, agora EMA20 no 4h)
        "ema200": ema50_val,  # EMA lenta (era EMA200 no 1d, agora EMA50 no 4h)
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

    logger.info(f"[TREND 4h] {symbol}: {trend} (EMA20={ema20_val:.2f}, EMA50={ema50_val:.2f}, "
                f"Preço={current_price:.2f}, Força={strength:.2f}) -> "
                f"Long={'SIM' if allow_long else 'NÃO'}, Short={'SIM' if allow_short else 'NÃO'}")

    return result
