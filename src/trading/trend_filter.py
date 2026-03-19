"""
Filtro de Tendência Dinâmico baseado em EMA 50/200 no timeframe DIÁRIO (1d).

Lógica:
- Busca klines 1d da Binance PRODUCTION (fapi.binance.com)
- Calcula EMA 50 e EMA 200 (50 dias e 200 dias — padrão institucional)
- Determina tendência: BULLISH (EMA50 > EMA200), BEARISH (EMA50 < EMA200), NEUTRAL (cruzamento recente)
- Filtra sinais contra-tendência (ex: bloqueia BUY em tendência BEARISH)
- Zona neutra: quando EMAs estão muito próximas (<0.5%), permite ambas as direções
- Cache de 1h para não sobrecarregar a API

NOTA: Usa timeframe diário (não 4h) para capturar a tendência MACRO real.
No 4h, um bounce de curto prazo fazia EMA50>EMA200 e o sistema achava
que ETH (caiu 55% de $4957 para $2199) estava em tendência de alta.
Com 1d, EMA50=50 dias e EMA200=200 dias — captura a tendência real.
"""
import time
from typing import Any, Dict, Optional

import aiohttp
import pandas as pd

from src.core.logger import get_logger

logger = get_logger(__name__)

# Cache: {symbol: {"trend": str, "ema50": float, "ema200": float, "timestamp": float, "strength": float}}
_trend_cache: Dict[str, Dict[str, Any]] = {}
CACHE_TTL_SECONDS = 3600  # 1 hora de cache (candle diário muda 1x/dia)


async def _fetch_klines_1d(symbol: str, limit: int = 250) -> Optional[pd.DataFrame]:
    """Busca klines diários diretamente da Binance PRODUCTION."""
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {
        "symbol": symbol,
        "interval": "1d",
        "limit": limit
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    logger.warning(f"[TREND] Erro ao buscar klines 1d de {symbol}: HTTP {resp.status}")
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
        logger.warning(f"[TREND] Erro ao buscar klines 1d de {symbol}: {e}")
        return None


def _calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Calcula EMA usando pandas."""
    return series.ewm(span=period, adjust=False).mean()


async def get_trend(symbol: str) -> Dict[str, Any]:
    """
    Retorna a tendência atual do símbolo baseada em EMA 50/200 no DIÁRIO.

    EMA50 = média de 50 dias, EMA200 = média de 200 dias.
    Padrão institucional para determinar tendência macro.

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

    # Buscar dados diários
    df = await _fetch_klines_1d(symbol, limit=250)

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

    # Verificar se houve cruzamento recente (últimas 5 velas = 5 dias)
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
    # Zona neutra mais ampla (0.5%) para diário — evita flip-flop em consolidação
    if ema_distance_pct < 0.5 or recent_cross:
        # EMAs muito próximas ou cruzamento recente = NEUTRO
        trend = "NEUTRAL"
        strength = ema_distance_pct / 0.5 if ema_distance_pct < 0.5 else 0.5
        allow_long = True
        allow_short = True
        description = f"Tendência NEUTRA - EMAs 1d muito próximas ({ema_distance_pct:.2f}%)"
        if recent_cross:
            description += " - Cruzamento recente detectado"

    elif ema50_val > ema200_val:
        # EMA 50 acima da 200 no DIÁRIO = BULLISH macro
        trend = "BULLISH"
        strength = min(ema_distance_pct / 5.0, 1.0)
        allow_long = True
        allow_short = False

        if price_above_ema50 and price_above_ema200:
            description = f"Tendência ALTA forte 1d - preço acima de EMA50 e EMA200 ({ema_distance_pct:.2f}%)"
        elif price_above_ema200:
            description = f"Tendência ALTA 1d - preço corrigindo entre EMAs ({ema_distance_pct:.2f}%)"
        else:
            description = f"Tendência ALTA 1d - preço testando suporte ({ema_distance_pct:.2f}%)"

    else:
        # EMA 50 abaixo da 200 no DIÁRIO = BEARISH macro
        trend = "BEARISH"
        strength = min(ema_distance_pct / 5.0, 1.0)
        allow_long = False
        allow_short = True

        if not price_above_ema50 and not price_above_ema200:
            description = f"Tendência BAIXA forte 1d - preço abaixo de EMA50 e EMA200 ({ema_distance_pct:.2f}%)"
        elif not price_above_ema200:
            description = f"Tendência BAIXA 1d - preço corrigindo entre EMAs ({ema_distance_pct:.2f}%)"
        else:
            description = f"Tendência BAIXA 1d - preço testando resistência ({ema_distance_pct:.2f}%)"

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

    logger.info(f"[TREND 1d] {symbol}: {trend} (EMA50={ema50_val:.2f}, EMA200={ema200_val:.2f}, "
                f"Preço={current_price:.2f}, Força={strength:.2f}) -> "
                f"Long={'SIM' if allow_long else 'NÃO'}, Short={'SIM' if allow_short else 'NÃO'}")

    return result


def clear_cache(symbol: str = None):
    """Limpa o cache de tendência."""
    if symbol:
        _trend_cache.pop(symbol, None)
    else:
        _trend_cache.clear()
