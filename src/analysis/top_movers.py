"""
Top Movers Dinâmico — Detecta os pares com maior movimento em Binance Futures
=============================================================================

Consulta a API pública da Binance para encontrar top gainers e losers
entre todos os pares USDT de Futures, filtrando por volume mínimo.

Retorna pares que NÃO estão na lista fixa (top_crypto_pairs) para complementá-la.
"""

import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import aiohttp

from src.core.logger import get_logger

logger = get_logger(__name__)

# Binance Futures API pública
BINANCE_FUTURES_TICKER_URL = "https://fapi.binance.com/fapi/v1/ticker/24hr"
BINANCE_FUTURES_EXCHANGE_INFO_URL = "https://fapi.binance.com/fapi/v1/exchangeInfo"

# Cache de símbolos ativos (TRADING status) — atualizado junto com o cache de movers
_active_symbols_cache: Optional[set] = None
_active_symbols_cache_time: Optional[datetime] = None

# Cache para evitar chamadas repetidas dentro do mesmo ciclo
_top_movers_cache: Optional[Dict] = None
_top_movers_cache_time: Optional[datetime] = None
_CACHE_TTL_SECONDS = 300  # 5 minutos


async def _fetch_active_trading_symbols(timeout_seconds: int = 15) -> set:
    """
    Busca exchangeInfo para obter apenas símbolos com status TRADING.
    Isso filtra pares em settling, delivering, closed, pre-trading, etc.
    """
    global _active_symbols_cache, _active_symbols_cache_time

    now = datetime.now(timezone.utc)
    if (
        _active_symbols_cache is not None
        and _active_symbols_cache_time is not None
        and (now - _active_symbols_cache_time).total_seconds() < _CACHE_TTL_SECONDS
    ):
        return _active_symbols_cache

    try:
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(BINANCE_FUTURES_EXCHANGE_INFO_URL) as resp:
                if resp.status != 200:
                    logger.warning(f"[TOP_MOVERS] exchangeInfo retornou {resp.status}")
                    return _active_symbols_cache or set()
                data = await resp.json()
                active = {
                    s["symbol"]
                    for s in data.get("symbols", [])
                    if s.get("status") == "TRADING"
                }
                _active_symbols_cache = active
                _active_symbols_cache_time = now
                logger.debug(f"[TOP_MOVERS] {len(active)} símbolos ativos em Futures")
                return active
    except asyncio.TimeoutError:
        logger.warning("[TOP_MOVERS] Timeout ao buscar exchangeInfo")
        return _active_symbols_cache or set()
    except Exception as e:
        logger.warning(f"[TOP_MOVERS] Erro ao buscar exchangeInfo: {e}")
        return _active_symbols_cache or set()


async def fetch_all_futures_tickers(timeout_seconds: int = 15) -> List[Dict]:
    """
    Busca ticker 24hr de TODOS os pares de Binance Futures.
    Usa a API pública sem autenticação.
    """
    try:
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(BINANCE_FUTURES_TICKER_URL) as resp:
                if resp.status != 200:
                    logger.warning(f"[TOP_MOVERS] Binance API retornou {resp.status}")
                    return []
                return await resp.json()
    except asyncio.TimeoutError:
        logger.warning("[TOP_MOVERS] Timeout ao buscar tickers da Binance")
        return []
    except Exception as e:
        logger.warning(f"[TOP_MOVERS] Erro ao buscar tickers: {e}")
        return []


def _filter_usdt_perpetuals(tickers: List[Dict], min_volume_usdt: float = 50_000_000) -> List[Dict]:
    """
    Filtra apenas pares USDT perpétuos com volume mínimo.
    Exclui pares de baixa liquidez e tokens de nicho.
    """
    filtered = []
    for t in tickers:
        symbol = t.get("symbol", "")

        # Apenas pares USDT (excluir BUSD, margined, etc.)
        if not symbol.endswith("USDT"):
            continue

        # Excluir pares de índice ou especiais
        if "_" in symbol or symbol.startswith("1000"):
            continue

        try:
            volume_usdt = float(t.get("quoteVolume", 0))
            price_change_pct = float(t.get("priceChangePercent", 0))
            last_price = float(t.get("lastPrice", 0))
        except (ValueError, TypeError):
            continue

        # Filtrar volume mínimo (evitar shitcoins sem liquidez)
        if volume_usdt < min_volume_usdt:
            continue

        # Filtrar preço válido
        if last_price <= 0:
            continue

        # Filtrar símbolos sem trades (settling/delisted)
        trade_count = int(t.get("count", 0))
        if trade_count < 100:
            continue

        filtered.append({
            "symbol": symbol,
            "price_change_pct": price_change_pct,
            "volume_usdt": volume_usdt,
            "last_price": last_price,
            "high_price": float(t.get("highPrice", 0)),
            "low_price": float(t.get("lowPrice", 0)),
        })

    return filtered


async def get_top_movers(
    n_gainers: int = 5,
    n_losers: int = 5,
    min_volume_usdt: float = 50_000_000,
    exclude_symbols: Optional[List[str]] = None,
) -> Dict[str, List[Dict]]:
    """
    Retorna os top N gainers e top N losers de Binance Futures (24h).

    Args:
        n_gainers: Número de top gainers a retornar
        n_losers: Número de top losers a retornar
        min_volume_usdt: Volume mínimo em USDT (default 50M) para filtrar liquidez
        exclude_symbols: Lista de símbolos a excluir (ex: top_crypto_pairs já fixos)

    Returns:
        Dict com "gainers" e "losers", cada um sendo lista de dicts com:
            symbol, price_change_pct, volume_usdt, last_price
    """
    global _top_movers_cache, _top_movers_cache_time

    # Verificar cache
    now = datetime.now(timezone.utc)
    if (
        _top_movers_cache is not None
        and _top_movers_cache_time is not None
        and (now - _top_movers_cache_time).total_seconds() < _CACHE_TTL_SECONDS
    ):
        return _top_movers_cache

    exclude_set = set(exclude_symbols or [])

    # Buscar símbolos ativos (status=TRADING) e todos os tickers em paralelo
    active_symbols, tickers = await asyncio.gather(
        _fetch_active_trading_symbols(),
        fetch_all_futures_tickers(),
    )
    if not tickers:
        logger.warning("[TOP_MOVERS] Sem dados de tickers")
        return {"gainers": [], "losers": []}

    # Filtrar pares válidos
    valid = _filter_usdt_perpetuals(tickers, min_volume_usdt)

    # Filtrar apenas símbolos com status TRADING (exclui settling, delivering, closed)
    if active_symbols:
        before = len(valid)
        valid = [t for t in valid if t["symbol"] in active_symbols]
        filtered_out = before - len(valid)
        if filtered_out > 0:
            logger.info(f"[TOP_MOVERS] {filtered_out} símbolos removidos (não-TRADING)")

    # Excluir pares já na lista fixa
    valid = [t for t in valid if t["symbol"] not in exclude_set]

    # Ordenar por variação de preço
    sorted_by_change = sorted(valid, key=lambda x: x["price_change_pct"], reverse=True)

    gainers = sorted_by_change[:n_gainers]
    losers = sorted_by_change[-n_losers:][::-1]  # Reverter para maior queda primeiro

    result = {"gainers": gainers, "losers": losers}

    # Atualizar cache
    _top_movers_cache = result
    _top_movers_cache_time = now

    # Log
    if gainers:
        g_str = ", ".join(f"{g['symbol']}(+{g['price_change_pct']:.1f}%)" for g in gainers)
        logger.info(f"[TOP_MOVERS] Gainers: {g_str}")
    if losers:
        l_str = ", ".join(f"{loser['symbol']}({loser['price_change_pct']:.1f}%)" for loser in losers)
        logger.info(f"[TOP_MOVERS] Losers: {l_str}")

    return result


async def get_dynamic_symbols(
    fixed_symbols: List[str],
    n_gainers: int = 5,
    n_losers: int = 5,
    min_volume_usdt: float = 50_000_000,
) -> Tuple[List[str], Dict[str, Dict]]:
    """
    Retorna lista combinada de símbolos (fixos + dinâmicos) e metadata dos dinâmicos.

    Args:
        fixed_symbols: Lista fixa de símbolos (settings.top_crypto_pairs)
        n_gainers: Quantos top gainers adicionar
        n_losers: Quantos top losers adicionar
        min_volume_usdt: Volume mínimo para filtrar

    Returns:
        Tuple de:
            - Lista combinada de símbolos (fixos primeiro, depois dinâmicos)
            - Dict de metadata dos dinâmicos: {symbol: {price_change_pct, volume_usdt, mover_type}}
    """
    movers = await get_top_movers(
        n_gainers=n_gainers,
        n_losers=n_losers,
        min_volume_usdt=min_volume_usdt,
        exclude_symbols=fixed_symbols,
    )

    dynamic_symbols = []
    dynamic_metadata = {}

    for g in movers.get("gainers", []):
        sym = g["symbol"]
        if sym not in dynamic_symbols:
            dynamic_symbols.append(sym)
            dynamic_metadata[sym] = {
                "price_change_pct": g["price_change_pct"],
                "volume_usdt": g["volume_usdt"],
                "mover_type": "gainer",
            }

    for loser in movers.get("losers", []):
        sym = loser["symbol"]
        if sym not in dynamic_symbols:
            dynamic_symbols.append(sym)
            dynamic_metadata[sym] = {
                "price_change_pct": loser["price_change_pct"],
                "volume_usdt": loser["volume_usdt"],
                "mover_type": "loser",
            }

    combined = list(fixed_symbols) + dynamic_symbols

    total_dynamic = len(dynamic_symbols)
    if total_dynamic > 0:
        logger.info(
            f"[TOP_MOVERS] Adicionados {total_dynamic} pares dinâmicos "
            f"(total: {len(combined)} pares para análise)"
        )

    return combined, dynamic_metadata
