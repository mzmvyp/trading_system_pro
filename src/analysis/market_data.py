"""
Market data collection from Binance
"""
from datetime import datetime
from typing import Any, Dict

from src.core.constants import DEFAULT_KLINES_LIMIT
from src.core.logger import get_logger

logger = get_logger(__name__)


async def get_market_data(symbol: str = "BTCUSDT") -> Dict[str, Any]:
    """
    Obtém dados de mercado da Binance para análise.
    """
    try:
        from src.exchange.client import BinanceClient

        logger.debug(f"Fetching market data for {symbol}")

        async with BinanceClient() as client:
            ticker = await client.get_ticker_24hr(symbol)
            klines_df = await client.get_klines(symbol, '1h', limit=DEFAULT_KLINES_LIMIT)
            funding = await client.get_funding_rate(symbol)
            open_interest = await client.get_open_interest(symbol)

        result = {
            "symbol": symbol,
            "current_price": float(ticker['lastPrice']),
            "price_change_24h": float(ticker['priceChangePercent']),
            "volume_24h": float(ticker['volume']),
            "high_24h": float(ticker['highPrice']),
            "low_24h": float(ticker['lowPrice']),
            "funding_rate": float(funding.get('lastFundingRate', 0)),
            "open_interest": float(open_interest.get('openInterest', 0)),
            "klines_count": len(klines_df),
            "timestamp": datetime.now().isoformat()
        }

        logger.info(f"Market data fetched for {symbol}: ${result['current_price']:.2f}")
        return result

    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        logger.exception(f"[{symbol}] Erro ao obter dados de mercado ({error_type}): {error_msg}")
        return {
            "error": f"Erro ao obter dados de mercado ({error_type}): {error_msg}",
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type
        }
