"""
Order flow analysis
"""
from datetime import datetime
from typing import Any, Dict

import aiohttp

from src.core.logger import get_logger

logger = get_logger(__name__)


async def analyze_order_flow(symbol: str) -> Dict[str, Any]:
    """Análise de fluxo de ordens e delta."""
    try:
        from src.exchange.client import BinanceClient

        async with BinanceClient() as client:
            orderbook = await client.get_orderbook(symbol, limit=20)

            bid_volume = sum([float(b[1]) for b in orderbook['bids'][:20]])
            ask_volume = sum([float(a[1]) for a in orderbook['asks'][:20]])

            total_volume = bid_volume + ask_volume
            imbalance = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{client.base_url}/fapi/v1/aggTrades",
                    params={'symbol': symbol, 'limit': 100},
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as trades_response:
                    buy_volume = 0
                    sell_volume = 0
                    if trades_response.status == 200:
                        trades = await trades_response.json()
                        for trade in trades:
                            if trade['m']:
                                sell_volume += float(trade['q'])
                            else:
                                buy_volume += float(trade['q'])

            cvd = buy_volume - sell_volume

            return {
                "symbol": symbol,
                "orderbook_imbalance": imbalance,
                "bid_volume": bid_volume,
                "ask_volume": ask_volume,
                "cvd": cvd,
                "buy_volume": buy_volume,
                "sell_volume": sell_volume,
                "buy_pressure": buy_volume > sell_volume * 1.2,
                "timestamp": datetime.now().isoformat()
            }

    except Exception as e:
        logger.exception(f"Erro na análise de order flow: {e}")
        return {
            "error": f"Erro na análise de order flow: {str(e)}",
            "symbol": symbol
        }
