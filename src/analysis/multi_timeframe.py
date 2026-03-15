"""
Multi-timeframe analysis
"""
from datetime import datetime, timezone
from typing import Any, Dict

import pandas as pd
import talib

from src.core.logger import get_logger

logger = get_logger(__name__)


async def analyze_multiple_timeframes(symbol: str) -> Dict[str, Any]:
    """Análise multi-timeframe para maior precisão."""
    try:
        from src.exchange.client import BinanceClient

        timeframes = ['5m', '15m', '1h', '4h', '1d']
        analyses = {}

        async with BinanceClient() as client:
            for tf in timeframes:
                try:
                    # exclude_forming=True: remove candle incompleto para evitar sinais falsos
                    klines_df = await client.get_klines(symbol, tf, limit=100, exclude_forming=True)

                    if not klines_df.empty and len(klines_df) >= 20:
                        df = klines_df.reset_index()
                        for col in ['open', 'high', 'low', 'close', 'volume']:
                            if col in df.columns:
                                df[col] = pd.to_numeric(df[col], errors='coerce')

                        close_prices = df['close'].values
                        sma_20 = talib.SMA(close_prices, timeperiod=20)[-1]
                        current_price = close_prices[-1]

                        if current_price > sma_20:
                            trend = "bullish"
                        elif current_price < sma_20:
                            trend = "bearish"
                        else:
                            trend = "neutral"

                        analyses[tf] = {
                            "trend": trend,
                            "current_price": float(current_price),
                            "sma_20": float(sma_20)
                        }
                except Exception as e:
                    logger.warning(f"Erro no timeframe {tf}: {e}")
                    continue

        bullish_timeframes = sum(1 for tf in analyses.values() if tf['trend'] == 'bullish')
        bearish_timeframes = sum(1 for tf in analyses.values() if tf['trend'] == 'bearish')
        neutral_timeframes = sum(1 for tf in analyses.values() if tf['trend'] == 'neutral')

        if bullish_timeframes > bearish_timeframes:
            confluence = "bullish"
        elif bearish_timeframes > bullish_timeframes:
            confluence = "bearish"
        else:
            confluence = "neutral"

        return {
            "symbol": symbol,
            "timeframes": analyses,
            "confluence": confluence,
            "bullish_count": bullish_timeframes,
            "bearish_count": bearish_timeframes,
            "neutral_count": neutral_timeframes,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        return {
            "error": f"Erro na análise multi-timeframe: {str(e)}",
            "symbol": symbol
        }
