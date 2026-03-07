"""
Market sentiment analysis based on market data
"""
from typing import Dict, Any
from datetime import datetime

from src.core.logger import get_logger
from src.analysis.market_data import get_market_data

logger = get_logger(__name__)


async def analyze_market_sentiment(symbol: str = "BTCUSDT") -> Dict[str, Any]:
    """Análise de sentimento baseada em dados de mercado (preço, volume, funding rate)."""
    try:
        market_data = await get_market_data(symbol)
        if "error" in market_data:
            return market_data

        price_change = market_data['price_change_24h']
        volume = market_data['volume_24h']
        funding_rate = market_data['funding_rate']
        open_interest = market_data.get('open_interest', 0)

        sentiment_score = 0
        confidence = 0.5

        # 1. Variação de preço
        if price_change > 8:
            sentiment_score += 3; confidence += 0.3
        elif price_change > 3:
            sentiment_score += 2; confidence += 0.2
        elif price_change > 1:
            sentiment_score += 1; confidence += 0.1
        elif price_change < -8:
            sentiment_score -= 3; confidence += 0.3
        elif price_change < -3:
            sentiment_score -= 2; confidence += 0.2
        elif price_change < -1:
            sentiment_score -= 1; confidence += 0.1

        # 2. Volume
        if volume > 2000000:
            sentiment_score += 2; confidence += 0.2
        elif volume > 1000000:
            sentiment_score += 1; confidence += 0.1
        elif volume < 50000:
            sentiment_score -= 1; confidence += 0.1

        # 3. Funding rate
        if funding_rate > 0.02:
            sentiment_score += 2; confidence += 0.2
        elif funding_rate > 0.005:
            sentiment_score += 1; confidence += 0.1
        elif funding_rate < -0.02:
            sentiment_score -= 2; confidence += 0.2
        elif funding_rate < -0.005:
            sentiment_score -= 1; confidence += 0.1

        # 4. Open Interest
        if open_interest > 100000000:
            sentiment_score += 1; confidence += 0.1
        elif open_interest < 10000000:
            sentiment_score -= 1; confidence += 0.1

        # Cap confidence before final determination
        confidence = min(confidence, 1.0)

        # Determine final sentiment
        if sentiment_score >= 3:
            sentiment = "very_positive"
            final_confidence = min(0.95, confidence)
        elif sentiment_score >= 1:
            sentiment = "positive"
            final_confidence = min(0.9, confidence)
        elif sentiment_score <= -3:
            sentiment = "very_negative"
            final_confidence = min(0.95, confidence)
        elif sentiment_score <= -1:
            sentiment = "negative"
            final_confidence = min(0.9, confidence)
        else:
            sentiment = "neutral"
            final_confidence = min(0.8, confidence)

        return {
            "symbol": symbol,
            "sentiment": sentiment,
            "confidence": float(final_confidence),
            "factors": {
                "price_change": price_change,
                "volume_level": "high" if volume > 1000000 else "low" if volume < 100000 else "normal",
                "funding_rate": funding_rate,
                "open_interest": open_interest
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "error": f"Erro na análise de sentimento: {str(e)}",
            "symbol": symbol,
            "timestamp": datetime.now().isoformat()
        }
