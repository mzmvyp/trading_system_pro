"""
Market Regime Filter - BTC-based regime filter for signal validation.
Source: trade_bot_new
Rules:
- STRONG_BULLISH: Block SHORTs
- STRONG_BEARISH: Block LONGs
- BULLISH/BEARISH: Warning but ALLOWS
- NEUTRAL: Allows both
"""

import logging
from datetime import datetime
from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np

from src.core.logger import get_logger

logger = get_logger(__name__)


class MarketRegime(Enum):
    STRONG_BULLISH = "STRONG_BULLISH"
    BULLISH = "BULLISH"
    NEUTRAL = "NEUTRAL"
    BEARISH = "BEARISH"
    STRONG_BEARISH = "STRONG_BEARISH"


class MarketRegimeFilter:
    """
    Filters trading signals based on BTC market regime.
    Blocks signals that contradict the overall market trend.
    """

    def __init__(self):
        self.current_regime = MarketRegime.NEUTRAL
        self.last_analysis_time: Optional[datetime] = None
        self.regime_confidence = 0.0
        self.btc_data: Dict = {}
        self.cache_duration_minutes = 15

    async def analyze_btc_regime(self, force: bool = False, client=None) -> Dict:
        """Analyze BTC regime using multi-timeframe analysis."""
        if not force and self._is_cache_valid():
            return {
                "regime": self.current_regime.value,
                "confidence": self.regime_confidence,
                "cached": True,
            }

        try:
            if client is None:
                from src.exchange.client import BinanceClient
                client = BinanceClient()

            klines_1h = await client.get_klines("BTCUSDT", "1h", limit=100)
            klines_4h = await client.get_klines("BTCUSDT", "4h", limit=50)

            if klines_1h is None or len(klines_1h) < 50:
                return {"regime": "NEUTRAL", "confidence": 0}

            close_1h = klines_1h["close"].values.astype(float)
            high_1h = klines_1h["high"].values.astype(float)
            low_1h = klines_1h["low"].values.astype(float)
            current_price = float(close_1h[-1])

            # Calculate indicators
            ema_20_1h = self._ema(close_1h, 20)
            ema_50_1h = self._ema(close_1h, 50)

            close_4h = klines_4h["close"].values.astype(float) if klines_4h is not None and len(klines_4h) > 20 else close_1h
            ema_20_4h = self._ema(close_4h, 20)
            ema_50_4h = self._ema(close_4h, 50)

            adx_1h = self._adx(high_1h, low_1h, close_1h)
            rsi_1h = self._rsi(close_1h)
            rsi_4h = self._rsi(close_4h)

            # MACD
            ema12 = self._ema(close_1h, 12)
            ema26 = self._ema(close_1h, 26)
            macd_hist = (ema12 - ema26) - self._ema(np.array([ema12 - ema26][-1:] if len(close_1h) < 50 else [0]), 9)
            macd_bullish = (ema12 - ema26) > 0

            # 24h price change approximation
            price_change_24h = ((current_price - float(close_1h[-24])) / float(close_1h[-24]) * 100) if len(close_1h) >= 24 else 0

            # Scoring
            bullish_score, bearish_score, total_weight = 0, 0, 0

            checks = [
                (current_price > ema_20_1h, 3),
                (current_price > ema_50_1h, 3),
                (ema_20_1h > ema_50_1h, 4),
                (current_price > ema_20_4h, 3),
                (ema_20_4h > ema_50_4h, 4),
            ]

            for check, weight in checks:
                if check:
                    bullish_score += weight
                else:
                    bearish_score += weight
                total_weight += weight

            # Price change
            if price_change_24h > 2:
                bullish_score += 2
            elif price_change_24h < -2:
                bearish_score += 2
            total_weight += 2

            # MACD
            if macd_bullish:
                bullish_score += 2
            else:
                bearish_score += 2
            total_weight += 2

            # RSI
            avg_rsi = (rsi_1h + rsi_4h) / 2
            if avg_rsi > 55:
                bullish_score += 2
            elif avg_rsi < 45:
                bearish_score += 2
            total_weight += 2

            # Net score
            net_score = (bullish_score - bearish_score) / total_weight if total_weight > 0 else 0
            trend_strength = min(adx_1h / 50, 1.0)

            if net_score > 0.6:
                regime = MarketRegime.STRONG_BULLISH
                confidence = min(0.9, 0.7 + net_score * 0.3)
            elif net_score > 0.3:
                regime = MarketRegime.BULLISH
                confidence = min(0.8, 0.5 + net_score * 0.4)
            elif net_score < -0.6:
                regime = MarketRegime.STRONG_BEARISH
                confidence = min(0.9, 0.7 + abs(net_score) * 0.3)
            elif net_score < -0.3:
                regime = MarketRegime.BEARISH
                confidence = min(0.8, 0.5 + abs(net_score) * 0.4)
            else:
                regime = MarketRegime.NEUTRAL
                confidence = 0.5

            if regime != MarketRegime.NEUTRAL:
                confidence *= (0.7 + 0.3 * trend_strength)

            self.current_regime = regime
            self.regime_confidence = confidence
            self.last_analysis_time = datetime.now()
            self.btc_data = {
                "current_price": current_price,
                "price_change_24h": price_change_24h,
                "net_score": net_score,
                "adx": adx_1h,
            }

            return {
                "regime": regime.value,
                "confidence": confidence,
                "btc_data": self.btc_data,
            }

        except Exception as e:
            logger.error(f"BTC regime analysis error: {e}")
            return {"regime": "NEUTRAL", "confidence": 0, "error": str(e)}

    def _is_cache_valid(self) -> bool:
        if not self.last_analysis_time:
            return False
        return (datetime.now() - self.last_analysis_time).total_seconds() < self.cache_duration_minutes * 60

    def should_allow_signal(self, signal_type: str) -> Tuple[bool, str]:
        """Check if a signal should be allowed based on current regime."""
        regime = self.current_regime

        if regime == MarketRegime.STRONG_BULLISH and signal_type in ("SELL", "SELL_SHORT"):
            return False, "BLOCKED: STRONG_BULLISH - SHORTs not allowed"

        if regime == MarketRegime.STRONG_BEARISH and signal_type in ("BUY", "BUY_LONG"):
            return False, "BLOCKED: STRONG_BEARISH - LONGs not allowed"

        if regime == MarketRegime.BULLISH and signal_type in ("SELL", "SELL_SHORT"):
            return True, "WARNING: BULLISH regime - SHORTs have elevated risk"

        if regime == MarketRegime.BEARISH and signal_type in ("BUY", "BUY_LONG"):
            return True, "WARNING: BEARISH regime - LONGs have elevated risk"

        return True, "OK"

    def get_allowed_signals(self) -> list:
        if self.current_regime == MarketRegime.STRONG_BULLISH:
            return ["BUY", "BUY_LONG"]
        if self.current_regime == MarketRegime.STRONG_BEARISH:
            return ["SELL", "SELL_SHORT"]
        return ["BUY", "BUY_LONG", "SELL", "SELL_SHORT"]

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> float:
        """Calculate EMA and return latest value."""
        import pandas as pd
        return float(pd.Series(data).ewm(span=period, adjust=False).mean().iloc[-1])

    @staticmethod
    def _rsi(data: np.ndarray, period: int = 14) -> float:
        import pandas as pd
        s = pd.Series(data)
        delta = s.diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1])

    @staticmethod
    def _adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        import pandas as pd
        h, l, c = pd.Series(high), pd.Series(low), pd.Series(close)
        up = h.diff()
        down = -l.diff()
        plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0))
        minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0))
        tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        pdi = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / (atr + 1e-10))
        mdi = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / (atr + 1e-10))
        dx = 100 * (abs(pdi - mdi) / (pdi + mdi + 1e-10))
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        return float(adx.iloc[-1])
