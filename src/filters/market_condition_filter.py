"""
Market Condition Filter - Multi-dimensional market state assessment.
Source: smart_trading_system
Conditions: 9 states including trending, ranging, choppy, breakout, reversal
Features: Multi-TF assessment, Fear/Greed-like score, strategy compatibility
"""

import logging
from enum import Enum
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.core.logger import get_logger

logger = get_logger(__name__)


class MarketCondition(Enum):
    STRONG_UPTREND = "STRONG_UPTREND"
    UPTREND = "UPTREND"
    RANGE_BOUND = "RANGE_BOUND"
    DOWNTREND = "DOWNTREND"
    STRONG_DOWNTREND = "STRONG_DOWNTREND"
    CHOPPY = "CHOPPY"
    BREAKOUT = "BREAKOUT"
    REVERSAL = "REVERSAL"
    UNCERTAIN = "UNCERTAIN"


class MarketConditionFilter:
    """
    Assess overall market condition and filter trades accordingly.
    Combines trend, volatility, and momentum into market state.
    """

    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        self.adx_threshold = config.get("adx_threshold", 25)
        self.rsi_overbought = config.get("rsi_overbought", 70)
        self.rsi_oversold = config.get("rsi_oversold", 30)

        # Strategy-condition compatibility
        self.compatible = {
            "trend_following": [
                MarketCondition.STRONG_UPTREND, MarketCondition.UPTREND,
                MarketCondition.DOWNTREND, MarketCondition.STRONG_DOWNTREND,
            ],
            "mean_reversion": [MarketCondition.RANGE_BOUND, MarketCondition.CHOPPY],
            "breakout": [MarketCondition.RANGE_BOUND, MarketCondition.BREAKOUT],
            "swing": [
                MarketCondition.UPTREND, MarketCondition.DOWNTREND,
                MarketCondition.RANGE_BOUND,
            ],
            "scalp": [
                MarketCondition.RANGE_BOUND, MarketCondition.UPTREND,
                MarketCondition.DOWNTREND,
            ],
        }

    def analyze(self, df: pd.DataFrame) -> Dict:
        """Analyze current market condition."""
        if len(df) < 50:
            return {"condition": MarketCondition.UNCERTAIN.value, "score": 50}

        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)

        # Trend indicators
        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        adx = self._calculate_adx(df)
        rsi = self._calculate_rsi(close)

        current_price = float(close.iloc[-1])
        adx_val = float(adx.iloc[-1])
        rsi_val = float(rsi.iloc[-1])
        ema20_val = float(ema20.iloc[-1])
        ema50_val = float(ema50.iloc[-1])

        # Choppiness detection
        tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
        atr14 = tr.ewm(span=14, adjust=False).mean()
        directional_move = abs(float(close.iloc[-1]) - float(close.iloc[-14]))
        choppiness = 1 - (directional_move / (float(atr14.iloc[-1]) * 14 + 1e-10))

        # Determine condition
        if adx_val > 40 and current_price > ema20_val > ema50_val:
            condition = MarketCondition.STRONG_UPTREND
        elif adx_val > self.adx_threshold and current_price > ema50_val:
            condition = MarketCondition.UPTREND
        elif adx_val > 40 and current_price < ema20_val < ema50_val:
            condition = MarketCondition.STRONG_DOWNTREND
        elif adx_val > self.adx_threshold and current_price < ema50_val:
            condition = MarketCondition.DOWNTREND
        elif choppiness > 0.7 and adx_val < 20:
            condition = MarketCondition.CHOPPY
        elif adx_val < self.adx_threshold:
            condition = MarketCondition.RANGE_BOUND
        else:
            condition = MarketCondition.UNCERTAIN

        # Fear/Greed-like score (0-100)
        # 0=extreme fear, 50=neutral, 100=extreme greed
        fg_score = 50
        fg_score += (rsi_val - 50) * 0.5  # RSI contribution
        fg_score += (adx_val - 25) * 0.3  # Trend strength
        if current_price > ema50_val:
            fg_score += 10
        else:
            fg_score -= 10
        fg_score = max(0, min(100, fg_score))

        return {
            "condition": condition.value,
            "score": round(fg_score, 1),
            "adx": round(adx_val, 1),
            "rsi": round(rsi_val, 1),
            "choppiness": round(choppiness, 3),
            "trend_direction": "UP" if current_price > ema50_val else "DOWN",
        }

    def should_trade(self, condition_data: Dict, strategy_type: str = "trend_following") -> Dict:
        """Check if market condition is compatible with strategy."""
        condition = MarketCondition(condition_data["condition"])
        allowed_conditions = self.compatible.get(strategy_type, [])

        is_compatible = condition in allowed_conditions

        reason = "OK" if is_compatible else f"{condition.value} not suitable for {strategy_type}"

        # Choppy market warning
        if condition == MarketCondition.CHOPPY:
            reason = "CHOPPY market - avoid trend-following strategies"

        return {
            "allowed": is_compatible,
            "reason": reason,
            "condition": condition.value,
            "fear_greed_score": condition_data.get("score", 50),
        }

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        h, l, c = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
        up = h.diff()
        down = -l.diff()
        plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
        minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
        tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        pdi = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / (atr + 1e-10))
        mdi = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / (atr + 1e-10))
        dx = 100 * (abs(pdi - mdi) / (pdi + mdi + 1e-10))
        return dx.ewm(alpha=1/period, adjust=False).mean().fillna(20)

    def _calculate_rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
        rs = gain / (loss + 1e-10)
        return (100 - (100 / (1 + rs))).fillna(50)
