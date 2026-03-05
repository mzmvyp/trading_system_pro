"""
Volatility Filter - Filter trades based on volatility regime.
Source: smart_trading_system
Regimes: VERY_LOW, LOW, NORMAL, HIGH, EXTREME
Features: ATR-based, BB width, historical percentile, regime-specific behavior
"""

import logging
from enum import Enum
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.core.logger import get_logger

logger = get_logger(__name__)


class VolatilityRegime(Enum):
    VERY_LOW = "VERY_LOW"      # < 10th percentile
    LOW = "LOW"                # 10-30th percentile
    NORMAL = "NORMAL"          # 30-70th percentile
    HIGH = "HIGH"              # 70-90th percentile
    EXTREME = "EXTREME"        # > 90th percentile


class VolatilityFilter:
    """
    Filter trades based on volatility conditions.
    Each strategy type has optimal volatility ranges.
    """

    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        self.lookback = config.get("lookback", 100)
        self.atr_period = config.get("atr_period", 14)

        # Strategy-specific volatility preferences
        self.strategy_preferences = {
            "breakout": [VolatilityRegime.LOW, VolatilityRegime.NORMAL],  # Breakout from low vol
            "trend_following": [VolatilityRegime.NORMAL, VolatilityRegime.HIGH],
            "mean_reversion": [VolatilityRegime.HIGH, VolatilityRegime.EXTREME],
            "swing": [VolatilityRegime.NORMAL, VolatilityRegime.HIGH],
            "scalp": [VolatilityRegime.LOW, VolatilityRegime.NORMAL],
        }

    def analyze(self, df: pd.DataFrame) -> Dict:
        """Analyze current volatility regime."""
        if len(df) < self.lookback:
            return {"regime": VolatilityRegime.NORMAL.value, "can_trade": True, "details": {}}

        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)

        # ATR
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(span=self.atr_period, adjust=False).mean()
        natr = atr / close  # Normalized ATR

        current_natr = float(natr.iloc[-1])

        # Historical percentile
        lookback_natr = natr.iloc[-self.lookback:]
        percentile = float((lookback_natr < current_natr).mean() * 100)

        # BB width
        sma = close.rolling(20).mean()
        std = close.rolling(20).std()
        bbw = ((sma + 2 * std) - (sma - 2 * std)) / close
        current_bbw = float(bbw.iloc[-1])

        # Determine regime
        if percentile < 10:
            regime = VolatilityRegime.VERY_LOW
        elif percentile < 30:
            regime = VolatilityRegime.LOW
        elif percentile < 70:
            regime = VolatilityRegime.NORMAL
        elif percentile < 90:
            regime = VolatilityRegime.HIGH
        else:
            regime = VolatilityRegime.EXTREME

        # Compression detection (potential breakout setup)
        is_compressed = current_bbw < float(bbw.iloc[-self.lookback:].quantile(0.15))

        # Expansion detection
        is_expanding = current_natr > float(natr.iloc[-5:].mean()) * 1.3

        return {
            "regime": regime.value,
            "natr": current_natr,
            "percentile": percentile,
            "bbw": current_bbw,
            "is_compressed": is_compressed,
            "is_expanding": is_expanding,
        }

    def should_trade(self, regime_data: Dict, strategy_type: str = "trend_following") -> Dict:
        """Check if current volatility allows trading for a given strategy."""
        regime = VolatilityRegime(regime_data["regime"])
        preferred = self.strategy_preferences.get(strategy_type, [VolatilityRegime.NORMAL])

        allowed = regime in preferred
        reason = "OK" if allowed else f"Volatility {regime.value} not optimal for {strategy_type}"

        # Special cases
        if regime == VolatilityRegime.EXTREME:
            reason = "EXTREME volatility - reduced position size recommended"
            # Allow but with warning
            allowed = True

        if regime_data.get("is_compressed") and strategy_type == "breakout":
            allowed = True
            reason = "Compression detected - breakout setup"

        return {
            "allowed": allowed,
            "reason": reason,
            "regime": regime.value,
            "recommended_size_factor": self._get_size_factor(regime),
        }

    def _get_size_factor(self, regime: VolatilityRegime) -> float:
        """Get position size multiplier based on volatility."""
        factors = {
            VolatilityRegime.VERY_LOW: 1.2,
            VolatilityRegime.LOW: 1.0,
            VolatilityRegime.NORMAL: 1.0,
            VolatilityRegime.HIGH: 0.7,
            VolatilityRegime.EXTREME: 0.5,
        }
        return factors.get(regime, 1.0)
