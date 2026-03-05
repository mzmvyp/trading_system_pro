"""
Divergence Detector - Detects divergences between price and indicators.
Source: smart_trading_system
Indicators: RSI, MACD, Stochastic, Momentum, Williams %R, CCI
Types: Bullish Regular, Bearish Regular, Bullish Hidden, Bearish Hidden
Scoring: strength (WEAK/MODERATE/STRONG/VERY_STRONG), confidence, reliability
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.core.logger import get_logger

logger = get_logger(__name__)


class DivergenceType(Enum):
    BULLISH_REGULAR = "BULLISH_REGULAR"
    BEARISH_REGULAR = "BEARISH_REGULAR"
    BULLISH_HIDDEN = "BULLISH_HIDDEN"
    BEARISH_HIDDEN = "BEARISH_HIDDEN"


class DivergenceStrength(Enum):
    WEAK = "WEAK"
    MODERATE = "MODERATE"
    STRONG = "STRONG"
    VERY_STRONG = "VERY_STRONG"


@dataclass
class Divergence:
    divergence_type: DivergenceType
    indicator: str
    strength: DivergenceStrength
    confidence: float  # 0-1
    reliability: float  # 0-1
    price_point_1: float
    price_point_2: float
    indicator_point_1: float
    indicator_point_2: float
    candles_apart: int
    metadata: Dict = field(default_factory=dict)


class DivergenceDetector:
    """
    Detects divergences between price action and technical indicators.
    Supports regular and hidden divergences across multiple indicators.
    """

    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        self.lookback = config.get("lookback", 30)
        self.pivot_len = config.get("pivot_len", 5)
        self.min_candles_apart = config.get("min_candles_apart", 5)
        self.max_candles_apart = config.get("max_candles_apart", 50)
        self.min_confidence = config.get("min_confidence", 0.4)

        # Indicator reliability weights
        self.reliability_weights = {
            "RSI": 0.85,
            "MACD": 0.80,
            "STOCHASTIC": 0.70,
            "CCI": 0.65,
            "MOMENTUM": 0.60,
            "WILLIAMS_R": 0.65,
        }

    def detect_all(self, df: pd.DataFrame) -> List[Divergence]:
        """Detect divergences across all indicators."""
        if len(df) < self.lookback + 10:
            return []

        divergences = []

        # Calculate indicators
        indicators = self._calculate_indicators(df)

        # Find swing points in price
        price_highs = self._find_pivots(df["high"].values, "HIGH")
        price_lows = self._find_pivots(df["low"].values, "LOW")

        # Check each indicator for divergences
        for name, series in indicators.items():
            ind_highs = self._find_pivots(series, "HIGH")
            ind_lows = self._find_pivots(series, "LOW")

            # Regular bullish: price makes LL, indicator makes HL
            for div in self._check_regular_bullish(price_lows, ind_lows, name):
                if div.confidence >= self.min_confidence:
                    divergences.append(div)

            # Regular bearish: price makes HH, indicator makes LH
            for div in self._check_regular_bearish(price_highs, ind_highs, name):
                if div.confidence >= self.min_confidence:
                    divergences.append(div)

            # Hidden bullish: price makes HL, indicator makes LL
            for div in self._check_hidden_bullish(price_lows, ind_lows, name):
                if div.confidence >= self.min_confidence:
                    divergences.append(div)

            # Hidden bearish: price makes LH, indicator makes HH
            for div in self._check_hidden_bearish(price_highs, ind_highs, name):
                if div.confidence >= self.min_confidence:
                    divergences.append(div)

        # Sort by confidence
        divergences.sort(key=lambda d: d.confidence, reverse=True)
        return divergences

    def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calculate all indicator series for divergence detection."""
        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)

        indicators = {}

        # RSI
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
        avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
        rs = avg_gain / (avg_loss + 1e-10)
        indicators["RSI"] = 100 - (100 / (1 + rs))

        # MACD histogram
        ema12 = pd.Series(close).ewm(span=12, adjust=False).mean().values
        ema26 = pd.Series(close).ewm(span=26, adjust=False).mean().values
        macd_line = ema12 - ema26
        signal_line = pd.Series(macd_line).ewm(span=9, adjust=False).mean().values
        indicators["MACD"] = macd_line - signal_line

        # Stochastic %K
        k_period = 14
        for i in range(k_period, len(close)):
            pass  # Pre-calculate
        lowest = pd.Series(low).rolling(k_period).min().values
        highest = pd.Series(high).rolling(k_period).max().values
        indicators["STOCHASTIC"] = 100 * (close - lowest) / (highest - lowest + 1e-10)

        # CCI
        typical_price = (high + low + close) / 3
        tp_sma = pd.Series(typical_price).rolling(20).mean().values
        tp_mad = pd.Series(typical_price).rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean()))).values
        indicators["CCI"] = (typical_price - tp_sma) / (0.015 * tp_mad + 1e-10)

        # Momentum
        period = 10
        indicators["MOMENTUM"] = np.concatenate([np.zeros(period), close[period:] - close[:-period]])

        # Williams %R
        indicators["WILLIAMS_R"] = -100 * (highest - close) / (highest - lowest + 1e-10)

        return indicators

    def _find_pivots(self, data: np.ndarray, pivot_type: str) -> List[Tuple[int, float]]:
        """Find pivot points (swing highs/lows)."""
        pivots = []
        n = len(data)

        for i in range(self.pivot_len, n - self.pivot_len):
            if pivot_type == "HIGH":
                is_pivot = all(data[i] >= data[i - j] for j in range(1, self.pivot_len + 1)) and \
                           all(data[i] >= data[i + j] for j in range(1, self.pivot_len + 1))
            else:
                is_pivot = all(data[i] <= data[i - j] for j in range(1, self.pivot_len + 1)) and \
                           all(data[i] <= data[i + j] for j in range(1, self.pivot_len + 1))

            if is_pivot:
                pivots.append((i, float(data[i])))

        return pivots

    def _check_regular_bullish(
        self, price_lows: List[Tuple[int, float]], ind_lows: List[Tuple[int, float]], indicator: str
    ) -> List[Divergence]:
        """Regular bullish: price LL, indicator HL."""
        return self._check_divergence(
            price_lows, ind_lows, indicator,
            DivergenceType.BULLISH_REGULAR,
            price_condition=lambda p1, p2: p2 < p1,  # Price lower low
            ind_condition=lambda i1, i2: i2 > i1,    # Indicator higher low
        )

    def _check_regular_bearish(
        self, price_highs: List[Tuple[int, float]], ind_highs: List[Tuple[int, float]], indicator: str
    ) -> List[Divergence]:
        """Regular bearish: price HH, indicator LH."""
        return self._check_divergence(
            price_highs, ind_highs, indicator,
            DivergenceType.BEARISH_REGULAR,
            price_condition=lambda p1, p2: p2 > p1,  # Price higher high
            ind_condition=lambda i1, i2: i2 < i1,    # Indicator lower high
        )

    def _check_hidden_bullish(
        self, price_lows: List[Tuple[int, float]], ind_lows: List[Tuple[int, float]], indicator: str
    ) -> List[Divergence]:
        """Hidden bullish: price HL, indicator LL."""
        return self._check_divergence(
            price_lows, ind_lows, indicator,
            DivergenceType.BULLISH_HIDDEN,
            price_condition=lambda p1, p2: p2 > p1,  # Price higher low
            ind_condition=lambda i1, i2: i2 < i1,    # Indicator lower low
        )

    def _check_hidden_bearish(
        self, price_highs: List[Tuple[int, float]], ind_highs: List[Tuple[int, float]], indicator: str
    ) -> List[Divergence]:
        """Hidden bearish: price LH, indicator HH."""
        return self._check_divergence(
            price_highs, ind_highs, indicator,
            DivergenceType.BEARISH_HIDDEN,
            price_condition=lambda p1, p2: p2 < p1,  # Price lower high
            ind_condition=lambda i1, i2: i2 > i1,    # Indicator higher high
        )

    def _check_divergence(
        self, price_pivots, ind_pivots, indicator: str,
        div_type: DivergenceType, price_condition, ind_condition,
    ) -> List[Divergence]:
        """Generic divergence checker."""
        divergences = []

        if len(price_pivots) < 2 or len(ind_pivots) < 2:
            return divergences

        # Check last few pivot pairs
        for i in range(len(price_pivots) - 1):
            p1_idx, p1_val = price_pivots[i]
            p2_idx, p2_val = price_pivots[i + 1]

            candles_apart = abs(p2_idx - p1_idx)
            if candles_apart < self.min_candles_apart or candles_apart > self.max_candles_apart:
                continue

            if not price_condition(p1_val, p2_val):
                continue

            # Find matching indicator pivots
            for j in range(len(ind_pivots) - 1):
                i1_idx, i1_val = ind_pivots[j]
                i2_idx, i2_val = ind_pivots[j + 1]

                # Check temporal alignment
                if abs(i1_idx - p1_idx) > 3 or abs(i2_idx - p2_idx) > 3:
                    continue

                if not ind_condition(i1_val, i2_val):
                    continue

                # Calculate strength
                price_diff = abs(p2_val - p1_val) / (p1_val + 1e-10)
                ind_diff = abs(i2_val - i1_val) / (abs(i1_val) + 1e-10)

                strength = self._calculate_strength(price_diff, ind_diff, candles_apart)
                reliability = self.reliability_weights.get(indicator, 0.5)
                confidence = (strength.value == "VERY_STRONG") * 0.3 + \
                             (strength.value == "STRONG") * 0.25 + \
                             (strength.value == "MODERATE") * 0.2 + \
                             0.3 * reliability + 0.2 * min(candles_apart / 20, 1.0)

                divergences.append(Divergence(
                    divergence_type=div_type,
                    indicator=indicator,
                    strength=strength,
                    confidence=confidence,
                    reliability=reliability,
                    price_point_1=p1_val,
                    price_point_2=p2_val,
                    indicator_point_1=i1_val,
                    indicator_point_2=i2_val,
                    candles_apart=candles_apart,
                ))
                break  # Take first match per price pivot pair

        return divergences

    def _calculate_strength(self, price_diff: float, ind_diff: float, candles: int) -> DivergenceStrength:
        """Calculate divergence strength."""
        score = 0
        if price_diff > 0.02:
            score += 2
        elif price_diff > 0.01:
            score += 1
        if ind_diff > 0.1:
            score += 2
        elif ind_diff > 0.05:
            score += 1
        if candles >= 15:
            score += 1

        if score >= 4:
            return DivergenceStrength.VERY_STRONG
        elif score >= 3:
            return DivergenceStrength.STRONG
        elif score >= 2:
            return DivergenceStrength.MODERATE
        return DivergenceStrength.WEAK

    def get_summary(self, divergences: List[Divergence]) -> Dict:
        """Get a summary of detected divergences."""
        if not divergences:
            return {"count": 0, "bias": "NEUTRAL", "strongest": None}

        bullish = [d for d in divergences if "BULLISH" in d.divergence_type.value]
        bearish = [d for d in divergences if "BEARISH" in d.divergence_type.value]

        bias = "NEUTRAL"
        if len(bullish) > len(bearish):
            bias = "BULLISH"
        elif len(bearish) > len(bullish):
            bias = "BEARISH"

        strongest = max(divergences, key=lambda d: d.confidence)

        return {
            "count": len(divergences),
            "bullish_count": len(bullish),
            "bearish_count": len(bearish),
            "bias": bias,
            "strongest": {
                "type": strongest.divergence_type.value,
                "indicator": strongest.indicator,
                "strength": strongest.strength.value,
                "confidence": strongest.confidence,
            },
        }
