"""
Breakout Strategy - Detection and trading of consolidation breakouts.
Source: smart_trading_system
Features:
- Consolidation pattern detection (RANGE/TRIANGLE/FLAG)
- Volume surge confirmation (2x average threshold)
- Retest opportunity identification
- False breakout filtering via price action
- Confluence scoring system
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.core.logger import get_logger

logger = get_logger(__name__)


class PatternType(Enum):
    RANGE = "RANGE"
    TRIANGLE = "TRIANGLE"
    FLAG = "FLAG"
    WEDGE = "WEDGE"
    PENNANT = "PENNANT"


class BreakoutDirection(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"


@dataclass
class ConsolidationPattern:
    pattern_type: PatternType
    upper_bound: float
    lower_bound: float
    duration_candles: int
    volume_profile: str  # "DECLINING", "FLAT", "INCREASING"
    start_index: int
    end_index: int
    tightening: bool = False  # For triangles/wedges

    @property
    def range_width(self) -> float:
        return self.upper_bound - self.lower_bound

    @property
    def range_pct(self) -> float:
        mid = (self.upper_bound + self.lower_bound) / 2
        return (self.range_width / mid) * 100 if mid > 0 else 0


@dataclass
class BreakoutSetup:
    pattern: ConsolidationPattern
    breakout_price: float
    breakout_direction: BreakoutDirection
    volume_surge: float  # Ratio vs average
    retest_opportunity: bool = False
    retest_level: Optional[float] = None


@dataclass
class BreakoutSignal:
    setup: BreakoutSetup
    direction: str  # "BUY" or "SELL"
    strength: str  # "WEAK", "MODERATE", "STRONG", "VERY_STRONG"
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    confluence_score: float  # 0-1
    risk_reward: float
    metadata: Dict = field(default_factory=dict)


class BreakoutStrategy:
    """
    Breakout trading strategy with consolidation detection,
    volume confirmation, and false breakout filtering.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.min_consolidation_candles = self.config.get("min_consolidation_candles", 10)
        self.max_consolidation_candles = self.config.get("max_consolidation_candles", 100)
        self.volume_surge_threshold = self.config.get("volume_surge_threshold", 2.0)
        self.min_range_pct = self.config.get("min_range_pct", 0.5)
        self.max_range_pct = self.config.get("max_range_pct", 5.0)
        self.false_breakout_threshold = self.config.get("false_breakout_threshold", 0.3)
        self.retest_tolerance_pct = self.config.get("retest_tolerance_pct", 0.2)
        self.min_confluence_score = self.config.get("min_confluence_score", 0.5)

    def detect_consolidation(self, df: pd.DataFrame) -> List[ConsolidationPattern]:
        """Detect consolidation patterns in price data."""
        patterns = []
        if len(df) < self.min_consolidation_candles:
            return patterns

        highs = df["high"].values
        lows = df["low"].values
        df["close"].values
        volumes = df["volume"].values

        # Sliding window approach
        for window_size in range(self.min_consolidation_candles, min(self.max_consolidation_candles, len(df) - 5)):
            start_idx = len(df) - window_size - 1
            end_idx = len(df) - 1

            window_highs = highs[start_idx:end_idx]
            window_lows = lows[start_idx:end_idx]
            window_volumes = volumes[start_idx:end_idx]

            upper = np.max(window_highs)
            lower = np.min(window_lows)
            mid = (upper + lower) / 2
            range_pct = ((upper - lower) / mid) * 100 if mid > 0 else 0

            if range_pct < self.min_range_pct or range_pct > self.max_range_pct:
                continue

            # Check if price stayed within range (90%+ of candles)
            within_range = np.sum(
                (window_highs <= upper * 1.002) & (window_lows >= lower * 0.998)
            )
            if within_range / len(window_highs) < 0.9:
                continue

            # Determine pattern type
            pattern_type = self._classify_pattern(window_highs, window_lows)

            # Volume profile
            vol_first_half = np.mean(window_volumes[:len(window_volumes)//2])
            vol_second_half = np.mean(window_volumes[len(window_volumes)//2:])
            if vol_second_half < vol_first_half * 0.8:
                volume_profile = "DECLINING"
            elif vol_second_half > vol_first_half * 1.2:
                volume_profile = "INCREASING"
            else:
                volume_profile = "FLAT"

            # Check tightening (for triangles)
            tightening = self._check_tightening(window_highs, window_lows)

            pattern = ConsolidationPattern(
                pattern_type=pattern_type,
                upper_bound=upper,
                lower_bound=lower,
                duration_candles=window_size,
                volume_profile=volume_profile,
                start_index=start_idx,
                end_index=end_idx,
                tightening=tightening,
            )
            patterns.append(pattern)
            break  # Take the first valid pattern

        return patterns

    def _classify_pattern(self, highs: np.ndarray, lows: np.ndarray) -> PatternType:
        """Classify the consolidation pattern type."""
        n = len(highs)
        if n < 4:
            return PatternType.RANGE

        # Check for converging highs/lows (triangle)
        high_slope = np.polyfit(range(n), highs, 1)[0]
        low_slope = np.polyfit(range(n), lows, 1)[0]

        if high_slope < 0 and low_slope > 0:
            return PatternType.TRIANGLE
        elif high_slope < 0 and abs(low_slope) < abs(high_slope) * 0.3:
            return PatternType.WEDGE
        elif abs(high_slope) < 0.001 and abs(low_slope) < 0.001:
            return PatternType.RANGE
        elif high_slope < 0 and low_slope < 0:
            return PatternType.FLAG

        return PatternType.RANGE

    def _check_tightening(self, highs: np.ndarray, lows: np.ndarray) -> bool:
        """Check if the range is tightening (converging)."""
        n = len(highs)
        if n < 6:
            return False
        first_half_range = np.max(highs[:n//2]) - np.min(lows[:n//2])
        second_half_range = np.max(highs[n//2:]) - np.min(lows[n//2:])
        return second_half_range < first_half_range * 0.75

    def identify_breakout(
        self, df: pd.DataFrame, pattern: ConsolidationPattern
    ) -> Optional[BreakoutSetup]:
        """Check if a breakout is occurring from the consolidation pattern."""
        if len(df) < 2:
            return None

        current = df.iloc[-1]
        prev = df.iloc[-2]
        current_close = current["close"]
        current_volume = current["volume"]

        # Average volume during consolidation
        consol_volumes = df["volume"].iloc[pattern.start_index:pattern.end_index]
        avg_volume = consol_volumes.mean() if len(consol_volumes) > 0 else current_volume

        volume_surge = current_volume / avg_volume if avg_volume > 0 else 1.0

        # Bullish breakout
        if current_close > pattern.upper_bound:
            breakout_pct = ((current_close - pattern.upper_bound) / pattern.upper_bound) * 100

            # Filter false breakout - price should close convincingly above
            if breakout_pct < self.false_breakout_threshold:
                return None

            return BreakoutSetup(
                pattern=pattern,
                breakout_price=current_close,
                breakout_direction=BreakoutDirection.BULLISH,
                volume_surge=volume_surge,
                retest_opportunity=False,
                retest_level=pattern.upper_bound,
            )

        # Bearish breakout
        elif current_close < pattern.lower_bound:
            breakout_pct = ((pattern.lower_bound - current_close) / pattern.lower_bound) * 100

            if breakout_pct < self.false_breakout_threshold:
                return None

            return BreakoutSetup(
                pattern=pattern,
                breakout_price=current_close,
                breakout_direction=BreakoutDirection.BEARISH,
                volume_surge=volume_surge,
                retest_opportunity=False,
                retest_level=pattern.lower_bound,
            )

        # Check for retest after breakout
        if prev["close"] > pattern.upper_bound and abs(current_close - pattern.upper_bound) / pattern.upper_bound * 100 < self.retest_tolerance_pct:
            return BreakoutSetup(
                pattern=pattern,
                breakout_price=current_close,
                breakout_direction=BreakoutDirection.BULLISH,
                volume_surge=volume_surge,
                retest_opportunity=True,
                retest_level=pattern.upper_bound,
            )

        return None

    def confirm_with_volume(self, setup: BreakoutSetup) -> bool:
        """Confirm breakout with volume surge."""
        return setup.volume_surge >= self.volume_surge_threshold

    def generate_signals(
        self, df: pd.DataFrame, market_data: Optional[Dict] = None
    ) -> List[BreakoutSignal]:
        """Generate breakout trading signals."""
        signals = []

        patterns = self.detect_consolidation(df)
        if not patterns:
            return signals

        for pattern in patterns:
            setup = self.identify_breakout(df, pattern)
            if setup is None:
                continue

            # Calculate confluence score
            confluence = self._calculate_confluence(setup, df, market_data)
            if confluence < self.min_confluence_score:
                continue

            # Volume confirmation bonus
            volume_confirmed = self.confirm_with_volume(setup)

            # Calculate entry, stop, targets
            atr = self._calculate_atr(df)

            if setup.breakout_direction == BreakoutDirection.BULLISH:
                direction = "BUY"
                entry = setup.breakout_price
                stop_loss = pattern.lower_bound - atr * 0.5
                target_range = pattern.range_width
                target_1 = entry + target_range
                target_2 = entry + target_range * 1.618
            else:
                direction = "SELL"
                entry = setup.breakout_price
                stop_loss = pattern.upper_bound + atr * 0.5
                target_range = pattern.range_width
                target_1 = entry - target_range
                target_2 = entry - target_range * 1.618

            risk = abs(entry - stop_loss)
            reward = abs(target_1 - entry)
            risk_reward = reward / risk if risk > 0 else 0

            # Determine strength
            strength = self._determine_strength(confluence, volume_confirmed, risk_reward, pattern)

            signal = BreakoutSignal(
                setup=setup,
                direction=direction,
                strength=strength,
                entry_price=entry,
                stop_loss=stop_loss,
                target_1=target_1,
                target_2=target_2,
                confluence_score=confluence,
                risk_reward=risk_reward,
                metadata={
                    "pattern_type": pattern.pattern_type.value,
                    "pattern_duration": pattern.duration_candles,
                    "volume_surge": setup.volume_surge,
                    "volume_confirmed": volume_confirmed,
                    "retest": setup.retest_opportunity,
                    "tightening": pattern.tightening,
                    "atr": atr,
                },
            )
            signals.append(signal)

        return signals

    def _calculate_confluence(
        self, setup: BreakoutSetup, df: pd.DataFrame, market_data: Optional[Dict]
    ) -> float:
        """Calculate confluence score for the breakout."""
        score = 0.0
        weights_total = 0.0

        # Volume confirmation (weight: 30%)
        weight = 0.30
        weights_total += weight
        if setup.volume_surge >= self.volume_surge_threshold * 1.5:
            score += weight * 1.0
        elif setup.volume_surge >= self.volume_surge_threshold:
            score += weight * 0.7
        elif setup.volume_surge >= self.volume_surge_threshold * 0.7:
            score += weight * 0.4

        # Pattern quality (weight: 25%)
        weight = 0.25
        weights_total += weight
        pattern = setup.pattern
        if pattern.volume_profile == "DECLINING" and pattern.tightening:
            score += weight * 1.0
        elif pattern.volume_profile == "DECLINING":
            score += weight * 0.7
        elif pattern.tightening:
            score += weight * 0.6
        else:
            score += weight * 0.3

        # Duration quality (weight: 15%)
        weight = 0.15
        weights_total += weight
        ideal_duration = 30
        duration_score = 1.0 - abs(pattern.duration_candles - ideal_duration) / ideal_duration
        score += weight * max(0, duration_score)

        # Retest bonus (weight: 15%)
        weight = 0.15
        weights_total += weight
        if setup.retest_opportunity:
            score += weight * 1.0
        else:
            score += weight * 0.5

        # Market alignment (weight: 15%)
        weight = 0.15
        weights_total += weight
        if market_data:
            trend = market_data.get("trend", "neutral")
            if setup.breakout_direction == BreakoutDirection.BULLISH and trend in ("bullish", "strong_bullish"):
                score += weight * 1.0
            elif setup.breakout_direction == BreakoutDirection.BEARISH and trend in ("bearish", "strong_bearish"):
                score += weight * 1.0
            elif trend == "neutral":
                score += weight * 0.5
            else:
                score += weight * 0.2

        return score / weights_total if weights_total > 0 else 0

    def _determine_strength(
        self, confluence: float, volume_confirmed: bool, risk_reward: float, pattern: ConsolidationPattern
    ) -> str:
        """Determine signal strength."""
        strength_score = 0

        if confluence >= 0.8:
            strength_score += 3
        elif confluence >= 0.6:
            strength_score += 2
        elif confluence >= 0.4:
            strength_score += 1

        if volume_confirmed:
            strength_score += 2

        if risk_reward >= 3.0:
            strength_score += 2
        elif risk_reward >= 2.0:
            strength_score += 1

        if pattern.tightening:
            strength_score += 1

        if strength_score >= 7:
            return "VERY_STRONG"
        elif strength_score >= 5:
            return "STRONG"
        elif strength_score >= 3:
            return "MODERATE"
        return "WEAK"

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR."""
        if len(df) < period + 1:
            return (df["high"].iloc[-1] - df["low"].iloc[-1])

        high = df["high"].values
        low = df["low"].values
        close = df["close"].values

        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )

        atr = pd.Series(tr).ewm(span=period, adjust=False).mean().iloc[-1]
        return float(atr)
