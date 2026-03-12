"""
Candlestick Pattern Detector - 5 high-reliability patterns.
Source: sinais
Patterns: Bullish Engulfing (0.85), Bearish Engulfing (0.90),
          Hammer (0.75), Shooting Star (0.75), Doji (0.60)
Validations: max risk 2.5%, min R:R 1.2:1, max target 5%
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from src.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CandlestickPattern:
    name: str
    pattern_type: str  # "BULLISH", "BEARISH", "NEUTRAL"
    entry: float
    stop_loss: float
    target_1: float
    target_2: Optional[float] = None
    reliability: float = 0.0
    strength: str = "MODERATE"  # "WEAK", "MODERATE", "STRONG"
    metadata: Dict = field(default_factory=dict)

    @property
    def risk_pct(self) -> float:
        if self.entry == 0:
            return 0
        return abs(self.entry - self.stop_loss) / self.entry * 100

    @property
    def reward_risk_ratio(self) -> float:
        risk = abs(self.entry - self.stop_loss)
        reward = abs(self.target_1 - self.entry)
        return reward / risk if risk > 0 else 0


class CandlestickPatternDetector:
    """
    Simplified candlestick pattern detector focused on 5 high-reliability patterns.
    Each pattern includes entry, stop, and target levels with risk validation.
    """

    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        self.max_risk_pct = config.get("max_risk_pct", 2.5)
        self.min_reward_risk = config.get("min_reward_risk", 1.2)
        self.max_target_pct = config.get("max_target_pct", 5.0)
        self.atr_period = config.get("atr_period", 14)

    def detect_patterns(self, df: pd.DataFrame) -> List[CandlestickPattern]:
        """Detect all candlestick patterns in the latest candles."""
        patterns = []

        if len(df) < 3:
            return patterns

        atr = self._calculate_atr(df)

        # Check each pattern
        pattern = self._check_bullish_engulfing(df, atr)
        if pattern:
            patterns.append(pattern)

        pattern = self._check_bearish_engulfing(df, atr)
        if pattern:
            patterns.append(pattern)

        pattern = self._check_hammer(df, atr)
        if pattern:
            patterns.append(pattern)

        pattern = self._check_shooting_star(df, atr)
        if pattern:
            patterns.append(pattern)

        pattern = self._check_doji(df, atr)
        if pattern:
            patterns.append(pattern)

        # Validate all patterns
        validated = []
        for p in patterns:
            if self._validate_pattern(p):
                validated.append(p)

        return validated

    def _check_bullish_engulfing(self, df: pd.DataFrame, atr: float) -> Optional[CandlestickPattern]:
        """Detect bullish engulfing pattern."""
        prev = df.iloc[-2]
        curr = df.iloc[-1]

        # Previous candle is bearish
        if prev["close"] >= prev["open"]:
            return None

        # Current candle is bullish and engulfs previous
        if curr["close"] <= curr["open"]:
            return None

        if curr["open"] >= prev["close"] or curr["close"] <= prev["open"]:
            return None

        entry = float(curr["close"])
        stop_loss = float(min(curr["low"], prev["low"])) - atr * 0.3
        target_1 = entry + abs(entry - stop_loss) * 2.0
        target_2 = entry + abs(entry - stop_loss) * 3.0

        # Strength based on engulfing size
        body_ratio = abs(curr["close"] - curr["open"]) / (abs(prev["close"] - prev["open"]) + 1e-10)
        strength = "STRONG" if body_ratio > 1.5 else "MODERATE"

        return CandlestickPattern(
            name="Bullish Engulfing",
            pattern_type="BULLISH",
            entry=entry,
            stop_loss=stop_loss,
            target_1=target_1,
            target_2=target_2,
            reliability=0.85,
            strength=strength,
            metadata={"body_ratio": body_ratio},
        )

    def _check_bearish_engulfing(self, df: pd.DataFrame, atr: float) -> Optional[CandlestickPattern]:
        """Detect bearish engulfing pattern."""
        prev = df.iloc[-2]
        curr = df.iloc[-1]

        # Previous candle is bullish
        if prev["close"] <= prev["open"]:
            return None

        # Current candle is bearish and engulfs previous
        if curr["close"] >= curr["open"]:
            return None

        if curr["open"] <= prev["close"] or curr["close"] >= prev["open"]:
            return None

        entry = float(curr["close"])
        stop_loss = float(max(curr["high"], prev["high"])) + atr * 0.3
        target_1 = entry - abs(stop_loss - entry) * 2.0
        target_2 = entry - abs(stop_loss - entry) * 3.0

        body_ratio = abs(curr["close"] - curr["open"]) / (abs(prev["close"] - prev["open"]) + 1e-10)
        strength = "STRONG" if body_ratio > 1.5 else "MODERATE"

        return CandlestickPattern(
            name="Bearish Engulfing",
            pattern_type="BEARISH",
            entry=entry,
            stop_loss=stop_loss,
            target_1=target_1,
            target_2=target_2,
            reliability=0.90,
            strength=strength,
            metadata={"body_ratio": body_ratio},
        )

    def _check_hammer(self, df: pd.DataFrame, atr: float) -> Optional[CandlestickPattern]:
        """Detect hammer pattern (bullish reversal)."""
        curr = df.iloc[-1]

        body = abs(curr["close"] - curr["open"])
        total_range = curr["high"] - curr["low"]
        lower_shadow = min(curr["open"], curr["close"]) - curr["low"]
        upper_shadow = curr["high"] - max(curr["open"], curr["close"])

        if total_range < atr * 0.3:
            return None

        # Hammer: small body, long lower shadow (2x+ body), small upper shadow
        if lower_shadow < body * 2.0 or upper_shadow > body * 0.5:
            return None

        entry = float(curr["close"])
        stop_loss = float(curr["low"]) - atr * 0.3
        target_1 = entry + abs(entry - stop_loss) * 1.5
        target_2 = entry + abs(entry - stop_loss) * 2.5

        shadow_ratio = lower_shadow / (body + 1e-10)
        strength = "STRONG" if shadow_ratio > 3.0 else "MODERATE"

        return CandlestickPattern(
            name="Hammer",
            pattern_type="BULLISH",
            entry=entry,
            stop_loss=stop_loss,
            target_1=target_1,
            target_2=target_2,
            reliability=0.75,
            strength=strength,
            metadata={"shadow_ratio": shadow_ratio},
        )

    def _check_shooting_star(self, df: pd.DataFrame, atr: float) -> Optional[CandlestickPattern]:
        """Detect shooting star pattern (bearish reversal)."""
        curr = df.iloc[-1]

        body = abs(curr["close"] - curr["open"])
        total_range = curr["high"] - curr["low"]
        lower_shadow = min(curr["open"], curr["close"]) - curr["low"]
        upper_shadow = curr["high"] - max(curr["open"], curr["close"])

        if total_range < atr * 0.3:
            return None

        # Shooting star: small body, long upper shadow (2x+ body), small lower shadow
        if upper_shadow < body * 2.0 or lower_shadow > body * 0.5:
            return None

        entry = float(curr["close"])
        stop_loss = float(curr["high"]) + atr * 0.3
        target_1 = entry - abs(stop_loss - entry) * 1.5
        target_2 = entry - abs(stop_loss - entry) * 2.5

        shadow_ratio = upper_shadow / (body + 1e-10)
        strength = "STRONG" if shadow_ratio > 3.0 else "MODERATE"

        return CandlestickPattern(
            name="Shooting Star",
            pattern_type="BEARISH",
            entry=entry,
            stop_loss=stop_loss,
            target_1=target_1,
            target_2=target_2,
            reliability=0.75,
            strength=strength,
            metadata={"shadow_ratio": shadow_ratio},
        )

    def _check_doji(self, df: pd.DataFrame, atr: float) -> Optional[CandlestickPattern]:
        """Detect doji pattern (indecision/reversal)."""
        curr = df.iloc[-1]

        body = abs(curr["close"] - curr["open"])
        total_range = curr["high"] - curr["low"]

        if total_range < atr * 0.2:
            return None

        # Doji: very small body relative to range
        if body > total_range * 0.1:
            return None

        # Determine direction based on context (previous 3 candles)
        prev_trend = df["close"].iloc[-4:-1]
        if prev_trend.iloc[-1] > prev_trend.iloc[0]:
            # After uptrend -> bearish signal
            pattern_type = "BEARISH"
            entry = float(curr["close"])
            stop_loss = float(curr["high"]) + atr * 0.3
            target_1 = entry - abs(stop_loss - entry) * 1.5
            target_2 = entry - abs(stop_loss - entry) * 2.0
        else:
            # After downtrend -> bullish signal
            pattern_type = "BULLISH"
            entry = float(curr["close"])
            stop_loss = float(curr["low"]) - atr * 0.3
            target_1 = entry + abs(entry - stop_loss) * 1.5
            target_2 = entry + abs(entry - stop_loss) * 2.0

        return CandlestickPattern(
            name="Doji",
            pattern_type=pattern_type,
            entry=entry,
            stop_loss=stop_loss,
            target_1=target_1,
            target_2=target_2,
            reliability=0.60,
            strength="WEAK",
        )

    def _validate_pattern(self, pattern: CandlestickPattern) -> bool:
        """Validate pattern risk parameters."""
        if pattern.risk_pct > self.max_risk_pct:
            return False

        if pattern.reward_risk_ratio < self.min_reward_risk:
            return False

        target_pct = abs(pattern.target_1 - pattern.entry) / pattern.entry * 100
        if target_pct > self.max_target_pct:
            return False

        return True

    def _calculate_atr(self, df: pd.DataFrame) -> float:
        """Calculate ATR."""
        if len(df) < self.atr_period + 1:
            return float(df["high"].iloc[-1] - df["low"].iloc[-1])

        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs()
        ], axis=1).max(axis=1)

        return float(tr.ewm(span=self.atr_period, adjust=False).mean().iloc[-1])
