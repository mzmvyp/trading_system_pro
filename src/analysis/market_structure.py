"""
Market Structure Analysis - Wyckoff phases and structural analysis.
Source: smart_trading_system
Features:
- MarketPhase: Accumulation, Markup, Distribution, Markdown
- StructureType: HH, HL, LH, LL, EH, EL
- Trend regression with R² confidence
- Breakout analysis with volume confirmation
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.core.logger import get_logger

logger = get_logger(__name__)


class MarketPhase(Enum):
    ACCUMULATION = "ACCUMULATION"
    MARKUP = "MARKUP"
    DISTRIBUTION = "DISTRIBUTION"
    MARKDOWN = "MARKDOWN"
    UNKNOWN = "UNKNOWN"


class StructureType(Enum):
    HH = "HH"  # Higher High
    HL = "HL"  # Higher Low
    LH = "LH"  # Lower High
    LL = "LL"  # Lower Low
    EH = "EH"  # Equal High
    EL = "EL"  # Equal Low


@dataclass
class StructurePoint:
    price: float
    index: int
    structure_type: StructureType
    significance: float = 0.0  # 0-1
    volume_at_point: float = 0.0
    timestamp: Optional[str] = None


@dataclass
class TrendAnalysis:
    direction: str  # "UP", "DOWN", "SIDEWAYS"
    slope: float
    r_squared: float
    confidence: float
    duration_candles: int


@dataclass
class BreakoutAnalysis:
    level: float
    direction: str  # "BULLISH", "BEARISH"
    volume_confirmed: bool
    false_breakout_risk: float  # 0-1
    distance_pct: float


@dataclass
class MarketStructureResult:
    phase: MarketPhase
    trend: TrendAnalysis
    structure_points: List[StructurePoint]
    breakouts: List[BreakoutAnalysis]
    support_levels: List[float]
    resistance_levels: List[float]
    metadata: Dict = field(default_factory=dict)


class MarketStructureAnalyzer:
    """
    Comprehensive market structure analysis including Wyckoff phases,
    structure points, trend regression, and breakout detection.
    """

    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        self.pivot_len = config.get("pivot_len", 5)
        self.equal_threshold_pct = config.get("equal_threshold_pct", 0.1)
        self.volume_surge_ratio = config.get("volume_surge_ratio", 1.5)
        self.trend_lookback = config.get("trend_lookback", 50)

    def analyze(self, df: pd.DataFrame) -> MarketStructureResult:
        """Full market structure analysis."""
        if len(df) < 30:
            return MarketStructureResult(
                phase=MarketPhase.UNKNOWN,
                trend=TrendAnalysis("SIDEWAYS", 0, 0, 0, 0),
                structure_points=[], breakouts=[],
                support_levels=[], resistance_levels=[],
            )

        # Find structure points
        structure_points = self._find_structure_points(df)

        # Classify structure
        self._classify_structure(structure_points)

        # Trend analysis via regression
        trend = self._analyze_trend(df)

        # Detect phase
        phase = self._detect_phase(structure_points, trend, df)

        # Find S/R levels
        support, resistance = self._find_sr_levels(structure_points, float(df["close"].iloc[-1]))

        # Detect breakouts
        breakouts = self._detect_breakouts(df, support, resistance)

        return MarketStructureResult(
            phase=phase,
            trend=trend,
            structure_points=structure_points,
            breakouts=breakouts,
            support_levels=support,
            resistance_levels=resistance,
            metadata={
                "hh_count": sum(1 for p in structure_points if p.structure_type == StructureType.HH),
                "hl_count": sum(1 for p in structure_points if p.structure_type == StructureType.HL),
                "lh_count": sum(1 for p in structure_points if p.structure_type == StructureType.LH),
                "ll_count": sum(1 for p in structure_points if p.structure_type == StructureType.LL),
            },
        )

    def _find_structure_points(self, df: pd.DataFrame) -> List[StructurePoint]:
        """Find swing highs and lows."""
        points = []
        highs = df["high"].values
        lows = df["low"].values
        volumes = df["volume"].values if "volume" in df.columns else np.ones(len(df))

        for i in range(self.pivot_len, len(df) - self.pivot_len):
            # Swing high
            is_high = all(highs[i] >= highs[i - j] for j in range(1, self.pivot_len + 1)) and \
                      all(highs[i] >= highs[i + j] for j in range(1, self.pivot_len + 1))
            if is_high:
                points.append(StructurePoint(
                    price=float(highs[i]),
                    index=i,
                    structure_type=StructureType.HH,  # Will be reclassified
                    volume_at_point=float(volumes[i]),
                ))

            # Swing low
            is_low = all(lows[i] <= lows[i - j] for j in range(1, self.pivot_len + 1)) and \
                     all(lows[i] <= lows[i + j] for j in range(1, self.pivot_len + 1))
            if is_low:
                points.append(StructurePoint(
                    price=float(lows[i]),
                    index=i,
                    structure_type=StructureType.LL,  # Will be reclassified
                    volume_at_point=float(volumes[i]),
                ))

        points.sort(key=lambda p: p.index)
        return points

    def _classify_structure(self, points: List[StructurePoint]):
        """Classify each point as HH/HL/LH/LL/EH/EL."""
        swing_highs = [p for p in points if p.price == max(p.price for p2 in points if abs(p2.index - p.index) < 2)]
        swing_lows = [p for p in points if p.price == min(p.price for p2 in points if abs(p2.index - p.index) < 2)]

        # Separate highs and lows by alternating through points
        highs = []
        lows = []
        for p in points:
            # Rough classification: if near local max -> high, else low
            if len(highs) == 0 and len(lows) == 0:
                # First point
                if len(points) > 1:
                    if p.price > points[1].price:
                        highs.append(p)
                    else:
                        lows.append(p)
                continue

            if highs and not lows:
                lows.append(p)
            elif lows and not highs:
                highs.append(p)
            elif p.price > lows[-1].price:
                highs.append(p)
            else:
                lows.append(p)

        # Classify highs
        for i in range(1, len(highs)):
            prev = highs[i - 1].price
            curr = highs[i].price
            diff_pct = abs(curr - prev) / prev * 100

            if diff_pct < self.equal_threshold_pct:
                highs[i].structure_type = StructureType.EH
            elif curr > prev:
                highs[i].structure_type = StructureType.HH
                highs[i].significance = (curr - prev) / prev
            else:
                highs[i].structure_type = StructureType.LH
                highs[i].significance = (prev - curr) / prev

        # Classify lows
        for i in range(1, len(lows)):
            prev = lows[i - 1].price
            curr = lows[i].price
            diff_pct = abs(curr - prev) / prev * 100

            if diff_pct < self.equal_threshold_pct:
                lows[i].structure_type = StructureType.EL
            elif curr > prev:
                lows[i].structure_type = StructureType.HL
                lows[i].significance = (curr - prev) / prev
            else:
                lows[i].structure_type = StructureType.LL
                lows[i].significance = (prev - curr) / prev

    def _analyze_trend(self, df: pd.DataFrame) -> TrendAnalysis:
        """Analyze trend using linear regression."""
        lookback = min(self.trend_lookback, len(df))
        close = df["close"].values[-lookback:]
        x = np.arange(lookback)

        # Linear regression
        coeffs = np.polyfit(x, close, 1)
        slope = coeffs[0]

        # R-squared
        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum((close - y_pred) ** 2)
        ss_tot = np.sum((close - np.mean(close)) ** 2)
        r_squared = 1 - (ss_res / (ss_tot + 1e-10))

        # Normalize slope
        slope_pct = (slope / close[0]) * 100  # % per candle

        if slope_pct > 0.05 and r_squared > 0.5:
            direction = "UP"
        elif slope_pct < -0.05 and r_squared > 0.5:
            direction = "DOWN"
        else:
            direction = "SIDEWAYS"

        confidence = r_squared * min(abs(slope_pct) * 10, 1.0)

        return TrendAnalysis(
            direction=direction,
            slope=slope_pct,
            r_squared=r_squared,
            confidence=min(confidence, 1.0),
            duration_candles=lookback,
        )

    def _detect_phase(
        self, points: List[StructurePoint], trend: TrendAnalysis, df: pd.DataFrame
    ) -> MarketPhase:
        """Detect Wyckoff market phase."""
        if len(points) < 4:
            return MarketPhase.UNKNOWN

        recent = points[-6:] if len(points) >= 6 else points
        hh = sum(1 for p in recent if p.structure_type == StructureType.HH)
        hl = sum(1 for p in recent if p.structure_type == StructureType.HL)
        lh = sum(1 for p in recent if p.structure_type == StructureType.LH)
        ll = sum(1 for p in recent if p.structure_type == StructureType.LL)
        eq = sum(1 for p in recent if p.structure_type in (StructureType.EH, StructureType.EL))

        # Markup: HH + HL dominant
        if hh + hl > lh + ll and trend.direction == "UP":
            return MarketPhase.MARKUP

        # Markdown: LH + LL dominant
        if lh + ll > hh + hl and trend.direction == "DOWN":
            return MarketPhase.MARKDOWN

        # Accumulation: after downtrend, range forming
        if trend.r_squared < 0.3 and eq >= 2:
            # Check if preceded by down move
            close_early = float(df["close"].iloc[-self.trend_lookback]) if len(df) > self.trend_lookback else float(df["close"].iloc[0])
            close_mid = float(df["close"].iloc[-self.trend_lookback // 2]) if len(df) > self.trend_lookback // 2 else float(df["close"].iloc[0])
            close_now = float(df["close"].iloc[-1])

            if close_early > close_mid and abs(close_now - close_mid) / close_mid < 0.03:
                return MarketPhase.ACCUMULATION
            elif close_early < close_mid and abs(close_now - close_mid) / close_mid < 0.03:
                return MarketPhase.DISTRIBUTION

        return MarketPhase.UNKNOWN

    def _find_sr_levels(
        self, points: List[StructurePoint], current_price: float
    ) -> Tuple[List[float], List[float]]:
        """Extract support and resistance levels from structure points."""
        support = []
        resistance = []

        for p in points:
            if p.price < current_price:
                support.append(p.price)
            else:
                resistance.append(p.price)

        # Deduplicate nearby levels
        support = self._cluster_levels(sorted(support, reverse=True))
        resistance = self._cluster_levels(sorted(resistance))

        return support[:5], resistance[:5]

    def _cluster_levels(self, levels: List[float], threshold_pct: float = 0.3) -> List[float]:
        """Cluster nearby price levels."""
        if not levels:
            return []

        clustered = [levels[0]]
        for level in levels[1:]:
            if abs(level - clustered[-1]) / clustered[-1] * 100 > threshold_pct:
                clustered.append(level)

        return clustered

    def _detect_breakouts(
        self, df: pd.DataFrame, support: List[float], resistance: List[float]
    ) -> List[BreakoutAnalysis]:
        """Detect breakouts from S/R levels."""
        breakouts = []
        current_price = float(df["close"].iloc[-1])
        current_volume = float(df["volume"].iloc[-1]) if "volume" in df.columns else 0
        avg_volume = float(df["volume"].iloc[-20:].mean()) if "volume" in df.columns else 1

        volume_confirmed = current_volume > avg_volume * self.volume_surge_ratio

        # Check resistance breakouts
        for level in resistance:
            if current_price > level:
                dist_pct = (current_price - level) / level * 100
                if dist_pct < 2.0:  # Recent breakout
                    false_risk = 0.3 if volume_confirmed else 0.6
                    if dist_pct < 0.3:
                        false_risk += 0.2

                    breakouts.append(BreakoutAnalysis(
                        level=level,
                        direction="BULLISH",
                        volume_confirmed=volume_confirmed,
                        false_breakout_risk=min(false_risk, 1.0),
                        distance_pct=dist_pct,
                    ))

        # Check support breakdowns
        for level in support:
            if current_price < level:
                dist_pct = (level - current_price) / level * 100
                if dist_pct < 2.0:
                    false_risk = 0.3 if volume_confirmed else 0.6
                    if dist_pct < 0.3:
                        false_risk += 0.2

                    breakouts.append(BreakoutAnalysis(
                        level=level,
                        direction="BEARISH",
                        volume_confirmed=volume_confirmed,
                        false_breakout_risk=min(false_risk, 1.0),
                        distance_pct=dist_pct,
                    ))

        return breakouts
