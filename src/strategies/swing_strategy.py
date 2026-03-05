"""
Swing Strategy - Multi-timeframe swing trading with structure breaks.
Source: smart_trading_system
Features:
- Multi-timeframe analysis (4H/1D/1H)
- SwingSetup types: STRUCTURE_BREAK, PULLBACK_ENTRY, SR_RETEST
- Confluence with weights: structure(35%), trend(25%), leading(25%), S/R(15%)
- Signal expiration: 72 hours
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd

from src.core.logger import get_logger

logger = get_logger(__name__)


class SwingSetup(Enum):
    STRUCTURE_BREAK = "STRUCTURE_BREAK"
    PULLBACK_ENTRY = "PULLBACK_ENTRY"
    SR_RETEST = "SR_RETEST"


class TrendDirection(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


@dataclass
class SwingLevel:
    price: float
    level_type: str  # "SUPPORT", "RESISTANCE"
    strength: int  # Number of touches
    timeframe: str
    last_touch: Optional[datetime] = None


@dataclass
class StructurePoint:
    price: float
    point_type: str  # "HH", "HL", "LH", "LL"
    index: int
    significance: float = 0.0


@dataclass
class SwingSignal:
    setup_type: SwingSetup
    direction: str  # "BUY" or "SELL"
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    confluence_score: float
    risk_reward: float
    expiration: datetime
    timeframes_aligned: List[str]
    metadata: Dict = field(default_factory=dict)


class SwingStrategy:
    """
    Swing trading strategy using multi-timeframe structure analysis.
    Focuses on higher-timeframe trends with lower-timeframe entries.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.primary_tf = self.config.get("primary_tf", "4h")
        self.higher_tf = self.config.get("higher_tf", "1d")
        self.entry_tf = self.config.get("entry_tf", "1h")
        self.signal_expiration_hours = self.config.get("signal_expiration_hours", 72)
        self.min_confluence = self.config.get("min_confluence", 0.5)
        self.min_risk_reward = self.config.get("min_risk_reward", 2.0)

        # Confluence weights
        self.weights = {
            "structure": self.config.get("weight_structure", 0.35),
            "trend": self.config.get("weight_trend", 0.25),
            "leading": self.config.get("weight_leading", 0.25),
            "sr_levels": self.config.get("weight_sr", 0.15),
        }

    def analyze_structure(self, df: pd.DataFrame) -> List[StructurePoint]:
        """Detect market structure points (HH, HL, LH, LL)."""
        points = []
        if len(df) < 10:
            return points

        highs = df["high"].values
        lows = df["low"].values

        # Find swing highs and lows using 5-bar pivot
        pivot_len = 5
        for i in range(pivot_len, len(df) - pivot_len):
            # Swing high
            if all(highs[i] >= highs[i - j] for j in range(1, pivot_len + 1)) and \
               all(highs[i] >= highs[i + j] for j in range(1, pivot_len + 1)):
                points.append(StructurePoint(
                    price=float(highs[i]),
                    point_type="HIGH",
                    index=i,
                ))

            # Swing low
            if all(lows[i] <= lows[i - j] for j in range(1, pivot_len + 1)) and \
               all(lows[i] <= lows[i + j] for j in range(1, pivot_len + 1)):
                points.append(StructurePoint(
                    price=float(lows[i]),
                    point_type="LOW",
                    index=i,
                ))

        # Classify as HH/HL/LH/LL
        self._classify_structure_points(points)

        return points

    def _classify_structure_points(self, points: List[StructurePoint]):
        """Classify structure points as HH, HL, LH, LL."""
        highs = [p for p in points if p.point_type == "HIGH"]
        lows = [p for p in points if p.point_type == "LOW"]

        for i in range(1, len(highs)):
            if highs[i].price > highs[i - 1].price:
                highs[i].point_type = "HH"
                highs[i].significance = abs(highs[i].price - highs[i - 1].price) / highs[i - 1].price
            else:
                highs[i].point_type = "LH"
                highs[i].significance = abs(highs[i - 1].price - highs[i].price) / highs[i - 1].price

        for i in range(1, len(lows)):
            if lows[i].price > lows[i - 1].price:
                lows[i].point_type = "HL"
                lows[i].significance = abs(lows[i].price - lows[i - 1].price) / lows[i - 1].price
            else:
                lows[i].point_type = "LL"
                lows[i].significance = abs(lows[i - 1].price - lows[i].price) / lows[i - 1].price

    def determine_trend(self, structure_points: List[StructurePoint]) -> TrendDirection:
        """Determine trend direction from structure points."""
        if len(structure_points) < 4:
            return TrendDirection.NEUTRAL

        recent = structure_points[-6:] if len(structure_points) >= 6 else structure_points
        hh_count = sum(1 for p in recent if p.point_type == "HH")
        hl_count = sum(1 for p in recent if p.point_type == "HL")
        lh_count = sum(1 for p in recent if p.point_type == "LH")
        ll_count = sum(1 for p in recent if p.point_type == "LL")

        bull_score = hh_count + hl_count
        bear_score = lh_count + ll_count

        if bull_score > bear_score and bull_score >= 2:
            return TrendDirection.BULLISH
        elif bear_score > bull_score and bear_score >= 2:
            return TrendDirection.BEARISH
        return TrendDirection.NEUTRAL

    def find_sr_levels(self, df: pd.DataFrame, num_levels: int = 5) -> List[SwingLevel]:
        """Find support and resistance levels."""
        levels = []
        if len(df) < 20:
            return levels

        price_range = df["high"].max() - df["low"].min()
        tolerance = price_range * 0.005  # 0.5% tolerance

        # Collect all swing points
        highs_values = []
        lows_values = []
        pivot_len = 3

        for i in range(pivot_len, len(df) - pivot_len):
            if all(df["high"].iloc[i] >= df["high"].iloc[i - j] for j in range(1, pivot_len + 1)) and \
               all(df["high"].iloc[i] >= df["high"].iloc[i + j] for j in range(1, pivot_len + 1)):
                highs_values.append(float(df["high"].iloc[i]))

            if all(df["low"].iloc[i] <= df["low"].iloc[i - j] for j in range(1, pivot_len + 1)) and \
               all(df["low"].iloc[i] <= df["low"].iloc[i + j] for j in range(1, pivot_len + 1)):
                lows_values.append(float(df["low"].iloc[i]))

        # Cluster nearby levels
        all_levels = [(p, "RESISTANCE") for p in highs_values] + [(p, "SUPPORT") for p in lows_values]
        all_levels.sort(key=lambda x: x[0])

        clustered = []
        used = set()
        for i, (price, level_type) in enumerate(all_levels):
            if i in used:
                continue
            cluster_prices = [price]
            used.add(i)
            for j in range(i + 1, len(all_levels)):
                if j in used:
                    continue
                if abs(all_levels[j][0] - price) <= tolerance:
                    cluster_prices.append(all_levels[j][0])
                    used.add(j)

            avg_price = np.mean(cluster_prices)
            current_price = float(df["close"].iloc[-1])
            lt = "SUPPORT" if avg_price < current_price else "RESISTANCE"

            clustered.append(SwingLevel(
                price=avg_price,
                level_type=lt,
                strength=len(cluster_prices),
                timeframe=self.primary_tf,
            ))

        # Sort by strength and return top N
        clustered.sort(key=lambda x: x.strength, reverse=True)
        return clustered[:num_levels]

    def detect_pullback(
        self, df: pd.DataFrame, trend: TrendDirection, sr_levels: List[SwingLevel]
    ) -> Optional[Dict]:
        """Detect pullback entry opportunities."""
        if trend == TrendDirection.NEUTRAL or len(df) < 5:
            return None

        current_price = float(df["close"].iloc[-1])
        recent_low = float(df["low"].iloc[-3:].min())
        recent_high = float(df["high"].iloc[-3:].max())

        if trend == TrendDirection.BULLISH:
            # Look for pullback to support
            for level in sr_levels:
                if level.level_type == "SUPPORT":
                    distance_pct = abs(current_price - level.price) / level.price * 100
                    if distance_pct < 1.0 and recent_low <= level.price * 1.005:
                        return {
                            "type": "PULLBACK_TO_SUPPORT",
                            "level": level.price,
                            "distance_pct": distance_pct,
                            "strength": level.strength,
                        }

        elif trend == TrendDirection.BEARISH:
            for level in sr_levels:
                if level.level_type == "RESISTANCE":
                    distance_pct = abs(current_price - level.price) / level.price * 100
                    if distance_pct < 1.0 and recent_high >= level.price * 0.995:
                        return {
                            "type": "PULLBACK_TO_RESISTANCE",
                            "level": level.price,
                            "distance_pct": distance_pct,
                            "strength": level.strength,
                        }

        return None

    def generate_signals(
        self,
        df_primary: pd.DataFrame,
        df_higher: Optional[pd.DataFrame] = None,
        df_entry: Optional[pd.DataFrame] = None,
        market_data: Optional[Dict] = None,
    ) -> List[SwingSignal]:
        """Generate swing trading signals."""
        signals = []

        # Structure analysis on primary timeframe
        structure = self.analyze_structure(df_primary)
        trend = self.determine_trend(structure)
        sr_levels = self.find_sr_levels(df_primary)

        # Higher timeframe trend confirmation
        higher_trend = TrendDirection.NEUTRAL
        if df_higher is not None:
            higher_structure = self.analyze_structure(df_higher)
            higher_trend = self.determine_trend(higher_structure)

        # Timeframe alignment
        aligned_tfs = []
        if trend != TrendDirection.NEUTRAL:
            aligned_tfs.append(self.primary_tf)
        if higher_trend == trend and trend != TrendDirection.NEUTRAL:
            aligned_tfs.append(self.higher_tf)

        entry_df = df_entry if df_entry is not None else df_primary
        current_price = float(entry_df["close"].iloc[-1])
        atr = self._calculate_atr(entry_df)

        # Setup 1: Structure break
        if len(structure) >= 2:
            last_point = structure[-1]
            if last_point.point_type == "HH" and trend == TrendDirection.BULLISH:
                signal = self._create_signal(
                    SwingSetup.STRUCTURE_BREAK, "BUY", current_price, atr,
                    structure, sr_levels, trend, higher_trend, aligned_tfs
                )
                if signal:
                    signals.append(signal)

            elif last_point.point_type == "LL" and trend == TrendDirection.BEARISH:
                signal = self._create_signal(
                    SwingSetup.STRUCTURE_BREAK, "SELL", current_price, atr,
                    structure, sr_levels, trend, higher_trend, aligned_tfs
                )
                if signal:
                    signals.append(signal)

        # Setup 2: Pullback entry
        pullback = self.detect_pullback(entry_df, trend, sr_levels)
        if pullback:
            direction = "BUY" if trend == TrendDirection.BULLISH else "SELL"
            signal = self._create_signal(
                SwingSetup.PULLBACK_ENTRY, direction, current_price, atr,
                structure, sr_levels, trend, higher_trend, aligned_tfs
            )
            if signal:
                signal.metadata["pullback"] = pullback
                signals.append(signal)

        return signals

    def _create_signal(
        self, setup_type: SwingSetup, direction: str, current_price: float,
        atr: float, structure: List[StructurePoint], sr_levels: List[SwingLevel],
        trend: TrendDirection, higher_trend: TrendDirection, aligned_tfs: List[str]
    ) -> Optional[SwingSignal]:
        """Create a swing signal with confluence scoring."""
        # Calculate confluence
        confluence = self._calculate_confluence(
            setup_type, direction, structure, sr_levels, trend, higher_trend
        )

        if confluence < self.min_confluence:
            return None

        # Entry, SL, TP
        if direction == "BUY":
            entry = current_price
            stop_loss = current_price - atr * 2.0
            target_1 = current_price + atr * 4.0
            target_2 = current_price + atr * 6.0

            # Adjust to nearest S/R
            for level in sr_levels:
                if level.level_type == "SUPPORT" and level.price < current_price:
                    stop_loss = min(stop_loss, level.price - atr * 0.5)
                if level.level_type == "RESISTANCE" and level.price > current_price:
                    target_1 = level.price
                    break
        else:
            entry = current_price
            stop_loss = current_price + atr * 2.0
            target_1 = current_price - atr * 4.0
            target_2 = current_price - atr * 6.0

            for level in sr_levels:
                if level.level_type == "RESISTANCE" and level.price > current_price:
                    stop_loss = max(stop_loss, level.price + atr * 0.5)
                if level.level_type == "SUPPORT" and level.price < current_price:
                    target_1 = level.price
                    break

        risk = abs(entry - stop_loss)
        reward = abs(target_1 - entry)
        risk_reward = reward / risk if risk > 0 else 0

        if risk_reward < self.min_risk_reward:
            return None

        return SwingSignal(
            setup_type=setup_type,
            direction=direction,
            entry_price=entry,
            stop_loss=stop_loss,
            target_1=target_1,
            target_2=target_2,
            confluence_score=confluence,
            risk_reward=risk_reward,
            expiration=datetime.now() + timedelta(hours=self.signal_expiration_hours),
            timeframes_aligned=aligned_tfs,
            metadata={
                "setup": setup_type.value,
                "trend": trend.value,
                "higher_trend": higher_trend.value,
                "atr": atr,
            },
        )

    def _calculate_confluence(
        self, setup_type: SwingSetup, direction: str,
        structure: List[StructurePoint], sr_levels: List[SwingLevel],
        trend: TrendDirection, higher_trend: TrendDirection
    ) -> float:
        """Calculate confluence score with weighted components."""
        score = 0.0

        # Structure alignment (35%)
        w = self.weights["structure"]
        if setup_type == SwingSetup.STRUCTURE_BREAK:
            score += w * 0.9
        elif setup_type == SwingSetup.PULLBACK_ENTRY:
            score += w * 0.8
        else:
            score += w * 0.6

        # Trend alignment (25%)
        w = self.weights["trend"]
        if trend != TrendDirection.NEUTRAL:
            if higher_trend == trend:
                score += w * 1.0
            else:
                score += w * 0.5
        else:
            score += w * 0.2

        # Leading indicators placeholder (25%)
        w = self.weights["leading"]
        score += w * 0.5  # Default mid-score without external data

        # S/R level support (15%)
        w = self.weights["sr_levels"]
        strong_levels = [l for l in sr_levels if l.strength >= 3]
        if len(strong_levels) >= 2:
            score += w * 0.8
        elif len(strong_levels) >= 1:
            score += w * 0.5
        else:
            score += w * 0.2

        return score

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR."""
        if len(df) < period + 1:
            return float(df["high"].iloc[-1] - df["low"].iloc[-1])

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

        return float(pd.Series(tr).ewm(span=period, adjust=False).mean().iloc[-1])
