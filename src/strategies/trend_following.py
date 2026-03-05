"""
Trend Following Strategy - Advanced trend detection and following.
Source: smart_trading_system (upgraded from 123-line version)
Features:
- TrendSetup enum (5 types): Pullback, Breakout Follow, MA Cross, Momentum, Trend Continuation
- TrendPhase enum: Early/Mature/Exhaustion
- TrendFollowingConfig (20+ configurable params)
- Confluence score and risk scoring
- Priority system (1-5)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

from src.core.logger import get_logger

logger = get_logger(__name__)


class TrendSetup(Enum):
    PULLBACK = "PULLBACK"
    BREAKOUT_FOLLOW = "BREAKOUT_FOLLOW"
    MA_CROSS = "MA_CROSS"
    MOMENTUM = "MOMENTUM"
    TREND_CONTINUATION = "TREND_CONTINUATION"


class TrendPhase(Enum):
    EARLY = "EARLY"
    MATURE = "MATURE"
    EXHAUSTION = "EXHAUSTION"


@dataclass
class TrendFollowingConfig:
    # EMA settings
    fast_ema: int = 9
    medium_ema: int = 21
    slow_ema: int = 50
    trend_ema: int = 200

    # ADX settings
    adx_period: int = 14
    adx_trend_threshold: float = 25.0
    adx_strong_threshold: float = 40.0

    # RSI settings
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    rsi_pullback_bull: float = 45.0
    rsi_pullback_bear: float = 55.0

    # Pullback settings
    pullback_min_pct: float = 0.5
    pullback_max_pct: float = 3.0
    pullback_ema_touch_tolerance: float = 0.2

    # Risk settings
    atr_stop_multiplier: float = 2.0
    atr_target_multiplier_1: float = 3.0
    atr_target_multiplier_2: float = 5.0
    min_risk_reward: float = 2.0

    # Signal settings
    min_confluence: float = 0.5
    min_priority: int = 3


@dataclass
class TrendSignal:
    setup_type: TrendSetup
    direction: str
    phase: TrendPhase
    priority: int  # 1-5, 5 is highest
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    confluence_score: float
    risk_reward: float
    risk_score: float  # 0-1, lower is better
    metadata: Dict = field(default_factory=dict)


class TrendFollowingStrategy:
    """Advanced trend following with multiple setup types and phases."""

    def __init__(self, config: Optional[TrendFollowingConfig] = None):
        self.config = config or TrendFollowingConfig()

    def analyze_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive trend analysis."""
        if len(df) < self.config.trend_ema + 10:
            return {"direction": "NEUTRAL", "phase": "UNKNOWN", "strength": 0}

        close = df["close"]

        # EMAs
        ema_fast = close.ewm(span=self.config.fast_ema, adjust=False).mean()
        ema_medium = close.ewm(span=self.config.medium_ema, adjust=False).mean()
        ema_slow = close.ewm(span=self.config.slow_ema, adjust=False).mean()
        ema_trend = close.ewm(span=self.config.trend_ema, adjust=False).mean()

        current_price = float(close.iloc[-1])
        ef = float(ema_fast.iloc[-1])
        em = float(ema_medium.iloc[-1])
        es = float(ema_slow.iloc[-1])
        et = float(ema_trend.iloc[-1])

        # ADX
        adx = self._calculate_adx(df, self.config.adx_period)
        current_adx = float(adx.iloc[-1]) if len(adx) > 0 else 20

        # RSI
        rsi = self._calculate_rsi(df, self.config.rsi_period)
        current_rsi = float(rsi.iloc[-1]) if len(rsi) > 0 else 50

        # Determine direction
        bull_count = sum([
            current_price > ef, current_price > em, current_price > es,
            current_price > et, ef > em, em > es, es > et,
        ])

        if bull_count >= 5 and current_adx >= self.config.adx_trend_threshold:
            direction = "BULLISH"
        elif bull_count <= 2 and current_adx >= self.config.adx_trend_threshold:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

        # Phase detection
        phase = self._detect_phase(direction, current_adx, current_rsi, ema_fast, ema_medium)

        # Strength (0-1)
        strength = min(current_adx / 60.0, 1.0)

        return {
            "direction": direction,
            "phase": phase.value,
            "strength": strength,
            "adx": current_adx,
            "rsi": current_rsi,
            "emas": {"fast": ef, "medium": em, "slow": es, "trend": et},
            "price": current_price,
            "ema_alignment": bull_count >= 5 if direction == "BULLISH" else bull_count <= 2,
        }

    def _detect_phase(
        self, direction: str, adx: float, rsi: float,
        ema_fast: pd.Series, ema_medium: pd.Series
    ) -> TrendPhase:
        """Detect current trend phase."""
        if direction == "NEUTRAL":
            return TrendPhase.EARLY

        # EMA separation rate
        if len(ema_fast) < 5:
            return TrendPhase.EARLY

        separation_now = abs(float(ema_fast.iloc[-1]) - float(ema_medium.iloc[-1]))
        separation_prev = abs(float(ema_fast.iloc[-5]) - float(ema_medium.iloc[-5]))

        if direction == "BULLISH":
            if adx < 30 and separation_now > separation_prev:
                return TrendPhase.EARLY
            elif adx >= 40 or rsi > 65:
                return TrendPhase.EXHAUSTION
            else:
                return TrendPhase.MATURE
        else:
            if adx < 30 and separation_now > separation_prev:
                return TrendPhase.EARLY
            elif adx >= 40 or rsi < 35:
                return TrendPhase.EXHAUSTION
            else:
                return TrendPhase.MATURE

    def detect_pullback(self, df: pd.DataFrame, trend: Dict) -> Optional[Dict]:
        """Detect pullback entry in a trend."""
        if trend["direction"] == "NEUTRAL":
            return None

        close = df["close"]
        current_price = float(close.iloc[-1])
        emas = trend["emas"]

        if trend["direction"] == "BULLISH":
            # Price pulled back to medium EMA zone
            distance_to_medium = (current_price - emas["medium"]) / emas["medium"] * 100
            distance_to_slow = (current_price - emas["slow"]) / emas["slow"] * 100

            if self.config.pullback_min_pct <= abs(distance_to_medium) <= self.config.pullback_max_pct:
                if trend["rsi"] <= self.config.rsi_pullback_bear:
                    return {
                        "type": "EMA_PULLBACK",
                        "ema_target": "medium",
                        "distance_pct": distance_to_medium,
                        "rsi": trend["rsi"],
                    }

            if abs(distance_to_medium) < self.config.pullback_ema_touch_tolerance:
                return {
                    "type": "EMA_TOUCH",
                    "ema_target": "medium",
                    "distance_pct": distance_to_medium,
                    "rsi": trend["rsi"],
                }

        elif trend["direction"] == "BEARISH":
            distance_to_medium = (emas["medium"] - current_price) / emas["medium"] * 100
            if self.config.pullback_min_pct <= abs(distance_to_medium) <= self.config.pullback_max_pct:
                if trend["rsi"] >= self.config.rsi_pullback_bear:
                    return {
                        "type": "EMA_PULLBACK",
                        "ema_target": "medium",
                        "distance_pct": distance_to_medium,
                        "rsi": trend["rsi"],
                    }

        return None

    def detect_ma_cross(self, df: pd.DataFrame) -> Optional[Dict]:
        """Detect moving average crossover signals."""
        if len(df) < self.config.slow_ema + 5:
            return None

        close = df["close"]
        ema_fast = close.ewm(span=self.config.fast_ema, adjust=False).mean()
        ema_medium = close.ewm(span=self.config.medium_ema, adjust=False).mean()

        # Check for recent crossover (last 3 candles)
        for i in range(-3, 0):
            prev_fast = float(ema_fast.iloc[i - 1])
            prev_medium = float(ema_medium.iloc[i - 1])
            curr_fast = float(ema_fast.iloc[i])
            curr_medium = float(ema_medium.iloc[i])

            if prev_fast <= prev_medium and curr_fast > curr_medium:
                return {"type": "BULLISH_CROSS", "candles_ago": abs(i)}
            elif prev_fast >= prev_medium and curr_fast < curr_medium:
                return {"type": "BEARISH_CROSS", "candles_ago": abs(i)}

        return None

    def detect_momentum(self, df: pd.DataFrame, trend: Dict) -> Optional[Dict]:
        """Detect momentum surge in trend direction."""
        if len(df) < 5 or trend["direction"] == "NEUTRAL":
            return None

        close = df["close"]
        volume = df["volume"]

        # Price momentum (last 3 candles)
        returns = close.pct_change().iloc[-3:]
        avg_return = float(returns.mean())
        vol_ratio = float(volume.iloc[-1]) / float(volume.iloc[-10:-1].mean()) if len(df) > 10 else 1.0

        if trend["direction"] == "BULLISH" and avg_return > 0.005 and vol_ratio > 1.5:
            return {"type": "BULL_MOMENTUM", "avg_return": avg_return, "volume_ratio": vol_ratio}
        elif trend["direction"] == "BEARISH" and avg_return < -0.005 and vol_ratio > 1.5:
            return {"type": "BEAR_MOMENTUM", "avg_return": avg_return, "volume_ratio": vol_ratio}

        return None

    def generate_signals(self, df: pd.DataFrame, market_data: Optional[Dict] = None) -> List[TrendSignal]:
        """Generate trend following signals."""
        signals = []

        trend = self.analyze_trend(df)
        if trend["direction"] == "NEUTRAL":
            return signals

        atr = self._calculate_atr_value(df)
        current_price = trend["price"]
        phase = TrendPhase(trend["phase"])

        # Check each setup type
        setups = []

        # Pullback
        pullback = self.detect_pullback(df, trend)
        if pullback:
            setups.append((TrendSetup.PULLBACK, pullback, 5 if phase == TrendPhase.EARLY else 4))

        # MA Cross
        ma_cross = self.detect_ma_cross(df)
        if ma_cross:
            is_aligned = (ma_cross["type"] == "BULLISH_CROSS" and trend["direction"] == "BULLISH") or \
                         (ma_cross["type"] == "BEARISH_CROSS" and trend["direction"] == "BEARISH")
            if is_aligned:
                setups.append((TrendSetup.MA_CROSS, ma_cross, 4 if phase == TrendPhase.EARLY else 3))

        # Momentum
        momentum = self.detect_momentum(df, trend)
        if momentum:
            setups.append((TrendSetup.MOMENTUM, momentum, 3))

        for setup_type, setup_data, priority in setups:
            if priority < self.config.min_priority:
                continue

            direction = "BUY" if trend["direction"] == "BULLISH" else "SELL"

            # Calculate levels
            if direction == "BUY":
                stop_loss = current_price - atr * self.config.atr_stop_multiplier
                target_1 = current_price + atr * self.config.atr_target_multiplier_1
                target_2 = current_price + atr * self.config.atr_target_multiplier_2
            else:
                stop_loss = current_price + atr * self.config.atr_stop_multiplier
                target_1 = current_price - atr * self.config.atr_target_multiplier_1
                target_2 = current_price - atr * self.config.atr_target_multiplier_2

            risk = abs(current_price - stop_loss)
            reward = abs(target_1 - current_price)
            rr = reward / risk if risk > 0 else 0

            if rr < self.config.min_risk_reward:
                continue

            # Confluence
            confluence = self._calculate_confluence(trend, setup_type, phase, setup_data)
            if confluence < self.config.min_confluence:
                continue

            # Risk score
            risk_score = self._calculate_risk_score(trend, phase)

            signal = TrendSignal(
                setup_type=setup_type,
                direction=direction,
                phase=phase,
                priority=priority,
                entry_price=current_price,
                stop_loss=stop_loss,
                target_1=target_1,
                target_2=target_2,
                confluence_score=confluence,
                risk_reward=rr,
                risk_score=risk_score,
                metadata={
                    "trend": trend,
                    "setup_data": setup_data,
                    "atr": atr,
                },
            )
            signals.append(signal)

        # Sort by priority
        signals.sort(key=lambda s: s.priority, reverse=True)
        return signals

    def _calculate_confluence(
        self, trend: Dict, setup: TrendSetup, phase: TrendPhase, setup_data: Dict
    ) -> float:
        """Calculate confluence score."""
        score = 0.0

        # Trend strength (30%)
        score += 0.30 * trend["strength"]

        # EMA alignment (20%)
        score += 0.20 * (1.0 if trend["ema_alignment"] else 0.3)

        # Phase bonus (20%)
        phase_scores = {TrendPhase.EARLY: 1.0, TrendPhase.MATURE: 0.7, TrendPhase.EXHAUSTION: 0.3}
        score += 0.20 * phase_scores.get(phase, 0.5)

        # Setup quality (30%)
        setup_scores = {
            TrendSetup.PULLBACK: 0.9,
            TrendSetup.MA_CROSS: 0.7,
            TrendSetup.MOMENTUM: 0.6,
            TrendSetup.BREAKOUT_FOLLOW: 0.8,
            TrendSetup.TREND_CONTINUATION: 0.7,
        }
        score += 0.30 * setup_scores.get(setup, 0.5)

        return score

    def _calculate_risk_score(self, trend: Dict, phase: TrendPhase) -> float:
        """Calculate risk score (0-1, lower is better)."""
        risk = 0.0

        # Exhaustion phase = higher risk
        if phase == TrendPhase.EXHAUSTION:
            risk += 0.4
        elif phase == TrendPhase.MATURE:
            risk += 0.2

        # RSI extremes
        rsi = trend["rsi"]
        if rsi > 75 or rsi < 25:
            risk += 0.3
        elif rsi > 65 or rsi < 35:
            risk += 0.1

        # Weak ADX
        if trend["adx"] < 20:
            risk += 0.2

        return min(risk, 1.0)

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ADX."""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
        minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)

        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)

        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10))
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        return adx.fillna(20)

    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    def _calculate_atr_value(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate current ATR value."""
        if len(df) < period + 1:
            return float(df["high"].iloc[-1] - df["low"].iloc[-1])

        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs()
        ], axis=1).max(axis=1)

        return float(tr.ewm(span=period, adjust=False).mean().iloc[-1])
