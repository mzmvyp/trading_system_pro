"""
Mean Reversion Strategy - Statistical mean reversion with multiple indicators.
Source: smart_trading_system (upgraded from 100-line version)
Features:
- Stochastic, Williams %R, RSI divergence detection
- Bollinger squeeze and expansion detection
- Support/Resistance bounce entries
- Overbought/Oversold zone entries
- Z-score based deviation measurement
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

from src.core.logger import get_logger

logger = get_logger(__name__)


class MeanReversionSetup(Enum):
    BOLLINGER_EXTREME = "BOLLINGER_EXTREME"
    RSI_EXTREME = "RSI_EXTREME"
    STOCHASTIC_EXTREME = "STOCHASTIC_EXTREME"
    ZSCORE_DEVIATION = "ZSCORE_DEVIATION"
    SR_BOUNCE = "SR_BOUNCE"


@dataclass
class MeanReversionConfig:
    # Bollinger
    bb_period: int = 20
    bb_std: float = 2.0
    bb_squeeze_threshold: float = 0.02  # BB width / price

    # RSI
    rsi_period: int = 14
    rsi_overbought: float = 75.0
    rsi_oversold: float = 25.0
    rsi_exit_upper: float = 60.0
    rsi_exit_lower: float = 40.0

    # Stochastic
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    stoch_overbought: float = 80.0
    stoch_oversold: float = 20.0

    # Z-score
    zscore_lookback: int = 50
    zscore_entry_threshold: float = 2.0
    zscore_exit_threshold: float = 0.5

    # Risk
    atr_stop_multiplier: float = 1.5
    min_risk_reward: float = 1.5
    min_confluence: float = 0.4
    max_risk_pct: float = 2.5


@dataclass
class MeanReversionSignal:
    setup_type: MeanReversionSetup
    direction: str
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    confluence_score: float
    risk_reward: float
    zscore: float
    metadata: Dict = field(default_factory=dict)


class MeanReversionStrategy:
    """Mean reversion strategy using statistical extremes."""

    def __init__(self, config: Optional[MeanReversionConfig] = None):
        self.config = config or MeanReversionConfig()

    def calculate_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate all mean reversion indicators."""
        close = df["close"]
        high = df["high"]
        low = df["low"]

        result = {}

        # Bollinger Bands
        bb_mid = close.rolling(self.config.bb_period).mean()
        bb_std = close.rolling(self.config.bb_period).std()
        bb_upper = bb_mid + self.config.bb_std * bb_std
        bb_lower = bb_mid - self.config.bb_std * bb_std
        bb_position = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)
        bb_width = (bb_upper - bb_lower) / (bb_mid + 1e-10)

        result["bb"] = {
            "upper": float(bb_upper.iloc[-1]),
            "middle": float(bb_mid.iloc[-1]),
            "lower": float(bb_lower.iloc[-1]),
            "position": float(bb_position.iloc[-1]),
            "width": float(bb_width.iloc[-1]),
            "squeeze": float(bb_width.iloc[-1]) < self.config.bb_squeeze_threshold,
        }

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/self.config.rsi_period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/self.config.rsi_period, adjust=False).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        result["rsi"] = float(rsi.iloc[-1])
        result["rsi_series"] = rsi

        # Stochastic
        lowest_low = low.rolling(self.config.stoch_k_period).min()
        highest_high = high.rolling(self.config.stoch_k_period).max()
        stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
        stoch_d = stoch_k.rolling(self.config.stoch_d_period).mean()

        result["stochastic"] = {
            "k": float(stoch_k.iloc[-1]),
            "d": float(stoch_d.iloc[-1]),
            "overbought": float(stoch_k.iloc[-1]) > self.config.stoch_overbought,
            "oversold": float(stoch_k.iloc[-1]) < self.config.stoch_oversold,
        }

        # Z-score
        zscore_mean = close.rolling(self.config.zscore_lookback).mean()
        zscore_std = close.rolling(self.config.zscore_lookback).std()
        zscore = (close - zscore_mean) / (zscore_std + 1e-10)
        result["zscore"] = float(zscore.iloc[-1])

        # Williams %R
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
        result["williams_r"] = float(williams_r.iloc[-1])

        # ATR
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = tr.ewm(span=14, adjust=False).mean()
        result["atr"] = float(atr.iloc[-1])

        return result

    def detect_divergence(self, df: pd.DataFrame, rsi_series: pd.Series) -> Optional[Dict]:
        """Detect RSI divergence with price."""
        if len(df) < 20:
            return None

        close = df["close"].values
        rsi_vals = rsi_series.values

        # Look at last 20 candles for divergence
        lookback = min(20, len(close) - 1)

        # Bullish divergence: price makes lower low, RSI makes higher low
        price_lows = []
        rsi_lows = []
        for i in range(len(close) - lookback, len(close) - 2):
            if close[i] < close[i - 1] and close[i] < close[i + 1]:
                price_lows.append((i, close[i]))
                rsi_lows.append((i, rsi_vals[i]))

        if len(price_lows) >= 2:
            if price_lows[-1][1] < price_lows[-2][1] and rsi_lows[-1][1] > rsi_lows[-2][1]:
                return {"type": "BULLISH_DIVERGENCE", "strength": "REGULAR"}

        # Bearish divergence: price makes higher high, RSI makes lower high
        price_highs = []
        rsi_highs = []
        for i in range(len(close) - lookback, len(close) - 2):
            if close[i] > close[i - 1] and close[i] > close[i + 1]:
                price_highs.append((i, close[i]))
                rsi_highs.append((i, rsi_vals[i]))

        if len(price_highs) >= 2:
            if price_highs[-1][1] > price_highs[-2][1] and rsi_highs[-1][1] < rsi_highs[-2][1]:
                return {"type": "BEARISH_DIVERGENCE", "strength": "REGULAR"}

        return None

    def generate_signals(self, df: pd.DataFrame, market_data: Optional[Dict] = None) -> List[MeanReversionSignal]:
        """Generate mean reversion signals."""
        signals = []

        if len(df) < self.config.zscore_lookback + 10:
            return signals

        indicators = self.calculate_indicators(df)
        current_price = float(df["close"].iloc[-1])
        atr = indicators["atr"]

        # Check divergence
        divergence = self.detect_divergence(df, indicators["rsi_series"])

        # Setup 1: Bollinger extreme
        bb = indicators["bb"]
        if bb["position"] > 0.95 or bb["position"] < 0.05:
            direction = "SELL" if bb["position"] > 0.95 else "BUY"
            confluence = self._calc_confluence(indicators, direction, divergence)
            signal = self._create_signal(
                MeanReversionSetup.BOLLINGER_EXTREME, direction, current_price,
                atr, indicators, confluence, bb["middle"]
            )
            if signal:
                signals.append(signal)

        # Setup 2: RSI extreme
        rsi = indicators["rsi"]
        if rsi > self.config.rsi_overbought or rsi < self.config.rsi_oversold:
            direction = "SELL" if rsi > self.config.rsi_overbought else "BUY"
            confluence = self._calc_confluence(indicators, direction, divergence)
            signal = self._create_signal(
                MeanReversionSetup.RSI_EXTREME, direction, current_price,
                atr, indicators, confluence, bb["middle"]
            )
            if signal:
                signals.append(signal)

        # Setup 3: Stochastic extreme
        stoch = indicators["stochastic"]
        if stoch["overbought"] or stoch["oversold"]:
            direction = "SELL" if stoch["overbought"] else "BUY"
            # Stochastic crossover confirmation
            if (stoch["overbought"] and stoch["k"] < stoch["d"]) or \
               (stoch["oversold"] and stoch["k"] > stoch["d"]):
                confluence = self._calc_confluence(indicators, direction, divergence)
                signal = self._create_signal(
                    MeanReversionSetup.STOCHASTIC_EXTREME, direction, current_price,
                    atr, indicators, confluence, bb["middle"]
                )
                if signal:
                    signals.append(signal)

        # Setup 4: Z-score deviation
        zscore = indicators["zscore"]
        if abs(zscore) > self.config.zscore_entry_threshold:
            direction = "SELL" if zscore > 0 else "BUY"
            confluence = self._calc_confluence(indicators, direction, divergence)
            signal = self._create_signal(
                MeanReversionSetup.ZSCORE_DEVIATION, direction, current_price,
                atr, indicators, confluence, bb["middle"]
            )
            if signal:
                signals.append(signal)

        return signals

    def _create_signal(
        self, setup: MeanReversionSetup, direction: str, price: float,
        atr: float, indicators: Dict, confluence: float, mean_price: float
    ) -> Optional[MeanReversionSignal]:
        """Create a mean reversion signal."""
        if confluence < self.config.min_confluence:
            return None

        if direction == "BUY":
            stop_loss = price - atr * self.config.atr_stop_multiplier
            target_1 = mean_price
            target_2 = mean_price + (mean_price - price) * 0.5
        else:
            stop_loss = price + atr * self.config.atr_stop_multiplier
            target_1 = mean_price
            target_2 = mean_price - (price - mean_price) * 0.5

        risk = abs(price - stop_loss)
        reward = abs(target_1 - price)
        rr = reward / risk if risk > 0 else 0

        if rr < self.config.min_risk_reward:
            return None

        # Max risk check
        risk_pct = (risk / price) * 100
        if risk_pct > self.config.max_risk_pct:
            return None

        return MeanReversionSignal(
            setup_type=setup,
            direction=direction,
            entry_price=price,
            stop_loss=stop_loss,
            target_1=target_1,
            target_2=target_2,
            confluence_score=confluence,
            risk_reward=rr,
            zscore=indicators["zscore"],
            metadata={
                "bb_position": indicators["bb"]["position"],
                "rsi": indicators["rsi"],
                "stochastic_k": indicators["stochastic"]["k"],
                "williams_r": indicators["williams_r"],
                "bb_squeeze": indicators["bb"]["squeeze"],
            },
        )

    def _calc_confluence(self, indicators: Dict, direction: str, divergence: Optional[Dict]) -> float:
        """Calculate confluence for mean reversion."""
        score = 0.0
        total = 0.0

        # BB position (25%)
        w = 0.25
        total += w
        bb_pos = indicators["bb"]["position"]
        if direction == "BUY" and bb_pos < 0.1:
            score += w * 1.0
        elif direction == "BUY" and bb_pos < 0.2:
            score += w * 0.6
        elif direction == "SELL" and bb_pos > 0.9:
            score += w * 1.0
        elif direction == "SELL" and bb_pos > 0.8:
            score += w * 0.6

        # RSI (20%)
        w = 0.20
        total += w
        rsi = indicators["rsi"]
        if direction == "BUY" and rsi < 25:
            score += w * 1.0
        elif direction == "BUY" and rsi < 35:
            score += w * 0.5
        elif direction == "SELL" and rsi > 75:
            score += w * 1.0
        elif direction == "SELL" and rsi > 65:
            score += w * 0.5

        # Stochastic (15%)
        w = 0.15
        total += w
        stoch_k = indicators["stochastic"]["k"]
        if direction == "BUY" and stoch_k < 20:
            score += w * 1.0
        elif direction == "SELL" and stoch_k > 80:
            score += w * 1.0
        elif (direction == "BUY" and stoch_k < 40) or (direction == "SELL" and stoch_k > 60):
            score += w * 0.4

        # Z-score (20%)
        w = 0.20
        total += w
        zscore = abs(indicators["zscore"])
        if zscore > 2.5:
            score += w * 1.0
        elif zscore > 2.0:
            score += w * 0.7
        elif zscore > 1.5:
            score += w * 0.4

        # Divergence bonus (20%)
        w = 0.20
        total += w
        if divergence:
            if (direction == "BUY" and divergence["type"] == "BULLISH_DIVERGENCE") or \
               (direction == "SELL" and divergence["type"] == "BEARISH_DIVERGENCE"):
                score += w * 1.0
        else:
            score += w * 0.2

        return score / total if total > 0 else 0
