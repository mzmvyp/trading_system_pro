"""
Estratégia de seguimento de tendência (portada e simplificada de smart_trading_system).
Usa EMA, MACD, ADX e RSI para identificar tendência e pullbacks.
"""
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd
import talib

from filters.market_condition_filter import MarketConditionFilter
from filters.volatility_filter import VolatilityFilter
from src.core.logger import get_logger
from strategies.base_strategy import BaseStrategy
from strategies.signal_types import SignalType

logger = get_logger(__name__)


@dataclass
class TrendFollowingConfig:
    """Configurações da estratégia."""
    fast_ma: int = 10
    slow_ma: int = 50
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    adx_period: int = 14
    adx_strong: float = 25.0
    rsi_period: int = 14
    rsi_pullback_bull: float = 45.0
    rsi_pullback_bear: float = 55.0
    atr_period: int = 14
    stop_atr_mult: float = 2.0


class TrendFollowingStrategy(BaseStrategy):
    """Trend following com MAs, MACD, ADX e RSI."""

    name = "trend_following"
    timeframes = ["4h", "1d"]

    def __init__(self, config: TrendFollowingConfig = None):
        self.config = config or TrendFollowingConfig()
        self.volatility_filter = VolatilityFilter()
        self.market_filter = MarketConditionFilter()

    def analyze(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, Any]:
        if df is None or len(df) < 100:
            return {"signals": [], "analysis": {"error": "Dados insuficientes"}}
        try:
            df = df.copy()
            if "close" not in df.columns and "Close" in df.columns:
                df["close"] = df["Close"]
            if "high" not in df.columns and "High" in df.columns:
                df["high"] = df["High"]
            if "low" not in df.columns and "Low" in df.columns:
                df["low"] = df["Low"]
            if "volume" not in df.columns and "Volume" in df.columns:
                df["volume"] = df["Volume"]
            close = df["close"].astype(float)
            high = df["high"].astype(float)
            low = df["low"].astype(float)

            ema_fast = talib.EMA(close, timeperiod=self.config.fast_ma)
            ema_slow = talib.EMA(close, timeperiod=self.config.slow_ma)
            macd, macd_signal, macd_hist = talib.MACD(
                close, self.config.macd_fast, self.config.macd_slow, self.config.macd_signal
            )
            adx = talib.ADX(high, low, close, timeperiod=self.config.adx_period)
            rsi = talib.RSI(close, timeperiod=self.config.rsi_period)
            atr = talib.ATR(high, low, close, timeperiod=self.config.atr_period)

            i = len(close) - 1
            if np.isnan(adx[i]) or np.isnan(rsi[i]) or adx[i] < self.config.adx_strong:
                return {"signals": [], "analysis": {"reason": "ADX fraco", "adx": float(adx[i]) if not np.isnan(adx[i]) else 0}}

            signals = []
            price = float(close.iloc[i])
            atr_val = float(atr.iloc[i]) if not np.isnan(atr.iloc[i]) else price * 0.02

            if ema_fast.iloc[i] > ema_slow.iloc[i] and rsi.iloc[i] >= self.config.rsi_pullback_bull and rsi.iloc[i] < 70:
                stop = price - self.config.stop_atr_mult * atr_val
                tp1 = price + 2 * (price - stop)
                signals.append({
                    "symbol": symbol,
                    "signal_type": SignalType.BUY.value,
                    "entry_price": price,
                    "stop_loss": stop,
                    "take_profit": tp1,
                    "confidence": min(9, 5 + (float(adx.iloc[i]) - self.config.adx_strong) / 20),
                    "strategy": self.name,
                    "timeframe": timeframe,
                })
            elif ema_fast.iloc[i] < ema_slow.iloc[i] and rsi.iloc[i] <= self.config.rsi_pullback_bear and rsi.iloc[i] > 30:
                stop = price + self.config.stop_atr_mult * atr_val
                tp1 = price - 2 * (stop - price)
                signals.append({
                    "symbol": symbol,
                    "signal_type": SignalType.SELL.value,
                    "entry_price": price,
                    "stop_loss": stop,
                    "take_profit": tp1,
                    "confidence": min(9, 5 + (float(adx.iloc[i]) - self.config.adx_strong) / 20),
                    "strategy": self.name,
                    "timeframe": timeframe,
                })

            vol_result = self.volatility_filter.apply(df, symbol, timeframe, {}, {})
            mkt_result = self.market_filter.apply(df, symbol, timeframe, {}, {})

            return {
                "signals": signals,
                "analysis": {
                    "filters_passed": vol_result.get("passed", True) and mkt_result.get("passed", True),
                    "adx": float(adx.iloc[i]),
                    "rsi": float(rsi.iloc[i]),
                    "trend_analysis": {"direction": "up" if ema_fast.iloc[i] > ema_slow.iloc[i] else "down"},
                },
            }
        except Exception as e:
            logger.exception("Erro na análise trend_following: %s", e)
            return {"signals": [], "analysis": {"error": str(e)}}
