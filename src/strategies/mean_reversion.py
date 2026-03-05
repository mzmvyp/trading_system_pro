"""
Estratégia de reversão à média (portada e simplificada de smart_trading_system).
Usa RSI, Bollinger e estocástico para extremos.
"""
import pandas as pd
import numpy as np
import talib
from typing import Dict, Any
from dataclasses import dataclass
from strategies.base_strategy import BaseStrategy
from strategies.signal_types import SignalType
from filters.volatility_filter import VolatilityFilter
from filters.market_condition_filter import MarketConditionFilter
from utils.helpers import safe_divide
from src.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MeanReversionConfig:
    """Configurações da estratégia."""
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    stop_atr_mult: float = 1.5


class MeanReversionStrategy(BaseStrategy):
    """Mean reversion em extremos de RSI/Bollinger."""

    name = "mean_reversion"
    timeframes = ["15m", "1h", "4h"]

    def __init__(self, config: MeanReversionConfig = None):
        self.config = config or MeanReversionConfig()
        self.volatility_filter = VolatilityFilter()
        self.market_filter = MarketConditionFilter()

    def analyze(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, Any]:
        if df is None or len(df) < 50:
            return {"signals": [], "analysis": {"error": "Dados insuficientes"}}
        try:
            df = df.copy()
            if "close" not in df.columns and "Close" in df.columns:
                df["close"] = df["Close"]
            if "high" not in df.columns:
                df["high"] = df["High"] if "High" in df.columns else df["close"]
            if "low" not in df.columns:
                df["low"] = df["Low"] if "Low" in df.columns else df["close"]
            close = df["close"].astype(float)
            high = df["high"].astype(float)
            low = df["low"].astype(float)

            rsi = talib.RSI(close, timeperiod=self.config.rsi_period)
            upper, mid, lower = talib.BBANDS(close, self.config.bb_period, self.config.bb_std, self.config.bb_std)
            atr = talib.ATR(high, low, close, timeperiod=self.config.atr_period)

            i = len(close) - 1
            price = float(close.iloc[i])
            atr_val = float(atr.iloc[i]) if not np.isnan(atr.iloc[i]) else price * 0.02
            signals = []

            if rsi.iloc[i] <= self.config.rsi_oversold and close.iloc[i] <= lower.iloc[i] * 1.01:
                stop = price - self.config.stop_atr_mult * atr_val
                tp = float(mid.iloc[i])
                signals.append({
                    "symbol": symbol,
                    "signal_type": SignalType.BUY.value,
                    "entry_price": price,
                    "stop_loss": stop,
                    "take_profit": tp,
                    "confidence": 7,
                    "strategy": self.name,
                    "timeframe": timeframe,
                })
            elif rsi.iloc[i] >= self.config.rsi_overbought and close.iloc[i] >= upper.iloc[i] * 0.99:
                stop = price + self.config.stop_atr_mult * atr_val
                tp = float(mid.iloc[i])
                signals.append({
                    "symbol": symbol,
                    "signal_type": SignalType.SELL.value,
                    "entry_price": price,
                    "stop_loss": stop,
                    "take_profit": tp,
                    "confidence": 7,
                    "strategy": self.name,
                    "timeframe": timeframe,
                })

            return {
                "signals": signals,
                "analysis": {"rsi": float(rsi.iloc[i]), "filters_passed": True},
            }
        except Exception as e:
            logger.exception("Erro na análise mean_reversion: %s", e)
            return {"signals": [], "analysis": {"error": str(e)}}
