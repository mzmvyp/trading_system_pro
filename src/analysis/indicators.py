"""
Indicadores técnicos centralizados (portado de agente_trade_futuros, usando talib).
Use este módulo em todo o projeto para consistência.
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    talib = None

from src.core.constants import (
    RSI_PERIOD,
    MACD_FAST,
    MACD_SLOW,
    MACD_SIGNAL,
    BOLLINGER_PERIOD,
    BOLLINGER_STD_DEV,
    ATR_PERIOD,
    SMA_SHORT,
    SMA_LONG,
)
from src.core.logger import get_logger

logger = get_logger(__name__)


def _ensure_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Garante colunas em minúsculo (close, high, low, open, volume)."""
    df = df.copy()
    for c in ["close", "high", "low", "open", "volume"]:
        if c not in df.columns and c.capitalize() in df.columns:
            df[c] = df[c.capitalize()]
    return df


class TechnicalIndicators:
    """Indicadores técnicos via talib."""

    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.Series:
        if not TALIB_AVAILABLE:
            return pd.Series(np.nan, index=df.index)
        c = df["close"].astype(float)
        return pd.Series(talib.RSI(c, timeperiod=period), index=df.index)

    @staticmethod
    def calculate_macd(
        df: pd.DataFrame,
        fast: int = MACD_FAST,
        slow: int = MACD_SLOW,
        signal: int = MACD_SIGNAL,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        if not TALIB_AVAILABLE:
            n = len(df)
            return pd.Series(np.nan, index=df.index), pd.Series(np.nan, index=df.index), pd.Series(np.nan, index=df.index)
        c = df["close"].astype(float)
        macd, sig, hist = talib.MACD(c, fastperiod=fast, slowperiod=slow, signalperiod=signal)
        return pd.Series(macd, index=df.index), pd.Series(sig, index=df.index), pd.Series(hist, index=df.index)

    @staticmethod
    def calculate_bollinger_bands(
        df: pd.DataFrame,
        period: int = BOLLINGER_PERIOD,
        std: float = BOLLINGER_STD_DEV,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        if not TALIB_AVAILABLE:
            n = len(df)
            return pd.Series(np.nan, index=df.index), pd.Series(np.nan, index=df.index), pd.Series(np.nan, index=df.index)
        c = df["close"].astype(float)
        u, m, l = talib.BBANDS(c, timeperiod=period, nbdevup=std, nbdevdn=std)
        return pd.Series(u, index=df.index), pd.Series(m, index=df.index), pd.Series(l, index=df.index)

    @staticmethod
    def calculate_sma(df: pd.DataFrame, period: int) -> pd.Series:
        if not TALIB_AVAILABLE:
            return pd.Series(np.nan, index=df.index)
        c = df["close"].astype(float)
        return pd.Series(talib.SMA(c, timeperiod=period), index=df.index)

    @staticmethod
    def calculate_ema(df: pd.DataFrame, period: int) -> pd.Series:
        if not TALIB_AVAILABLE:
            return pd.Series(np.nan, index=df.index)
        c = df["close"].astype(float)
        return pd.Series(talib.EMA(c, timeperiod=period), index=df.index)

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
        if not TALIB_AVAILABLE:
            return pd.Series(np.nan, index=df.index)
        h, l, c = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
        return pd.Series(talib.ATR(h, l, c, timeperiod=period), index=df.index)

    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona colunas de indicadores ao DataFrame."""
        df = _ensure_ohlc(df.copy())
        if not TALIB_AVAILABLE:
            logger.warning("talib não disponível; indicadores não calculados")
            return df
        df["rsi"] = TechnicalIndicators.calculate_rsi(df)
        macd, sig, hist = TechnicalIndicators.calculate_macd(df)
        df["macd"], df["macd_signal"], df["macd_histogram"] = macd, sig, hist
        u, m, l = TechnicalIndicators.calculate_bollinger_bands(df)
        df["bb_upper"], df["bb_middle"], df["bb_lower"] = u, m, l
        df["sma_20"] = TechnicalIndicators.calculate_sma(df, SMA_SHORT)
        df["sma_50"] = TechnicalIndicators.calculate_sma(df, SMA_LONG)
        df["ema_12"] = TechnicalIndicators.calculate_ema(df, 12)
        df["ema_26"] = TechnicalIndicators.calculate_ema(df, 26)
        df["atr"] = TechnicalIndicators.calculate_atr(df)
        if "volume" in df.columns:
            df["volume_sma"] = df["volume"].rolling(20).mean()
        return df

    @staticmethod
    def get_latest_indicators(df: pd.DataFrame) -> Dict[str, float]:
        """Valores mais recentes dos indicadores."""
        if df.empty:
            return {}
        latest = df.iloc[-1]
        return {
            "close": latest.get("close", np.nan),
            "volume": latest.get("volume", np.nan),
            "rsi": latest.get("rsi", np.nan),
            "macd": latest.get("macd", np.nan),
            "macd_signal": latest.get("macd_signal", np.nan),
            "macd_histogram": latest.get("macd_histogram", np.nan),
            "bb_upper": latest.get("bb_upper", np.nan),
            "bb_middle": latest.get("bb_middle", np.nan),
            "bb_lower": latest.get("bb_lower", np.nan),
            "sma_20": latest.get("sma_20", np.nan),
            "sma_50": latest.get("sma_50", np.nan),
            "ema_12": latest.get("ema_12", np.nan),
            "ema_26": latest.get("ema_26", np.nan),
            "atr": latest.get("atr", np.nan),
            "volume_sma": latest.get("volume_sma", np.nan),
        }
