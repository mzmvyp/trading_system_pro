"""
Feature Engineering - ML feature preparation pipeline.
Source: sinais
Features:
- Technical indicators as features (RSI, MACD, BB, ATR, EMAs)
- Lag features (price, volume, returns)
- Rolling statistics (mean, std, min, max, position)
- Volume features (VWAP, volume ratio, VWMA)
- Price pattern features (body, shadows, consecutive)
- Temporal features (hour, day_of_week with sin/cos encoding)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.core.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """Prepare ML features from OHLCV data."""

    def prepare_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Add all feature columns to a dataframe with OHLCV data."""
        try:
            if len(df) < 60:
                return None

            df = df.copy()

            # Normalize column names
            col_map = {
                "close_price": "close", "open_price": "open",
                "high_price": "high", "low_price": "low",
            }
            df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

            df = self._add_technical_indicators(df)
            df = self._add_lag_features(df)
            df = self._add_rolling_features(df)
            df = self._add_volume_features(df)
            df = self._add_price_patterns(df)

            if "timestamp" in df.columns:
                df = self._add_temporal_features(df)

            df = df.dropna()
            if len(df) < 50:
                return None

            return df

        except Exception as e:
            logger.error(f"Feature preparation error: {e}")
            return None

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicator features."""
        close = df["close"]

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df["rsi"] = 100 - (100 / (1 + rs))

        # MACD
        ema_fast = close.ewm(span=12, adjust=False).mean()
        ema_slow = close.ewm(span=26, adjust=False).mean()
        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]

        # Bollinger Bands
        df["bb_middle"] = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        df["bb_upper"] = df["bb_middle"] + 2 * bb_std
        df["bb_lower"] = df["bb_middle"] - 2 * bb_std
        df["bb_position"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)

        # ATR
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - close.shift(1)).abs(),
            (df["low"] - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        df["atr"] = tr.ewm(span=14, adjust=False).mean()

        # EMAs
        for period in [9, 21, 50]:
            df[f"ema_{period}"] = close.ewm(span=period, adjust=False).mean()

        return df

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged price and volume features."""
        for lag in [1, 2, 3, 5, 10]:
            df[f"close_lag_{lag}"] = df["close"].shift(lag)
            df[f"volume_lag_{lag}"] = df["volume"].shift(lag)

        for lag in [1, 2, 3, 5]:
            df[f"return_lag_{lag}"] = df["close"].pct_change(lag).shift(1)

        return df

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling window statistics."""
        for window in [5, 10, 20, 50]:
            df[f"close_mean_{window}"] = df["close"].rolling(window).mean()
            df[f"close_std_{window}"] = df["close"].rolling(window).std()
            df[f"close_min_{window}"] = df["close"].rolling(window).min()
            df[f"close_max_{window}"] = df["close"].rolling(window).max()
            df[f"close_position_{window}"] = (
                (df["close"] - df[f"close_min_{window}"]) /
                (df[f"close_max_{window}"] - df[f"close_min_{window}"] + 1e-10)
            )

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features including VWAP."""
        df["volume_mean_20"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / (df["volume_mean_20"] + 1e-10)

        # VWAP
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        df["vwap"] = (
            (typical_price * df["volume"]).rolling(20).sum() /
            (df["volume"].rolling(20).sum() + 1e-10)
        )
        df["vwap_distance"] = (df["close"] - df["vwap"]) / (df["vwap"] + 1e-10)

        # VWMA
        for window in [10, 20]:
            df[f"vwma_{window}"] = (
                (df["close"] * df["volume"]).rolling(window).sum() /
                (df["volume"].rolling(window).sum() + 1e-10)
            )

        return df

    def _add_price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add candlestick pattern features."""
        df["body_size"] = abs(df["close"] - df["open"])
        df["body_size_pct"] = df["body_size"] / (df["open"] + 1e-10)
        df["upper_shadow"] = df["high"] - df[["open", "close"]].max(axis=1)
        df["lower_shadow"] = df[["open", "close"]].min(axis=1) - df["low"]
        df["is_bullish"] = (df["close"] > df["open"]).astype(int)
        df["candle_range"] = df["high"] - df["low"]
        df["candle_range_pct"] = df["candle_range"] / (df["open"] + 1e-10)

        # Consecutive candles
        df["consecutive_up"] = 0
        df["consecutive_down"] = 0
        for i in range(1, len(df)):
            if df.iloc[i]["is_bullish"] == 1:
                df.iloc[i, df.columns.get_loc("consecutive_up")] = df.iloc[i - 1]["consecutive_up"] + 1
            else:
                df.iloc[i, df.columns.get_loc("consecutive_down")] = df.iloc[i - 1]["consecutive_down"] + 1

        return df

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features with cyclical encoding."""
        ts = pd.to_datetime(df["timestamp"])
        df["hour"] = ts.dt.hour
        df["day_of_week"] = ts.dt.dayofweek
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        return df

    def create_target(self, df: pd.DataFrame, prediction_horizon: int = 12, threshold: float = 0.01) -> pd.DataFrame:
        """Create binary target: 1 if price goes up by threshold within horizon."""
        df = df.copy()
        df["future_return"] = df["close"].shift(-prediction_horizon) / df["close"] - 1
        df["target"] = (df["future_return"] > threshold).astype(int)
        df = df[~df["target"].isna()].copy()
        df.drop("future_return", axis=1, inplace=True)
        return df

    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature column names (exclude raw OHLCV and metadata)."""
        exclude = {
            "timestamp", "symbol", "timeframe", "target",
            "open", "high", "low", "close", "volume",
            "open_price", "high_price", "low_price", "close_price",
        }
        return [col for col in df.columns if col not in exclude]

    def prepare_training_data(
        self, dataframes: Dict[str, pd.DataFrame],
        prediction_horizon: int = 12,
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], List[str]]:
        """Prepare training data from multiple symbol dataframes."""
        all_data = []

        for symbol, df in dataframes.items():
            featured = self.prepare_features(df)
            if featured is not None and len(featured) > 0:
                featured = self.create_target(featured, prediction_horizon)
                if len(featured) > 0:
                    featured["symbol_encoded"] = hash(symbol) % 1000
                    all_data.append(featured)

        if not all_data:
            return None, None, []

        combined = pd.concat(all_data, ignore_index=True)
        feature_names = self.get_feature_names(combined)
        X = combined[feature_names]
        y = combined["target"]

        return X, y, feature_names
