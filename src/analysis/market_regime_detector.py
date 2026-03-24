"""
Market Regime Detector for Futures - Multi-dimensional regime analysis.
Source: agente_trade_futuros
Features:
- Multi-timeframe trend analysis (15m/1h/4h)
- Volatility regime detection (SQUEEZE/HIGH/LOW/NORMAL)
- Momentum scoring
- Funding rate & Open Interest analysis
- Combined regime: BULL/BEAR/SIDEWAYS + volatility suffix
- Special regimes: SQUEEZE_BREAKOUT_PENDING, LONG_SQUEEZE, SHORT_SQUEEZE
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.core.logger import get_logger

logger = get_logger(__name__)


class MarketRegimeDetectorFutures:
    """
    Detector de Regime de Mercado otimizado para Binance Futures.
    Analisa tendência, volatilidade, momentum e dados de futuros.
    """

    def __init__(self):
        self.ema_periods = [9, 21, 50, 200]

    async def detect_regime(self, symbol: str = "BTCUSDT", client=None) -> Dict:
        """Detect current market regime using multiple data sources."""
        try:
            owns_client = client is None
            if owns_client:
                from src.exchange.client import BinanceClient
                client = BinanceClient()
                await client.__aenter__()

            try:
                df_15m = await client.get_klines(symbol, interval="15m", limit=200)
                df_1h = await client.get_klines(symbol, interval="1h", limit=200)
                df_4h = await client.get_klines(symbol, interval="4h", limit=100)

                if df_1h is None or len(df_1h) < 50:
                    return self._get_default_regime()

                trend = self._analyze_trend(df_1h, df_4h)
                volatility = self._analyze_volatility(df_15m if df_15m is not None else df_1h)
                momentum = self._analyze_momentum(
                    df_15m if df_15m is not None else df_1h, df_1h
                )

                # Funding rate
                funding = {"bias": "NEUTRAL", "rate": 0.0, "confidence": 0.0}
                try:
                    rate = await client.get_funding_rate(symbol)
                    if rate is not None:
                        if rate > 0.0002:
                            funding = {"bias": "OVERBOUGHT", "rate": rate, "confidence": min(abs(rate) * 1000, 1.0)}
                        elif rate < -0.0002:
                            funding = {"bias": "OVERSOLD", "rate": rate, "confidence": min(abs(rate) * 1000, 1.0)}
                except Exception:
                    pass

                # Open interest
                oi = {"trend": "NEUTRAL", "change": 0.0, "confidence": 0.0}
                try:
                    oi_data = await client.get_open_interest(symbol)
                    if oi_data:
                        oi = oi_data
                except Exception:
                    pass

                return self._combine_analyses(trend, volatility, momentum, funding, oi)
            finally:
                if owns_client:
                    await client.__aexit__(None, None, None)

        except Exception as e:
            logger.error(f"Regime detection error: {e}")
            return self._get_default_regime()

    def _analyze_trend(self, df_1h: pd.DataFrame, df_4h: Optional[pd.DataFrame]) -> Dict:
        """Analyze trend direction using EMAs and ADX."""
        try:
            close = df_1h["close"].astype(float)
            for period in self.ema_periods:
                if len(close) >= period:
                    df_1h[f"ema_{period}"] = close.ewm(span=period, adjust=False).mean()

            current_price = float(close.iloc[-1])
            emas = {}
            for p in self.ema_periods:
                col = f"ema_{p}"
                if col in df_1h.columns:
                    emas[p] = float(df_1h[col].iloc[-1])

            # Score EMA alignment
            score = 0
            if emas.get(9) and current_price > emas[9]:
                score += 0.2
            if emas.get(21) and current_price > emas[21]:
                score += 0.2
            if emas.get(50) and current_price > emas[50]:
                score += 0.3
            if emas.get(200) and current_price > emas[200]:
                score += 0.3

            # Perfect alignment bonus
            if all(p in emas for p in self.ema_periods):
                if emas[9] > emas[21] > emas[50] > emas[200]:
                    score = 1.0
                elif emas[9] < emas[21] < emas[50] < emas[200]:
                    score = 0.0

            # ADX
            adx = self._calculate_adx(df_1h)
            current_adx = float(adx.iloc[-1]) if len(adx) > 0 else 25

            if current_adx < 25:
                trend = "SIDEWAYS"
                confidence = 1 - (current_adx / 25)
            else:
                if score > 0.6:
                    trend = "BULL"
                elif score < 0.4:
                    trend = "BEAR"
                else:
                    trend = "SIDEWAYS"
                confidence = min(abs(score - 0.5) * 2, current_adx / 50)

            return {"trend": trend, "confidence": confidence, "adx": current_adx, "score": score}
        except Exception:
            return {"trend": "SIDEWAYS", "confidence": 0.5, "adx": 25, "score": 0.5}

    def _analyze_volatility(self, df: pd.DataFrame) -> Dict:
        """Analyze volatility regime."""
        try:
            close = df["close"].astype(float)
            high = df["high"].astype(float)
            low = df["low"].astype(float)

            # ATR
            tr = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs()
            ], axis=1).max(axis=1)
            atr = tr.ewm(span=14, adjust=False).mean()
            natr = (atr / close).iloc[-1]

            # Bollinger Band Width
            sma = close.rolling(20).mean()
            std = close.rolling(20).std()
            upper = sma + 2 * std
            lower = sma - 2 * std
            bbw = ((upper - lower) / close).iloc[-1]

            # Percentile ranking
            natr_series = atr / close
            if len(natr_series) >= 100:
                vol_high = natr_series.rolling(100).quantile(0.80).iloc[-1]
                vol_low = natr_series.rolling(100).quantile(0.20).iloc[-1]
            else:
                vol_high = natr * 1.5
                vol_low = natr * 0.5

            if bbw < 0.025:
                vol_regime = "SQUEEZE"
            elif natr > vol_high:
                vol_regime = "HIGH"
            elif natr < vol_low:
                vol_regime = "LOW"
            else:
                vol_regime = "NORMAL"

            return {"volatility": vol_regime, "value": float(natr), "bbw": float(bbw)}
        except Exception:
            return {"volatility": "NORMAL", "value": 0.03, "bbw": 0.05}

    def _analyze_momentum(self, df_15m: pd.DataFrame, df_1h: pd.DataFrame) -> Dict:
        """Analyze momentum across timeframes."""
        try:
            rsi_15m = self._calculate_rsi(df_15m)
            rsi_1h = self._calculate_rsi(df_1h)

            r15 = float(rsi_15m.iloc[-1]) if len(rsi_15m) > 0 else 50
            r1h = float(rsi_1h.iloc[-1]) if len(rsi_1h) > 0 else 50

            # MACD
            close = df_15m["close"].astype(float)
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd_hist = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()

            score = 0
            if r15 > 55 and r1h > 52:
                score += 0.5
            if r15 < 45 and r1h < 48:
                score -= 0.5

            if len(macd_hist) >= 2:
                if float(macd_hist.iloc[-1]) > 0 and float(macd_hist.iloc[-1]) > float(macd_hist.iloc[-2]):
                    score += 0.5
                if float(macd_hist.iloc[-1]) < 0 and float(macd_hist.iloc[-1]) < float(macd_hist.iloc[-2]):
                    score -= 0.5

            if score > 0.5:
                momentum = "BULLISH"
            elif score < -0.5:
                momentum = "BEARISH"
            else:
                momentum = "NEUTRAL"

            return {"momentum": momentum, "score": score, "rsi_15m": r15, "rsi_1h": r1h}
        except Exception:
            return {"momentum": "NEUTRAL", "score": 0, "rsi_15m": 50, "rsi_1h": 50}

    def _combine_analyses(self, trend, volatility, momentum, funding, oi) -> Dict:
        """Combine all analyses into final regime."""
        bull_score, bear_score = 0.0, 0.0

        if trend["trend"] == "BULL":
            bull_score += 1.0 * trend["confidence"]
        if trend["trend"] == "BEAR":
            bear_score += 1.0 * trend["confidence"]

        if momentum["momentum"] == "BULLISH":
            bull_score += 0.8 * abs(momentum["score"])
        if momentum["momentum"] == "BEARISH":
            bear_score += 0.8 * abs(momentum["score"])

        if oi.get("trend") == "INCREASING" and trend["trend"] == "BULL":
            bull_score += 0.5 * oi.get("confidence", 0)
        if oi.get("trend") == "INCREASING" and trend["trend"] == "BEAR":
            bear_score += 0.5 * oi.get("confidence", 0)

        if funding["bias"] == "OVERBOUGHT":
            bear_score += 0.3 * funding["confidence"]
        if funding["bias"] == "OVERSOLD":
            bull_score += 0.3 * funding["confidence"]

        # Base regime
        if bull_score > bear_score and trend["adx"] > 25:
            base_regime = "BULL"
        elif bear_score > bull_score and trend["adx"] > 25:
            base_regime = "BEAR"
        else:
            base_regime = "SIDEWAYS"

        # Final regime with volatility
        final_regime = f"{base_regime}_{volatility['volatility']}_VOL"

        # Special regimes
        if volatility["volatility"] == "SQUEEZE":
            final_regime = "SQUEEZE_BREAKOUT_PENDING"
        if base_regime == "BULL" and funding["bias"] == "OVERBOUGHT" and momentum["momentum"] != "BULLISH":
            final_regime = "LONG_SQUEEZE"
        if base_regime == "BEAR" and funding["bias"] == "OVERSOLD" and momentum["momentum"] != "BEARISH":
            final_regime = "SHORT_SQUEEZE"

        confidence = (bull_score + bear_score) / 2 if (bull_score + bear_score) > 0 else 0.5

        return {
            "regime": final_regime,
            "base_regime": base_regime,
            "confidence": min(confidence, 1.0),
            "details": {
                "trend": trend,
                "volatility": volatility,
                "momentum": momentum,
                "funding": funding,
                "open_interest": oi,
            },
        }

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)

        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
        minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)

        tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()

        plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / (atr + 1e-10))

        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10))
        return dx.ewm(alpha=1/period, adjust=False).mean().fillna(25)

    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        close = df["close"].astype(float)
        delta = close.diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
        rs = gain / (loss + 1e-10)
        return (100 - (100 / (1 + rs))).fillna(50)

    def _get_default_regime(self) -> Dict:
        return {
            "regime": "SIDEWAYS_NORMAL_VOL",
            "base_regime": "SIDEWAYS",
            "confidence": 0.5,
            "details": {},
        }
