"""
Technical indicators analysis using TA-Lib
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
import talib

from src.core.logger import get_logger

logger = get_logger(__name__)


def _analyze_market_structure(df: pd.DataFrame) -> Dict[str, Any]:
    """Analisa estrutura de mercado identificando suporte, resistência e estrutura de tendência."""
    try:
        if len(df) < 10:
            return {"structure": "insufficient_data"}

        highs = df['high'].rolling(window=5).max()
        lows = df['low'].rolling(window=5).min()

        recent_highs = highs.tail(10).dropna()
        recent_lows = lows.tail(10).dropna()

        if len(recent_highs) >= 3 and len(recent_lows) >= 3:
            if (recent_highs.iloc[-1] > recent_highs.iloc[-3] and
                recent_lows.iloc[-1] > recent_lows.iloc[-3]):
                structure = "UPTREND"
                strength = "strong" if (recent_highs.iloc[-1] > recent_highs.iloc[-5] and
                                      recent_lows.iloc[-1] > recent_lows.iloc[-5]) else "moderate"
            elif (recent_highs.iloc[-1] < recent_highs.iloc[-3] and
                  recent_lows.iloc[-1] < recent_lows.iloc[-3]):
                structure = "DOWNTREND"
                strength = "strong" if (recent_highs.iloc[-1] < recent_highs.iloc[-5] and
                                      recent_lows.iloc[-1] < recent_lows.iloc[-5]) else "moderate"
            else:
                structure = "RANGE"
                strength = "neutral"
        else:
            structure = "RANGE"
            strength = "neutral"

        support_level = recent_lows.min() if len(recent_lows) > 0 else df['low'].min()
        resistance_level = recent_highs.max() if len(recent_highs) > 0 else df['high'].max()

        return {
            "structure": structure,
            "strength": strength,
            "support_level": float(support_level),
            "resistance_level": float(resistance_level),
            "recent_highs_count": len(recent_highs),
            "recent_lows_count": len(recent_lows)
        }

    except Exception as e:
        return {"structure": "error", "error": str(e)}


async def analyze_technical_indicators(symbol: str = "BTCUSDT") -> Dict[str, Any]:
    """
    Analisa indicadores técnicos REAIS usando TA-Lib.
    """
    try:
        from src.exchange.client import BinanceClient

        async with BinanceClient() as client:
            klines_df = await client.get_klines(symbol, '1h', limit=200)

        if klines_df.empty or len(klines_df) < 50:
            return {
                "error": "Dados insuficientes para análise técnica (mínimo 50 candles)",
                "symbol": symbol
            }

        df = klines_df.reset_index()

        if 'timestamp' not in df.columns:
            df['timestamp'] = df.index if hasattr(df.index, 'values') else range(len(df))

        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df['close'] = df['close'].ffill().bfill().fillna(0)
        df['high'] = df['high'].ffill().bfill().fillna(0)
        df['low'] = df['low'].ffill().bfill().fillna(0)
        df['volume'] = df['volume'].fillna(0)

        close_prices = np.asarray(df['close'].values, dtype=np.float64)
        high_prices = np.asarray(df['high'].values, dtype=np.float64)
        low_prices = np.asarray(df['low'].values, dtype=np.float64)
        volume = np.asarray(df['volume'].values, dtype=np.float64)

        close_prices = np.nan_to_num(close_prices, nan=0.0)
        high_prices = np.nan_to_num(high_prices, nan=0.0)
        low_prices = np.nan_to_num(low_prices, nan=0.0)
        volume = np.nan_to_num(volume, nan=0.0)

        current_price = close_prices[-1]

        # RSI
        rsi = talib.RSI(close_prices, timeperiod=14)[-1]

        # MACD
        macd, macd_signal, macd_hist = talib.MACD(close_prices)
        macd_value = macd[-1]
        macd_signal_value = macd_signal[-1]
        macd_histogram = macd_hist[-1]
        macd_crossover = "bullish" if macd_histogram > 0 and macd_value > macd_signal_value else "bearish" if macd_histogram < 0 else "neutral"

        # ADX
        adx = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)[-1]

        # ATR
        atr = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)[-1]

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close_prices, timeperiod=20)
        bb_position = (close_prices[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1]) if (bb_upper[-1] - bb_lower[-1]) > 0 else 0.5

        # SMA
        sma_20 = talib.SMA(close_prices, timeperiod=20)[-1]
        sma_50 = talib.SMA(close_prices, timeperiod=50)[-1]
        sma_200 = talib.SMA(close_prices, timeperiod=200) if len(close_prices) >= 200 else None
        sma_200_value = float(sma_200[-1]) if sma_200 is not None and not np.isnan(sma_200[-1]) else None

        # EMA
        ema_20 = talib.EMA(close_prices, timeperiod=20)[-1]
        ema_50 = talib.EMA(close_prices, timeperiod=50)[-1]
        ema_200 = talib.EMA(close_prices, timeperiod=200) if len(close_prices) >= 200 else None
        ema_200_value = float(ema_200[-1]) if ema_200 is not None and not np.isnan(ema_200[-1]) else None

        # OBV
        obv = talib.OBV(close_prices, volume)
        obv_value = float(obv[-1]) if not np.isnan(obv[-1]) else 0
        obv_trend = "bullish" if len(obv) >= 5 and obv[-1] > obv[-5] else "bearish"

        # Volume Profile
        try:
            price_ranges = np.linspace(df['low'].min(), df['high'].max(), 20)
            volume_profile = {}
            for i in range(len(price_ranges) - 1):
                mask = (df['close'] >= price_ranges[i]) & (df['close'] < price_ranges[i+1])
                volume_profile[float(price_ranges[i])] = float(df[mask]['volume'].sum())
            if volume_profile and len(volume_profile) > 0:
                poc_price = max(volume_profile, key=volume_profile.get)
            else:
                poc_price = current_price
        except (ValueError, TypeError, KeyError) as e:
            logger.warning(f"Erro ao calcular volume profile: {e}")
            poc_price = current_price
            volume_profile = {}

        # Fibonacci
        period_high = df['high'].max()
        period_low = df['low'].min()
        fib_range = period_high - period_low

        fib_levels = {
            "fib_0": float(period_high),
            "fib_23.6": float(period_high - (fib_range * 0.236)),
            "fib_38.2": float(period_high - (fib_range * 0.382)),
            "fib_50": float(period_high - (fib_range * 0.50)),
            "fib_61.8": float(period_high - (fib_range * 0.618)),
            "fib_100": float(period_low)
        }

        # Trend determination
        adx_value = float(adx) if not np.isnan(adx) else 25

        if ema_200_value:
            if current_price > ema_20 > ema_50 > ema_200_value:
                trend = "strong_bullish" if adx_value > 25 else "bullish"
            elif current_price > ema_20 > ema_50:
                trend = "bullish"
            elif current_price < ema_20 < ema_50 < ema_200_value:
                trend = "strong_bearish" if adx_value > 25 else "bearish"
            elif current_price < ema_20 < ema_50:
                trend = "bearish"
            else:
                trend = "neutral"
        else:
            if current_price > ema_20 > ema_50:
                trend = "strong_bullish" if adx_value > 25 else "bullish"
            elif current_price < ema_20 < ema_50:
                trend = "strong_bearish" if adx_value > 25 else "bearish"
            else:
                trend = "neutral"

        # Momentum
        if rsi > 70:
            momentum = "overbought"
        elif rsi < 30:
            momentum = "oversold"
        elif rsi > 50:
            momentum = "bullish"
        elif rsi < 50:
            momentum = "bearish"
        else:
            momentum = "neutral"

        support = min([fib_levels["fib_61.8"], bb_lower[-1], poc_price])
        resistance = max([fib_levels["fib_38.2"], bb_upper[-1], poc_price])

        market_structure = _analyze_market_structure(df)

        return {
            "symbol": symbol,
            "trend": trend,
            "momentum": momentum,
            "volatility": "high" if atr > current_price * 0.02 else "normal",
            "market_structure": market_structure,
            "indicators": {
                "rsi": float(rsi) if not np.isnan(rsi) else 50,
                "macd": float(macd_value) if not np.isnan(macd_value) else 0,
                "macd_signal": float(macd_signal_value) if not np.isnan(macd_signal_value) else 0,
                "macd_histogram": float(macd_histogram) if not np.isnan(macd_histogram) else 0,
                "macd_crossover": macd_crossover,
                "adx": float(adx) if not np.isnan(adx) else 25,
                "atr": float(atr) if not np.isnan(atr) else current_price * 0.01,
                "bb_position": float(bb_position) if not np.isnan(bb_position) else 0.5,
                "bb_upper": float(bb_upper[-1]) if not np.isnan(bb_upper[-1]) else current_price * 1.05,
                "bb_middle": float(bb_middle[-1]) if not np.isnan(bb_middle[-1]) else current_price,
                "bb_lower": float(bb_lower[-1]) if not np.isnan(bb_lower[-1]) else current_price * 0.95,
                "sma_20": float(sma_20) if not np.isnan(sma_20) else current_price,
                "sma_50": float(sma_50) if not np.isnan(sma_50) else current_price,
                "sma_200": sma_200_value,
                "ema_20": float(ema_20) if not np.isnan(ema_20) else current_price,
                "ema_50": float(ema_50) if not np.isnan(ema_50) else current_price,
                "ema_200": ema_200_value,
                "obv": obv_value,
                "obv_trend": obv_trend
            },
            "volume_profile": {
                "poc_price": float(poc_price),
                "high_volume_zones": sorted(volume_profile.items(), key=lambda x: x[1], reverse=True)[:3] if volume_profile and len(volume_profile) > 0 else []
            },
            "fibonacci_levels": fib_levels,
            "support": float(support) if not np.isnan(support) else current_price * 0.95,
            "resistance": float(resistance) if not np.isnan(resistance) else current_price * 1.05,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        logger.exception(f"[{symbol}] Erro na análise técnica ({error_type}): {error_msg}")
        return {
            "error": f"Erro na análise técnica ({error_type}): {error_msg}",
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type
        }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _classify_rsi(rsi: float) -> Dict[str, str]:
    """Classifica RSI em zona e hint de ação"""
    try:
        if rsi < 30:
            return {"zone": "oversold", "action_hint": "potential_buy"}
        elif rsi < 40:
            return {"zone": "approaching_oversold", "action_hint": "potential_buy"}
        elif rsi < 60:
            return {"zone": "neutral", "action_hint": "wait"}
        elif rsi < 70:
            return {"zone": "approaching_overbought", "action_hint": "potential_sell"}
        else:
            return {"zone": "overbought", "action_hint": "potential_sell"}
    except Exception as e:
        logger.warning(f"Erro em _classify_rsi: {e}")
        return {"zone": "neutral", "action_hint": "wait"}

def _interpret_adx(adx: float) -> str:
    """Interpreta força da tendência baseado no ADX"""
    try:
        if adx < 20:
            return "no_trend"
        elif adx < 25:
            return "weak"
        elif adx < 50:
            return "moderate"
        else:
            return "strong"
    except Exception as e:
        logger.warning(f"Erro em _interpret_adx: {e}")
        return "no_trend"

def _interpret_macd_momentum(histogram: float, prev_histogram: Optional[float] = None) -> str:
    """Determina direção do momentum do MACD"""
    try:
        if prev_histogram is None:
            return "accelerating_up" if histogram > 0 else "accelerating_down"
        if histogram > 0 and histogram > prev_histogram:
            return "accelerating_up"
        elif histogram > 0 and histogram < prev_histogram:
            return "decelerating_up"
        elif histogram < 0 and histogram < prev_histogram:
            return "accelerating_down"
        else:
            return "decelerating_down"
    except Exception as e:
        logger.warning(f"Erro em _interpret_macd_momentum: {e}")
        return "neutral"

def _classify_bollinger_position(position: float) -> str:
    """Classifica posição nas Bollinger Bands"""
    try:
        if position < 0.2:
            return "lower_band"
        elif position < 0.4:
            return "below_middle"
        elif position < 0.6:
            return "middle"
        elif position < 0.8:
            return "above_middle"
        else:
            return "upper_band"
    except Exception as e:
        logger.warning(f"Erro em _classify_bollinger_position: {e}")
        return "middle"

def _detect_ema_alignment(ema20: float, ema50: float, ema200: Optional[float], price: float) -> str:
    """Detecta alinhamento das EMAs"""
    try:
        if ema200 is None:
            if price > ema20 > ema50:
                return "bullish_stack"
            elif price < ema20 < ema50:
                return "bearish_stack"
            else:
                return "mixed"
        else:
            if price > ema20 > ema50 > ema200:
                return "bullish_stack"
            elif price < ema20 < ema50 < ema200:
                return "bearish_stack"
            else:
                return "mixed"
    except Exception as e:
        logger.warning(f"Erro em _detect_ema_alignment: {e}")
        return "mixed"

def _interpret_funding_rate(rate: float) -> str:
    """Interpreta funding rate"""
    try:
        if rate > 0.01:
            return "crowded_long"
        elif rate > 0.005:
            return "slightly_long"
        elif rate > -0.005:
            return "neutral"
        elif rate > -0.01:
            return "slightly_short"
        else:
            return "crowded_short"
    except Exception as e:
        logger.warning(f"Erro em _interpret_funding_rate: {e}")
        return "neutral"

def _classify_orderbook_imbalance(imbalance: float) -> str:
    """Classifica pressão do orderbook"""
    try:
        if imbalance > 0.5:
            return "strong_buy_pressure"
        elif imbalance > 0.2:
            return "buy_pressure"
        elif imbalance > -0.2:
            return "neutral"
        elif imbalance > -0.5:
            return "sell_pressure"
        else:
            return "strong_sell_pressure"
    except Exception as e:
        logger.warning(f"Erro em _classify_orderbook_imbalance: {e}")
        return "neutral"

def _calculate_suggested_stops(atr: float, price: float, signal_type: str = "BUY") -> Dict[str, float]:
    """Calcula stop loss e take profits sugeridos baseado em ATR"""
    try:
        atr_pct = (atr / price) * 100
        stop_pct = 1.5 * atr_pct
        tp1_pct = 2.0 * atr_pct
        tp2_pct = 3.0 * atr_pct
        return {
            "suggested_stop_pct": stop_pct,
            "suggested_tp1_pct": tp1_pct,
            "suggested_tp2_pct": tp2_pct
        }
    except Exception as e:
        logger.warning(f"Erro em _calculate_suggested_stops: {e}")
        return {
            "suggested_stop_pct": 2.0,
            "suggested_tp1_pct": 2.5,
            "suggested_tp2_pct": 5.0
        }
