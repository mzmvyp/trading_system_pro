"""Tests for indicators module"""
import pytest
from src.analysis.indicators import (
    _classify_rsi,
    _interpret_adx,
    _interpret_macd_momentum,
    _classify_bollinger_position,
    _detect_ema_alignment,
    _interpret_funding_rate,
    _classify_orderbook_imbalance,
    _calculate_suggested_stops
)


class TestClassifyRsi:
    def test_oversold(self):
        result = _classify_rsi(25)
        assert result["zone"] == "oversold"
        assert result["action_hint"] == "potential_buy"

    def test_overbought(self):
        result = _classify_rsi(75)
        assert result["zone"] == "overbought"
        assert result["action_hint"] == "potential_sell"

    def test_neutral(self):
        result = _classify_rsi(50)
        assert result["zone"] == "neutral"
        assert result["action_hint"] == "wait"

    def test_approaching_oversold(self):
        result = _classify_rsi(35)
        assert result["zone"] == "approaching_oversold"

    def test_approaching_overbought(self):
        result = _classify_rsi(65)
        assert result["zone"] == "approaching_overbought"


class TestInterpretAdx:
    def test_strong_trend(self):
        assert _interpret_adx(55) == "strong"

    def test_moderate_trend(self):
        assert _interpret_adx(30) == "moderate"

    def test_weak_trend(self):
        assert _interpret_adx(22) == "weak"

    def test_no_trend(self):
        assert _interpret_adx(15) == "no_trend"


class TestInterpretMacdMomentum:
    def test_accelerating_up(self):
        assert _interpret_macd_momentum(5, 3) == "accelerating_up"

    def test_decelerating_up(self):
        assert _interpret_macd_momentum(3, 5) == "decelerating_up"

    def test_accelerating_down(self):
        assert _interpret_macd_momentum(-5, -3) == "accelerating_down"

    def test_decelerating_down(self):
        assert _interpret_macd_momentum(-3, -5) == "decelerating_down"


class TestDetectEmaAlignment:
    def test_bullish_stack(self):
        result = _detect_ema_alignment(100, 95, 90, 105)
        assert result == "bullish_stack"

    def test_bearish_stack(self):
        result = _detect_ema_alignment(100, 105, 110, 95)
        assert result == "bearish_stack"

    def test_mixed(self):
        result = _detect_ema_alignment(100, 95, 110, 105)
        assert result == "mixed"

    def test_bullish_without_ema200(self):
        result = _detect_ema_alignment(100, 95, None, 105)
        assert result == "bullish_stack"

    def test_bearish_without_ema200(self):
        result = _detect_ema_alignment(100, 105, None, 95)
        assert result == "bearish_stack"


class TestClassifyBollingerPosition:
    def test_lower_band(self):
        assert _classify_bollinger_position(0.1) == "lower_band"

    def test_upper_band(self):
        assert _classify_bollinger_position(0.9) == "upper_band"

    def test_middle(self):
        assert _classify_bollinger_position(0.5) == "middle"


class TestInterpretFundingRate:
    def test_crowded_long(self):
        assert _interpret_funding_rate(0.02) == "crowded_long"

    def test_crowded_short(self):
        assert _interpret_funding_rate(-0.02) == "crowded_short"

    def test_neutral(self):
        assert _interpret_funding_rate(0.001) == "neutral"


class TestClassifyOrderbookImbalance:
    def test_strong_buy_pressure(self):
        assert _classify_orderbook_imbalance(0.6) == "strong_buy_pressure"

    def test_strong_sell_pressure(self):
        assert _classify_orderbook_imbalance(-0.6) == "strong_sell_pressure"

    def test_neutral(self):
        assert _classify_orderbook_imbalance(0.0) == "neutral"


class TestCalculateSuggestedStops:
    def test_returns_valid_stops(self):
        result = _calculate_suggested_stops(1800, 90000)
        assert "suggested_stop_pct" in result
        assert "suggested_tp1_pct" in result
        assert "suggested_tp2_pct" in result
        assert result["suggested_stop_pct"] > 0
        assert result["suggested_tp1_pct"] > result["suggested_stop_pct"]
        assert result["suggested_tp2_pct"] > result["suggested_tp1_pct"]
