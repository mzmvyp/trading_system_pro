"""Tests for risk manager module"""
import pytest
from unittest.mock import patch, MagicMock
from src.trading.risk_manager import validate_risk_and_position


class TestValidateRiskAndPosition:
    """Tests for validate_risk_and_position function"""

    @patch('src.trading.risk_manager._calculate_current_drawdown', return_value=0.0)
    @patch('src.trading.risk_manager._get_daily_trades_count', return_value=0)
    @patch('src.trading.risk_manager.os.path.exists', return_value=False)
    def test_validate_buy_signal_high_confidence(self, mock_exists, mock_daily, mock_dd):
        """High confidence buy signal should be executable"""
        signal = {
            "signal": "BUY",
            "entry_price": 90000,
            "stop_loss": 88200,  # 2% risk
            "confidence": 8,
            "source": "DEEPSEEK"
        }
        result = validate_risk_and_position(signal, "BTCUSDT")
        assert result["can_execute"] is True
        assert result["risk_level"] == "acceptable"
        assert result["recommended_position_size"] > 0

    @patch('src.trading.risk_manager._calculate_current_drawdown', return_value=0.0)
    @patch('src.trading.risk_manager._get_daily_trades_count', return_value=0)
    @patch('src.trading.risk_manager.os.path.exists', return_value=False)
    def test_validate_buy_signal_low_confidence(self, mock_exists, mock_daily, mock_dd):
        """Low confidence signal should be rejected"""
        signal = {
            "signal": "BUY",
            "entry_price": 90000,
            "stop_loss": 88200,
            "confidence": 5,
            "source": "DEEPSEEK"
        }
        result = validate_risk_and_position(signal, "BTCUSDT")
        assert result["can_execute"] is False
        assert "Confianca muito baixa" in result["reason"]

    @patch('src.trading.risk_manager._calculate_current_drawdown', return_value=0.0)
    @patch('src.trading.risk_manager._get_daily_trades_count', return_value=0)
    @patch('src.trading.risk_manager.os.path.exists', return_value=False)
    def test_validate_high_risk_percentage(self, mock_exists, mock_daily, mock_dd):
        """Signal with risk > 5% should be rejected"""
        signal = {
            "signal": "BUY",
            "entry_price": 90000,
            "stop_loss": 83000,  # ~7.8% risk
            "confidence": 8,
            "source": "DEEPSEEK"
        }
        result = validate_risk_and_position(signal, "BTCUSDT")
        assert result["can_execute"] is False
        assert "Risco muito alto" in result["reason"]

    @patch('src.trading.risk_manager._calculate_current_drawdown', return_value=0.0)
    @patch('src.trading.risk_manager._get_daily_trades_count', return_value=5)
    @patch('src.trading.risk_manager.os.path.exists', return_value=False)
    def test_validate_daily_trade_limit(self, mock_exists, mock_daily, mock_dd):
        """Should reject when daily trade limit reached"""
        signal = {
            "signal": "BUY",
            "entry_price": 90000,
            "stop_loss": 88200,
            "confidence": 8,
            "source": "DEEPSEEK"
        }
        result = validate_risk_and_position(signal, "BTCUSDT")
        assert result["can_execute"] is False
        assert "Limite diário" in result["reason"]

    def test_validate_no_signal(self):
        """NO_SIGNAL should not execute"""
        signal = {"signal": "NO_SIGNAL", "confidence": 0}
        result = validate_risk_and_position(signal, "BTCUSDT")
        assert result["can_execute"] is False

    def test_validate_hold_signal(self):
        """HOLD signal should not execute"""
        signal = {"signal": "HOLD", "confidence": 0}
        result = validate_risk_and_position(signal, "BTCUSDT")
        assert result["can_execute"] is False

    @patch('src.trading.risk_manager._calculate_current_drawdown', return_value=0.0)
    @patch('src.trading.risk_manager._get_daily_trades_count', return_value=0)
    @patch('src.trading.risk_manager.os.path.exists', return_value=False)
    def test_validate_missing_prices(self, mock_exists, mock_daily, mock_dd):
        """Signal without entry_price should be rejected"""
        signal = {
            "signal": "BUY",
            "entry_price": 0,
            "stop_loss": 0,
            "confidence": 8,
            "source": "DEEPSEEK"
        }
        result = validate_risk_and_position(signal, "BTCUSDT")
        assert result["can_execute"] is False
