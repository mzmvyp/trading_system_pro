"""Tests for config module"""
import pytest
import os
from unittest.mock import patch


class TestSettings:
    def test_default_settings(self):
        """Test that default settings are loaded correctly"""
        # Import fresh to get defaults
        from src.core.config import Settings
        s = Settings()
        assert s.trading_mode in ("paper", "real")
        assert s.min_confidence_0_10 >= 1
        assert s.min_confidence_0_10 <= 10

    def test_env_override(self):
        """Test that environment variables override defaults"""
        with patch.dict(os.environ, {"TRADING_MODE": "real"}):
            from src.core.config import Settings
            s = Settings()
            assert s.trading_mode == "real"
