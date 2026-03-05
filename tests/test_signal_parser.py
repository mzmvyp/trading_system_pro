"""Tests for signal parser module"""
import pytest
from src.trading.signal_parser import (
    extract_balanced_json,
    extract_price_from_text,
    get_default_price
)


class TestExtractBalancedJson:
    def test_simple_json(self):
        text = '```json\n{"signal": "BUY", "confidence": 8}\n```'
        result = extract_balanced_json(text)
        assert result is not None
        assert '"signal": "BUY"' in result

    def test_nested_json(self):
        text = '```json\n{"signal": "BUY", "nested": {"key": "value"}}\n```'
        result = extract_balanced_json(text)
        assert result is not None
        assert '"nested"' in result
        assert result.count('{') == result.count('}')

    def test_json_without_code_block(self):
        text = 'The analysis shows {"signal": "SELL", "confidence": 7}'
        result = extract_balanced_json(text)
        assert result is not None
        assert '"signal": "SELL"' in result

    def test_no_json(self):
        text = 'No JSON here, just plain text analysis'
        result = extract_balanced_json(text)
        assert result is None

    def test_json_with_surrounding_text(self):
        text = 'Here is my analysis:\n```json\n{"signal": "BUY", "entry_price": 90000}\n```\nEnd of analysis.'
        result = extract_balanced_json(text)
        assert result is not None
        assert '"entry_price": 90000' in result


class TestExtractPriceFromText:
    def test_dollar_sign_price(self):
        text = "Current price is $90,563.50"
        result = extract_price_from_text(text)
        assert result == 90563.50

    def test_entry_price(self):
        text = "entry_price: $3,200.50"
        result = extract_price_from_text(text)
        assert result == 3200.50

    def test_no_price(self):
        text = "No price mentioned here"
        result = extract_price_from_text(text)
        assert result is None

    def test_empty_text(self):
        result = extract_price_from_text("")
        assert result is None

    def test_none_text(self):
        result = extract_price_from_text(None)
        assert result is None


class TestGetDefaultPrice:
    def test_btc_default(self):
        assert get_default_price("BTCUSDT") == 90000

    def test_eth_default(self):
        assert get_default_price("ETHUSDT") == 3000

    def test_unknown_symbol(self):
        assert get_default_price("UNKNOWNUSDT") == 100
