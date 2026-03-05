"""
Custom exceptions for the trading system
"""


class TradingSystemError(Exception):
    """Base exception for trading system errors"""
    pass


class ExchangeError(TradingSystemError):
    """Error communicating with exchange API"""
    pass


class RateLimitError(ExchangeError):
    """Rate limit exceeded"""
    pass


class CircuitBreakerOpenError(ExchangeError):
    """Circuit breaker is open, rejecting calls"""
    pass


class InsufficientDataError(TradingSystemError):
    """Not enough data for analysis"""
    pass


class SignalValidationError(TradingSystemError):
    """Signal failed validation"""
    pass


class RiskLimitExceededError(TradingSystemError):
    """Risk limits exceeded"""
    pass


class ConfigurationError(TradingSystemError):
    """Configuration error"""
    pass
