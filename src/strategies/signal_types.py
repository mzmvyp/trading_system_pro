"""
Tipos de sinal para estratégias (portado de smart_trading_system).
"""
from enum import Enum


class SignalType(Enum):
    """Tipos de sinal."""
    BUY = "buy"
    SELL = "sell"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"


class SignalPriority(Enum):
    """Prioridades de sinal."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
