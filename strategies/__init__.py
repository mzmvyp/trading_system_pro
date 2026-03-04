"""Módulo de estratégias de trading."""
from .base_strategy import BaseStrategy
from .signal_types import SignalType, SignalPriority
from .trend_following import TrendFollowingStrategy, TrendFollowingConfig
from .mean_reversion import MeanReversionStrategy, MeanReversionConfig
from .breakout import BreakoutStrategy
from .swing import SwingStrategy

__all__ = [
    "BaseStrategy",
    "SignalType",
    "SignalPriority",
    "TrendFollowingStrategy",
    "TrendFollowingConfig",
    "MeanReversionStrategy",
    "MeanReversionConfig",
    "BreakoutStrategy",
    "SwingStrategy",
]
