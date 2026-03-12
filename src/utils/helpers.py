"""
Funções auxiliares para estratégias e filtros (portado de smart_trading_system).
"""
from typing import Any, Dict, List

import numpy as np


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divisão segura evitando divisão por zero."""
    try:
        if denominator == 0 or np.isnan(denominator) or np.isinf(denominator):
            return default
        result = numerator / denominator
        return default if np.isnan(result) or np.isinf(result) else result
    except Exception:
        return default


def normalize_value(value: float, min_val: float, max_val: float) -> float:
    """Normaliza valor entre 0 e 1."""
    if max_val == min_val:
        return 0.5
    return (value - min_val) / (max_val - min_val)


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """Calcula mudança percentual."""
    if old_value == 0:
        return 0.0
    return ((new_value - old_value) / old_value) * 100


def find_local_extremes(prices: List[float], window: int = 5) -> Dict[str, List[int]]:
    """Encontra máximos e mínimos locais."""
    if len(prices) < window * 2 + 1:
        return {'highs': [], 'lows': []}
    highs = []
    lows = []
    for i in range(window, len(prices) - window):
        if all(prices[i] >= prices[j] for j in range(i - window, i + window + 1)):
            if prices[i] > max(prices[i - window:i] + prices[i + 1:i + window + 1]):
                highs.append(i)
        if all(prices[i] <= prices[j] for j in range(i - window, i + window + 1)):
            if prices[i] < min(prices[i - window:i] + prices[i + 1:i + window + 1]):
                lows.append(i)
    return {'highs': highs, 'lows': lows}


def calculate_risk_reward_ratio(entry: float, stop_loss: float, take_profit: float) -> float:
    """Calcula relação risco/recompensa."""
    risk = abs(entry - stop_loss)
    reward = abs(take_profit - entry)
    return safe_divide(reward, risk, 0.0)


def sanitize_numeric_input(value: Any, default: float = 0.0) -> float:
    """Sanitiza entrada numérica."""
    try:
        if value is None:
            return default
        if hasattr(value, '__float__'):
            float_val = float(value)
        else:
            float_val = float(value)
        return default if np.isnan(float_val) or np.isinf(float_val) else float_val
    except Exception:
        return default
