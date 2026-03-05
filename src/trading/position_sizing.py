"""
Position Sizing - Risk-based position size calculator.
Source: agente_trade_futuros
Features:
- Risk percentage-based sizing
- Position validation against capital and risk limits
- Multi-position total risk calculation
"""

import logging
from typing import Dict, Optional

from src.core.logger import get_logger

logger = get_logger(__name__)


class PositionSizing:
    """Risk-based position sizing calculator."""

    def __init__(
        self,
        risk_per_trade: float = 0.01,
        max_risk_per_trade: float = 0.025,
        max_total_risk: float = 0.06,
    ):
        self.risk_per_trade = risk_per_trade
        self.max_risk_per_trade = max_risk_per_trade
        self.max_total_risk = max_total_risk
        self.total_capital = 0.0
        self.available_capital = 0.0

    def set_capital(self, total_capital: float, available_capital: Optional[float] = None):
        """Set current capital levels."""
        self.total_capital = total_capital
        self.available_capital = available_capital if available_capital is not None else total_capital

    def calculate(
        self, entry_price: float, stop_loss: float,
        risk_percentage: Optional[float] = None,
    ) -> Dict:
        """
        Calculate position size based on entry, stop, and risk tolerance.

        Returns dict with success, position_size, position_value, risk details.
        """
        if self.available_capital <= 0:
            return {"success": False, "error": "Capital not set"}

        risk_pct = risk_percentage if risk_percentage is not None else self.risk_per_trade

        risk_per_unit = abs(entry_price - stop_loss)
        if risk_per_unit <= 0:
            return {"success": False, "error": "Stop too close to entry"}

        risk_amount = self.available_capital * risk_pct
        position_size = risk_amount / risk_per_unit
        position_value = position_size * entry_price

        # Cap at available capital
        if position_value > self.available_capital:
            position_value = self.available_capital * 0.95
            position_size = position_value / entry_price
            risk_amount = position_size * risk_per_unit

        actual_risk_pct = (risk_amount / self.available_capital) * 100

        return {
            "success": True,
            "position_size": position_size,
            "position_value": position_value,
            "risk_amount": risk_amount,
            "risk_percentage": actual_risk_pct,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "risk_per_unit": risk_per_unit,
        }

    def validate(self, position_size: float, entry_price: float, stop_loss: float) -> Dict:
        """Validate a position against risk limits."""
        position_value = position_size * entry_price
        risk_per_unit = abs(entry_price - stop_loss)
        risk_amount = position_size * risk_per_unit
        risk_pct = (risk_amount / self.available_capital) * 100 if self.available_capital > 0 else 100

        validations = {
            "capital_available": position_value <= self.available_capital,
            "risk_per_trade": risk_pct <= self.max_risk_per_trade * 100,
            "position_size_positive": position_size > 0,
        }

        return {
            "valid": all(validations.values()),
            "risk_percentage": risk_pct,
            "validations": validations,
        }

    def check_total_risk(self, positions: list) -> Dict:
        """Check total portfolio risk across all positions."""
        total_risk = 0
        for pos in positions:
            risk_per_unit = abs(pos["entry_price"] - pos["stop_loss"])
            total_risk += pos["position_size"] * risk_per_unit

        total_risk_pct = (total_risk / self.available_capital) * 100 if self.available_capital > 0 else 100

        return {
            "total_risk_percentage": total_risk_pct,
            "within_limits": total_risk_pct <= self.max_total_risk * 100,
        }
