"""
Filtro fundamental (stub portado de smart_trading_system).
"""
from typing import Dict, Any
import pandas as pd
from logger import get_logger

logger = get_logger(__name__)


class FundamentalFilter:
    """Filtro baseado em dados fundamentais (quando disponíveis)."""

    def apply(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Retorna se passa no filtro fundamental."""
        return {"passed": True, "reasons": []}
