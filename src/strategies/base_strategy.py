"""
Classe abstrata base para estratégias de trading.
Todas as estratégias devem herdar de BaseStrategy e implementar analyze().
"""
from abc import ABC, abstractmethod
from typing import Any, Dict

import pandas as pd


class BaseStrategy(ABC):
    """Estratégia abstrata. Nome e timeframes recomendados devem ser definidos nas subclasses."""

    name: str = "base"
    timeframes: list = []

    @abstractmethod
    def analyze(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Analisa o DataFrame de OHLCV e retorna sinais e análise.

        Args:
            df: DataFrame com colunas de preço (open, high, low, close, volume).
            symbol: Símbolo do ativo (ex: BTCUSDT).
            timeframe: Timeframe (ex: 1h, 4h).

        Returns:
            Dict com chaves como 'signals', 'analysis', 'metadata'.
        """
        pass
