"""
Backtesting and optimization module.
=====================================

Origem:
- Score composto (30/30/20/20): sinais/backtesting/optimization_engine.py
- Backtest engine: implementação local usando BinanceClient + TA-Lib
- Walk-forward: implementação local

Exports:
- BacktestParams: parâmetros configuráveis do backtest
- BacktestMetrics: métricas calculadas (win rate, return, sharpe, drawdown)
- BacktestEngine: motor de simulação de trades
- OptimizationEngine: otimizador de parâmetros com score composto
- run_optimization: função de conveniência para otimização rápida
- apply_best_params: converte resultado em BacktestParams
"""

from src.backtesting.backtest_engine import (
    BacktestEngine,
    BacktestMetrics,
    BacktestParams,
    Trade,
)
from src.backtesting.optimization_engine import (
    OptimizationEngine,
    OptimizationResult,
    WalkForwardWindow,
    apply_best_params,
    run_optimization,
)

__all__ = [
    "BacktestEngine",
    "BacktestMetrics",
    "BacktestParams",
    "Trade",
    "OptimizationEngine",
    "OptimizationResult",
    "WalkForwardWindow",
    "run_optimization",
    "apply_best_params",
]
