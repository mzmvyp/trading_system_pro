"""Backtesting and optimization module."""
__all__ = ["OptimizationParams", "BacktestMetrics", "run_optimization", "apply_best_params"]


def __getattr__(name: str):
    if name in ("OptimizationParams", "BacktestMetrics", "run_optimization", "apply_best_params"):
        from src.backtesting import optimization_engine
        return getattr(optimization_engine, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
