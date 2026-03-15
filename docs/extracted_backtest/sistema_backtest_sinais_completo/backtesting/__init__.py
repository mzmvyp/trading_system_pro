# -*- coding: utf-8 -*-
"""
Backtesting Module - Sistema de backtesting e otimização
"""

from .backtest_engine import BacktestEngine, BacktestResult, TradeResult
from .optimization_engine import OptimizationEngine, OptimizationConfig, OptimizationResult
from .data_analyzer import DataAnalyzer, DataQualityReport, StreamPerformanceReport

__all__ = [
    'BacktestEngine',
    'BacktestResult', 
    'TradeResult',
    'OptimizationEngine',
    'OptimizationConfig',
    'OptimizationResult',
    'DataAnalyzer',
    'DataQualityReport',
    'StreamPerformanceReport'
]
