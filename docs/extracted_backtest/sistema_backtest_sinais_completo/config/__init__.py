"""
Módulo de Configuração do Trading Analyzer
"""

from .settings import (
    settings,
    DatabaseConfig,
    AnalysisConfig, 
    IndicatorConfig,
    PatternConfig,
    SystemConfig,
    Settings
)

__all__ = [
    'settings',
    'DatabaseConfig',
    'AnalysisConfig',
    'IndicatorConfig', 
    'PatternConfig',
    'SystemConfig',
    'Settings'
]

__version__ = "1.0.0"