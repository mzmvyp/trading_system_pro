"""
Configurações do sistema de trading
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, validator

class Settings(BaseSettings):
    # ========================================
    # MODO DE TRADING: "paper" ou "real"
    # ========================================
    trading_mode: str = Field(default="paper", description="Modo de trading: 'paper' ou 'real'")

    # ========================================
    # FILTRO DE SINAIS: Quais fontes de sinal aceitar
    # ========================================
    accept_agno_signals: bool = True
    accept_deepseek_signals: bool = False

    # Configurações da API Binance
    binance_api_key: Optional[str] = None
    binance_secret_key: Optional[str] = None

    # Configurações do DeepSeek
    deepseek_base_url: str = "https://api.deepseek.com/v1"

    # Configurações do sistema
    log_level: str = "INFO"
    trading_symbol: str = "BTCUSDT"

    # Configurações de Risk Management
    max_risk_per_trade: float = 0.02
    max_drawdown: float = 0.15
    max_exposure: float = 0.50
    max_daily_trades: int = 3
    base_risk_percentage: float = 0.01

    # ========================================
    # GESTAO DE CAPITAL E RISCO POR TRADE
    # ========================================
    initial_capital: float = 10000.0
    risk_percent_per_trade: float = 2.0

    # Configurações de Confiança
    min_confidence_0_10: int = 7

    # Configurações de Intervalo de Análise
    min_analysis_interval_hours: float = 2.0

    # Top 10 criptomoedas para análise
    top_crypto_pairs: list = [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT",
        "XRPUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
        "PAXGUSDT"
    ]

    # ========================================
    # TIMEOUT POR TIPO DE OPERAÇÃO
    # ========================================
    timeout_scalp_hours: float = 0.5
    timeout_day_trade_hours: float = 8.0
    timeout_swing_trade_hours: float = 120.0
    timeout_position_trade_hours: float = 672.0

    # ========================================
    # REAVALIAÇÃO DE SINAIS ATIVOS
    # ========================================
    reevaluation_enabled: bool = True
    reevaluation_interval_hours: float = 2.0
    reevaluation_min_time_open_hours: float = 1.0
    reevaluation_min_confidence: int = 7

    # ========================================
    # VALIDAÇÃO ML
    # ========================================
    ml_validation_enabled: bool = True
    ml_validation_threshold: float = 0.65
    ml_validation_required: bool = False

    # ========================================
    # ONLINE LEARNING
    # ========================================
    ml_online_learning_enabled: bool = True
    ml_retrain_threshold: int = 50
    ml_min_improvement: float = 0.0


    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"

settings = Settings()
