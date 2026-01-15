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
    # "paper" = Simulado (não executa ordens reais)
    # "real" = Executa ordens reais na Binance Futures
    trading_mode: str = Field(default="paper", description="Modo de trading: 'paper' ou 'real'")

    # ========================================
    # FILTRO DE SINAIS: Quais fontes de sinal aceitar
    # ========================================
    # Por padrão, aceita apenas sinais do AGNO (não do DEEPSEEK direto)
    accept_agno_signals: bool = True
    accept_deepseek_signals: bool = False  # DESABILITADO por padrão

    # Configurações da API Binance (pública - não precisa de chaves)
    binance_api_key: Optional[str] = None
    binance_secret_key: Optional[str] = None

    # Twitter removido - análise de sentimento baseada apenas em dados de mercado

    # Configurações do DeepSeek
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    
    # Configurações do sistema
    log_level: str = "INFO"
    trading_symbol: str = "BTCUSDT"
    
    # Configurações de Risk Management
    max_risk_per_trade: float = 0.05  # Máximo 5% de risco por trade (aumentado para paper trading)
    max_drawdown: float = 0.40  # Máximo 40% de drawdown (aumentado para paper trading)
    max_exposure: float = 0.80  # Máximo 80% de exposição total (aumentado para paper trading)
    max_daily_trades: int = 5  # Máximo 5 trades por dia
    base_risk_percentage: float = 0.02  # 2% base de risco

    # ========================================
    # GESTAO DE CAPITAL E RISCO POR TRADE
    # ========================================
    # Capital inicial para paper trading (usado para calcular tamanho de posicao)
    initial_capital: float = 10000.0  # Capital inicial em USDT

    # Porcentagem do capital a arriscar por trade
    # Exemplo: 5% significa que se o stop loss for atingido, voce perde 5% do capital
    risk_percent_per_trade: float = 5.0  # 5% do capital arriscado por trade

    # Como calcular o tamanho da posicao:
    # 1. Risco em $ = capital * (risk_percent_per_trade / 100)
    # 2. Distancia do stop = |entry_price - stop_loss|
    # 3. Risco por unidade = distancia do stop
    # 4. Tamanho da posicao (unidades) = Risco em $ / Risco por unidade
    # 5. Valor total da posicao = tamanho * entry_price
    #
    # Exemplo com capital = $10,000, risco = 5%, entry = $100,000, stop = $98,000:
    # - Risco em $ = $10,000 * 0.05 = $500
    # - Distancia do stop = $2,000 (2%)
    # - Tamanho = $500 / $2,000 = 0.25 BTC
    # - Valor total = 0.25 * $100,000 = $25,000 (alavancagem ~2.5x)
    
    # Configurações de Confiança
    # UNIFICADO: Sempre usar escala 0-10
    min_confidence_0_10: int = 7  # Mínimo 7/10 para executar sinais
    # DEPRECATED: min_confidence_0_5 removido - sempre usar escala 0-10
    
    # Configurações de Intervalo de Análise
    min_analysis_interval_hours: float = 1.0  # Mínimo 1 hora entre análises do mesmo símbolo
    
    # Top 10 criptomoedas para análise
    top_crypto_pairs: list = [
        "BTCUSDT",   # Bitcoin
        "ETHUSDT",   # Ethereum
        "SOLUSDT",   # Solana
        "BNBUSDT",   # Binance Coin
        "ADAUSDT",   # Cardano
        "XRPUSDT",   # Ripple
        "DOGEUSDT",  # Dogecoin
        "AVAXUSDT",  # Avalanche
        "DOTUSDT",   # Polkadot
        "LINKUSDT",  # Chainlink
        "PAXGUSDT"   # PAX Gold
    ]

    # ========================================
    # TIMEOUT POR TIPO DE OPERAÇÃO
    # ========================================
    timeout_scalp_hours: float = 0.5        # 30 minutos
    timeout_day_trade_hours: float = 8.0    # 8 horas
    timeout_swing_trade_hours: float = 120.0  # 5 dias
    timeout_position_trade_hours: float = 672.0  # 28 dias

    # ========================================
    # REAVALIAÇÃO DE SINAIS ATIVOS
    # ========================================
    reevaluation_enabled: bool = True
    reevaluation_interval_hours: float = 2.0  # Reavaliar a cada 2 horas
    reevaluation_min_time_open_hours: float = 1.0  # Só reavaliar após 1h aberta
    reevaluation_min_confidence: int = 7  # Confiança mínima para agir na reavaliação

    # ========================================
    # VALIDAÇÃO ML - MODELO DE CONFLUÊNCIA
    # ========================================
    # Habilita validação de sinais usando modelo ML treinado
    ml_validation_enabled: bool = True
    # Threshold de probabilidade para considerar confluência (0.0 a 1.0)
    ml_validation_threshold: float = 0.5
    # Se True, só executa sinais que passam na validação ML
    # Se False, apenas loga a validação mas executa de qualquer forma
    ml_validation_required: bool = False
    
    # ========================================
    # ONLINE LEARNING - RETREINAMENTO AUTOMÁTICO
    # ========================================
    # Habilita coleta de dados para retreinamento do modelo
    ml_online_learning_enabled: bool = True
    # Número de novos exemplos necessários para disparar retreinamento
    ml_retrain_threshold: int = 50
    # Melhoria mínima necessária no F1 para salvar novo modelo
    ml_min_improvement: float = 0.0


    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignorar campos extras

settings = Settings()
