"""
Configurações do sistema de trading
"""
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


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
    max_risk_per_trade: float = 0.02  # Máximo 2% de risco por trade
    max_drawdown: float = 0.15  # Máximo 15% de drawdown
    max_exposure: float = 0.50  # Máximo 50% de exposição total
    max_daily_trades: int = 5  # Máximo 5 trades por dia
    base_risk_percentage: float = 0.01  # 1% base de risco
    max_open_positions: int = 6  # Máximo 6 posições simultâneas

    # ========================================
    # GESTAO DE CAPITAL E RISCO POR TRADE
    # ========================================
    # Capital disponivel para trading (DEVE refletir saldo real na Binance)
    # IMPORTANTE: Atualizar este valor quando o saldo mudar significativamente
    initial_capital: float = 100.0  # Capital atual em USDT (saldo real Binance)

    # Porcentagem do capital a arriscar por trade
    # 5% = se o stop loss for atingido, perde 5% do capital
    risk_percent_per_trade: float = 2.0  # 2% do capital arriscado por trade (antes 5% — muito agressivo)

    # Como calcular o tamanho da posicao:
    # 1. Risco em $ = capital * (risk_percent_per_trade / 100)
    # 2. Distancia do stop em % = |entry - stop| / entry
    # 3. Tamanho da posicao = Risco em $ / distancia do stop em %
    # 4. Alavancagem implicita = Tamanho da posicao / capital
    #
    # Exemplo com capital = $100, risco = 5%, entry = $100, stop = $98:
    # - Risco em $ = $100 * 0.05 = $5
    # - Distancia do stop = 2%
    # - Tamanho da posicao = $5 / 2% = $250
    # - Alavancagem implicita = $250 / $100 = 2.5x
    # - Se stop bater, perde exatamente $5 (5% do capital)

    # Configurações de Confiança
    # UNIFICADO: Sempre usar escala 0-10
    # Mínimo 6/10 — sinais fracos passam mas são filtrados por ML, confluência, LSTM
    min_confidence_0_10: int = 7  # Mínimo 7/10 (antes 6 — executava sinais fracos)
    # DEPRECATED: min_confidence_0_5 removido - sempre usar escala 0-10

    # Configurações de Intervalo de Análise
    # CORRIGIDO: Aumentado para 4h para evitar overtrading severo
    # Análise dos 267 sinais mostrou que sinais a cada 2-7min destruíam performance
    min_analysis_interval_hours: float = 4.0  # Mínimo 4 horas entre análises do mesmo símbolo (aumentado de 2h)

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
        "PAXGUSDT",  # PAX Gold
    ]

    # ========================================
    # TOP MOVERS DINÂMICOS
    # ========================================
    # Adiciona pares com maior movimento (gainers + losers) à análise
    top_movers_enabled: bool = True
    top_movers_n_gainers: int = 3   # Top N que mais subiram (reduzido de 5 — evitar shitcoins)
    top_movers_n_losers: int = 2    # Top N que mais caíram (reduzido de 5 — shorts perdem muito)
    top_movers_min_volume_usdt: float = 200_000_000  # Volume mínimo 200M USDT (antes 50M — trazia lixo)

    # ========================================
    # BLACKLIST DE TOKENS (ilíquidos/consistentemente perdedores)
    # ========================================
    # Tokens que devem ser ignorados mesmo que apareçam como top movers
    # Identificados pela análise de 2000+ trades (Win Rate < 30% ou ilíquidos)
    token_blacklist: list = [
        "JCTUSDT",     # Ilíquido - preço manipulado (entry $20 → $0.003)
        "BRUSDT",      # SL > 20% - volatilidade extrema
        "SIRENUSDT",   # SL > 22% - volatilidade extrema
        "LYNUSDT",     # 0% Win Rate - consistentemente perdedor
        "UAIUSDT",     # 16.7% Win Rate - consistentemente perdedor
        "PIPPINUSDT",  # -38% ROI - meme coin volátil
        "NIGHTUSDT",   # -23% ROI - micro-cap sem liquidez
        "EDGEUSDT",    # -43% ROI - pump & dump
        "SKYAIUSDT",   # -33% ROI - micro-cap especulativo
        "RIVERUSDT",   # Micro-cap - volatilidade extrema
    ]

    # ========================================
    # FILTRO DE SELL (SHORT) - Mais restritivo
    # ========================================
    # Dados mostram SELL com 46% WR vs BUY com 67% WR
    # Shorts precisam de confiança maior para compensar menor acerto
    sell_min_confidence: int = 8  # Shorts precisam confiança 8/10 (WR 46% vs BUY 67%)
    sell_require_strong_trend: bool = True  # SELL só em tendência forte de baixa

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
    reevaluation_interval_hours: float = 0.5  # Reavaliar a cada 30min (antes 1h)
    reevaluation_min_time_open_hours: float = 0.25  # Primeira reavaliação após 15min (antes 1h)
    reevaluation_min_confidence: int = 7  # Confiança mínima para agir na reavaliação
    reevaluation_require_tp1_hit: bool = False  # Se True, só reavalia após TP1 ser atingido

    # ========================================
    # PROTEÇÃO AUTOMÁTICA DE LUCRO
    # ========================================
    # Move stop para breakeven quando lucro atinge X%
    auto_breakeven_enabled: bool = True
    auto_breakeven_trigger_pct: float = 1.5  # Ativa breakeven com 1.5% de lucro
    # Trailing stop automático quando lucro atinge X%
    auto_trailing_stop_enabled: bool = True
    auto_trailing_stop_trigger_pct: float = 2.5  # Ativa trailing com 2.5% de lucro
    auto_trailing_stop_distance_pct: float = 1.0  # Distância do trailing: 1% abaixo do máximo

    # ========================================
    # VALIDAÇÃO ML - MODELO COMO VOTO (NÃO VETO)
    # ========================================
    # ML é apenas 1 voto no sistema de confluência (10 votos possíveis)
    # Nunca bloqueia sozinho — precisa de maioria dos indicadores contra
    ml_validation_enabled: bool = True
    ml_validation_threshold: float = 0.6  # prob >= 0.6 = voto a favor, < 0.4 = voto contra
    ml_validation_required: bool = False  # ML não tem mais poder de veto

    # ========================================
    # ONLINE LEARNING - RETREINAMENTO AUTOMÁTICO
    # ========================================
    # Habilita coleta de dados para retreinamento do modelo
    ml_online_learning_enabled: bool = True
    # Número de novos exemplos necessários para disparar retreinamento
    ml_retrain_threshold: int = 50
    # Melhoria mínima necessária no F1 para salvar novo modelo
    ml_min_improvement: float = 0.0


    # ========================================
    # NOTIFICAÇÕES POR EMAIL (Yahoo Mail / SMTP)
    # ========================================
    # Configure no .env:
    #   EMAIL_SMTP_USER=seu_email@yahoo.com
    #   EMAIL_SMTP_PASSWORD=sua_senha_de_app
    #   EMAIL_SMTP_HOST=smtp.mail.yahoo.com  (padrão Yahoo)
    #   EMAIL_SMTP_PORT=587  (padrão TLS)
    #   EMAIL_TO=destinatario@email.com  (padrão: mesmo que EMAIL_SMTP_USER)
    email_notifications_enabled: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignorar campos extras

settings = Settings()
