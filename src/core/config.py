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
    max_daily_trades: int = 8  # Máximo 8 trades/dia (up from 5 — agora só passam sinais bons)
    base_risk_percentage: float = 0.01  # 1% base de risco
    # REDUZIDO 6 -> 3: 6 posições altcoin simultâneas em mercados correlacionados
    # produzem liquidações em cascata quando BTC se move (15/abr: LINK+DOT+ADA+XRP
    # todos LONG perderam juntos). 3 posições com filtro de correlação evitam isso.
    max_open_positions: int = 3

    # ========================================
    # GESTAO DE CAPITAL E RISCO POR TRADE
    # ========================================
    # Capital disponivel para trading (DEVE refletir saldo real na Binance)
    # IMPORTANTE: Atualizar este valor quando o saldo mudar significativamente
    initial_capital: float = 100.0  # Capital atual em USDT (saldo real Binance)

    # Porcentagem do capital a arriscar por trade
    # REDUZIDO 5% -> 1%: com 5% e leverage 40x, sequência de 3 stops = -15% capital
    # 1% mantém capital sustentável durante drawdowns. Kelly aproximado para
    # WR 38% / RR 1.0 sugere risco ótimo abaixo de 1.5%.
    risk_percent_per_trade: float = 1.0  # 1% do capital arriscado por trade

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
        "JCTUSDT",     # Dados corrompidos (entry $20 -> $0.003)
        "4USDT",       # PnL gigante (micro-cap, dados nao confiaveis)
        "BRUSDT",      # SL > 20% - volatilidade extrema
        "SIRENUSDT",   # SL > 22% - volatilidade extrema
        "LYNUSDT",     # WR 25%, PnL -66.81% (N=12)
        "UAIUSDT",     # WR 16.7% - consistentemente perdedor
        "ZECUSDT",     # WR 0%, PnL -26.10% (N=8)
        "KNCUSDT",     # WR 0%, PnL -14.82% (N=5)
        "PLAYUSDT",    # WR 12%, PnL -30.60% (N=8)
        "DUSKUSDT",    # WR 11%, PnL -22.35% (N=9)
        "BANKUSDT",    # WR 14%, PnL -18.25% (N=7)
        "MUSDT",       # WR 14%, PnL -17.77% (N=7)
        "PIPPINUSDT",  # WR 20%, PnL -37.57% (N=20)
        "CTSIUSDT",    # WR 20%, PnL -27.65% (N=10)
        "EDGEUSDT",    # WR 18%, PnL -10.90% (N=17)
        "RIVERUSDT",   # WR 16.7%, PnL -31.19% (N=6) — abril analise
        "AVAXUSDT",    # WR 10%, PnL -15.29% (N=10) — abril analise
        "DOGEUSDT",    # WR 20%, PnL -10.60% (N=5) — abril analise
    ]

    # ========================================
    # FILTRO DE SELL (SHORT)
    # ========================================
    # Dados reais abril: SELL WR 32.4% (N=111) vs BUY WR 24% (N=25)
    # SELL e a direcao principal do sistema; confianca minima 7 (igual BUY)
    sell_min_confidence: int = 7
    sell_require_strong_trend: bool = True

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
    reevaluation_interval_hours: float = 2.0  # Reavaliar a cada 2h (antes 30min — fechava prematuramente)
    reevaluation_min_time_open_hours: float = 1.0  # Primeira reavaliação após 1h (antes 15min)
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
    # DESATIVADO: deep_analysis_report.txt mostra correlação ML<->outcome r=-0.1257
    # (estatisticamente INVERSO, p<0.001). Causa raiz: features risk_distance_pct,
    # reward_distance_pct e risk_reward_ratio têm semântica diferente entre
    # treino (bootstrap em backtest_dataset_generator.py) e inferência. Backtest
    # com ML desligado: PnL +1764% vs PnL +398% com ML ligado.
    # Reativar APENAS após corrigir feature alignment no dataset generator.
    ml_validation_enabled: bool = False
    ml_validation_threshold: float = 0.6  # prob >= 0.6 = voto a favor, < 0.4 = voto contra
    ml_validation_required: bool = False  # ML não tem mais poder de veto

    # ========================================
    # VALIDAÇÃO LSTM SEQUENCE - DESATIVADA
    # ========================================
    # deep_analysis_report.txt mostra correlação LSTM<->outcome r=-0.0255 (p=0.32)
    # — estatisticamente ZERO poder discriminativo. Mesmos issues de data leakage
    # do ML clássico. Reativar APENAS após retreinar com dataset corrigido.
    lstm_validation_enabled: bool = False

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
