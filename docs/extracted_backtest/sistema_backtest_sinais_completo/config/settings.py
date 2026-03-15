# settings.py - CORRIGIDO PARA APENAS 1h, 4h e 1d

# -*- coding: utf-8 -*-
"""
Configuracoes Multi-Timeframe do Sistema de Trading - VERSÃO FINAL CORRIGIDA
APENAS 1h, 4h e 1d ATIVOS
"""
import os
from dataclasses import dataclass, field
from typing import List, Dict

# _#_NOVO_: Classe para configurar a precisão decimal de cada ativo.
@dataclass
class PrecisionConfig:
    """Configurações de precisão para a exchange."""
    # Mapeia o símbolo para o número de casas decimais do PREÇO.
    # Adicione ou modifique os pares conforme sua necessidade.
    symbol_price_precision: Dict[str, int] = field(default_factory=lambda: {
        'BTC': 2,
        'ETH': 2,
        'BNB': 2,
        'SOL': 2,
        'ENA': 4,
        'HBAR': 5,
        'NEAR': 3,
        'OMNI': 3,
        'SUI': 4,
        'PEPE': 8,
        'TURBO': 6,
        'IMX': 4,
        'CRV': 4,
        'HYPE': 6, # Exemplo, ajuste se necessário
        # Adicione outros símbolos aqui
        'DEFAULT': 4 # Valor padrão para símbolos não listados
    })

@dataclass
class DatabaseConfig:
    """Configuracoes de banco de dados"""
    # Caminhos alternativos para buscar os bancos
    _base_paths: List[str] = field(default_factory=lambda: [
        r"C:\Users\mzmvy\Documents\python\trading_system\data",
        r"C:\Users\mzmvy\Documents\python\bot_trade\sinais\data",
        r".\data",
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    ])
    
    stream_db_path: str = ""
    signals_db_path: str = ""
    stream_table: str = "crypto_ohlc"
    signals_table: str = "trading_signals_v2"
    backup_table: str = "signal_backup_v2"
    
    def __post_init__(self):
        """Detecta automaticamente os bancos de dados"""
        if not self.stream_db_path:
            self.stream_db_path = self._find_db("crypto_stream.db")
        if not self.signals_db_path:
            self.signals_db_path = self._find_db("trading_analyzer_v2.db")
    
    def _find_db(self, db_name: str) -> str:
        """Procura banco de dados em múltiplos caminhos"""
        for base_path in self._base_paths:
            db_path = os.path.join(base_path, db_name)
            if os.path.exists(db_path):
                return os.path.abspath(db_path)
        
        # Se não encontrou, retorna o caminho padrão (será criado se necessário)
        default_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            "data", 
            db_name
        )
        return os.path.abspath(default_path)

@dataclass
class TimeframeConfig:
    """Configuracao para cada timeframe INDIVIDUAL"""
    timeframe: str
    min_data_points: int
    lookback_hours: int
    confidence_threshold: float
    max_signals_per_symbol: int
    analysis_priority: int
    enabled_detectors: List[str] = field(default_factory=lambda: ['technical', 'patterns', 'candlestick'])
    rsi_sensitivity: float = 1.0
    volume_threshold_multiplier: float = 1.0
    pattern_min_strength: float = 0.6

@dataclass
class MultiTimeframeConfig:
    """Configuracao multi-timeframe - APENAS 1h, 4h e 1d"""
    enabled_timeframes: List[str] = None
    timeframe_configs: Dict[str, TimeframeConfig] = None
    allow_conflicting_signals: bool = False
    cross_timeframe_confirmation: bool = True
    hierarchy_priority: bool = True

    def __post_init__(self):
        # CORRIGIDO: Apenas 1h, 4h e 1d habilitados conforme especificação
        if self.enabled_timeframes is None:
            self.enabled_timeframes = ["1h", "4h", "1d"]
        
        if self.timeframe_configs is None:
            self.timeframe_configs = {
                "1h": TimeframeConfig(
                    timeframe="1h", min_data_points=100, lookback_hours=48,
                    confidence_threshold=0.70, max_signals_per_symbol=1, analysis_priority=1,
                    enabled_detectors=['technical', 'candlestick'], rsi_sensitivity=1.0,
                    volume_threshold_multiplier=1.2, pattern_min_strength=0.6
                ),
                "4h": TimeframeConfig(
                    timeframe="4h", min_data_points=50, lookback_hours=192,  # 8 dias
                    confidence_threshold=0.75, max_signals_per_symbol=1, analysis_priority=2,
                    enabled_detectors=['technical', 'candlestick'], rsi_sensitivity=1.0,
                    volume_threshold_multiplier=1.3, pattern_min_strength=0.7
                ),
                "1d": TimeframeConfig(
                    timeframe="1d", min_data_points=30, lookback_hours=720,  # 30 dias
                    confidence_threshold=0.80, max_signals_per_symbol=1, analysis_priority=3,
                    enabled_detectors=['technical', 'candlestick'], rsi_sensitivity=1.0,
                    volume_threshold_multiplier=1.5, pattern_min_strength=0.8
                )
            }

@dataclass
class AnalysisConfig:
    """Configuracoes de analise - OTIMIZADA"""
    multi_timeframe: MultiTimeframeConfig = None
    default_timeframe: str = "1h"  # MUDADO para 1h conforme especificação
    min_data_points: int = 50  # REDUZIDO
    lookback_hours: int = 24
    confidence_threshold: float = 0.70  # REDUZIDO para ser mais permissivo
    symbols: List[str] = None

    def __post_init__(self):
        if self.symbols is None:
            # Usa todos os pares coletados pela Binance
            self.symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT', 'UNIUSDT', 'SOLUSDT', 'MATICUSDT']
        if self.multi_timeframe is None:
            self.multi_timeframe = MultiTimeframeConfig()

@dataclass
class IndicatorConfig:
    """Configuracoes dos indicadores tecnicos - APENAS 1h, 4h e 1d"""
    rsi_period: int = 14
    rsi_overbought: Dict[str, float] = None
    rsi_oversold: Dict[str, float] = None
    macd_fast: Dict[str, int] = None
    macd_slow: Dict[str, int] = None
    macd_signal: Dict[str, int] = None
    volume_ma_period: int = 20
    min_volume_ratio: Dict[str, float] = None

    def __post_init__(self):
        # APENAS 1h, 4h e 1d
        if self.rsi_overbought is None: 
            self.rsi_overbought = {"1h": 70, "4h": 75, "1d": 80}
        if self.rsi_oversold is None: 
            self.rsi_oversold = {"1h": 30, "4h": 25, "1d": 20}
        if self.macd_fast is None: 
            self.macd_fast = {"1h": 12, "4h": 10, "1d": 8}
        if self.macd_slow is None: 
            self.macd_slow = {"1h": 26, "4h": 21, "1d": 17}
        if self.macd_signal is None: 
            self.macd_signal = {"1h": 9, "4h": 8, "1d": 7}
        if self.min_volume_ratio is None: 
            self.min_volume_ratio = {"1h": 1.2, "4h": 1.3, "1d": 1.5}

@dataclass
class PatternConfig:
    """Configuracoes para deteccao de padroes SIMPLIFICADAS"""
    double_tolerance: float = 0.02
    double_min_distance: int = 15
    double_min_significance: float = 0.08
    min_pattern_strength: float = 0.60  # REDUZIDO de 0.65
    max_patterns_per_analysis: int = 2
    # ADICIONADO: Configurações de habilitação
    enable_head_shoulders: bool = False      # DESABILITADO
    enable_cup_handle: bool = False         # DESABILITADO  
    enable_double_patterns: bool = True     # HABILITADO

@dataclass
class SystemConfig:
    """Configuracoes do sistema OTIMIZADAS - APENAS 1h/4h/1d"""
    multi_timeframe_enabled: bool = True
    analysis_interval: int = 180
    backup_all_signals: bool = True
    max_total_signals_per_symbol: int = 1
    log_level: str = "INFO"
    log_file: str = "trading_analyzer_optimized.log"
    parallel_analysis: bool = True
    max_workers: int = 4
    
    # ADICIONADO: Configurações de limpeza automática
    auto_cleanup_enabled: bool = True
    cleanup_interval_hours: int = 24      # ADICIONADO
    signal_lifecycle_hours: int = 48      # ADICIONADO
    
    # NOVO: Sistema de retenção de dados por 4 anos
    data_retention_years: int = 4         # Manter apenas 4 anos de dados
    data_cleanup_interval_hours: int = 24  # Limpeza a cada 24h
    
    live_data_timeframes: List[str] = field(default_factory=lambda: ["1h"])  
    live_data_enabled: bool = True
    
    def get_live_data_timeframes(self) -> List[str]:
        """Retorna timeframes que usam dados live"""
        if self.live_data_enabled:
            return self.live_data_timeframes
        else:
            return []

@dataclass
class ValidationConfig:
    """Configurações para a validação de sinais com microestrutura (Sniper) - CORRIGIDA."""
    
    enabled: bool = True
    # _#_NOVO_: Nome da tabela de microestrutura adicionado aqui.
    microstructure_table: str = "kline_microstructure_1m"
    validation_window_minutes: int = 5  # Quantos minutos olhar à frente na microestrutura.
    search_window_extend_minutes: int = 30  # NOVO: Janela de busca ampliada
    min_data_points_required: int = 3  # NOVO: Mínimo de pontos de dados
    momentum_period: int = 5            # Período para o RSI de momentum na microestrutura.
    buy_momentum_threshold: float = 50.0  # REDUZIDO de 55 para 50 (mais flexível)
    sell_momentum_threshold: float = 50.0 # ALTERADO de 45 para 50 (mais flexível)

@dataclass
class MLConfig:
    """Configurações de Machine Learning"""
    enabled: bool = True
    model_path: str = "data/models/xgboost_model.pkl"
    retrain_interval_days: int = 7
    min_data_points: int = 1000
    prediction_horizon_hours: int = 3  # 3 horas = 36 períodos de 5m
    confidence_threshold: float = 0.65
    ml_weight: float = 0.25  # Peso do ML no score final (25%)

@dataclass  
class LLMConfig:
    """Configurações de LLM Sentiment Analysis"""
    enabled: bool = True  # Habilitado com API key fornecida
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key: str = ""  # Definir via env OPENAI_API_KEY ou .env
    max_tokens: int = 150
    temperature: float = 0.3
    cache_duration_minutes: int = 60
    max_cost_per_month: float = 50.0
    llm_weight: float = 0.20  # Peso do LLM no score final (20%)

@dataclass
class PaperTradingConfig:
    """Configurações de Paper Trading"""
    enabled: bool = True
    initial_capital: float = 10000.0
    max_position_size_pct: float = 0.10  # 10% por posição
    max_open_positions: int = 5
    fee_percentage: float = 0.001  # 0.1% Binance
    slippage_percentage: float = 0.0005  # 0.05%

class Settings:
    """Classe principal de configuracoes - CORRIGIDA PARA 1h/4h/1d APENAS + NOVAS FEATURES"""
    def __init__(self):
        self.database = DatabaseConfig()
        self.analysis = AnalysisConfig()
        self.indicators = IndicatorConfig()
        self.patterns = PatternConfig()
        self.system = SystemConfig()
        self.precisions = PrecisionConfig()
        self.validation = ValidationConfig()
        # 🔧 CORREÇÃO: Linha comentada para evitar erro
        # self.candlestick = CandlestickConfig()  # COMENTADO - classe não definida
        
        # 🆕 NOVAS CONFIGURAÇÕES
        self.ml = MLConfig()
        self.llm = LLMConfig()
        self.paper_trading = PaperTradingConfig()

    def get_timeframe_config(self, timeframe: str) -> TimeframeConfig:
        """CORRIGIDO: Fallback para 1h se timeframe não encontrado"""
        valid_timeframes = ["1h", "4h", "1d"]
        if timeframe not in valid_timeframes:
            timeframe = "1h"  # Fallback para 1h
        return self.analysis.multi_timeframe.timeframe_configs.get(timeframe, self.analysis.multi_timeframe.timeframe_configs["1h"])

    def get_enabled_timeframes(self) -> List[str]:
        """GARANTIDO: Retorna apenas 1h, 4h e 1d"""
        enabled = ["1h", "4h", "1d"]  # HARDCODED para evitar problemas
        if self.system.multi_timeframe_enabled:
            return enabled
        else:
            return [self.analysis.default_timeframe]  # "1h"

    def get_rsi_levels(self, timeframe: str) -> Dict[str, float]:
        """CORRIGIDO: Fallback para timeframes válidos"""
        if timeframe not in ["1h", "4h", "1d"]:
            timeframe = "1h"
        return {'overbought': self.indicators.rsi_overbought.get(timeframe, 70), 'oversold': self.indicators.rsi_oversold.get(timeframe, 30)}

    def get_macd_params(self, timeframe: str) -> Dict[str, int]:
        """CORRIGIDO: Fallback para timeframes válidos"""
        if timeframe not in ["1h", "4h", "1d"]:
            timeframe = "1h"
        return {'fast': self.indicators.macd_fast.get(timeframe, 12), 'slow': self.indicators.macd_slow.get(timeframe, 26), 'signal': self.indicators.macd_signal.get(timeframe, 9)}

    def get_analysis_symbols(self) -> List[str]:
        return self.analysis.symbols
    
    def get_price_precision(self, symbol: str) -> int:
        return self.precisions.symbol_price_precision.get(symbol, self.precisions.symbol_price_precision['DEFAULT'])

    # NOVO: Configurações para stop loss técnico (compatibilidade)
    def get_stop_target_config(self, timeframe: str) -> Dict:
        """Configurações para cálculo de stop loss e targets técnicos"""
        configs = {
            "5m": {
                "atr_period": 14,
                "stop_atr_multiplier": 1.8,
                "min_atr_mult": 1.2,
                "max_atr_mult": 2.5,
                "target_1_ratio": 1.5,
                "target_2_ratio": 3.0
            },
            "15m": {
                "atr_period": 14,
                "stop_atr_multiplier": 2.2,
                "min_atr_mult": 1.5,
                "max_atr_mult": 3.0,
                "target_1_ratio": 1.8,
                "target_2_ratio": 3.5
            }
        }
        
        if timeframe not in configs:
            timeframe = "5m"  # Fallback
        
        return configs[timeframe]

# _#_CORRIGIDO_: Instância única do settings
settings = Settings()