"""
Módulo de Indicadores e Padrões Gráficos - CORRIGIDO
Trading Analyzer v2.0 - TODOS OS DETECTORES INTEGRADOS
"""

# Importações dos indicadores técnicos
from .technical import (
    TechnicalAnalyzer,
    RSIAnalyzer, 
    MACDAnalyzer,
    IndicatorResult
)


# ✅ NOVA IMPORTAÇÃO CORRIGIDA: Candlestick Patterns
try:
    from .candlestick_patterns_detector import (
        CandlestickDetector,
        CandlestickPattern,
        generate_candlestick_signals,
        verify_patterns_implementation
    )
    CANDLESTICK_AVAILABLE = True
    print("OK: Detector de Candlestick carregado: 43 padroes disponiveis")
    
    # Verifica se todos os 43 padroes estao implementados
    try:
        implementation_complete = verify_patterns_implementation()
        if implementation_complete:
            print("Target: Implementacao COMPLETA: Todos os 43 padroes de candlestick")
        else:
            print("Warning: Implementacao INCOMPLETA: Alguns padroes podem estar faltando")
    except Exception as verify_error:
        print(f"Warning: Erro na verificacao de implementacao: {verify_error}")
        
except ImportError as e:
    CANDLESTICK_AVAILABLE = False
    print(f"Warning: Detector de Candlestick nao disponivel: {e}")
except Exception as e:
    CANDLESTICK_AVAILABLE = False
    print(f"Error: Erro no detector de Candlestick: {e}")

# Lista base de exportações
__all__ = [
    'TechnicalAnalyzer',
    'RSIAnalyzer',
    'MACDAnalyzer', 
    'IndicatorResult'
]



# ADICIONADO: Candlestick patterns se disponiveis
if CANDLESTICK_AVAILABLE:
    __all__.extend([
        'CandlestickDetector',
        'CandlestickPattern',
        'generate_candlestick_signals',
        'verify_patterns_implementation'
    ])



def get_available_detectors():
    """Retorna lista de detectores OTIMIZADOS"""
    detectors = ['Technical Indicators (RSI, MACD) - Timeframes: 1h, 4h, 1d']
    
    if CANDLESTICK_AVAILABLE:
        detectors.append('Candlestick Patterns (43 patterns - filtrados)')
    
    return detectors


def get_system_status():
    """Status do sistema OTIMIZADO - CORRIGIDO"""
    return {
        'technical_indicators': 'OK - 1h/4h/1d only',
        'candlestick_patterns': 'OK - High confidence only' if CANDLESTICK_AVAILABLE else 'DISABLED',
        'total_pattern_types': sum([
            1,  # Technical sempre disponível
            1 if CANDLESTICK_AVAILABLE else 0  # Candlestick se disponível
        ]),
        'candlestick_patterns_count': 43 if CANDLESTICK_AVAILABLE else 0,
        'disabled_patterns': ['Head&Shoulders', 'Cup&Handle'],
        'optimization': 'Single signal per crypto + 1h priority',
        'anti_hang_protection': 'ACTIVE'
    }

__version__ = "2.0.1"
__author__ = "Trading Analyzer System - COMPLETE EDITION - ANTI-HANG"

# Informações de inicialização CORRIGIDAS
print(f"\nTRADING ANALYZER INDICATORS v{__version__}")
print("=" * 50)
print("Componentes carregados:")
for detector in get_available_detectors():
    print(f"  OK {detector}")

status = get_system_status()
print(f"\nSistema OTIMIZADO:")
print(f"  • Timeframes ativos: 1h, 4h, 1d (preferencia 1h)")
print(f"  • Candlesticks: {status['candlestick_patterns_count']} padroes (filtrados)" if CANDLESTICK_AVAILABLE else "  • Candlesticks: DESABILITADOS")
print(f"  • Otimizacao: {status['optimization']}")
print(f"  • Protecao: {status['anti_hang_protection']}")
if status.get('disabled_patterns'):
    print(f"  • Desabilitados: {', '.join(status['disabled_patterns'])}")
print("=" * 50)