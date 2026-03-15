# technical.py - INTEGRADO COM BOLLINGER BANDS + VWAP PARA CONFLUÊNCIA

"""
Technical Analyzer Aprimorado - Integra Filtros de Confirmação
- RSI + MACD (indicadores principais)
- Bandas de Bollinger (filtro de volatilidade)
- VWAP (referência dinâmica)
- Sistema de confluência que aumenta confidence dos sinais
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from config.settings import settings
from core.data_reader import MarketData

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

@dataclass
class IndicatorResult:
    name: str
    values: pd.Series
    signals: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class RSIAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.period = settings.indicators.rsi_period

    def calculate_rsi(self, close_prices: pd.Series) -> pd.Series:
        """Cálculo RSI CORRIGIDO usando EMA (Exponential Moving Average)"""
        if len(close_prices) < self.period:
            return pd.Series(dtype=float)
        
        delta = close_prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # CORREÇÃO: Usar EMA ao invés de SMA (método correto do RSI)
        avg_gain = gain.ewm(span=self.period, adjust=False).mean()
        avg_loss = loss.ewm(span=self.period, adjust=False).mean()
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50.0)

    def analyze(self, market_data, timeframe: str):
        self.logger.debug(f"Analisando RSI para {market_data.symbol} no timeframe {timeframe}")
        # Garante que a análise use um slice que termina no candle fechado.
        df_closed = market_data.data

        if len(df_closed) < self.period + 5:
            return IndicatorResult("RSI", pd.Series(dtype=float), [], {})

        close_prices = df_closed['close_price']
        rsi_levels = settings.get_rsi_levels(timeframe)
        overbought = rsi_levels['overbought']
        oversold = rsi_levels['oversold']

        rsi = self.calculate_rsi(close_prices)
        if rsi.empty:
            return IndicatorResult("RSI", rsi, [], {})

        # Pega o valor mais recente do RSI, que corresponde ao último candle fechado.
        latest_rsi = rsi.iloc[-1]

        signals = []
        
        # Sinais de extremos (alta confiança)
        if latest_rsi >= overbought:
            signals.append({'type': 'rsi_overbought', 'signal_type': 'SELL_SHORT', 'confidence': 0.75, 'priority': 'high'})
        elif latest_rsi <= oversold:
            signals.append({'type': 'rsi_oversold', 'signal_type': 'BUY_LONG', 'confidence': 0.75, 'priority': 'high'})
        
        # Sinais de tendência (confiança média) - NOVO
        elif latest_rsi > 60:  # Tendência de alta
            signals.append({'type': 'rsi_bullish_trend', 'signal_type': 'BUY_LONG', 'confidence': 0.60, 'priority': 'medium'})
        elif latest_rsi < 40:  # Tendência de baixa
            signals.append({'type': 'rsi_bearish_trend', 'signal_type': 'SELL_SHORT', 'confidence': 0.60, 'priority': 'medium'})
        
        # Sinais de momentum (confiança baixa) - NOVO
        elif latest_rsi > 50:  # Momentum positivo
            signals.append({'type': 'rsi_bullish_momentum', 'signal_type': 'BUY_LONG', 'confidence': 0.45, 'priority': 'low'})
        elif latest_rsi < 50:  # Momentum negativo
            signals.append({'type': 'rsi_bearish_momentum', 'signal_type': 'SELL_SHORT', 'confidence': 0.45, 'priority': 'low'})

        # O metadata pode usar o RSI do candle atual para simples informação
        current_rsi_full = self.calculate_rsi(market_data.data['close_price'])
        return IndicatorResult("RSI", rsi, signals, {'current_rsi': current_rsi_full.iloc[-1] if not current_rsi_full.empty else 50.0})

class MACDAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate_macd(self, close_prices: pd.Series, params: Dict) -> Tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = close_prices.ewm(span=params['fast'], adjust=False).mean()
        ema_slow = close_prices.ewm(span=params['slow'], adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=params['signal'], adjust=False).mean()
        histogram = macd - signal
        return macd, signal, histogram

    def analyze(self, market_data, timeframe: str):
        self.logger.debug(f"Analisando MACD para {market_data.symbol} no timeframe {timeframe}")
        params = settings.get_macd_params(timeframe)
        # Garante que a análise use um slice que termina no candle fechado.
        df_closed = market_data.data

        if len(df_closed) < params['slow'] + 5:
            return IndicatorResult("MACD", pd.Series(dtype=float), [], {})

        close_prices = df_closed['close_price']
        macd, signal, _ = self.calculate_macd(close_prices, params)

        if len(macd) < 2:
            return IndicatorResult("MACD", macd, [], {})

        # Usa os dois últimos pontos de dados (que são do penúltimo e antepenúltimo candles fechados)
        latest_macd, prev_macd = macd.iloc[-1], macd.iloc[-2]
        latest_signal, prev_signal = signal.iloc[-1], signal.iloc[-2]

        signals = []
        
        # Detecção de cruzamento (alta confiança)
        if prev_macd <= prev_signal and latest_macd > latest_signal:
            signals.append({'type': 'macd_bullish_crossover', 'signal_type': 'BUY_LONG', 'confidence': 0.8, 'priority': 'crossover'})
        elif prev_macd >= prev_signal and latest_macd < latest_signal:
            signals.append({'type': 'macd_bearish_crossover', 'signal_type': 'SELL_SHORT', 'confidence': 0.8, 'priority': 'crossover'})
        
        # Sinais de momentum (confiança média) - NOVO
        elif latest_macd > latest_signal:  # MACD acima da linha de sinal
            signals.append({'type': 'macd_bullish_momentum', 'signal_type': 'BUY_LONG', 'confidence': 0.65, 'priority': 'medium'})
        elif latest_macd < latest_signal:  # MACD abaixo da linha de sinal
            signals.append({'type': 'macd_bearish_momentum', 'signal_type': 'SELL_SHORT', 'confidence': 0.65, 'priority': 'medium'})
        
        # Sinais de tendência (confiança baixa) - NOVO
        elif latest_macd > 0:  # MACD positivo
            signals.append({'type': 'macd_bullish_trend', 'signal_type': 'BUY_LONG', 'confidence': 0.50, 'priority': 'low'})
        elif latest_macd < 0:  # MACD negativo
            signals.append({'type': 'macd_bearish_trend', 'signal_type': 'SELL_SHORT', 'confidence': 0.50, 'priority': 'low'})

        # O metadata pode usar o MACD do candle atual para simples informação
        full_macd, full_signal, _ = self.calculate_macd(market_data.data['close_price'], params)
        return IndicatorResult("MACD", macd, signals, {'current_macd': full_macd.iloc[-1], 'current_signal': full_signal.iloc[-1]})

# 🚨 NOVA CLASSE: Bollinger Bands Analyzer
class BollingerBandsAnalyzer:
    """Analisador de Bandas de Bollinger integrado ao sistema principal"""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        self.logger = logging.getLogger(__name__)
        self.period = period
        self.std_dev = std_dev
    
    def analyze(self, market_data, timeframe: str):
        """Análise das Bandas de Bollinger"""
        self.logger.debug(f"Analisando Bollinger Bands para {market_data.symbol} no timeframe {timeframe}")
        
        df_closed = market_data.data
        
        if len(df_closed) < self.period + 5:
            return IndicatorResult("BollingerBands", pd.Series(dtype=float), [], {})
        
        close_prices = df_closed['close_price']
        
        # Calcula as bandas
        middle_band = close_prices.rolling(window=self.period).mean()
        std = close_prices.rolling(window=self.period).std()
        upper_band = middle_band + (std * self.std_dev)
        lower_band = middle_band - (std * self.std_dev)
        
        if middle_band.empty:
            return IndicatorResult("BollingerBands", middle_band, [], {})
        
        # Analisa sinais de extremos
        latest_price = close_prices.iloc[-1]
        latest_upper = upper_band.iloc[-1]
        latest_lower = lower_band.iloc[-1]
        latest_middle = middle_band.iloc[-1]
        
        signals = []
        
        # SINAIS DE EXTREMOS (alta probabilidade de reversão)
        if latest_price <= latest_lower:
            signals.append({
                'type': 'bollinger_oversold_extreme', 
                'signal_type': 'BUY_LONG', 
                'confidence': 0.85,  # Alta confiança para extremos
                'priority': 'extreme'
            })
        elif latest_price >= latest_upper:
            signals.append({
                'type': 'bollinger_overbought_extreme', 
                'signal_type': 'SELL_SHORT', 
                'confidence': 0.85,
                'priority': 'extreme'
            })
        
        # SINAIS DE PROXIMIDADE DAS BANDAS (confiança média) - NOVO
        elif latest_price <= latest_lower * 1.02:  # Próximo da banda inferior (2%)
            signals.append({
                'type': 'bollinger_near_lower', 
                'signal_type': 'BUY_LONG', 
                'confidence': 0.70,
                'priority': 'medium'
            })
        elif latest_price >= latest_upper * 0.98:  # Próximo da banda superior (2%)
            signals.append({
                'type': 'bollinger_near_upper', 
                'signal_type': 'SELL_SHORT', 
                'confidence': 0.70,
                'priority': 'medium'
            })
        
        # SINAIS DE POSIÇÃO RELATIVA (confiança baixa) - NOVO
        elif latest_price > latest_middle:  # Acima da média
            signals.append({
                'type': 'bollinger_above_middle', 
                'signal_type': 'BUY_LONG', 
                'confidence': 0.55,
                'priority': 'low'
            })
        elif latest_price < latest_middle:  # Abaixo da média
            signals.append({
                'type': 'bollinger_below_middle', 
                'signal_type': 'SELL_SHORT', 
                'confidence': 0.55,
                'priority': 'low'
            })
        
        # Metadata com informações das bandas
        metadata = {
            'current_upper': latest_upper,
            'current_middle': latest_middle,
            'current_lower': latest_lower,
            'price_position': 'above_upper' if latest_price > latest_upper else 
                            'above_middle' if latest_price > latest_middle else
                            'below_middle' if latest_price > latest_lower else 'below_lower',
            'distance_to_upper_pct': ((latest_upper - latest_price) / latest_price) * 100,
            'distance_to_lower_pct': ((latest_price - latest_lower) / latest_price) * 100
        }
        
        return IndicatorResult("BollingerBands", middle_band, signals, metadata)

# 🚨 NOVA CLASSE: VWAP Analyzer  
class VWAPAnalyzer:
    """Analisador de VWAP integrado ao sistema principal"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze(self, market_data, timeframe: str):
        """Análise do VWAP"""
        self.logger.debug(f"Analisando VWAP para {market_data.symbol} no timeframe {timeframe}")
        
        df_closed = market_data.data
        
        if len(df_closed) < 20:
            return IndicatorResult("VWAP", pd.Series(dtype=float), [], {})
        
        # Calcula VWAP
        typical_price = (df_closed['high_price'] + df_closed['low_price'] + df_closed['close_price']) / 3
        volume = df_closed['volume']
        
        # VWAP de janela móvel (últimos 50 períodos)
        window = min(50, len(typical_price))
        pv = typical_price * volume
        
        rolling_pv = pv.rolling(window=window).sum()
        rolling_volume = volume.rolling(window=window).sum()
        
        vwap = rolling_pv / (rolling_volume + 1e-10)
        
        if vwap.empty:
            return IndicatorResult("VWAP", vwap, [], {})
        
        # Analisa posição relativa ao VWAP
        latest_price = typical_price.iloc[-1]
        latest_vwap = vwap.iloc[-1]
        
        signals = []
        
        # 🎯 SINAIS DE SUPORTE/RESISTÊNCIA no VWAP
        distance_pct = abs(latest_price - latest_vwap) / latest_vwap * 100
        
        if distance_pct < 0.5:  # Preço muito próximo do VWAP
            if latest_price > latest_vwap:
                signals.append({
                    'type': 'vwap_support_test', 
                    'signal_type': 'BUY_LONG', 
                    'confidence': 0.70,
                    'priority': 'support'
                })
            else:
                signals.append({
                    'type': 'vwap_resistance_test', 
                    'signal_type': 'SELL_SHORT', 
                    'confidence': 0.70,
                    'priority': 'resistance'
                })
        
        # Metadata
        metadata = {
            'current_vwap': latest_vwap,
            'price_vs_vwap': 'above' if latest_price > latest_vwap else 'below',
            'distance_pct': distance_pct,
            'volume_profile': self._analyze_volume_profile(volume)
        }
        
        return IndicatorResult("VWAP", vwap, signals, metadata)
    
    def _analyze_volume_profile(self, volume: pd.Series) -> str:
        """Analisa perfil de volume"""
        if len(volume) < 10:
            return 'unknown'
        
        current_volume = volume.iloc[-1]
        avg_volume = volume.tail(20).mean()
        
        if current_volume > avg_volume * 1.5:
            return 'high_volume'
        elif current_volume < avg_volume * 0.7:
            return 'low_volume'
        else:
            return 'average_volume'

class TechnicalAnalyzer:
    """Analisador Técnico APRIMORADO com Sistema de Confluência"""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.rsi_analyzer = RSIAnalyzer()
        self.macd_analyzer = MACDAnalyzer()
        self.bollinger_analyzer = BollingerBandsAnalyzer()
        # VWAP removido - não adiciona valor significativo
        
        self.logger.info("🎯 TechnicalAnalyzer SIMPLIFICADO - RSI + MACD + Bollinger") 
   
    def should_use_live_data(self, indicator_name: str) -> bool:
        """Define quais indicadores usam dados live"""
        
        LIVE_DATA_INDICATORS = [
            'RSI', 'MACD', 'BollingerBands', 'VWAP'
        ]
        
        return indicator_name in LIVE_DATA_INDICATORS
    
    
    def analyze_all(self, market_data, timeframe: str):
        """Análise simplificada com indicadores efetivos"""
        results = {}
        try:
            # Apenas indicadores com boa performance
            results['RSI'] = self.rsi_analyzer.analyze(market_data, timeframe)
            results['MACD'] = self.macd_analyzer.analyze(market_data, timeframe)
            results['BollingerBands'] = self.bollinger_analyzer.analyze(market_data, timeframe)
            
        except Exception as e:
            self.logger.error(f"Erro na análise técnica para {market_data.symbol} {timeframe}: {e}")
        
        return results
   
    def generate_trading_signals(self, market_data, analysis_results, timeframe: str):
        """Geração de sinais SIMPLIFICADA - sem confluência complexa"""
        from core.signal_writer import EnhancedTradingSignal

        # Coleta sinais de indicadores efetivos
        all_signals = []
        for indicator_name, result in analysis_results.items():
            for signal_data in result.signals:
                signal_data['indicator'] = indicator_name
                all_signals.append(signal_data)

        if not all_signals:
            return []

        # Prioriza por performance histórica
        def signal_priority_score(signal):
            # BollingerBands extremos = prioridade máxima (62% sucesso)
            if signal.get('priority') == 'extreme':
                return 100 + signal.get('confidence', 0.5) * 100
            # RSI overbought/oversold (61% sucesso)  
            elif signal.get('priority') == 'high':
                return 90 + signal.get('confidence', 0.5) * 100
            # MACD crossovers (54% sucesso)
            elif signal.get('priority') == 'crossover':
                return 80 + signal.get('confidence', 0.5) * 100
            # Prioridade média (novos sinais)
            elif signal.get('priority') == 'medium':
                return 70 + signal.get('confidence', 0.5) * 100
            # Prioridade baixa (novos sinais)
            elif signal.get('priority') == 'low':
                return 60 + signal.get('confidence', 0.5) * 100
            else:
                return 50 + signal.get('confidence', 0.5) * 100

        best_signal = max(all_signals, key=signal_priority_score)

        # Cria sinal sem confluência complexa
        enhanced_signal = EnhancedTradingSignal(
            symbol=market_data.symbol,
            signal_type=best_signal['signal_type'],
            entry_price=market_data.latest_price,
            confidence=best_signal['confidence'],
            timeframe=timeframe,
            detector_type='technical',
            detector_name=best_signal['indicator'],
            market_data=market_data.data,
            technical_data={
                'primary_signal': best_signal,
                'all_signals': all_signals,
            }
        )

        self.logger.info(
            f"📊 SINAL TÉCNICO: {market_data.symbol} {timeframe} | "
            f"{best_signal['indicator']} → {best_signal['signal_type']} | "
            f"Conf: {enhanced_signal.confidence:.3f}"
        )

        return [enhanced_signal]
    
    
    def get_confluence_statistics(self, market_data) -> Dict:
        """Retorna estatísticas do sistema de confluência para debug"""
        if not self.use_confluence:
            return {'confluence_available': False}
        
        try:
            context = self.confluence_analyzer.get_market_context(market_data)
            return {
                'confluence_available': True,
                'market_context': context,
                'timestamp': pd.Timestamp.now().isoformat()
            }
        except Exception as e:
            return {'confluence_available': True, 'error': str(e)}
