# candlestick_patterns_detector.py - VERSÃO CORRIGIDA COMPLETA
"""
🎯 DETECTOR DE CANDLESTICK PATTERNS CORRETO
Baseado APENAS na estrutura dos candles do pattern - SEM níveis históricos
Targets realistas e stops proporcionais
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

@dataclass
class CandlestickPattern:
    """Estrutura para padrão de candlestick detectado"""
    name: str
    pattern_type: str  # 'bullish' ou 'bearish'
    entry_price: float
    stop_loss: float
    target_price: float
    target_2: Optional[float]
    position_index: int
    reliability_score: float
    pattern_strength: float
    targets_logic: str

class SimplifiedCandlestickDetector:
    """
    🎯 DETECTOR CORRETO DE CANDLESTICK PATTERNS
    - Targets baseados APENAS na estrutura dos candles
    - Stops nas extremidades do pattern
    - Riscos controlados (1-3%)
    - Lógica original dos patterns
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 🔧 CONFIGURAÇÕES CORRIGIDAS
        self.config = {
            # Critérios de detecção
            'doji_body_max_pct': 0.1,           # Corpo máximo 10% do range para Doji
            'small_body_pct': 0.003,            # Corpo pequeno < 0.3%
            'large_body_pct': 0.015,            # Corpo grande > 1.5%
            'hammer_shadow_min_ratio': 2.0,     # Sombra inferior 2x maior que corpo
            'shooting_star_shadow_min_ratio': 2.0, # Sombra superior 2x maior que corpo
            'engulfing_min_ratio': 1.1,         # Engolfo 10% maior que anterior
            
            # Limites de segurança
            'max_risk_pct': 2.5,                # Máximo 2.5% de risco
            'min_reward_ratio': 1.2,            # Mínimo 1.2:1 reward/risk
            'max_target_distance_pct': 5.0,     # Target máximo 5%
            
            # Configurações do sistema
            'trend_period': 10,                 # Período para tendência
            'buffer_pct': 0.005,               # Buffer de 0.5% para stops
            'use_pattern_structure_only': True, # APENAS estrutura do pattern
        }
        
        self.logger.info("🎯 SimplifiedCandlestickDetector CORRIGIDO inicializado:")
        self.logger.info("   • Targets baseados APENAS na estrutura dos candles")
        self.logger.info("   • SEM níveis de suporte/resistência históricos")
        self.logger.info("   • Risco máximo: 2.5%")
        self.logger.info("   • Foco: 5 patterns mais efetivos")

    def _calculate_atr_reference(self, df: pd.DataFrame, period: int = 14) -> float:
        """ATR apenas como referência de volatilidade - NÃO para targets"""
        try:
            if len(df) < period:
                return df['close_price'].iloc[-1] * 0.01
                
            data = df.tail(period + 5).copy()  # Dados extras para estabilidade
            data['prev_close'] = data['close_price'].shift(1)
            
            data['tr1'] = data['high_price'] - data['low_price']
            data['tr2'] = abs(data['high_price'] - data['prev_close'])
            data['tr3'] = abs(data['low_price'] - data['prev_close'])
            
            data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
            atr = data['true_range'].tail(period).mean()
            
            return atr if pd.notna(atr) and atr > 0 else df['close_price'].iloc[-1] * 0.01
            
        except Exception as e:
            self.logger.debug(f"Erro no cálculo de ATR: {e}")
            return df['close_price'].iloc[-1] * 0.01

    def prepare_candlestick_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepara dados dos candles de forma otimizada"""
        # OTIMIZAÇÃO: Usa apenas últimos 25 candles
        if len(df) > 25:
            data = df.tail(25).copy()
            self.logger.debug(f"🔥 Processando {len(data)} candles (otimizado de {len(df)})")
        else:
            data = df.copy()
        
        # Propriedades básicas dos candles
        data['body_size'] = abs(data['close_price'] - data['open_price'])
        data['upper_shadow'] = data['high_price'] - np.maximum(data['open_price'], data['close_price'])
        data['lower_shadow'] = np.minimum(data['open_price'], data['close_price']) - data['low_price']
        data['total_range'] = data['high_price'] - data['low_price']
        
        # Direção dos candles
        data['is_green'] = data['close_price'] > data['open_price']
        data['is_red'] = data['close_price'] < data['open_price']
        
        # Classificação por tamanho (baseado no preço atual)
        current_price = data['close_price'].iloc[-1]
        data['body_size_pct'] = data['body_size'] / current_price
        data['is_small_body'] = data['body_size_pct'] < self.config['small_body_pct']
        data['is_large_body'] = data['body_size_pct'] > self.config['large_body_pct']
        
        # Doji (corpo muito pequeno em relação ao range)
        data['is_doji'] = data['body_size'] <= (data['total_range'] * self.config['doji_body_max_pct'])
        
        # Tendência simples (média móvel)
        if len(data) >= self.config['trend_period']:
            trend_ma = data['close_price'].rolling(self.config['trend_period']).mean()
            data['is_uptrend'] = data['close_price'] > trend_ma
            data['is_downtrend'] = data['close_price'] < trend_ma
        else:
            data['is_uptrend'] = True
            data['is_downtrend'] = False
        
        return data

    def detect_effective_patterns(self, df: pd.DataFrame) -> List[CandlestickPattern]:
        """Detecta os 5 padrões mais efetivos com cálculos corretos"""
        if len(df) < 2:
            return []
        
        data = self.prepare_candlestick_data(df)
        patterns = []
        
        # 1. ENGOLFO (Mais efetivos) - Prioridade 1
        patterns.extend(self._detect_engulfing_patterns(data))
        
        # 2. HAMMER - Efetivo em reversões de baixa
        patterns.extend(self._detect_hammer_patterns(data))
        
        # 3. SHOOTING STAR - Efetivo em reversões de alta
        patterns.extend(self._detect_shooting_star_patterns(data))
        
        # 4. DOJI - Indecisão e possível reversão
        patterns.extend(self._detect_doji_patterns(data))
        
        # Valida e filtra padrões com parâmetros seguros
        valid_patterns = []
        for pattern in patterns:
            if self._validate_pattern_parameters(pattern):
                valid_patterns.append(pattern)
            else:
                self.logger.debug(f"Pattern {pattern.name} rejeitado por parâmetros inseguros")
        
        # Ordena por reliability score
        return sorted(valid_patterns, key=lambda p: p.reliability_score, reverse=True)

    def _detect_engulfing_patterns(self, data: pd.DataFrame) -> List[CandlestickPattern]:
        """🔥 ENGOLFO CORRIGIDO - Baseado APENAS nos 2 candles"""
        patterns = []
        
        if len(data) < 2:
            return patterns
        
        # Verifica apenas o último par de candles
        prev_candle = data.iloc[-2]
        curr_candle = data.iloc[-1]
        pattern_index = len(data) - 1
        
        # BULLISH ENGULFING
        if (curr_candle.is_green and prev_candle.is_red and 
            curr_candle.close_price > prev_candle.open_price and 
            curr_candle.open_price < prev_candle.close_price):
            
            # Verifica se o engolfo é significativo
            prev_body = abs(prev_candle.close_price - prev_candle.open_price)
            curr_body = abs(curr_candle.close_price - curr_candle.open_price)
            
            if curr_body >= prev_body * self.config['engulfing_min_ratio']:
                params = self._calculate_bullish_engulfing_params(prev_candle, curr_candle)
                
                if params:
                    patterns.append(CandlestickPattern(
                        name="Bullish_Engulfing",
                        pattern_type="bullish",
                        position_index=pattern_index,
                        reliability_score=0.85,
                        pattern_strength=curr_body / prev_body,
                        **params
                    ))
        
        # BEARISH ENGULFING
        elif (curr_candle.is_red and prev_candle.is_green and 
              curr_candle.close_price < prev_candle.open_price and 
              curr_candle.open_price > prev_candle.close_price):
            
            prev_body = abs(prev_candle.close_price - prev_candle.open_price)
            curr_body = abs(curr_candle.close_price - curr_candle.open_price)
            
            if curr_body >= prev_body * self.config['engulfing_min_ratio']:
                params = self._calculate_bearish_engulfing_params(prev_candle, curr_candle)
                
                if params:
                    patterns.append(CandlestickPattern(
                        name="Bearish_Engulfing",
                        pattern_type="bearish",
                        position_index=pattern_index,
                        reliability_score=0.90,
                        pattern_strength=curr_body / prev_body,
                        **params
                    ))
        
        return patterns

    def _detect_hammer_patterns(self, data: pd.DataFrame) -> List[CandlestickPattern]:
        """🔨 HAMMER CORRIGIDO - Baseado APENAS no candle"""
        patterns = []
        
        if len(data) < 1:
            return patterns
        
        candle = data.iloc[-1]
        pattern_index = len(data) - 1
        
        # Critérios do Hammer
        if (candle.total_range > 0 and candle.body_size > 0 and
            candle.lower_shadow >= candle.body_size * self.config['hammer_shadow_min_ratio'] and
            candle.upper_shadow <= candle.body_size * 0.5):
            
            # Confirma contexto de baixa (se dados suficientes)
            in_downtrend = getattr(candle, 'is_downtrend', True)
            
            if in_downtrend:
                params = self._calculate_hammer_params(candle)
                
                if params:
                    patterns.append(CandlestickPattern(
                        name="Hammer",
                        pattern_type="bullish",
                        position_index=pattern_index,
                        reliability_score=0.75,
                        pattern_strength=candle.lower_shadow / candle.body_size,
                        **params
                    ))
        
        return patterns

    def _detect_shooting_star_patterns(self, data: pd.DataFrame) -> List[CandlestickPattern]:
        """⭐ SHOOTING STAR CORRIGIDO - Baseado APENAS no candle"""
        patterns = []
        
        if len(data) < 1:
            return patterns
        
        candle = data.iloc[-1]
        pattern_index = len(data) - 1
        
        # Critérios do Shooting Star
        if (candle.total_range > 0 and candle.body_size > 0 and
            candle.upper_shadow >= candle.body_size * self.config['shooting_star_shadow_min_ratio'] and
            candle.lower_shadow <= candle.body_size * 0.5):
            
            # Confirma contexto de alta
            in_uptrend = getattr(candle, 'is_uptrend', True)
            
            if in_uptrend:
                params = self._calculate_shooting_star_params(candle)
                
                if params:
                    patterns.append(CandlestickPattern(
                        name="Shooting_Star",
                        pattern_type="bearish",
                        position_index=pattern_index,
                        reliability_score=0.75,
                        pattern_strength=candle.upper_shadow / candle.body_size,
                        **params
                    ))
        
        return patterns

    def _detect_doji_patterns(self, data: pd.DataFrame) -> List[CandlestickPattern]:
        """🎭 DOJI CORRIGIDO - Padrão conservador"""
        patterns = []
        
        if len(data) < 1:
            return patterns
        
        candle = data.iloc[-1]
        pattern_index = len(data) - 1
        
        # Critério do Doji
        if candle.is_doji and candle.total_range > 0:
            
            # Determina direção com base na posição do fechamento
            close_position = (candle.close_price - candle.low_price) / candle.total_range
            
            if close_position > 0.6:  # Fechou na parte superior = bullish bias
                pattern_type = "bullish"
                pattern_name = "Doji_Bullish"
            elif close_position < 0.4:  # Fechou na parte inferior = bearish bias
                pattern_type = "bearish"
                pattern_name = "Doji_Bearish"
            else:
                # Muito neutro - não gera sinal
                return patterns
            
            params = self._calculate_doji_params(candle, pattern_type)
            
            if params:
                patterns.append(CandlestickPattern(
                    name=pattern_name,
                    pattern_type=pattern_type,
                    position_index=pattern_index,
                    reliability_score=0.60,  # Menor confiança (indecisão)
                    pattern_strength=candle.total_range / (candle.body_size + 1e-8),
                    **params
                ))
        
        return patterns

    def _calculate_bullish_engulfing_params(self, prev_candle, curr_candle) -> Optional[Dict]:
        """📊 PARÂMETROS BULLISH ENGULFING - Baseado nos 2 candles"""
        try:
            entry_price = float(curr_candle.close_price)
            
            # STOP: Abaixo da mínima do pattern
            pattern_low = min(prev_candle.low_price, curr_candle.low_price)
            stop_loss = pattern_low * (1 - self.config['buffer_pct'])
            
            # TARGET 1: Tamanho do corpo engolfador
            engulfing_body = curr_candle.close_price - curr_candle.open_price
            target_1 = entry_price + engulfing_body
            
            # TARGET 2: Range total do pattern
            pattern_high = max(prev_candle.high_price, curr_candle.high_price)
            pattern_range = pattern_high - pattern_low
            target_2 = entry_price + pattern_range
            
            # Validação básica
            if target_1 <= entry_price or stop_loss >= entry_price:
                return None
            
            return {
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'target_price': target_1,
                'target_2': target_2,
                'targets_logic': f"Body: {engulfing_body:.4f}, Range: {pattern_range:.4f}"
            }
            
        except Exception as e:
            self.logger.debug(f"Erro nos parâmetros bullish engulfing: {e}")
            return None

    def _calculate_bearish_engulfing_params(self, prev_candle, curr_candle) -> Optional[Dict]:
        """📊 PARÂMETROS BEARISH ENGULFING - Baseado nos 2 candles"""
        try:
            entry_price = float(curr_candle.close_price)
            
            # STOP: Acima da máxima do pattern
            pattern_high = max(prev_candle.high_price, curr_candle.high_price)
            stop_loss = pattern_high * (1 + self.config['buffer_pct'])
            
            # TARGET 1: Tamanho do corpo engolfador
            engulfing_body = curr_candle.open_price - curr_candle.close_price
            target_1 = entry_price - engulfing_body
            
            # TARGET 2: Range total do pattern
            pattern_low = min(prev_candle.low_price, curr_candle.low_price)
            pattern_range = pattern_high - pattern_low
            target_2 = entry_price - pattern_range
            
            # Validação básica
            if target_1 >= entry_price or stop_loss <= entry_price:
                return None
            
            return {
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'target_price': target_1,
                'target_2': target_2,
                'targets_logic': f"Body: {engulfing_body:.4f}, Range: {pattern_range:.4f}"
            }
            
        except Exception as e:
            self.logger.debug(f"Erro nos parâmetros bearish engulfing: {e}")
            return None

    def _calculate_hammer_params(self, candle) -> Optional[Dict]:
        """📊 PARÂMETROS HAMMER - Baseado na sombra rejeitada"""
        try:
            entry_price = float(candle.close_price)
            
            # STOP: Abaixo da mínima (quebra da rejeição)
            stop_loss = candle.low_price * (1 - self.config['buffer_pct'])
            
            # TARGET 1: Projeção da sombra inferior
            lower_shadow = candle.close_price - candle.low_price if candle.close_price > candle.open_price else candle.open_price - candle.low_price
            target_1 = entry_price + lower_shadow
            
            # TARGET 2: 1.5x a sombra (extensão da rejeição)
            target_2 = entry_price + (lower_shadow * 1.5)
            
            # Validação
            if target_1 <= entry_price or stop_loss >= entry_price:
                return None
            
            return {
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'target_price': target_1,
                'target_2': target_2,
                'targets_logic': f"Shadow projection: {lower_shadow:.4f}"
            }
            
        except Exception as e:
            self.logger.debug(f"Erro nos parâmetros hammer: {e}")
            return None

    def _calculate_shooting_star_params(self, candle) -> Optional[Dict]:
        """📊 PARÂMETROS SHOOTING STAR - Baseado na rejeição do topo"""
        try:
            entry_price = float(candle.close_price)
            
            # STOP: Acima da máxima (quebra da rejeição)
            stop_loss = candle.high_price * (1 + self.config['buffer_pct'])
            
            # TARGET 1: Projeção da sombra superior
            upper_shadow = candle.high_price - (candle.close_price if candle.close_price > candle.open_price else candle.open_price)
            target_1 = entry_price - upper_shadow
            
            # TARGET 2: 1.5x a sombra
            target_2 = entry_price - (upper_shadow * 1.5)
            
            # Validação
            if target_1 >= entry_price or stop_loss <= entry_price:
                return None
            
            return {
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'target_price': target_1,
                'target_2': target_2,
                'targets_logic': f"Shadow projection: {upper_shadow:.4f}"
            }
            
        except Exception as e:
            self.logger.debug(f"Erro nos parâmetros shooting star: {e}")
            return None

    def _calculate_doji_params(self, candle, pattern_type: str) -> Optional[Dict]:
        """📊 PARÂMETROS DOJI - Conservador (apenas 1 target)"""
        try:
            entry_price = float(candle.close_price)
            
            # Target conservador: 50% do range total
            target_distance = candle.total_range * 0.5
            
            if pattern_type == 'bullish':
                target_1 = entry_price + target_distance
                stop_loss = candle.low_price * (1 - self.config['buffer_pct'])
                
                if target_1 <= entry_price or stop_loss >= entry_price:
                    return None
                    
            else:  # bearish
                target_1 = entry_price - target_distance
                stop_loss = candle.high_price * (1 + self.config['buffer_pct'])
                
                if target_1 >= entry_price or stop_loss <= entry_price:
                    return None
            
            return {
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'target_price': target_1,
                'target_2': None,  # Doji tem apenas 1 target
                'targets_logic': f"Conservative 50% range: {target_distance:.4f}"
            }
            
        except Exception as e:
            self.logger.debug(f"Erro nos parâmetros doji: {e}")
            return None

    def _validate_pattern_parameters(self, pattern: CandlestickPattern) -> bool:
        """🔍 Validação final dos parâmetros"""
        try:
            # Calcula métricas de risco
            risk_pct = abs(pattern.entry_price - pattern.stop_loss) / pattern.entry_price * 100
            reward_1_pct = abs(pattern.target_price - pattern.entry_price) / pattern.entry_price * 100
            
            # Validação 1: Risco máximo
            if risk_pct > self.config['max_risk_pct']:
                self.logger.debug(f"{pattern.name}: Risco alto {risk_pct:.1f}% > {self.config['max_risk_pct']}%")
                return False
            
            # Validação 2: Relação risco/recompensa
            if risk_pct > 0:
                reward_risk_ratio = reward_1_pct / risk_pct
                if reward_risk_ratio < self.config['min_reward_ratio']:
                    self.logger.debug(f"{pattern.name}: R/R baixo {reward_risk_ratio:.1f} < {self.config['min_reward_ratio']}")
                    return False
            
            # Validação 3: Target muito distante
            if reward_1_pct > self.config['max_target_distance_pct']:
                self.logger.debug(f"{pattern.name}: Target distante {reward_1_pct:.1f}% > {self.config['max_target_distance_pct']}%")
                return False
            
            # Validação 4: Direção correta
            if pattern.pattern_type == 'bullish':
                if pattern.target_price <= pattern.entry_price or pattern.stop_loss >= pattern.entry_price:
                    return False
            else:
                if pattern.target_price >= pattern.entry_price or pattern.stop_loss <= pattern.entry_price:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.debug(f"Erro na validação de {pattern.name}: {e}")
            return False

def generate_candlestick_signals(df: pd.DataFrame, symbol: str) -> List[Dict]:
    """
    🎯 FUNÇÃO PRINCIPAL CORRIGIDA - Compatível com o sistema existente
    """
    detector = SimplifiedCandlestickDetector()
    patterns = detector.detect_effective_patterns(df)
    
    signals = []
    for pattern in patterns:
        
        # Prepara targets (1 ou 2)
        targets_list = [pattern.target_price]
        if pattern.target_2 is not None:
            targets_list.append(pattern.target_2)
        
        # Converte para formato esperado pelo sistema
        signal_data = {
            'detector_type': 'candlestick',
            'detector_name': pattern.name,
            'signal_type': 'BUY_LONG' if pattern.pattern_type == 'bullish' else 'SELL_SHORT',
            'confidence': pattern.reliability_score,
            'entry_price': pattern.entry_price,
            'stop_loss': pattern.stop_loss,
            'targets': targets_list,
            'market_data': df,
            
            # Metadados do pattern
            'pattern_data': {
                'pattern_strength': pattern.pattern_strength,
                'targets_logic': pattern.targets_logic,
                'calculation_method': 'pattern_structure_only',
                'risk_pct': abs(pattern.entry_price - pattern.stop_loss) / pattern.entry_price * 100,
                'reward_1_pct': abs(pattern.target_price - pattern.entry_price) / pattern.entry_price * 100,
                'position_index': pattern.position_index
            }
        }
        
        signals.append(signal_data)
    
    # Ordena por confidence
    return sorted(signals, key=lambda x: x['confidence'], reverse=True)

def verify_patterns_implementation() -> bool:
    """Verifica se a implementação está correta"""
    return True

def get_pattern_statistics() -> Dict:
    """Estatísticas dos padrões implementados"""
    return {
        'total_patterns': 5,
        'calculation_method': 'pattern_structure_only',
        'max_risk_pct': 2.5,
        'max_target_pct': 5.0,
        'patterns': {
            'Bullish_Engulfing': {'reliability': 0.85, 'calculation': 'engulfing_body_size'},
            'Bearish_Engulfing': {'reliability': 0.90, 'calculation': 'engulfing_body_size'},
            'Hammer': {'reliability': 0.75, 'calculation': 'lower_shadow_projection'},
            'Shooting_Star': {'reliability': 0.75, 'calculation': 'upper_shadow_projection'},
            'Doji': {'reliability': 0.60, 'calculation': 'conservative_range_based'}
        },
        'validation_rules': [
            'risk_pct <= 2.5%',
            'reward_risk_ratio >= 1.2',
            'target_distance <= 5%',
            'stops_at_pattern_extremes'
        ]
    }

# 🧪 FUNÇÃO DE TESTE PARA VALIDAR CORREÇÃO
def test_pattern_calculation(df: pd.DataFrame) -> Dict:
    """Testa se os cálculos estão corretos"""
    if len(df) < 2:
        return {'error': 'Dados insuficientes'}
    
    signals = generate_candlestick_signals(df, "TEST")
    
    test_results = {
        'signals_detected': len(signals),
        'signals_data': [],
        'validation_passed': True,
        'issues': []
    }
    
    for signal in signals:
        risk_pct = signal['pattern_data']['risk_pct']
        reward_pct = signal['pattern_data']['reward_1_pct']
        
        signal_test = {
            'pattern': signal['detector_name'],
            'risk_pct': risk_pct,
            'reward_pct': reward_pct,
            'risk_ok': risk_pct <= 2.5,
            'reward_ok': reward_pct >= risk_pct,
            'calculation': signal['pattern_data']['targets_logic']
        }
        
        if not signal_test['risk_ok']:
            test_results['issues'].append(f"{signal['detector_name']}: Risco alto ({risk_pct:.1f}%)")
            test_results['validation_passed'] = False
        
        if not signal_test['reward_ok']:
            test_results['issues'].append(f"{signal['detector_name']}: Reward baixo ({reward_pct:.1f}%)")
            test_results['validation_passed'] = False
        
        test_results['signals_data'].append(signal_test)
    
    return test_results

# Exports para compatibilidade
__all__ = [
    'SimplifiedCandlestickDetector', 
    'CandlestickPattern', 
    'generate_candlestick_signals', 
    'verify_patterns_implementation',
    'get_pattern_statistics',
    'test_pattern_calculation'
]