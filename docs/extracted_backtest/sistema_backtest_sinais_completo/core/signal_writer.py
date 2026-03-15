# signal_writer.py - CORREÇÃO COMPLETA DOS TARGETS TÉCNICOS

"""
Signal Writer - CORREÇÃO DOS ERROS DE TARGETS:
1. Validação robusta de tipos antes de conversões float()
2. Fallbacks seguros para todos os cálculos
3. Serialização JSON 100% segura
"""
import sqlite3
import json
import time
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field, asdict
import logging
import hashlib

from config.settings import settings

def safe_float_conversion(value, fallback: float = 0.0) -> float:
    """Converte qualquer valor para float de forma segura"""
    if value is None:
        return fallback
    
    # Se já é float ou int
    if isinstance(value, (int, float)):
        return float(value)
    
    # Se é string
    if isinstance(value, str):
        try:
            return float(value)
        except (ValueError, TypeError):
            return fallback
    
    # Se é numpy type
    if hasattr(value, 'item'):
        try:
            return float(value.item())
        except:
            return fallback
    
    # Se é pandas Series (pega o último valor)
    if hasattr(value, 'iloc'):
        try:
            return float(value.iloc[-1])
        except:
            return fallback
    
    # Se é lista ou array (pega o último)
    if isinstance(value, (list, tuple)) and len(value) > 0:
        return safe_float_conversion(value[-1], fallback)
    
    # Se é dict, tenta extrair valor numérico
    if isinstance(value, dict):
        # Procura por chaves comuns
        for key in ['value', 'price', 'level', 'target', 'close', 'last']:
            if key in value:
                return safe_float_conversion(value[key], fallback)
        
        # Se não encontrou, tenta o primeiro valor numérico
        for v in value.values():
            try:
                return safe_float_conversion(v, fallback)
            except:
                continue
        
        return fallback
    
    # Último recurso: tenta converter diretamente
    try:
        return float(value)
    except:
        return fallback


def safe_json_serialize(obj):
    """Serializa objetos de forma segura para JSON"""
    if obj is None:
        return None
    
    # Se já é um tipo básico JSON-serializável
    if isinstance(obj, (str, int, float, bool)):
        return obj
    
    # Se é lista ou tupla
    if isinstance(obj, (list, tuple)):
        return [safe_json_serialize(item) for item in obj]
    
    # Se é um dicionário
    if isinstance(obj, dict):
        return {str(k): safe_json_serialize(v) for k, v in obj.items()}
    
    # Se é um pandas Series ou DataFrame
    if hasattr(obj, 'to_list'):
        try:
            return obj.to_list()
        except:
            pass
    elif hasattr(obj, 'to_dict'):
        try:
            return safe_json_serialize(obj.to_dict())
        except:
            pass
    
    # Se é um objeto com __dict__
    if hasattr(obj, '__dict__'):
        try:
            return {k: safe_json_serialize(v) for k, v in obj.__dict__.items()}
        except:
            pass
    
    # Para datetime
    if isinstance(obj, datetime):
        return obj.isoformat()
    
    # Para numpy types
    if hasattr(obj, 'item'):  # numpy scalar
        try:
            return obj.item()
        except:
            pass
    
    # Para objetos não serializáveis, converte para string
    try:
        return str(obj)
    except:
        return None

@dataclass
class EnhancedTradingSignal:
    """Estrutura de sinal com VALIDAÇÃO ROBUSTA DE TARGETS"""
    symbol: str
    signal_type: str
    entry_price: float
    confidence: float
    timeframe: str
    detector_type: str
    detector_name: str
    
    # Dados de mercado para cálculos técnicos
    market_data: Optional[pd.DataFrame] = None
    
    id: str = None
    signal_hash: str = None
    signal_source: str = None
    targets: List[float] = None
    stop_loss: float = None
    confluence_score: int = 95
    status: str = "ACTIVE"
    indicators_used: List[str] = None
    targets_hit: List[bool] = None
    timeframe_analysis: Dict = field(default_factory=dict)
    market_conditions: Dict = field(default_factory=dict)
    pattern_data: Optional[Dict] = None
    technical_data: Optional[Dict] = None
    strategy: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    entry_timestamp: Optional[datetime] = None
    
    # Análises técnicas
    stop_loss_analysis: Optional[Dict] = None
    targets_analysis: Optional[Dict] = None

    def __post_init__(self):
        self.logger = logging.getLogger(__name__)
        
        self._normalize_signal_type()

        if self.id is None:
            ts = int(self.timestamp.timestamp() * 1000)
            self.id = f"{self.symbol}_{self.signal_type}_{ts}"

        hash_content = f"{self.symbol}_{self.timeframe}_{self.detector_name}_{int(self.timestamp.timestamp())}"
        if self.signal_hash is None:
            self.signal_hash = hashlib.md5(hash_content.encode()).hexdigest()[:12]

        if self.signal_source is None:
            direction = "bullish" if "BUY" in self.signal_type else "bearish"
            self.signal_source = f"{self.detector_name}_{direction}_{self.timeframe}"

        # FORÇA STATUS COMO ACTIVE
        self.status = "ACTIVE"

        # Valida entry_price
        self.entry_price = safe_float_conversion(self.entry_price, 100.0)

        # Cálculo técnico completo COM VALIDAÇÃO ROBUSTA
        if self.stop_loss is None: 
            self.stop_loss, self.stop_loss_analysis = self._calculate_technical_stop_loss_safe()
        if self.targets is None: 
            self.targets, self.targets_analysis = self._calculate_technical_targets_safe()
        
        if self.indicators_used is None: 
            self.indicators_used = [f"{self.detector_name.lower()}_analyze"]
        if self.targets_hit is None: 
            self.targets_hit = [False, False]  # Apenas 2 targets
        
        self._apply_precisions()
        self._validate_stop_and_targets()
        
        # Limpa objetos não serializáveis APÓS todos os cálculos
        # CORRIGE ENTRY_TIMESTAMP se não foi definido
        if self.entry_timestamp is None:
            if self.market_data is not None and len(self.market_data) > 0:
                try:
                    last_candle_timestamp = self.market_data.iloc[-1]['timestamp']
                    if pd.notna(last_candle_timestamp):
                        self.entry_timestamp = pd.to_datetime(last_candle_timestamp).to_pydatetime()
                    else:
                        self.entry_timestamp = self.timestamp
                except Exception:
                    self.entry_timestamp = self.timestamp
            else:
                self.entry_timestamp = self.timestamp

        # Limpa objetos não serializáveis APÓS todos os cálculos
        self._prepare_for_serialization()

    def _prepare_for_serialization(self):
        """Prepara o objeto para serialização JSON removendo objetos problemáticos"""
        
        # Remove/converte market_data que não é serializável
        if self.market_data is not None:
            self.market_data = None  # Remove para evitar problemas de serialização
        
        # Limpa technical_data de objetos não serializáveis
        if self.technical_data:
            self.technical_data = safe_json_serialize(self.technical_data)
        
        # Limpa pattern_data  
        if self.pattern_data:
            self.pattern_data = safe_json_serialize(self.pattern_data)
            
        # Limpa timeframe_analysis
        if self.timeframe_analysis:
            self.timeframe_analysis = safe_json_serialize(self.timeframe_analysis)
            
        # Limpa market_conditions
        if self.market_conditions:
            self.market_conditions = safe_json_serialize(self.market_conditions)
            
        # Limpa análises técnicas
        if self.stop_loss_analysis:
            self.stop_loss_analysis = safe_json_serialize(self.stop_loss_analysis)
            
        if self.targets_analysis:
            self.targets_analysis = safe_json_serialize(self.targets_analysis)

    def _normalize_signal_type(self):
        if self.signal_type.upper() in ['BUY', 'BULLISH']: 
            self.signal_type = 'BUY_LONG'
        elif self.signal_type.upper() in ['SELL', 'BEARISH']: 
            self.signal_type = 'SELL_SHORT'

    def _calculate_technical_stop_loss_safe(self) -> tuple[float, Dict]:
        """Cálcula stop loss técnico COM VALIDAÇÃO ROBUSTA"""
        try:
            from core.technical_stop_loss import TechnicalStopLossCalculator
            from core.data_reader import MarketData
            
            calculator = TechnicalStopLossCalculator()
            
            if isinstance(self.market_data, pd.DataFrame) and len(self.market_data) > 10:
                market_data_obj = MarketData(
                    symbol=self.symbol,
                    timeframe=self.timeframe,
                    data=self.market_data,
                    last_update=datetime.now()
                )
            else:
                # Força usar sistema inteligente mesmo com poucos dados
                market_data_obj = MarketData(
                    symbol=self.symbol,
                    timeframe=self.timeframe, 
                    data=self.market_data,
                    last_update=datetime.now()
                )
            
            stop_result = calculator.calculate_intelligent_stop_loss(
                market_data_obj, 
                self.signal_type, 
                self.entry_price, 
                self.timeframe
            )
            
            # VALIDAÇÃO ROBUSTA DO RESULTADO
            recommended_stop = safe_float_conversion(stop_result.recommended_stop, self._calculate_fallback_stop_loss())
            
            analysis_dict = {
                'method_used': str(stop_result.method_used),
                'confidence': safe_float_conversion(stop_result.confidence, 0.3),
                'risk_percentage': safe_float_conversion(stop_result.risk_percentage, 2.0),
                'atr_value': safe_float_conversion(stop_result.atr_value, 0),
                'nearest_support_resistance': safe_float_conversion(stop_result.nearest_support_resistance, None),
                'analysis_details': safe_json_serialize(stop_result.analysis_details)
            }
            
            self.logger.debug(f"Stop loss técnico calculado: {stop_result.method_used} para {self.symbol}")
            return recommended_stop, analysis_dict
            
        except Exception as e:
            self.logger.error(f"Erro no cálculo técnico de stop loss para {self.symbol}: {e}")
            return self._calculate_fallback_stop_loss(), {
                'method_used': 'Error_Fallback',
                'confidence': 0.2,
                'error': str(e),
                'risk_percentage': 2.0,
                'atr_value': 0,
                'analysis_details': {'error': str(e)}
            }
    
    def _calculate_technical_targets_safe(self) -> tuple[List[float], Dict]:
        """Cálcula targets técnicos COM VALIDAÇÃO ROBUSTA - APENAS 2 TARGETS"""
        try:
            from core.technical_targets import TechnicalTargetsCalculator
            from core.data_reader import MarketData
            
            calculator = TechnicalTargetsCalculator()
            
            if isinstance(self.market_data, pd.DataFrame) and len(self.market_data) > 10:
                market_data_obj = MarketData(
                    symbol=self.symbol,
                    timeframe=self.timeframe,
                    data=self.market_data,
                    last_update=datetime.now()
                )
            else:
                return self._calculate_fallback_targets(), {
                    'method_used': 'Fallback_No_Data',
                    'confidence': 0.3,
                    'analysis_details': {'error': 'No market data available'}
                }
            
            temp_stop = safe_float_conversion(self.stop_loss if self.stop_loss else self._calculate_fallback_stop_loss())
            
            targets_result = calculator.calculate_intelligent_targets(
                market_data_obj,
                self.signal_type,
                self.entry_price,
                temp_stop,
                self.timeframe
            )
            
            # VALIDAÇÃO SUPER ROBUSTA DOS TARGETS - TRATA NoneType
            raw_targets = getattr(targets_result, 'targets', None)
            
            # Se targets é None ou não iterável, usa fallback
            if raw_targets is None:
                self.logger.warning(f"Targets retornaram None para {self.symbol}, usando fallback")
                return self._calculate_fallback_targets(), {
                    'method_used': 'None_Targets_Fallback',
                    'confidence': 0.3,
                    'analysis_details': {'error': 'Targets returned None'}
                }
            
            # Tenta converter para lista se não for iterável
            if not hasattr(raw_targets, '__iter__'):
                self.logger.warning(f"Targets não iteráveis para {self.symbol}, usando fallback")
                return self._calculate_fallback_targets(), {
                    'method_used': 'Non_Iterable_Targets_Fallback',
                    'confidence': 0.3,
                    'analysis_details': {'error': 'Targets not iterable'}
                }
            
            # Converte todos os targets para float de forma segura
            safe_targets = []
            try:
                for target in raw_targets:
                    converted_target = safe_float_conversion(target, None)
                    if converted_target is not None and converted_target > 0:
                        safe_targets.append(converted_target)
            except Exception as e:
                self.logger.warning(f"Erro ao iterar targets para {self.symbol}: {e}")
                return self._calculate_fallback_targets(), {
                    'method_used': 'Iteration_Error_Fallback',
                    'confidence': 0.3,
                    'analysis_details': {'error': f'Iteration error: {e}'}
                }
            
            # GARANTE APENAS 2 TARGETS
            if len(safe_targets) >= 2:
                final_targets = safe_targets[:2]
            elif len(safe_targets) == 1:
                # Se só tem 1 target, cria o segundo baseado no primeiro
                if 'BUY' in self.signal_type:
                    final_targets = [safe_targets[0], safe_targets[0] * 1.02]
                else:
                    final_targets = [safe_targets[0], safe_targets[0] * 0.98]
            else:
                # Se não tem targets válidos, usa fallback
                self.logger.warning(f"Nenhum target válido encontrado para {self.symbol}, usando fallback")
                final_targets = self._calculate_fallback_targets()
            
            # VALIDAÇÃO DOS RESULTADOS DE ANÁLISE COM PROTEÇÃO None
            target_levels = getattr(targets_result, 'target_levels', None)
            resistance_levels = getattr(targets_result, 'resistance_levels', None)
            support_levels = getattr(targets_result, 'support_levels', None)
            risk_reward_ratios = getattr(targets_result, 'risk_reward_ratios', None)
            
            analysis_dict = {
                'method_used': str(getattr(targets_result, 'method_used', 'Technical_Calculated')),
                'confidence': safe_float_conversion(getattr(targets_result, 'confidence', 0.5)),
                'target_levels': [safe_float_conversion(x) for x in (target_levels or final_targets)[:2]],
                'resistance_levels': [safe_float_conversion(x) for x in (resistance_levels or [])] if resistance_levels else [],
                'support_levels': [safe_float_conversion(x) for x in (support_levels or [])] if support_levels else [],
                'risk_reward_ratios': [safe_float_conversion(x) for x in (risk_reward_ratios or [])[:2]] if risk_reward_ratios else [],
                'analysis_details': safe_json_serialize(getattr(targets_result, 'analysis_details', {}))
            }
            
            self.logger.debug(f"Targets técnicos calculados: {analysis_dict['method_used']} para {self.symbol} (2 targets)")
            return final_targets, analysis_dict
            
        except ImportError:
            self.logger.warning("Sistema técnico de targets não disponível, usando fallback")
            return self._calculate_simple_technical_targets(), {
                'method_used': 'Simple_Technical_Fallback',
                'confidence': 0.5,
                'analysis_details': {'note': 'Technical targets system not available'}
            }
        except Exception as e:
            self.logger.error(f"Erro no cálculo técnico de targets para {self.symbol}: {e}")
            return self._calculate_fallback_targets(), {
                'method_used': 'Error_Fallback',
                'confidence': 0.2,
                'error': str(e),
                'analysis_details': {'error': str(e)}
            }

    def _calculate_simple_technical_targets(self) -> List[float]:
        """Targets técnicos simples baseados em ATR - 2 TARGETS"""
        try:
            if self.market_data is None or len(self.market_data) < 20:
                return self._calculate_fallback_targets()
            
            atr = self._calculate_atr(self.market_data, 14)
            resistance_levels, support_levels = self._find_nearby_levels(self.market_data)
            
            if 'BUY' in self.signal_type:
                valid_resistances = [safe_float_conversion(r) for r in resistance_levels if safe_float_conversion(r) > self.entry_price]
                valid_resistances = [r for r in valid_resistances if r > 0]
                
                if len(valid_resistances) >= 2:
                    targets = sorted(valid_resistances)[:2]
                else:
                    targets = [
                        self.entry_price + atr * 2.0,  # Target 1: 2x ATR
                        self.entry_price + atr * 3.5   # Target 2: 3.5x ATR
                    ]
            else:
                valid_supports = [safe_float_conversion(s) for s in support_levels if safe_float_conversion(s) < self.entry_price]
                valid_supports = [s for s in valid_supports if s > 0]
                
                if len(valid_supports) >= 2:
                    targets = sorted(valid_supports, reverse=True)[:2]
                else:
                    targets = [
                        self.entry_price - atr * 2.0,  # Target 1: 2x ATR
                        self.entry_price - atr * 3.5   # Target 2: 3.5x ATR
                    ]
            
            # Validação final
            final_targets = [safe_float_conversion(t, self.entry_price) for t in targets]
            return final_targets[:2]  # Garante apenas 2
            
        except Exception as e:
            self.logger.warning(f"Erro no cálculo simples de targets técnicos: {e}")
            return self._calculate_fallback_targets()

    
    def _find_nearby_levels(self, df: pd.DataFrame, lookback: int = 50) -> tuple[List[float], List[float]]:
        """🎯 BUSCA NÍVEIS S/R PRÓXIMOS E SIGNIFICATIVOS"""
        try:
            if df is None or len(df) < 10:
                return [], []
            
            # Usa dados recentes para níveis relevantes
            recent_data = df.tail(lookback)
            current_price = float(df.iloc[-1]['close_price'])
            
            if 'high_price' not in recent_data.columns or 'low_price' not in recent_data.columns:
                return [], []
            
            highs = recent_data['high_price'].values
            lows = recent_data['low_price'].values
            
            # 🔍 DETECTA PICOS E VALES SIGNIFICATIVOS
            resistance_levels = []
            support_levels = []
            
            # Busca máximas locais (resistências)
            for i in range(2, len(highs) - 2):
                current_high = highs[i]
                
                # Verifica se é pico local
                if (current_high > highs[i-1] and current_high > highs[i-2] and
                    current_high > highs[i+1] and current_high > highs[i+2]):
                    
                    # Só adiciona se for relevante (não muito longe do preço atual)
                    distance_pct = abs(current_high - current_price) / current_price
                    if distance_pct < 0.15:  # Máximo 15% de distância
                        resistance_levels.append(float(current_high))
            
            # Busca mínimas locais (suportes)
            for i in range(2, len(lows) - 2):
                current_low = lows[i]
                
                # Verifica se é vale local
                if (current_low < lows[i-1] and current_low < lows[i-2] and
                    current_low < lows[i+1] and current_low < lows[i+2]):
                    
                    # Só adiciona se for relevante
                    distance_pct = abs(current_price - current_low) / current_price
                    if distance_pct < 0.15:  # Máximo 15% de distância
                        support_levels.append(float(current_low))
            
            # 🎯 REMOVE NÍVEIS MUITO PRÓXIMOS (consolidação)
            resistance_levels = self._consolidate_levels(resistance_levels, current_price)
            support_levels = self._consolidate_levels(support_levels, current_price)
            
            # Ordena por proximidade ao preço atual
            resistance_levels.sort()
            support_levels.sort(reverse=True)
            
            return resistance_levels[:5], support_levels[:5]  # Máximo 5 de cada
            
        except Exception as e:
            return [], []

    def _consolidate_levels(self, levels: List[float], current_price: float) -> List[float]:
        """Remove níveis muito próximos entre si"""
        if not levels:
            return []
        
        consolidated = []
        levels_sorted = sorted(levels)
        
        for level in levels_sorted:
            # Só adiciona se não for muito próximo dos existentes
            is_unique = True
            for existing in consolidated:
                distance_pct = abs(level - existing) / current_price
                if distance_pct < 0.02:  # Menos de 2% de diferença
                    is_unique = False
                    break
            
            if is_unique:
                consolidated.append(level)
        
        return consolidated
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calcula ATR (Average True Range) COM VALIDAÇÃO"""
        try:
            if data is None or len(data) < period + 2:
                fallback_atr = self.entry_price * 0.015
                return fallback_atr
            
            df = data.iloc[:-1].copy() if len(data) > 1 else data.copy()
            
            if len(df) < period:
                fallback_atr = self.entry_price * 0.015
                return fallback_atr
            
            # Conversão segura das colunas
            df['prev_close'] = df['close_price'].shift(1)
            df['high_safe'] = df['high_price'].apply(lambda x: safe_float_conversion(x, 0))
            df['low_safe'] = df['low_price'].apply(lambda x: safe_float_conversion(x, 0))
            df['close_safe'] = df['close_price'].apply(lambda x: safe_float_conversion(x, 0))
            df['prev_close_safe'] = df['prev_close'].apply(lambda x: safe_float_conversion(x, 0))
            
            df['tr1'] = df['high_safe'] - df['low_safe']
            df['tr2'] = abs(df['high_safe'] - df['prev_close_safe'])
            df['tr3'] = abs(df['low_safe'] - df['prev_close_safe'])
            
            df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
            atr = df['true_range'].ewm(span=period, adjust=False).mean().iloc[-1]
            
            min_atr = self.entry_price * 0.005
            max_atr = self.entry_price * 0.03
            atr_final = max(min_atr, min(max_atr, safe_float_conversion(atr, self.entry_price * 0.015)))
            
            return atr_final
            
        except Exception as e:
            self.logger.warning(f"Erro no cálculo de ATR: {e}")
            fallback_atr = self.entry_price * 0.015
            return fallback_atr
    
    def _calculate_fallback_stop_loss(self) -> float:
        """Stop loss de emergência"""
        stop_percentage = 0.02  # 2% conservador
        
        if 'BUY' in self.signal_type:
            return self.entry_price * (1 - stop_percentage)
        else:
            return self.entry_price * (1 + stop_percentage)

    def _calculate_fallback_targets(self) -> List[float]:
        """Targets de emergência - APENAS 2"""
        if 'BUY' in self.signal_type:
            return [
                self.entry_price * 1.02,  # Target 1: +2%
                self.entry_price * 1.04   # Target 2: +4%
            ]
        else:
            return [
                self.entry_price * 0.98,  # Target 1: -2%
                self.entry_price * 0.96   # Target 2: -4%
            ]

    def _apply_precisions(self):
        """Aplica precisão de preços COM VALIDAÇÃO"""
        try:
            precision = settings.get_price_precision(self.symbol)
            
            self.entry_price = round(safe_float_conversion(self.entry_price), precision)
            self.stop_loss = round(safe_float_conversion(self.stop_loss), precision)
            self.targets = [round(safe_float_conversion(t), precision) for t in self.targets]
            
        except Exception as e:
            self.logger.warning(f"Erro na aplicação de precisão: {e}")

    def _validate_stop_and_targets(self):
        """Valida stop loss e targets COM CORREÇÃO AUTOMÁTICA"""
        try:
            # INICIALIZA risk_pct PARA TODOS OS CASOS
            risk_pct = 0.0
            
            # VALIDAÇÃO 1: Direção correta do stop loss
            if 'BUY' in self.signal_type:
                if self.stop_loss >= self.entry_price:
                    self.stop_loss = self.entry_price * 0.985
                    self.logger.warning(f"🔧 Stop loss LONG corrigido para {self.symbol}: {self.stop_loss:.4f}")
                
                # Calcula risco para BUY
                risk_pct = abs(self.entry_price - self.stop_loss) / self.entry_price * 100
                if risk_pct > 15:  # Máximo 15% de risco
                    self.stop_loss = self.entry_price * 0.85  # 15% de risco
                    risk_pct = 15.0  # Atualiza o valor
                    self.logger.warning(f"🔧 Stop loss LONG ajustado para risco máximo: {self.stop_loss:.4f}")
            
            elif 'SELL' in self.signal_type:
                if self.stop_loss <= self.entry_price:
                    self.stop_loss = self.entry_price * 1.015
                    self.logger.warning(f"🔧 Stop loss SHORT corrigido para {self.symbol}: {self.stop_loss:.4f}")
                
                # Calcula risco para SELL
                risk_pct = abs(self.stop_loss - self.entry_price) / self.entry_price * 100
                if risk_pct > 15:  # Máximo 15% de risco
                    self.stop_loss = self.entry_price * 1.15  # 15% de risco
                    risk_pct = 15.0  # Atualiza o valor
                    self.logger.warning(f"🔧 Stop loss SHORT ajustado para risco máximo: {self.stop_loss:.4f}")  
                    
            # VALIDAÇÃO 2: Garante apenas 2 targets
            if len(self.targets) < 1:
                self.targets = self._calculate_fallback_targets()
                self.logger.warning(f"🎯 Nenhum target para {self.symbol}, calculados automaticamente")
            elif len(self.targets) == 1:
                # Aceita 1 target (patterns como Doji)
                self.logger.debug(f"🎯 {self.symbol}: 1 target aceito (pattern conservador)")
            
            # VALIDAÇÃO 3: Targets na direção correta
            self.logger.info(f"🎯 Validando targets para {self.symbol}: Entry=${self.entry_price:.4f}")

            # VALIDAÇÃO 3: Targets na direção correta
            self.logger.info(f"🎯 Validando targets para {self.symbol}: Entry=${self.entry_price:.4f}")

            for i, target in enumerate(self.targets):
                target_safe = safe_float_conversion(target, self.entry_price)
                
                if 'BUY' in self.signal_type:
                    if target_safe <= self.entry_price:
                        # Targets corrigidos baseados no entry_price correto
                        self.targets[i] = self.entry_price * (1.015 + i * 0.015)  # 1.5% e 3%
                        self.logger.warning(f"🎯 Target {i+1} LONG corrigido: ${self.targets[i]:.4f} (era ${target_safe:.4f})")
                else:  # SELL
                    if target_safe >= self.entry_price:
                        # Targets SHORT corrigidos
                        self.targets[i] = self.entry_price * (0.985 - i * 0.015)  # -1.5% e -3%
                        self.logger.warning(f"🎯 Target {i+1} SHORT corrigido: ${self.targets[i]:.4f} (era ${target_safe:.4f})")
            
            # VALIDAÇÃO 4: Ordem dos targets
            if 'BUY' in self.signal_type:
                self.targets.sort()  # Crescente para LONG
            else:
                self.targets.sort(reverse=True)  # Decrescente para SHORT
                
        except Exception as e:
            self.logger.error(f"❌ Erro na validação de stop/targets para {self.symbol}: {e}")
            # Log adicional para debug
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Em caso de erro, usa valores seguros
            self.targets = self._calculate_fallback_targets()
            self.stop_loss = self._calculate_fallback_stop_loss()

class EnhancedSignalWriter:
    """Signal Writer com VALIDAÇÃO ROBUSTA DE TARGETS"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.db_path = settings.database.signals_db_path
        self.signals_table = settings.database.signals_table
        self.backup_table = settings.database.backup_table
        
        # Locks para evitar race conditions
        self._lock = threading.Lock()
        self._db_lock = threading.Lock()
        
        self._ensure_tables_exist()
        self.logger.info("🚨 EnhancedSignalWriter CORRIGIDO - Targets e JSON seguros + Thread-safe")
        
    def _get_connection(self):
        return sqlite3.connect(self.db_path, timeout=10)

    def _ensure_tables_exist(self):
        """Garante que as tabelas existam"""
        create_signals_table = f"""
        CREATE TABLE IF NOT EXISTS {self.signals_table} (
            id TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            signal_type TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            detector_type TEXT NOT NULL,
            detector_name TEXT NOT NULL,
            signal_source TEXT,
            signal_hash TEXT,
            entry_price REAL NOT NULL,
            targets TEXT,
            stop_loss REAL NOT NULL,
            confidence REAL NOT NULL,
            confluence_score INTEGER DEFAULT 95,
            status TEXT DEFAULT 'ACTIVE',
            created_at TEXT NOT NULL,
            entry_time TEXT,
            current_price REAL,
            targets_hit TEXT,
            indicators_used TEXT,
            updated_at TEXT,
            timeframe_analysis TEXT,
            market_conditions TEXT,
            pattern_data TEXT,
            technical_data TEXT,
            stop_loss_analysis TEXT,
            targets_analysis TEXT
        )
        """
        
        create_backup_table = f"""
        CREATE TABLE IF NOT EXISTS {self.backup_table} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_id TEXT,
            symbol TEXT,
            signal_type TEXT,
            timeframe TEXT,
            detector_type TEXT,
            detector_name TEXT,
            signal_source TEXT,
            signal_hash TEXT,
            entry_price REAL,
            confidence REAL,
            confluence_score INTEGER,
            status TEXT,
            created_at TEXT,
            backup_reason TEXT,
            targets TEXT,
            stop_loss REAL,
            indicators_used TEXT,
            timeframe_analysis TEXT,
            market_conditions TEXT,
            pattern_data TEXT,
            technical_data TEXT,
            stop_loss_analysis TEXT,
            targets_analysis TEXT,
            backup_timestamp TEXT
        )
        """
        
        try:
            with self._get_connection() as conn:
                conn.execute(create_signals_table)
                conn.execute(create_backup_table)
                conn.commit()
                self.logger.debug("✅ Tabelas verificadas/criadas")
        except Exception as e:
            self.logger.error(f"❌ Erro ao criar tabelas: {e}")
    
    
    def validate_signal_pricing(self, signal):
        """Validação FINAL com dados mais atuais possíveis"""
        try:
            # Buscar preço MAIS ATUAL do banco (pode ser mais novo que os dados da análise)
            current_price = self.get_current_price_from_db(signal.symbol)
            if not current_price:
                return True, "Sem preço atual para validar"
            
            # Diferença entre preço de análise e preço atual
            price_diff_pct = abs(current_price - signal.entry_price) / signal.entry_price * 100
            
            
            self.logger.info(f"Predict price {price_diff_pct} ")
            # VALIDAÇÃO 1: Divergência máxima permitida
            if price_diff_pct > 3:  # Máximo 1.5%
                return False, f"DIVERGÊNCIA ALTA: {price_diff_pct:.2f}% (análise: ${signal.entry_price:.4f}, atual: ${current_price:.4f})"
            
            # VALIDAÇÃO 2: Sinal não pode estar "pré-executado"
            if signal.signal_type == 'BUY_LONG':
                if current_price >= signal.targets[0] * 0.995:  # 0.5% tolerância
                    return False, f"PRÉ-EXECUTADO: atual ${current_price:.4f} próximo/acima target ${signal.targets[0]:.4f}"
                if current_price <= signal.stop_loss * 1.005:  # 0.5% tolerância
                    return False, f"JÁ STOPADO: atual ${current_price:.4f} próximo/abaixo stop ${signal.stop_loss:.4f}"
                    
            elif signal.signal_type == 'SELL_SHORT':
                if current_price <= signal.targets[0] * 1.005:  # 0.5% tolerância
                    return False, f"PRÉ-EXECUTADO: atual ${current_price:.4f} próximo/abaixo target ${signal.targets[0]:.4f}"
                if current_price >= signal.stop_loss * 0.995:  # 0.5% tolerância
                    return False, f"JÁ STOPADO: atual ${current_price:.4f} próximo/acima stop ${signal.stop_loss:.4f}"
            
            return True, f"OK (diff: {price_diff_pct:.2f}%, atual: ${current_price:.4f})"
            
        except Exception as e:
            return True, f"Erro na validação: {e}"
    
    def get_current_price_from_db(self, symbol):
        """Busca o preço MAIS ATUAL possível (incluindo candle dinâmico)"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Busca primeiro por dados de 1m (mais atuais), depois 5m
            for tf in ['1h', '4h', '1d']:
                cursor.execute("""
                    SELECT close_price FROM crypto_ohlc 
                    WHERE symbol = ? AND timeframe = ? 
                    ORDER BY timestamp DESC LIMIT 1
                """, (symbol, tf))
                
                result = cursor.fetchone()
                if result:
                    conn.close()
                    return float(result[0])
            
            conn.close()
            return None
            
        except Exception as e:
            return None
    
    def check_existing_active_signals(self, symbol: str) -> bool:
        """
        Verifica sinais que REALMENTE BLOQUEIAM novos sinais baseado na lógica de targets:
        - ACTIVE: sempre bloqueia
        - TARGET_1_HIT: bloqueia apenas se tem 2 targets (ainda não finalizou)
        - TARGET_2_HIT, STOP_HIT, EXPIRED, MANUALLY_CLOSED: nunca bloqueiam
        """
        query = f"""
        SELECT id, status, targets, created_at, timeframe, detector_name
        FROM {self.signals_table} 
        WHERE symbol = ? AND status IN ('ACTIVE', 'TARGET_1_HIT')
        ORDER BY created_at DESC
        """
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (symbol,))
                potential_blocking_signals = cursor.fetchall()
                
                if not potential_blocking_signals:
                    return False
                
                actually_blocking = []
                
                for signal_id, status, targets_json, created_at, timeframe, detector_name in potential_blocking_signals:
                    # Parse targets
                    try:
                        targets = json.loads(targets_json) if targets_json else []
                        targets_count = len(targets) if targets else 0
                    except:
                        targets_count = 0
                    
                    is_blocking = False
                    blocking_reason = ""
                    
                    if status == 'ACTIVE':
                        # ACTIVE sempre bloqueia
                        is_blocking = True
                        blocking_reason = "Sinal ativo aguardando resultado"
                        
                    elif status == 'TARGET_1_HIT':
                        if targets_count >= 2:
                            # Target 1 atingido, mas ainda tem target 2 pendente
                            is_blocking = True
                            blocking_reason = f"Target 1/2 atingido, target 2 ainda pendente"
                        else:
                            # Só tinha 1 target e já foi atingido - não bloqueia mais
                            is_blocking = False
                            blocking_reason = f"Target único atingido - não bloqueia"
                            self.logger.debug(f"✅ {symbol}: Sinal {signal_id} com 1 target atingido não bloqueia mais")
                    
                    if is_blocking:
                        actually_blocking.append({
                            'id': signal_id,
                            'status': status,
                            'targets_count': targets_count,
                            'reason': blocking_reason,
                            'timeframe': timeframe,
                            'detector_name': detector_name
                        })
                
                if actually_blocking:
                    self.logger.info(f"🚫 {symbol} BLOQUEADO: {len(actually_blocking)} sinal(s) que impedem novos sinais:")
                    for signal in actually_blocking:
                        self.logger.info(f"   • {signal['id']} | {signal['status']} | {signal['timeframe']} | {signal['detector_name']} | {signal['reason']}")
                    return True
                else:
                    self.logger.debug(f"✅ {symbol}: Sem sinais bloqueadores (targets únicos já atingidos)")
                    return False
                        
        except Exception as e:
            self.logger.error(f"❌ Erro ao verificar sinais bloqueadores para {symbol}: {e}")
            return False
    
    def _check_duplicate_signal(self, signal):
        """Verifica se já existe sinal muito similar (anti-duplicação)"""
        try:
            # Busca sinais dos últimos 30 minutos
            time_threshold = datetime.now() - timedelta(minutes=30)
            
            query = f"""
            SELECT entry_price, detector_name, created_at 
            FROM {self.signals_table} 
            WHERE symbol = ? 
            AND timeframe = ? 
            AND datetime(created_at) > ? 
            ORDER BY created_at DESC
            """
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (
                    signal.symbol, 
                    signal.timeframe, 
                    time_threshold.isoformat()
                ))
                
                recent_signals = cursor.fetchall()
                
                for entry_price, detector_name, created_at in recent_signals:
                    # Mesmo detector e preço muito similar = duplicado
                    if (detector_name == signal.detector_name and
                        abs(float(entry_price) - signal.entry_price) / signal.entry_price < 0.005):  # 0.5% tolerância
                        
                        self.logger.debug(f"Sinal duplicado detectado: {detector_name} preço similar {entry_price}")
                        return True
                
                return False
                
        except Exception as e:
            self.logger.error(f"Erro na verificação de duplicação: {e}")
            return False
    
    
    def write_enhanced_signal(self, signal: EnhancedTradingSignal) -> bool:
        """GRAVAÇÃO CORRIGIDA com validação robusta + Thread-safe"""
        
        with self._lock:  # Lock para operações concorrentes
            # Validação de sinal obsoleto
            if not self._validate_signal_freshness(signal):
                self._backup_signal(signal, "blocked_stale_signal")
                return False
            
            # Verificação de bloqueio
            if self.check_existing_active_signals(signal.symbol):
                self.logger.info(f"🚫 SINAL BLOQUEADO: {signal.symbol} tem sinal ativo/TARGET_1_HIT")
                self._backup_signal(signal, "blocked_existing_active_signal")
                return False
            # VALIDAÇÃO ANTI-DUPLICAÇÃO: Verifica se já existe sinal muito similar
            if self._check_duplicate_signal(signal):
                self.logger.warning(f"🚫 SINAL DUPLICADO: {signal.symbol} {signal.detector_name} muito similar ao existente")
                self._backup_signal(signal, "blocked_duplicate_signal")
                return False
            # ADICIONAR validação antes de INSERT
            is_valid, validation_msg = self.validate_signal_pricing(signal)
            if not is_valid:
                self.logger.warning(f"❌ Sinal rejeitado: {signal.symbol} - {validation_msg}")
                return False
            
            # CORREÇÃO: Validar targets e stops
            if not self.validate_targets_and_stops(signal):
                self.logger.warning(f"❌ Sinal rejeitado: {signal.symbol} - Targets/stops inválidos")
                return False
            
            # FORÇA STATUS ACTIVE
            signal.status = "ACTIVE"
            
            self.logger.info(f"💾 GRAVANDO NOVO SINAL: {signal.symbol} | {signal.timeframe} | {signal.detector_name} | Status: {signal.status}")
                
        sql = f"""
        INSERT OR REPLACE INTO {self.signals_table} (
            id, symbol, signal_type, timeframe, detector_type, detector_name,
            signal_source, signal_hash, entry_price, targets, stop_loss,
            confidence, confluence_score, status, created_at, entry_time,
            current_price, targets_hit, indicators_used, updated_at,
            timeframe_analysis, market_conditions, pattern_data, technical_data,
            stop_loss_analysis, targets_analysis
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        try:
            # Faz backup primeiro
            self._backup_signal(signal, "generated")

            # SERIALIZAÇÃO JSON SEGURA COM VALIDAÇÃO + Thread-safe
            with self._db_lock:  # Lock para operações de banco
                with self._get_connection() as conn:
                    entry_time = signal.entry_timestamp if signal.entry_timestamp else signal.timestamp
                    # TIMESTAMPS CORRETOS
                    current_timestamp = datetime.now()  # SYSDATE para created_at

                    # ENTRY_TIME = timestamp do candle de análise
                    if hasattr(signal, 'entry_timestamp') and signal.entry_timestamp:
                        candle_close_time = signal.entry_timestamp
                    else:
                        candle_close_time = current_timestamp

                    values = (
                        signal.id, signal.symbol, signal.signal_type, signal.timeframe,
                        signal.detector_type, signal.detector_name, signal.signal_source,
                        signal.signal_hash, safe_float_conversion(signal.entry_price), 
                        json.dumps([safe_float_conversion(t) for t in signal.targets], default=safe_json_serialize),
                        safe_float_conversion(signal.stop_loss), safe_float_conversion(signal.confidence), 
                        signal.confluence_score, signal.status, 
                        current_timestamp.isoformat(),          # created_at = SYSDATE
                        candle_close_time.isoformat(),         # entry_time = close time do candle
                        safe_float_conversion(signal.entry_price),  # current_price
                        json.dumps(signal.targets_hit, default=safe_json_serialize),
                        json.dumps(signal.indicators_used, default=safe_json_serialize), 
                        current_timestamp.isoformat(),         # updated_at
                        json.dumps(signal.timeframe_analysis, default=safe_json_serialize),
                        json.dumps(signal.market_conditions, default=safe_json_serialize),
                        json.dumps(signal.pattern_data, default=safe_json_serialize),
                        json.dumps(signal.technical_data, default=safe_json_serialize),
                        json.dumps(signal.stop_loss_analysis, default=safe_json_serialize),
                        json.dumps(signal.targets_analysis, default=safe_json_serialize)
                    )
                    conn.execute(sql, values)
                    conn.commit()
            
            # Log detalhado de sucesso
            risk_pct = abs(signal.stop_loss - signal.entry_price) / signal.entry_price * 100
            target1_pct = abs(signal.targets[0] - signal.entry_price) / signal.entry_price * 100
            target2_pct = abs(signal.targets[1] - signal.entry_price) / signal.entry_price * 100
            
            stop_method = signal.stop_loss_analysis.get('method_used', 'Unknown') if signal.stop_loss_analysis else 'Unknown'
            targets_method = signal.targets_analysis.get('method_used', 'Unknown') if signal.targets_analysis else 'Unknown'
            
            self.logger.info(
                f"✅ SINAL GRAVADO COMO ACTIVE: {signal.symbol} {signal.timeframe} | "
                f"Entry: {signal.entry_price:.4f} | "
                f"Stop: {signal.stop_loss:.4f} ({risk_pct:.1f}% risco) [{stop_method}] | "
                f"T1: {signal.targets[0]:.4f} ({target1_pct:.1f}%) | "
                f"T2: {signal.targets[1]:.4f} ({target2_pct:.1f}%) [{targets_method}] | "
                f"Conf: {signal.confidence:.3f}"
            )
            
            # Verificação pós-gravação
            self._verify_signal_saved_correctly(signal)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ ERRO AO GRAVAR SINAL: {signal.symbol} - {e}")
            self._backup_signal(signal, f"insert_error: {e}")
            return False

    def _validate_signal_freshness(self, signal: EnhancedTradingSignal) -> bool:
        """Valida se o sinal não está obsoleto"""
        try:
            # LOG ADICIONAL para debug
            self.logger.info(f"🔍 Validando freshness: {signal.symbol} | Entry: {signal.entry_price:.4f} | T1: {signal.targets[0]:.4f} | Stop: {signal.stop_loss:.4f}")
            
            if signal.signal_type == 'BUY_LONG':
                if signal.entry_price >= signal.targets[0]:
                    self.logger.warning(f"🚫 Sinal BUY obsoleto para {signal.symbol}: Entry {signal.entry_price:.4f} >= T1 {signal.targets[0]:.4f}")
                    return False
            
            elif signal.signal_type == 'SELL_SHORT':
                if signal.entry_price <= signal.targets[0]:
                    self.logger.warning(f"🚫 Sinal SELL obsoleto para {signal.symbol}: Entry {signal.entry_price:.4f} <= T1 {signal.targets[0]:.4f}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erro na validação de sinal obsoleto: {e}")
            return True

    def _verify_signal_saved_correctly(self, signal: EnhancedTradingSignal):
        """Verifica se o sinal foi gravado corretamente"""
        try:
            query = f"SELECT status, targets_hit FROM {self.signals_table} WHERE id = ?"
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (signal.id,))
                result = cursor.fetchone()
                
                if result:
                    saved_status, saved_targets_hit = result
                    self.logger.debug(f"✅ VERIFICAÇÃO: {signal.symbol} salvo com status={saved_status}, targets_hit={saved_targets_hit}")
                    
                    if saved_status != "ACTIVE":
                        self.logger.error(f"❌ ERRO: {signal.symbol} salvo com status incorreto: {saved_status} (deveria ser ACTIVE)")
                else:
                    self.logger.error(f"❌ ERRO: {signal.symbol} não encontrado no banco após gravação!")
                    
        except Exception as e:
            self.logger.error(f"Erro na verificação pós-gravação: {e}")

    def validate_targets_and_stops(self, signal) -> bool:
        """Validação robusta de coerência de targets e stops"""
        try:
            # Para LONG
            if 'BUY' in signal.signal_type:
                # Stop deve estar abaixo da entrada
                if signal.stop_loss >= signal.entry_price:
                    self.logger.error(f"LONG: Stop {signal.stop_loss} >= Entry {signal.entry_price}")
                    return False
                
                # Targets devem estar acima da entrada e em ordem crescente
                prev_target = signal.entry_price
                for i, target in enumerate(signal.targets):
                    if target <= prev_target:
                        self.logger.error(f"LONG: Target {i+1} ({target}) <= {prev_target}")
                        return False
                    prev_target = target
                
                # Risk/Reward mínimo
                risk = signal.entry_price - signal.stop_loss
                reward = signal.targets[0] - signal.entry_price
                if reward / risk < 1.2:  # Mínimo 1.2:1
                    self.logger.warning(f"LONG: R/R baixo: {reward/risk:.2f}")
                    return False
            
            # Para SHORT
            elif 'SELL' in signal.signal_type:
                # Stop deve estar acima da entrada
                if signal.stop_loss <= signal.entry_price:
                    self.logger.error(f"SHORT: Stop {signal.stop_loss} <= Entry {signal.entry_price}")
                    return False
                
                # Targets devem estar abaixo da entrada e em ordem decrescente
                prev_target = signal.entry_price
                for i, target in enumerate(signal.targets):
                    if target >= prev_target:
                        self.logger.error(f"SHORT: Target {i+1} ({target}) >= {prev_target}")
                        return False
                    prev_target = target
                
                # Risk/Reward mínimo
                risk = signal.stop_loss - signal.entry_price
                reward = signal.entry_price - signal.targets[0]
                if reward / risk < 1.2:
                    self.logger.warning(f"SHORT: R/R baixo: {reward/risk:.2f}")
                    return False
            
            return True
        except Exception as e:
            self.logger.error(f"Erro na validação de targets/stops: {e}")
            return False

    def get_active_signals_count(self, symbol: str) -> int:
        """Conta sinais que REALMENTE bloqueiam novos sinais"""
        query = f"""
        SELECT COUNT(*) 
        FROM {self.signals_table} 
        WHERE symbol = ? AND status IN ('ACTIVE', 'TARGET_1_HIT')
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (symbol,))
                count = cursor.fetchone()[0]
                return count
                
        except Exception as e:
            self.logger.error(f"Erro ao contar sinais bloqueadores para {symbol}: {e}")
            return 0

    def move_inactive_signals_to_backup(self) -> Dict[str, int]:
        """Move sinais FINALIZADOS para backup"""
        moved_counts = {'TARGET_2_HIT': 0, 'STOP_HIT': 0, 'EXPIRED': 0, 'MANUALLY_CLOSED': 0}
        
        final_statuses = ['TARGET_2_HIT', 'STOP_HIT', 'EXPIRED', 'MANUALLY_CLOSED']
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                for status in final_statuses:
                    select_query = f"""
                    SELECT * FROM {self.signals_table} 
                    WHERE status = ?
                    """
                    cursor.execute(select_query, (status,))
                    signals_to_move = cursor.fetchall()
                    
                    if signals_to_move:
                        for signal in signals_to_move:
                            self._backup_signal_from_row(signal, f"moved_to_backup_{status.lower()}")
                        
                        delete_query = f"""
                        DELETE FROM {self.signals_table} 
                        WHERE status = ?
                        """
                        cursor.execute(delete_query, (status,))
                        
                        moved_counts[status] = len(signals_to_move)
                        self.logger.info(f"📦 Movidos {len(signals_to_move)} sinais {status} para backup")
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"❌ Erro ao mover sinais finalizados para backup: {e}")
        
        return moved_counts
    
    def mark_expired_signals_as_killed(self) -> int:
        """Marca apenas sinais ACTIVE antigos como expirados"""
        hours_limit = settings.system.signal_lifecycle_hours
        cutoff_time = datetime.now() - timedelta(hours=hours_limit)
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                update_query = f"""
                UPDATE {self.signals_table} 
                SET status = 'EXPIRED', updated_at = ?
                WHERE status = 'ACTIVE' AND datetime(created_at) < ?
                """
                
                cursor.execute(update_query, (datetime.now().isoformat(), cutoff_time.isoformat()))
                expired_count = cursor.rowcount
                conn.commit()
                
                if expired_count > 0:
                    self.logger.info(f"⏰ {expired_count} sinais ACTIVE marcados como EXPIRED (lifecycle: {hours_limit}h)")
                
                return expired_count
                
        except Exception as e:
            self.logger.error(f"❌ Erro ao marcar sinais como EXPIRED: {e}")
            return 0
    
    def _backup_signal_from_row(self, signal_row: tuple, reason: str):
        """Faz backup de um sinal a partir de uma row do banco"""
        sql = f"""
        INSERT INTO {self.backup_table} (
            original_id, symbol, signal_type, timeframe, detector_type, detector_name,
            signal_source, signal_hash, entry_price, confidence, confluence_score,
            status, created_at, backup_reason,
            targets, stop_loss, indicators_used, timeframe_analysis, market_conditions,
            pattern_data, technical_data, stop_loss_analysis, targets_analysis, backup_timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        try:
            with self._get_connection() as conn:
                values = (
                    signal_row[0],  # id
                    signal_row[1],  # symbol
                    signal_row[2],  # signal_type
                    signal_row[3],  # timeframe
                    signal_row[4],  # detector_type
                    signal_row[5],  # detector_name
                    signal_row[6],  # signal_source
                    signal_row[7],  # signal_hash
                    signal_row[8],  # entry_price
                    signal_row[11], # confidence
                    signal_row[12], # confluence_score
                    signal_row[13], # status
                    signal_row[14], # created_at
                    reason,         # backup_reason
                    signal_row[9],  # targets
                    signal_row[10], # stop_loss
                    signal_row[18], # indicators_used
                    signal_row[20], # timeframe_analysis
                    signal_row[21], # market_conditions
                    signal_row[22], # pattern_data
                    signal_row[23], # technical_data
                    signal_row[24] if len(signal_row) > 24 else None,  # stop_loss_analysis
                    signal_row[25] if len(signal_row) > 25 else None,  # targets_analysis
                    datetime.now().isoformat() # backup_timestamp
                )
                conn.execute(sql, values)
                conn.commit()
        except Exception as e:
            self.logger.error(f"❌ Erro ao fazer backup da row: {e}")

    def _backup_signal(self, signal: EnhancedTradingSignal, reason: str):
        """Faz backup com serialização JSON segura"""
        sql = f"""
        INSERT INTO {self.backup_table} (
            original_id, symbol, signal_type, timeframe, detector_type, detector_name,
            signal_source, signal_hash, entry_price, confidence, confluence_score,
            status, created_at, backup_reason,
            targets, stop_loss, indicators_used, timeframe_analysis, market_conditions,
            pattern_data, technical_data, stop_loss_analysis, targets_analysis, backup_timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        try:
            with self._get_connection() as conn:
                # SERIALIZAÇÃO SEGURA
                # TIMESTAMPS CORRETOS
                current_timestamp = datetime.now()  # SYSDATE para created_at

                # ENTRY_TIME = timestamp do candle de análise
                if hasattr(signal, 'entry_timestamp') and signal.entry_timestamp:
                    candle_close_time = signal.entry_timestamp
                else:
                    candle_close_time = current_timestamp
                
                values = (
                    signal.id, signal.symbol, signal.signal_type, signal.timeframe,
                    signal.detector_type, signal.detector_name, signal.signal_source,
                    signal.signal_hash, safe_float_conversion(signal.entry_price), 
                    safe_float_conversion(signal.confidence), signal.confluence_score, signal.status,
                    signal.timestamp.isoformat(), reason,
                    json.dumps([safe_float_conversion(t) for t in signal.targets], default=safe_json_serialize), 
                    safe_float_conversion(signal.stop_loss),
                    json.dumps(signal.indicators_used, default=safe_json_serialize),
                    json.dumps(signal.timeframe_analysis, default=safe_json_serialize),
                    json.dumps(signal.market_conditions, default=safe_json_serialize),
                    json.dumps(signal.pattern_data, default=safe_json_serialize),
                    json.dumps(signal.technical_data, default=safe_json_serialize),
                    json.dumps(signal.stop_loss_analysis, default=safe_json_serialize),
                    json.dumps(signal.targets_analysis, default=safe_json_serialize),
                    datetime.now().isoformat()
                )
                conn.execute(sql, values)
                conn.commit()
        except Exception as e:
            self.logger.error(f"❌ Erro ao fazer backup do sinal: {e}")

    def get_current_signal_status_summary(self) -> Dict:
        """Resumo dos status atuais dos sinais"""
        try:
            query = f"""
            SELECT status, COUNT(*) as count, 
                   GROUP_CONCAT(DISTINCT symbol) as symbols
            FROM {self.signals_table}
            GROUP BY status
            ORDER BY count DESC
            """
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                results = cursor.fetchall()
                
                summary = {
                    'status_distribution': {},
                    'total_signals': 0,
                    'blocking_signals': 0,
                    'completed_signals': 0
                }
                
                for status, count, symbols in results:
                    summary['status_distribution'][status] = {
                        'count': count,
                        'symbols': symbols.split(',') if symbols else []
                    }
                    summary['total_signals'] += count
                    
                    if status in ['ACTIVE', 'TARGET_1_HIT']:
                        summary['blocking_signals'] += count
                    elif status in ['TARGET_2_HIT', 'STOP_HIT', 'EXPIRED']:
                        summary['completed_signals'] += count
                
                return summary
                
        except Exception as e:
            self.logger.error(f"Erro ao obter resumo de status: {e}")
            return {'error': str(e)}

# Aliases para compatibilidade
TradingSignal = EnhancedTradingSignal
SignalWriter = EnhancedSignalWriter