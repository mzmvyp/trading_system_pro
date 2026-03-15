# analyzer.py - PRIORIDADE 1d + SCORE RIGOROSO PARA 1h/4h + SEM LOCKS + FILTRO CANDLESTICK

import logging
import time
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta

from core.data_reader import DataReader, MarketData
from core.signal_writer import EnhancedSignalWriter, EnhancedTradingSignal
from indicators.technical import TechnicalAnalyzer, RSIAnalyzer

# NOVO: Imports para integração ML e LLM
try:
    from ml.ml_integration import MLSignalEnhancer, create_ml_enhancer
    from llm.llm_integration import LLMSignalEnhancer, create_llm_enhancer
    ML_LLM_AVAILABLE = True
    print("OK ML e LLM enhancers disponiveis")
except ImportError as e:
    ML_LLM_AVAILABLE = False
    print(f"Warning ML/LLM enhancers nao disponiveis: {e}") 

try:
    from core.improved_signal_quality_system import (
        create_rigorous_quality_system,
        SignalEffectivenessAnalyzer,
        RigorousQualityConfig
    )
    RIGOROUS_QUALITY_AVAILABLE = True
except ImportError:
    RIGOROUS_QUALITY_AVAILABLE = False
    logging.warning("⚠️ Sistema rigoroso de qualidade não disponível")


# 🔧 CORREÇÃO 1: Adicionar imports de candlestick
try:
    from indicators.candlestick_patterns_detector import generate_candlestick_signals
    CANDLESTICK_AVAILABLE = True
except ImportError as e:
    CANDLESTICK_AVAILABLE = False
    logging.warning(f"Detector de Candlestick não disponível: {e}")

# Sistema de monitoramento em tempo real DESABILITADO para evitar locks
REAL_TIME_MONITORING_AVAILABLE = False
logging.info("🚫 Monitoramento em tempo real DESABILITADO (evita locks de DB)")

from config.settings import settings

class MultiTimeframeAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.data_reader = DataReader()
        self.signal_writer = EnhancedSignalWriter()
        
        enabled_timeframes = settings.get_enabled_timeframes()
        self.technical_analyzers = {tf: TechnicalAnalyzer() for tf in enabled_timeframes}
        
        # NOVO: Inicialização dos enhancers ML e LLM
        self.ml_enhancer = None
        self.llm_enhancer = None
        
        if ML_LLM_AVAILABLE:
            try:
                # ML Enhancer com peso de 25%
                self.ml_enhancer = create_ml_enhancer(
                    enabled=settings.ml.enabled,
                    ml_weight=settings.ml.ml_weight
                )
                
                # LLM Enhancer com peso de 20%
                self.llm_enhancer = create_llm_enhancer(
                    enabled=settings.llm.enabled,
                    llm_weight=settings.llm.llm_weight
                )
                
                self.logger.info(f"✅ Integração ML/LLM ativa - ML: {settings.ml.ml_weight*100:.0f}%, LLM: {settings.llm.llm_weight*100:.0f}%")
            except Exception as e:
                self.logger.error(f"❌ Erro ao inicializar enhancers ML/LLM: {e}")
                self.ml_enhancer = None
                self.llm_enhancer = None
        else:
            self.logger.warning("⚠️ ML/LLM não disponível - usando apenas análise técnica")

        # NOVA LÓGICA: Processamento específico por timeframe (aguarda stream)
        self.timeframe_specific_mode = True
        self.quality_mode = "timeframe_specific_with_stream_delay"

        # SCHEDULER ESPECÍFICO POR TIMEFRAME
        try:
            from core.timeframe_scheduler import get_global_scheduler
            self.scheduler = get_global_scheduler()
            self.scheduler_enabled = True
            
            # Registra callbacks para cada timeframe
            self.scheduler.register_timeframe_callback("1h", self._process_1h_event)
            
            self.logger.info("✅ Scheduler específico inicializado (aguarda 35s após fechamento)")
        except ImportError:
            self.scheduler = None
            self.scheduler_enabled = False
            self.logger.warning("⚠️ Scheduler não disponível - usando modo tradicional")
            
        self._last_cleanup = datetime.now()
        
        # Monitoramento em tempo real DESABILITADO
        self.real_time_monitor = None
        self.monitoring_enabled = False

        # CONFIGURAÇÕES OTIMIZADAS - sem locks
        self.MAX_SYMBOL_TIME = 8  # Reduzido para 8s
        self.MAX_VALIDATION_TIME = 2  # Reduzido para 2s
        self.DISABLE_MICROSTRUCTURE = True  # Desabilita microestrutura
        
        self.logger.info("MultiTimeframeAnalyzer com PROCESSAMENTO ESPECÍFICO POR TIMEFRAME:")
        self.logger.info(f"  • Timeframes: {enabled_timeframes} (PROCESSAMENTO ESPECÍFICO)")
        self.logger.info(f"  • Integração ML/LLM: {'✅ ATIVA' if (self.ml_enhancer or self.llm_enhancer) else '❌ INATIVA'}")
        self.logger.info(f"  • Modo: Cada timeframe processado independentemente")
        self.logger.info(f"  • Delay: 35s após fechamento (30s stream + 5s análise)")
        self.logger.info(f"  • Cronograma 1h: XX:00:35 (conforme especificação)")
        self.logger.info(f"  • Microestrutura: DESABILITADA (evita locks)")
        self.logger.info(f"  • Scheduler: {'ATIVO' if self.scheduler_enabled else 'INATIVO'}")
        # 🔧 CORREÇÃO 3: Logs do filtro candlestick
        
    def _process_1h_event(self, event):
        """Processa evento de fechamento de candle 1h (aguarda stream gravar)"""
        try:
            self.logger.info(f"🕒 EVENTO 1h: Candle {event.candle_close_time.strftime('%H:%M')} fechado + stream gravado")
            
            symbols = self.data_reader.get_valid_symbols_for_analysis()
            if not symbols:
                self.logger.warning("❌ Nenhum símbolo válido para análise 1h")
                return
            
            processed_count = 0
            signals_generated = 0
            
            for symbol in symbols:
                try:
                    result = self._analyze_single_timeframe_at_event(symbol, "1h", event)
                    processed_count += 1
                    
                    if result.get('signals_saved', 0) > 0:
                        signals_generated += result['signals_saved']
                        self.logger.info(f"✅ {symbol} 1h: {result['signals_saved']} sinais gerados")
                    
                except Exception as e:
                    self.logger.error(f"❌ Erro processando {symbol} 1h: {e}")
            
            self.logger.info(
                f"📊 EVENTO 1h CONCLUÍDO: {processed_count} símbolos processados, "
                f"{signals_generated} sinais gerados"
            )
            
        except Exception as e:
            self.logger.error(f"❌ Erro no evento 1h: {e}")
    
    
    def _analyze_single_timeframe_at_event(self, symbol: str, timeframe: str, event) -> Dict[str, Any]:
        """Análise ESPECÍFICA de um timeframe quando seu candle fecha (com dados do stream)"""
        
        start_time = time.time()
        
        # VERIFICAÇÃO INTELIGENTE de sinais bloqueadores
        if self.signal_writer.check_existing_active_signals(symbol):
            # Log mais detalhado sobre o bloqueio
            from core.signal_manager import SignalManager
            manager = SignalManager()
            blocking_info = manager.get_truly_blocking_signals(symbol)
            
            blocking_details = []
            for signal in blocking_info.get('signals', []):
                reason = signal.get('blocking_reason', 'Unknown')
                blocking_details.append(f"{signal['timeframe']} {signal['detector_name']} ({reason})")
            
            self.logger.info(f"🚫 {symbol} BLOQUEADO por: {'; '.join(blocking_details)}")
            
            return {
                'symbol': symbol, 
                'timeframe': timeframe,
                'status': 'blocked', 
                'reason': 'existing_blocking_signal',
                'blocking_details': blocking_details,
                'signals_detected': 0, 
                'signals_validated': 0, 
                'signals_saved': 0,
                'execution_time': time.time() - start_time,
                'event_time': event.trigger_time.isoformat()
            }
        
        self.logger.debug(f"🔍 {symbol} {timeframe}: Análise específica pós-stream")    
        # Busca dados ESPECÍFICOS para o timeframe (dados já gravados pelo stream)
        try:
            market_data = self.data_reader.get_latest_data(symbol, timeframe)
            self.logger.debug(f"📊 {symbol} {timeframe}: Dados pós-stream carregados")
            
            if not market_data or not market_data.is_sufficient_data:
                return {
                    'symbol': symbol, 
                    'timeframe': timeframe,
                    'status': 'insufficient_data', 
                    'signals_detected': 0, 
                    'signals_validated': 0, 
                    'signals_saved': 0,
                    'execution_time': time.time() - start_time,
                    'event_time': event.trigger_time.isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"❌ {symbol} {timeframe}: Erro nos dados - {e}")
            return {
                'symbol': symbol, 
                'timeframe': timeframe,
                'status': 'data_error', 
                'reason': str(e),
                'signals_detected': 0, 
                'signals_validated': 0, 
                'signals_saved': 0,
                'execution_time': time.time() - start_time,
                'event_time': event.trigger_time.isoformat()
            }
        
        # Análise do timeframe específico
        signals_detected = []
        try:
            tf_result = self._analyze_single_timeframe_fast(symbol, timeframe, market_data)
            signals_detected = tf_result.get('signals', [])
            
            self.logger.debug(f"🔍 {symbol} {timeframe}: {len(signals_detected)} sinais detectados")
                
        except Exception as e:
            self.logger.error(f"❌ {symbol} {timeframe}: Erro na análise - {e}")
            return {
                'symbol': symbol, 
                'timeframe': timeframe,
                'status': 'analysis_error', 
                'reason': str(e),
                'signals_detected': 0, 
                'signals_validated': 0, 
                'signals_saved': 0,
                'execution_time': time.time() - start_time,
                'event_time': event.trigger_time.isoformat()
            }
        
        # Validação SIMPLIFICADA
        validated_signals = self._simple_validation_no_locks(signals_detected, {timeframe: market_data})
        
        # Gravação
        signals_saved = 0
        if validated_signals:
            signal = validated_signals[0]
            try:
                is_valid, validation_msg = self.validate_signal_before_saving(signal, {timeframe: market_data})
                
                if not is_valid:
                    self.logger.warning(f"❌ {symbol} {timeframe}: SINAL REJEITADO - {validation_msg}")
                    return {
                        'symbol': symbol, 
                        'timeframe': timeframe,
                        'status': 'rejected', 
                        'reason': validation_msg,
                        'signals_detected': len(signals_detected), 
                        'signals_validated': len(validated_signals), 
                        'signals_saved': 0,
                        'execution_time': time.time() - start_time,
                        'event_time': event.trigger_time.isoformat()
                    }
                
                signal.status = "ACTIVE"
                
                if self.signal_writer.write_enhanced_signal(signal):
                    signals_saved = 1
                    
                    total_time = time.time() - start_time
                    score = signal.confidence * 100
                    self.logger.info(
                        f"💾 {symbol} {timeframe}: GRAVADO | {signal.detector_name} | "
                        f"Score: {score:.1f} | Entry: ${signal.entry_price:.4f} | "
                        f"Stop: ${signal.stop_loss:.4f} | T1: ${signal.targets[0]:.4f} | "
                        f"T2: ${signal.targets[1]:.4f} | {total_time:.1f}s | Pós-stream: OK"
                    )
                        
                else:
                    self.logger.warning(f"❌ {symbol} {timeframe}: FALHA NA GRAVAÇÃO")
            except Exception as e:
                self.logger.error(f"❌ {symbol} {timeframe}: Erro ao salvar - {e}")
        
        total_time = time.time() - start_time
        
        return {
            'symbol': symbol, 
            'timeframe': timeframe,
            'status': 'success', 
            'signals_detected': len(signals_detected), 
            'signals_validated': len(validated_signals), 
            'signals_saved': signals_saved,
            'execution_time': total_time,
            'processing_mode': 'timeframe_specific_with_stream',
            'event_time': event.trigger_time.isoformat(),
            'candle_close_time': event.candle_close_time.isoformat()
        }    
        
    def validate_signal_before_saving(self, signal, market_data_by_tf):
        """Valida sinal RIGOROSAMENTE antes de salvar"""
        try:
            market_data = market_data_by_tf.get(signal.timeframe)
            if not market_data or len(market_data.data) == 0:
                return False, "Sem dados para validar"
            
            # Preço atual (último candle disponível)
            current_price = float(market_data.data.iloc[-1]['close_price'])
            
            # VALIDAÇÃO 1: Divergência de preço
            price_diff_pct = abs(current_price - signal.entry_price) / signal.entry_price * 100
            if price_diff_pct > 1.0:  # Máximo 1% de divergência
                return False, f"PREÇO DIVERGIU: {price_diff_pct:.2f}% (atual: ${current_price:.4f}, entrada: ${signal.entry_price:.4f})"
            
            # VALIDAÇÃO 2: Sinal não pode estar "pré-executado"
            if signal.signal_type == 'BUY_LONG':
                # Para BUY, preço atual não pode estar acima do target 1
                if current_price >= signal.targets[0]:
                    return False, f"SINAL PRÉ-EXECUTADO: Preço atual ${current_price:.4f} >= Target1 ${signal.targets[0]:.4f}"
                # Preço atual não pode estar abaixo do stop
                if current_price <= signal.stop_loss:
                    return False, f"SINAL JÁ STOPADO: Preço atual ${current_price:.4f} <= Stop ${signal.stop_loss:.4f}"
            
            elif signal.signal_type == 'SELL_SHORT':
                # Para SELL, preço atual não pode estar abaixo do target 1
                if current_price <= signal.targets[0]:
                    return False, f"SINAL PRÉ-EXECUTADO: Preço atual ${current_price:.4f} <= Target1 ${signal.targets[0]:.4f}"
                # Preço atual não pode estar acima do stop
                if current_price >= signal.stop_loss:
                    return False, f"SINAL JÁ STOPADO: Preço atual ${current_price:.4f} >= Stop ${signal.stop_loss:.4f}"
            
            # VALIDAÇÃO 3: Timeout do sinal
            signal_age_minutes = (datetime.now() - signal.timestamp).total_seconds() / 60
            max_age = 10  # Máximo 10min para 1h
            
            if signal_age_minutes > max_age:
                return False, f"SINAL EXPIRADO: {signal_age_minutes:.1f}min > {max_age}min"
            
            return True, f"VÁLIDO (diff: {price_diff_pct:.2f}%, age: {signal_age_minutes:.1f}min)"
            
        except Exception as e:
            return False, f"Erro na validação: {e}"    
    
      
    # 🔧 CORREÇÃO 4: Adicionar método de backup
    def process_candlesticks_for_backup(self, backup_patterns: List[Dict], symbol: str, timeframe: str):
        """
        🗄️ PROCESSA BACKUP DE TODOS OS 43 CANDLESTICK PATTERNS
        """
        if not backup_patterns:
            return
        
        try:
            # Log detalhado do backup
            pattern_names = [p.get('pattern_name', 'unknown') for p in backup_patterns]
            unique_patterns = len(set(pattern_names))
            
            self.logger.debug(f"🗄️ Backup {symbol} {timeframe}: {len(backup_patterns)} detecções de {unique_patterns} patterns únicos")
            
            # Estatísticas rápidas
            high_quality = [p for p in backup_patterns if p.get('would_be_signal', False)]
            if high_quality:
                self.logger.debug(f"🎯 {len(high_quality)} patterns com qualidade para sinal")
            
            # TODO: Implementar salvamento no banco se necessário
            # Por enquanto, dados estão sendo salvos automaticamente pelo filtro
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao processar backup de patterns: {e}")
        
    def analyze_symbol_all_timeframes(self, symbol: str) -> Dict[str, Any]:
        """Análise manual de símbolo (sem scheduler) - compatibilidade"""
        
        symbol_start_time = time.time()
        
        # VERIFICAÇÃO RÁPIDA de sinais bloqueadores
        if self.signal_writer.check_existing_active_signals(symbol):
            return {
                'symbol': symbol, 
                'status': 'blocked', 
                'reason': 'existing_blocking_signal',
                'signals_detected': 0, 
                'signals_validated': 0, 
                'signals_saved': 0,
                'execution_time': time.time() - symbol_start_time,
                'mode': 'manual_analysis'
            }
        
        self.logger.info(f"🔍 {symbol}: Análise manual (todos os timeframes)")
        
        # Para análise manual, processa apenas 1h
        timeframes = ["1h"]
        all_signals = []

        # Busca dados para ambos timeframes
        market_data_by_tf = {}
        for tf in timeframes:
            try:
                market_data = self.data_reader.get_latest_data(symbol, tf)
                market_data_by_tf[tf] = market_data
                    
            except Exception as e:
                self.logger.warning(f"❌ {symbol} {tf}: Erro nos dados - {e}")
                market_data_by_tf[tf] = None

        # Análise por timeframe
        for timeframe in timeframes:
            market_data = market_data_by_tf[timeframe]
            if market_data and market_data.is_sufficient_data:
                try:
                    tf_result = self._analyze_single_timeframe_fast(symbol, timeframe, market_data)
                    tf_signals = tf_result.get('signals', [])
                    all_signals.extend(tf_signals)
                    
                    self.logger.info(f"🔍 {symbol} {timeframe}: {len(tf_signals)} sinais detectados")
                            
                except Exception as e:
                    self.logger.error(f"❌ {symbol} {timeframe}: Erro - {e}")

        # Pega apenas o melhor sinal (sem conflitos)
        if len(all_signals) > 1:
            # Prioriza por confidence
            best_signal = max(all_signals, key=lambda s: s.confidence)
            filtered_signals = [best_signal]
            self.logger.debug(f"🔧 {symbol}: {len(all_signals)} → 1 melhor sinal (conf: {best_signal.confidence:.3f})")
        else:
            filtered_signals = all_signals

        # Validação SIMPLIFICADA
        validated_signals = self._simple_validation_no_locks(filtered_signals, market_data_by_tf)

        # NOVO: Integração ML/LLM nos sinais
        enhanced_signals = []
        for signal in validated_signals:
            enhanced_signal = self._enhance_signal_with_ml_llm(signal)
            enhanced_signals.append(enhanced_signal)

        # Gravação
        signals_saved = 0
        if enhanced_signals:
            signal = enhanced_signals[0]
            try:
                # VALIDAÇÃO ANTES DE SALVAR
                is_valid, validation_msg = self.validate_signal_before_saving(signal, market_data_by_tf)
                
                if not is_valid:
                    self.logger.warning(f"❌ {symbol}: SINAL REJEITADO - {validation_msg}")
                    return {
                        'symbol': symbol, 
                        'status': 'rejected', 
                        'reason': validation_msg,
                        'signals_detected': len(all_signals), 
                        'signals_validated': len(validated_signals), 
                        'signals_saved': 0,
                        'execution_time': time.time() - symbol_start_time,
                        'mode': 'manual_analysis'
                    }
                
                signal.status = "ACTIVE"
                
                if self.signal_writer.write_enhanced_signal(signal):
                    signals_saved = 1
                    
                    total_time = time.time() - symbol_start_time
                    score = signal.confidence * 100
                    self.logger.info(
                        f"💾 {symbol}: GRAVADO | {signal.timeframe} | {signal.detector_name} | "
                        f"Score: {score:.1f} | Entry: ${signal.entry_price:.4f} | "
                        f"Stop: ${signal.stop_loss:.4f} | T1: ${signal.targets[0]:.4f} | "
                        f"T2: ${signal.targets[1]:.4f} | {validation_msg} | {total_time:.1f}s"
                    )
                        
                else:
                    self.logger.warning(f"❌ {symbol}: FALHA NA GRAVAÇÃO")
            except Exception as e:
                self.logger.error(f"❌ {symbol}: Erro ao salvar - {e}")

        total_time = time.time() - symbol_start_time
        
        return {
            'symbol': symbol, 
            'status': 'success', 
            'signals_detected': len(all_signals), 
            'signals_validated': len(validated_signals), 
            'signals_saved': signals_saved,
            'execution_time': total_time,
            'mode': 'manual_analysis'
        }
    
    def _analyze_single_timeframe_fast(self, symbol: str, timeframe: str, market_data: MarketData) -> Dict[str, Any]:
        """Análise SEMPRE usa candle fechado, validação usa candle dinâmico"""
        
        tf_config = settings.get_timeframe_config(timeframe)
        signals = []

        if len(market_data.data) < 3:  # Precisa de pelo menos 3 candles
            return {'signals': []}

        # 🔥 ANÁLISE: SEMPRE usa CANDLE FECHADO (confirmado)
        try:
            # CORREÇÃO: Valida se dados são do timeframe correto
            if len(market_data.data) < 2:
                self.logger.error(f"❌ {symbol} {timeframe}: Dados insuficientes")
                return {'signals': []}

            # Para 1h, 4h, 1d: usa último candle fechado
            analysis_candle = market_data.data.iloc[-1]
            analysis_price = float(analysis_candle['close_price'])
            analysis_timestamp = analysis_candle['timestamp'].to_pydatetime()

            # VALIDAÇÃO FLEXÍVEL: Verifica se timestamp é compatível com timeframe (tolerância de 60s)
            TIMESTAMP_TOLERANCE = 60  # segundos de tolerância
            
            if not self._validate_timestamp_flexible(analysis_timestamp, timeframe, TIMESTAMP_TOLERANCE):
                self.logger.debug(f"🔍 {symbol} {timeframe}: Timestamp rejeitado: {analysis_timestamp}")
                return {'signals': []}

            # LOG para debug
            self.logger.info(f"🔍 {symbol} {timeframe}: Entry = ${analysis_price:.4f} | Timestamp: {analysis_timestamp}")

            # Log para debug do preço
            self.logger.info(f"🔍 {symbol} {timeframe}: Analysis price = {analysis_price:.4f} | Timestamp: {analysis_timestamp}")

            # Validação com o mesmo candle
            dynamic_candle = market_data.data.iloc[-1]
            dynamic_price = float(dynamic_candle['close_price'])
            
            self.logger.debug(f"📊 {symbol} {timeframe}: Análise=${analysis_price:.4f} | Dinâmico=${dynamic_price:.4f}")
            
        except Exception as e:
            self.logger.error(f"Erro ao obter preços para {symbol} {timeframe}: {e}")
            return {'signals': []}
        
        # 🔧 CORREÇÃO 5: Definir create_signal_fast no local correto
        # 🔥 FUNÇÃO PARA CRIAR SINAIS (usa preço de análise)
        def create_signal_fast(**kwargs):
            try:
                analysis_data = market_data.data[:-1] if len(market_data.data) > 50 else market_data.data
        
                
                base_args = {
                    'symbol': symbol, 
                    'timeframe': timeframe, 
                    'entry_price': analysis_price,  # 🔥 SEMPRE preço do candle fechado
                    'timestamp': datetime.now(),      # 🔧 Data de criação do sinal
                    'entry_timestamp': analysis_timestamp,  # 🔧 NOVO: timestamp do candle
                    'market_data': analysis_data,  # 🔥 Dados SEM candle dinâmico
                    'status': 'ACTIVE',
                    'dynamic_validation_price': dynamic_price  # 🔥 NOVO: para validação
                }
                final_args = {**kwargs, **base_args}
                allowed_keys = EnhancedTradingSignal.__annotations__.keys()
                filtered_args = {k: v for k, v in final_args.items() if k in allowed_keys}
                
                signal = EnhancedTradingSignal(**filtered_args)
                
                # 🔥 VALIDAÇÃO DINÂMICA antes de adicionar
                if dynamic_price > 0 and abs(dynamic_price - analysis_price) / analysis_price < 0.05:
                    # Evita duplicatas
                    existing_detectors = [s.detector_name for s in signals]
                    if signal.detector_name not in existing_detectors:
                        signals.append(signal)
                        score = signal.confidence * 100
                        self.logger.debug(f"➕ {timeframe} {signal.detector_name} | Score: {score:.1f} | Dinâmico OK")
                else:
                    self.logger.debug(f"❌ {timeframe} {kwargs.get('detector_name', 'unknown')}: Invalidado por candle dinâmico")
                    
            except Exception as e:
                self.logger.debug(f"⏭️ Sinal descartado: {e}")

        # 🔧 CORREÇÃO 6: CANDLESTICK COM FILTRO RIGOROSO
        # CANDLESTICK SIMPLIFICADO - apenas engolfo de alta performance
        if 'candlestick' in tf_config.enabled_detectors and CANDLESTICK_AVAILABLE:
            try:
                cs_start = time.time()
                # USA dados fechados sempre
                df_for_cs = market_data.data.tail(30)
                
                self.logger.debug(f"🕯️ {symbol} {timeframe}: Usando {len(df_for_cs)} candles para patterns (otimizado)")
                
                if len(df_for_cs) >= 10:
                    # SEM FILTRO - usa detector direto
                    cs_signals_raw = generate_candlestick_signals(df_for_cs, symbol)
                    cs_time = time.time() - cs_start
                    
                    # ACEITA TODOS OS 5 PADRÕES EFETIVOS com limiares ajustados
                    quality_signals = []

                    for cs in cs_signals_raw:
                        detector_name = cs.get('detector_name', '')
                        confidence = cs.get('confidence', 0)
                        
                        # Limiares por tipo de padrão (baseado na confiabilidade)
                        if detector_name in ['Bullish_Engulfing', 'Bearish_Engulfing'] and confidence >= 0.75:
                            quality_signals.append(cs)
                            self.logger.debug(f"✅ Engulfing aceito: {detector_name} conf={confidence:.3f}")
                            
                        elif detector_name in ['Hammer', 'Shooting_Star'] and confidence >= 0.70:
                            quality_signals.append(cs)
                            self.logger.debug(f"✅ Reversal aceito: {detector_name} conf={confidence:.3f}")
                            
                        elif detector_name in ['Doji_Bullish', 'Doji_Bearish'] and confidence >= 0.60:
                            quality_signals.append(cs)
                            self.logger.debug(f"✅ Doji aceito: {detector_name} conf={confidence:.3f}")
                            
                        else:
                            self.logger.debug(f"❌ Rejeitado: {detector_name} conf={confidence:.3f} (abaixo do limiar)")

                    # Ordena por confiança e pega os melhores
                    quality_signals.sort(key=lambda x: x.get('confidence', 0), reverse=True)

                    for cs in quality_signals[:2]:  # Máximo 2 padrões
                        create_signal_fast(**cs)
                        
                    self.logger.debug(f"🕯️ Candlestick {symbol} {timeframe}: {len(quality_signals)} padrões válidos em {cs_time:.2f}s")
                                
            except Exception as e:
                self.logger.warning(f"❌ Erro candlestick {symbol} {timeframe}: {e}")
        
        # 🔧 CORREÇÃO: ADICIONAR INDICADORES TÉCNICOS
        self.logger.debug(f"🔍 {symbol} {timeframe}: Verificando detector 'technical'...")
        if 'technical' in tf_config.enabled_detectors:
            self.logger.debug(f"✅ {symbol} {timeframe}: Detector 'technical' habilitado")
            try:
                tech_start = time.time()
                technical_analyzer = self.technical_analyzers.get(timeframe)
                self.logger.debug(f"🔍 {symbol} {timeframe}: Technical analyzer: {technical_analyzer}")
                if technical_analyzer:
                    # Usa dados fechados para análise técnica
                    analysis_data = market_data.data[:-1] if len(market_data.data) > 50 else market_data.data
                    
                    # Cria MarketData com dados de análise
                    from core.data_reader import MarketData
                    analysis_market_data = MarketData(
                        symbol=symbol,
                        timeframe=timeframe,
                        data=analysis_data,
                        last_update=market_data.last_update
                    )
                    
                    # Analisa indicadores técnicos
                    self.logger.debug(f"🔍 {symbol} {timeframe}: Analisando indicadores técnicos...")
                    tech_results = technical_analyzer.analyze_all(analysis_market_data, timeframe)
                    self.logger.debug(f"🔍 {symbol} {timeframe}: Resultados técnicos: {len(tech_results)} indicadores")
                    
                    total_tech_signals = 0
                    for indicator_name, result in tech_results.items():
                        self.logger.debug(f"🔍 {symbol} {timeframe}: {indicator_name} - {len(result.signals)} sinais")
                        if result.signals:
                            for signal_data in result.signals:
                                self.logger.debug(f"🔍 {symbol} {timeframe}: Processando sinal {signal_data['type']}")
                                # Converte signal_data para EnhancedTradingSignal
                                create_signal_fast(
                                    detector_name=f"Technical_{indicator_name}",
                                    detector_type="technical",
                                    signal_type=signal_data['signal_type'],
                                    confidence=signal_data['confidence'],
                                    priority=signal_data.get('priority', 'medium'),
                                    entry_price=analysis_price,
                                    stop_loss=analysis_price * 0.98 if 'BUY' in signal_data['signal_type'] else analysis_price * 1.02,
                                    target_1=analysis_price * 1.02 if 'BUY' in signal_data['signal_type'] else analysis_price * 0.98,
                                    target_2=analysis_price * 1.04 if 'BUY' in signal_data['signal_type'] else analysis_price * 0.96,
                                    market_data=analysis_data
                                )
                                total_tech_signals += 1
                    
                    tech_time = time.time() - tech_start
                    self.logger.debug(f"📊 Technical {symbol} {timeframe}: {total_tech_signals} sinais em {tech_time:.2f}s")
                    
            except Exception as e:
                self.logger.warning(f"❌ Erro technical {symbol} {timeframe}: {e}")
        
        return {'signals': signals}
   
    def _simple_validation_no_locks(self, signals: List[EnhancedTradingSignal], market_data_by_tf: Dict) -> List[EnhancedTradingSignal]:
        """Validação SUPER SIMPLES sem microestrutura para evitar locks"""
        if not signals:
            return []

        validated_signals = []
        
        for signal in signals:
            validation_start = time.time()
            
            # VALIDAÇÃO ULTRA SIMPLIFICADA
            validation_score = 0
            max_score = 2
            
            # 1. Confidence - peso maior
            if signal.timeframe == "1h":
                min_confidence = 0.60  # Mais permissivo para 1h
            elif signal.timeframe == "4h":
                min_confidence = 0.65  # Médio para 4h
            elif signal.timeframe == "1d":
                min_confidence = 0.70  # Mais rigoroso para 1d
            else:
                min_confidence = 0.70  # Padrão
            
            if signal.confidence >= min_confidence:
                validation_score += 2
            elif signal.confidence >= min_confidence - 0.05:  # Tolerance
                validation_score += 1
            
            # Sem outras validações para evitar locks de DB
            
            validation_time = time.time() - validation_start
            
            # Decisão
            success_rate = validation_score / max_score
            required_rate = 0.5  # Bem permissivo
            
            if success_rate >= required_rate:
                signal.status = "ACTIVE"
                validated_signals.append(signal)
                score = signal.confidence * 100
                self.logger.debug(f"✅ {signal.symbol} {signal.timeframe}: Score {score:.1f} validado em {validation_time:.2f}s")
            else:
                score = signal.confidence * 100
                self.logger.debug(f"❌ {signal.symbol} {signal.timeframe}: Score {score:.1f} rejeitado")

        return validated_signals

    def run_continuous_multi_timeframe_analysis(self, base_interval: int = None):
        """Execução contínua COM SCHEDULER ESPECÍFICO (aguarda stream gravar)"""
        
        if not self.scheduler_enabled:
            self.logger.error("❌ Scheduler não disponível - modo específico indisponível")
            self.logger.info("⚠️ Execute: pip install threading (se necessário)")
            return
        
        self.logger.info("🚀 ANÁLISE CONTÍNUA COM SCHEDULER ESPECÍFICO + AGUARDA STREAM")
        self.logger.info("=" * 70)
        self.logger.info("🕒 CRONOGRAMA DE DISPAROS (pós-stream):")
        self.logger.info("   • 5m:  XX:00:35, XX:05:35, XX:10:35, XX:15:35...")
        self.logger.info("   • 15m: XX:00:35, XX:15:35, XX:30:35, XX:45:35...")
        self.logger.info("📊 CARACTERÍSTICAS:")
        self.logger.info("   • Stream grava candle em 30s")
        self.logger.info("   • Análise dispara 35s após fechamento (5s extra)")
        self.logger.info("   • Cada timeframe processado independentemente")
        self.logger.info("   • SEM conflitos entre timeframes")
        self.logger.info("   • SEM gaps de candles")
        self.logger.info("=" * 70)
        
        try:
            # Inicia scheduler
            self.scheduler.start()
            
            # Mostra próximos disparos
            status = self.scheduler.get_status()
            self.logger.info("📅 PRÓXIMOS DISPAROS (aguarda stream):")
            for tf, trigger_info in status['next_triggers'].items():
                trigger_time = trigger_info['next_trigger_time']
                time_until = trigger_info['time_until_minutes']
                candle_close = trigger_info['candle_close_time']
                
                self.logger.info(
                    f"   • {tf}: Candle fecha às {candle_close[-8:-3]}, "
                    f"análise às {trigger_time[-8:-3]} (em {time_until:.1f} min)"
                )
            
            self.logger.info(f"\n🎯 Sistema ativo - aguardando eventos (stream delay: 35s)...")
            self.logger.info("💡 Pressione Ctrl+C para parar")
            
            # Loop principal
            cycle_count = 0
            last_cleanup = time.time()
            last_status = time.time()
            
            while True:
                time.sleep(30)
                
                cycle_count += 1
                current_time = time.time()
                
                # Limpeza periódica
                if current_time - last_cleanup > 3600:
                    self._perform_quick_cleanup()
                    last_cleanup = current_time
                
                # Status periódico
                if current_time - last_status > 600:
                    scheduler_status = self.scheduler.get_status()
                    self.logger.info(f"💓 Sistema ativo - Scheduler: {scheduler_status['status']} (delay: {scheduler_status['delay_seconds']}s)")
                    last_status = current_time
                
        except KeyboardInterrupt:
            self.logger.info("\n🛑 Análise interrompida pelo usuário")
        except Exception as e:
            self.logger.error(f"❌ Erro crítico no scheduler: {e}")
            import traceback
            self.logger.error(f"Stack trace: {traceback.format_exc()}")
        finally:
            if self.scheduler:
                self.scheduler.stop()
            self.logger.info("🏁 Análise contínua finalizada")
    
    def _perform_quick_cleanup(self):
        """Limpeza rápida sem locks"""
        now = datetime.now()
        hours_since_cleanup = (now - self._last_cleanup).total_seconds() / 3600
        
        if hours_since_cleanup >= settings.system.cleanup_interval_hours:
            self.logger.info("🧹 Limpeza rápida...")
            
            try:
                cleanup_start = time.time()
                killed_count = self.signal_writer.mark_expired_signals_as_killed()
                moved_counts = self.signal_writer.move_inactive_signals_to_backup()
                total_moved = sum(moved_counts.values())
                cleanup_time = time.time() - cleanup_start
                
                if killed_count > 0 or total_moved > 0:
                    self.logger.info(f"✅ Limpeza: {killed_count} EXPIRED + {total_moved} movidos em {cleanup_time:.1f}s")
                    
                self._last_cleanup = now
                
            except Exception as e:
                self.logger.error(f"❌ Erro na limpeza: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Status com nova lógica específica por timeframe"""
        try:
            symbols = settings.get_analysis_symbols()
            enabled_timeframes = settings.get_enabled_timeframes()
            
            # Status do scheduler
            scheduler_status = {}
            if self.scheduler_enabled and self.scheduler:
                scheduler_status = self.scheduler.get_status()
                processing_mode = "timeframe_specific_with_stream_delay"
                processing_description = "Cada timeframe processado quando seu candle fecha (aguarda stream 30s)"
            else:
                processing_mode = "traditional_batch"
                processing_description = "Processamento tradicional em lotes"
            
            components = {
                'database': 'OK_NO_LOCKS',
                'technical_analyzer': 'OPTIMIZED',
                'microstructure_validation': 'DISABLED_NO_LOCKS',
                'processing_mode': processing_mode,
                'scheduler': 'ACTIVE' if self.scheduler_enabled else 'DISABLED',
                'stream_integration': 'ACTIVE_35S_DELAY' if self.scheduler_enabled else 'DISABLED',
                'signal_status_logic': 'CORRECTED_2_TARGETS',
                'new_signals_status': 'ALWAYS_ACTIVE',
                'blocking_logic': 'ACTIVE_AND_TARGET_1_HIT_ONLY',
                'real_time_monitoring': 'DISABLED_NO_LOCKS',
                'anti_lock_protection': 'ACTIVE',
                'timeframe_isolation': 'ACTIVE' if self.timeframe_specific_mode else 'DISABLED'
            }
            
            status_data = {
                'status': 'OK',
                'system_type': 'Trading Analyzer - SCHEDULER ESPECÍFICO + STREAM DELAY',
                'timestamp': datetime.now().isoformat(),
                'components': components,
                'symbols_available': len(symbols),
                'enabled_timeframes': enabled_timeframes,
                'processing_description': processing_description,
                'signal_flow': 'ACTIVE → TARGET_1_HIT → TARGET_2_HIT/STOP_HIT',
                'blocking_states': ['ACTIVE', 'TARGET_1_HIT'],
                'completed_states': ['TARGET_2_HIT', 'STOP_HIT', 'EXPIRED'],
                
                'timeframe_processing': {
                    'mode': 'stream_aware_specific_events',
                    'description': 'Aguarda stream gravar (30s) + análise específica (5s)',
                    'timing': {
                        '5m': 'XX:00:35, XX:05:35, XX:10:35, XX:15:35...',
                        '15m': 'XX:00:35, XX:15:35, XX:30:35, XX:45:35...'
                    },
                    'stream_delay': '30 segundos',
                    'analysis_delay': '5 segundos',
                    'total_delay': '35 segundos',
                    'no_gaps': True,
                    'no_conflicts': True
                },
                
                'anti_lock_settings': {
                    'microstructure_disabled': self.DISABLE_MICROSTRUCTURE,
                    'monitoring_disabled': True,
                    'max_symbol_time': self.MAX_SYMBOL_TIME,
                    'max_validation_time': self.MAX_VALIDATION_TIME,
                    'connection_optimized': True
                },
                
                'configuration': {
                    'multi_timeframe_enabled': True,
                    'timeframes_active': enabled_timeframes,
                    'single_signal_per_crypto': True,
                    'signal_status_corrected': True,
                    'targets_count': 2,
                    'new_signals_always_active': True,
                    'timeframe_specific_processing': self.timeframe_specific_mode,
                    'scheduler_enabled': self.scheduler_enabled,
                    'stream_integration': True,
                    'anti_lock_protection': True,
                    'no_database_locks': True
                }
            }
            
            # Adiciona informações do scheduler se disponível
            if self.scheduler_enabled and scheduler_status:
                status_data['scheduler_status'] = scheduler_status
            
            return status_data
            
        except Exception as e:
            return {
                'status': 'ERROR',
                'message': str(e),
                'timestamp': datetime.now().isoformat(),
                'processing_mode': 'error'
            }
    
    # Métodos de compatibilidade
    def analyze_symbol(self, symbol: str, timeframe: str = None) -> Dict[str, Any]:
        return self.analyze_symbol_all_timeframes(symbol)

    def analyze_multiple_symbols(self, symbols: List[str] = None, timeframe: str = None) -> Dict[str, Any]:
        if symbols is None:
            symbols = settings.get_analysis_symbols()
        
        results = {}
        successful_analyses = 0
        total_signals = 0
        blocked_analyses = 0
        start_time = time.time()
        
        for symbol in symbols:
            try:
                result = self.analyze_symbol_all_timeframes(symbol)
                results[symbol] = result
                if result.get('status') == 'success':
                    successful_analyses += 1
                    total_signals += result.get('signals_saved', 0)
                elif result.get('status') == 'blocked':
                    blocked_analyses += 1
            except Exception as e:
                results[symbol] = {'status': 'error', 'message': str(e)}
        
        results['_summary'] = {
            'symbols_analyzed': len(symbols),
            'successful_analyses': successful_analyses,
            'blocked_analyses': blocked_analyses,
            'total_signals_generated': total_signals,
            'total_execution_time': time.time() - start_time,
            'priority_logic': '15M_PREFERRED_5M_RIGOROUS',
            'anti_lock_protection': True,
            'signal_status_logic': 'CORRECTED - ACTIVE → TARGET_1_HIT → TARGET_2_HIT/STOP_HIT'
        }
        
        return results

    def _enhance_signal_with_ml_llm(self, signal: EnhancedTradingSignal) -> EnhancedTradingSignal:
        """
        Integra ML e LLM no sinal para gerar confiança final ponderada
        
        Args:
            signal: Sinal técnico base
            
        Returns:
            Sinal com confiança final integrada
        """
        if not (self.ml_enhancer or self.llm_enhancer):
            # Sem enhancers - retorna sinal original
            return signal
        
        original_confidence = signal.confidence
        current_confidence = original_confidence
        enhancement_log = []
        
        # 1. Aplicar ML Enhancer (se disponível)
        if self.ml_enhancer and self.ml_enhancer.enabled:
            try:
                ml_result = self.ml_enhancer.enhance_signal(
                    symbol=signal.symbol,
                    signal_type=signal.signal_type,
                    technical_confidence=current_confidence,
                    timeframe=signal.timeframe
                )
                
                if ml_result.get('ml_enabled'):
                    current_confidence = ml_result['final_confidence']
                    ml_contribution = ml_result.get('ml_contribution', 0)
                    agreement = ml_result.get('ml_agrees', False)
                    
                    enhancement_log.append(f"ML: {ml_contribution:+.3f} ({'✅' if agreement else '⚠️'})")
                    
                    # Adiciona metadata ML ao sinal
                    if not hasattr(signal, 'ml_data'):
                        signal.ml_data = {}
                    signal.ml_data.update({
                        'ml_prediction': ml_result.get('ml_prediction'),
                        'ml_confidence': ml_result.get('ml_confidence'),
                        'ml_agrees': agreement,
                        'ml_contribution': ml_contribution
                    })
                    
            except Exception as e:
                self.logger.error(f"❌ Erro no ML enhancer para {signal.symbol}: {e}")
        
        # 2. Aplicar LLM Enhancer (se disponível)
        if self.llm_enhancer and self.llm_enhancer.enabled:
            try:
                # Contexto do mercado para o LLM
                market_context = {
                    'rsi': getattr(signal, 'rsi', None),
                    'trend': signal.signal_type,
                    'timeframe': signal.timeframe
                }
                
                llm_result = self.llm_enhancer.enhance_signal(
                    symbol=signal.symbol,
                    signal_type=signal.signal_type,
                    current_confidence=current_confidence,
                    market_context=market_context
                )
                
                if llm_result.get('llm_enabled'):
                    current_confidence = llm_result['final_confidence']
                    llm_contribution = llm_result.get('llm_contribution', 0)
                    agreement = llm_result.get('llm_agrees', False)
                    
                    enhancement_log.append(f"LLM: {llm_contribution:+.3f} ({'✅' if agreement else '⚠️'})")
                    
                    # Adiciona metadata LLM ao sinal
                    if not hasattr(signal, 'llm_data'):
                        signal.llm_data = {}
                    signal.llm_data.update({
                        'sentiment_score': llm_result.get('sentiment_score'),
                        'sentiment_confidence': llm_result.get('sentiment_confidence'),
                        'llm_agrees': agreement,
                        'llm_contribution': llm_contribution,
                        'news_count': llm_result.get('news_count')
                    })
                    
            except Exception as e:
                self.logger.error(f"❌ Erro no LLM enhancer para {signal.symbol}: {e}")
        
        # 3. Atualiza confiança final do sinal
        signal.confidence = current_confidence
        
        # 4. Log da integração
        if enhancement_log:
            confidence_change = current_confidence - original_confidence
            self.logger.info(
                f"🧠 {signal.symbol} Integração: "
                f"Técnico: {original_confidence:.3f} → "
                f"Final: {current_confidence:.3f} ({confidence_change:+.3f}) | "
                f"{' | '.join(enhancement_log)}"
            )
        
        return signal

    def cleanup_old_data(self, days: int) -> Dict[str, Any]:
        """Limpeza manual"""
        try:
            moved_counts = self.signal_writer.move_inactive_signals_to_backup()
            total_moved = sum(moved_counts.values())
            
            return {
                'status': 'success',
                'removed_signals': total_moved,
                'details': moved_counts,
                'message': f'Limpeza concluída: {total_moved} sinais finalizados movidos para backup',
                'signal_status_info': 'Apenas TARGET_2_HIT, STOP_HIT, EXPIRED, MANUALLY_CLOSED são considerados finalizados',
                'priority_logic': '15M_PREFERRED_5M_RIGOROUS',
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def _validate_timestamp_flexible(self, timestamp, timeframe: str, tolerance_seconds: int = 60) -> bool:
        """Validação flexível de timestamp com tolerância"""
        try:
            now = datetime.now()
            time_diff = abs((timestamp - now).total_seconds())
            
            # Se está muito longe no tempo, rejeita
            if time_diff > tolerance_seconds:
                return False
            
            if timeframe == "1h":
                # Para 1h, aceita qualquer minuto se dentro da tolerância
                return True
            elif timeframe == "4h":
                # Para 4h, verifica se hora é múltiplo de 4 (com tolerância)
                return timestamp.hour % 4 == 0
            elif timeframe == "1d":
                # Para 1d, aceita se hora é 0 ou 23 (com tolerância)
                return timestamp.hour in [0, 23]
            
            return True
        except Exception as e:
            self.logger.warning(f"Erro na validação de timestamp: {e}")
            return True  # Em caso de erro, aceita

    def validate_signal_coherence(self, signals: List) -> List:
        """Valida coerência de sinais - evita contradições"""
        if not signals:
            return []
        
        # Agrupar por direção
        bullish_signals = [s for s in signals if hasattr(s, 'signal_type') and 'BUY' in s.signal_type]
        bearish_signals = [s for s in signals if hasattr(s, 'signal_type') and 'SELL' in s.signal_type]
        
        # Se há conflito, usar votação ponderada
        if bullish_signals and bearish_signals:
            bullish_strength = sum(getattr(s, 'confidence', 0) for s in bullish_signals)
            bearish_strength = sum(getattr(s, 'confidence', 0) for s in bearish_signals)
            
            if bullish_strength > bearish_strength * 1.2:  # 20% margem
                self.logger.info(f"✅ Sinais coerentes: Bullish escolhido (Bull={bullish_strength:.2f} vs Bear={bearish_strength:.2f})")
                return bullish_signals
            elif bearish_strength > bullish_strength * 1.2:
                self.logger.info(f"✅ Sinais coerentes: Bearish escolhido (Bear={bearish_strength:.2f} vs Bull={bullish_strength:.2f})")
                return bearish_signals
            else:
                # Conflito muito próximo - não gerar sinal
                self.logger.warning(f"🚫 Sinais contraditórios rejeitados: Bull={bullish_strength:.2f} Bear={bearish_strength:.2f}")
                return []
        
        return signals

# Alias para compatibilidade
TradingAnalyzer = MultiTimeframeAnalyzer