# -*- coding: utf-8 -*-
"""
ML Integration - Integração do XGBoost com sistema de análise
Adiciona score ML aos sinais existentes
"""
import logging
from typing import Dict, Optional
from datetime import datetime

from ml.optimized_xgboost_predictor import OptimizedXGBoostPredictor, XGBOOST_AVAILABLE


class MLSignalEnhancer:
    """
    Adiciona scores de Machine Learning aos sinais
    
    Features:
    - Predição de movimento futuro
    - Score de confiança ML
    - Integração transparente (graceful degradation)
    - Weight configurável no score final
    """
    
    def __init__(self, enabled: bool = True, ml_weight: float = 0.25):
        self.logger = logging.getLogger(__name__)
        self.enabled = enabled and XGBOOST_AVAILABLE
        self.ml_weight = ml_weight  # Peso do ML no score final (25%)
        
        self.predictor: Optional[OptimizedXGBoostPredictor] = None
        
        if self.enabled:
            try:
                self.predictor = OptimizedXGBoostPredictor()
                
                if self.predictor.model is None:
                    self.logger.warning("⚠️ Modelo ML não treinado - ML desabilitado")
                    self.enabled = False
                else:
                    model_info = self.predictor.get_model_info()
                    self.logger.info(f"✅ ML Enhancer ativo | Peso: {ml_weight*100:.0f}% | Acc: {model_info['metrics'].get('accuracy', 0):.3f}")
                    
                    if self.predictor.should_retrain():
                        self.logger.warning("⚠️ Modelo precisa ser retreinado (>7 dias)")
            except Exception as e:
                self.logger.error(f"❌ Erro ao inicializar ML: {e}")
                self.enabled = False
        else:
            if not XGBOOST_AVAILABLE:
                self.logger.warning("⚠️ XGBoost não disponível - ML desabilitado")
            else:
                self.logger.info("ML Enhancer desabilitado por configuração")
    
    def enhance_signal(
        self,
        symbol: str,
        signal_type: str,
        technical_confidence: float,
        timeframe: str = "5m"
    ) -> Dict:
        """
        Adiciona score ML a um sinal
        
        Args:
            symbol: Símbolo da crypto
            signal_type: Tipo do sinal ('BUY_LONG' ou 'SELL_SHORT')
            technical_confidence: Confiança da análise técnica (0-1)
            timeframe: Timeframe
        
        Returns:
            Dict com confidence ajustada e metadata ML
        """
        if not self.enabled:
            return {
                'final_confidence': technical_confidence,
                'ml_enabled': False,
                'ml_prediction': None,
                'ml_confidence': 0.0,
                'ml_contribution': 0.0
            }
        
        try:
            # Faz predição ML
            ml_result = self.predictor.predict(symbol, timeframe)
            
            if ml_result is None:
                # ML falhou - usa apenas técnico
                return {
                    'final_confidence': technical_confidence,
                    'ml_enabled': True,
                    'ml_prediction': None,
                    'ml_confidence': 0.0,
                    'ml_contribution': 0.0,
                    'ml_error': 'Prediction failed'
                }
            
            # Verifica concordância entre técnico e ML
            ml_bullish = ml_result['prediction'] == 'BULLISH'
            signal_bullish = 'BUY' in signal_type
            
            agreement = ml_bullish == signal_bullish
            
            if agreement:
                # ML concorda - aumenta confiança
                ml_contribution = ml_result['confidence'] * self.ml_weight
                final_confidence = technical_confidence * (1 - self.ml_weight) + ml_contribution
            else:
                # ML discorda - reduz confiança
                ml_contribution = -ml_result['confidence'] * self.ml_weight
                final_confidence = technical_confidence * (1 - self.ml_weight) + (1 - ml_result['confidence']) * self.ml_weight
            
            # Garante que confidence está em [0, 1]
            final_confidence = max(0.0, min(1.0, final_confidence))
            
            result = {
                'final_confidence': final_confidence,
                'ml_enabled': True,
                'ml_prediction': ml_result['prediction'],
                'ml_confidence': ml_result['confidence'],
                'ml_probability_up': ml_result['probability_up'],
                'ml_probability_down': ml_result['probability_down'],
                'ml_agrees': agreement,
                'ml_contribution': ml_contribution,
                'technical_confidence': technical_confidence,
                'ml_weight': self.ml_weight
            }
            
            # Log
            agreement_emoji = "✅" if agreement else "⚠️"
            self.logger.debug(
                f"{agreement_emoji} ML {symbol}: {ml_result['prediction']} "
                f"(conf: {ml_result['confidence']:.3f}) | "
                f"Final: {final_confidence:.3f} (tech: {technical_confidence:.3f})"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao processar ML para {symbol}: {e}")
            return {
                'final_confidence': technical_confidence,
                'ml_enabled': True,
                'ml_error': str(e)
            }
    
    def get_status(self) -> Dict:
        """Retorna status do ML enhancer"""
        if not self.enabled:
            return {
                'enabled': False,
                'reason': 'Disabled or XGBoost not available'
            }
        
        model_info = self.predictor.get_model_info()
        
        return {
            'enabled': True,
            'ml_weight': self.ml_weight,
            'model_info': model_info,
            'needs_retrain': model_info.get('needs_retrain', False)
        }


# Função auxiliar para uso no analyzer
def create_ml_enhancer(enabled: bool = True, ml_weight: float = 0.25) -> MLSignalEnhancer:
    """
    Factory function para criar ML enhancer
    
    Args:
        enabled: Se deve habilitar ML
        ml_weight: Peso do ML no score final (0-1)
    
    Returns:
        MLSignalEnhancer instance
    """
    return MLSignalEnhancer(enabled=enabled, ml_weight=ml_weight)

