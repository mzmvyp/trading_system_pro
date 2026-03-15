# -*- coding: utf-8 -*-
"""
LLM Integration - Integração do Sentiment Analysis com sistema
Adiciona score de sentimento LLM aos sinais
"""
import os
import logging
from typing import Dict, Optional

from llm.sentiment_analyzer import SentimentAnalyzer, OPENAI_AVAILABLE, SentimentResult
from llm.news_fetcher import CachedNewsFetcher


class LLMSignalEnhancer:
    """
    Adiciona scores de Sentiment LLM aos sinais
    
    Features:
    - Análise de notícias em tempo real
    - Score de sentimento de -100 a +100
    - Integração com cache agressivo
    - Weight configurável (20% padrão)
    - Graceful degradation se API falhar
    """
    
    def __init__(
        self,
        enabled: bool = True,
        llm_weight: float = 0.20,
        api_key: Optional[str] = None,
        cache_duration_minutes: int = 60,
        max_monthly_cost: float = 50.0
    ):
        self.logger = logging.getLogger(__name__)
        self.enabled = enabled and OPENAI_AVAILABLE
        self.llm_weight = llm_weight  # Peso do LLM no score final (20%)
        
        self.sentiment_analyzer: Optional[SentimentAnalyzer] = None
        self.news_fetcher: Optional[CachedNewsFetcher] = None
        
        if self.enabled:
            # CORREÇÃO: Usar API key do settings se não fornecida
            if not api_key:
                try:
                    from config.settings import settings
                    api_key = getattr(settings.llm, 'api_key', None)
                except:
                    api_key = os.getenv('OPENAI_API_KEY')
            
            if not api_key:
                self.logger.warning("⚠️ OPENAI_API_KEY não configurada - LLM desabilitado")
                self.enabled = False
            else:
                try:
                    self.sentiment_analyzer = SentimentAnalyzer(
                        api_key=api_key,
                        cache_duration_minutes=cache_duration_minutes,
                        max_cost_per_month=max_monthly_cost
                    )
                    
                    self.news_fetcher = CachedNewsFetcher(cache_duration_minutes=30)
                    
                    self.logger.info(f"✅ LLM Enhancer ativo | Peso: {llm_weight*100:.0f}% | Cache: {cache_duration_minutes}min")
                    
                except Exception as e:
                    self.logger.error(f"❌ Erro ao inicializar LLM: {e}")
                    self.enabled = False
        else:
            if not OPENAI_AVAILABLE:
                self.logger.warning("⚠️ OpenAI não disponível - LLM desabilitado")
            else:
                self.logger.info("LLM Enhancer desabilitado por configuração")
    
    def enhance_signal(
        self,
        symbol: str,
        signal_type: str,
        current_confidence: float,
        market_context: Optional[Dict] = None
    ) -> Dict:
        """
        Adiciona score LLM a um sinal
        
        Args:
            symbol: Símbolo da crypto
            signal_type: Tipo do sinal ('BUY_LONG' ou 'SELL_SHORT')
            current_confidence: Confiança atual (0-1)
            market_context: Contexto do mercado (RSI, trend, etc)
        
        Returns:
            Dict com confidence ajustada e metadata LLM
        """
        if not self.enabled:
            return {
                'final_confidence': current_confidence,
                'llm_enabled': False,
                'sentiment_score': 0,
                'sentiment_confidence': 0.0
            }
        
        try:
            # Busca notícias
            news_items = self.news_fetcher.fetch_news(symbol, hours_back=24, max_items=5)
            
            if not news_items:
                self.logger.debug(f"⚠️ Sem notícias para {symbol} - usando apenas técnico")
                return {
                    'final_confidence': current_confidence,
                    'llm_enabled': True,
                    'sentiment_score': 0,
                    'sentiment_confidence': 0.0,
                    'no_news': True
                }
            
            # Converte para strings
            news_summaries = self.news_fetcher.fetcher.get_news_summary(news_items)
            
            # Analisa sentiment
            sentiment_result = self.sentiment_analyzer.analyze_symbol(
                symbol,
                news_items=news_summaries,
                market_context=market_context
            )
            
            if not sentiment_result:
                return {
                    'final_confidence': current_confidence,
                    'llm_enabled': True,
                    'sentiment_error': 'Analysis failed'
                }
            
            # Normaliza sentiment score para [0, 1]
            # -100 a +100 → 0 a 1
            normalized_sentiment = (sentiment_result.sentiment_score + 100) / 200
            
            # Verifica concordância
            signal_bullish = 'BUY' in signal_type
            sentiment_bullish = sentiment_result.sentiment_score > 0
            
            agreement = signal_bullish == sentiment_bullish
            
            # Calcula contribuição do LLM
            if agreement:
                # LLM concorda - aumenta confiança
                llm_contribution = normalized_sentiment * sentiment_result.confidence * self.llm_weight
            else:
                # LLM discorda - reduz confiança
                llm_contribution = -(1 - normalized_sentiment) * sentiment_result.confidence * self.llm_weight
            
            # Confidence final
            final_confidence = current_confidence * (1 - self.llm_weight) + current_confidence + llm_contribution
            
            # Garante [0, 1]
            final_confidence = max(0.0, min(1.0, final_confidence))
            
            result = {
                'final_confidence': final_confidence,
                'llm_enabled': True,
                'sentiment_score': sentiment_result.sentiment_score,
                'sentiment_confidence': sentiment_result.confidence,
                'sentiment_reasoning': sentiment_result.reasoning,
                'news_count': sentiment_result.news_count,
                'llm_agrees': agreement,
                'llm_contribution': llm_contribution,
                'original_confidence': current_confidence,
                'llm_weight': self.llm_weight,
                'cached': sentiment_result.cached
            }
            
            # Log
            agreement_emoji = "✅" if agreement else "⚠️"
            cached_emoji = "💾" if sentiment_result.cached else "🆕"
            
            self.logger.info(
                f"{agreement_emoji}{cached_emoji} LLM {symbol}: {sentiment_result.sentiment_score:+.0f} "
                f"(conf: {sentiment_result.confidence:.2f}) | "
                f"Final: {final_confidence:.3f} (base: {current_confidence:.3f})"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao processar LLM para {symbol}: {e}")
            return {
                'final_confidence': current_confidence,
                'llm_enabled': True,
                'error': str(e)
            }
    
    def get_status(self) -> Dict:
        """Retorna status do LLM enhancer"""
        if not self.enabled:
            return {
                'enabled': False,
                'reason': 'Disabled or OpenAI not available'
            }
        
        sentiment_status = self.sentiment_analyzer.get_status()
        
        return {
            'enabled': True,
            'llm_weight': self.llm_weight,
            'sentiment_status': sentiment_status,
            'news_cache_stats': self.news_fetcher.cache if self.news_fetcher else {}
        }


# Função auxiliar para criar enhancer
def create_llm_enhancer(
    enabled: bool = True,
    llm_weight: float = 0.20,
    api_key: Optional[str] = None
) -> LLMSignalEnhancer:
    """
    Factory function para criar LLM enhancer
    
    Args:
        enabled: Se deve habilitar LLM
        llm_weight: Peso do LLM no score final (0-1)
        api_key: OpenAI API key
    
    Returns:
        LLMSignalEnhancer instance
    """
    return LLMSignalEnhancer(
        enabled=enabled,
        llm_weight=llm_weight,
        api_key=api_key
    )

