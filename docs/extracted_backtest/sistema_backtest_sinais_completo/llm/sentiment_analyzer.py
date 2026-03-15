# -*- coding: utf-8 -*-
"""
Sentiment Analyzer - Análise de sentimento usando OpenAI GPT-4o-mini
Analisa notícias e contexto de mercado para gerar score de sentimento
"""
import os
import json
import logging
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Dicionário de sentiment keywords para crypto
CRYPTO_SENTIMENT_KEYWORDS = {
    'BULLISH_STRONG': [
        'breakout', 'moon', 'pump', 'surge', 'rally', 'ATH', 'all-time high',
        'institutional adoption', 'ETF approved', 'major partnership'
    ],
    'BULLISH_MODERATE': [
        'support', 'bounce', 'recovery', 'accumulation', 'bullish divergence',
        'golden cross', 'uptick', 'positive momentum'
    ],
    'BEARISH_STRONG': [
        'crash', 'dump', 'plunge', 'collapse', 'bear market', 'FUD',
        'hack', 'exploit', 'regulatory crackdown', 'delisting'
    ],
    'BEARISH_MODERATE': [
        'resistance', 'rejection', 'correction', 'pullback', 'death cross',
        'overbought', 'distribution', 'negative divergence'
    ]
}


@dataclass
class SentimentResult:
    """Resultado de análise de sentimento"""
    symbol: str
    sentiment_score: float  # -100 (muito bearish) a +100 (muito bullish)
    confidence: float  # 0 a 1
    reasoning: str
    news_count: int
    timestamp: datetime
    sources: List[str]
    cached: bool = False


class SentimentCache:
    """Cache agressivo para economizar API calls"""
    
    def __init__(self, cache_duration_minutes: int = 60):
        self.cache: Dict[str, SentimentResult] = {}
        self.cache_duration = timedelta(minutes=cache_duration_minutes)
        self.logger = logging.getLogger(__name__)
    
    def _get_cache_key(self, symbol: str) -> str:
        """Gera chave de cache"""
        return f"sentiment_{symbol.upper()}"
    
    def get(self, symbol: str) -> Optional[SentimentResult]:
        """Busca do cache se não expirou"""
        key = self._get_cache_key(symbol)
        
        if key not in self.cache:
            return None
        
        result = self.cache[key]
        age = datetime.now() - result.timestamp
        
        if age > self.cache_duration:
            # Cache expirado
            del self.cache[key]
            return None
        
        self.logger.debug(f"💾 Cache hit: {symbol} (age: {age.seconds // 60}min)")
        result.cached = True
        return result
    
    def set(self, symbol: str, result: SentimentResult):
        """Salva no cache"""
        key = self._get_cache_key(symbol)
        self.cache[key] = result
        self.logger.debug(f"💾 Cached: {symbol}")
    
    def clear(self):
        """Limpa cache"""
        self.cache.clear()
    
    def get_stats(self) -> Dict:
        """Estatísticas do cache"""
        return {
            'cached_symbols': len(self.cache),
            'symbols': list(self.cache.keys())
        }


class SentimentAnalyzer:
    """
    Analisador de Sentimento com GPT-4o-mini
    
    Features:
    - Análise de notícias em tempo real
    - Cache agressivo (1h padrão)
    - Controle de custo (limite mensal)
    - Score estruturado (-100 a +100)
    - Contexto de mercado
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_duration_minutes: int = 60,
        max_cost_per_month: float = 50.0,
        model: str = "gpt-4o-mini"
    ):
        self.logger = logging.getLogger(__name__)
        
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI não instalado. Execute: pip install openai")
        
        # API Key
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY não configurada (env ou parâmetro)")
        
        openai.api_key = self.api_key
        
        # Configurações
        self.model = model
        self.max_cost_per_month = max_cost_per_month
        self.current_month_cost = 0.0
        
        # Cache
        self.cache = SentimentCache(cache_duration_minutes)
        
        # Custo estimado por request (GPT-4o-mini é muito barato)
        self.cost_per_request = 0.002  # ~$0.002 por análise
        
        self.logger.info(f"✅ Sentiment Analyzer ativo | Model: {model} | Cache: {cache_duration_minutes}min")
    
    def analyze_symbol(
        self,
        symbol: str,
        news_items: Optional[List[str]] = None,
        market_context: Optional[Dict] = None,
        force_refresh: bool = False
    ) -> Optional[SentimentResult]:
        """
        Analisa sentimento para um símbolo
        
        Args:
            symbol: Símbolo da crypto
            news_items: Lista de notícias (opcional, usa mock se None)
            market_context: Contexto adicional do mercado
            force_refresh: Força nova análise ignorando cache
        
        Returns:
            SentimentResult ou None se erro
        """
        # Verifica cache
        if not force_refresh:
            cached = self.cache.get(symbol)
            if cached:
                return cached
        
        # Verifica limite de custo
        if self.current_month_cost >= self.max_cost_per_month:
            self.logger.warning(
                f"⚠️ Limite de custo atingido: ${self.current_month_cost:.2f} / ${self.max_cost_per_month:.2f}"
            )
            return None
        
        try:
            # Prepara contexto
            if news_items is None:
                # Mock de notícias para desenvolvimento
                news_items = self._get_mock_news(symbol)
            
            if not news_items:
                self.logger.warning(f"⚠️ Sem notícias para {symbol}")
                return None
            
            # Cria prompt
            prompt = self._create_sentiment_prompt(symbol, news_items, market_context)
            
            # Chama GPT
            start_time = time.time()
            response = self._call_gpt(prompt)
            elapsed = time.time() - start_time
            
            if response is None:
                return None
            
            # Atualiza custo
            self.current_month_cost += self.cost_per_request
            
            # Parse resposta
            result = self._parse_gpt_response(symbol, response, news_items)
            
            if result:
                # Salva no cache
                self.cache.set(symbol, result)
                
                self.logger.info(
                    f"🤖 LLM Sentiment {symbol}: {result.sentiment_score:+.1f} "
                    f"(conf: {result.confidence:.2f}) | "
                    f"Cost: ${self.cost_per_request:.4f} | Time: {elapsed:.1f}s"
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Erro na análise de sentimento para {symbol}: {e}")
            return None
    
    def _create_sentiment_prompt(
        self,
        symbol: str,
        news_items: List[str],
        market_context: Optional[Dict]
    ) -> str:
        """Cria prompt estruturado para GPT"""
        
        # Concatena notícias
        news_text = "\n".join([f"- {news}" for news in news_items])
        
        # Adiciona contexto
        context_text = ""
        if market_context:
            if 'rsi' in market_context:
                context_text += f"\n- RSI: {market_context['rsi']:.1f}"
            if 'trend' in market_context:
                context_text += f"\n- Trend: {market_context['trend']}"
            if 'volume_profile' in market_context:
                context_text += f"\n- Volume: {market_context['volume_profile']}"
        
        prompt = f"""You are a professional cryptocurrency trading analyst specializing in sentiment analysis for SHORT-TERM price movements.

**TASK**: Analyze {symbol} sentiment for TRADING SIGNALS (not investment advice).

**RECENT NEWS** (ordered by relevance):
{news_text}

**TECHNICAL CONTEXT**:
{context_text if context_text else "- No additional context"}

**SENTIMENT SCORING RULES**:
1. Score from -100 (extremely bearish SHORT signal) to +100 (extremely bullish LONG signal)
2. Consider ONLY short-term impact (next 1-4 hours)
3. Weight factors:
   - Breaking news: 40%
   - Market sentiment shifts: 30%
   - Technical alignment: 20%
   - Social sentiment: 10%

**CONFIDENCE SCORING**:
- High (0.8-1.0): Multiple strong signals align
- Medium (0.5-0.7): Mixed signals, some uncertainty
- Low (0.0-0.4): Conflicting or weak signals

**OUTPUT FORMAT (JSON)**:
{{
  "sentiment_score": <-100 to 100>,
  "confidence": <0.0 to 1.0>,
  "reasoning": "<max 50 words explaining the KEY driver>",
  "primary_driver": "<news|technical|social|regulatory>",
  "time_horizon": "<1h|4h|1d>"
}}

Focus on ACTIONABLE trading sentiment, not long-term investment outlook."""
        
        return prompt
    
    def analyze_keyword_sentiment(self, text: str) -> tuple[float, float]:
        """Analisa sentiment baseado em keywords específicas de crypto"""
        text_lower = text.lower()
        scores = {
            'BULLISH_STRONG': 0,
            'BULLISH_MODERATE': 0,
            'BEARISH_STRONG': 0,
            'BEARISH_MODERATE': 0
        }
        
        for category, keywords in CRYPTO_SENTIMENT_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    scores[category] += 1
        
        # Calcular score ponderado
        total_score = (
            scores['BULLISH_STRONG'] * 100 +
            scores['BULLISH_MODERATE'] * 50 -
            scores['BEARISH_MODERATE'] * 50 -
            scores['BEARISH_STRONG'] * 100
        )
        
        total_keywords = sum(scores.values())
        if total_keywords > 0:
            normalized_score = total_score / total_keywords
            confidence = min(1.0, total_keywords / 10)  # Mais keywords = mais confiança
            return normalized_score, confidence
        
        return 0, 0
    
    def _call_gpt(self, prompt: str) -> Optional[str]:
        """Chama API do OpenAI"""
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional cryptocurrency market analyst. Provide concise, objective sentiment analysis."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,  # Mais determinístico
                max_tokens=150,   # Limita tokens = reduz custo
                response_format={"type": "json_object"}
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"❌ Erro na chamada GPT: {e}")
            return None
    
    def _parse_gpt_response(
        self,
        symbol: str,
        response: str,
        news_items: List[str]
    ) -> Optional[SentimentResult]:
        """Parse resposta JSON do GPT"""
        try:
            data = json.loads(response)
            
            sentiment_score = float(data.get('sentiment_score', 0))
            confidence = float(data.get('confidence', 0.5))
            reasoning = str(data.get('reasoning', 'No reasoning provided'))
            
            # Valida ranges
            sentiment_score = max(-100, min(100, sentiment_score))
            confidence = max(0, min(1, confidence))
            
            return SentimentResult(
                symbol=symbol,
                sentiment_score=sentiment_score,
                confidence=confidence,
                reasoning=reasoning,
                news_count=len(news_items),
                timestamp=datetime.now(),
                sources=[],  # TODO: adicionar fontes reais
                cached=False
            )
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao parsear resposta GPT: {e}")
            return None
    
    def _get_mock_news(self, symbol: str) -> List[str]:
        """Notícias mock para desenvolvimento (será substituído por news_fetcher)"""
        mock_news = {
            'BTC': [
                "Bitcoin ETF sees record inflows of $500M",
                "Major institution announces Bitcoin treasury strategy",
                "Technical analysis suggests bullish breakout pattern"
            ],
            'ETH': [
                "Ethereum upgrade successfully deployed on testnet",
                "DeFi Total Value Locked reaches new all-time high",
                "Major protocol announces migration to Ethereum"
            ],
            'BNB': [
                "Binance announces new product launch",
                "BNB Chain sees surge in daily transactions",
                "New partnerships announced for BNB ecosystem"
            ]
        }
        
        # Retorna notícias mock ou genéricas
        return mock_news.get(symbol, [
            f"{symbol} showing strong market momentum",
            f"Trading volume for {symbol} increases significantly",
            f"Technical indicators suggest positive outlook for {symbol}"
        ])
    
    def analyze_batch(
        self,
        symbols: List[str],
        market_contexts: Optional[Dict[str, Dict]] = None
    ) -> Dict[str, SentimentResult]:
        """
        Analisa múltiplos símbolos
        
        Args:
            symbols: Lista de símbolos
            market_contexts: Dict opcional {symbol: context}
        
        Returns:
            Dict {symbol: SentimentResult}
        """
        results = {}
        
        for symbol in symbols:
            context = market_contexts.get(symbol) if market_contexts else None
            
            result = self.analyze_symbol(symbol, market_context=context)
            
            if result:
                results[symbol] = result
            
            # Pequeno delay para evitar rate limit (não necessário com cache)
            time.sleep(0.1)
        
        return results
    
    def get_status(self) -> Dict:
        """Status do sentiment analyzer"""
        return {
            'enabled': True,
            'model': self.model,
            'current_month_cost': self.current_month_cost,
            'max_monthly_cost': self.max_cost_per_month,
            'cost_remaining': self.max_cost_per_month - self.current_month_cost,
            'cache_stats': self.cache.get_stats(),
            'cost_per_request': self.cost_per_request
        }
    
    def reset_monthly_cost(self):
        """Reseta custo mensal (chamar no início do mês)"""
        self.logger.info(f"💰 Resetando custo mensal de ${self.current_month_cost:.2f} para $0.00")
        self.current_month_cost = 0.0


# Função auxiliar para uso direto
def analyze_sentiment(
    symbol: str,
    api_key: Optional[str] = None,
    force_refresh: bool = False
) -> Optional[SentimentResult]:
    """
    Função helper para análise rápida de sentimento
    
    Args:
        symbol: Símbolo da crypto
        api_key: OpenAI API key (ou usa env)
        force_refresh: Ignora cache
    
    Returns:
        SentimentResult ou None
    """
    analyzer = SentimentAnalyzer(api_key=api_key)
    return analyzer.analyze_symbol(symbol, force_refresh=force_refresh)

