# -*- coding: utf-8 -*-
"""
LLM Sentiment Analyzer - Análise de sentimento com OpenAI (portado de sinais).
Usa variável de ambiente OPENAI_API_KEY.
"""
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

from logger import get_logger

logger = get_logger(__name__)


@dataclass
class SentimentResult:
    """Resultado de análise de sentimento."""
    symbol: str
    sentiment_score: float  # -100 a +100
    confidence: float
    reasoning: str
    news_count: int
    timestamp: datetime
    sources: List[str]
    cached: bool = False


class LLMSentimentAnalyzer:
    """Analisador de sentimento com LLM (OpenAI)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_duration_minutes: int = 60,
        model: str = "gpt-4o-mini"
    ):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI não instalado. pip install openai")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY não configurada (env ou parâmetro)")
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self._cache: Dict[str, SentimentResult] = {}
        self._cache_duration_sec = cache_duration_minutes * 60

    def analyze_symbol(
        self,
        symbol: str,
        news_items: Optional[List[str]] = None,
        market_context: Optional[Dict] = None,
        force_refresh: bool = False
    ) -> Optional[SentimentResult]:
        """Analisa sentimento para um símbolo."""
        if not force_refresh:
            cached = self._get_cached(symbol)
            if cached:
                return cached
        if news_items is None:
            news_items = self._get_mock_news(symbol)
        if not news_items:
            return None
        prompt = self._create_prompt(symbol, news_items, market_context)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a crypto market analyst. Provide concise sentiment analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=150,
            )
            content = response.choices[0].message.content
            result = self._parse_response(symbol, content, len(news_items))
            if result:
                self._set_cached(symbol, result)
            return result
        except Exception as e:
            logger.error("Erro LLM sentiment %s: %s", symbol, e)
            return None

    def _get_cached(self, symbol: str) -> Optional[SentimentResult]:
        key = f"sentiment_{symbol.upper()}"
        if key not in self._cache:
            return None
        r = self._cache[key]
        if (datetime.now() - r.timestamp).total_seconds() > self._cache_duration_sec:
            del self._cache[key]
            return None
        r.cached = True
        return r

    def _set_cached(self, symbol: str, result: SentimentResult):
        self._cache[f"sentiment_{symbol.upper()}"] = result

    def _create_prompt(self, symbol: str, news_items: List[str], market_context: Optional[Dict]) -> str:
        news_text = "\n".join([f"- {n}" for n in news_items])
        context_text = ""
        if market_context:
            if "rsi" in market_context:
                context_text += f"\n- RSI: {market_context['rsi']:.1f}"
            if "trend" in market_context:
                context_text += f"\n- Trend: {market_context['trend']}"
        return f"""Analyze {symbol} sentiment for TRADING (short-term). Score -100 (bearish) to +100 (bullish).

NEWS:
{news_text}
{context_text}

Reply with JSON only: {{"sentiment_score": number, "confidence": 0-1, "reasoning": "short text"}}"""

    def _parse_response(self, symbol: str, response: str, news_count: int) -> Optional[SentimentResult]:
        try:
            data = json.loads(response)
            score = max(-100, min(100, float(data.get("sentiment_score", 0))))
            conf = max(0, min(1, float(data.get("confidence", 0.5))))
            return SentimentResult(
                symbol=symbol,
                sentiment_score=score,
                confidence=conf,
                reasoning=str(data.get("reasoning", "")),
                news_count=news_count,
                timestamp=datetime.now(),
                sources=[],
                cached=False
            )
        except Exception as e:
            logger.warning("Parse sentiment response: %s", e)
            return None

    def _get_mock_news(self, symbol: str) -> List[str]:
        return [
            f"{symbol} market momentum update",
            f"Trading volume and technical outlook for {symbol}",
        ]


def analyze_sentiment(
    symbol: str,
    api_key: Optional[str] = None,
    force_refresh: bool = False
) -> Optional[SentimentResult]:
    """Helper para análise rápida de sentimento."""
    try:
        analyzer = LLMSentimentAnalyzer(api_key=api_key)
        return analyzer.analyze_symbol(symbol, force_refresh=force_refresh)
    except (ImportError, ValueError) as e:
        logger.warning("Sentiment não disponível: %s", e)
        return None
