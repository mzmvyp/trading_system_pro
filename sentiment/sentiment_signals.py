# -*- coding: utf-8 -*-
"""
Integra sentimento como feature/sinal para os modelos existentes.
Pode ser usado como input adicional no ML ou no agente AGNO.
"""
from typing import Dict, Optional
from logger import get_logger

logger = get_logger(__name__)


def get_sentiment_feature(symbol: str, sentiment_result: Optional[object]) -> float:
    """
    Retorna score de sentimento normalizado para uso como feature (0-1 ou -1 a 1).
    sentiment_result: SentimentResult do llm_analyzer (ou None).
    """
    if sentiment_result is None:
        return 0.0
    score = getattr(sentiment_result, "sentiment_score", 0)
    return max(-1.0, min(1.0, score / 100.0))
