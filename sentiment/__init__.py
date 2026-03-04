"""Módulo de análise de sentimento (LLM e notícias)."""
from .news_fetcher import NewsFetcher, NewsItem
from .llm_analyzer import LLMSentimentAnalyzer, SentimentResult, analyze_sentiment

__all__ = ["NewsFetcher", "NewsItem", "LLMSentimentAnalyzer", "SentimentResult", "analyze_sentiment"]
