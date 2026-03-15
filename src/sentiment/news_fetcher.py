# -*- coding: utf-8 -*-
"""
News Fetcher - Coleta notícias de crypto (portado de sinais).
Suporta CryptoCompare e mock para desenvolvimento.
"""
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import requests

from src.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class NewsItem:
    """Item de notícia."""
    title: str
    source: str
    url: str
    published_at: datetime
    sentiment_hint: Optional[str] = None


class NewsFetcher:
    """Busca notícias de crypto (CryptoCompare e mock)."""

    def __init__(self, cryptocompare_api_key: Optional[str] = None):
        self.cryptocompare_api_key = cryptocompare_api_key or os.getenv("CRYPTOCOMPARE_API_KEY")
        self.cryptocompare_base = "https://min-api.cryptocompare.com/data/v2/news/"

    def fetch_news(
        self,
        symbol: str,
        hours_back: int = 24,
        max_items: int = 10
    ) -> List[NewsItem]:
        """Busca notícias recentes para um símbolo."""
        news_items = []
        try:
            cc_news = self._fetch_cryptocompare(symbol, max_items)
            news_items.extend(cc_news)
        except Exception as e:
            logger.debug("CryptoCompare falhou, usando mock: %s", e)
        if not news_items:
            news_items = self._fetch_mock_news(symbol)
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)
        news_items = [n for n in news_items if n.published_at >= cutoff_time]
        return news_items[:max_items]

    def _fetch_cryptocompare(self, symbol: str, max_items: int) -> List[NewsItem]:
        try:
            params = {"categories": symbol.upper(), "lang": "EN"}
            if self.cryptocompare_api_key:
                params["api_key"] = self.cryptocompare_api_key
            response = requests.get(self.cryptocompare_base, params=params, timeout=10)
            if response.status_code != 200:
                return []
            data = response.json()
            if not data.get("Data"):
                return []
            items = []
            for item in data["Data"][:max_items]:
                try:
                    items.append(NewsItem(
                        title=item.get("title", ""),
                        source=item.get("source", "CryptoCompare"),
                        url=item.get("url", ""),
                        published_at=datetime.fromtimestamp(item.get("published_on", 0))
                    ))
                except Exception:
                    continue
            return items
        except Exception as e:
            logger.warning("Erro CryptoCompare: %s", e)
            return []

    def _fetch_mock_news(self, symbol: str) -> List[NewsItem]:
        now = datetime.now(timezone.utc)
        mock = {
            "BTC": [("Bitcoin institutional adoption milestone", "positive"), ("Record BTC volume", "positive")],
            "ETH": [("Ethereum upgrade successful", "positive"), ("DeFi TVL growth", "positive")],
        }
        templates = mock.get(symbol, [(f"{symbol} market update", "neutral")])
        return [
            NewsItem(title=t, source="Mock", url="", published_at=now - timedelta(hours=i * 2), sentiment_hint=h)
            for i, (t, h) in enumerate(templates)
        ]

    def get_news_summary(self, news_items: List[NewsItem]) -> List[str]:
        """Converte NewsItems em strings para o LLM."""
        summaries = []
        for item in news_items:
            age = datetime.now(timezone.utc) - item.published_at
            hours_ago = int(age.total_seconds() / 3600)
            summaries.append(f"[{hours_ago}h ago] {item.title} ({item.source})")
        return summaries
