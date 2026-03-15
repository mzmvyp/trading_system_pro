# -*- coding: utf-8 -*-
"""
News Fetcher - Coleta notícias de crypto em tempo real
Suporta múltiplas fontes (CoinDesk, CoinTelegraph, Reddit, Twitter)
"""
import logging
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class NewsItem:
    """Item de notícia"""
    title: str
    source: str
    url: str
    published_at: datetime
    sentiment_hint: Optional[str] = None  # 'positive', 'negative', 'neutral'


class NewsFetcher:
    """
    Busca notícias de crypto de múltiplas fontes
    
    Fontes suportadas:
    - CryptoCompare API (gratuita)
    - RSS feeds públicos
    - Twitter API (opcional)
    
    TODO: Implementar scraping de CoinDesk, CoinTelegraph
    """
    
    def __init__(self, cryptocompare_api_key: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.cryptocompare_api_key = cryptocompare_api_key
        
        # URLs base
        self.cryptocompare_base = "https://min-api.cryptocompare.com/data/v2/news/"
        
        self.logger.info("✅ News Fetcher inicializado")
    
    def fetch_news(
        self,
        symbol: str,
        hours_back: int = 24,
        max_items: int = 10
    ) -> List[NewsItem]:
        """
        Busca notícias recentes para um símbolo
        
        Args:
            symbol: Símbolo da crypto (BTC, ETH, etc)
            hours_back: Horas atrás para buscar
            max_items: Máximo de notícias
        
        Returns:
            Lista de NewsItem
        """
        news_items = []
        
        # Tenta CryptoCompare
        cc_news = self._fetch_cryptocompare(symbol, max_items)
        news_items.extend(cc_news)
        
        # TODO: Adicionar outras fontes aqui
        # rss_news = self._fetch_rss_feeds(symbol)
        # news_items.extend(rss_news)
        
        # Filtra por data
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        news_items = [n for n in news_items if n.published_at >= cutoff_time]
        
        # Limita quantidade
        news_items = news_items[:max_items]
        
        self.logger.info(f"📰 Fetched {len(news_items)} news for {symbol}")
        
        return news_items
    
    def _fetch_cryptocompare(self, symbol: str, max_items: int) -> List[NewsItem]:
        """Busca notícias do CryptoCompare"""
        try:
            params = {
                'categories': symbol.upper(),
                'lang': 'EN'
            }
            
            if self.cryptocompare_api_key:
                params['api_key'] = self.cryptocompare_api_key
            
            response = requests.get(
                f"{self.cryptocompare_base}",
                params=params,
                timeout=10
            )
            
            if response.status_code != 200:
                self.logger.warning(f"⚠️ CryptoCompare error: {response.status_code}")
                return []
            
            data = response.json()
            
            if not data.get('Data'):
                return []
            
            news_items = []
            
            for item in data['Data'][:max_items]:
                try:
                    news_items.append(NewsItem(
                        title=item.get('title', ''),
                        source=item.get('source', 'CryptoCompare'),
                        url=item.get('url', ''),
                        published_at=datetime.fromtimestamp(item.get('published_on', 0))
                    ))
                except Exception as e:
                    self.logger.debug(f"Erro ao parsear notícia: {e}")
                    continue
            
            return news_items
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao buscar CryptoCompare: {e}")
            return []
    
    def _fetch_mock_news(self, symbol: str) -> List[NewsItem]:
        """Mock de notícias para desenvolvimento"""
        now = datetime.now()
        
        mock_data = {
            'BTC': [
                ("Bitcoin reaches new milestone in institutional adoption", "positive"),
                ("Major exchange reports record BTC trading volume", "positive"),
                ("Analysts predict strong Q4 for Bitcoin", "positive")
            ],
            'ETH': [
                ("Ethereum network upgrade successful", "positive"),
                ("DeFi protocols on Ethereum reach $100B TVL", "positive"),
                ("New scaling solution launches on Ethereum", "positive")
            ],
            'BNB': [
                ("Binance announces new feature for BNB holders", "positive"),
                ("BNB Chain sees surge in daily active users", "positive"),
                ("New DeFi projects choose BNB Chain", "positive")
            ]
        }
        
        templates = mock_data.get(symbol, [
            (f"{symbol} shows strong performance in recent trading", "positive"),
            (f"Market sentiment improves for {symbol}", "positive"),
            (f"Technical analysis bullish for {symbol}", "positive")
        ])
        
        return [
            NewsItem(
                title=title,
                source="MockNews",
                url=f"https://example.com/news/{symbol.lower()}/{i}",
                published_at=now - timedelta(hours=i*2),
                sentiment_hint=hint
            )
            for i, (title, hint) in enumerate(templates)
        ]
    
    def get_news_summary(self, news_items: List[NewsItem]) -> List[str]:
        """
        Converte lista de NewsItem em strings para LLM
        
        Args:
            news_items: Lista de NewsItem
        
        Returns:
            Lista de strings formatadas
        """
        summaries = []
        
        for item in news_items:
            # Calcula idade da notícia
            age = datetime.now() - item.published_at
            hours_ago = int(age.total_seconds() / 3600)
            
            # Formato: "[2h ago] Title (Source)"
            summary = f"[{hours_ago}h ago] {item.title} ({item.source})"
            summaries.append(summary)
        
        return summaries


class CachedNewsFetcher:
    """
    News Fetcher com cache para reduzir chamadas
    """
    
    def __init__(self, cache_duration_minutes: int = 30):
        self.fetcher = NewsFetcher()
        self.cache: Dict[str, tuple] = {}  # {symbol: (news, timestamp)}
        self.cache_duration = timedelta(minutes=cache_duration_minutes)
        self.logger = logging.getLogger(__name__)
    
    def fetch_news(
        self,
        symbol: str,
        hours_back: int = 24,
        max_items: int = 10,
        use_cache: bool = True
    ) -> List[NewsItem]:
        """
        Busca notícias com cache
        
        Args:
            symbol: Símbolo
            hours_back: Horas atrás
            max_items: Máximo de items
            use_cache: Se deve usar cache
        
        Returns:
            Lista de NewsItem
        """
        # Verifica cache
        if use_cache and symbol in self.cache:
            cached_news, cached_time = self.cache[symbol]
            age = datetime.now() - cached_time
            
            if age < self.cache_duration:
                self.logger.debug(f"💾 Cache hit: {symbol} (age: {age.seconds//60}min)")
                return cached_news
        
        # Busca novas notícias
        news_items = self.fetcher.fetch_news(symbol, hours_back, max_items)
        
        # Atualiza cache
        self.cache[symbol] = (news_items, datetime.now())
        
        return news_items
    
    def clear_cache(self):
        """Limpa cache"""
        self.cache.clear()
        self.logger.info("🗑️ News cache cleared")


# Função auxiliar para uso direto
def fetch_crypto_news(
    symbol: str,
    hours_back: int = 24,
    max_items: int = 10,
    use_mock: bool = False
) -> List[str]:
    """
    Função helper para buscar notícias
    
    Args:
        symbol: Símbolo da crypto
        hours_back: Horas atrás
        max_items: Máximo de items
        use_mock: Se deve usar notícias mock
    
    Returns:
        Lista de strings formatadas
    """
    fetcher = NewsFetcher()
    
    if use_mock:
        news_items = fetcher._fetch_mock_news(symbol)
    else:
        news_items = fetcher.fetch_news(symbol, hours_back, max_items)
    
    return fetcher.get_news_summary(news_items)

