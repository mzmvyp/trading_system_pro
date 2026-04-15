"""
Twitter/X Sentiment Analysis via Grok (xAI API).

Usa o Grok para buscar e analisar sentimento em tempo real do Twitter/X
sobre criptomoedas específicas. O Grok tem acesso nativo ao X.

Resultados são cacheados por 30 minutos para evitar chamadas excessivas.
"""
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import aiohttp

from src.core.logger import get_logger

logger = get_logger(__name__)

# Cache em memória: {symbol: {"data": {...}, "timestamp": float}}
_sentiment_cache: Dict[str, Dict] = {}
CACHE_TTL_SECONDS = 1800  # 30 minutos

XAI_API_URL = "https://api.x.ai/v1/chat/completions"


def _get_api_key() -> Optional[str]:
    return os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")


def _is_cached(symbol: str) -> bool:
    if symbol not in _sentiment_cache:
        return False
    age = time.time() - _sentiment_cache[symbol]["timestamp"]
    return age < CACHE_TTL_SECONDS


async def analyze_twitter_sentiment(symbol: str) -> Dict[str, Any]:
    """
    Analisa sentimento do Twitter/X para um símbolo crypto via Grok.

    Args:
        symbol: Par de trading (ex: BTCUSDT, ETHUSDT)

    Returns:
        Dict com sentiment, score (-10 a +10), keywords, e detalhes
    """
    api_key = _get_api_key()
    if not api_key:
        return {
            "available": False,
            "reason": "XAI_API_KEY não configurada",
            "sentiment": "neutral",
            "score": 0,
        }

    # Retornar cache se válido
    if _is_cached(symbol):
        cached = _sentiment_cache[symbol]["data"]
        cached["from_cache"] = True
        return cached

    # Extrair nome do token (BTCUSDT → BTC)
    token = symbol.replace("USDT", "").replace("BUSD", "").replace("USDC", "")

    prompt = (
        f"Search recent tweets (last 4 hours) about ${token} cryptocurrency. "
        f"Analyze the overall sentiment. Reply ONLY with this JSON, no extra text:\n"
        f'{{"sentiment": "bullish/bearish/neutral", '
        f'"score": <-10 to +10>, '
        f'"confidence": <0.0 to 1.0>, '
        f'"tweet_volume": "high/medium/low", '
        f'"key_topics": ["topic1", "topic2"], '
        f'"notable_accounts": ["@account if any influential"], '
        f'"summary": "1-2 sentence summary of the mood"}}'
    )

    try:
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": "grok-3-mini",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a crypto Twitter sentiment analyzer. "
                            "Search X/Twitter for recent posts about the given token "
                            "and provide structured sentiment analysis. "
                            "Reply ONLY with valid JSON, no markdown, no explanation."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
                "max_tokens": 300,
            }

            async with session.post(
                XAI_API_URL, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.warning(f"[GROK] API error {resp.status}: {error_text[:200]}")
                    return {
                        "available": False,
                        "reason": f"API error {resp.status}",
                        "sentiment": "neutral",
                        "score": 0,
                    }

                data = await resp.json()

        # Extrair resposta
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

        # Limpar markdown se Grok envolver em ```json
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[-1]
        if content.endswith("```"):
            content = content.rsplit("```", 1)[0]
        content = content.strip()

        parsed = json.loads(content)

        result = {
            "available": True,
            "symbol": symbol,
            "token": token,
            "sentiment": parsed.get("sentiment", "neutral"),
            "score": max(-10, min(10, parsed.get("score", 0))),
            "confidence": max(0.0, min(1.0, parsed.get("confidence", 0.5))),
            "tweet_volume": parsed.get("tweet_volume", "unknown"),
            "key_topics": parsed.get("key_topics", []),
            "notable_accounts": parsed.get("notable_accounts", []),
            "summary": parsed.get("summary", ""),
            "from_cache": False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Cachear
        _sentiment_cache[symbol] = {
            "data": result,
            "timestamp": time.time(),
        }

        logger.info(
            f"[TWITTER] {symbol}: {result['sentiment']} (score={result['score']}, "
            f"volume={result['tweet_volume']}, conf={result['confidence']:.0%})"
        )

        return result

    except json.JSONDecodeError as e:
        logger.warning(f"[GROK] JSON parse error para {symbol}: {e}")
        return {
            "available": False,
            "reason": f"JSON parse error: {e}",
            "sentiment": "neutral",
            "score": 0,
        }
    except asyncio.TimeoutError:
        logger.warning(f"[GROK] Timeout para {symbol}")
        return {
            "available": False,
            "reason": "timeout",
            "sentiment": "neutral",
            "score": 0,
        }
    except Exception as e:
        logger.warning(f"[GROK] Erro ao analisar sentimento Twitter para {symbol}: {e}")
        return {
            "available": False,
            "reason": str(e),
            "sentiment": "neutral",
            "score": 0,
        }


# Precisa do import asyncio para o TimeoutError
import asyncio  # noqa: E402
