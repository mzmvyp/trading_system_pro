"""
Cliente para integração com a API da Binance
Includes rate limiting, circuit breaker, and robust error handling
"""
import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import aiohttp

from src.core.logger import get_logger
from src.core.config import settings
from src.exchange.utils import binance_rate_limiter, binance_circuit_breaker, exponential_backoff_retry

logger = get_logger(__name__)

class BinanceClient:
    def __init__(self):
        self.base_url = "https://fapi.binance.com"
        self.session = None
        self.timeout = aiohttp.ClientTimeout(total=10)
        logger.info("BinanceClient initialized")

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        logger.debug("HTTP session created")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            logger.debug("HTTP session closed")

    async def _request_with_protection(self, url: str, params: dict) -> dict:
        async def make_request():
            async with binance_rate_limiter:
                async with self.session.get(url, params=params) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"API error {response.status}: {error_text}")
                        raise aiohttp.ClientError(f"HTTP {response.status}: {error_text}")
                    return await response.json()
        return await binance_circuit_breaker.call(make_request)

    async def get_klines(self, symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
        try:
            url = f"{self.base_url}/fapi/v1/klines"
            params = {'symbol': symbol, 'interval': interval, 'limit': limit}
            data = await exponential_backoff_retry(
                self._request_with_protection, max_retries=3, url=url, params=params
            )
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base_volume',
                'taker_buy_quote_volume', 'ignore'
            ])
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            logger.debug(f"Fetched {len(df)} klines for {symbol} @ {interval}")
            return df
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.error(f"Network error fetching klines for {symbol}: {e}")
            raise
        except (KeyError, ValueError) as e:
            logger.error(f"Data parsing error for {symbol}: {e}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error fetching klines for {symbol}: {e}")
            raise

    async def get_ticker_24hr(self, symbol: str) -> Dict:
        try:
            url = f"{self.base_url}/fapi/v1/ticker/24hr"
            params = {'symbol': symbol}
            data = await self._request_with_protection(url, params)
            logger.debug(f"Fetched 24hr ticker for {symbol}")
            return data
        except Exception as e:
            logger.error(f"Error fetching 24hr ticker for {symbol}: {e}")
            raise

    async def get_orderbook(self, symbol: str, limit: int = 100) -> Dict:
        try:
            url = f"{self.base_url}/fapi/v1/depth"
            params = {'symbol': symbol, 'limit': limit}
            data = await self._request_with_protection(url, params)
            logger.debug(f"Fetched orderbook for {symbol} (limit={limit})")
            return data
        except Exception as e:
            logger.error(f"Error fetching orderbook for {symbol}: {e}")
            raise

    async def get_funding_rate(self, symbol: str) -> Dict:
        try:
            url = f"{self.base_url}/fapi/v1/premiumIndex"
            params = {'symbol': symbol}
            data = await self._request_with_protection(url, params)
            logger.debug(f"Fetched funding rate for {symbol}")
            return data
        except Exception as e:
            logger.error(f"Error fetching funding rate for {symbol}: {e}")
            raise

    async def get_open_interest(self, symbol: str) -> Dict:
        try:
            url = f"{self.base_url}/fapi/v1/openInterest"
            params = {'symbol': symbol}
            data = await self._request_with_protection(url, params)
            logger.debug(f"Fetched open interest for {symbol}")
            return data
        except Exception as e:
            logger.error(f"Error fetching open interest for {symbol}: {e}")
            raise

    async def get_historical_klines(self, symbol: str, interval: str,
                                  start_time: datetime, end_time: datetime) -> pd.DataFrame:
        all_klines = []
        current_start = int(start_time.timestamp() * 1000)
        end_timestamp = int(end_time.timestamp() * 1000)

        while current_start < end_timestamp:
            url = f"{self.base_url}/fapi/v1/klines"
            params = {
                'symbol': symbol, 'interval': interval,
                'startTime': current_start,
                'endTime': min(current_start + 100 * 60 * 60 * 1000, end_timestamp),
                'limit': 1000
            }
            try:
                batch = await self._request_with_protection(url, params)
                if not batch:
                    break
                all_klines.extend(batch)
                current_start = int(batch[-1][0]) + 1
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.error(f"Network error fetching historical data: {e}")
                break
            except Exception as e:
                logger.exception(f"Error fetching historical klines: {e}")
                break

        if not all_klines:
            return pd.DataFrame()

        df = pd.DataFrame(all_klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base_volume',
            'taker_buy_quote_volume', 'ignore'
        ])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[df.index <= end_time]
        return df
