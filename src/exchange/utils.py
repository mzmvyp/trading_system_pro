"""
API utilities for rate limiting and circuit breaker pattern
Prevents API bans and handles failures gracefully
"""

import asyncio
import time
from collections import deque
from functools import wraps
from typing import Any, Callable, Optional

from src.core.logger import get_logger

logger = get_logger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter for API calls
    Binance Futures: 1200 requests/minute, 10 orders/second
    """

    def __init__(self, max_calls: int = 1200, period: int = 60):
        self.max_calls = max_calls
        self.period = period
        self.calls = deque()
        self._lock = asyncio.Lock()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def acquire(self):
        async with self._lock:
            now = time.time()
            while self.calls and self.calls[0] <= now - self.period:
                self.calls.popleft()
            if len(self.calls) >= self.max_calls:
                sleep_time = self.period - (now - self.calls[0])
                if sleep_time > 0:
                    logger.debug(f"Rate limit reached, waiting {sleep_time:.2f}s")
                    await asyncio.sleep(sleep_time)
                    return await self.acquire()
            self.calls.append(now)

    def get_stats(self) -> dict:
        now = time.time()
        recent_calls = sum(1 for call_time in self.calls if call_time > now - self.period)
        return {
            "recent_calls": recent_calls,
            "max_calls": self.max_calls,
            "period": self.period,
            "utilization_percent": (recent_calls / self.max_calls) * 100
        }


class CircuitBreaker:
    """
    Circuit breaker pattern for API calls
    States: CLOSED (normal), OPEN (failing), HALF_OPEN (testing recovery)
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "CLOSED"
        self._lock = asyncio.Lock()

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        async with self._lock:
            if self.state == "OPEN":
                if self.last_failure_time and \
                   time.time() - self.last_failure_time >= self.recovery_timeout:
                    self.state = "HALF_OPEN"
                    logger.info("Circuit breaker transitioning to HALF_OPEN (testing recovery)")
                else:
                    logger.warning(f"Circuit breaker OPEN - rejecting call to {func.__name__}")
                    raise Exception(f"Circuit breaker is OPEN for {func.__name__}")

        try:
            result = await func(*args, **kwargs)
            async with self._lock:
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                    logger.info(f"Circuit breaker CLOSED - {func.__name__} recovered")
                self.failure_count = 0
            return result

        except self.expected_exception as e:
            async with self._lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                logger.error(
                    f"Circuit breaker failure {self.failure_count}/{self.failure_threshold} "
                    f"for {func.__name__}: {e}"
                )
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
                    logger.error(
                        f"Circuit breaker OPEN for {func.__name__} - "
                        f"too many failures ({self.failure_count})"
                    )
            raise

    def reset(self):
        self.failure_count = 0
        self.state = "CLOSED"
        self.last_failure_time = None
        logger.info("Circuit breaker manually reset")

    def get_state(self) -> dict:
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "last_failure_time": self.last_failure_time,
            "recovery_timeout": self.recovery_timeout
        }


def rate_limited(max_calls: int = 1200, period: int = 60):
    limiter = RateLimiter(max_calls, period)
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async with limiter:
                return await func(*args, **kwargs)
        return wrapper
    return decorator


# Global rate limiter for Binance API (1200 req/min)
binance_rate_limiter = RateLimiter(max_calls=1200, period=60)

# Global circuit breaker for Binance API
binance_circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=60,
    expected_exception=Exception
)


def _is_non_retryable_error(error: Exception) -> bool:
    """Check if error is a client error (4xx) that should NOT be retried."""
    error_str = str(error)
    # HTTP 400 (Bad Request), 401 (Unauthorized), 403 (Forbidden), 404 (Not Found), 422 (Unprocessable)
    # These indicate the request itself is wrong — retrying won't fix it
    for code in ("HTTP 400", "HTTP 401", "HTTP 403", "HTTP 404", "HTTP 422"):
        if code in error_str:
            return True
    return False


async def exponential_backoff_retry(
    func: Callable,
    max_retries: int = 4,
    base_delay: float = 2.0,
    *args,
    **kwargs
) -> Any:
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            # Fail-fast on client errors (4xx) — retrying won't help
            if _is_non_retryable_error(e):
                logger.error(f"Non-retryable error for {func.__name__}: {e}")
                raise
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}. "
                    f"Retrying in {delay}s..."
                )
                await asyncio.sleep(delay)
            else:
                logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}")
    raise last_exception
