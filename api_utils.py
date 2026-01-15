"""
API utilities for rate limiting and circuit breaker pattern
Prevents API bans and handles failures gracefully
"""

import asyncio
import time
from typing import Callable, Any, Optional
from functools import wraps
from collections import deque
from datetime import datetime, timedelta

from logger import get_logger

logger = get_logger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter for API calls
    Binance Futures: 1200 requests/minute, 10 orders/second
    """

    def __init__(self, max_calls: int = 1200, period: int = 60):
        """
        Args:
            max_calls: Maximum number of calls allowed
            period: Time period in seconds
        """
        self.max_calls = max_calls
        self.period = period
        self.calls = deque()
        self._lock = asyncio.Lock()

    async def __aenter__(self):
        """Async context manager entry"""
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        pass

    async def acquire(self):
        """Wait until a call can be made within rate limits"""
        async with self._lock:
            now = time.time()

            # Remove calls outside the time window
            while self.calls and self.calls[0] <= now - self.period:
                self.calls.popleft()

            # If at limit, wait until oldest call expires
            if len(self.calls) >= self.max_calls:
                sleep_time = self.period - (now - self.calls[0])
                if sleep_time > 0:
                    logger.debug(f"Rate limit reached, waiting {sleep_time:.2f}s")
                    await asyncio.sleep(sleep_time)
                    # Retry acquisition after waiting
                    return await self.acquire()

            # Record this call
            self.calls.append(now)

    def get_stats(self) -> dict:
        """Get current rate limiter statistics"""
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
        """
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            expected_exception: Exception type that triggers circuit breaker
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = asyncio.Lock()

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection

        Args:
            func: Async function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            Exception: If circuit is OPEN or function fails
        """
        async with self._lock:
            # Check if circuit should transition to HALF_OPEN
            if self.state == "OPEN":
                if self.last_failure_time and \
                   time.time() - self.last_failure_time >= self.recovery_timeout:
                    self.state = "HALF_OPEN"
                    logger.info("Circuit breaker transitioning to HALF_OPEN (testing recovery)")
                else:
                    logger.warning(f"Circuit breaker OPEN - rejecting call to {func.__name__}")
                    raise Exception(f"Circuit breaker is OPEN for {func.__name__}")

        # Execute function
        try:
            result = await func(*args, **kwargs)

            # Success - reset failure count
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

                # Open circuit if threshold reached
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
                    logger.error(
                        f"Circuit breaker OPEN for {func.__name__} - "
                        f"too many failures ({self.failure_count})"
                    )

            raise

    def reset(self):
        """Manually reset circuit breaker"""
        self.failure_count = 0
        self.state = "CLOSED"
        self.last_failure_time = None
        logger.info("Circuit breaker manually reset")

    def get_state(self) -> dict:
        """Get current circuit breaker state"""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "last_failure_time": self.last_failure_time,
            "recovery_timeout": self.recovery_timeout
        }


def rate_limited(max_calls: int = 1200, period: int = 60):
    """
    Decorator for rate-limited async functions

    Usage:
        @rate_limited(max_calls=100, period=60)
        async def my_api_call():
            pass
    """
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


async def exponential_backoff_retry(
    func: Callable,
    max_retries: int = 4,
    base_delay: float = 2.0,
    *args,
    **kwargs
) -> Any:
    """
    Retry async function with exponential backoff

    Args:
        func: Async function to retry
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds (doubles each retry)
        *args: Function arguments
        **kwargs: Function keyword arguments

    Returns:
        Function result

    Raises:
        Last exception if all retries fail
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)

        except Exception as e:
            last_exception = e

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
