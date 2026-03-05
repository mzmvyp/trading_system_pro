"""
Time Filter - Filter trades based on session timing and liquidity.
Source: smart_trading_system
Sessions: Asia, Europe, US with overlap periods
Features: Session-specific behavior, liquidity windows, strategy timing
"""

import logging
from datetime import datetime, time
from enum import Enum
from typing import Dict, Optional

from src.core.logger import get_logger

logger = get_logger(__name__)


class TradingSession(Enum):
    ASIA = "ASIA"              # 00:00-08:00 UTC
    EUROPE = "EUROPE"          # 07:00-16:00 UTC
    US = "US"                  # 13:00-22:00 UTC
    ASIA_EUROPE = "ASIA_EU"   # 07:00-08:00 UTC overlap
    EUROPE_US = "EU_US"       # 13:00-16:00 UTC overlap
    OFF_HOURS = "OFF_HOURS"   # 22:00-00:00 UTC


class TimeFilter:
    """
    Filter trades based on market session and time of day.
    Different strategies perform better in different sessions.
    """

    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        self.timezone_offset = config.get("timezone_offset_hours", 0)

        # Session definitions (UTC)
        self.sessions = {
            TradingSession.ASIA: (time(0, 0), time(8, 0)),
            TradingSession.EUROPE: (time(7, 0), time(16, 0)),
            TradingSession.US: (time(13, 0), time(22, 0)),
            TradingSession.ASIA_EUROPE: (time(7, 0), time(8, 0)),
            TradingSession.EUROPE_US: (time(13, 0), time(16, 0)),
        }

        # High liquidity windows
        self.high_liquidity = [
            (time(7, 0), time(11, 0)),    # EU open
            (time(13, 0), time(16, 0)),   # EU/US overlap
            (time(13, 30), time(15, 0)),  # US session peak
        ]

        # Strategy-session preferences
        self.strategy_sessions = {
            "scalp": [TradingSession.EUROPE_US, TradingSession.US],
            "day_trade": [TradingSession.EUROPE, TradingSession.US, TradingSession.EUROPE_US],
            "swing": None,  # Any session
            "breakout": [TradingSession.EUROPE, TradingSession.US],
            "trend_following": [TradingSession.US, TradingSession.EUROPE_US],
            "mean_reversion": [TradingSession.ASIA, TradingSession.OFF_HOURS],
        }

    def get_current_session(self, now: Optional[datetime] = None) -> Dict:
        """Get current trading session information."""
        now = now or datetime.utcnow()
        current_time = now.time()

        active_sessions = []
        for session, (start, end) in self.sessions.items():
            if start <= current_time < end:
                active_sessions.append(session)

        if not active_sessions:
            active_sessions = [TradingSession.OFF_HOURS]

        # Check liquidity
        is_high_liquidity = any(
            start <= current_time < end
            for start, end in self.high_liquidity
        )

        # Primary session (most specific)
        primary = active_sessions[-1] if active_sessions else TradingSession.OFF_HOURS

        return {
            "primary_session": primary.value,
            "active_sessions": [s.value for s in active_sessions],
            "is_high_liquidity": is_high_liquidity,
            "utc_time": current_time.strftime("%H:%M"),
            "is_weekend": now.weekday() >= 5,
        }

    def should_trade(self, session_data: Dict, strategy_type: str = "day_trade") -> Dict:
        """Check if current time is suitable for trading."""
        # Weekend check (crypto trades 24/7, but lower liquidity)
        if session_data.get("is_weekend"):
            if strategy_type in ("scalp", "breakout"):
                return {
                    "allowed": False,
                    "reason": "Weekend - low liquidity for scalp/breakout",
                }

        preferred = self.strategy_sessions.get(strategy_type)
        if preferred is None:
            return {"allowed": True, "reason": "Swing/position - any session OK"}

        active = session_data.get("active_sessions", [])
        matching = [s for s in preferred if s.value in active]

        if matching:
            return {
                "allowed": True,
                "reason": f"Good session for {strategy_type}: {matching[0].value}",
                "liquidity": "HIGH" if session_data.get("is_high_liquidity") else "NORMAL",
            }

        return {
            "allowed": False,
            "reason": f"Current session not optimal for {strategy_type}",
            "recommended_sessions": [s.value for s in preferred],
        }
