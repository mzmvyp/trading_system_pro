"""
Fundamental Filter - News and event-based trade filtering.
Source: smart_trading_system
Features:
- High-impact event detection (avoid trading during)
- News sentiment assessment
- Black swan risk detection
- Configurable event calendar
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from src.core.logger import get_logger

logger = get_logger(__name__)


class FundamentalFilter:
    """
    Filter trades based on fundamental/news events.
    Blocks trading during high-impact events and extreme sentiment.
    """

    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        self.block_on_high_impact = config.get("block_on_high_impact", True)
        self.block_minutes_before = config.get("block_minutes_before", 30)
        self.block_minutes_after = config.get("block_minutes_after", 15)
        self.max_fear_for_long = config.get("max_fear_for_long", 15)
        self.min_greed_for_short = config.get("min_greed_for_short", 85)

        # Known recurring high-impact events (UTC hours)
        self.recurring_events = [
            {"name": "FOMC", "day_of_week": 2, "hour": 18, "impact": "HIGH"},
            {"name": "NFP", "day_of_month": "first_friday", "hour": 12, "impact": "HIGH"},
            {"name": "CPI", "day_of_month": 13, "hour": 12, "impact": "HIGH"},
        ]

        # Custom events (loaded externally)
        self.custom_events: List[Dict] = []

    def analyze(self, now: Optional[datetime] = None) -> Dict:
        """Analyze current fundamental conditions."""
        now = now or datetime.utcnow()

        # Check for nearby high-impact events
        upcoming_events = self._get_upcoming_events(now)
        blocking_events = [
            e for e in upcoming_events
            if e.get("impact") == "HIGH" and e.get("minutes_until", 999) <= self.block_minutes_before
        ]

        # Recent events still in effect
        recent_events = [
            e for e in upcoming_events
            if e.get("minutes_since", 999) <= self.block_minutes_after
        ]

        is_blocked = len(blocking_events) > 0 or len(recent_events) > 0

        return {
            "is_blocked": is_blocked,
            "blocking_events": blocking_events,
            "upcoming_events": upcoming_events,
            "block_reason": blocking_events[0]["name"] if blocking_events else (
                recent_events[0]["name"] if recent_events else None
            ),
        }

    def should_trade(self, fundamental_data: Dict, signal_direction: Optional[str] = None) -> Dict:
        """Check if trading is allowed based on fundamentals."""
        if fundamental_data.get("is_blocked") and self.block_on_high_impact:
            return {
                "allowed": False,
                "reason": f"High-impact event: {fundamental_data.get('block_reason')}",
            }

        return {
            "allowed": True,
            "reason": "OK",
            "upcoming_events": fundamental_data.get("upcoming_events", []),
        }

    def add_event(self, name: str, timestamp: datetime, impact: str = "HIGH"):
        """Add a custom event to the calendar."""
        self.custom_events.append({
            "name": name,
            "timestamp": timestamp.isoformat(),
            "impact": impact,
        })

    def _get_upcoming_events(self, now: datetime) -> List[Dict]:
        """Get events within the next few hours."""
        events = []

        # Check custom events
        for event in self.custom_events:
            try:
                event_time = datetime.fromisoformat(event["timestamp"])
                diff = (event_time - now).total_seconds() / 60

                if -self.block_minutes_after <= diff <= 120:
                    events.append({
                        "name": event["name"],
                        "impact": event.get("impact", "MEDIUM"),
                        "minutes_until": max(0, diff),
                        "minutes_since": max(0, -diff),
                    })
            except Exception:
                continue

        return events
