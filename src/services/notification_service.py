"""
Notification Service - Multi-channel notification system.
Source: trader_monitor
Channels: Email (SMTP), Slack, Discord, Telegram, Webhook
Features:
- Priority levels (low/medium/high/critical)
- Rate limiting per channel
- Notification history
"""

import os
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional

from src.core.logger import get_logger

logger = get_logger(__name__)


class NotificationPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class NotificationChannel(Enum):
    TELEGRAM = "telegram"
    DISCORD = "discord"
    SLACK = "slack"
    EMAIL = "email"
    WEBHOOK = "webhook"


class NotificationService:
    """
    Multi-channel notification service with rate limiting and priority.
    Supports Telegram, Discord, Slack, Email, and custom webhooks.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.channels: Dict[str, Dict] = {}
        self.rate_limits: Dict[str, List[datetime]] = {}
        self.history: List[Dict] = []

        # Rate limits per channel (max messages per hour)
        self.max_per_hour = {
            "telegram": self.config.get("telegram_rate", 30),
            "discord": self.config.get("discord_rate", 20),
            "slack": self.config.get("slack_rate", 20),
            "email": self.config.get("email_rate", 10),
            "webhook": self.config.get("webhook_rate", 60),
        }

        self._configure_channels()

    def _configure_channels(self):
        """Configure available channels from environment variables."""
        # Telegram
        telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        telegram_chat = os.getenv("TELEGRAM_CHAT_ID")
        if telegram_token and telegram_chat:
            self.channels["telegram"] = {
                "token": telegram_token,
                "chat_id": telegram_chat,
            }

        # Discord
        discord_webhook = os.getenv("DISCORD_WEBHOOK_URL")
        if discord_webhook:
            self.channels["discord"] = {"webhook_url": discord_webhook}

        # Slack
        slack_webhook = os.getenv("SLACK_WEBHOOK_URL")
        if slack_webhook:
            self.channels["slack"] = {"webhook_url": slack_webhook}

        # Custom webhook
        webhook_url = os.getenv("NOTIFICATION_WEBHOOK_URL")
        if webhook_url:
            self.channels["webhook"] = {"url": webhook_url}

        logger.info(f"Notification channels configured: {list(self.channels.keys())}")

    def _check_rate_limit(self, channel: str) -> bool:
        """Check if we're within rate limits for a channel."""
        now = datetime.now(timezone.utc)
        if channel not in self.rate_limits:
            self.rate_limits[channel] = []

        # Clean old entries
        cutoff = now - timedelta(hours=1)
        self.rate_limits[channel] = [t for t in self.rate_limits[channel] if t > cutoff]

        max_rate = self.max_per_hour.get(channel, 30)
        return len(self.rate_limits[channel]) < max_rate

    def _record_send(self, channel: str):
        """Record a sent notification for rate limiting."""
        if channel not in self.rate_limits:
            self.rate_limits[channel] = []
        self.rate_limits[channel].append(datetime.now(timezone.utc))

    async def send(
        self,
        message: str,
        title: Optional[str] = None,
        priority: NotificationPriority = NotificationPriority.MEDIUM,
        channels: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        """
        Send a notification to specified channels.

        Args:
            message: Notification message text
            title: Optional title/subject
            priority: Priority level
            channels: Specific channels to use (None = all configured)
            metadata: Additional data to include
        """
        target_channels = channels or list(self.channels.keys())
        results = {}

        for channel in target_channels:
            if channel not in self.channels:
                results[channel] = {"success": False, "error": "not_configured"}
                continue

            if not self._check_rate_limit(channel):
                results[channel] = {"success": False, "error": "rate_limited"}
                continue

            try:
                if channel == "telegram":
                    success = await self._send_telegram(message, title, priority)
                elif channel == "discord":
                    success = await self._send_discord(message, title, priority)
                elif channel == "slack":
                    success = await self._send_slack(message, title, priority)
                elif channel == "webhook":
                    success = await self._send_webhook(message, title, priority, metadata)
                else:
                    success = False

                results[channel] = {"success": success}
                if success:
                    self._record_send(channel)

            except Exception as e:
                logger.error(f"Notification error ({channel}): {e}")
                results[channel] = {"success": False, "error": str(e)}

        # Record in history
        self.history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "title": title,
            "message": message[:200],
            "priority": priority.value,
            "results": results,
        })

        # Keep history manageable
        if len(self.history) > 1000:
            self.history = self.history[-500:]

        return results

    async def _send_telegram(self, message: str, title: Optional[str], priority: NotificationPriority) -> bool:
        """Send via Telegram Bot API."""
        try:
            import aiohttp

            config = self.channels["telegram"]
            text = f"*{title}*\n\n{message}" if title else message

            # Add priority emoji
            priority_emoji = {"low": "ℹ️", "medium": "⚠️", "high": "🔴", "critical": "🚨"}
            text = f"{priority_emoji.get(priority.value, '')} {text}"

            url = f"https://api.telegram.org/bot{config['token']}/sendMessage"
            payload = {
                "chat_id": config["chat_id"],
                "text": text,
                "parse_mode": "Markdown",
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    return resp.status == 200

        except Exception as e:
            logger.error(f"Telegram send error: {e}")
            return False

    async def _send_discord(self, message: str, title: Optional[str], priority: NotificationPriority) -> bool:
        """Send via Discord webhook."""
        try:
            import aiohttp

            config = self.channels["discord"]
            color_map = {"low": 0x3498DB, "medium": 0xF39C12, "high": 0xE74C3C, "critical": 0x8E44AD}

            payload = {
                "embeds": [{
                    "title": title or "Trading System",
                    "description": message,
                    "color": color_map.get(priority.value, 0x3498DB),
                    "timestamp": datetime.utcnow().isoformat(),
                }]
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(config["webhook_url"], json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    return resp.status in (200, 204)

        except Exception as e:
            logger.error(f"Discord send error: {e}")
            return False

    async def _send_slack(self, message: str, title: Optional[str], priority: NotificationPriority) -> bool:
        """Send via Slack webhook."""
        try:
            import aiohttp

            config = self.channels["slack"]
            text = f"*{title}*\n{message}" if title else message

            payload = {"text": text}

            async with aiohttp.ClientSession() as session:
                async with session.post(config["webhook_url"], json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    return resp.status == 200

        except Exception as e:
            logger.error(f"Slack send error: {e}")
            return False

    async def _send_webhook(self, message: str, title: Optional[str], priority: NotificationPriority, metadata: Optional[Dict]) -> bool:
        """Send via custom webhook."""
        try:
            import aiohttp

            config = self.channels["webhook"]
            payload = {
                "title": title,
                "message": message,
                "priority": priority.value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": metadata or {},
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(config["url"], json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    return resp.status in (200, 201, 202, 204)

        except Exception as e:
            logger.error(f"Webhook send error: {e}")
            return False

    async def send_trade_signal(self, signal: Dict):
        """Convenience method to send a trade signal notification."""
        direction = signal.get("direction", "UNKNOWN")
        symbol = signal.get("symbol", "UNKNOWN")
        confidence = signal.get("confidence", 0)
        entry = signal.get("entry_price", 0)

        title = f"Trade Signal: {direction} {symbol}"
        message = (
            f"Direction: {direction}\n"
            f"Symbol: {symbol}\n"
            f"Confidence: {confidence}\n"
            f"Entry: {entry}\n"
            f"SL: {signal.get('stop_loss', 'N/A')}\n"
            f"TP1: {signal.get('target_1', 'N/A')}"
        )

        priority = NotificationPriority.HIGH if confidence >= 8 else NotificationPriority.MEDIUM
        await self.send(message, title, priority, metadata=signal)

    def get_status(self) -> Dict:
        """Get notification service status."""
        return {
            "configured_channels": list(self.channels.keys()),
            "recent_notifications": len(self.history),
            "rate_limits": {
                ch: len(times) for ch, times in self.rate_limits.items()
            },
        }
