"""
Position Manager - Advanced position management.
Source: trade_bot_new
Features:
- Trailing Stop (ATR-based, activation threshold, step minimum)
- Time-Based Exit (configurable per operation type)
- Post-Profit Cooldown
- Daily Trade Limits (per pair and total)
- Exit Distribution (TP1/TP2/Runner split)
- Dynamic Confidence Threshold (based on recent results)
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.core.logger import get_logger

logger = get_logger(__name__)


class PositionManagerConfig:
    """Configuration for position manager with sensible defaults."""

    def __init__(self, **kwargs):
        # Trailing stop
        self.trailing_stop_enabled = kwargs.get("trailing_stop_enabled", True)
        self.trailing_stop_activation_pct = kwargs.get("trailing_stop_activation_pct", 1.0)
        self.trailing_stop_atr_multiplier = kwargs.get("trailing_stop_atr_multiplier", 1.5)
        self.trailing_stop_step_pct = kwargs.get("trailing_stop_step_pct", 0.3)

        # Time-based exit
        self.time_based_exit_enabled = kwargs.get("time_based_exit_enabled", True)
        self.time_exit_scalp_candles = kwargs.get("time_exit_scalp_candles", 12)
        self.time_exit_day_trade_candles = kwargs.get("time_exit_day_trade_candles", 24)
        self.time_exit_swing_candles = kwargs.get("time_exit_swing_candles", 72)
        self.time_exit_action = kwargs.get("time_exit_action", "CLOSE_AT_MARKET")

        # Cooldown
        self.post_profit_cooldown_enabled = kwargs.get("post_profit_cooldown_enabled", True)
        self.post_profit_cooldown_hours = kwargs.get("post_profit_cooldown_hours", 4.0)
        self.post_profit_cooldown_partial_hours = kwargs.get("post_profit_cooldown_partial_hours", 2.0)

        # Trade limits
        self.trade_limit_enabled = kwargs.get("trade_limit_enabled", True)
        self.max_trades_per_pair_per_day = kwargs.get("max_trades_per_pair_per_day", 3)
        self.max_total_trades_per_day = kwargs.get("max_total_trades_per_day", 10)

        # Exit distribution
        self.exit_distribution = kwargs.get("exit_distribution", [40, 40, 20])

        # Dynamic confidence
        self.dynamic_confidence_enabled = kwargs.get("dynamic_confidence_enabled", True)
        self.base_confidence_threshold = kwargs.get("base_confidence_threshold", 7)
        self.confidence_after_profit = kwargs.get("confidence_after_profit", 6)
        self.confidence_after_loss = kwargs.get("confidence_after_loss", 8)
        self.recent_trade_hours = kwargs.get("recent_trade_hours", 24)


class PositionManager:
    """Advanced position management with trailing stops, cooldowns, and limits."""

    def __init__(self, config: Optional[PositionManagerConfig] = None):
        self.config = config or PositionManagerConfig()
        self.trade_history_file = Path("data/trade_history.json")
        self.trade_history: List[Dict] = []

        Path("data").mkdir(exist_ok=True)
        self._load_trade_history()

    def _load_trade_history(self):
        try:
            if self.trade_history_file.exists():
                with open(self.trade_history_file, "r", encoding="utf-8") as f:
                    self.trade_history = json.load(f)
        except Exception:
            self.trade_history = []

    def _save_trade_history(self):
        try:
            with open(self.trade_history_file, "w", encoding="utf-8") as f:
                json.dump(self.trade_history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving trade history: {e}")

    def record_trade(
        self, symbol: str, signal_type: str, result: str,
        pnl_percent: float, entry_price: float, exit_price: float,
        duration_hours: float,
    ):
        """Record a completed trade."""
        trade = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "signal_type": signal_type,
            "result": result,
            "pnl_percent": pnl_percent,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "duration_hours": duration_hours,
        }
        self.trade_history.append(trade)
        self._save_trade_history()

    def calculate_trailing_stop(
        self, position: Dict, current_price: float, atr: float
    ) -> Optional[float]:
        """Calculate new trailing stop level."""
        if not self.config.trailing_stop_enabled:
            return None

        entry_price = position.get("entry_price", 0)
        signal_type = position.get("signal", "BUY")
        current_stop = position.get("stop_loss", 0) or 0

        if entry_price <= 0 or current_price <= 0:
            return None

        # Calculate P&L
        if signal_type == "BUY":
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
        else:
            pnl_pct = ((entry_price - current_price) / entry_price) * 100

        # Only activate after threshold
        if pnl_pct < self.config.trailing_stop_activation_pct:
            return None

        trailing_distance = atr * self.config.trailing_stop_atr_multiplier

        if signal_type == "BUY":
            new_stop = current_price - trailing_distance
            if current_stop and new_stop <= current_stop:
                return None
            if current_stop > 0:
                improvement = ((new_stop - current_stop) / current_stop) * 100
                if improvement < self.config.trailing_stop_step_pct:
                    return None
        else:
            new_stop = current_price + trailing_distance
            if current_stop and new_stop >= current_stop:
                return None
            if current_stop > 0:
                improvement = ((current_stop - new_stop) / current_stop) * 100
                if improvement < self.config.trailing_stop_step_pct:
                    return None

        return new_stop

    def check_time_based_exit(self, position: Dict) -> Optional[str]:
        """Check if position should be closed based on time."""
        if not self.config.time_based_exit_enabled:
            return None

        operation_type = position.get("operation_type", "DAY_TRADE").upper()
        timestamp_str = position.get("timestamp")

        if not timestamp_str:
            return None

        try:
            open_time = datetime.fromisoformat(timestamp_str)
            hours_open = (datetime.now(timezone.utc) - open_time).total_seconds() / 3600

            if "SCALP" in operation_type:
                max_hours = self.config.time_exit_scalp_candles * (5 / 60)
            elif "DAY" in operation_type:
                max_hours = self.config.time_exit_day_trade_candles * 1
            else:  # SWING
                max_hours = self.config.time_exit_swing_candles * 4

            if hours_open > max_hours:
                return self.config.time_exit_action

            return None
        except Exception:
            return None

    def is_in_post_profit_cooldown(self, symbol: str) -> Tuple[bool, float]:
        """Check if symbol is in post-profit cooldown period."""
        if not self.config.post_profit_cooldown_enabled:
            return False, 0

        now = datetime.now(timezone.utc)
        for trade in reversed(self.trade_history):
            if trade.get("symbol") != symbol:
                continue
            try:
                trade_time = datetime.fromisoformat(trade.get("timestamp", ""))
                hours_since = (now - trade_time).total_seconds() / 3600

                if trade.get("result") == "WIN":
                    pnl = trade.get("pnl_percent", 0)
                    cooldown = (
                        self.config.post_profit_cooldown_hours
                        if pnl >= 3.0
                        else self.config.post_profit_cooldown_partial_hours
                    )
                    if hours_since < cooldown:
                        return True, cooldown - hours_since
                break
            except Exception:
                continue

        return False, 0

    def can_open_trade(self, symbol: str) -> Tuple[bool, str]:
        """Check if a new trade can be opened based on daily limits."""
        if not self.config.trade_limit_enabled:
            return True, "OK"

        now = datetime.now(timezone.utc)
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        trades_today_symbol = 0
        trades_today_total = 0

        for trade in self.trade_history:
            try:
                trade_time = datetime.fromisoformat(trade.get("timestamp", ""))
                if trade_time >= day_start:
                    trades_today_total += 1
                    if trade.get("symbol") == symbol:
                        trades_today_symbol += 1
            except Exception:
                continue

        if trades_today_symbol >= self.config.max_trades_per_pair_per_day:
            return False, f"Daily limit for {symbol} reached ({trades_today_symbol})"

        if trades_today_total >= self.config.max_total_trades_per_day:
            return False, f"Total daily limit reached ({trades_today_total})"

        return True, "OK"

    def calculate_position_sizes(self, total_quantity: float) -> Dict[str, float]:
        """Split position for distributed exit (TP1/TP2/Runner)."""
        dist = self.config.exit_distribution
        total = sum(dist)
        ratios = [d / total for d in dist]

        return {
            "tp1": total_quantity * ratios[0],
            "tp2": total_quantity * ratios[1],
            "runner": total_quantity * ratios[2] if len(ratios) > 2 else 0,
        }

    def get_dynamic_confidence_threshold(self, symbol: str) -> int:
        """Get confidence threshold adjusted by recent trade results."""
        if not self.config.dynamic_confidence_enabled:
            return self.config.base_confidence_threshold

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=self.config.recent_trade_hours)

        for trade in reversed(self.trade_history):
            if trade.get("symbol") != symbol:
                continue
            try:
                trade_time = datetime.fromisoformat(trade.get("timestamp", ""))
                if trade_time >= cutoff:
                    if trade.get("result") == "WIN":
                        return self.config.confidence_after_profit
                    elif trade.get("result") == "LOSS":
                        return self.config.confidence_after_loss
                break
            except Exception:
                continue

        return self.config.base_confidence_threshold

    def should_take_signal(self, symbol: str, signal_confidence: int) -> Tuple[bool, str]:
        """Full pre-trade validation: limits, cooldown, and confidence."""
        can_trade, reason = self.can_open_trade(symbol)
        if not can_trade:
            return False, reason

        in_cooldown, remaining = self.is_in_post_profit_cooldown(symbol)
        if in_cooldown:
            return False, f"Post-profit cooldown ({remaining:.1f}h remaining)"

        threshold = self.get_dynamic_confidence_threshold(symbol)
        if signal_confidence < threshold:
            return False, f"Confidence {signal_confidence} < threshold {threshold}"

        return True, "OK"
