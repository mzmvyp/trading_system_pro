"""
Performance Tracker - Advanced trading performance metrics.
Source: sinais
Features:
- TradeStats: win rate, profit factor, avg win/loss, largest win/loss
- Sharpe Ratio (annualized)
- Sortino Ratio (downside deviation)
- Expectancy (mathematical edge per trade)
- Comprehensive reports
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np

from src.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TradeStats:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    profit_factor: float
    avg_duration_minutes: float


class PerformanceTracker:
    """Calculates advanced performance metrics from trade history."""

    def calculate_trade_stats(self, trades: List[Dict]) -> TradeStats:
        """Calculate basic trade statistics."""
        if not trades:
            return TradeStats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        wins = [t for t in trades if t.get("pnl", 0) > 0]
        losses = [t for t in trades if t.get("pnl", 0) < 0]
        total = len(trades)

        win_rate = (len(wins) / total) * 100 if total > 0 else 0
        avg_win = sum(t["pnl"] for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t["pnl"] for t in losses) / len(losses) if losses else 0

        total_gains = sum(t["pnl"] for t in wins)
        total_losses = abs(sum(t["pnl"] for t in losses))
        profit_factor = total_gains / total_losses if total_losses > 0 else float("inf")

        durations = [t.get("duration_minutes", 0) for t in trades]
        avg_duration = sum(durations) / total if total > 0 else 0

        return TradeStats(
            total_trades=total,
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=max((t["pnl"] for t in wins), default=0),
            largest_loss=min((t["pnl"] for t in losses), default=0),
            profit_factor=profit_factor,
            avg_duration_minutes=avg_duration,
        )

    def calculate_sharpe_ratio(self, trades: List[Dict], risk_free_rate: float = 0.02) -> float:
        """Calculate annualized Sharpe ratio."""
        if not trades or len(trades) < 2:
            return 0.0

        returns = [t.get("pnl_percentage", 0) / 100 for t in trades]
        avg_return = float(np.mean(returns))
        std_return = float(np.std(returns, ddof=1))

        if std_return == 0:
            return 0.0

        durations = [t.get("duration_minutes", 60) for t in trades]
        avg_duration_days = np.mean(durations) / (60 * 24)
        if avg_duration_days <= 0:
            avg_duration_days = 1

        period_risk_free = risk_free_rate * (avg_duration_days / 365)
        sharpe = (avg_return - period_risk_free) / std_return
        trades_per_year = 365 / avg_duration_days

        return float(sharpe * np.sqrt(trades_per_year))

    def calculate_sortino_ratio(self, trades: List[Dict], risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (penalizes only downside volatility)."""
        if not trades or len(trades) < 2:
            return 0.0

        returns = [t.get("pnl_percentage", 0) / 100 for t in trades]
        avg_return = float(np.mean(returns))

        downside_returns = [r for r in returns if r < 0]
        if not downside_returns:
            return float("inf")

        downside_std = float(np.std(downside_returns, ddof=1))
        if downside_std == 0:
            return 0.0

        durations = [t.get("duration_minutes", 60) for t in trades]
        avg_duration_days = np.mean(durations) / (60 * 24)
        if avg_duration_days <= 0:
            avg_duration_days = 1

        period_risk_free = risk_free_rate * (avg_duration_days / 365)
        sortino = (avg_return - period_risk_free) / downside_std
        trades_per_year = 365 / avg_duration_days

        return float(sortino * np.sqrt(trades_per_year))

    def calculate_expectancy(self, trades: List[Dict]) -> float:
        """Calculate mathematical expectancy per trade."""
        if not trades:
            return 0.0

        wins = [t for t in trades if t.get("pnl", 0) > 0]
        losses = [t for t in trades if t.get("pnl", 0) < 0]
        total = len(trades)

        win_rate = len(wins) / total if total > 0 else 0
        avg_win = sum(t["pnl"] for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t["pnl"] for t in losses) / len(losses) if losses else 0

        return (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    def calculate_max_drawdown(self, trades: List[Dict]) -> Dict:
        """Calculate maximum drawdown from equity curve."""
        if not trades:
            return {"max_drawdown_pct": 0, "max_drawdown_amount": 0}

        cumulative = 0
        peak = 0
        max_dd = 0
        max_dd_pct = 0

        for trade in trades:
            cumulative += trade.get("pnl", 0)
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd
                max_dd_pct = (dd / peak * 100) if peak > 0 else 0

        return {
            "max_drawdown_pct": max_dd_pct,
            "max_drawdown_amount": max_dd,
        }

    def get_comprehensive_report(self, trades: List[Dict], period_days: int = 7) -> Dict:
        """Generate a full performance report."""
        stats = self.calculate_trade_stats(trades)
        drawdown = self.calculate_max_drawdown(trades)

        return {
            "period_days": period_days,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "basic_stats": {
                "total_trades": stats.total_trades,
                "win_rate": round(stats.win_rate, 2),
                "profit_factor": round(stats.profit_factor, 2),
                "avg_win": round(stats.avg_win, 4),
                "avg_loss": round(stats.avg_loss, 4),
                "largest_win": round(stats.largest_win, 4),
                "largest_loss": round(stats.largest_loss, 4),
                "avg_duration_min": round(stats.avg_duration_minutes, 1),
            },
            "advanced_metrics": {
                "sharpe_ratio": round(self.calculate_sharpe_ratio(trades), 3),
                "sortino_ratio": round(self.calculate_sortino_ratio(trades), 3),
                "expectancy": round(self.calculate_expectancy(trades), 4),
                "max_drawdown_pct": round(drawdown["max_drawdown_pct"], 2),
                "max_drawdown_amount": round(drawdown["max_drawdown_amount"], 4),
            },
        }
