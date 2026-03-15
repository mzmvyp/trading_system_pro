"""
Backtesting: Reports Generator
Gerador de relatórios detalhados com gráficos e análises
Origem: mzmvyp/smart_trading_system
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import json
import base64
from io import BytesIO
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from jinja2 import Template

# Nota: no repo original BacktestResult pode vir de database.models
# Aqui usamos tipo genérico para referência; adaptar para BacktestResults ou modelo local
try:
    from .backtest_engine import BacktestResults as BacktestResult
except ImportError:
    BacktestResult = Any  # type: ignore

# Adaptar para o projeto: utils.logger e utils.helpers
import logging
logger = logging.getLogger(__name__)

def format_currency(x: float) -> str:
    return f"${x:,.2f}" if x is not None else "N/A"
def format_percentage(x: float) -> str:
    return f"{x:.2f}%" if x is not None else "N/A"
def format_number(x: float) -> str:
    return f"{x:,.2f}" if x is not None else "N/A"

# Configure matplotlib
try:
    plt.style.use("seaborn-v0_8")
except Exception:
    plt.style.use("default")
try:
    sns.set_palette("husl")
except Exception:
    pass


class ReportGenerator:
    """Gerador de relatórios de backtesting"""

    def __init__(self):
        self.charts_data = {}
        self.report_style = {
            "primary_color": "#2E86C1",
            "success_color": "#28B463",
            "danger_color": "#E74C3C",
            "warning_color": "#F39C12",
            "info_color": "#85C1E9",
            "background_color": "#F8F9FA",
            "text_color": "#2C3E50",
        }

    def generate_full_report(
        self,
        result: BacktestResult,
        output_path: str = "backtest_report.html",
        include_charts: bool = True,
    ) -> str:
        """Gera relatório completo HTML. result deve ter: total_return, sharpe_ratio, max_drawdown, equity_curve, trade_history, monthly_returns, strategy_performance, win_rate, profit_factor, etc."""
        try:
            logger.info("Gerando relatório de backtesting...")
            summary_data = self._prepare_summary_data(result)
            performance_data = self._prepare_performance_data(result)
            trade_analysis = self._prepare_trade_analysis(result)
            strategy_analysis = self._prepare_strategy_analysis(result)
            charts = {}
            if include_charts:
                charts = self._generate_all_charts(result)
            html_content = self._generate_html_report(
                summary_data, performance_data, trade_analysis, strategy_analysis, charts
            )
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            logger.info(f"Relatório salvo em: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Erro ao gerar relatório: {e}")
            raise

    def _prepare_summary_data(self, result: BacktestResult) -> Dict:
        total_return = getattr(result, "total_return_pct", None) or getattr(result, "total_return", 0)
        total_return = total_return * 100 if abs(total_return) < 1 else total_return
        sharpe = getattr(result, "sharpe_ratio", 0) or 0
        max_dd = getattr(result, "max_drawdown_pct", None) or getattr(result, "max_drawdown", 0)
        max_dd = max_dd * 100 if max_dd and abs(max_dd) < 1 else (max_dd or 0)
        win_rate = getattr(result, "win_rate", 0) or 0
        win_rate = win_rate * 100 if win_rate and abs(win_rate) <= 1 else win_rate
        if total_return >= 50:
            performance_rating, performance_class = "Excelente", "success"
        elif total_return >= 20:
            performance_rating, performance_class = "Bom", "info"
        elif total_return >= 0:
            performance_rating, performance_class = "Moderado", "warning"
        else:
            performance_rating, performance_class = "Ruim", "danger"
        sharpe_rating = "Excelente" if sharpe >= 1.5 else "Bom" if sharpe >= 1.0 else "Aceitável" if sharpe >= 0.5 else "Ruim"
        start_date = getattr(result, "config", None) and getattr(result.config, "start_date", None)
        end_date = getattr(result, "config", None) and getattr(result.config, "end_date", None)
        return {
            "period": {
                "start": start_date.strftime("%d/%m/%Y") if start_date else "N/A",
                "end": end_date.strftime("%d/%m/%Y") if end_date else "N/A",
                "days": getattr(result, "data_points_processed", 0),
            },
            "performance": {"total_return": total_return, "annual_return": getattr(result, "cagr", 0) * 100 if getattr(result, "cagr", None) else 0, "rating": performance_rating, "class": performance_class},
            "risk": {"sharpe_ratio": sharpe, "sharpe_rating": sharpe_rating, "max_drawdown": max_dd, "volatility": self._calculate_volatility(getattr(result, "equity_curve", []) or [])},
            "trades": {"total": getattr(result, "total_trades", 0), "win_rate": win_rate, "profit_factor": getattr(result, "profit_factor", 0)},
        }

    def _prepare_performance_data(self, result: BacktestResult) -> Dict:
        eq = getattr(result, "equity_curve", []) or []
        equity_values = [e[1] if isinstance(e, (list, tuple)) else e for e in eq]
        return {
            "returns": {
                "total_return": (getattr(result, "total_return_pct", 0) or 0) * 100,
                "annual_return": (getattr(result, "cagr", 0) or 0) * 100,
                "monthly_avg": 0,
                "best_month": 0,
                "worst_month": 0,
            },
            "risk_metrics": {
                "sharpe_ratio": getattr(result, "sharpe_ratio", 0) or 0,
                "sortino_ratio": getattr(result, "sortino_ratio", 0) or 0,
                "max_drawdown": (getattr(result, "max_drawdown_pct", 0) or 0) * 100,
                "volatility": self._calculate_volatility(equity_values),
                "var_95": self._calculate_var(equity_values),
                "calmar_ratio": getattr(result, "calmar_ratio", 0) or 0,
            },
            "profit_loss": {
                "net_profit": getattr(result, "total_return", 0) or 0,
                "gross_profit": 0,
                "gross_loss": 0,
                "profit_factor": getattr(result, "profit_factor", 0) or 0,
                "recovery_factor": self._calculate_recovery_factor(getattr(result, "total_return", 0), (getattr(result, "max_drawdown_pct", 0) or 0) * (getattr(result, "initial_balance", 1) or 1)),
            },
        }

    def _prepare_trade_analysis(self, result: BacktestResult) -> Dict:
        trade_log = getattr(result, "trade_log", []) or []
        if not trade_log:
            return {"summary": {"total_trades": 0}, "wins_vs_losses": {}, "duration_analysis": {}, "monthly_distribution": {}}
        trades_df = pd.DataFrame(trade_log)
        pnl_col = "pnl" if "pnl" in trades_df.columns else "realized_pnl"
        if pnl_col not in trades_df.columns:
            return {"summary": {"total_trades": len(trades_df)}, "wins_vs_losses": {}, "duration_analysis": {}, "monthly_distribution": {}}
        winning = trades_df[trades_df[pnl_col] > 0]
        losing = trades_df[trades_df[pnl_col] <= 0]
        summary = {
            "total_trades": len(trades_df),
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "win_rate": getattr(result, "win_rate", 0) or 0,
            "avg_win": getattr(result, "avg_win_pct", 0) or 0,
            "avg_loss": getattr(result, "avg_loss_pct", 0) or 0,
            "largest_win": getattr(result, "largest_win_pct", 0) or 0,
            "largest_loss": getattr(result, "largest_loss_pct", 0) or 0,
            "avg_trade": trades_df[pnl_col].mean(),
            "consecutive_wins": self._calculate_consecutive_wins(trades_df, pnl_col),
            "consecutive_losses": self._calculate_consecutive_losses(trades_df, pnl_col),
        }
        duration_analysis = {}
        if "duration_hours" in trades_df.columns:
            duration_analysis = {
                "avg_duration_hours": trades_df["duration_hours"].mean(),
                "avg_winning_duration": winning["duration_hours"].mean() if len(winning) > 0 else 0,
                "avg_losing_duration": losing["duration_hours"].mean() if len(losing) > 0 else 0,
                "shortest_trade": trades_df["duration_hours"].min(),
                "longest_trade": trades_df["duration_hours"].max(),
            }
        return {"summary": summary, "wins_vs_losses": {}, "duration_analysis": duration_analysis, "monthly_distribution": {}}

    def _prepare_strategy_analysis(self, result: BacktestResult) -> Dict:
        return getattr(result, "strategy_performance", {}) or {}

    def _generate_all_charts(self, result: BacktestResult) -> Dict:
        charts = {}
        try:
            eq = getattr(result, "equity_curve", []) or []
            equity_values = [e[1] if isinstance(e, (list, tuple)) else e for e in eq]
            if equity_values:
                charts["equity_curve"] = self._generate_equity_curve_chart(equity_values, result)
                charts["drawdown_chart"] = self._generate_drawdown_chart(equity_values, result)
        except Exception as e:
            logger.error(f"Erro ao gerar gráficos: {e}")
        return charts

    def _generate_equity_curve_chart(self, equity_curve: List[float], result: Any) -> str:
        if not equity_curve:
            return ""
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(range(len(equity_curve)), equity_curve, linewidth=2, color=self.report_style["primary_color"])
        ax.fill_between(range(len(equity_curve)), equity_curve, alpha=0.3, color=self.report_style["primary_color"])
        if equity_curve:
            ax.axhline(y=equity_curve[0], color=self.report_style["danger_color"], linestyle="--", alpha=0.7, label="Capital Inicial")
        ax.set_title("Curva de Equity", fontsize=16, fontweight="bold")
        ax.set_ylabel("Valor do Portfolio")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_currency(x)))
        plt.tight_layout()
        out = self._chart_to_base64(fig)
        plt.close(fig)
        return out

    def _generate_drawdown_chart(self, equity_curve: List[float], result: Any) -> str:
        if not equity_curve:
            return ""
        equity_series = pd.Series(equity_curve)
        running_max = equity_series.expanding().max()
        drawdown = ((equity_series - running_max) / running_max.replace(0, np.nan)) * 100
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.7, color=self.report_style["danger_color"])
        ax.set_title("Drawdown", fontsize=16, fontweight="bold")
        ax.set_ylabel("Drawdown (%)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = self._chart_to_base64(fig)
        plt.close(fig)
        return out

    def _chart_to_base64(self, fig) -> str:
        try:
            buffer = BytesIO()
            fig.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/png;base64,{image_base64}"
        except Exception as e:
            logger.error(f"Erro ao converter gráfico: {e}")
            return ""

    def _calculate_volatility(self, equity_curve: List[float]) -> float:
        if len(equity_curve) < 2:
            return 0.0
        returns = [(equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1] for i in range(1, len(equity_curve))]
        return float(np.std(returns) * np.sqrt(252) * 100)

    def _calculate_var(self, equity_curve: List[float], confidence: float = 0.05) -> float:
        if len(equity_curve) < 2:
            return 0.0
        returns = [(equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1] for i in range(1, len(equity_curve))]
        return float(np.percentile(returns, confidence * 100) * 100)

    def _calculate_recovery_factor(self, net_profit: float, max_drawdown: float) -> float:
        if max_drawdown == 0:
            return float("inf") if net_profit > 0 else 0
        return float(abs(net_profit / max_drawdown))

    def _calculate_consecutive_wins(self, trades_df: pd.DataFrame, pnl_col: str = "pnl") -> int:
        if trades_df.empty or pnl_col not in trades_df.columns:
            return 0
        consecutive = max_consecutive = 0
        for pnl in trades_df[pnl_col]:
            if pnl > 0:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0
        return max_consecutive

    def _calculate_consecutive_losses(self, trades_df: pd.DataFrame, pnl_col: str = "pnl") -> int:
        if trades_df.empty or pnl_col not in trades_df.columns:
            return 0
        consecutive = max_consecutive = 0
        for pnl in trades_df[pnl_col]:
            if pnl < 0:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0
        return max_consecutive

    def _generate_html_report(
        self,
        summary_data: Dict,
        performance_data: Dict,
        trade_analysis: Dict,
        strategy_analysis: Dict,
        charts: Dict,
    ) -> str:
        style = self.report_style
        summary = summary_data
        performance = performance_data
        trade_analysis = trade_analysis
        strategy_analysis = strategy_analysis
        html = f"""
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Relatório de Backtesting</title></head>
<body style="font-family: Segoe UI, sans-serif; background: {style['background_color']}; padding: 2rem;">
<div class="header" style="background: linear-gradient(135deg, {style['primary_color']}, {style['info_color']}); color: white; padding: 2rem; border-radius: 10px;">
<h1>Relatório de Backtesting</h1>
<p>Retorno Total: {summary.get('performance', {}).get('total_return', 0):.2f}% | Sharpe: {summary.get('risk', {}).get('sharpe_ratio', 0):.2f} | Max DD: {summary.get('risk', {}).get('max_drawdown', 0):.2f}%</p>
</div>
<div style="margin-top: 2rem;">
<p><b>Período:</b> {summary.get('period', {}).get('start', 'N/A')} a {summary.get('period', {}).get('end', 'N/A')}</p>
<p><b>Trades:</b> {summary.get('trades', {}).get('total', 0)} | Win rate: {summary.get('trades', {}).get('win_rate', 0):.1f}%</p>
</div>
{"<div style='margin-top:2rem'><img src='" + charts.get("equity_curve", "") + "' alt='Equity' style='max-width:100%'/></div>" if charts.get("equity_curve") else ""}
{"<div style='margin-top:2rem'><img src='" + charts.get("drawdown_chart", "") + "' alt='Drawdown' style='max-width:100%'/></div>" if charts.get("drawdown_chart") else ""}
<p style="margin-top:3rem; color: #666;">Relatório gerado pelo Smart Trading System (extraído). Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}</p>
</body>
</html>
"""
        return html


def generate_backtest_report(
    result: Any,
    output_path: str = "backtest_report.html",
    include_charts: bool = True,
) -> str:
    """Gera relatório de backtesting completo."""
    generator = ReportGenerator()
    return generator.generate_full_report(result, output_path, include_charts)
