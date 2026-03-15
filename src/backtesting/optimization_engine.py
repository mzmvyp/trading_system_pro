"""
Optimization Engine - Otimização de parâmetros para trading_system_pro
=======================================================================
Testa configurações de indicadores (RSI, MACD, BB, volume, confidence)
em dados históricos Binance, calcula score composto e suporta walk-forward.

Score composto: 30% win rate + 30% return + 20% Sharpe + 20% (1 - drawdown)
Origem conceitual: extração de repos mzmvyp (sinais/optimization_engine não acessível).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Pesos do score composto (soma = 1.0)
WEIGHT_WIN_RATE = 0.30
WEIGHT_RETURN = 0.30
WEIGHT_SHARPE = 0.20
WEIGHT_DRAWDOWN = 0.20  # score += weight * (1 - drawdown_normalized)


@dataclass
class OptimizationParams:
    """Espaço de parâmetros a otimizar."""
    rsi_period: int = 14
    rsi_overbought: int = 70
    rsi_oversold: int = 30
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    volume_ma_period: int = 20
    min_volume_ratio: float = 1.0  # volume > min_ratio * volume_ma
    min_confidence: float = 7.0    # escala 0-10

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rsi_period": self.rsi_period,
            "rsi_overbought": self.rsi_overbought,
            "rsi_oversold": self.rsi_oversold,
            "macd_fast": self.macd_fast,
            "macd_slow": self.macd_slow,
            "macd_signal": self.macd_signal,
            "bb_period": self.bb_period,
            "bb_std": self.bb_std,
            "volume_ma_period": self.volume_ma_period,
            "min_volume_ratio": self.min_volume_ratio,
            "min_confidence": self.min_confidence,
        }


@dataclass
class BacktestMetrics:
    """Métricas de uma rodada de backtest."""
    win_rate: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    total_trades: int
    profit_factor: float

    def composite_score(
        self,
        w_wr: float = WEIGHT_WIN_RATE,
        w_ret: float = WEIGHT_RETURN,
        w_sh: float = WEIGHT_SHARPE,
        w_dd: float = WEIGHT_DRAWDOWN,
    ) -> float:
        """Score no intervalo [0, 1] (normaliza cada métrica para 0-1)."""
        wr_norm = max(0, min(1, self.win_rate))
        ret_norm = max(0, min(1, (self.total_return_pct + 20) / 40))  # -20% -> 0, +20% -> 1
        sharpe_norm = max(0, min(1, (self.sharpe_ratio + 1) / 2))     # -1 -> 0, 1 -> 1
        dd_norm = max(0, min(1, self.max_drawdown_pct / 30))          # 0% -> 0, 30% -> 1
        return (
            w_wr * wr_norm
            + w_ret * ret_norm
            + w_sh * sharpe_norm
            + w_dd * (1 - dd_norm)
        )


def _compute_indicators(df: pd.DataFrame, params: OptimizationParams) -> pd.DataFrame:
    """Adiciona colunas RSI, MACD, BB e volume_ratio ao DataFrame."""
    if df.empty or len(df) < max(params.rsi_period, params.macd_slow, params.bb_period, params.volume_ma_period):
        return df
    out = df.copy()
    close = pd.Series(pd.to_numeric(out["close"], errors="coerce"))
    volume = pd.Series(pd.to_numeric(out["volume"], errors="coerce"))

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(params.rsi_period).mean()
    avg_loss = loss.rolling(params.rsi_period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    ema_fast = close.ewm(span=params.macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=params.macd_slow, adjust=False).mean()
    out["macd"] = ema_fast - ema_slow
    out["macd_signal"] = out["macd"].ewm(span=params.macd_signal, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    # Bollinger
    out["bb_mid"] = close.rolling(params.bb_period).mean()
    out["bb_std"] = close.rolling(params.bb_period).std()
    out["bb_upper"] = out["bb_mid"] + params.bb_std * out["bb_std"]
    out["bb_lower"] = out["bb_mid"] - params.bb_std * out["bb_std"]

    # Volume ratio
    out["volume_ma"] = volume.rolling(params.volume_ma_period).mean()
    out["volume_ratio"] = volume / out["volume_ma"].replace(0, np.nan)
    return out


def _run_simple_backtest(
    df: pd.DataFrame,
    params: OptimizationParams,
    initial_balance: float = 10000.0,
) -> BacktestMetrics:
    """
    Backtest simplificado: long quando RSI oversold + MACD cruzamento positivo + volume ok;
    saída em take-profit 2% ou stop -1%.
    """
    df = _compute_indicators(df, params)
    if "rsi" not in df.columns or df["rsi"].isna().all():
        return BacktestMetrics(0.0, 0.0, 0.0, 0.0, 0, 0.0)

    balance = initial_balance
    equity_curve = [balance]
    trades: List[Dict[str, Any]] = []
    position: Optional[Dict[str, Any]] = None
    min_bars = max(params.rsi_period, params.macd_slow, params.bb_period) + 5

    for i in range(min_bars, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        high, low, close = float(row["high"]), float(row["low"]), float(row["close"])
        rsi = row.get("rsi")
        macd_hist = row.get("macd_hist")
        macd_hist_prev = prev.get("macd_hist")
        vol_ratio = row.get("volume_ratio", 1.0)
        if pd.isna(rsi) or pd.isna(macd_hist):
            if position:
                if position["side"] == "long":
                    pnl_pct = (close - position["entry"]) / position["entry"]
                else:
                    pnl_pct = (position["entry"] - close) / position["entry"]
                equity_curve.append(balance + position["size_usd"] * pnl_pct)
            else:
                equity_curve.append(balance)
            continue

        if position is None:
            if (
                rsi <= params.rsi_oversold
                and (macd_hist_prev is not None and macd_hist_prev <= 0 and macd_hist > 0)
                and (pd.isna(vol_ratio) or vol_ratio >= params.min_volume_ratio)
            ):
                size_usd = balance * 0.2
                position = {
                    "entry": close,
                    "size_usd": size_usd,
                    "side": "long",
                    "tp": close * 1.02,
                    "sl": close * 0.99,
                }
            equity_curve.append(balance)
        else:
            exit_price = None
            if position["side"] == "long":
                if low <= position["sl"]:
                    exit_price = position["sl"]
                elif high >= position["tp"]:
                    exit_price = position["tp"]
            if exit_price is not None:
                pnl_pct = (exit_price - position["entry"]) / position["entry"]
                pnl_usd = position["size_usd"] * pnl_pct
                balance += pnl_usd
                trades.append({"pnl_pct": pnl_pct, "pnl_usd": pnl_usd})
                position = None
                equity_curve.append(balance)
            else:
                pnl_pct = (close - position["entry"]) / position["entry"]
                equity_curve.append(balance + position["size_usd"] * pnl_pct)

    if position is not None:
        # Close at last price
        last = float(df.iloc[-1]["close"])
        pnl_pct = (last - position["entry"]) / position["entry"]
        balance += position["size_usd"] * pnl_pct
        trades.append({"pnl_pct": pnl_pct, "pnl_usd": position["size_usd"] * pnl_pct})

    # Metrics
    total_trades = len(trades)
    if total_trades == 0:
        return BacktestMetrics(0.0, 0.0, 0.0, 0.0, 0, 0.0)
    wins = [t["pnl_pct"] for t in trades if t["pnl_pct"] > 0]
    losses = [t["pnl_pct"] for t in trades if t["pnl_pct"] <= 0]
    win_rate = len(wins) / total_trades
    total_return_pct = (balance - initial_balance) / initial_balance * 100
    eq = np.array(equity_curve, dtype=float)
    eq = eq[eq > 0]
    returns = np.diff(eq) / eq[:-1] if len(eq) > 1 else np.array([])
    returns = returns[~np.isnan(returns)]
    sharpe = 0.0
    if len(returns) > 0 and np.std(returns) > 0:
        sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252))
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / np.where(peak > 0, peak, 1)
    max_drawdown_pct = abs(float(np.min(dd))) * 100
    gross_win = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0
    profit_factor = gross_win / gross_loss if gross_loss > 0 else (float("inf") if gross_win > 0 else 0)

    return BacktestMetrics(
        win_rate=win_rate,
        total_return_pct=total_return_pct,
        sharpe_ratio=sharpe,
        max_drawdown_pct=max_drawdown_pct,
        total_trades=total_trades,
        profit_factor=profit_factor,
    )


def _grid_params(
    rsi_periods: List[int] = None,
    bb_periods: List[int] = None,
    min_confidences: List[float] = None,
) -> List[OptimizationParams]:
    """Gera grade de parâmetros (reduzida para exemplo)."""
    rsi_periods = rsi_periods or [14]
    bb_periods = bb_periods or [20]
    min_confidences = min_confidences or [7.0]
    params_list: List[OptimizationParams] = []
    for rsi in rsi_periods:
        for bb in bb_periods:
            for conf in min_confidences:
                params_list.append(OptimizationParams(
                    rsi_period=rsi,
                    bb_period=bb,
                    min_confidence=conf,
                ))
    return params_list


async def run_optimization(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    data_df: Optional[pd.DataFrame] = None,
    walk_forward_windows: Optional[int] = None,
    param_grid: Optional[List[OptimizationParams]] = None,
) -> Tuple[OptimizationParams, BacktestMetrics, List[Dict[str, Any]]]:
    """
    Executa otimização sobre dados históricos.

    Args:
        symbol: Símbolo Binance.
        interval: Intervalo dos candles.
        start_date / end_date: Período (usado se data_df for None).
        data_df: Se fornecido, usa este DataFrame em vez de buscar.
        walk_forward_windows: Se > 1, divide o período em janelas e valida walk-forward.
        param_grid: Lista de parâmetros a testar; se None, usa grade padrão.

    Returns:
        (melhor_params, melhores_metricas, lista de resultados por combinação)
    """
    if data_df is None:
        from src.exchange.client import BinanceClient
        client = BinanceClient()
        end_date = end_date or datetime.utcnow()
        start_date = start_date or (end_date - timedelta(days=90))
        data_df = await client.get_historical_klines(symbol, interval, start_date, end_date)
        if data_df is None or data_df.empty:
            logger.warning("Sem dados históricos para otimização")
            return OptimizationParams(), BacktestMetrics(0.0, 0.0, 0.0, 0.0, 0, 0.0), []

    param_list = param_grid or _grid_params()
    results: List[Dict[str, Any]] = []
    best_score = -1.0
    best_params = param_list[0] if param_list else OptimizationParams()
    best_metrics = BacktestMetrics(0.0, 0.0, 0.0, 0.0, 0, 0.0)

    for params in param_list:
        if walk_forward_windows and walk_forward_windows > 1:
            # Walk-forward: média dos scores em cada janela
            n = len(data_df)
            window_size = n // walk_forward_windows
            scores_list: List[float] = []
            for w in range(walk_forward_windows):
                start_i = w * window_size
                end_i = min((w + 1) * window_size, n)
                if end_i - start_i < 50:
                    continue
                chunk = data_df.iloc[start_i:end_i].copy()
                met = _run_simple_backtest(chunk, params)
                scores_list.append(met.composite_score())
            score = float(np.mean(scores_list)) if scores_list else 0.0
            metrics = _run_simple_backtest(data_df, params)  # overall metrics
        else:
            metrics = _run_simple_backtest(data_df, params)
            score = metrics.composite_score()
        results.append({"params": params.to_dict(), "metrics": metrics, "score": score})
        if score > best_score:
            best_score = score
            best_params = params
            best_metrics = metrics

    logger.info(f"[OPTIMIZATION] Melhor score: {best_score:.4f} | Win rate: {best_metrics.win_rate:.2%} | Return: {best_metrics.total_return_pct:.2f}%")
    return best_params, best_metrics, results


def apply_best_params(params: OptimizationParams, config_target: str = "config") -> None:
    """
    Aplica os melhores parâmetros ao sistema (ex.: atualizar config ou variáveis de ambiente).
    config_target: 'config' para src.core.config, ou caminho de arquivo.
    """
    d = params.to_dict()
    logger.info(f"[OPTIMIZATION] Aplicar parâmetros: {d}")
    # Implementação depende de onde o trading_system_pro guarda RSI/MACD/BB/confidence.
    # Ex.: atualizar settings ou escrever em .env / JSON para o dashboard ler.
    # Por ora apenas log.
    return


# CLI opcional
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    best, metrics, all_results = asyncio.run(run_optimization(
        symbol="BTCUSDT",
        interval="1h",
        start_date=datetime.utcnow() - timedelta(days=60),
        walk_forward_windows=3,
    ))
    print("Best params:", best.to_dict())
    print("Metrics:", metrics.win_rate, metrics.total_return_pct, metrics.sharpe_ratio, metrics.max_drawdown_pct)
    print("Composite score:", metrics.composite_score())
