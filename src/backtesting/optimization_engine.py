"""
Optimization Engine - Otimização de parâmetros com score composto
=================================================================

Origem: sinais/backtesting/optimization_engine.py (especificação).
Score composto: 30% win rate + 30% return + 20% Sharpe + 20% (1 - drawdown).
Implementa: random search, walk-forward validation, thread-safe execution.

Fluxo:
1. Gera N configurações aleatórias de parâmetros (random search)
2. Para cada configuração, roda backtest completo
3. Calcula score composto 30/30/20/20
4. Retorna melhores configurações ordenadas por score
5. Walk-forward: treina em janela in-sample, valida em out-of-sample
"""

import asyncio
import json
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.backtesting.backtest_engine import BacktestEngine, BacktestMetrics, BacktestParams
from src.core.logger import get_logger

logger = get_logger(__name__)


# ==============================================================================
# Espaço de parâmetros para otimização
# ==============================================================================

PARAM_SPACE = {
    'rsi_period': (7, 21),
    'rsi_oversold': (20.0, 40.0),
    'rsi_overbought': (60.0, 80.0),
    'ema_fast': (8, 30),
    'ema_slow': (30, 100),
    'macd_fast': (8, 16),
    'macd_slow': (20, 35),
    'macd_signal': (5, 12),
    'bb_period': (14, 30),
    'bb_std': (1.5, 3.0),
    'adx_period': (10, 20),
    'adx_min_strength': (15.0, 30.0),
    'volume_ma_period': (10, 30),
    'volume_surge_multiplier': (1.2, 2.5),
    'min_confluence': (2, 5),
    'sl_atr_multiplier': (1.0, 3.0),
    'tp1_atr_multiplier': (1.5, 5.0),
    'tp2_atr_multiplier': (3.0, 8.0),
    'tp1_close_pct': (0.3, 0.7),
}


@dataclass
class OptimizationResult:
    """Resultado de uma iteração de otimização."""
    params: BacktestParams
    metrics: BacktestMetrics
    score: float
    iteration: int


@dataclass
class WalkForwardWindow:
    """Uma janela de walk-forward."""
    in_sample_start: datetime
    in_sample_end: datetime
    out_of_sample_start: datetime
    out_of_sample_end: datetime
    best_params: Optional[BacktestParams] = None
    in_sample_metrics: Optional[BacktestMetrics] = None
    out_of_sample_metrics: Optional[BacktestMetrics] = None
    in_sample_score: float = 0.0
    out_of_sample_score: float = 0.0


class OptimizationEngine:
    """
    Motor de otimização de parâmetros de trading.

    Origem do score: sinais/backtesting/optimization_engine.py
    Score composto = 30% win_rate + 30% return + 20% Sharpe + 20% (1 - drawdown)

    Componentes normalizados para [0, 1]:
    - win_rate: valor / 100 (já é percentual)
    - return: sigmoid(return / 10) para normalizar retornos extremos
    - sharpe: sigmoid(sharpe) para normalizar
    - drawdown: 1 - (drawdown / 100), invertido (menor DD = melhor)
    """

    def __init__(self, symbol: str = "BTCUSDT", interval: str = "1h"):
        self.symbol = symbol
        self.interval = interval
        self.results: List[OptimizationResult] = []
        self.best_result: Optional[OptimizationResult] = None
        self._running = False

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Sigmoid para normalizar valores em [0, 1]."""
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def calculate_score(metrics: BacktestMetrics) -> float:
        """
        Score composto: 30% win rate + 30% return + 20% Sharpe + 20% (1 - drawdown)

        Origem: sinais/backtesting/optimization_engine.py
        """
        if metrics.total_trades < 5:
            return 0.0  # mínimo de trades para score válido

        # Normalizar componentes para [0, 1]
        win_rate_norm = min(metrics.win_rate / 100, 1.0)
        return_norm = OptimizationEngine._sigmoid(metrics.total_return_pct / 10)
        sharpe_norm = OptimizationEngine._sigmoid(metrics.sharpe_ratio)
        drawdown_norm = max(1 - metrics.max_drawdown_pct / 100, 0.0)

        score = (
            0.30 * win_rate_norm +
            0.30 * return_norm +
            0.20 * sharpe_norm +
            0.20 * drawdown_norm
        )
        return round(score, 6)

    @staticmethod
    def random_params() -> BacktestParams:
        """Gera um conjunto aleatório de parâmetros dentro do espaço definido."""
        kwargs = {}
        for param_name, (low, high) in PARAM_SPACE.items():
            if isinstance(low, int) and isinstance(high, int):
                kwargs[param_name] = random.randint(low, high)
            else:
                kwargs[param_name] = round(random.uniform(low, high), 2)

        # Garantir restrições lógicas
        if kwargs['ema_fast'] >= kwargs['ema_slow']:
            kwargs['ema_slow'] = kwargs['ema_fast'] + 10
        if kwargs['macd_fast'] >= kwargs['macd_slow']:
            kwargs['macd_slow'] = kwargs['macd_fast'] + 8
        if kwargs['tp1_atr_multiplier'] >= kwargs['tp2_atr_multiplier']:
            kwargs['tp2_atr_multiplier'] = kwargs['tp1_atr_multiplier'] + 1.5

        return BacktestParams(**kwargs)

    async def run_optimization(
        self,
        start_time: datetime,
        end_time: datetime,
        n_iterations: int = 50,
        max_hold_bars: int = 48,
    ) -> List[OptimizationResult]:
        """
        Executa otimização por random search.

        Busca dados uma vez e roda N combinações de parâmetros.
        """
        self._running = True
        self.results = []

        logger.info(
            f"Starting optimization: {self.symbol} {self.interval}, "
            f"{n_iterations} iterations, {start_time} to {end_time}"
        )

        # Buscar dados uma vez (reutilizar para todas as iterações)
        engine = BacktestEngine()
        df = await engine.fetch_data(self.symbol, self.interval, start_time, end_time)

        if df.empty or len(df) < 100:
            logger.error(f"Insufficient data for optimization ({len(df)} candles)")
            self._running = False
            return []

        logger.info(f"Data loaded: {len(df)} candles. Running {n_iterations} parameter sets...")

        for i in range(n_iterations):
            if not self._running:
                logger.info("Optimization stopped by user")
                break

            params = self.random_params()
            bt = BacktestEngine(params=params)
            metrics = bt.run_on_dataframe(df, max_hold_bars=max_hold_bars)
            score = self.calculate_score(metrics)

            result = OptimizationResult(
                params=params,
                metrics=metrics,
                score=score,
                iteration=i + 1,
            )
            self.results.append(result)

            if self.best_result is None or score > self.best_result.score:
                self.best_result = result
                logger.info(
                    f"[{i+1}/{n_iterations}] New best: score={score:.4f}, "
                    f"WR={metrics.win_rate:.1f}%, Return={metrics.total_return_pct:.2f}%, "
                    f"Sharpe={metrics.sharpe_ratio:.2f}, DD={metrics.max_drawdown_pct:.2f}%"
                )

            if (i + 1) % 10 == 0:
                logger.info(f"[{i+1}/{n_iterations}] Progress... best score={self.best_result.score:.4f}")

        # Ordenar por score
        self.results.sort(key=lambda r: r.score, reverse=True)
        self._running = False

        logger.info(
            f"Optimization complete. Best score={self.results[0].score:.4f}, "
            f"{self.results[0].metrics.total_trades} trades"
        )

        return self.results

    async def walk_forward(
        self,
        full_start: datetime,
        full_end: datetime,
        n_windows: int = 4,
        in_sample_ratio: float = 0.7,
        n_iterations_per_window: int = 30,
        max_hold_bars: int = 48,
    ) -> List[WalkForwardWindow]:
        """
        Walk-forward optimization.

        Divide os dados em N janelas. Para cada janela:
        1. Treina (otimiza) nos dados in-sample (70%)
        2. Valida nos dados out-of-sample (30%)

        Retorna a performance agregada out-of-sample.
        """
        total_days = (full_end - full_start).days
        window_days = total_days // n_windows

        if window_days < 7:
            logger.error(f"Window too small: {window_days} days. Need at least 7.")
            return []

        in_sample_days = int(window_days * in_sample_ratio)
        oos_days = window_days - in_sample_days

        logger.info(
            f"Walk-forward: {n_windows} windows, {in_sample_days}d in-sample, "
            f"{oos_days}d out-of-sample, {n_iterations_per_window} iterations/window"
        )

        # Buscar dados completos uma vez
        engine = BacktestEngine()
        full_df = await engine.fetch_data(self.symbol, self.interval, full_start, full_end)

        if full_df.empty or len(full_df) < 200:
            logger.error(f"Insufficient data for walk-forward ({len(full_df)} candles)")
            return []

        # Garantir coluna timestamp
        if 'timestamp' not in full_df.columns:
            full_df = full_df.reset_index()

        windows: List[WalkForwardWindow] = []

        for w in range(n_windows):
            window_start = full_start + timedelta(days=w * window_days)
            is_end = window_start + timedelta(days=in_sample_days)
            oos_start = is_end
            oos_end = window_start + timedelta(days=window_days)

            # Filtrar DataFrames para cada período
            is_mask = (full_df['timestamp'] >= window_start) & (full_df['timestamp'] < is_end)
            oos_mask = (full_df['timestamp'] >= oos_start) & (full_df['timestamp'] < oos_end)

            is_df = full_df[is_mask].copy()
            oos_df = full_df[oos_mask].copy()

            if len(is_df) < 50 or len(oos_df) < 10:
                logger.warning(f"Window {w+1}: insufficient data (IS={len(is_df)}, OOS={len(oos_df)})")
                continue

            logger.info(f"Window {w+1}/{n_windows}: IS={len(is_df)} candles, OOS={len(oos_df)} candles")

            # Otimizar em in-sample
            best_score = 0.0
            best_params = BacktestParams()
            best_is_metrics = BacktestMetrics()

            for i in range(n_iterations_per_window):
                params = self.random_params()
                bt = BacktestEngine(params=params)
                is_metrics = bt.run_on_dataframe(is_df, max_hold_bars=max_hold_bars)
                score = self.calculate_score(is_metrics)

                if score > best_score:
                    best_score = score
                    best_params = params
                    best_is_metrics = is_metrics

            # Validar melhores params em out-of-sample
            bt_oos = BacktestEngine(params=best_params)
            oos_metrics = bt_oos.run_on_dataframe(oos_df, max_hold_bars=max_hold_bars)
            oos_score = self.calculate_score(oos_metrics)

            wf_window = WalkForwardWindow(
                in_sample_start=window_start,
                in_sample_end=is_end,
                out_of_sample_start=oos_start,
                out_of_sample_end=oos_end,
                best_params=best_params,
                in_sample_metrics=best_is_metrics,
                out_of_sample_metrics=oos_metrics,
                in_sample_score=best_score,
                out_of_sample_score=oos_score,
            )
            windows.append(wf_window)

            logger.info(
                f"Window {w+1}: IS score={best_score:.4f} → OOS score={oos_score:.4f} "
                f"(WR={oos_metrics.win_rate:.1f}%, Return={oos_metrics.total_return_pct:.2f}%)"
            )

        # Resumo
        if windows:
            avg_is = np.mean([w.in_sample_score for w in windows])
            avg_oos = np.mean([w.out_of_sample_score for w in windows])
            degradation = (avg_is - avg_oos) / avg_is * 100 if avg_is > 0 else 0
            logger.info(
                f"Walk-forward complete: avg IS={avg_is:.4f}, avg OOS={avg_oos:.4f}, "
                f"degradation={degradation:.1f}%"
            )

        return windows

    def stop(self):
        """Para a otimização em andamento."""
        self._running = False

    def get_top_results(self, n: int = 10) -> List[OptimizationResult]:
        """Retorna os N melhores resultados."""
        return sorted(self.results, key=lambda r: r.score, reverse=True)[:n]

    def save_results(self, filepath: Optional[str] = None) -> str:
        """Salva resultados em JSON."""
        if not filepath:
            ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            filepath = f"data/optimization_{self.symbol}_{ts}.json"

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        output = {
            "symbol": self.symbol,
            "interval": self.interval,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_iterations": len(self.results),
            "score_formula": "30% win_rate + 30% return + 20% sharpe + 20% (1-drawdown)",
            "score_origin": "sinais/backtesting/optimization_engine.py",
            "top_results": [],
        }

        for r in self.get_top_results(20):
            output["top_results"].append({
                "iteration": r.iteration,
                "score": r.score,
                "params": asdict(r.params),
                "metrics": {
                    "total_trades": r.metrics.total_trades,
                    "win_rate": round(r.metrics.win_rate, 2),
                    "total_return_pct": round(r.metrics.total_return_pct, 2),
                    "sharpe_ratio": round(r.metrics.sharpe_ratio, 2),
                    "sortino_ratio": round(r.metrics.sortino_ratio, 2),
                    "max_drawdown_pct": round(r.metrics.max_drawdown_pct, 2),
                    "profit_factor": round(r.metrics.profit_factor, 2),
                    "avg_winner_pct": round(r.metrics.avg_winner_pct, 2),
                    "avg_loser_pct": round(r.metrics.avg_loser_pct, 2),
                    "max_consecutive_losses": r.metrics.max_consecutive_losses,
                },
            })

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Results saved to {filepath}")
        return filepath

    def save_walk_forward_results(self, windows: List[WalkForwardWindow],
                                  filepath: Optional[str] = None) -> str:
        """Salva resultados do walk-forward em JSON."""
        if not filepath:
            ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            filepath = f"data/walk_forward_{self.symbol}_{ts}.json"

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        output = {
            "symbol": self.symbol,
            "interval": self.interval,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_windows": len(windows),
            "score_formula": "30% win_rate + 30% return + 20% sharpe + 20% (1-drawdown)",
            "score_origin": "sinais/backtesting/optimization_engine.py",
            "windows": [],
        }

        for w in windows:
            oos_m = w.out_of_sample_metrics or BacktestMetrics()
            window_data = {
                "in_sample": {
                    "start": w.in_sample_start.isoformat(),
                    "end": w.in_sample_end.isoformat(),
                    "score": round(w.in_sample_score, 4),
                },
                "out_of_sample": {
                    "start": w.out_of_sample_start.isoformat(),
                    "end": w.out_of_sample_end.isoformat(),
                    "score": round(w.out_of_sample_score, 4),
                    "trades": oos_m.total_trades,
                    "win_rate": round(oos_m.win_rate, 2),
                    "return_pct": round(oos_m.total_return_pct, 2),
                    "sharpe": round(oos_m.sharpe_ratio, 2),
                    "max_drawdown": round(oos_m.max_drawdown_pct, 2),
                },
                "best_params": asdict(w.best_params) if w.best_params else {},
            }
            output["windows"].append(window_data)

        # Agregados
        if windows:
            output["summary"] = {
                "avg_in_sample_score": round(np.mean([w.in_sample_score for w in windows]), 4),
                "avg_oos_score": round(np.mean([w.out_of_sample_score for w in windows]), 4),
                "total_oos_trades": sum(
                    (w.out_of_sample_metrics.total_trades if w.out_of_sample_metrics else 0)
                    for w in windows
                ),
                "avg_oos_win_rate": round(np.mean([
                    w.out_of_sample_metrics.win_rate
                    for w in windows if w.out_of_sample_metrics
                ]), 2),
            }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Walk-forward results saved to {filepath}")
        return filepath


# ==============================================================================
# Funções de conveniência (exportadas via __init__.py)
# ==============================================================================

async def run_optimization(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    days_back: int = 90,
    n_iterations: int = 50,
) -> Dict[str, Any]:
    """
    Função de conveniência para rodar otimização rápida.

    Returns:
        Dict com melhores parâmetros, score e métricas.
    """
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days_back)

    engine = OptimizationEngine(symbol=symbol, interval=interval)
    results = await engine.run_optimization(start_time, end_time, n_iterations=n_iterations)

    if not results:
        return {"error": "No results from optimization"}

    best = results[0]
    filepath = engine.save_results()

    return {
        "best_score": best.score,
        "best_params": asdict(best.params),
        "metrics": {
            "total_trades": best.metrics.total_trades,
            "win_rate": best.metrics.win_rate,
            "total_return_pct": best.metrics.total_return_pct,
            "sharpe_ratio": best.metrics.sharpe_ratio,
            "max_drawdown_pct": best.metrics.max_drawdown_pct,
        },
        "results_file": filepath,
        "total_iterations": len(results),
    }


def apply_best_params(result: Dict[str, Any]) -> BacktestParams:
    """
    Converte resultado da otimização em BacktestParams aplicáveis.
    """
    if "best_params" not in result:
        return BacktestParams()
    return BacktestParams(**result["best_params"])
