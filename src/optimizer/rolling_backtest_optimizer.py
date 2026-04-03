"""
Rolling Backtest Optimizer — Walk-Forward GLOBAL (multi-symbol)
=================================================================

Roda como container separado. A cada ciclo (default: 24h):
1. Baixa últimos 90 dias de candles de TODOS os pares (Binance Futures)
2. Walk-forward: 75 dias in-sample → 15 dias out-of-sample
3. Testa N combinações de parâmetros em TODOS os pares simultaneamente
4. Score = performance AGREGADA (evita overfit em 1 par)
5. Valida no out-of-sample com Monte Carlo
6. Se melhor que o atual → aplica automaticamente para todos os pares
7. Salva em data/optimization/ (agent.py lê em runtime)

GLOBAL vs PER-SYMBOL:
- Cada combinação de params é testada em TODOS os pares
- Score = média ponderada (pares com mais trades pesam mais)
- Um param set que só funciona em BTC mas perde em ETH/SOL é rejeitado
- Muito mais trades no OOS → confiança estatística real

Métricas de decisão:
- Win Rate >= 50% (agregado de todos os pares)
- Profit Factor >= 1.2
- Min 30 trades totais no out-of-sample
- OOS score não degradar mais que 30% do IS score (anti-overfit)
"""

import asyncio
import json
import os
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Garantir que o projeto esteja no path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.backtesting.backtest_engine import BacktestEngine, BacktestMetrics, BacktestParams
from src.backtesting.optimization_engine import OptimizationEngine, PARAM_SPACE
from src.core.logger import get_logger

logger = get_logger(__name__)

# Arquivo central que o agent.py lê
DYNAMIC_PARAMS_FILE = Path("data/optimization/dynamic_params.json")
OPTIMIZER_LOG_FILE = Path("data/optimization/optimizer_log.json")
BEST_CONFIG_DIR = Path("data/optimization")


def _aggregate_metrics(all_metrics: List[BacktestMetrics]) -> Dict:
    """Agrega métricas de múltiplos símbolos em um resumo global."""
    if not all_metrics:
        return {"total_trades": 0, "win_rate": 0, "profit_factor": 0}

    total_trades = sum(m.total_trades for m in all_metrics)
    total_winners = sum(m.winning_trades for m in all_metrics)
    total_return = sum(m.total_return_pct for m in all_metrics)

    all_pnls = []
    for m in all_metrics:
        all_pnls.extend([t.pnl_pct for t in m.trades])

    win_rate = total_winners / total_trades * 100 if total_trades > 0 else 0
    avg_return = total_return / len(all_metrics) if all_metrics else 0

    # Profit factor agregado
    total_gains = sum(max(0, p) for p in all_pnls)
    total_losses = abs(sum(min(0, p) for p in all_pnls))
    profit_factor = total_gains / total_losses if total_losses > 0 else 0

    # Sharpe ratio agregado
    if len(all_pnls) >= 2:
        std = np.std(all_pnls, ddof=1)
        sharpe = np.mean(all_pnls) / std * np.sqrt(252) if std > 0 else 0
    else:
        sharpe = 0

    # Max drawdown (pior de todos)
    max_dd = max((m.max_drawdown_pct for m in all_metrics), default=0)

    return {
        "total_trades": total_trades,
        "winning_trades": total_winners,
        "win_rate": round(win_rate, 2),
        "total_return_pct": round(total_return, 2),
        "avg_return_pct": round(avg_return, 2),
        "profit_factor": round(profit_factor, 2),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "n_symbols": len(all_metrics),
        "all_pnls": all_pnls,
    }


def _global_score(agg: Dict) -> float:
    """
    Score composto global: 30% win_rate + 30% return + 20% Sharpe + 20% (1-drawdown)
    Mesmo critério do OptimizationEngine mas com dados agregados.
    """
    if agg["total_trades"] < 10:
        return 0.0

    win_rate_norm = min(agg["win_rate"] / 100, 1.0)
    return_norm = 1 / (1 + np.exp(-agg["avg_return_pct"] / 10))
    sharpe_norm = 1 / (1 + np.exp(-agg["sharpe_ratio"]))
    drawdown_norm = max(1 - agg["max_drawdown_pct"] / 100, 0.0)

    score = 0.30 * win_rate_norm + 0.30 * return_norm + 0.20 * sharpe_norm + 0.20 * drawdown_norm
    return round(score, 6)


class RollingBacktestOptimizer:
    """
    Otimizador walk-forward GLOBAL que testa params em todos os pares.

    Diferente do ContinuousOptimizer (per-symbol, sem walk-forward):
    - GLOBAL: cada param set testado em TODOS os pares
    - Walk-forward obrigatório (anti-overfit)
    - Score = performance agregada (mais trades = mais confiança)
    - Monte Carlo para robustez
    - Regras de troca com degradation check
    """

    def __init__(
        self,
        symbols: List[str] = None,
        interval: str = "1h",
        cycle_hours: int = 24,
        days_back: int = 180,
        n_iterations: int = 500,
        n_windows: int = 3,
        in_sample_ratio: float = 0.80,
        min_oos_trades: int = 20,
        min_oos_win_rate: float = 50.0,
        min_profit_factor: float = 1.2,
        max_degradation_pct: float = 30.0,
        monte_carlo_runs: int = 50,
    ):
        self.symbols = symbols or [
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
            "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT",
            "DOTUSDT", "LINKUSDT",
        ]
        self.interval = interval
        self.cycle_hours = cycle_hours
        self.days_back = days_back
        self.n_iterations = n_iterations
        self.n_windows = n_windows
        self.in_sample_ratio = in_sample_ratio
        self.min_oos_trades = min_oos_trades
        self.min_oos_win_rate = min_oos_win_rate
        self.min_profit_factor = min_profit_factor
        self.max_degradation_pct = max_degradation_pct
        self.monte_carlo_runs = monte_carlo_runs

        # Estado
        self._running = False
        self._cycle_count = 0
        self._status: Dict[str, Any] = {}
        self._log: List[Dict] = []

        # Criar diretórios
        BEST_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        self._load_log()

    def _load_log(self):
        if OPTIMIZER_LOG_FILE.exists():
            try:
                with open(OPTIMIZER_LOG_FILE, "r") as f:
                    self._log = json.load(f)
                if len(self._log) > 500:
                    self._log = self._log[-500:]
            except Exception:
                self._log = []

    def _save_log(self):
        try:
            with open(OPTIMIZER_LOG_FILE, "w") as f:
                json.dump(self._log, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Erro ao salvar log: {e}")

    def _load_current_global_score(self) -> float:
        """Carrega o score global atual."""
        try:
            if DYNAMIC_PARAMS_FILE.exists():
                with open(DYNAMIC_PARAMS_FILE, "r") as f:
                    data = json.load(f)
                return data.get("global_score", 0)
        except Exception:
            pass
        return 0

    def _save_global_params(self, params: BacktestParams, score: float,
                            metrics: Dict, wf_summary: Dict, per_symbol: Dict):
        """
        Salva parâmetros otimizados globalmente.
        Um único conjunto de params para todos os pares.
        """
        data = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "optimizer_version": "rolling_global_v3",
            "mode": "global",
            "global_score": round(score, 6),
            "global_params": asdict(params),
            "global_metrics": {k: v for k, v in metrics.items() if k != "all_pnls"},
            "walk_forward": wf_summary,
            "per_symbol_metrics": per_symbol,
        }

        # Salvar arquivo central
        with open(DYNAMIC_PARAMS_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)

        # Salvar no formato per-symbol (compatibilidade com load_best_config)
        from src.backtesting.continuous_optimizer import save_best_config
        for symbol in self.symbols:
            sym_metrics = per_symbol.get(symbol, metrics)
            save_best_config(symbol, self.interval, params, score, sym_metrics)

        logger.info(f"[OPTIMIZER] Params GLOBAIS salvos: score={score:.4f}")

    async def _fetch_all_data(self) -> Dict[str, Any]:
        """Baixa dados históricos de todos os símbolos."""
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=self.days_back)
        engine = BacktestEngine()

        all_data = {}
        for symbol in self.symbols:
            try:
                df = await engine.fetch_data(symbol, self.interval, start_time, end_time)
                if df.empty or len(df) < 200:
                    logger.warning(f"[OPTIMIZER] {symbol}: dados insuficientes ({len(df)} candles)")
                    continue
                if "timestamp" not in df.columns:
                    df = df.reset_index()
                all_data[symbol] = df
                logger.info(f"[OPTIMIZER] {symbol}: {len(df)} candles")
            except Exception as e:
                logger.error(f"[OPTIMIZER] {symbol}: erro ao buscar dados: {e}")

        return all_data

    def _run_params_on_all_symbols(
        self, params: BacktestParams, symbol_dfs: Dict[str, Any], max_hold: int = 48
    ) -> tuple:
        """
        Roda um conjunto de params em TODOS os símbolos.

        Returns:
            (aggregated_metrics_dict, list_of_per_symbol_metrics)
        """
        all_metrics = []
        per_symbol = {}
        for symbol, df in symbol_dfs.items():
            bt = BacktestEngine(params=params)
            metrics = bt.run_on_dataframe(df, max_hold_bars=max_hold)
            all_metrics.append(metrics)
            per_symbol[symbol] = {
                "trades": metrics.total_trades,
                "win_rate": round(metrics.win_rate, 1),
                "return": round(metrics.total_return_pct, 2),
                "profit_factor": round(metrics.profit_factor, 2),
                "max_dd": round(metrics.max_drawdown_pct, 2),
            }

        agg = _aggregate_metrics(all_metrics)
        return agg, per_symbol

    async def run_global_optimization(self) -> Optional[Dict]:
        """
        Otimização global walk-forward: testa cada param set em TODOS os pares.

        Fluxo:
        1. Baixar dados de todos os pares
        2. Para cada janela walk-forward:
           a. Dividir dados em IS/OOS
           b. Random search: testar N combinações em TODOS os pares (IS)
           c. Validar melhor no OOS de TODOS os pares
        3. Monte Carlo no OOS agregado
        4. Se passar critérios → aplicar
        """
        logger.info(f"[OPTIMIZER] === Otimização GLOBAL ({len(self.symbols)} pares) ===")

        # 1. Buscar todos os dados
        all_data = await self._fetch_all_data()
        if len(all_data) < 3:
            logger.warning(f"[OPTIMIZER] Apenas {len(all_data)} pares com dados. Mínimo: 3.")
            return None

        symbols_ok = list(all_data.keys())
        logger.info(f"[OPTIMIZER] {len(symbols_ok)} pares carregados: {symbols_ok}")

        # 2. Walk-forward
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=self.days_back)
        total_days = (end_time - start_time).days
        window_days = total_days // self.n_windows
        is_days = int(window_days * self.in_sample_ratio)
        oos_days = window_days - is_days

        if is_days < 14 or oos_days < 5:
            logger.warning(f"[OPTIMIZER] Janelas muito pequenas: IS={is_days}d, OOS={oos_days}d")
            return None

        logger.info(
            f"[OPTIMIZER] Walk-forward: {self.n_windows} janelas, "
            f"IS={is_days}d, OOS={oos_days}d, {self.n_iterations} iter/janela"
        )

        windows_results = []
        best_overall_params = None
        best_overall_oos_score = 0.0
        best_overall_oos_metrics = {}
        best_overall_per_symbol = {}

        for w in range(self.n_windows):
            w_start = start_time + timedelta(days=w * window_days)
            is_end = w_start + timedelta(days=is_days)
            oos_start = is_end
            oos_end = w_start + timedelta(days=window_days)

            # Dividir dados IS/OOS para TODOS os pares
            is_dfs = {}
            oos_dfs = {}
            for sym, df in all_data.items():
                is_mask = (df["timestamp"] >= w_start) & (df["timestamp"] < is_end)
                oos_mask = (df["timestamp"] >= oos_start) & (df["timestamp"] < oos_end)
                is_df = df[is_mask].copy().reset_index(drop=True)
                oos_df = df[oos_mask].copy().reset_index(drop=True)
                if len(is_df) >= 50 and len(oos_df) >= 10:
                    is_dfs[sym] = is_df
                    oos_dfs[sym] = oos_df

            if len(is_dfs) < 3:
                logger.warning(f"[OPTIMIZER] W{w+1}: apenas {len(is_dfs)} pares com dados suficientes")
                continue

            logger.info(f"[OPTIMIZER] W{w+1}: {len(is_dfs)} pares, IS/OOS split ok")

            # Random search no IS GLOBAL
            best_is_score = 0.0
            best_is_params = BacktestParams()
            best_is_agg = {}

            for i in range(self.n_iterations):
                params = OptimizationEngine.random_params()
                agg, _ = self._run_params_on_all_symbols(params, is_dfs)
                score = _global_score(agg)

                if score > best_is_score:
                    best_is_score = score
                    best_is_params = params
                    best_is_agg = agg

                if (i + 1) % 100 == 0:
                    logger.info(
                        f"[OPTIMIZER] W{w+1} [{i+1}/{self.n_iterations}] "
                        f"best_IS={best_is_score:.4f} "
                        f"(WR={best_is_agg.get('win_rate', 0):.1f}%, "
                        f"trades={best_is_agg.get('total_trades', 0)}, "
                        f"PF={best_is_agg.get('profit_factor', 0):.2f})"
                    )

            # Validar no OOS GLOBAL
            oos_agg, oos_per_sym = self._run_params_on_all_symbols(best_is_params, oos_dfs)
            oos_score = _global_score(oos_agg)

            degradation = 0.0
            if best_is_score > 0:
                degradation = (best_is_score - oos_score) / best_is_score * 100

            window_result = {
                "window": w + 1,
                "n_symbols": len(is_dfs),
                "is_score": round(best_is_score, 4),
                "oos_score": round(oos_score, 4),
                "degradation_pct": round(degradation, 1),
                "oos_trades": oos_agg["total_trades"],
                "oos_win_rate": oos_agg["win_rate"],
                "oos_return": oos_agg.get("total_return_pct", 0),
                "oos_profit_factor": oos_agg["profit_factor"],
                "oos_max_dd": oos_agg["max_drawdown_pct"],
                "oos_sharpe": oos_agg["sharpe_ratio"],
                "per_symbol": oos_per_sym,
            }
            windows_results.append(window_result)

            logger.info(
                f"[OPTIMIZER] W{w+1}: IS={best_is_score:.4f} → OOS={oos_score:.4f} "
                f"(degrad={degradation:.1f}%) | "
                f"trades={oos_agg['total_trades']}, WR={oos_agg['win_rate']:.1f}%, "
                f"PF={oos_agg['profit_factor']:.2f}"
            )

            # Log per-symbol no OOS
            for sym, sm in sorted(oos_per_sym.items()):
                logger.info(
                    f"  {sym}: trades={sm['trades']}, WR={sm['win_rate']}%, "
                    f"PF={sm['profit_factor']}, return={sm['return']}%"
                )

            if oos_score > best_overall_oos_score:
                best_overall_oos_score = oos_score
                best_overall_params = best_is_params
                best_overall_oos_metrics = oos_agg
                best_overall_per_symbol = oos_per_sym

        if not windows_results or best_overall_params is None:
            logger.warning("[OPTIMIZER] Nenhuma janela válida")
            return None

        # 3. Monte Carlo no OOS agregado
        last_window = windows_results[-1]
        avg_oos_score = np.mean([w["oos_score"] for w in windows_results])
        avg_degradation = np.mean([w["degradation_pct"] for w in windows_results])

        mc_pass = True
        oos_pnls = best_overall_oos_metrics.get("all_pnls", [])
        if self.monte_carlo_runs > 0 and len(oos_pnls) >= 10:
            mc_win_rates = []
            for _ in range(self.monte_carlo_runs):
                shuffled = random.sample(oos_pnls, len(oos_pnls))
                wins = sum(1 for p in shuffled if p > 0)
                mc_win_rates.append(wins / len(shuffled) * 100)

            mc_5th = np.percentile(mc_win_rates, 5)
            mc_pass = mc_5th >= 45.0
            logger.info(
                f"[OPTIMIZER] Monte Carlo ({len(oos_pnls)} trades): "
                f"median WR={np.median(mc_win_rates):.1f}%, "
                f"5th pct={mc_5th:.1f}% → {'PASS' if mc_pass else 'FAIL'}"
            )

        # 4. Critérios de aceitação
        passes_quality = (
            best_overall_oos_metrics.get("total_trades", 0) >= self.min_oos_trades and
            best_overall_oos_metrics.get("win_rate", 0) >= self.min_oos_win_rate and
            best_overall_oos_metrics.get("profit_factor", 0) >= self.min_profit_factor and
            avg_degradation <= self.max_degradation_pct and
            mc_pass
        )

        current_score = self._load_current_global_score()
        # Require at least 2% improvement (was 5%, which rejected valid improvements)
        is_better = best_overall_oos_score > current_score * 1.02

        should_apply = passes_quality and (is_better or current_score == 0)

        result = {
            "mode": "global",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbols": symbols_ok,
            "n_symbols": len(symbols_ok),
            "windows": windows_results,
            "avg_oos_score": round(avg_oos_score, 4),
            "avg_degradation": round(avg_degradation, 1),
            "best_oos_score": round(best_overall_oos_score, 4),
            "current_score": round(current_score, 4),
            "oos_trades_total": best_overall_oos_metrics.get("total_trades", 0),
            "oos_win_rate": best_overall_oos_metrics.get("win_rate", 0),
            "oos_profit_factor": best_overall_oos_metrics.get("profit_factor", 0),
            "passes_quality": passes_quality,
            "is_better": is_better,
            "applied": should_apply,
            "monte_carlo_pass": mc_pass,
            "per_symbol": best_overall_per_symbol,
            "params": asdict(best_overall_params) if best_overall_params else {},
        }

        if should_apply:
            clean_metrics = {k: v for k, v in best_overall_oos_metrics.items() if k != "all_pnls"}
            self._save_global_params(
                params=best_overall_params,
                score=best_overall_oos_score,
                metrics=clean_metrics,
                wf_summary={
                    "n_windows": len(windows_results),
                    "avg_oos_score": round(avg_oos_score, 4),
                    "avg_degradation": round(avg_degradation, 1),
                    "monte_carlo_pass": mc_pass,
                },
                per_symbol=best_overall_per_symbol,
            )
            logger.warning(
                f"[OPTIMIZER] ★ PARAMS GLOBAIS APLICADOS! "
                f"score={best_overall_oos_score:.4f} (era {current_score:.4f}) | "
                f"WR={best_overall_oos_metrics['win_rate']:.1f}%, "
                f"PF={best_overall_oos_metrics['profit_factor']:.2f}, "
                f"trades={best_overall_oos_metrics['total_trades']} "
                f"({len(symbols_ok)} pares)"
            )
        else:
            reasons = []
            if not passes_quality:
                t = best_overall_oos_metrics.get("total_trades", 0)
                wr = best_overall_oos_metrics.get("win_rate", 0)
                pf = best_overall_oos_metrics.get("profit_factor", 0)
                if t < self.min_oos_trades:
                    reasons.append(f"trades={t} < {self.min_oos_trades}")
                if wr < self.min_oos_win_rate:
                    reasons.append(f"WR={wr:.1f}% < {self.min_oos_win_rate}%")
                if pf < self.min_profit_factor:
                    reasons.append(f"PF={pf:.2f} < {self.min_profit_factor}")
                if avg_degradation > self.max_degradation_pct:
                    reasons.append(f"degradation={avg_degradation:.1f}% > {self.max_degradation_pct}%")
                if not mc_pass:
                    reasons.append("Monte Carlo falhou")
            if not is_better:
                needed = current_score * 1.02
                reasons.append(
                    f"score {best_overall_oos_score:.4f} não superou mínimo de 2% sobre current "
                    f"{current_score:.4f} (precisa {needed:.4f})"
                )

            logger.info(f"[OPTIMIZER] Params NÃO aplicados — {', '.join(reasons)}")

        self._log.append(result)
        self._save_log()
        return result

    async def run_cycle(self):
        """Executa um ciclo completo de otimização global."""
        self._cycle_count += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"[OPTIMIZER] CICLO {self._cycle_count} — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        logger.info(f"[OPTIMIZER] Modo: GLOBAL ({len(self.symbols)} pares)")
        logger.info(f"[OPTIMIZER] Config: {self.n_iterations} iter, {self.n_windows} janelas, {self.days_back}d dados")
        logger.info(f"{'='*60}")

        result = await self.run_global_optimization()

        # Setup Validator: gerar sinais históricos e validar contextos
        setup_validation = None
        try:
            from src.optimizer.setup_validator import SetupValidator
            sv = SetupValidator()
            setup_validation = await sv.run_full_validation(
                symbols=self.symbols, days_back=self.days_back
            )
            logger.info(
                f"[SETUP_VALIDATOR] {setup_validation['total_signals']} sinais validados, "
                f"{setup_validation['total_setups']} setups rastreados"
            )
        except Exception as e:
            logger.warning(f"[SETUP_VALIDATOR] Erro na validação periódica: {e}")

        # Drift Detector: verificar se features mudaram vs baseline
        drift_report = None
        try:
            from src.analysis.drift_detector import get_drift_detector
            import glob
            import json as _json

            detector = get_drift_detector()

            # Coletar sinais recentes (últimos 7 dias) para comparar com baseline
            recent_signals = []
            signal_files = sorted(glob.glob("signals/agno_*.json"), reverse=True)[:200]
            for sf in signal_files:
                try:
                    with open(sf, "r") as f:
                        sig = _json.load(f)
                    # Extrair features relevantes
                    indicators = sig.get("key_indicators", {})
                    recent_signals.append({
                        "rsi": indicators.get("rsi", {}).get("value", sig.get("rsi", 50)),
                        "macd_histogram": indicators.get("macd", {}).get("histogram", sig.get("macd_histogram", 0)),
                        "adx": sig.get("adx", indicators.get("adx", 25)),
                        "atr": sig.get("atr", 0),
                        "bb_position": indicators.get("bollinger", {}).get("position", 0.5),
                        "confidence": sig.get("confidence", 5),
                        "risk_distance_pct": sig.get("risk_distance_pct", 0),
                        "reward_distance_pct": sig.get("reward_distance_pct", 0),
                        "risk_reward_ratio": sig.get("risk_reward_ratio", 0),
                    })
                except Exception:
                    continue

            if len(recent_signals) >= 20:
                # Criar baseline se não existe
                if not detector.baseline:
                    detector.create_baseline_from_signals(recent_signals)
                    logger.info("[DRIFT] Baseline criado a partir dos sinais recentes")
                else:
                    # Coletar predições ML recentes
                    ml_predictions = []
                    votes_log = "signals/model_votes_log.jsonl"
                    if os.path.exists(votes_log):
                        try:
                            with open(votes_log, "r") as f:
                                for line in f:
                                    line = line.strip()
                                    if line:
                                        try:
                                            rec = _json.loads(line)
                                            prob = rec.get("ml_probability", rec.get("ml_prob"))
                                            if prob is not None:
                                                ml_predictions.append(float(prob))
                                        except Exception:
                                            continue
                        except Exception:
                            pass

                    drift_report = detector.generate_report(
                        recent_signals,
                        predictions=ml_predictions[-100:] if ml_predictions else None,
                    )
                    if drift_report.overall_drift_detected:
                        logger.warning(
                            f"[DRIFT] ⚠ DRIFT DETECTADO: {drift_report.overall_severity} | "
                            f"Features: {drift_report.features_with_drift} | "
                            f"Recomendações: {drift_report.recommendations}"
                        )
                    else:
                        logger.info("[DRIFT] Modelo estável — sem drift significativo")
            else:
                logger.info(f"[DRIFT] Apenas {len(recent_signals)} sinais — aguardando mais dados")

        except Exception as e:
            logger.warning(f"[DRIFT] Erro na detecção de drift: {e}")

        # Salvar status
        self._status = {
            "running": self._running,
            "mode": "global",
            "cycle": self._cycle_count,
            "last_run": datetime.now(timezone.utc).isoformat(),
            "next_run": (datetime.now(timezone.utc) + timedelta(hours=self.cycle_hours)).isoformat(),
        }

        if result:
            self._status.update({
                "symbols_count": result.get("n_symbols", 0),
                "best_oos_score": result.get("best_oos_score", 0),
                "oos_win_rate": result.get("oos_win_rate", 0),
                "oos_profit_factor": result.get("oos_profit_factor", 0),
                "oos_trades": result.get("oos_trades_total", 0),
                "applied": result.get("applied", False),
                "passes_quality": result.get("passes_quality", False),
                "per_symbol": result.get("per_symbol", {}),
            })
            status_str = "APLICADO" if result["applied"] else "MANTIDO"
            logger.info(
                f"\n[OPTIMIZER] Ciclo {self._cycle_count}: {status_str} "
                f"(score={result['best_oos_score']:.4f})"
            )
        else:
            logger.warning(f"[OPTIMIZER] Ciclo {self._cycle_count}: sem resultado válido")

        if setup_validation:
            self._status["setup_validation"] = {
                "total_signals": setup_validation.get("total_signals", 0),
                "total_setups": setup_validation.get("total_setups", 0),
                "per_symbol": setup_validation.get("per_symbol", {}),
            }

        if drift_report:
            self._status["drift"] = {
                "detected": drift_report.overall_drift_detected,
                "severity": drift_report.overall_severity,
                "features_with_drift": drift_report.features_with_drift,
                "recommendations": drift_report.recommendations,
            }

        status_file = BEST_CONFIG_DIR / "optimizer_status.json"
        with open(status_file, "w") as f:
            json.dump(self._status, f, indent=2, default=str)

        return [result] if result else []

    async def run_continuous(self):
        """Loop contínuo: roda um ciclo, dorme, repete."""
        self._running = True
        logger.info(f"[OPTIMIZER] Iniciando modo contínuo (ciclo a cada {self.cycle_hours}h)")

        while self._running:
            try:
                await self.run_cycle()
            except Exception as e:
                logger.error(f"[OPTIMIZER] Erro no ciclo: {e}")
                import traceback
                traceback.print_exc()

            sleep_secs = self.cycle_hours * 3600
            logger.info(f"[OPTIMIZER] Próximo ciclo em {self.cycle_hours}h...")
            for _ in range(sleep_secs // 30):
                if not self._running:
                    break
                await asyncio.sleep(30)

    def stop(self):
        self._running = False


def get_optimizer_status() -> Dict:
    """Lê status do optimizer (para o dashboard)."""
    status_file = BEST_CONFIG_DIR / "optimizer_status.json"
    if status_file.exists():
        try:
            with open(status_file, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"running": False, "message": "Optimizer não rodou ainda"}


def get_optimizer_log() -> List[Dict]:
    """Lê log do optimizer (para o dashboard)."""
    if OPTIMIZER_LOG_FILE.exists():
        try:
            with open(OPTIMIZER_LOG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def get_current_dynamic_params() -> Dict:
    """Lê parâmetros dinâmicos atuais (para o dashboard)."""
    if DYNAMIC_PARAMS_FILE.exists():
        try:
            with open(DYNAMIC_PARAMS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}
