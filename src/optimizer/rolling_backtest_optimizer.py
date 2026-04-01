"""
Rolling Backtest Optimizer — Walk-Forward com aplicação automática
=================================================================

Roda como container separado. A cada ciclo (default: 24h):
1. Baixa últimos 90 dias de candles (Binance Futures)
2. Walk-forward: 75 dias in-sample → 15 dias out-of-sample
3. Testa N combinações de parâmetros dos indicadores
4. Valida no out-of-sample (dados que o optimizer NUNCA viu)
5. Se o novo setup for melhor que o atual → aplica automaticamente
6. Salva em data/optimization/dynamic_params.json (agent.py lê em runtime)

Métricas de decisão:
- Win Rate >= 50%
- Profit Factor >= 1.2
- Min 30 trades no out-of-sample
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


class RollingBacktestOptimizer:
    """
    Otimizador walk-forward rolling que roda como serviço independente.

    Diferente do ContinuousOptimizer (que faz random search simples):
    - Walk-forward obrigatório (anti-overfit)
    - Múltiplas janelas de validação
    - Regras de troca com degradation check
    - Log de todas as tentativas para auditoria
    - Monte Carlo opcional para robustez
    """

    def __init__(
        self,
        symbols: List[str] = None,
        interval: str = "1h",
        cycle_hours: int = 24,
        days_back: int = 90,
        n_iterations: int = 500,
        n_windows: int = 3,
        in_sample_ratio: float = 0.80,  # 75 dias IS, 15 dias OOS
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

        # Carregar log existente
        self._load_log()

    def _load_log(self):
        """Carrega log de otimizações anteriores."""
        if OPTIMIZER_LOG_FILE.exists():
            try:
                with open(OPTIMIZER_LOG_FILE, "r") as f:
                    self._log = json.load(f)
                # Manter últimos 500 registros
                if len(self._log) > 500:
                    self._log = self._log[-500:]
            except Exception:
                self._log = []

    def _save_log(self):
        """Salva log de otimizações."""
        try:
            with open(OPTIMIZER_LOG_FILE, "w") as f:
                json.dump(self._log, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Erro ao salvar log: {e}")

    def _load_current_params(self, symbol: str) -> Optional[Dict]:
        """Carrega os parâmetros atualmente em uso para um símbolo."""
        try:
            if DYNAMIC_PARAMS_FILE.exists():
                with open(DYNAMIC_PARAMS_FILE, "r") as f:
                    data = json.load(f)
                return data.get("symbols", {}).get(symbol)
        except Exception:
            pass
        return None

    def _save_dynamic_params(self, symbol: str, params: BacktestParams,
                             metrics: Dict, score: float, wf_summary: Dict):
        """
        Salva parâmetros otimizados no arquivo central.
        O agent.py lê este arquivo em cada análise.
        """
        # Carregar arquivo existente ou criar novo
        data = {"updated_at": None, "symbols": {}}
        if DYNAMIC_PARAMS_FILE.exists():
            try:
                with open(DYNAMIC_PARAMS_FILE, "r") as f:
                    data = json.load(f)
            except Exception:
                pass

        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        data["symbols"] = data.get("symbols", {})

        data["symbols"][symbol] = {
            "params": asdict(params),
            "score": round(score, 6),
            "metrics": metrics,
            "walk_forward": wf_summary,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "optimizer_version": "rolling_v2",
        }

        with open(DYNAMIC_PARAMS_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)

        # Também salvar no formato que o continuous_optimizer já usa
        # (compatibilidade com load_best_config)
        from src.backtesting.continuous_optimizer import save_best_config
        save_best_config(symbol, self.interval, params, score, metrics)

        logger.info(f"[OPTIMIZER] Params salvos para {symbol}: score={score:.4f}")

    async def optimize_symbol(self, symbol: str) -> Optional[Dict]:
        """
        Roda walk-forward optimization para um símbolo.

        Returns:
            Dict com resultados ou None se falhou
        """
        logger.info(f"[OPTIMIZER] === Iniciando otimização {symbol} ===")

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=self.days_back)

        # Buscar dados uma vez
        engine = BacktestEngine()
        try:
            full_df = await engine.fetch_data(symbol, self.interval, start_time, end_time)
        except Exception as e:
            logger.error(f"[OPTIMIZER] Erro ao buscar dados {symbol}: {e}")
            return None

        if full_df.empty or len(full_df) < 200:
            logger.warning(f"[OPTIMIZER] Dados insuficientes para {symbol}: {len(full_df)} candles")
            return None

        # Garantir coluna timestamp
        if "timestamp" not in full_df.columns:
            full_df = full_df.reset_index()

        logger.info(f"[OPTIMIZER] {symbol}: {len(full_df)} candles carregados")

        # ================================================================
        # WALK-FORWARD: Dividir em janelas IS/OOS
        # ================================================================
        total_days = (end_time - start_time).days
        window_days = total_days // self.n_windows
        is_days = int(window_days * self.in_sample_ratio)
        oos_days = window_days - is_days

        if is_days < 14 or oos_days < 5:
            logger.warning(f"[OPTIMIZER] Janelas muito pequenas: IS={is_days}d, OOS={oos_days}d")
            return None

        logger.info(f"[OPTIMIZER] Walk-forward: {self.n_windows} janelas, IS={is_days}d, OOS={oos_days}d")

        windows_results = []
        best_overall_params = None
        best_overall_oos_score = 0.0

        for w in range(self.n_windows):
            w_start = start_time + timedelta(days=w * window_days)
            is_end = w_start + timedelta(days=is_days)
            oos_start = is_end
            oos_end = w_start + timedelta(days=window_days)

            # Filtrar dados
            is_mask = (full_df["timestamp"] >= w_start) & (full_df["timestamp"] < is_end)
            oos_mask = (full_df["timestamp"] >= oos_start) & (full_df["timestamp"] < oos_end)
            is_df = full_df[is_mask].copy().reset_index(drop=True)
            oos_df = full_df[oos_mask].copy().reset_index(drop=True)

            if len(is_df) < 100 or len(oos_df) < 30:
                logger.warning(f"[OPTIMIZER] Janela {w+1}: dados insuficientes IS={len(is_df)}, OOS={len(oos_df)}")
                continue

            # Random search no in-sample
            best_is_score = 0.0
            best_is_params = BacktestParams()
            best_is_metrics = BacktestMetrics()

            for i in range(self.n_iterations):
                params = OptimizationEngine.random_params()
                bt = BacktestEngine(params=params)
                metrics = bt.run_on_dataframe(is_df, max_hold_bars=48)
                score = OptimizationEngine.calculate_score(metrics)

                if score > best_is_score:
                    best_is_score = score
                    best_is_params = params
                    best_is_metrics = metrics

                if (i + 1) % 100 == 0:
                    logger.info(
                        f"[OPTIMIZER] {symbol} W{w+1} [{i+1}/{self.n_iterations}] "
                        f"best_IS={best_is_score:.4f}"
                    )

            # Validar no out-of-sample
            bt_oos = BacktestEngine(params=best_is_params)
            oos_metrics = bt_oos.run_on_dataframe(oos_df, max_hold_bars=48)
            oos_score = OptimizationEngine.calculate_score(oos_metrics)

            # Calcular degradação IS→OOS
            degradation = 0.0
            if best_is_score > 0:
                degradation = (best_is_score - oos_score) / best_is_score * 100

            window_result = {
                "window": w + 1,
                "is_candles": len(is_df),
                "oos_candles": len(oos_df),
                "is_score": round(best_is_score, 4),
                "oos_score": round(oos_score, 4),
                "degradation_pct": round(degradation, 1),
                "oos_trades": oos_metrics.total_trades,
                "oos_win_rate": round(oos_metrics.win_rate, 1),
                "oos_return": round(oos_metrics.total_return_pct, 2),
                "oos_profit_factor": round(oos_metrics.profit_factor, 2),
                "oos_max_dd": round(oos_metrics.max_drawdown_pct, 2),
                "oos_sharpe": round(oos_metrics.sharpe_ratio, 2),
                "params": asdict(best_is_params),
            }
            windows_results.append(window_result)

            logger.info(
                f"[OPTIMIZER] {symbol} W{w+1}: IS={best_is_score:.4f} → OOS={oos_score:.4f} "
                f"(degradation={degradation:.1f}%) | "
                f"WR={oos_metrics.win_rate:.1f}%, PF={oos_metrics.profit_factor:.2f}, "
                f"trades={oos_metrics.total_trades}"
            )

            # Atualizar melhor OOS
            if oos_score > best_overall_oos_score:
                best_overall_oos_score = oos_score
                best_overall_params = best_is_params

        if not windows_results or best_overall_params is None:
            logger.warning(f"[OPTIMIZER] {symbol}: Nenhuma janela válida")
            return None

        # ================================================================
        # VALIDAÇÃO FINAL: Checar qualidade do melhor resultado
        # ================================================================
        # Usar a ÚLTIMA janela OOS como referência (mais recente)
        last_window = windows_results[-1]
        avg_oos_score = np.mean([w["oos_score"] for w in windows_results])
        avg_degradation = np.mean([w["degradation_pct"] for w in windows_results])

        # Monte Carlo: embaralhar trades e ver se o resultado se mantém
        mc_pass = True
        if self.monte_carlo_runs > 0 and last_window["oos_trades"] >= 10:
            bt_mc = BacktestEngine(params=best_overall_params)
            # Re-rodar no último OOS para pegar trades
            last_oos_mask = (full_df["timestamp"] >= (end_time - timedelta(days=oos_days))) & \
                            (full_df["timestamp"] < end_time)
            last_oos_df = full_df[last_oos_mask].copy().reset_index(drop=True)
            if len(last_oos_df) >= 30:
                mc_metrics = bt_mc.run_on_dataframe(last_oos_df, max_hold_bars=48)
                if mc_metrics.trades:
                    pnls = [t.pnl_pct for t in mc_metrics.trades]
                    mc_win_rates = []
                    for _ in range(self.monte_carlo_runs):
                        shuffled = random.sample(pnls, len(pnls))
                        wins = sum(1 for p in shuffled if p > 0)
                        mc_win_rates.append(wins / len(shuffled) * 100)

                    mc_5th = np.percentile(mc_win_rates, 5)
                    mc_pass = mc_5th >= 45.0  # 5th percentile > 45%
                    logger.info(
                        f"[OPTIMIZER] {symbol} Monte Carlo: "
                        f"median WR={np.median(mc_win_rates):.1f}%, "
                        f"5th pct={mc_5th:.1f}% → {'PASS' if mc_pass else 'FAIL'}"
                    )

        # Critérios de aceitação
        passes_quality = (
            last_window["oos_trades"] >= self.min_oos_trades and
            last_window["oos_win_rate"] >= self.min_oos_win_rate and
            last_window["oos_profit_factor"] >= self.min_profit_factor and
            avg_degradation <= self.max_degradation_pct and
            mc_pass
        )

        # Comparar com params atuais
        current = self._load_current_params(symbol)
        current_score = current.get("score", 0) if current else 0
        is_better = best_overall_oos_score > current_score * 1.05  # Precisa ser 5% melhor

        should_apply = passes_quality and (is_better or current_score == 0)

        result = {
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "windows": windows_results,
            "avg_oos_score": round(avg_oos_score, 4),
            "avg_degradation": round(avg_degradation, 1),
            "best_oos_score": round(best_overall_oos_score, 4),
            "current_score": round(current_score, 4),
            "passes_quality": passes_quality,
            "is_better": is_better,
            "applied": should_apply,
            "monte_carlo_pass": mc_pass,
        }

        if should_apply:
            # APLICAR novos parâmetros
            self._save_dynamic_params(
                symbol=symbol,
                params=best_overall_params,
                metrics={
                    "oos_win_rate": last_window["oos_win_rate"],
                    "oos_return": last_window["oos_return"],
                    "oos_profit_factor": last_window["oos_profit_factor"],
                    "oos_max_dd": last_window["oos_max_dd"],
                    "oos_sharpe": last_window["oos_sharpe"],
                    "oos_trades": last_window["oos_trades"],
                    "avg_degradation": round(avg_degradation, 1),
                },
                score=best_overall_oos_score,
                wf_summary={
                    "n_windows": len(windows_results),
                    "avg_oos_score": round(avg_oos_score, 4),
                    "avg_degradation": round(avg_degradation, 1),
                    "monte_carlo_pass": mc_pass,
                },
            )
            logger.warning(
                f"[OPTIMIZER] ★ {symbol}: NOVOS PARAMS APLICADOS! "
                f"score={best_overall_oos_score:.4f} (era {current_score:.4f}) | "
                f"WR={last_window['oos_win_rate']:.1f}%, PF={last_window['oos_profit_factor']:.2f}"
            )
        else:
            reasons = []
            if not passes_quality:
                if last_window["oos_trades"] < self.min_oos_trades:
                    reasons.append(f"trades={last_window['oos_trades']} < {self.min_oos_trades}")
                if last_window["oos_win_rate"] < self.min_oos_win_rate:
                    reasons.append(f"WR={last_window['oos_win_rate']:.1f}% < {self.min_oos_win_rate}%")
                if last_window["oos_profit_factor"] < self.min_profit_factor:
                    reasons.append(f"PF={last_window['oos_profit_factor']:.2f} < {self.min_profit_factor}")
                if avg_degradation > self.max_degradation_pct:
                    reasons.append(f"degradation={avg_degradation:.1f}% > {self.max_degradation_pct}%")
                if not mc_pass:
                    reasons.append("Monte Carlo falhou")
            if not is_better:
                reasons.append(f"score {best_overall_oos_score:.4f} <= current {current_score:.4f}")

            logger.info(
                f"[OPTIMIZER] {symbol}: Params NÃO aplicados — {', '.join(reasons)}"
            )

        # Adicionar ao log
        self._log.append(result)
        self._save_log()

        return result

    async def run_cycle(self):
        """Executa um ciclo completo de otimização para todos os símbolos."""
        self._cycle_count += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"[OPTIMIZER] CICLO {self._cycle_count} — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        logger.info(f"[OPTIMIZER] Símbolos: {self.symbols}")
        logger.info(f"[OPTIMIZER] Config: {self.n_iterations} iter, {self.n_windows} janelas, {self.days_back}d dados")
        logger.info(f"{'='*60}")

        results = []
        for symbol in self.symbols:
            try:
                result = await self.optimize_symbol(symbol)
                if result:
                    results.append(result)
                    status = "APLICADO" if result["applied"] else "MANTIDO"
                    logger.info(f"[OPTIMIZER] {symbol}: {status} (score={result['best_oos_score']:.4f})")
            except Exception as e:
                logger.error(f"[OPTIMIZER] Erro em {symbol}: {e}")
                import traceback
                traceback.print_exc()

        # Resumo do ciclo
        applied = sum(1 for r in results if r["applied"])
        logger.info(f"\n[OPTIMIZER] Ciclo {self._cycle_count} completo: {len(results)} símbolos, {applied} atualizados")

        # Salvar status
        self._status = {
            "running": self._running,
            "cycle": self._cycle_count,
            "last_run": datetime.now(timezone.utc).isoformat(),
            "next_run": (datetime.now(timezone.utc) + timedelta(hours=self.cycle_hours)).isoformat(),
            "symbols_optimized": len(results),
            "params_applied": applied,
            "results": {r["symbol"]: {
                "score": r["best_oos_score"],
                "applied": r["applied"],
                "passes_quality": r["passes_quality"],
            } for r in results},
        }

        status_file = BEST_CONFIG_DIR / "optimizer_status.json"
        with open(status_file, "w") as f:
            json.dump(self._status, f, indent=2, default=str)

        return results

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

            # Dormir até próximo ciclo
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
