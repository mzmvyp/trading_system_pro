"""
Continuous Optimizer - Otimização contínua que retroalimenta o sistema live.
============================================================================

Inspirado em: sinais/backtesting/optimization_engine.py
- Roda a cada N horas em background (daemon thread)
- Busca melhores parâmetros de indicadores via backtest
- Salva best_config.json que o sistema live consome
- Score composto: 30% win_rate + 30% return + 20% Sharpe + 20% (1-drawdown)

Diferença do sinais original:
- Usa BinanceClient async (não SQLite)
- Usa TA-Lib (não cálculos manuais)
- Integra com BacktestEngine existente
"""

import asyncio
import json
import threading
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

from src.backtesting.backtest_engine import BacktestParams
from src.backtesting.optimization_engine import OptimizationEngine
from src.core.logger import get_logger

logger = get_logger(__name__)

# Arquivo onde o best config é salvo (o agent.py lê daqui)
BEST_CONFIG_DIR = Path("data/optimization")
BEST_CONFIG_FILE = BEST_CONFIG_DIR / "best_config_{symbol}_{interval}.json"


def get_best_config_path(symbol: str, interval: str) -> Path:
    return BEST_CONFIG_DIR / f"best_config_{symbol}_{interval}.json"


def load_best_config(symbol: str, interval: str = "1h") -> Optional[BacktestParams]:
    """
    Carrega o melhor config salvo pelo otimizador contínuo.
    Usado pelo agent.py para aplicar parâmetros otimizados.

    Returns:
        BacktestParams ou None se não existir.
    """
    filepath = get_best_config_path(symbol, interval)
    if not filepath.exists():
        return None

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        config_data = data.get("params", {})
        if not config_data:
            return None

        # Filtrar apenas campos válidos do BacktestParams
        valid_fields = BacktestParams.__dataclass_fields__.keys()
        filtered = {k: v for k, v in config_data.items() if k in valid_fields}

        params = BacktestParams(**filtered)
        age_hours = (datetime.now(timezone.utc) - datetime.fromisoformat(data.get("updated_at", "2020-01-01T00:00:00+00:00"))).total_seconds() / 3600

        logger.info(
            f"[OPTIMIZER] Loaded best config for {symbol} {interval}: "
            f"score={data.get('score', 0):.4f}, age={age_hours:.1f}h"
        )
        return params

    except Exception as e:
        logger.warning(f"[OPTIMIZER] Error loading best config for {symbol}: {e}")
        return None


def save_best_config(symbol: str, interval: str, params: BacktestParams,
                     score: float, metrics: Dict) -> str:
    """Salva o melhor config encontrado."""
    BEST_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    filepath = get_best_config_path(symbol, interval)

    data = {
        "symbol": symbol,
        "interval": interval,
        "params": asdict(params),
        "score": round(score, 6),
        "score_formula": "30% win_rate + 30% return + 20% sharpe + 20% (1-drawdown)",
        "metrics": metrics,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "origin": "continuous_optimizer (inspired by sinais/optimization_engine.py)",
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"[OPTIMIZER] Best config saved: {filepath} (score={score:.4f})")
    return str(filepath)


class ContinuousOptimizer:
    """
    Otimizador contínuo que roda em background e salva best_config.json.

    Inspirado em sinais/optimization_engine.py:
    - start_continuous() → daemon thread com loop
    - A cada cycle_hours: roda otimização → salva melhor config
    - O agent.py lê o best_config antes de gerar sinais
    """

    def __init__(
        self,
        symbols: list = None,
        interval: str = "1h",
        cycle_hours: int = 6,
        days_back: int = 60,
        n_iterations: int = 300,
        min_score: float = 0.35,
    ):
        self.symbols = symbols or ["BTCUSDT"]
        self.interval = interval
        self.cycle_hours = cycle_hours
        self.days_back = days_back
        self.n_iterations = n_iterations
        self.min_score = min_score
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._status: Dict = {
            "running": False,
            "last_cycle": None,
            "next_cycle": None,
            "best_scores": {},
            "errors": [],
        }

    @property
    def status(self) -> Dict:
        return self._status.copy()

    def start_continuous(self):
        """Inicia otimização contínua em daemon thread."""
        if self._running:
            logger.warning("[OPTIMIZER] Already running")
            return

        self._running = True
        self._status["running"] = True

        def _loop():
            while self._running:
                try:
                    logger.info(f"[OPTIMIZER] Starting cycle for {self.symbols}")

                    # Rodar o loop async dentro da thread
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(self._run_cycle())
                    finally:
                        loop.close()

                    self._status["last_cycle"] = datetime.now(timezone.utc).isoformat()
                    next_time = datetime.now(timezone.utc) + timedelta(hours=self.cycle_hours)
                    self._status["next_cycle"] = next_time.isoformat()

                    logger.info(f"[OPTIMIZER] Next cycle at {next_time.strftime('%H:%M')} UTC")

                    # Dormir até próximo ciclo (verificar parada a cada 60s)
                    sleep_seconds = self.cycle_hours * 3600
                    for _ in range(sleep_seconds // 60):
                        if not self._running:
                            break
                        time.sleep(60)

                except Exception as e:
                    error_msg = f"Cycle error: {e}"
                    logger.error(f"[OPTIMIZER] {error_msg}")
                    self._status["errors"].append({
                        "time": datetime.now(timezone.utc).isoformat(),
                        "error": error_msg,
                    })
                    # Esperar 1h em caso de erro (como no sinais)
                    for _ in range(60):
                        if not self._running:
                            break
                        time.sleep(60)

        self._thread = threading.Thread(target=_loop, daemon=True, name="continuous_optimizer")
        self._thread.start()
        logger.info(f"[OPTIMIZER] Continuous optimization started (cycle={self.cycle_hours}h)")

    def stop(self):
        """Para o otimizador."""
        self._running = False
        self._status["running"] = False
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("[OPTIMIZER] Stopped")

    async def _run_cycle(self):
        """Executa um ciclo de otimização para todos os símbolos."""
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=self.days_back)

        for symbol in self.symbols:
            try:
                logger.info(f"[OPTIMIZER] Optimizing {symbol} {self.interval}...")

                engine = OptimizationEngine(symbol=symbol, interval=self.interval)
                results = await engine.run_optimization(
                    start_time=start_time,
                    end_time=end_time,
                    n_iterations=self.n_iterations,
                )

                if not results:
                    logger.warning(f"[OPTIMIZER] No results for {symbol}")
                    continue

                best = results[0]

                # Salvar apenas se score >= min_score (evita gravar estratégias ruins)
                if best.score >= self.min_score:
                    save_best_config(
                        symbol=symbol,
                        interval=self.interval,
                        params=best.params,
                        score=best.score,
                        metrics={
                            "total_trades": best.metrics.total_trades,
                            "win_rate": round(best.metrics.win_rate, 2),
                            "total_return_pct": round(best.metrics.total_return_pct, 2),
                            "sharpe_ratio": round(best.metrics.sharpe_ratio, 2),
                            "max_drawdown_pct": round(best.metrics.max_drawdown_pct, 2),
                            "profit_factor": round(best.metrics.profit_factor, 2),
                        },
                    )
                    self._status["best_scores"][symbol] = best.score
                else:
                    logger.warning(
                        f"[OPTIMIZER] Score too low for {symbol}: {best.score:.4f} "
                        f"(min {self.min_score}). Config NOT saved."
                    )

                # Salvar resultados completos
                engine.save_results()

            except Exception as e:
                logger.error(f"[OPTIMIZER] Error optimizing {symbol}: {e}")


# Singleton global para acesso de qualquer módulo
_global_optimizer: Optional[ContinuousOptimizer] = None


def get_global_optimizer() -> Optional[ContinuousOptimizer]:
    return _global_optimizer


def start_global_optimizer(symbols: list, interval: str = "1h",
                           cycle_hours: int = 6, min_score: float = 0.35, **kwargs) -> ContinuousOptimizer:
    """Inicia o otimizador global (chamado de main.py)."""
    global _global_optimizer
    if _global_optimizer and _global_optimizer._running:
        return _global_optimizer

    _global_optimizer = ContinuousOptimizer(
        symbols=symbols, interval=interval, cycle_hours=cycle_hours,
        min_score=min_score, **kwargs
    )
    _global_optimizer.start_continuous()
    return _global_optimizer
