"""
Entrypoint para o container Docker do optimizer.

Uso:
    python -m src.optimizer.run_optimizer              # Modo contínuo (default)
    python -m src.optimizer.run_optimizer --once        # Roda uma vez e sai
    python -m src.optimizer.run_optimizer --status      # Mostra status atual

Variáveis de ambiente:
    OPTIMIZER_CYCLE_HOURS=24      Intervalo entre ciclos
    OPTIMIZER_ITERATIONS=500      Combinações a testar por janela
    OPTIMIZER_DAYS_BACK=90        Dias de dados históricos
    OPTIMIZER_WINDOWS=3           Número de janelas walk-forward
    OPTIMIZER_MIN_WR=50           Win rate mínimo para aceitar (%)
    OPTIMIZER_MIN_PF=1.2          Profit factor mínimo
    OPTIMIZER_MC_RUNS=50          Rodadas de Monte Carlo (0=desabilitar)
    OPTIMIZER_SYMBOLS=BTCUSDT,ETHUSDT  Pares (virgula-separados, vazio=top10)
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Garantir root no path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

from src.core.logger import get_logger
from src.optimizer.rolling_backtest_optimizer import (
    RollingBacktestOptimizer,
    get_current_dynamic_params,
    get_optimizer_log,
    get_optimizer_status,
)

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Rolling Backtest Optimizer")
    parser.add_argument("--once", action="store_true", help="Rodar um único ciclo")
    parser.add_argument("--status", action="store_true", help="Mostrar status atual")
    parser.add_argument("--params", action="store_true", help="Mostrar params atuais")
    args = parser.parse_args()

    if args.status:
        status = get_optimizer_status()
        print(json.dumps(status, indent=2))
        return

    if args.params:
        params = get_current_dynamic_params()
        print(json.dumps(params, indent=2))
        return

    # Ler configuração de variáveis de ambiente
    cycle_hours = int(os.getenv("OPTIMIZER_CYCLE_HOURS", "24"))
    n_iterations = int(os.getenv("OPTIMIZER_ITERATIONS", "500"))
    days_back = int(os.getenv("OPTIMIZER_DAYS_BACK", "90"))
    n_windows = int(os.getenv("OPTIMIZER_WINDOWS", "3"))
    min_wr = float(os.getenv("OPTIMIZER_MIN_WR", "50"))
    min_pf = float(os.getenv("OPTIMIZER_MIN_PF", "1.2"))
    mc_runs = int(os.getenv("OPTIMIZER_MC_RUNS", "50"))

    symbols_env = os.getenv("OPTIMIZER_SYMBOLS", "")
    symbols = [s.strip().upper() for s in symbols_env.split(",") if s.strip()] if symbols_env else None

    optimizer = RollingBacktestOptimizer(
        symbols=symbols,
        interval="1h",
        cycle_hours=cycle_hours,
        days_back=days_back,
        n_iterations=n_iterations,
        n_windows=n_windows,
        min_oos_win_rate=min_wr,
        min_profit_factor=min_pf,
        monte_carlo_runs=mc_runs,
    )

    logger.info("=" * 60)
    logger.info("ROLLING BACKTEST OPTIMIZER")
    logger.info(f"Símbolos: {optimizer.symbols}")
    logger.info(f"Ciclo: {cycle_hours}h | Iterações: {n_iterations} | Dados: {days_back}d")
    logger.info(f"Walk-forward: {n_windows} janelas | Monte Carlo: {mc_runs} runs")
    logger.info(f"Critérios: WR>={min_wr}%, PF>={min_pf}")
    logger.info("=" * 60)

    if args.once:
        logger.info("Modo: SINGLE RUN")
        results = asyncio.run(optimizer.run_cycle())
        # Print summary
        for r in results:
            status = "APLICADO" if r["applied"] else "MANTIDO"
            logger.info(f"  {r['symbol']}: {status} (OOS score={r['best_oos_score']:.4f})")
    else:
        logger.info("Modo: CONTÍNUO")
        try:
            asyncio.run(optimizer.run_continuous())
        except KeyboardInterrupt:
            optimizer.stop()
            logger.info("Optimizer parado pelo usuário")


if __name__ == "__main__":
    main()
