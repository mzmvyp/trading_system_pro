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


def _load_config_file(filepath: Path, label: str) -> Optional[BacktestParams]:
    """Carrega um arquivo de config e retorna BacktestParams ou None."""
    if not filepath.exists():
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        config_data = data.get("params", {})
        if not config_data:
            return None

        valid_fields = BacktestParams.__dataclass_fields__.keys()
        filtered = {k: v for k, v in config_data.items() if k in valid_fields}

        params = BacktestParams(**filtered)
        age_hours = (datetime.now(timezone.utc) - datetime.fromisoformat(data.get("updated_at", "2020-01-01T00:00:00+00:00"))).total_seconds() / 3600

        logger.info(
            f"[OPTIMIZER] Loaded {label} config: "
            f"score={data.get('score', 0):.4f}, age={age_hours:.1f}h, "
            f"origin={data.get('origin', 'unknown')}"
        )
        return params
    except Exception as e:
        logger.warning(f"[OPTIMIZER] Error loading {label} config: {e}")
        return None


def _widen_sl_for_dynamic_pair(params: BacktestParams, mover_type: str) -> BacktestParams:
    """
    Amplia multiplicadores de SL/TP para pares dinâmicos (gainers/losers).

    Os category configs são média de BTC/ETH/SOL — pares fixos com volatilidade baixa.
    Micro-cap altcoins (top movers) têm 3-10x mais volatilidade, então precisam de SL
    mais largo para não serem stopados no ruído normal do candle.

    Fator: 1.5x para SL, 1.3x para TP (TP não precisa tanto ajuste).
    """
    SL_WIDEN_FACTOR = 1.5   # SL 50% mais largo
    TP_WIDEN_FACTOR = 1.3   # TP 30% mais largo (para compensar R:R)

    from dataclasses import replace
    widened = replace(
        params,
        sl_atr_multiplier=round(params.sl_atr_multiplier * SL_WIDEN_FACTOR, 2),
        tp1_atr_multiplier=round(params.tp1_atr_multiplier * TP_WIDEN_FACTOR, 2),
        tp2_atr_multiplier=round(params.tp2_atr_multiplier * TP_WIDEN_FACTOR, 2),
    )
    logger.info(
        f"[OPTIMIZER] SL ampliado para par dinâmico ({mover_type}): "
        f"SL_ATR={params.sl_atr_multiplier} → {widened.sl_atr_multiplier}, "
        f"TP1_ATR={params.tp1_atr_multiplier} → {widened.tp1_atr_multiplier}"
    )
    return widened


def load_best_config(symbol: str, interval: str = "1h", mover_type: Optional[str] = None) -> Optional[BacktestParams]:
    """
    Carrega o melhor config com fallback por categoria.

    Prioridade:
    1. Config específica do símbolo (best_config_BTCUSDT_1h.json)
    2. Config da categoria gainer/loser (best_config__GAINERS__1h.json)
    3. Config default genérica (best_config__DEFAULT__1h.json)
    4. None (usa hardcoded defaults)

    Args:
        symbol: Par de trading (ex: BTCUSDT)
        interval: Timeframe (default: 1h)
        mover_type: "gainer", "loser" ou None para pares fixos
    """
    # 1. Config específica do símbolo
    result = _load_config_file(
        get_best_config_path(symbol, interval),
        f"{symbol} {interval}"
    )
    if result:
        return result

    # 2. Config da categoria (gainer/loser)
    # Para pares dinâmicos: ampliar SL porque são mais voláteis que os pares fixos
    # de onde os params foram derivados (BTC/ETH/SOL → micro-caps)
    if mover_type:
        category = "_GAINERS_" if mover_type == "gainer" else "_LOSERS_"
        result = _load_config_file(
            get_best_config_path(category, interval),
            f"{category} {interval}"
        )
        if result:
            result = _widen_sl_for_dynamic_pair(result, mover_type)
            return result

    # 3. Config default genérica (otimizada multi-par)
    result = _load_config_file(
        get_best_config_path("_DEFAULT_", interval),
        f"_DEFAULT_ {interval}"
    )
    if result:
        if mover_type:
            result = _widen_sl_for_dynamic_pair(result, mover_type)
        return result

    # 4. Nenhuma config encontrada
    logger.debug(f"[OPTIMIZER] Nenhuma config encontrada para {symbol} (mover_type={mover_type})")
    return None


def save_best_config(symbol: str, interval: str, params: BacktestParams,
                     score: float, metrics: Dict, origin: str = "continuous_optimizer") -> str:
    """Salva o melhor config encontrado.

    Só sobrescreve se o novo score for >= ao existente (evita que um optimizer
    com score inferior apague o resultado de outro).
    """
    BEST_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    filepath = get_best_config_path(symbol, interval)

    # Verificar se config existente tem score superior
    if filepath.exists():
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                existing = json.load(f)
            existing_score = existing.get("score", 0)
            existing_origin = existing.get("origin", "unknown")
            age_hours = (datetime.now(timezone.utc) - datetime.fromisoformat(
                existing.get("updated_at", "2020-01-01T00:00:00+00:00")
            )).total_seconds() / 3600

            # Manter config existente se: score maior E não muito antiga (<48h)
            if existing_score > score and age_hours < 48:
                logger.info(
                    f"[OPTIMIZER] Config existente mantida para {symbol}: "
                    f"existing={existing_score:.4f} ({existing_origin}) > new={score:.4f} ({origin})"
                )
                return str(filepath)
        except Exception:
            pass

    data = {
        "symbol": symbol,
        "interval": interval,
        "params": asdict(params),
        "score": round(score, 6),
        "score_formula": "30% win_rate + 30% return + 20% sharpe + 20% (1-drawdown)",
        "metrics": metrics,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "origin": origin,
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"[OPTIMIZER] Best config saved: {filepath} (score={score:.4f}, origin={origin})")
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
                        origin="continuous_optimizer",
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

        # Gerar configs de categoria a partir dos melhores resultados individuais
        await self._generate_category_configs()

    async def _generate_category_configs(self):
        """
        Gera configs agregadas por categoria para pares dinâmicos.

        - _DEFAULT_: média dos melhores params de todos os pares fixos
        - _GAINERS_: média dos pares com tendência de alta (top movers positivos)
        - _LOSERS_: média dos pares com tendência de queda (top movers negativos)

        Pares dinâmicos (top gainers/losers) que não têm config própria
        usam essas configs como fallback.
        """
        try:
            from dataclasses import fields

            # Carregar todas as configs existentes
            all_configs = {}
            for symbol in self.symbols:
                cfg = _load_config_file(
                    get_best_config_path(symbol, self.interval),
                    f"{symbol} (para agregação)"
                )
                if cfg:
                    all_configs[symbol] = cfg

            if len(all_configs) < 2:
                logger.warning("[OPTIMIZER] Configs insuficientes para gerar categorias")
                return

            # Separar por tendência recente (últimas 24h)
            gainers_configs = []
            losers_configs = []
            try:
                from src.analysis.top_movers import get_top_movers
                movers = await get_top_movers()
                gainer_symbols = {m["symbol"] for m in movers.get("gainers", [])}
                loser_symbols = {m["symbol"] for m in movers.get("losers", [])}

                for sym, cfg in all_configs.items():
                    if sym in gainer_symbols:
                        gainers_configs.append(cfg)
                    elif sym in loser_symbols:
                        losers_configs.append(cfg)
            except Exception as e:
                logger.debug(f"[OPTIMIZER] Não foi possível classificar por movers: {e}")

            # Função para calcular média de BacktestParams
            def _average_params(configs: list) -> BacktestParams:
                if not configs:
                    return BacktestParams()  # defaults
                avg = {}
                for field in fields(BacktestParams):
                    values = [getattr(c, field.name) for c in configs]
                    if field.type == int:
                        avg[field.name] = int(round(sum(values) / len(values)))
                    else:
                        avg[field.name] = round(sum(values) / len(values), 4)
                return BacktestParams(**avg)

            all_list = list(all_configs.values())

            # _DEFAULT_: média de todos os pares fixos
            default_params = _average_params(all_list)
            save_best_config("_DEFAULT_", self.interval, default_params,
                             score=0.5, metrics={"source": "average_all_fixed", "n_symbols": len(all_list)})
            logger.info(f"[OPTIMIZER] _DEFAULT_ config gerada (média de {len(all_list)} pares)")

            # _GAINERS_: média dos pares que são gainers (ou todos se não há dados)
            if gainers_configs:
                gainer_params = _average_params(gainers_configs)
                save_best_config("_GAINERS_", self.interval, gainer_params,
                                 score=0.5, metrics={"source": "average_gainers", "n_symbols": len(gainers_configs)})
                logger.info(f"[OPTIMIZER] _GAINERS_ config gerada ({len(gainers_configs)} pares)")

            # _LOSERS_: média dos pares que são losers (ou todos se não há dados)
            if losers_configs:
                loser_params = _average_params(losers_configs)
                save_best_config("_LOSERS_", self.interval, loser_params,
                                 score=0.5, metrics={"source": "average_losers", "n_symbols": len(losers_configs)})
                logger.info(f"[OPTIMIZER] _LOSERS_ config gerada ({len(losers_configs)} pares)")

        except Exception as e:
            logger.warning(f"[OPTIMIZER] Erro ao gerar configs de categoria: {e}")


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
