"""
Sistema de Trading com AGNO Agent
Updated with logging and improved error handling
"""
import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timedelta, timezone

from src.core.logger import get_logger
from src.trading.agent import AgnoTradingAgent

logger = get_logger(__name__)

def get_active_positions():
    """Retorna símbolos com posições ativas (CORRIGIDO: verifica BUY e SELL)"""
    active_symbols = []

    try:
        if os.path.exists("portfolio/state.json"):
            with open("portfolio/state.json", "r", encoding='utf-8') as f:
                state = json.load(f)
                positions = state.get("positions", {})

                # MODIFICADO: Verificar posições considerando novas chaves (SYMBOL_DEEPSEEK, SYMBOL_AGNO, etc.)
                for key, pos in positions.items():
                    if pos.get("status") == "OPEN":
                        symbol = pos.get("symbol")
                        if symbol:
                            # Extrair símbolo base da chave (pode ser SYMBOL_DEEPSEEK, SYMBOL_AGNO, SYMBOL_DEEPSEEK_SHORT, etc.)
                            base_symbol = symbol  # O símbolo já está limpo na posição
                            if base_symbol not in active_symbols:
                                active_symbols.append(base_symbol)

                logger.debug(f"Loaded {len(active_symbols)} active positions")
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Erro ao carregar posicoes ativas: {e}")
    except Exception as e:
        logger.exception(f"Erro inesperado ao carregar posicoes ativas: {e}")

    return active_symbols


async def log_monitor_summary(settings):
    """No início de cada ciclo do monitor: posições abertas, saldo, próxima análise."""
    logger.info("--- [CICLO] Resumo ---")
    try:
        if settings.trading_mode == "real":
            from src.exchange.executor import BinanceFuturesExecutor
            executor = BinanceFuturesExecutor()
            balance = await executor.get_balance()
            positions = await executor.get_all_positions()
            if isinstance(balance, dict) and "error" not in balance:
                avail = balance.get("available_balance", 0)
                total = balance.get("total_balance", avail)
                logger.info(f"  Saldo: ${total:,.2f} USDT (disponível: ${avail:,.2f})")
            if isinstance(positions, list):
                open_pos = [p for p in positions if isinstance(p, dict) and "error" not in p]
                if open_pos:
                    for p in open_pos:
                        sym = p.get("symbol", "?")
                        side = p.get("side", "?")
                        amt = p.get("position_amt", 0)
                        pnl = p.get("unrealized_pnl", 0)
                        logger.info(f"  Posição: {sym} {side} {abs(amt)} | PnL: ${pnl:,.2f}")
                else:
                    logger.info("  Posições abertas: Nenhuma")
        else:
            active = get_active_positions()
            logger.info(f"  Posições ativas (paper): {active if active else 'Nenhuma'}")
    except Exception as e:
        logger.warning(f"Erro ao obter resumo: {e}")
    # Calcular próxima análise baseada no fechamento do candle de 1h
    from datetime import datetime as _dt
    from datetime import timezone as _tz
    now_utc = _dt.now(_tz.utc)
    next_hour = now_utc.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    next_analysis = next_hour + timedelta(seconds=35)
    wait_minutes = int((next_analysis - now_utc).total_seconds() // 60)
    logger.info(f"  Próxima análise em {wait_minutes}min (candle close: {next_analysis.strftime('%H:%M:%S')} UTC)")


def _register_trade_result(symbol: str, add_trade_result_fn):
    """Registra resultado de trade fechado para online learning."""
    try:
        # Buscar ultimo sinal salvo para este simbolo
        import glob as glob_mod
        signal_files = glob_mod.glob(f"signals/agno_{symbol}_*.json")
        if not signal_files:
            return

        # Pegar o mais recente
        signal_files.sort(key=os.path.getmtime, reverse=True)
        with open(signal_files[0], 'r', encoding='utf-8') as f:
            signal = json.load(f)

        if signal.get('signal') not in ('BUY', 'SELL'):
            return

        # Buscar resultado do trade no portfolio
        result = 'TIMEOUT'
        return_pct = 0.0

        # Try portfolio state (paper trading)
        if os.path.exists("portfolio/state.json"):
            with open("portfolio/state.json", 'r', encoding='utf-8') as f:
                state = json.load(f)
                trade_history = state.get("trade_history", [])
                for trade in reversed(trade_history):
                    if trade.get("symbol") == symbol:
                        pnl = trade.get("pnl", trade.get("pnl_pct", 0))
                        return_pct = float(pnl) if pnl else 0.0
                        if return_pct > 0:
                            result = 'TP1'
                        else:
                            result = 'SL'
                        break

        # Fallback for real mode: try to get PnL from Binance income history
        if result == 'TIMEOUT':
            try:
                import asyncio
                from src.exchange.executor import BinanceFuturesExecutor
                _exec = BinanceFuturesExecutor()
                # Get recent income (realized PnL) for this symbol
                income = asyncio.get_event_loop().run_until_complete(
                    _exec.client.futures_income_history(symbol=symbol, incomeType="REALIZED_PNL", limit=5)
                ) if hasattr(_exec, 'client') else []
                if income:
                    latest = income[-1]
                    pnl_val = float(latest.get("income", 0))
                    return_pct = pnl_val
                    result = 'TP1' if pnl_val > 0 else 'SL'
            except Exception as e:
                logger.debug(f"[ML] Fallback Binance income falhou para {symbol}: {e}")

        # Last resort: infer from signal entry vs TP/SL (positive = TP1, negative = SL)
        if result == 'TIMEOUT':
            # Can't determine result, skip this signal (add_signal_result filters TIMEOUT)
            logger.info(f"[ML] Resultado indeterminado para {symbol}, pulando registro")
            return

        add_trade_result_fn(signal, result, return_pct)
        logger.info(f"[ML] Resultado registrado para online learning: {symbol} -> {result} ({return_pct:+.2f}%)")

    except Exception as e:
        logger.warning(f"Erro ao registrar trade result para {symbol}: {e}")


async def main():
    parser = argparse.ArgumentParser(
        description='Sistema de Trading de Criptomoedas com AGNO Agent'
    )
    parser.add_argument(
        '--symbol',
        default='BTCUSDT',
        help='Símbolo para trading (ex: BTCUSDT)'
    )
    parser.add_argument(
        '--mode',
        choices=['single', 'monitor', 'top5', 'top10', 'cleanup', 'train_ml'],
        default='single',
        help='Modo de operação (cleanup = limpar ordens órfãs, train_ml = treinar modelo ML)'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=900,
        help='Intervalo para monitoramento em segundos (default: 900s = 15min)'
    )
    parser.add_argument(
        '--paper',
        action='store_true',
        default=True,
        help='Usar paper trading (simulado)'
    )

    args = parser.parse_args()

    # Respeitar TRADING_MODE do .env: real = executar na Binance (testnet/prod)
    from src.core.config import settings
    if settings.trading_mode == "real":
        args.paper = False

    # Banner
    logger.info("=" * 60)
    logger.info("SISTEMA DE TRADING COM AGNO AGENT")
    logger.info(f"Simbolo: {args.symbol} | Modo: {args.mode} | Paper: {'Sim' if args.paper else 'Nao'}")
    logger.info("=" * 60)

    logger.info(f"Starting trading system - Mode: {args.mode}, Symbol: {args.symbol}, Paper: {args.paper}")

    # Criar agent
    agent = AgnoTradingAgent(paper_trading=args.paper)

    try:
        if args.mode == 'single':
            # Análise única
            signal = await agent.analyze(args.symbol)

            if signal.get('signal') in ['BUY', 'SELL'] and signal.get('confidence', 0) >= 7:
                logger.info("[ALERTA] Sinal forte detectado! Considere executar o trade com cautela.")

        elif args.mode == 'monitor':
            # Monitoramento contínuo do Top 10 + Top Movers dinâmicos
            from src.core.config import settings

            symbols = list(settings.top_crypto_pairs)  # Pares fixos (cópia)
            dynamic_metadata = {}  # Metadata dos pares dinâmicos (preenchido a cada ciclo)

            logger.info(f"[MONITOR] Monitoramento continuo | Pares fixos: {symbols} | Top Movers: {'ON' if settings.top_movers_enabled else 'OFF'} | Sync: candle 1h close | Modo: {settings.trading_mode.upper()}")

            # ============================================================
            # Otimizador contínuo: roda a cada N h, testa M combinações
            # aleatórias de indicadores (RSI, EMA, MACD, BB, ADX, SL/TP)
            # e salva o melhor em best_config. Agent usa esse config.
            # Ajuste via .env: OPTIMIZER_ITERATIONS, OPTIMIZER_DAYS_BACK, etc.
            # ============================================================
            try:
                from src.backtesting.continuous_optimizer import start_global_optimizer
                _opt_iter = int(os.getenv("OPTIMIZER_ITERATIONS", "300"))
                _opt_days = int(os.getenv("OPTIMIZER_DAYS_BACK", "60"))
                _opt_cycle = int(os.getenv("OPTIMIZER_CYCLE_HOURS", "6"))
                _opt_min_score = float(os.getenv("OPTIMIZER_MIN_SCORE", "0.35"))
                start_global_optimizer(
                    symbols=symbols,
                    interval="1h",
                    cycle_hours=_opt_cycle,
                    days_back=_opt_days,
                    n_iterations=_opt_iter,
                    min_score=_opt_min_score,
                )
                logger.info(f"[OPTIMIZER] Otimizacao continua: {_opt_iter} iter/simbolo, {_opt_days}d dados, ciclo {_opt_cycle}h, min_score={_opt_min_score}")
            except Exception as e:
                logger.warning(f"[OPTIMIZER] Falha ao iniciar otimizador: {e} (continuando sem)")

            # ============================================================
            # Treino automático periódico: ML e LSTM
            # Roda a cada N horas em thread daemon separada
            # ============================================================
            try:
                _ml_train_hours = int(os.getenv("ML_AUTO_TRAIN_HOURS", "6"))
                if _ml_train_hours > 0:
                    import threading

                    def _auto_train_ml_loop(interval_hours: int):
                        """Thread daemon que treina ML/LSTM periodicamente."""
                        import time as _time
                        import traceback as _tb

                        # Garantir diretórios existem
                        import pathlib
                        pathlib.Path("data/drift").mkdir(parents=True, exist_ok=True)
                        pathlib.Path("data/ml").mkdir(parents=True, exist_ok=True)
                        pathlib.Path("logs").mkdir(parents=True, exist_ok=True)

                        logger.info(f"[ML-AUTO] Treino automático iniciado. Ciclo: {interval_hours}h")
                        while True:
                            _time.sleep(interval_hours * 3600)
                            try:
                                logger.info("[ML-AUTO] Iniciando treino automático do ML...")

                                # 1. Alimentar buffer com sinais avaliados
                                from src.ml.online_learning import seed_from_evaluated_signals
                                seed_result = seed_from_evaluated_signals(force_retrain=True)
                                if seed_result.get("success"):
                                    rt = seed_result.get("retrain_result", {})
                                    if rt and rt.get("success"):
                                        logger.info(
                                            f"[ML-AUTO] ML retreinado! Accuracy={rt.get('new_accuracy', 0):.1%}, "
                                            f"F1={rt.get('new_f1', 0):.3f}, Amostras={rt.get('samples_used', 0)}"
                                        )
                                    else:
                                        logger.info(f"[ML-AUTO] Sinais alimentados: {seed_result.get('signals_added', 0)}")
                                else:
                                    logger.warning(f"[ML-AUTO] Seed falhou: {seed_result.get('error', 'unknown')}")

                                # 2. Treinar Bi-LSTM se disponível
                                try:
                                    from src.ml.lstm_sequence_validator import LSTMSequenceValidator
                                    lstm = LSTMSequenceValidator()
                                    lstm_result = lstm.train_from_backtest()
                                    if lstm_result and lstm_result.get("success"):
                                        test_acc = lstm_result.get("test_accuracy", 0)
                                        test_f1 = lstm_result.get("test_f1", 0)
                                        logger.info(
                                            f"[ML-AUTO] Bi-LSTM retreinado! "
                                            f"Accuracy={test_acc:.1%}, F1={test_f1:.3f}, "
                                            f"Amostras={lstm_result.get('total_samples', 0)}"
                                        )
                                    else:
                                        reason = lstm_result.get("reason", "unknown") if lstm_result else "no result"
                                        logger.info(f"[ML-AUTO] Bi-LSTM treino: {reason}")
                                except Exception as lstm_err:
                                    logger.warning(f"[ML-AUTO] LSTM treino falhou: {lstm_err}")

                            except Exception as e:
                                logger.error(f"[ML-AUTO] Erro no treino automático: {e}\n{_tb.format_exc()}")

                    ml_thread = threading.Thread(
                        target=_auto_train_ml_loop,
                        args=(_ml_train_hours,),
                        daemon=True,
                        name="ml-auto-train"
                    )
                    ml_thread.start()
                    logger.info(f"[ML-AUTO] Thread de treino automático iniciada (ciclo: {_ml_train_hours}h)")
            except Exception as e:
                logger.warning(f"[ML-AUTO] Falha ao iniciar treino automático: {e}")

            # Rastrear posições anteriores para detectar fechamentos
            previous_positions = set()
            previous_positions_info = {}  # {symbol: {"side": "SHORT"/"LONG"}} para cooldown direcional
            _cleanup_cycle = 0  # Contador para limpeza periódica de ordens órfãs

            while True:
                try:
                    await log_monitor_summary(settings)
                    # Verificar posições ativas: em modo real = Binance; em paper = state.json
                    if settings.trading_mode == "real":
                        try:
                            from src.exchange.executor import BinanceFuturesExecutor
                            exec_real = BinanceFuturesExecutor()
                            all_pos = await exec_real.get_all_positions()
                            active_positions = [p["symbol"] for p in all_pos if isinstance(p, dict) and "error" not in p and p.get("symbol")]
                            # Salvar info de direção para cooldown direcional
                            for p in all_pos:
                                if isinstance(p, dict) and "error" not in p and p.get("symbol"):
                                    previous_positions_info[p["symbol"]] = {
                                        "side": p.get("side", "UNKNOWN")
                                    }
                        except Exception as e:
                            logger.warning(f"Erro ao obter posições Binance: {e}")
                            active_positions = get_active_positions()
                    else:
                        active_positions = get_active_positions()
                    active_positions_set = set(active_positions)

                    # Detectar posições fechadas (estavam ativas antes, mas não estão mais)
                    closed_positions = previous_positions - active_positions_set
                    if closed_positions:
                        logger.info(f"[POSICAO FECHADA] Detectado fechamento: {closed_positions}")

                        # Registrar cooldowns para evitar whipsaw e reentrada na mesma direção
                        # CORRIGIDO: Aplica cooldown para QUALQUER fechamento (TP ou SL)
                        # Bug fix: DRIFTUSDT bateu TP2 e reabriu imediatamente no mesmo ciclo
                        try:
                            from src.trading.risk_manager import register_position_closed, register_sl_hit
                            for sym in closed_positions:
                                # Cooldown geral: bloqueia símbolo por 4h após QUALQUER fechamento
                                register_sl_hit(sym)
                                logger.warning(f"[COOLDOWN] {sym}: bloqueado por 4h após fechamento de posição (anti-reentrada)")
                                # Cooldown direcional: bloquear mesma direção por 6h
                                pos_info = previous_positions_info.get(sym, {})
                                pos_side = pos_info.get("side", "UNKNOWN")
                                if pos_side in ("SHORT", "LONG"):
                                    # SHORT = sinal era SELL, LONG = sinal era BUY
                                    signal_dir = "SELL" if pos_side == "SHORT" else "BUY"
                                    register_position_closed(sym, signal_dir)
                        except Exception as e:
                            logger.warning(f"Erro ao registrar cooldown: {e}")

                        # CORRIGIDO: Cancelar ordens órfãs das posições fechadas
                        if settings.trading_mode == "real":
                            try:
                                from src.exchange.executor import BinanceFuturesExecutor
                                exec_cleanup = BinanceFuturesExecutor()
                                for sym in closed_positions:
                                    await exec_cleanup.cancel_all_orders(sym)
                                    logger.info(f"[LIMPEZA] Ordens de {sym} canceladas (posicao fechada)")
                            except Exception as e:
                                logger.warning(f"Erro ao limpar ordens de posicoes fechadas: {e}")

                        # Online Learning: registrar resultado do trade fechado
                        if settings.ml_online_learning_enabled:
                            try:
                                from src.ml.online_learning import add_trade_result
                                for sym in closed_positions:
                                    _register_trade_result(sym, add_trade_result)
                            except Exception as e:
                                logger.warning(f"Erro ao registrar resultado para online learning: {e}")

                    # Limpeza periódica de ordens órfãs (CADA ciclo - SL pode fechar posição a qualquer momento)
                    _cleanup_cycle += 1
                    if settings.trading_mode == "real":
                        try:
                            from src.trading.orphan_cleaner import cleanup_orphan_orders
                            await cleanup_orphan_orders()
                        except Exception as e:
                            logger.warning(f"Erro na limpeza periodica de ordens orfas: {e}")

                    # MONITORAMENTO DE POSIÇÕES: verificar SL, circuit breaker, reavaliação
                    # Roda em TODOS os modos (real e paper) e TODOS os ciclos
                    if active_positions:
                        try:
                            from src.trading.position_monitor import get_monitor
                            monitor = get_monitor()

                            if settings.trading_mode == "real":
                                from src.exchange.executor import BinanceFuturesExecutor
                                exec_monitor = BinanceFuturesExecutor()

                                # Verificar saúde (SL ativo, circuit breaker)
                                await monitor.check_all_positions(exec_monitor)

                                # Trailing stop: proteger lucros progressivamente (CADA ciclo)
                                await monitor.apply_trailing_stop(exec_monitor)

                                # Reavaliar posições (análise técnica) - respeita min_time_open
                                await monitor.reevaluate_positions(exec_monitor, agent)
                            else:
                                # Paper mode: reavaliar posições paper
                                await monitor.reevaluate_paper_positions(agent)
                        except Exception as e:
                            logger.warning(f"Erro no monitoramento de posicoes: {e}")

                    logger.info(f"[POSICOES] Ativas: {active_positions if active_positions else 'Nenhuma'} | Modo: {settings.trading_mode.upper()}")

                    # ============================================================
                    # TOP MOVERS DINÂMICOS: Adicionar pares com maior movimento
                    # Busca top gainers/losers de Binance Futures a cada ciclo
                    # ============================================================
                    all_symbols = list(settings.top_crypto_pairs)
                    if settings.top_movers_enabled:
                        try:
                            from src.analysis.top_movers import get_dynamic_symbols
                            all_symbols, dynamic_metadata = await get_dynamic_symbols(
                                fixed_symbols=settings.top_crypto_pairs,
                                n_gainers=settings.top_movers_n_gainers,
                                n_losers=settings.top_movers_n_losers,
                                min_volume_usdt=settings.top_movers_min_volume_usdt,
                            )
                        except Exception as e:
                            logger.warning(f"[TOP_MOVERS] Erro ao buscar top movers: {e} (usando apenas pares fixos)")

                    # Filtrar símbolos sem posição ativa E sem cooldown ativo
                    from src.trading.risk_manager import _check_sl_cooldown, _sl_cooldown_registry, _sl_cooldown_hours
                    from src.trading.risk_manager import _direction_cooldown_registry
                    symbols_to_analyze = []
                    for s in all_symbols:
                        if s in active_positions_set:
                            continue
                        if _check_sl_cooldown(s):
                            remaining = _sl_cooldown_hours - (datetime.now(timezone.utc) - _sl_cooldown_registry[s]).total_seconds() / 3600
                            logger.info(f"[COOLDOWN] {s}: pulando análise — cooldown pós-fechamento ({remaining:.1f}h restantes)")
                            continue
                        symbols_to_analyze.append(s)

                    if not symbols_to_analyze:
                        logger.info("[OK] Todos os pares tem posicoes ativas. Aguardando...")
                    else:
                        n_fixed = sum(1 for s in symbols_to_analyze if s in settings.top_crypto_pairs)
                        n_dynamic = len(symbols_to_analyze) - n_fixed
                        logger.info(
                            f"[ANALISE] Analisando {len(symbols_to_analyze)} pares "
                            f"({n_fixed} fixos + {n_dynamic} dinâmicos) sem posicoes ativas..."
                        )

                        for symbol in symbols_to_analyze:
                            # Log extra para pares dinâmicos
                            if symbol in dynamic_metadata:
                                meta = dynamic_metadata[symbol]
                                logger.info(
                                    f"[TOP_MOVER] {symbol}: {meta['mover_type'].upper()} "
                                    f"({meta['price_change_pct']:+.1f}% 24h, "
                                    f"vol=${meta['volume_usdt']/1e6:.0f}M)"
                                )

                            try:
                                _mover_meta = dynamic_metadata.get(symbol, {}) if dynamic_metadata else {}
                                _mover_type = _mover_meta.get("mover_type")
                                _price_change_24h = _mover_meta.get("price_change_pct", 0)
                                await asyncio.wait_for(
                                    agent.analyze(symbol, mover_type=_mover_type, price_change_24h=_price_change_24h),
                                    timeout=300  # 5 minutos max por par
                                )
                            except asyncio.TimeoutError:
                                logger.warning(f"[TIMEOUT] {symbol} excedeu 300s - pulando para proximo par")
                            except Exception as e:
                                logger.error(f"[ERRO] Erro em {symbol}: {e}")

                            await asyncio.sleep(3)  # Pausa entre análises

                    # Atualizar rastreamento de posições
                    previous_positions = active_positions_set

                    # ============================================================
                    # MELHORIA do sinais: Sincronizar com fechamento de candle
                    # Ao invés de dormir 900s fixo, calcula tempo até próxima
                    # hora cheia + 35s (30s pro dado gravar + 5s margem).
                    # Isso garante que a análise usa candles RECÉM-FECHADOS.
                    # Origem: sinais/core/timeframe_scheduler.py
                    # ============================================================
                    from datetime import datetime as _dt
                    from datetime import timezone as _tz
                    now_utc = _dt.now(_tz.utc)
                    # Próxima hora cheia
                    next_hour = now_utc.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
                    # Adicionar 35s de delay (como no sinais: 30s stream + 5s análise)
                    next_analysis = next_hour + timedelta(seconds=35)
                    wait_seconds = (next_analysis - now_utc).total_seconds()

                    # Se falta pouco (< 60s), esperar a hora seguinte
                    if wait_seconds < 60:
                        next_analysis += timedelta(hours=1)
                        wait_seconds = (next_analysis - now_utc).total_seconds()

                    wait_minutes = int(wait_seconds // 60)
                    logger.info(f"[AGUARDANDO] Proxima analise em {wait_minutes}min (sync candle close: {next_analysis.strftime('%H:%M:%S')} UTC)")
                    await asyncio.sleep(wait_seconds)

                except KeyboardInterrupt:
                    raise
                except asyncio.CancelledError:
                    logger.error("[MAIN LOOP] Task cancelada — reiniciando ciclo...")
                    await asyncio.sleep(10)
                except Exception as e:
                    logger.exception(f"Erro no ciclo de monitoramento: {e}")
                    await asyncio.sleep(30)
                except BaseException as e:
                    logger.error(f"[MAIN LOOP] BaseException não tratada: {type(e).__name__}: {e}")
                    await asyncio.sleep(30)

        elif args.mode == 'top5':
            # Todos os pares configurados + top movers dinâmicos
            from src.core.config import settings
            symbols = list(settings.top_crypto_pairs)
            dynamic_metadata = {}

            # Adicionar top movers dinâmicos
            if settings.top_movers_enabled:
                try:
                    from src.analysis.top_movers import get_dynamic_symbols
                    symbols, dynamic_metadata = await get_dynamic_symbols(
                        fixed_symbols=settings.top_crypto_pairs,
                        n_gainers=settings.top_movers_n_gainers,
                        n_losers=settings.top_movers_n_losers,
                        min_volume_usdt=settings.top_movers_min_volume_usdt,
                    )
                except Exception as e:
                    logger.warning(f"[TOP_MOVERS] Erro: {e} (usando apenas pares fixos)")

            logger.info(f"[TOP] Analisando {len(symbols)} pares ({len(symbols) - len(dynamic_metadata)} fixos + {len(dynamic_metadata)} dinâmicos)...")

            # Verificar posições ativas
            active_positions = get_active_positions()
            logger.info(f"[POSICOES] Ativas: {active_positions if active_positions else 'Nenhuma'}")

            # Filtrar apenas símbolos sem posição ativa
            symbols_to_analyze = [s for s in symbols if s not in active_positions]

            if not symbols_to_analyze:
                logger.info("[OK] Todos os pares tem posicoes ativas. Aguardando fechamento...")
            else:
                logger.info(f"[ANALISE] Analisando {len(symbols_to_analyze)} pares sem posicoes ativas...")

                for i, symbol in enumerate(symbols_to_analyze, 1):
                    mover_tag = ""
                    if symbol in dynamic_metadata:
                        meta = dynamic_metadata[symbol]
                        mover_tag = f" [TOP_MOVER {meta['mover_type'].upper()} {meta['price_change_pct']:+.1f}%]"
                    logger.info(f"[{i}/{len(symbols_to_analyze)}] Analisando {symbol}...{mover_tag}")
                    signal = await agent.analyze(symbol)

                    # Mostrar resumo rápido
                    if signal.get('signal') in ['BUY', 'SELL']:
                        logger.info(f"[ALERTA] {signal.get('signal')} com confianca {signal.get('confidence', 0)}/10")
                    else:
                        logger.info(f"[SINAL] {signal.get('signal')} - Confianca: {signal.get('confidence', 0)}/10")

                    await asyncio.sleep(3)  # Pausa entre análises

        elif args.mode == 'top10':
            # Todos os pares configurados + top movers dinâmicos
            from src.core.config import settings
            symbols = list(settings.top_crypto_pairs)
            dynamic_metadata = {}

            if settings.top_movers_enabled:
                try:
                    from src.analysis.top_movers import get_dynamic_symbols
                    symbols, dynamic_metadata = await get_dynamic_symbols(
                        fixed_symbols=settings.top_crypto_pairs,
                        n_gainers=settings.top_movers_n_gainers,
                        n_losers=settings.top_movers_n_losers,
                        min_volume_usdt=settings.top_movers_min_volume_usdt,
                    )
                except Exception as e:
                    logger.warning(f"[TOP_MOVERS] Erro: {e} (usando apenas pares fixos)")

            logger.info(f"[TOP] Analisando {len(symbols)} pares ({len(symbols) - len(dynamic_metadata)} fixos + {len(dynamic_metadata)} dinâmicos)...")
            for i, symbol in enumerate(symbols, 1):
                mover_tag = ""
                if symbol in dynamic_metadata:
                    meta = dynamic_metadata[symbol]
                    mover_tag = f" [TOP_MOVER {meta['mover_type'].upper()} {meta['price_change_pct']:+.1f}%]"
                logger.info(f"[{i}/{len(symbols)}] Analisando {symbol}...{mover_tag}")
                await agent.analyze(symbol)
                await asyncio.sleep(5)  # Pausa entre análises

        elif args.mode == 'cleanup':
            # Limpeza imediata de ordens órfãs e excessivas
            logger.info("[CLEANUP] Limpeza imediata de ordens orfas e excessivas")
            from src.trading.orphan_cleaner import cleanup_orphan_orders
            result = await cleanup_orphan_orders()
            logger.info(f"[CLEANUP] Resultado: {result}")

        elif args.mode == 'train_ml':
            # Treinamento do modelo ML
            logger.info("[ML] Iniciando treinamento do modelo ML")
            from src.ml.train_from_signals import run_training_pipeline
            success = run_training_pipeline()
            if not success:
                logger.error("[ML] Falha no treinamento ML")
                sys.exit(1)

    except KeyboardInterrupt:
        logger.info("[PARADO] Sistema interrompido pelo usuario")
    except Exception as e:
        logger.exception(f"[ERRO FATAL] {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
