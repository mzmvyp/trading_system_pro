"""
Risk management and position validation
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from src.core.logger import get_logger

logger = get_logger(__name__)

# Cooldown pós-stop: símbolo bloqueado por N horas após SL ser atingido
_sl_cooldown_hours = 4.0
_sl_cooldown_registry: Dict[str, datetime] = {}

# Cooldown direcional pós-close: bloqueia MESMA DIREÇÃO por N horas após fechar
_direction_cooldown_hours = 10.0
_direction_cooldown_registry: Dict[str, dict] = {}

# Arquivo de persistência para cooldowns
_COOLDOWN_FILE = Path("data/cooldowns.json")


def _save_cooldowns():
    """Persiste cooldowns em disco (sobrevive restarts)"""
    try:
        _COOLDOWN_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "sl_cooldowns": {
                sym: t.isoformat() for sym, t in _sl_cooldown_registry.items()
            },
            "direction_cooldowns": {
                sym: {"direction": d["direction"], "time": d["time"].isoformat()}
                for sym, d in _direction_cooldown_registry.items()
            }
        }
        with open(_COOLDOWN_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.warning(f"[COOLDOWN] Erro ao salvar cooldowns: {e}")


def _load_cooldowns():
    """Carrega cooldowns do disco no startup"""
    global _sl_cooldown_registry, _direction_cooldown_registry
    try:
        if not _COOLDOWN_FILE.exists():
            return
        with open(_COOLDOWN_FILE) as f:
            data = json.load(f)

        now = datetime.now(timezone.utc)

        # Carregar SL cooldowns (descartar expirados)
        for sym, t_str in data.get("sl_cooldowns", {}).items():
            t = datetime.fromisoformat(t_str)
            if (now - t).total_seconds() / 3600 < _sl_cooldown_hours:
                _sl_cooldown_registry[sym] = t

        # Carregar direction cooldowns (descartar expirados)
        for sym, d in data.get("direction_cooldowns", {}).items():
            t = datetime.fromisoformat(d["time"])
            if (now - t).total_seconds() / 3600 < _direction_cooldown_hours:
                _direction_cooldown_registry[sym] = {
                    "direction": d["direction"],
                    "time": t
                }

        loaded = len(_sl_cooldown_registry) + len(_direction_cooldown_registry)
        if loaded > 0:
            logger.info(f"[COOLDOWN] {loaded} cooldowns carregados do disco")
    except Exception as e:
        logger.warning(f"[COOLDOWN] Erro ao carregar cooldowns: {e}")


# Carregar cooldowns ao importar o módulo
_load_cooldowns()


def register_sl_hit(symbol: str):
    """Registra que um símbolo teve SL atingido (para cooldown)"""
    _sl_cooldown_registry[symbol] = datetime.now(timezone.utc)
    logger.warning(f"[COOLDOWN] {symbol}: bloqueado por {_sl_cooldown_hours}h após stop loss")
    _save_cooldowns()


def register_position_closed(symbol: str, direction: str):
    """Registra fechamento de posição com direção (para cooldown direcional)"""
    _direction_cooldown_registry[symbol] = {
        "direction": direction.upper(),
        "time": datetime.now(timezone.utc)
    }
    logger.warning(
        f"[COOLDOWN DIRECIONAL] {symbol}: {direction} bloqueado por "
        f"{_direction_cooldown_hours}h (evita reabrir mesma direção)"
    )
    _save_cooldowns()


def _check_sl_cooldown(symbol: str) -> bool:
    """Retorna True se o símbolo ainda está em cooldown pós-stop"""
    last_sl = _sl_cooldown_registry.get(symbol)
    if not last_sl:
        return False
    hours_since = (datetime.now(timezone.utc) - last_sl).total_seconds() / 3600
    if hours_since < _sl_cooldown_hours:
        return True
    # Cooldown expirado, limpar
    del _sl_cooldown_registry[symbol]
    return False


def _calculate_current_drawdown() -> float:
    """Calcula drawdown real desde o pico máximo do equity.
    Funciona tanto em modo paper (trade_history) quanto modo real (real_orders/)."""
    try:
        from src.core.config import settings as _cfg
        initial_capital = _cfg.initial_capital
        pnl_list = []

        # Tentar paper trading primeiro
        try:
            from src.trading.paper_trading import real_paper_trading
            trades = real_paper_trading.get_trade_history()
            if trades:
                pnl_list = [t.get('pnl', t.get('realized_pnl', 0)) or 0 for t in trades]
        except Exception:
            pass

        # Se paper vazio, ler de real_orders/ (modo produção real)
        if not pnl_list:
            import glob
            order_files = sorted(glob.glob("real_orders/orders_*.json"))
            for fpath in order_files:
                try:
                    with open(fpath, 'r') as f:
                        orders = json.load(f)
                    for o in orders:
                        pnl = o.get("realized_pnl", o.get("pnl", 0))
                        if pnl and isinstance(pnl, (int, float)) and pnl != 0:
                            pnl_list.append(pnl)
                except Exception:
                    continue

        if not pnl_list:
            return 0.0

        equity = initial_capital
        peak = initial_capital
        for pnl in pnl_list:
            equity += pnl
            if equity > peak:
                peak = equity

        current_dd = (peak - equity) / peak if peak > 0 else 0
        if current_dd > 0.01:
            logger.info(f"[DRAWDOWN] Equity={equity:.2f} | Pico={peak:.2f} | DD={current_dd:.1%}")
        return current_dd
    except Exception as e:
        logger.warning(f"Erro ao calcular drawdown: {e}")
        return 0.0


def _calculate_total_exposure() -> float:
    """Calcula a exposição total atual usando preço atual (mark_price) se disponível."""
    try:
        if not os.path.exists("portfolio/state.json"):
            return 0.0
        with open("portfolio/state.json", "r", encoding='utf-8') as f:
            state = json.load(f)
        positions = state.get("positions", {})
        total_value = 0.0
        for p in positions.values():
            if p.get("status") != "OPEN":
                continue
            size = p.get("position_size", 0)
            current_price = p.get("mark_price", p.get("current_price", p.get("entry_price", 0)))
            total_value += size * current_price if size and current_price else p.get("position_value", 0)
        from src.core.config import settings as _cfg
        capital = state.get("capital", state.get("initial_capital", _cfg.initial_capital))
        if capital <= 0:
            return 0.0
        return total_value / capital
    except Exception as e:
        logger.warning(f"Erro ao calcular exposição: {e}")
        return 0.0


def _get_daily_trades_count() -> int:
    """Retorna o número de trades executados hoje."""
    try:
        from src.trading.paper_trading import real_paper_trading
        today = datetime.now(timezone.utc).date()
        trades = real_paper_trading.get_trade_history()
        daily_trades = [t for t in trades if datetime.fromisoformat(t['timestamp']).date() == today]
        return len(daily_trades)
    except Exception as e:
        logger.warning(f"Erro em _get_daily_trades_count: {e}")
        return 0


def validate_risk_and_position(
    signal: Dict[str, Any],
    symbol: str,
    account_balance: float = None,
    _trend_data: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Valida risco e calcula tamanho de posição apropriado com circuit breakers.
    Inclui filtro de tendência dinâmico e validação de R:R.
    """
    try:
        if account_balance is None:
            # Tentar buscar saldo real da Binance
            try:
                from src.core.config import settings as _settings
                account_balance = _settings.initial_capital
            except Exception:
                account_balance = 300.0  # Fallback conservador

        if signal.get('signal') == 'HOLD' or signal.get('signal') == 'NO_SIGNAL':
            return {
                "can_execute": False,
                "reason": "Sinal HOLD/NO_SIGNAL - não executar",
                "risk_level": "low"
            }

        # Check blacklist de tokens
        from src.core.config import settings as _cfg
        if symbol in _cfg.token_blacklist:
            return {
                "can_execute": False,
                "reason": f"{symbol} está na blacklist (ilíquido ou consistentemente perdedor)",
                "risk_level": "high"
            }

        # Check SELL precisa de confiança maior
        sig_type = signal.get("signal", "").upper()
        sig_conf = signal.get("confidence", 0)
        if sig_type == "SELL" and sig_conf < _cfg.sell_min_confidence:
            return {
                "can_execute": False,
                "reason": f"SELL requer confiança mínima {_cfg.sell_min_confidence}/10 (recebido {sig_conf}/10)",
                "risk_level": "medium"
            }

        # Check cooldown pós-stop (evita whipsaw)
        if _check_sl_cooldown(symbol):
            remaining = _sl_cooldown_hours - (datetime.now(timezone.utc) - _sl_cooldown_registry[symbol]).total_seconds() / 3600
            return {
                "can_execute": False,
                "reason": f"Cooldown pos-stop ativo para {symbol}: {remaining:.1f}h restantes (evita whipsaw)",
                "risk_level": "medium"
            }

        # Check cooldown direcional pós-close (evita reabrir mesma direção)
        dir_cd = _direction_cooldown_registry.get(symbol)
        if dir_cd:
            hours_since = (datetime.now(timezone.utc) - dir_cd["time"]).total_seconds() / 3600
            if hours_since < _direction_cooldown_hours and sig_type == dir_cd["direction"]:
                remaining = _direction_cooldown_hours - hours_since
                return {
                    "can_execute": False,
                    "reason": f"Cooldown direcional: {sig_type} {symbol} bloqueado por {remaining:.1f}h "
                              f"(última posição {dir_cd['direction']} fechada há {hours_since:.1f}h)",
                    "risk_level": "medium"
                }
            elif hours_since >= _direction_cooldown_hours:
                del _direction_cooldown_registry[symbol]

        # Check existing position + GUARDA DIRECIONAL (anti stop em massa)
        try:
            if os.path.exists("portfolio/state.json"):
                with open("portfolio/state.json", "r", encoding='utf-8') as f:
                    state = json.load(f)
                    positions = state.get("positions", {})
                    open_positions = [p for p in positions.values() if p.get("status") == "OPEN"]

                    # Check 1: já existe posição neste símbolo?
                    for pos in open_positions:
                        if pos.get("symbol") == symbol:
                            existing_signal = pos.get("signal", "UNKNOWN")
                            existing_source = pos.get("source", "UNKNOWN")
                            return {
                                "can_execute": False,
                                "reason": f"Ja existe posicao {existing_signal} ({existing_source}) aberta para {symbol}. Feche antes de abrir nova.",
                                "risk_level": "medium"
                            }

                    # Check 1b: LIMITE TOTAL — máximo 3 posições simultâneas
                    from src.core.config import settings as _cfg_risk
                    max_positions = getattr(_cfg_risk, 'max_open_positions', 3)
                    if len(open_positions) >= max_positions:
                        open_symbols = [p.get("symbol", "?") for p in open_positions]
                        logger.warning(
                            f"[MAX POSICOES] {symbol} BLOQUEADO: já existem {len(open_positions)} "
                            f"posições abertas ({', '.join(open_symbols)}). "
                            f"Máximo {max_positions} posições simultâneas."
                        )
                        return {
                            "can_execute": False,
                            "reason": f"Limite de posições: {len(open_positions)}/{max_positions} "
                                      f"({', '.join(open_symbols)}). Feche alguma antes.",
                            "risk_level": "high"
                        }

                    # Guarda direcional removida — respeitamos apenas max_open_positions
        except Exception as e:
            logger.warning(f"Erro ao verificar posicoes existentes: {e}")

        # Filtro de tendência dinâmico (EMA 50/200 no 4h)
        signal_type = signal.get("signal", "").upper()
        if _trend_data and signal_type in ("BUY", "SELL"):
            allow_long = _trend_data.get("allow_long", True)
            allow_short = _trend_data.get("allow_short", True)
            trend_desc = _trend_data.get("description", "")
            trend = _trend_data.get("trend", "UNKNOWN")

            if signal_type == "BUY" and not allow_long:
                return {
                    "can_execute": False,
                    "reason": f"Sinal BUY bloqueado pelo filtro de tendencia: {trend_desc}",
                    "risk_level": "medium",
                    "trend": trend
                }
            if signal_type == "SELL" and not allow_short:
                return {
                    "can_execute": False,
                    "reason": f"Sinal SELL bloqueado pelo filtro de tendencia: {trend_desc}",
                    "risk_level": "medium",
                    "trend": trend
                }

        # Filtro de volatilidade: não operar se stop muito apertado (mercado choppy)
        entry_price = signal.get('entry_price', 0)
        stop_loss_check = signal.get('stop_loss', 0)
        if entry_price and stop_loss_check:
            sl_distance_pct = abs(entry_price - stop_loss_check) / entry_price * 100
            if sl_distance_pct < 0.5:
                return {
                    "can_execute": False,
                    "reason": f"Stop loss muito apertado: {sl_distance_pct:.2f}% (minimo 0.5%). Mercado choppy, evitar.",
                    "risk_level": "high"
                }

        # Validação de R:R mínimo
        entry_rr = signal.get('entry_price', 0)
        sl_rr = signal.get('stop_loss', 0)
        tp1_rr = signal.get('tp1') or signal.get('take_profit_1') or signal.get('target_1', 0)
        if entry_rr and sl_rr and tp1_rr:
            risk_d = abs(entry_rr - sl_rr)
            reward_d = abs(tp1_rr - entry_rr)
            if risk_d > 0:
                rr = reward_d / risk_d
                if rr < 1.48:  # Tolerância para floating point (1.4999 exibe "1.50")
                    return {
                        "can_execute": False,
                        "reason": f"Risk:Reward inadequado: {rr:.2f}:1 (minimo 1.5:1)",
                        "risk_level": "high",
                        "rr_ratio": rr
                    }

        entry_price = signal.get('entry_price', 0)
        stop_loss = signal.get('stop_loss', 0)
        confidence = signal.get('confidence', 0)

        from src.core.config import settings

        # CORRIGIDO: Não converter confiança automaticamente.
        # O prompt do DeepSeek já pede escala 1-10 explicitamente.
        # A conversão anterior dobrava confiança 4 para 8, causando falsos positivos.

        if confidence < 1 or confidence > 10:
            return {
                "can_execute": False,
                "reason": f"Confianca invalida: {confidence} (deve ser entre 1 e 10)",
                "risk_level": "medium",
                "confidence": confidence
            }

        min_confidence = settings.min_confidence_0_10
        if confidence < min_confidence:
            return {
                "can_execute": False,
                "reason": f"Confianca muito baixa: {confidence}/10 (minimo {min_confidence}/10 para executar)",
                "risk_level": "medium",
                "confidence": confidence,
                "min_confidence": min_confidence
            }

        if not entry_price or not stop_loss:
            return {
                "can_execute": False,
                "reason": "Precos de entrada ou stop loss nao definidos",
                "risk_level": "high"
            }

        risk_per_trade = abs(entry_price - stop_loss)
        risk_percentage = (risk_per_trade / entry_price) * 100

        # Stop largo NÃO bloqueia - ajusta tamanho da posição
        # Só bloqueia se stop absurdo (> 20%) = provavelmente bug
        if risk_percentage > 20.0:
            return {
                "can_execute": False,
                "reason": f"Stop loss provavelmente inválido: {risk_percentage:.2f}% de distância (> 20%)",
                "risk_level": "high"
            }

        if risk_percentage > 5.0:
            logger.info(f"[RISCO] Stop largo ({risk_percentage:.2f}%) - posição será reduzida proporcionalmente")

        from src.core.config import settings as _risk_cfg
        current_drawdown = _calculate_current_drawdown()
        max_drawdown_allowed = _risk_cfg.max_drawdown  # config.py: 0.15
        if current_drawdown > max_drawdown_allowed:
            return {
                "can_execute": False,
                "reason": f"Drawdown atual muito alto: {current_drawdown:.2%} (máximo {max_drawdown_allowed:.0%})",
                "risk_level": "high"
            }
        elif current_drawdown > max_drawdown_allowed * 0.6:
            logger.warning(f"[RISCO] Drawdown elevado ({current_drawdown:.2%}), reduzindo tamanho de posição")

        total_exposure = _calculate_total_exposure()
        max_exposure_allowed = _risk_cfg.max_exposure  # config.py: 0.50
        if total_exposure > max_exposure_allowed:
            return {
                "can_execute": False,
                "reason": f"Exposição total muito alta: {total_exposure:.0%} (máximo {max_exposure_allowed:.0%})",
                "risk_level": "high"
            }
        elif total_exposure > max_exposure_allowed * 0.8:
            logger.warning(f"[RISCO] Exposição elevada ({total_exposure:.0%}), considere reduzir posições")

        logger.info(f"[P&L MODE] Exposição atual: {total_exposure:.0%}")

        daily_trades = _get_daily_trades_count()
        max_daily = _risk_cfg.max_daily_trades  # config.py: 8
        if daily_trades >= max_daily:
            return {
                "can_execute": False,
                "reason": f"Limite diário atingido: {daily_trades}/{max_daily} trades hoje",
                "risk_level": "medium"
            }

        # Calcular position size baseado no capital REAL e risco configurado
        # Formula: risk_amount = capital * risk_percent
        #          position_size = risk_amount / stop_distance
        # Se stop bater, perde exatamente risk_percent do capital
        from src.core.config import settings as _cfg
        risk_percent = _cfg.risk_percent_per_trade / 100.0

        if stop_loss:
            risk_per_unit = abs(entry_price - stop_loss)
            if risk_per_unit > 0:
                risk_amount = account_balance * risk_percent
                position_size = risk_amount / risk_per_unit
            else:
                position_size = 0.0
        else:
            position_size = 0.0

        position_value = position_size * entry_price
        max_risk_amount = abs(entry_price - stop_loss) * position_size if stop_loss else 0
        implied_leverage = position_value / account_balance if account_balance > 0 else 0

        logger.info(f"[POSITION SIZING] Capital: ${account_balance:.2f} | Risco: {_cfg.risk_percent_per_trade}% (${max_risk_amount:.2f})")
        logger.info(f"[POSITION SIZING] Tamanho: {position_size:.6f} unidades | Valor: ${position_value:.2f} | Alavancagem: {implied_leverage:.1f}x")

        daily_trades = _get_daily_trades_count()
        return {
            "can_execute": True,
            "recommended_position_size": position_size,
            "position_value": position_size * entry_price,
            "max_risk_amount": max_risk_amount,
            "risk_level": "acceptable",
            "current_drawdown": current_drawdown,
            "daily_trades": daily_trades
        }
    except Exception as e:
        return {
            "can_execute": False,
            "reason": f"Erro na validação: {str(e)}",
            "risk_level": "high"
        }
