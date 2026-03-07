"""
Risk management and position validation
"""
from typing import Dict, Any, Optional
from datetime import datetime
import json
import os

from src.core.logger import get_logger

logger = get_logger(__name__)


def _calculate_current_drawdown() -> float:
    """Calcula o drawdown atual baseado no histórico de trades."""
    try:
        from src.trading.paper_trading import real_paper_trading
        summary = real_paper_trading.get_portfolio_summary()
        total_pnl_percent = summary.get('total_pnl_percent', 0)
        if total_pnl_percent < 0:
            return abs(total_pnl_percent) / 100.0
        return 0.0
    except Exception as e:
        logger.warning(f"Erro ao calcular drawdown: {e}")
        return 0.0


def _calculate_total_exposure() -> float:
    """Calcula a exposição total atual baseado nas posições abertas."""
    try:
        if not os.path.exists("portfolio/state.json"):
            return 0.0
        with open("portfolio/state.json", "r", encoding='utf-8') as f:
            state = json.load(f)
        positions = state.get("positions", {})
        total_value = sum(
            p.get("position_value", p.get("position_size", 0) * p.get("entry_price", 0))
            for p in positions.values()
            if p.get("status") == "OPEN"
        )
        capital = state.get("capital", state.get("initial_capital", 10000.0))
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
        today = datetime.now().date()
        trades = real_paper_trading.get_trade_history()
        daily_trades = [t for t in trades if datetime.fromisoformat(t['timestamp']).date() == today]
        return len(daily_trades)
    except Exception as e:
        logger.warning(f"Erro em _get_daily_trades_count: {e}")
        return 0


def validate_risk_and_position(
    signal: Dict[str, Any],
    symbol: str,
    account_balance: float = None
) -> Dict[str, Any]:
    """
    Valida risco e calcula tamanho de posição apropriado com circuit breakers.
    """
    try:
        if account_balance is None:
            account_balance = 10000.0

        if signal.get('signal') == 'HOLD' or signal.get('signal') == 'NO_SIGNAL':
            return {
                "can_execute": False,
                "reason": "Sinal HOLD/NO_SIGNAL - não executar",
                "risk_level": "low"
            }

        # Check existing position
        try:
            signal_source = signal.get("source", "UNKNOWN")
            if os.path.exists("portfolio/state.json"):
                with open("portfolio/state.json", "r", encoding='utf-8') as f:
                    state = json.load(f)
                    positions = state.get("positions", {})
                    signal_type = signal.get("signal", "")
                    if signal_type == "BUY":
                        position_key = f"{symbol}_{signal_source}"
                    elif signal_type == "SELL":
                        position_key = f"{symbol}_{signal_source}_SHORT"
                    else:
                        position_key = None

                    if position_key and position_key in positions:
                        existing_position = positions[position_key]
                        if existing_position.get("status") == "OPEN":
                            return {
                                "can_execute": False,
                                "reason": f"Ja existe uma posicao {signal_type} {signal_source} aberta para {symbol}.",
                                "risk_level": "medium"
                            }
        except Exception as e:
            logger.warning(f"Erro ao verificar posicoes existentes: {e}")

        entry_price = signal.get('entry_price', 0)
        stop_loss = signal.get('stop_loss', 0)
        confidence = signal.get('confidence', 0)

        from src.core.config import settings

        # Normalize confidence to 0-10 scale
        if confidence > 0 and confidence <= 5:
            confidence = confidence * 2
            logger.warning(f"[CONFIANCA] Convertendo escala 0-5 para 0-10: {signal.get('confidence')} -> {confidence}")

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

        max_risk_per_trade = 5.0
        if risk_percentage > max_risk_per_trade:
            return {
                "can_execute": False,
                "reason": f"Risco muito alto: {risk_percentage:.2f}% (máximo {max_risk_per_trade}%)",
                "risk_level": "high"
            }
        elif risk_percentage > 3.0:
            logger.warning(f"[RISCO] Risco elevado ({risk_percentage:.2f}%), reduzindo tamanho de posição")

        current_drawdown = _calculate_current_drawdown()
        max_drawdown_allowed = 0.40
        if current_drawdown > max_drawdown_allowed:
            return {
                "can_execute": False,
                "reason": f"Drawdown atual muito alto: {current_drawdown:.2%} (máximo {max_drawdown_allowed:.0%})",
                "risk_level": "high"
            }
        elif current_drawdown > 0.15:
            logger.warning(f"[RISCO] Drawdown elevado ({current_drawdown:.2%}), reduzindo tamanho de posição")

        total_exposure = _calculate_total_exposure()
        max_exposure_allowed = 0.80
        if total_exposure > max_exposure_allowed:
            return {
                "can_execute": False,
                "reason": f"Exposição total muito alta: {total_exposure:.0%} (máximo {max_exposure_allowed:.0%})",
                "risk_level": "high"
            }
        elif total_exposure > 0.50:
            logger.warning(f"[RISCO] Exposição elevada ({total_exposure:.0%}), considere reduzir posições")

        logger.info(f"[P&L MODE] Exposição atual: {total_exposure:.0%}")

        daily_trades = _get_daily_trades_count()
        if daily_trades >= 5:
            return {
                "can_execute": False,
                "reason": f"Limite diário de trades atingido: {daily_trades} (máximo 5)",
                "risk_level": "medium"
            }

        if stop_loss:
            risk_per_unit = abs(entry_price - stop_loss)
            if risk_per_unit > 0:
                position_size = 100.0 / risk_per_unit
            else:
                position_size = 1.0
        else:
            position_size = 1.0

        position_value = position_size * entry_price
        max_risk_amount = abs(entry_price - stop_loss) * position_size if stop_loss else 0

        logger.info(f"[P&L MODE] Tamanho posição: {position_size:.6f} unidades, Valor: ${position_value:.2f}, Risco: ${max_risk_amount:.2f}")

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
