"""
Position Reevaluator — Gestão técnica ativa de posições abertas
===============================================================

Reavalia posições abertas usando indicadores técnicos locais (sem LLM).
Roda periodicamente no loop principal do agent.

Ações disponíveis:
- HOLD: manter posição
- MOVE_SL_BREAKEVEN: mover SL para o preço de entrada
- TRAILING_STOP: ajustar SL para travar lucro parcial
- CLOSE: fechar posição (indicadores fortemente contra)
- PARTIAL_CLOSE: realizar 50% do lucro

Regras baseadas no position_reevaluator.py do bot_trade_20260115.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.logger import get_logger

logger = get_logger(__name__)

REEVALUATION_LOG_DIR = Path("reevaluation_logs")
REEVALUATION_LOG_DIR.mkdir(exist_ok=True)


class PositionReevaluator:
    """
    Reavaliação técnica de posições abertas.

    3 camadas:
    1. Indicadores locais (rápido, toda rodada)
    2. Regras baseadas em tempo e progresso
    3. Log de ações para análise
    """

    def __init__(self):
        self._last_check: Dict[str, datetime] = {}
        self._breakeven_moved: Dict[str, bool] = {}
        self._partial_closed: Dict[str, bool] = {}
        self.min_hours_between_checks = 0.083  # ~5 min (antes era 30min — muito lento para alta alavancagem)
        self.min_hours_open = 1.0  # mínimo 1h aberta antes de reavaliar

    async def reevaluate(self, position: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reavalia uma posição aberta e retorna ação recomendada.

        Args:
            position: dict com symbol, signal/side, entry_price, stop_loss,
                      take_profit_1, take_profit_2, timestamp, current_price

        Returns:
            Dict com action, reason, e opcionalmente new_stop_loss
        """
        symbol = position.get("symbol", "")
        pos_key = f"{symbol}_{position.get('signal', position.get('side', 'BUY'))}"

        # Throttle: não checar com muita frequência
        now = datetime.now(timezone.utc)
        last = self._last_check.get(pos_key)
        if last and (now - last).total_seconds() < self.min_hours_between_checks * 3600:
            return {"action": "HOLD", "reason": "Muito cedo para reavaliar"}

        self._last_check[pos_key] = now

        signal_type = position.get("signal", position.get("side", "BUY"))
        if signal_type in ("LONG",):
            signal_type = "BUY"
        elif signal_type in ("SHORT",):
            signal_type = "SELL"

        entry_price = float(position.get("entry_price", 0))
        stop_loss = float(position.get("stop_loss", 0))
        tp1 = float(position.get("take_profit_1", position.get("tp1", 0)))
        tp2 = float(position.get("take_profit_2", position.get("tp2", 0)))
        current_price = float(position.get("current_price", 0))

        if not entry_price or not current_price:
            return {"action": "HOLD", "reason": "Preços indisponíveis"}

        # Calcular PnL e progresso
        if signal_type == "BUY":
            pnl_pct = (current_price - entry_price) / entry_price * 100
            tp1_total_dist = abs(tp1 - entry_price) if tp1 else 0
            current_dist = current_price - entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price * 100
            tp1_total_dist = abs(entry_price - tp1) if tp1 else 0
            current_dist = entry_price - current_price

        tp1_progress = (current_dist / tp1_total_dist) if tp1_total_dist > 0 else 0

        # Tempo aberto
        entry_time_str = position.get("timestamp", position.get("entry_time", ""))
        hours_open = 0
        if entry_time_str:
            try:
                entry_time = datetime.fromisoformat(str(entry_time_str).replace("Z", "+00:00"))
                if entry_time.tzinfo is None:
                    entry_time = entry_time.replace(tzinfo=timezone.utc)
                hours_open = (now - entry_time).total_seconds() / 3600
            except Exception:
                hours_open = 0

        if hours_open < self.min_hours_open:
            return {"action": "HOLD", "reason": f"Aberta há {hours_open:.1f}h (mín {self.min_hours_open}h)"}

        # Indicadores (se disponíveis)
        rsi = position.get("rsi", 50)
        adx = position.get("adx", 25)
        macd_hist = position.get("macd_histogram", 0)

        result = {
            "symbol": symbol,
            "signal_type": signal_type,
            "pnl_pct": round(pnl_pct, 2),
            "tp1_progress": round(tp1_progress, 2),
            "hours_open": round(hours_open, 1),
            "current_price": current_price,
        }

        # ============================================
        # REGRA 1: Breakeven — lucro >= 50% do TP1
        # ============================================
        if (tp1_progress >= 0.5 and pnl_pct > 0
                and not self._breakeven_moved.get(pos_key, False)):
            new_sl = entry_price
            self._breakeven_moved[pos_key] = True
            result.update({
                "action": "MOVE_SL_BREAKEVEN",
                "reason": f"Lucro {tp1_progress:.0%} do TP1 — mover SL para breakeven",
                "new_stop_loss": new_sl,
            })
            self._log_action(result)
            return result

        # ============================================
        # REGRA 2: Trailing Stop — lucro >= 75% do TP1
        # ADX removido: altcoins voláteis têm ADX baixo mas PRECISAM de trailing
        # ============================================
        if tp1_progress >= 0.75 and pnl_pct > 0:
            if signal_type == "BUY":
                new_sl = entry_price + (current_price - entry_price) * 0.5
            else:
                new_sl = entry_price - (entry_price - current_price) * 0.5

            # Só mover se o novo SL é melhor que o atual
            if signal_type == "BUY" and new_sl > stop_loss:
                result.update({
                    "action": "TRAILING_STOP",
                    "reason": (
                        f"Lucro {tp1_progress:.0%} TP1, ADX={adx:.0f} — "
                        f"trailing SL {stop_loss:.2f} → {new_sl:.2f}"
                    ),
                    "new_stop_loss": new_sl,
                })
                self._log_action(result)
                return result
            elif signal_type == "SELL" and new_sl < stop_loss:
                result.update({
                    "action": "TRAILING_STOP",
                    "reason": (
                        f"Lucro {tp1_progress:.0%} TP1, ADX={adx:.0f} — "
                        f"trailing SL {stop_loss:.2f} → {new_sl:.2f}"
                    ),
                    "new_stop_loss": new_sl,
                })
                self._log_action(result)
                return result

        # ============================================
        # REGRA 3: RSI extremo contrário — alerta/parcial
        # ============================================
        if signal_type == "BUY" and rsi > 80 and pnl_pct > 0.5:
            if not self._partial_closed.get(pos_key, False):
                self._partial_closed[pos_key] = True
                result.update({
                    "action": "PARTIAL_CLOSE",
                    "reason": f"RSI sobrecomprado ({rsi:.0f}) com lucro — realizar 50%",
                    "partial_pct": 0.5,
                })
                self._log_action(result)
                return result

        if signal_type == "SELL" and rsi < 20 and pnl_pct > 0.5:
            if not self._partial_closed.get(pos_key, False):
                self._partial_closed[pos_key] = True
                result.update({
                    "action": "PARTIAL_CLOSE",
                    "reason": f"RSI sobrevendido ({rsi:.0f}) com lucro — realizar 50%",
                    "partial_pct": 0.5,
                })
                self._log_action(result)
                return result

        # ============================================
        # REGRA 4: MACD inverteu + prejuízo → fechar
        # ============================================
        # REGRA 4: MACD inverteu — apenas logar, NÃO fechar
        # DESATIVADO: MACD inverte facilmente em crypto e fechava trades prematuramente
        # O stop loss original deve ser a proteção principal
        if signal_type == "BUY" and macd_hist < 0 and pnl_pct < -1.0 and hours_open > 1:
            logger.info(
                f"[REEVAL WATCH] {pos_key}: MACD inverteu ({macd_hist:.4f}) + "
                f"PnL={pnl_pct:.1f}% há {hours_open:.0f}h — mantendo (SL protege)"
            )

        if signal_type == "SELL" and macd_hist > 0 and pnl_pct < -1.0 and hours_open > 1:
            logger.info(
                f"[REEVAL WATCH] {pos_key}: MACD inverteu ({macd_hist:.4f}) + "
                f"PnL={pnl_pct:.1f}% há {hours_open:.0f}h — mantendo (SL protege)"
            )

        # ============================================
        # REGRA 5: Timeout — posição aberta > 48h sem TP
        # ============================================
        if hours_open > 48 and pnl_pct < 0.5:
            result.update({
                "action": "CLOSE",
                "reason": f"Timeout: {hours_open:.0f}h aberta com PnL={pnl_pct:.1f}% — fechar",
            })
            self._log_action(result)
            return result

        # Default: manter
        result.update({
            "action": "HOLD",
            "reason": f"PnL={pnl_pct:.1f}%, TP1 progress={tp1_progress:.0%}, {hours_open:.0f}h aberta",
        })
        return result

    async def reevaluate_all(self, positions: List[Dict]) -> List[Dict]:
        """Reavalia todas as posições e retorna lista de ações."""
        results = []
        for pos in positions:
            result = await self.reevaluate(pos)
            results.append(result)
        return results

    def clear_position(self, symbol: str, signal_type: str = "BUY"):
        """Limpa estado de uma posição fechada."""
        pos_key = f"{symbol}_{signal_type}"
        self._breakeven_moved.pop(pos_key, None)
        self._partial_closed.pop(pos_key, None)
        self._last_check.pop(pos_key, None)

    def _log_action(self, result: Dict):
        """Salva ação no log para análise."""
        action = result.get("action", "HOLD")
        if action == "HOLD":
            return

        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **result,
        }

        log_file = REEVALUATION_LOG_DIR / "actions.json"
        history = []
        if log_file.exists():
            try:
                with open(log_file, "r") as f:
                    history = json.load(f)
            except Exception:
                history = []

        history.append(log_entry)
        # Manter últimos 500
        if len(history) > 500:
            history = history[-500:]

        with open(log_file, "w") as f:
            json.dump(history, f, indent=2, default=str)

        logger.info(
            f"[REEVALUATOR] {result.get('symbol')} {result.get('signal_type')}: "
            f"{action} — {result.get('reason')}"
        )


# Instância global
_reevaluator: Optional[PositionReevaluator] = None


def get_position_reevaluator() -> PositionReevaluator:
    global _reevaluator
    if _reevaluator is None:
        _reevaluator = PositionReevaluator()
    return _reevaluator
