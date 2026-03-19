"""
Position Monitor - Monitoramento e Reavaliação de Posições Abertas
===================================================================

Verifica saúde das posições em tempo real:
1. Garante que toda posição tem SL ativo na Binance
2. Circuit breaker: fecha forçado se ROI < threshold
3. Reavalia posição — SÓ fecha em reversão COMPLETA confirmada

FILOSOFIA DE REAVALIAÇÃO:
- O Stop Loss é a proteção principal. A reavaliação NÃO substitui o SL.
- Só fecha posição por reavaliação se TODOS os indicadores confirmarem reversão E posição em prejuízo.
- Posição em LUCRO + reversão completa: move SL para breakeven+0.5% (protege lucro).
- Posição em LUCRO + sinais parciais (2/3): aperta SL para 50% entre entry e preço atual.
- Indicadores que CONFIRMAM a posição (ex: RSI oversold para SHORT) não contam contra.
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.core.logger import get_logger

logger = get_logger(__name__)


class PositionMonitor:
    """Monitora e reavalia posições abertas na Binance"""

    # Circuit breaker: fecha posição se ROI margem atingir esse limite
    MAX_LOSS_ROI_PERCENT = -15.0  # -15% do ROI na margem = fechar

    def __init__(self):
        self._last_reeval: Dict[str, datetime] = {}

    async def check_all_positions(self, executor) -> Dict[str, Any]:
        """
        Verifica saúde de todas as posições abertas.

        Returns:
            Dict com resultado da verificação
        """
        results = {
            "checked": 0,
            "sl_missing": 0,
            "sl_replaced": 0,
            "circuit_breaker_closed": 0,
            "errors": []
        }

        try:
            # Obter todas as posições abertas
            positions = await executor.get_all_positions()
            if not positions or (isinstance(positions, list) and len(positions) > 0 and isinstance(positions[0], dict) and "error" in positions[0]):
                return results

            # Obter ordens abertas regulares
            all_orders = await executor.get_open_orders()
            if isinstance(all_orders, dict) and "error" in all_orders:
                all_orders = []

            # Indexar ordens por símbolo
            orders_by_symbol: Dict[str, List] = {}
            for order in all_orders:
                sym = order.get("symbol", "")
                if sym not in orders_by_symbol:
                    orders_by_symbol[sym] = []
                orders_by_symbol[sym].append(order)

            # Incluir ordens algo (SL/TP) por símbolo — Binance usa Algo API para STOP_MARKET/TAKE_PROFIT_MARKET
            for pos in positions:
                sym = pos.get("symbol", "")
                if not sym:
                    continue
                algo_orders = await executor.get_open_algo_orders(sym)
                if sym not in orders_by_symbol:
                    orders_by_symbol[sym] = []
                orders_by_symbol[sym].extend(algo_orders)

            for pos in positions:
                symbol = pos.get("symbol", "")
                if not symbol:
                    continue

                results["checked"] += 1

                try:
                    # 1. CIRCUIT BREAKER: verificar ROI da margem
                    entry_price = pos.get("entry_price", 0)
                    pnl = pos.get("unrealized_pnl", 0)
                    leverage = pos.get("leverage", 20)
                    position_amt = abs(pos.get("position_amt", 0))
                    side = pos.get("side", "LONG")

                    # Calcular ROI na margem (= PnL / margem_usada)
                    if entry_price > 0 and position_amt > 0:
                        notional = position_amt * entry_price
                        margin_used = notional / leverage if leverage > 0 else notional
                        roi_percent = (pnl / margin_used * 100) if margin_used > 0 else 0
                    else:
                        roi_percent = 0

                    if roi_percent < self.MAX_LOSS_ROI_PERCENT:
                        logger.warning(
                            f"[CIRCUIT BREAKER] {symbol}: ROI={roi_percent:.1f}% < {self.MAX_LOSS_ROI_PERCENT}% - FECHANDO!"
                        )
                        logger.info(f"[CIRCUIT BREAKER] {symbol}: ROI {roi_percent:.1f}% - FECHANDO POSICAO!")
                        try:
                            close_result = await executor.close_position(symbol)
                            if isinstance(close_result, dict) and "error" not in close_result:
                                results["circuit_breaker_closed"] += 1
                                logger.warning(f"[CIRCUIT BREAKER] {symbol} fechado com sucesso")
                            else:
                                logger.error(f"[CIRCUIT BREAKER] Erro ao fechar {symbol}: {close_result}")
                                results["errors"].append(f"Circuit breaker {symbol}: {close_result}")
                        except Exception as e:
                            logger.error(f"[CIRCUIT BREAKER] Exceção ao fechar {symbol}: {e}")
                            results["errors"].append(f"Circuit breaker {symbol}: {e}")
                        continue  # Não precisa verificar SL se já fechou

                    # 2. VERIFICAR SL ATIVO: toda posição precisa de stop loss
                    symbol_orders = orders_by_symbol.get(symbol, [])
                    has_sl = False
                    for order in symbol_orders:
                        order_type = order.get("type", "").upper()
                        # Stop Loss pode ser STOP_MARKET, STOP, ou ordens algo
                        if order_type in ("STOP_MARKET", "STOP", "STOP_LOSS"):
                            has_sl = True
                            break

                    if not has_sl:
                        results["sl_missing"] += 1
                        logger.warning(f"[SL AUSENTE] {symbol} ({side}) sem Stop Loss ativo!")
                        logger.info(f"[SL AUSENTE] {symbol} ({side}) - Recolocando Stop Loss...")

                        # Cancelar qualquer ordem algo STOP_MARKET existente (evita duplicação)
                        algo_list = await executor.get_open_algo_orders(symbol)
                        for order in algo_list:
                            if (order.get("type") or "").upper() == "STOP_MARKET" and order.get("algoId") is not None:
                                try:
                                    await executor.cancel_algo_order(order["algoId"])
                                    logger.info(f"[SL LIMPEZA] {symbol}: cancelada ordem SL duplicada algoId={order['algoId']}")
                                except Exception as e:
                                    logger.warning(f"[SL LIMPEZA] {symbol}: erro ao cancelar algoId {order.get('algoId')}: {e}")

                        # Tentar recolocar SL baseado no execution record
                        sl_price = await self._find_sl_price(symbol, entry_price, side)
                        if sl_price:
                            try:
                                close_side = "SELL" if side == "LONG" else "BUY"
                                sl_result = await executor.place_stop_loss(
                                    symbol, close_side, position_amt, sl_price
                                )
                                if isinstance(sl_result, dict) and "error" not in sl_result:
                                    results["sl_replaced"] += 1
                                    logger.info(f"[SL RECOLOCADO] {symbol}: SL em ${sl_price:.4f}")
                                else:
                                    logger.error(f"[SL ERRO] {symbol}: {sl_result}")
                                    results["errors"].append(f"SL replace {symbol}: {sl_result}")
                            except Exception as e:
                                logger.error(f"[SL ERRO] {symbol}: {e}")
                                results["errors"].append(f"SL replace {symbol}: {e}")
                        else:
                            # Sem SL conhecido: usar 3% de distância como emergência
                            if side == "LONG":
                                emergency_sl = entry_price * 0.97
                            else:
                                emergency_sl = entry_price * 1.03
                            logger.warning(f"[SL EMERGENCIA] {symbol}: SL original não encontrado, usando 3% = ${emergency_sl:.4f}")
                            logger.info(f"[SL EMERGENCIA] {symbol}: SL de emergência em ${emergency_sl:.4f}")
                            try:
                                close_side = "SELL" if side == "LONG" else "BUY"
                                sl_result = await executor.place_stop_loss(
                                    symbol, close_side, position_amt, emergency_sl
                                )
                                if isinstance(sl_result, dict) and "error" not in sl_result:
                                    results["sl_replaced"] += 1
                                else:
                                    results["errors"].append(f"Emergency SL {symbol}: {sl_result}")
                            except Exception as e:
                                results["errors"].append(f"Emergency SL {symbol}: {e}")

                except Exception as e:
                    logger.error(f"[MONITOR] Erro ao verificar {symbol}: {e}")
                    results["errors"].append(f"Check {symbol}: {e}")

        except Exception as e:
            logger.exception(f"[MONITOR] Erro geral: {e}")
            results["errors"].append(f"General: {e}")

        # Log resumo
        if results["checked"] > 0:
            logger.info(f"[MONITOR] Verificadas {results['checked']} posicoes: "
                  f"{results['sl_missing']} sem SL, "
                  f"{results['sl_replaced']} SL recolocados, "
                  f"{results['circuit_breaker_closed']} fechadas por circuit breaker")

        return results

    def _evaluate_reversal(self, side: str, trend: str, macd_hist: float, rsi: float,
                           pnl: float, entry_price: float = 0, current_price: float = 0) -> tuple:
        """
        Avalia reversão contra a posição e decide ação.

        Ações possíveis:
        - "close": fechar posição (só se em prejuízo + 3/3 sinais contra)
        - "move_sl_breakeven": mover SL para breakeven+0.5% (em lucro + 3/3 contra)
        - "tighten_sl": apertar SL para 50% entre entry e preço atual (em lucro + 2/3 contra)
        - None: manter tudo como está

        Returns:
            (action: str|None, reason: str, details: list)
        """
        against_details = []

        # Contar sinais contra
        trend_against = False
        macd_against = False
        rsi_against = False

        if side == "LONG":
            if "bearish" in trend:
                trend_against = True
                against_details.append(f"trend={trend}")
            if macd_hist < 0:
                macd_against = True
                against_details.append(f"MACD={macd_hist:.4f}")
            if rsi < 40:
                rsi_against = True
                against_details.append(f"RSI={rsi:.0f}")
        else:  # SHORT
            if "bullish" in trend:
                trend_against = True
                against_details.append(f"trend={trend}")
            if macd_hist > 0:
                macd_against = True
                against_details.append(f"MACD={macd_hist:.4f}")
            if rsi > 60:
                rsi_against = True
                against_details.append(f"RSI={rsi:.0f}")

        against_count = sum([trend_against, macd_against, rsi_against])
        all_against = against_count == 3

        # ============================================================
        # POSIÇÃO EM PREJUÍZO + REVERSÃO COMPLETA (3/3) = FECHAR
        # ============================================================
        if pnl < 0 and all_against:
            reason = f"REVERSAO COMPLETA {side} (PnL: ${pnl:.2f}): {', '.join(against_details)}"
            return "close", reason, against_details

        # ============================================================
        # POSIÇÃO EM LUCRO + REVERSÃO COMPLETA (3/3) = MOVER SL PARA BREAKEVEN+0.5%
        # Protege o lucro sem fechar prematuramente. Se o preço voltar
        # a favor, a posição continua. Se não, sai no breakeven com lucro.
        # ============================================================
        if pnl >= 0 and all_against and entry_price > 0:
            reason = (f"PROTECAO LUCRO {side} (PnL: ${pnl:.2f}): "
                      f"{', '.join(against_details)} - movendo SL para breakeven+0.5%")
            return "move_sl_breakeven", reason, against_details

        # ============================================================
        # POSIÇÃO EM LUCRO + 2/3 SINAIS CONTRA = APERTAR SL
        # Aperta SL para 50% entre entry e preço atual (garante parte do lucro)
        # ============================================================
        if pnl > 0 and against_count >= 2 and entry_price > 0 and current_price > 0:
            reason = (f"APERTANDO SL {side} (PnL: ${pnl:.2f}): "
                      f"{against_count}/3 sinais contra ({', '.join(against_details)})")
            return "tighten_sl", reason, against_details

        # ============================================================
        # SINAIS PARCIAIS (prejuízo ou lucro insuficiente) = MANTER
        # ============================================================
        if against_details:
            logger.info(
                f"[REAVALIACAO] {side}: {against_count}/3 sinais contra "
                f"({', '.join(against_details)}) - MANTENDO"
            )

        return None, "", against_details

    async def _move_stop_loss(self, executor, symbol: str, side: str,
                              position_amt: float, new_sl: float) -> bool:
        """
        Move o SL de uma posição: cancela SL antigos e coloca novo.
        Usa o mesmo padrão do stop_adjuster.py.

        Returns:
            True se moveu com sucesso
        """
        try:
            # 1. Cancelar SL existentes (apenas STOP_MARKET, preserva TP)
            algo_orders = await executor.get_open_algo_orders(symbol)
            for order in algo_orders:
                if (order.get("type") or "").upper() == "STOP_MARKET" and order.get("algoId"):
                    try:
                        await executor.cancel_algo_order(order["algoId"])
                        logger.info(f"[REAVALIACAO] {symbol}: cancelado SL antigo algoId={order['algoId']}")
                    except Exception as e:
                        logger.warning(f"[REAVALIACAO] {symbol}: erro ao cancelar SL: {e}")

            # 2. Colocar novo SL
            close_side = "SELL" if side == "LONG" else "BUY"
            result = await executor.place_stop_loss(symbol, close_side, position_amt, new_sl)
            if isinstance(result, dict) and "error" not in result:
                logger.info(f"[REAVALIACAO] {symbol}: SL movido para ${new_sl:.4f}")
                return True
            else:
                logger.error(f"[REAVALIACAO] {symbol}: erro ao colocar novo SL: {result}")
                return False
        except Exception as e:
            logger.error(f"[REAVALIACAO] {symbol}: exceção ao mover SL: {e}")
            return False

    def _calculate_new_sl(self, action: str, side: str, entry_price: float,
                          current_price: float) -> Optional[float]:
        """
        Calcula o novo preço de SL baseado na ação.

        - move_sl_breakeven: entry + 0.5% na direção favorável
        - tighten_sl: 50% entre entry e preço atual
        """
        if action == "move_sl_breakeven":
            if side == "LONG":
                return entry_price * 1.005  # Breakeven + 0.5%
            else:
                return entry_price * 0.995  # Breakeven + 0.5% para short
        elif action == "tighten_sl":
            if side == "LONG":
                # SL = meio entre entry e preço atual (garante ~50% do lucro)
                return entry_price + (current_price - entry_price) * 0.5
            else:
                # SL = meio entre entry e preço atual (short)
                return entry_price - (entry_price - current_price) * 0.5
        return None

    async def reevaluate_positions(self, executor, agent) -> Dict[str, Any]:
        """
        Reavalia posições abertas usando análise técnica.

        Ações:
        - Em prejuízo + 3/3 sinais contra: FECHA posição
        - Em lucro + 3/3 sinais contra: MOVE SL para breakeven+0.5%
        - Em lucro + 2/3 sinais contra: APERTA SL (50% do lucro)
        - Caso contrário: mantém tudo
        """
        from src.core.config import settings

        if not settings.reevaluation_enabled:
            return {"skipped": True, "reason": "Reevaluation disabled"}

        results = {
            "reevaluated": 0,
            "closed_by_reversal": 0,
            "sl_moved": 0,
            "kept": 0,
            "errors": []
        }

        try:
            positions = await executor.get_all_positions()
            if not positions:
                return results

            for pos in positions:
                symbol = pos.get("symbol", "")
                side = pos.get("side", "LONG")
                pnl = pos.get("unrealized_pnl", 0)
                entry_price = pos.get("entry_price", 0)
                position_amt = abs(pos.get("position_amt", 0))
                mark_price = pos.get("mark_price", 0)
                if not symbol:
                    continue

                # Verificar intervalo mínimo de reavaliação
                now = datetime.now(timezone.utc)
                last_reeval = self._last_reeval.get(symbol)
                if last_reeval:
                    hours_since = (now - last_reeval).total_seconds() / 3600
                    if hours_since < settings.reevaluation_interval_hours:
                        logger.debug(f"[REAVALIACAO] {symbol}: pulando, reavaliado ha {hours_since:.1f}h (min: {settings.reevaluation_interval_hours}h)")
                        continue

                # Verificar tempo mínimo aberto
                entry_time = await self._find_entry_time(symbol)
                if entry_time:
                    if entry_time.tzinfo is None:
                        entry_time = entry_time.replace(tzinfo=timezone.utc)
                    hours_open = (now - entry_time).total_seconds() / 3600
                    if hours_open < settings.reevaluation_min_time_open_hours:
                        logger.debug(f"[REAVALIACAO] {symbol}: pulando, aberta ha {hours_open:.1f}h (min: {settings.reevaluation_min_time_open_hours}h)")
                        continue

                results["reevaluated"] += 1
                self._last_reeval[symbol] = now

                try:
                    from src.analysis.agno_tools import analyze_technical_indicators
                    logger.info(f"[REAVALIACAO] Analisando {symbol} ({side}, PnL: ${pnl:.2f})...")
                    tech = await analyze_technical_indicators(symbol)

                    if not tech or "indicators" not in tech:
                        continue

                    indicators = tech["indicators"]
                    rsi = indicators.get("rsi", 50)
                    trend = indicators.get("trend", "neutral")
                    macd_hist = indicators.get("macd_histogram", 0)
                    current_price = indicators.get("close", mark_price) or mark_price

                    action, reason, details = self._evaluate_reversal(
                        side, trend, macd_hist, rsi, pnl, entry_price, current_price
                    )

                    if action == "close":
                        logger.warning(f"[REAVALIACAO] {symbol} ({side}): {reason} - FECHANDO!")
                        try:
                            close_result = await executor.close_position(symbol)
                            if isinstance(close_result, dict) and "error" not in close_result:
                                results["closed_by_reversal"] += 1
                            else:
                                results["errors"].append(f"Close {symbol}: {close_result}")
                        except Exception as e:
                            results["errors"].append(f"Close {symbol}: {e}")

                    elif action in ("move_sl_breakeven", "tighten_sl"):
                        new_sl = self._calculate_new_sl(action, side, entry_price, current_price)
                        if new_sl:
                            logger.info(f"[REAVALIACAO] {symbol} ({side}): {reason} - SL -> ${new_sl:.4f}")
                            success = await self._move_stop_loss(
                                executor, symbol, side, position_amt, new_sl
                            )
                            if success:
                                results["sl_moved"] += 1
                            else:
                                results["errors"].append(f"Move SL {symbol}: failed")
                        else:
                            results["kept"] += 1

                    else:
                        results["kept"] += 1
                        logger.debug(f"[REAVALIACAO] {symbol}: RSI={rsi:.0f}, trend={trend}, PnL=${pnl:.2f} - Mantendo")

                except Exception as e:
                    logger.error(f"[REAVALIACAO] Erro ao reavaliar {symbol}: {e}")
                    results["errors"].append(f"Reeval {symbol}: {e}")

        except Exception as e:
            logger.exception(f"[REAVALIACAO] Erro geral: {e}")

        if results["reevaluated"] > 0:
            logger.info(f"[REAVALIACAO] {results['reevaluated']} posicoes reavaliadas: "
                  f"{results['closed_by_reversal']} fechadas, "
                  f"{results['sl_moved']} SL movidos, "
                  f"{results['kept']} mantidas")

        return results

    async def reevaluate_paper_positions(self, agent) -> Dict[str, Any]:
        """
        Reavalia posições abertas em paper trading.
        Mesma lógica do modo real, com ajuste de SL no state.json.
        """
        from src.core.config import settings

        if not settings.reevaluation_enabled:
            return {"skipped": True, "reason": "Reevaluation disabled"}

        results = {
            "reevaluated": 0,
            "closed_by_reversal": 0,
            "sl_moved": 0,
            "kept": 0,
            "errors": []
        }

        try:
            if not os.path.exists("portfolio/state.json"):
                return results

            with open("portfolio/state.json", "r", encoding="utf-8") as f:
                state = json.load(f)

            positions = state.get("positions", {})
            if not positions:
                return results

            state_modified = False

            for pos_key, pos in positions.items():
                if pos.get("status") != "OPEN":
                    continue

                symbol = pos.get("symbol", "")
                side = pos.get("side", "BUY")
                if not symbol:
                    continue

                normalized_side = "LONG" if side in ("BUY", "LONG") else "SHORT"
                entry_price = pos.get("entry_price", 0)
                position_size = pos.get("position_size", 0)
                current_price = 0
                pnl = 0

                # Verificar intervalo mínimo de reavaliação
                now = datetime.now(timezone.utc)
                last_reeval = self._last_reeval.get(symbol)
                if last_reeval:
                    hours_since = (now - last_reeval).total_seconds() / 3600
                    if hours_since < settings.reevaluation_interval_hours:
                        continue

                # Verificar tempo mínimo aberto
                entry_time_str = pos.get("entry_time") or pos.get("timestamp")
                if entry_time_str:
                    try:
                        entry_time = datetime.fromisoformat(entry_time_str)
                        hours_open = (now - entry_time).total_seconds() / 3600
                        if hours_open < settings.reevaluation_min_time_open_hours:
                            continue
                    except (ValueError, TypeError):
                        pass

                results["reevaluated"] += 1
                self._last_reeval[symbol] = now

                try:
                    from src.analysis.agno_tools import analyze_technical_indicators
                    logger.info(f"[REAVALIACAO PAPER] Analisando {symbol} ({normalized_side})...")
                    tech = await analyze_technical_indicators(symbol)

                    if not tech or "indicators" not in tech:
                        continue

                    indicators = tech["indicators"]
                    rsi = indicators.get("rsi", 50)
                    trend = indicators.get("trend", "neutral")
                    macd_hist = indicators.get("macd_histogram", 0)
                    current_price = indicators.get("close", entry_price)

                    # Calcular PnL
                    if entry_price > 0 and position_size > 0 and current_price > 0:
                        if normalized_side == "LONG":
                            pnl = (current_price - entry_price) * position_size
                        else:
                            pnl = (entry_price - current_price) * position_size

                    logger.info(f"[REAVALIACAO PAPER] {symbol} PnL: ${pnl:.2f}")

                    action, reason, details = self._evaluate_reversal(
                        normalized_side, trend, macd_hist, rsi, pnl, entry_price, current_price
                    )

                    if action == "close":
                        logger.warning(f"[REAVALIACAO PAPER] {symbol} ({normalized_side}): {reason} - FECHANDO!")
                        try:
                            if hasattr(agent, 'paper_system') and agent.paper_system:
                                cp = await agent.paper_system.get_current_price(symbol)
                                if cp:
                                    await agent.paper_system.close_position_manual(pos_key, cp)
                                    results["closed_by_reversal"] += 1
                                else:
                                    results["errors"].append(f"Preço não disponível para {symbol}")
                            else:
                                results["errors"].append(f"Paper system não disponível para fechar {symbol}")
                        except Exception as e:
                            results["errors"].append(f"Close paper {symbol}: {e}")

                    elif action in ("move_sl_breakeven", "tighten_sl"):
                        new_sl = self._calculate_new_sl(action, normalized_side, entry_price, current_price)
                        if new_sl:
                            old_sl = pos.get("stop_loss", 0)
                            # Só move se o novo SL é MELHOR que o atual
                            sl_is_better = False
                            if normalized_side == "LONG":
                                sl_is_better = new_sl > old_sl if old_sl else True
                            else:
                                sl_is_better = new_sl < old_sl if old_sl else True

                            if sl_is_better:
                                pos["stop_loss"] = new_sl
                                state_modified = True
                                results["sl_moved"] += 1
                                logger.info(f"[REAVALIACAO PAPER] {symbol}: {reason} - SL ${old_sl:.4f} -> ${new_sl:.4f}")
                            else:
                                results["kept"] += 1
                                logger.debug(f"[REAVALIACAO PAPER] {symbol}: novo SL ${new_sl:.4f} não é melhor que atual ${old_sl:.4f}")
                        else:
                            results["kept"] += 1
                    else:
                        results["kept"] += 1
                        logger.debug(f"[REAVALIACAO PAPER] {symbol}: RSI={rsi:.0f}, trend={trend}, PnL=${pnl:.2f} - Mantendo")

                except Exception as e:
                    logger.error(f"[REAVALIACAO PAPER] Erro ao reavaliar {symbol}: {e}")
                    results["errors"].append(f"Reeval paper {symbol}: {e}")

            # Salvar state.json se houve mudanças no SL
            if state_modified:
                with open("portfolio/state.json", "w", encoding="utf-8") as f:
                    json.dump(state, f, indent=2, default=str)

        except Exception as e:
            logger.exception(f"[REAVALIACAO PAPER] Erro geral: {e}")

        if results["reevaluated"] > 0:
            logger.info(f"[REAVALIACAO PAPER] {results['reevaluated']} posicoes reavaliadas: "
                  f"{results['closed_by_reversal']} fechadas, "
                  f"{results['sl_moved']} SL movidos, "
                  f"{results['kept']} mantidas")

        return results

    async def _find_sl_price(self, symbol: str, entry_price: float, side: str) -> Optional[float]:
        """Busca preço de SL original nos execution records"""
        try:
            records_dir = "real_orders"
            if not os.path.exists(records_dir):
                return None

            # Buscar o execution record mais recente para o símbolo
            import glob
            pattern = os.path.join(records_dir, f"execution_{symbol}_*.json")
            files = sorted(glob.glob(pattern), reverse=True)

            for f in files:
                try:
                    with open(f, "r") as fh:
                        record = json.load(fh)
                        if record.get("status") == "OPEN" and record.get("stop_loss"):
                            return float(record["stop_loss"])
                except (json.JSONDecodeError, KeyError):
                    continue

        except Exception as e:
            logger.warning(f"Erro ao buscar SL de {symbol}: {e}")

        return None

    async def _find_entry_time(self, symbol: str) -> Optional[datetime]:
        """Busca timestamp de entrada nos execution records"""
        try:
            records_dir = "real_orders"
            if not os.path.exists(records_dir):
                return None

            import glob
            pattern = os.path.join(records_dir, f"execution_{symbol}_*.json")
            files = sorted(glob.glob(pattern), reverse=True)

            for f in files:
                try:
                    with open(f, "r") as fh:
                        record = json.load(fh)
                        if record.get("status") == "OPEN" and record.get("timestamp"):
                            return datetime.fromisoformat(record["timestamp"])
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue

        except Exception:
            pass

        return None


# Instância singleton
_monitor_instance = None


def get_monitor() -> PositionMonitor:
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = PositionMonitor()
    return _monitor_instance
