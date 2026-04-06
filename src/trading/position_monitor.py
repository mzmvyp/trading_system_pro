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
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.logger import get_logger

logger = get_logger(__name__)


class PositionMonitor:
    """Monitora e reavalia posições abertas na Binance"""

    # Circuit breaker: fecha posição se prejuízo no PREÇO atingir esse limite
    # Agora DINÂMICO baseado na alavancagem: quanto maior a alavancagem, menor o limite
    MAX_LOSS_PRICE_PERCENT = -5.0  # Default para baixa alavancagem (1-5x)
    # Com alta alavancagem: limite reduzido para evitar liquidação
    # 10x → -3%, 20x → -2%, 40x → -1.5%

    # Trailing stop: níveis progressivos de proteção
    # Quando o preço atinge X% de lucro, mover SL para garantir Y% de lucro
    TRAILING_LEVELS = [
        # (lucro_trigger_pct, sl_lock_pct)
        # Exemplo: quando lucro >= 1.0%, mover SL para garantir 0.3% de lucro
        (0.8, 0.2),    # 0.8% lucro → SL garante 0.2%
        (1.5, 0.7),    # 1.5% lucro → SL garante 0.7%
        (2.5, 1.5),    # 2.5% lucro → SL garante 1.5%
        (4.0, 2.8),    # 4.0% lucro → SL garante 2.8%
        (6.0, 4.5),    # 6.0% lucro → SL garante 4.5%
        (10.0, 8.0),   # 10.0% lucro → SL garante 8.0%
    ]

    def __init__(self):
        self._last_reeval: Dict[str, datetime] = {}
        self._trailing_sl_applied: Dict[str, float] = {}  # {symbol: último SL trailing aplicado}

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
                    # 1. CIRCUIT BREAKER: verificar variação do PREÇO ajustada pela alavancagem
                    # Com alta alavancagem, limites mais apertados para evitar liquidação
                    entry_price = pos.get("entry_price", 0)
                    mark_price = pos.get("mark_price", 0)
                    position_amt = abs(pos.get("position_amt", 0))
                    side = pos.get("side", "LONG")
                    leverage = int(pos.get("leverage", 1)) or 1

                    # Calcular variação do preço
                    if entry_price > 0 and mark_price > 0:
                        if side == "LONG":
                            price_change_pct = ((mark_price - entry_price) / entry_price) * 100
                        else:  # SHORT
                            price_change_pct = ((entry_price - mark_price) / entry_price) * 100
                    else:
                        price_change_pct = 0

                    # Limite dinâmico baseado na alavancagem:
                    # Liquidação acontece em ~(100/leverage)% de movimento
                    # Circuit breaker fecha em 60% do caminho até liquidação
                    if leverage >= 10:
                        max_loss = -(100.0 / leverage) * 0.6  # 60% do caminho até liquidação
                        max_loss = max(max_loss, -5.0)  # Nunca menos que -5% (antes -3% era muito agressivo)
                    else:
                        max_loss = self.MAX_LOSS_PRICE_PERCENT  # -5% para baixa alavancagem

                    if price_change_pct < max_loss:
                        logger.warning(
                            f"[CIRCUIT BREAKER] {symbol}: Preço variou {price_change_pct:.1f}% contra "
                            f"(limite: {max_loss:.1f}%, leverage={leverage}x) - FECHANDO!"
                        )
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

    async def apply_trailing_stop(self, executor) -> Dict[str, Any]:
        """
        Trailing stop progressivo: roda CADA ciclo (sem restrição de tempo).
        Quando o preço atinge certos níveis de lucro, move o SL para proteger ganhos.

        Usa TRAILING_LEVELS: lista de (trigger_pct, lock_pct).
        Exemplo: lucro de 1.5% → SL move para garantir 0.7% de lucro.

        IMPORTANTE: Só move SL para CIMA (LONG) ou para BAIXO (SHORT) — nunca piora.
        """
        results = {"checked": 0, "sl_trailed": 0, "errors": []}

        try:
            positions = await executor.get_all_positions()
            if not positions:
                return results

            for pos in positions:
                symbol = pos.get("symbol", "")
                side = pos.get("side", "LONG")
                entry_price = pos.get("entry_price", 0)
                mark_price = pos.get("mark_price", 0)
                position_amt = abs(pos.get("position_amt", 0))

                if not symbol or entry_price <= 0 or mark_price <= 0 or position_amt <= 0:
                    continue

                results["checked"] += 1

                # Calcular lucro % do preço
                if side == "LONG":
                    profit_pct = ((mark_price - entry_price) / entry_price) * 100
                else:
                    profit_pct = ((entry_price - mark_price) / entry_price) * 100

                if profit_pct <= 0:
                    continue  # Sem lucro, nada a proteger

                # Encontrar o melhor nível de trailing aplicável
                best_lock_pct = None
                for trigger_pct, lock_pct in self.TRAILING_LEVELS:
                    if profit_pct >= trigger_pct:
                        best_lock_pct = lock_pct

                if best_lock_pct is None:
                    continue  # Lucro ainda não atingiu nenhum nível

                # Calcular novo SL
                if side == "LONG":
                    new_sl = entry_price * (1 + best_lock_pct / 100)
                else:
                    new_sl = entry_price * (1 - best_lock_pct / 100)

                # Verificar se é MELHOR que o SL trailing já aplicado
                current_trailing = self._trailing_sl_applied.get(symbol, 0)
                sl_is_better = False
                if side == "LONG":
                    sl_is_better = new_sl > current_trailing if current_trailing else True
                else:
                    sl_is_better = new_sl < current_trailing if current_trailing else True

                if not sl_is_better:
                    continue  # Já tem trailing melhor aplicado

                # Buscar SL atual da exchange para comparar
                algo_orders = await executor.get_open_algo_orders(symbol)
                current_sl_price = None
                for order in algo_orders:
                    if (order.get("type") or "").upper() == "STOP_MARKET":
                        current_sl_price = float(order.get("stopPrice", 0) or order.get("triggerPrice", 0) or 0)
                        break

                # Só mover se o novo SL é melhor que o SL atual na exchange
                if current_sl_price:
                    if side == "LONG" and new_sl <= current_sl_price:
                        continue
                    if side == "SHORT" and new_sl >= current_sl_price:
                        continue

                # Aplicar trailing stop
                logger.info(
                    f"[TRAILING STOP] {symbol} ({side}): lucro {profit_pct:.1f}% atingiu nível → "
                    f"SL movendo de ${current_sl_price or 0:.4f} para ${new_sl:.4f} "
                    f"(garante {best_lock_pct:.1f}% de lucro)"
                )

                success = await self._move_stop_loss(executor, symbol, side, position_amt, new_sl)
                if success:
                    self._trailing_sl_applied[symbol] = new_sl
                    results["sl_trailed"] += 1
                    logger.info(f"[TRAILING STOP] {symbol}: SL trailing aplicado em ${new_sl:.4f}")
                else:
                    results["errors"].append(f"Trailing {symbol}: failed to move SL")

        except Exception as e:
            logger.exception(f"[TRAILING STOP] Erro geral: {e}")
            results["errors"].append(f"General: {e}")

        if results["sl_trailed"] > 0:
            logger.info(f"[TRAILING STOP] {results['sl_trailed']} stops ajustados de {results['checked']} posições")

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
        # POSIÇÃO EM PREJUÍZO SIGNIFICATIVO + REVERSÃO COMPLETA (3/3) = FECHAR
        # Antes: qualquer PnL negativo fechava — agora exige perda > $1.50
        # O SL é a proteção principal, não a reavaliação
        # ============================================================
        if pnl < -1.50 and all_against:
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
        # POSIÇÃO EM LUCRO + 2/3 SINAIS CONTRA = APENAS LOGAR
        # DESATIVADO: apertar SL por reavaliação matava trades antes do TP
        # O trailing stop já protege lucro quando ativado
        # ============================================================
        if pnl > 0 and against_count >= 2 and entry_price > 0 and current_price > 0:
            logger.info(
                f"[REAVALIACAO] {side}: {against_count}/3 contra com lucro ${pnl:.2f} — "
                f"mantendo (trailing stop protege)"
            )
            return None, "", []

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
                else:
                    # SEM TIMESTAMP = não reavaliar (posição pode ter sido aberta agora)
                    # Evita fechar posições recém-abertas por falta de execution record
                    logger.info(f"[REAVALIACAO] {symbol}: sem entry_time encontrado — pulando reavaliação (segurança)")
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
                        # DESATIVADO: Reavaliação NÃO fecha posições na exchange
                        # O SL/TP já estão colocados na Binance — são a proteção principal
                        # Fechar por reavaliação matava trades que depois atingiriam TP
                        logger.warning(
                            f"[REAVALIACAO] {symbol} ({side}): {reason} — "
                            f"NÃO fechando (SL na exchange protege)"
                        )
                        results["kept"] += 1

                    elif action in ("move_sl_breakeven",):
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
        DESATIVADO: Reavaliação de paper positions agora é feita APENAS pelo
        paper_trading._monitor_positions() com trailing stop integrado.
        Este método era DUPLICADO e escrevia diretamente no state.json sem
        coordenação com o paper_trading, causando corrupção de estado e
        fechamentos prematuros.
        """
        return {"skipped": True, "reason": "Unified in paper_trading monitor (avoid state.json race)"}

        # === CÓDIGO ABAIXO DESATIVADO ===
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
                        if entry_time.tzinfo is None:
                            entry_time = entry_time.replace(tzinfo=timezone.utc)
                        hours_open = (now - entry_time).total_seconds() / 3600
                        if hours_open < settings.reevaluation_min_time_open_hours:
                            continue
                    except (ValueError, TypeError):
                        pass
                else:
                    # Sem timestamp = não reavaliar (segurança)
                    logger.info(f"[REAVALIACAO PAPER] {symbol}: sem entry_time — pulando reavaliação")
                    continue

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

                    # PositionReevaluator: gestão ativa (breakeven, trailing, parcial)
                    try:
                        from src.trading.position_reevaluator import get_position_reevaluator
                        reeval = get_position_reevaluator()
                        reeval_result = await reeval.reevaluate({
                            "symbol": symbol,
                            "signal": side,
                            "entry_price": entry_price,
                            "stop_loss": pos.get("stop_loss", 0),
                            "take_profit_1": pos.get("take_profit_1", pos.get("tp1", 0)),
                            "take_profit_2": pos.get("take_profit_2", pos.get("tp2", 0)),
                            "current_price": current_price,
                            "timestamp": entry_time_str,
                            "rsi": rsi,
                            "adx": indicators.get("adx", 25),
                            "macd_histogram": macd_hist,
                        })
                        reeval_action = reeval_result.get("action", "HOLD")
                    except Exception as e:
                        logger.warning(f"[REEVALUATOR] Erro: {e}")
                        reeval_action = "HOLD"
                        reeval_result = {}

                    # Combinar: reevaluator + _evaluate_reversal
                    action, reason, details = self._evaluate_reversal(
                        normalized_side, trend, macd_hist, rsi, pnl, entry_price, current_price
                    )

                    # PositionReevaluator tem prioridade para ações de gestão
                    if reeval_action in ("MOVE_SL_BREAKEVEN", "TRAILING_STOP"):
                        new_sl = reeval_result.get("new_stop_loss")
                        if new_sl:
                            old_sl = pos.get("stop_loss", 0)
                            sl_is_better = False
                            if normalized_side == "LONG":
                                sl_is_better = new_sl > old_sl if old_sl else True
                            else:
                                sl_is_better = new_sl < old_sl if old_sl else True
                            if sl_is_better:
                                pos["stop_loss"] = new_sl
                                state_modified = True
                                results["sl_moved"] += 1
                                logger.info(f"[REEVALUATOR] {symbol}: {reeval_result.get('reason')} — SL ${old_sl:.4f} -> ${new_sl:.4f}")
                            else:
                                results["kept"] += 1
                        else:
                            results["kept"] += 1

                    elif reeval_action == "CLOSE" or action == "close":
                        close_reason = reeval_result.get("reason", reason)
                        # DESATIVADO: Reavaliação NÃO fecha posições — SL/TP são a proteção
                        # Antes: fechava posições prematuramente, impedindo que atingissem TP
                        logger.warning(
                            f"[REAVALIACAO PAPER] {symbol} ({normalized_side}): {close_reason} — "
                            f"NÃO fechando (SL protege, reavaliação apenas monitora)"
                        )
                        results["kept"] += 1

                    elif action in ("move_sl_breakeven", "tighten_sl"):
                        new_sl = self._calculate_new_sl(action, normalized_side, entry_price, current_price)
                        if new_sl:
                            old_sl = pos.get("stop_loss", 0)
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
        """Busca timestamp de entrada nos execution records ou state.json"""
        # 1. Tentar real_orders
        try:
            records_dir = "real_orders"
            if os.path.exists(records_dir):
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

        # 2. Fallback: buscar no state.json (portfolio)
        try:
            state_file = Path("portfolio/state.json")
            if state_file.exists():
                with open(state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
                for pos_key, pos in state.get("positions", {}).items():
                    if pos.get("symbol") == symbol and pos.get("status") == "OPEN":
                        ts = pos.get("entry_time") or pos.get("timestamp")
                        if ts:
                            return datetime.fromisoformat(ts)
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
