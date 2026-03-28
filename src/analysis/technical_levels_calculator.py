"""
Technical SL/TP Calculator - Calcula Stop Loss e Take Profit baseados em niveis tecnicos reais.

Usa: suportes/resistencias, Fibonacci, Bollinger Bands, EMAs, Volume POC, market structure.
Nunca usa percentuais fixos arbitrarios.
"""
from typing import Any, Dict, List, Optional, Tuple

from src.core.logger import get_logger

logger = get_logger(__name__)

# Margem minima de seguranca: SL nao pode ser mais perto que 0.3% do entry
MIN_SL_DISTANCE_PCT = 0.3
# SL maximo tecnico: se nenhum nivel tecnico estiver a menos de 15%, provavelmente nao ha setup
# O tamanho da posição é ajustado automaticamente para compensar stop largo
MAX_SL_DISTANCE_PCT = 15.0
# Minimo risk:reward aceitavel (1.5:1 — alvos realistas para mercado lateral)
MIN_RISK_REWARD = 1.5


def calculate_technical_sl_tp(
    entry_price: float,
    direction: str,  # "BUY" or "SELL"
    analysis_data: Dict[str, Any],
    operation_type: str = "DAY_TRADE",
) -> Dict[str, Any]:
    """
    Calcula SL, TP1 e TP2 baseados em niveis tecnicos reais.

    Hierarquia de niveis (ordem de prioridade):
    1. Suporte/Resistencia de structure points (swing highs/lows)
    2. Fibonacci retracement levels
    3. EMAs importantes (20, 50, 200)
    4. Bollinger Bands
    5. Volume POC
    6. Fallback: ATR-based (ultimo recurso)

    Args:
        entry_price: Preco de entrada
        direction: BUY ou SELL
        analysis_data: Dados completos da analise (de prepare_analysis_for_llm)
        operation_type: SCALP, DAY_TRADE, SWING_TRADE, POSITION_TRADE

    Returns:
        Dict com stop_loss, take_profit_1, take_profit_2, metodo usado, e detalhes
    """
    if entry_price <= 0:
        return {"error": "entry_price invalido"}

    # Coletar todos os niveis tecnicos disponiveis
    levels = _collect_all_levels(entry_price, analysis_data)

    if direction == "BUY":
        result = _calculate_buy_levels(entry_price, levels, operation_type)
    else:
        result = _calculate_sell_levels(entry_price, levels, operation_type)

    # Validar risk:reward minimo
    result = _validate_risk_reward(entry_price, direction, result)

    logger.info(
        f"[TECH SL/TP] {direction} @ ${entry_price:,.2f} | "
        f"SL=${result['stop_loss']:,.2f} ({result['sl_method']}) | "
        f"TP1=${result['take_profit_1']:,.2f} ({result['tp1_method']}) | "
        f"TP2=${result['take_profit_2']:,.2f} ({result['tp2_method']}) | "
        f"R:R={result.get('risk_reward', 0):.1f}"
    )

    return result


def _collect_all_levels(
    current_price: float, analysis_data: Dict[str, Any]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Coleta todos os niveis de preco tecnicos e os organiza em:
    - supports: niveis abaixo do preco atual (ordenados do mais proximo ao mais distante)
    - resistances: niveis acima do preco atual (ordenados do mais proximo ao mais distante)

    Cada nivel tem: price, source, strength (0-1)
    """
    supports = []
    resistances = []

    key_levels = analysis_data.get("key_levels", {})
    volatility = analysis_data.get("volatility", {})
    indicators = analysis_data.get("key_indicators", {})

    # 1. Suporte/Resistencia imediatos (alta prioridade)
    imm_support = key_levels.get("immediate_support")
    imm_resistance = key_levels.get("immediate_resistance")
    if imm_support and imm_support > 0:
        supports.append({"price": imm_support, "source": "S/R_imediato", "strength": 0.9})
    if imm_resistance and imm_resistance > 0:
        resistances.append({"price": imm_resistance, "source": "S/R_imediato", "strength": 0.9})

    # 2. Fibonacci levels (media-alta prioridade)
    fib_382 = key_levels.get("fib_382")
    fib_50 = key_levels.get("fib_50")
    fib_618 = key_levels.get("fib_618")

    for fib_price, fib_name in [(fib_382, "Fib_38.2%"), (fib_50, "Fib_50%"), (fib_618, "Fib_61.8%")]:
        if fib_price and fib_price > 0:
            if fib_price < current_price:
                supports.append({"price": fib_price, "source": fib_name, "strength": 0.8})
            elif fib_price > current_price:
                resistances.append({"price": fib_price, "source": fib_name, "strength": 0.8})

    # 3. EMAs como suporte/resistencia dinamico
    ema_struct = indicators.get("ema_structure", {})
    # Tentar obter valores de EMA do analysis_data
    # O analysis_data de prepare_analysis_for_llm nao tem os valores brutos de EMA,
    # mas temos as posicoes relativas. Vamos calcular a partir de volatility/atr
    # Buscar dados brutos se disponiveis (via tech_data passado separadamente)
    raw_indicators = analysis_data.get("_raw_indicators", {})

    ema_20 = raw_indicators.get("ema_20")
    ema_50 = raw_indicators.get("ema_50")
    ema_200 = raw_indicators.get("ema_200")
    sma_200 = raw_indicators.get("sma_200")
    bb_upper = raw_indicators.get("bb_upper")
    bb_lower = raw_indicators.get("bb_lower")
    bb_middle = raw_indicators.get("bb_middle")

    for ema_val, ema_name, strength in [
        (ema_20, "EMA20", 0.6),
        (ema_50, "EMA50", 0.75),
        (ema_200, "EMA200", 0.9),
        (sma_200, "SMA200", 0.85),
    ]:
        if ema_val and ema_val > 0:
            if ema_val < current_price:
                supports.append({"price": ema_val, "source": ema_name, "strength": strength})
            elif ema_val > current_price:
                resistances.append({"price": ema_val, "source": ema_name, "strength": strength})

    # 4. Bollinger Bands
    if bb_lower and bb_lower > 0:
        supports.append({"price": bb_lower, "source": "BB_lower", "strength": 0.65})
    if bb_upper and bb_upper > 0:
        resistances.append({"price": bb_upper, "source": "BB_upper", "strength": 0.65})
    if bb_middle and bb_middle > 0:
        if bb_middle < current_price:
            supports.append({"price": bb_middle, "source": "BB_middle", "strength": 0.5})
        elif bb_middle > current_price:
            resistances.append({"price": bb_middle, "source": "BB_middle", "strength": 0.5})

    # 5. Volume POC
    poc = key_levels.get("volume_poc")
    if poc and poc > 0:
        if poc < current_price:
            supports.append({"price": poc, "source": "Volume_POC", "strength": 0.7})
        elif poc > current_price:
            resistances.append({"price": poc, "source": "Volume_POC", "strength": 0.7})

    # 6. Market structure levels (swing highs/lows)
    market_struct = analysis_data.get("_market_structure", {})
    struct_support = market_struct.get("support_level")
    struct_resistance = market_struct.get("resistance_level")
    if struct_support and struct_support > 0 and struct_support < current_price:
        supports.append({"price": struct_support, "source": "Swing_Low", "strength": 0.85})
    if struct_resistance and struct_resistance > 0 and struct_resistance > current_price:
        resistances.append({"price": struct_resistance, "source": "Swing_High", "strength": 0.85})

    # 7. High/Low 24h como niveis psicologicos
    price_ctx = analysis_data.get("price_context", {})
    high_24h = price_ctx.get("high_24h")
    low_24h = price_ctx.get("low_24h")
    if low_24h and low_24h > 0 and low_24h < current_price:
        supports.append({"price": low_24h, "source": "Low_24h", "strength": 0.6})
    if high_24h and high_24h > 0 and high_24h > current_price:
        resistances.append({"price": high_24h, "source": "High_24h", "strength": 0.6})

    # Remover duplicatas proximas (0.15%) e ordenar por distancia ao preco atual
    supports = _deduplicate_levels(supports, current_price)
    resistances = _deduplicate_levels(resistances, current_price)

    # Ordenar: mais proximo primeiro
    supports.sort(key=lambda x: current_price - x["price"])
    resistances.sort(key=lambda x: x["price"] - current_price)

    # Adicionar ATR como ultimo recurso
    atr = volatility.get("atr_value", volatility.get("atr", 0))
    if not atr or atr <= 0:
        atr = current_price * 0.015  # fallback 1.5%

    return {
        "supports": supports,
        "resistances": resistances,
        "atr": atr,
        "current_price": current_price,
    }


def _deduplicate_levels(
    levels: List[Dict], reference_price: float, threshold_pct: float = 0.15
) -> List[Dict]:
    """Remove niveis muito proximos entre si, mantendo o de maior strength."""
    if not levels:
        return []

    # Ordenar por preco
    levels.sort(key=lambda x: x["price"])
    deduped = [levels[0]]

    for level in levels[1:]:
        last = deduped[-1]
        distance_pct = abs(level["price"] - last["price"]) / reference_price * 100
        if distance_pct < threshold_pct:
            # Manter o de maior strength
            if level["strength"] > last["strength"]:
                deduped[-1] = level
        else:
            deduped.append(level)

    return deduped


def _calculate_buy_levels(
    entry: float,
    levels: Dict,
    operation_type: str,
) -> Dict[str, Any]:
    """
    Para BUY:
    - SL abaixo do suporte mais proximo (com margem de seguranca)
    - TP1 na resistencia mais proxima
    - TP2 na segunda resistencia
    """
    supports = levels["supports"]
    resistances = levels["resistances"]
    atr = levels["atr"]

    # === STOP LOSS ===
    sl, sl_method = _find_buy_sl(entry, supports, atr, operation_type)

    # === TAKE PROFIT 1 ===
    tp1, tp1_method = _find_buy_tp1(entry, resistances, atr, sl, operation_type)

    # === TAKE PROFIT 2 ===
    tp2, tp2_method = _find_buy_tp2(entry, resistances, atr, tp1, operation_type)

    sl_distance = abs(entry - sl) / entry * 100
    tp1_distance = abs(tp1 - entry) / entry * 100
    rr = tp1_distance / sl_distance if sl_distance > 0 else 0

    return {
        "stop_loss": round(sl, 8),
        "take_profit_1": round(tp1, 8),
        "take_profit_2": round(tp2, 8),
        "sl_method": sl_method,
        "tp1_method": tp1_method,
        "tp2_method": tp2_method,
        "sl_distance_pct": round(sl_distance, 2),
        "tp1_distance_pct": round(tp1_distance, 2),
        "risk_reward": round(rr, 2),
        "supports_found": len(supports),
        "resistances_found": len(resistances),
        "levels_used": {
            "supports": [{"price": s["price"], "source": s["source"]} for s in supports[:5]],
            "resistances": [{"price": r["price"], "source": r["source"]} for r in resistances[:5]],
        },
    }


def _calculate_sell_levels(
    entry: float,
    levels: Dict,
    operation_type: str,
) -> Dict[str, Any]:
    """
    Para SELL:
    - SL acima da resistencia mais proxima (com margem de seguranca)
    - TP1 no suporte mais proximo
    - TP2 no segundo suporte
    """
    supports = levels["supports"]
    resistances = levels["resistances"]
    atr = levels["atr"]

    # === STOP LOSS (acima da resistencia) ===
    sl, sl_method = _find_sell_sl(entry, resistances, atr, operation_type)

    # === TAKE PROFIT 1 (no suporte) ===
    tp1, tp1_method = _find_sell_tp1(entry, supports, atr, sl, operation_type)

    # === TAKE PROFIT 2 (no segundo suporte) ===
    tp2, tp2_method = _find_sell_tp2(entry, supports, atr, tp1, operation_type)

    sl_distance = abs(sl - entry) / entry * 100
    tp1_distance = abs(entry - tp1) / entry * 100
    rr = tp1_distance / sl_distance if sl_distance > 0 else 0

    return {
        "stop_loss": round(sl, 8),
        "take_profit_1": round(tp1, 8),
        "take_profit_2": round(tp2, 8),
        "sl_method": sl_method,
        "tp1_method": tp1_method,
        "tp2_method": tp2_method,
        "sl_distance_pct": round(sl_distance, 2),
        "tp1_distance_pct": round(tp1_distance, 2),
        "risk_reward": round(rr, 2),
        "supports_found": len(supports),
        "resistances_found": len(resistances),
        "levels_used": {
            "supports": [{"price": s["price"], "source": s["source"]} for s in supports[:5]],
            "resistances": [{"price": r["price"], "source": r["source"]} for r in resistances[:5]],
        },
    }


# ===== BUY SL/TP helpers =====

def _find_buy_sl(
    entry: float, supports: List[Dict], atr: float, op_type: str
) -> Tuple[float, str]:
    """SL para BUY: abaixo do suporte mais proximo, com margem ATR."""
    # Margem de seguranca abaixo do suporte (para nao ser stopado no pavio)
    margin = atr * _sl_margin_multiplier(op_type)

    for s in supports:
        sl_candidate = s["price"] - margin
        distance_pct = (entry - sl_candidate) / entry * 100

        # Verificar limites
        if distance_pct < MIN_SL_DISTANCE_PCT:
            continue  # Muito perto, pular para proximo suporte
        if distance_pct > MAX_SL_DISTANCE_PCT:
            continue  # Muito longe, pular

        return sl_candidate, f"abaixo_{s['source']}"

    # Fallback: usar ATR se nenhum suporte valido encontrado
    sl_atr = entry - (atr * _sl_atr_multiplier(op_type))
    return sl_atr, "ATR_fallback"


def _find_buy_tp1(
    entry: float, resistances: List[Dict], atr: float, sl: float, op_type: str
) -> Tuple[float, str]:
    """TP1 para BUY: na resistencia mais proxima."""
    sl_distance = entry - sl
    min_tp1 = entry + (sl_distance * MIN_RISK_REWARD)  # Minimo 1.5:1 R:R

    for r in resistances:
        if r["price"] > min_tp1:
            return r["price"], f"em_{r['source']}"

    # Se nenhuma resistencia atende o R:R minimo, usar 2x ATR
    tp1_atr = entry + (atr * _tp1_atr_multiplier(op_type))
    if tp1_atr > min_tp1:
        return tp1_atr, "ATR_fallback"
    return min_tp1, "min_RR_1.5"


def _find_buy_tp2(
    entry: float, resistances: List[Dict], atr: float, tp1: float, op_type: str
) -> Tuple[float, str]:
    """TP2 para BUY: na segunda resistencia acima de TP1."""
    for r in resistances:
        # TP2 deve ser acima de TP1 com pelo menos 0.3% de margem
        if r["price"] > tp1 * 1.003:
            return r["price"], f"em_{r['source']}"

    # Fallback: TP2 = 2x a distancia de TP1
    tp1_distance = tp1 - entry
    tp2 = entry + (tp1_distance * 1.5)
    return tp2, "1.5x_TP1_dist"


# ===== SELL SL/TP helpers =====

def _find_sell_sl(
    entry: float, resistances: List[Dict], atr: float, op_type: str
) -> Tuple[float, str]:
    """SL para SELL: acima da resistencia mais proxima, com margem ATR."""
    margin = atr * _sl_margin_multiplier(op_type)

    for r in resistances:
        sl_candidate = r["price"] + margin
        distance_pct = (sl_candidate - entry) / entry * 100

        if distance_pct < MIN_SL_DISTANCE_PCT:
            continue
        if distance_pct > MAX_SL_DISTANCE_PCT:
            continue

        return sl_candidate, f"acima_{r['source']}"

    # Fallback ATR
    sl_atr = entry + (atr * _sl_atr_multiplier(op_type))
    return sl_atr, "ATR_fallback"


def _find_sell_tp1(
    entry: float, supports: List[Dict], atr: float, sl: float, op_type: str
) -> Tuple[float, str]:
    """TP1 para SELL: no suporte mais proximo."""
    sl_distance = sl - entry
    min_tp1 = entry - (sl_distance * MIN_RISK_REWARD)

    for s in supports:
        if s["price"] < min_tp1:
            return s["price"], f"em_{s['source']}"

    tp1_atr = entry - (atr * _tp1_atr_multiplier(op_type))
    if tp1_atr < min_tp1:
        return tp1_atr, "ATR_fallback"
    return min_tp1, "min_RR_1.5"


def _find_sell_tp2(
    entry: float, supports: List[Dict], atr: float, tp1: float, op_type: str
) -> Tuple[float, str]:
    """TP2 para SELL: no segundo suporte abaixo de TP1."""
    for s in supports:
        if s["price"] < tp1 * 0.997:
            return s["price"], f"em_{s['source']}"

    tp1_distance = entry - tp1
    tp2 = entry - (tp1_distance * 1.5)
    return tp2, "1.5x_TP1_dist"


# ===== Multiplicadores por tipo de operacao =====

def _sl_margin_multiplier(op_type: str) -> float:
    """Margem de seguranca abaixo/acima do nivel tecnico (em multiplos de ATR)."""
    return {
        "SCALP": 0.2,
        "DAY_TRADE": 0.3,
        "SWING_TRADE": 0.5,
        "POSITION_TRADE": 0.7,
    }.get(op_type, 0.3)


def _sl_atr_multiplier(op_type: str) -> float:
    """Multiplicador ATR para SL fallback."""
    return {
        "SCALP": 1.0,
        "DAY_TRADE": 1.5,
        "SWING_TRADE": 2.0,
        "POSITION_TRADE": 3.0,
    }.get(op_type, 1.5)


def _tp1_atr_multiplier(op_type: str) -> float:
    """Multiplicador ATR para TP1 fallback (reduzido para alvos realistas)."""
    return {
        "SCALP": 1.0,
        "DAY_TRADE": 2.0,
        "SWING_TRADE": 3.0,
        "POSITION_TRADE": 4.5,
    }.get(op_type, 2.0)


def _validate_risk_reward(
    entry: float, direction: str, result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Valida e ajusta para garantir risk:reward minimo.
    Se TP1 nao atinge 1.5:1, ajusta TP1.
    """
    sl = result["stop_loss"]
    tp1 = result["take_profit_1"]
    tp2 = result["take_profit_2"]

    if direction == "BUY":
        sl_dist = entry - sl
        tp1_dist = tp1 - entry
        if sl_dist > 0 and tp1_dist / sl_dist < MIN_RISK_REWARD:
            tp1 = entry + (sl_dist * MIN_RISK_REWARD)
            result["take_profit_1"] = round(tp1, 8)
            result["tp1_method"] += "+RR_ajustado"
            # Ajustar TP2 se ficou abaixo do TP1
            if tp2 <= tp1:
                tp2 = entry + (sl_dist * MIN_RISK_REWARD * 1.5)
                result["take_profit_2"] = round(tp2, 8)
                result["tp2_method"] += "+RR_ajustado"
    else:
        sl_dist = sl - entry
        tp1_dist = entry - tp1
        if sl_dist > 0 and tp1_dist / sl_dist < MIN_RISK_REWARD:
            tp1 = entry - (sl_dist * MIN_RISK_REWARD)
            result["take_profit_1"] = round(tp1, 8)
            result["tp1_method"] += "+RR_ajustado"
            if tp2 >= tp1:
                tp2 = entry - (sl_dist * MIN_RISK_REWARD * 1.5)
                result["take_profit_2"] = round(tp2, 8)
                result["tp2_method"] += "+RR_ajustado"

    # Recalcular risk:reward final
    if direction == "BUY":
        sl_dist = entry - result["stop_loss"]
        tp1_dist = result["take_profit_1"] - entry
    else:
        sl_dist = result["stop_loss"] - entry
        tp1_dist = entry - result["take_profit_1"]

    result["risk_reward"] = round(tp1_dist / sl_dist, 2) if sl_dist > 0 else 0
    result["sl_distance_pct"] = round(sl_dist / entry * 100, 2)
    result["tp1_distance_pct"] = round(tp1_dist / entry * 100, 2)

    return result
