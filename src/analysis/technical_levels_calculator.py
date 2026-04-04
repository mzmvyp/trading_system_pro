"""
Technical SL/TP Calculator - Calcula Stop Loss e Take Profit baseados em niveis tecnicos reais.

Usa: suportes/resistencias, Fibonacci, Bollinger Bands, EMAs, Volume POC, market structure.
Nunca usa percentuais fixos arbitrarios.
"""
from typing import Any, Dict, List, Tuple

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
    optimized_params: Dict = None,
) -> Dict[str, Any]:
    """
    Calcula SL, TP1 e TP2 baseados em niveis tecnicos reais.

    Hierarquia de niveis (ordem de prioridade):
    1. Suporte/Resistencia de structure points (swing highs/lows)
    2. Fibonacci retracement levels
    3. EMAs importantes (20, 50, 200)
    4. Bollinger Bands
    5. Volume POC
    6. Fallback: ATR-based (ultimo recurso) — usa params otimizados se disponíveis

    Args:
        entry_price: Preco de entrada
        direction: BUY ou SELL
        analysis_data: Dados completos da analise (de prepare_analysis_for_llm)
        operation_type: SCALP, DAY_TRADE, SWING_TRADE, POSITION_TRADE
        optimized_params: Params otimizados do optimizer (sl_atr_multiplier, tp1_atr_multiplier, etc.)

    Returns:
        Dict com stop_loss, take_profit_1, take_profit_2, metodo usado, e detalhes
    """
    if entry_price <= 0:
        return {"error": "entry_price invalido"}

    # Coletar todos os niveis tecnicos disponiveis
    levels = _collect_all_levels(entry_price, analysis_data)

    if direction == "BUY":
        result = _calculate_buy_levels(entry_price, levels, operation_type, optimized_params)
    else:
        result = _calculate_sell_levels(entry_price, levels, operation_type, optimized_params)

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
    indicators.get("ema_structure", {})  # usado para contexto
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
    # Cap ATR: MAX_SL_DISTANCE_PCT (15%) já limita o SL nos helpers,
    # mas aqui limitamos a 10% para evitar outliers extremos de ATR.
    # Antigo cap de 5% era muito restritivo para altcoins voláteis.
    max_atr = current_price * 0.10
    if atr > max_atr:
        atr = max_atr

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
    optimized_params: Dict = None,
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
    sl, sl_method = _find_buy_sl(entry, supports, atr, operation_type, optimized_params)

    # === TAKE PROFIT 1 ===
    tp1, tp1_method = _find_buy_tp1(entry, resistances, atr, sl, operation_type, optimized_params)

    # === TAKE PROFIT 2 ===
    tp2, tp2_method = _find_buy_tp2(entry, resistances, atr, tp1, operation_type, optimized_params)

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
    optimized_params: Dict = None,
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
    sl, sl_method = _find_sell_sl(entry, resistances, atr, operation_type, optimized_params)

    # === TAKE PROFIT 1 (no suporte) ===
    tp1, tp1_method = _find_sell_tp1(entry, supports, atr, sl, operation_type, optimized_params)

    # === TAKE PROFIT 2 (no segundo suporte) ===
    tp2, tp2_method = _find_sell_tp2(entry, supports, atr, tp1, operation_type, optimized_params)

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
    entry: float, supports: List[Dict], atr: float, op_type: str,
    optimized_params: Dict = None,
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

    # Fallback: usar ATR com multiplicador do optimizer (se disponível)
    sl_mult = _sl_atr_multiplier(op_type, optimized_params)
    sl_atr = entry - (atr * sl_mult)
    # Clamp ATR fallback to respect MAX_SL_DISTANCE_PCT
    sl_distance_pct = (entry - sl_atr) / entry * 100 if entry > 0 else 0
    if sl_distance_pct > MAX_SL_DISTANCE_PCT:
        sl_atr = entry * (1 - MAX_SL_DISTANCE_PCT / 100)
    elif sl_distance_pct < MIN_SL_DISTANCE_PCT:
        sl_atr = entry * (1 - MIN_SL_DISTANCE_PCT / 100)
    # Never allow SL <= 0
    if sl_atr <= 0:
        sl_atr = entry * (1 - min(MAX_SL_DISTANCE_PCT, 5.0) / 100)
    method = f"ATR_x{sl_mult:.1f}" if optimized_params else "ATR_fallback"
    return sl_atr, method


def _find_buy_tp1(
    entry: float, resistances: List[Dict], atr: float, sl: float, op_type: str,
    optimized_params: Dict = None,
) -> Tuple[float, str]:
    """TP1 para BUY: na resistencia mais proxima."""
    sl_distance = entry - sl
    min_tp1 = entry + (sl_distance * MIN_RISK_REWARD)  # Minimo 1.5:1 R:R

    for r in resistances:
        if r["price"] > min_tp1:
            return r["price"], f"em_{r['source']}"

    # Se nenhuma resistencia atende o R:R minimo, usar ATR com multiplicador otimizado
    tp1_mult = _tp1_atr_multiplier(op_type, optimized_params)
    tp1_atr = entry + (atr * tp1_mult)
    if tp1_atr > min_tp1:
        method = f"ATR_x{tp1_mult:.1f}" if optimized_params else "ATR_fallback"
        return tp1_atr, method
    return min_tp1, "min_RR_1.5"


def _find_buy_tp2(
    entry: float, resistances: List[Dict], atr: float, tp1: float, op_type: str,
    optimized_params: Dict = None,
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
    entry: float, resistances: List[Dict], atr: float, op_type: str,
    optimized_params: Dict = None,
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

    # Fallback ATR com multiplicador do optimizer (se disponível)
    sl_mult = _sl_atr_multiplier(op_type, optimized_params)
    sl_atr = entry + (atr * sl_mult)
    # Clamp ATR fallback to respect MAX_SL_DISTANCE_PCT
    sl_distance_pct = (sl_atr - entry) / entry * 100 if entry > 0 else 0
    if sl_distance_pct > MAX_SL_DISTANCE_PCT:
        sl_atr = entry * (1 + MAX_SL_DISTANCE_PCT / 100)
    elif sl_distance_pct < MIN_SL_DISTANCE_PCT:
        sl_atr = entry * (1 + MIN_SL_DISTANCE_PCT / 100)
    method = f"ATR_x{sl_mult:.1f}" if optimized_params else "ATR_fallback"
    return sl_atr, method


def _find_sell_tp1(
    entry: float, supports: List[Dict], atr: float, sl: float, op_type: str,
    optimized_params: Dict = None,
) -> Tuple[float, str]:
    """TP1 para SELL: no suporte mais proximo."""
    sl_distance = sl - entry
    min_tp1 = entry - (sl_distance * MIN_RISK_REWARD)

    for s in supports:
        if s["price"] < min_tp1:
            return s["price"], f"em_{s['source']}"

    tp1_mult = _tp1_atr_multiplier(op_type, optimized_params)
    tp1_atr = entry - (atr * tp1_mult)
    if tp1_atr < min_tp1 and tp1_atr > 0:
        method = f"ATR_x{tp1_mult:.1f}" if optimized_params else "ATR_fallback"
        return tp1_atr, method
    # Ensure TP never goes negative
    if min_tp1 <= 0:
        min_tp1 = entry * 0.97  # Fallback: 3% below entry
    return min_tp1, "min_RR_1.5"


def _find_sell_tp2(
    entry: float, supports: List[Dict], atr: float, tp1: float, op_type: str,
    optimized_params: Dict = None,
) -> Tuple[float, str]:
    """TP2 para SELL: no segundo suporte abaixo de TP1."""
    for s in supports:
        if s["price"] < tp1 * 0.997:
            return s["price"], f"em_{s['source']}"

    tp1_distance = entry - tp1
    tp2 = entry - (tp1_distance * 1.5)
    # Ensure TP2 never goes negative
    if tp2 <= 0:
        tp2 = tp1 * 0.97  # Fallback: 3% below TP1
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


def _sl_atr_multiplier(op_type: str, optimized_params: Dict = None) -> float:
    """Multiplicador ATR para SL fallback. Usa params otimizados se disponíveis."""
    if optimized_params:
        opt_sl = optimized_params.get("sl_atr_multiplier")
        if opt_sl and opt_sl > 0:
            return float(opt_sl)
    return {
        "SCALP": 1.0,
        "DAY_TRADE": 1.5,
        "SWING_TRADE": 2.0,
        "POSITION_TRADE": 3.0,
    }.get(op_type, 1.5)


def _tp1_atr_multiplier(op_type: str, optimized_params: Dict = None) -> float:
    """Multiplicador ATR para TP1 fallback. Usa params otimizados se disponíveis."""
    if optimized_params:
        opt_tp1 = optimized_params.get("tp1_atr_multiplier")
        if opt_tp1 and opt_tp1 > 0:
            return float(opt_tp1)
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


# ===== SIDEWAYS MEAN REVERSION MODE =====

def assess_range_quality(analysis_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Avalia a qualidade do range lateral para decidir se vale operar mean reversion.

    Critérios:
    - BB width estável (não expandindo rapidamente)
    - Preço dentro das bandas (não rompendo)
    - RSI não em extremo absoluto (< 20 ou > 80 = possível breakout)
    - ADX baixo e estável (confirma lateral)

    Returns:
        Dict com: tradeable (bool), quality_score (0-1), reason (str), bb_data (dict)
    """
    indicators = analysis_data.get("key_indicators", {})
    raw = analysis_data.get("_raw_indicators", {})
    trend_data = analysis_data.get("trend_analysis", {})

    bb_upper = raw.get("bb_upper", 0)
    bb_lower = raw.get("bb_lower", 0)
    bb_middle = raw.get("bb_middle", 0)
    close = raw.get("close", indicators.get("close", indicators.get("price", 0)))
    adx = trend_data.get("adx", indicators.get("adx", {}).get("value", 25) if isinstance(indicators.get("adx"), dict) else indicators.get("adx", 25))
    rsi = indicators.get("rsi", {}).get("value", 50) if isinstance(indicators.get("rsi"), dict) else indicators.get("rsi", 50)

    if not bb_upper or not bb_lower or bb_upper <= bb_lower or not close:
        return {"tradeable": False, "quality_score": 0, "reason": "BB data indisponível",
                "bb_data": {"upper": 0, "lower": 0, "middle": 0, "width_pct": 0}}

    bb_width_pct = (bb_upper - bb_lower) / close * 100
    bb_position = (close - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5

    score = 0.0
    reasons = []

    # 1. Range deve ter tamanho mínimo (> 0.5%) para ter lucro viável
    if bb_width_pct < 0.5:
        return {"tradeable": False, "quality_score": 0, "reason": "Range muito estreito (BB width < 0.5%)",
                "bb_data": {"upper": bb_upper, "lower": bb_lower, "middle": bb_middle,
                            "width_pct": round(bb_width_pct, 3), "position": round(bb_position, 3)}}

    # 2. Range não pode ser excessivo (> 8%) — seria volátil demais para mean reversion
    if bb_width_pct > 8.0:
        return {"tradeable": False, "quality_score": 0, "reason": f"Range muito amplo ({bb_width_pct:.1f}% > 8%)",
                "bb_data": {"upper": bb_upper, "lower": bb_lower, "middle": bb_middle,
                            "width_pct": round(bb_width_pct, 3), "position": round(bb_position, 3)}}

    # 3. Preço deve estar PERTO de uma extremidade (< 0.25 ou > 0.75)
    # Mean reversion = comprar na banda inferior, vender na superior
    near_extreme = bb_position < 0.25 or bb_position > 0.75
    if near_extreme:
        score += 0.4
        reasons.append(f"BB pos={bb_position:.2f} (perto de extremo)")
    else:
        score += 0.1
        reasons.append(f"BB pos={bb_position:.2f} (meio do range)")

    # 4. ADX baixo confirma lateral
    if adx < 15:
        score += 0.3
        reasons.append(f"ADX={adx:.1f} (forte lateral)")
    elif adx < 20:
        score += 0.2
        reasons.append(f"ADX={adx:.1f} (lateral)")
    elif adx < 25:
        score += 0.1
        reasons.append(f"ADX={adx:.1f} (fraco lateral)")

    # 5. RSI moderado (não em extremo absoluto que indique breakout)
    if 25 <= rsi <= 75:
        score += 0.2
        reasons.append(f"RSI={rsi:.1f} (dentro do range)")
    elif 20 <= rsi <= 80:
        score += 0.1
        reasons.append(f"RSI={rsi:.1f} (perto do extremo)")
    else:
        # RSI < 20 ou > 80 pode indicar breakout iminente — não operar mean reversion
        return {"tradeable": False, "quality_score": 0,
                "reason": f"RSI em extremo absoluto ({rsi:.1f}) — possível breakout",
                "bb_data": {"upper": bb_upper, "lower": bb_lower, "middle": bb_middle,
                            "width_pct": round(bb_width_pct, 3), "position": round(bb_position, 3)}}

    # 6. Width ideal para scalp (1% - 4%) = bonus
    if 1.0 <= bb_width_pct <= 4.0:
        score += 0.1
        reasons.append(f"BB width={bb_width_pct:.2f}% (ideal)")

    tradeable = score >= 0.4 and near_extreme
    reason = " | ".join(reasons)

    return {
        "tradeable": tradeable,
        "quality_score": round(score, 2),
        "reason": reason,
        "bb_data": {
            "upper": bb_upper,
            "lower": bb_lower,
            "middle": bb_middle,
            "width_pct": round(bb_width_pct, 3),
            "position": round(bb_position, 3),
        },
    }


def calculate_sideways_mean_reversion_levels(
    entry_price: float,
    direction: str,
    bb_data: Dict[str, Any],
    atr: float = 0,
) -> Dict[str, Any]:
    """
    Calcula SL/TP específicos para mean reversion em mercado lateral.

    Lógica:
    - BUY perto da BB inferior → TP1 = BB middle, TP2 = BB upper (com margem)
    - SELL perto da BB superior → TP1 = BB middle, TP2 = BB lower (com margem)
    - SL = além da banda (com margem ATR pequena)

    Isso gera R:R bom porque o SL é curto (logo além da banda) e o TP é a banda oposta.
    """
    bb_upper = bb_data["upper"]
    bb_lower = bb_data["lower"]
    bb_middle = bb_data["middle"]

    if not atr or atr <= 0:
        atr = entry_price * 0.005  # fallback 0.5%

    # Margem de segurança: 30% do ATR para não ser stopado no pavio
    margin = atr * 0.3

    if direction == "BUY":
        # BUY: entrada perto da BB lower
        # SL logo abaixo da BB lower
        sl = bb_lower - margin
        sl_method = "BB_lower_mean_rev"

        # TP1 = BB middle (centro do range)
        tp1 = bb_middle
        tp1_method = "BB_middle_mean_rev"

        # TP2 = 75% do caminho até BB upper (não esperar toque exato)
        tp2 = bb_middle + (bb_upper - bb_middle) * 0.75
        tp2_method = "BB_upper_75pct_mean_rev"

        # Validações
        if sl <= 0:
            sl = entry_price * 0.995
        if tp1 <= entry_price:
            tp1 = entry_price + (entry_price - sl) * 1.5
            tp1_method = "min_RR_mean_rev"
        if tp2 <= tp1:
            tp2 = tp1 + (tp1 - entry_price) * 0.5
    else:
        # SELL: entrada perto da BB upper
        # SL logo acima da BB upper
        sl = bb_upper + margin
        sl_method = "BB_upper_mean_rev"

        # TP1 = BB middle
        tp1 = bb_middle
        tp1_method = "BB_middle_mean_rev"

        # TP2 = 75% do caminho até BB lower
        tp2 = bb_middle - (bb_middle - bb_lower) * 0.75
        tp2_method = "BB_lower_75pct_mean_rev"

        # Validações
        if tp1 >= entry_price:
            tp1 = entry_price - (sl - entry_price) * 1.5
            tp1_method = "min_RR_mean_rev"
        if tp2 >= tp1:
            tp2 = tp1 - (entry_price - tp1) * 0.5
        if tp2 <= 0:
            tp2 = tp1 * 0.99

    sl_dist = abs(entry_price - sl) / entry_price * 100
    tp1_dist = abs(tp1 - entry_price) / entry_price * 100
    rr = tp1_dist / sl_dist if sl_dist > 0 else 0

    # Clampar SL dentro dos limites
    if sl_dist < MIN_SL_DISTANCE_PCT:
        if direction == "BUY":
            sl = entry_price * (1 - MIN_SL_DISTANCE_PCT / 100)
        else:
            sl = entry_price * (1 + MIN_SL_DISTANCE_PCT / 100)
        sl_dist = MIN_SL_DISTANCE_PCT
        sl_method += "+clamped"

    return {
        "stop_loss": round(sl, 8),
        "take_profit_1": round(tp1, 8),
        "take_profit_2": round(tp2, 8),
        "sl_method": sl_method,
        "tp1_method": tp1_method,
        "tp2_method": tp2_method,
        "sl_distance_pct": round(sl_dist, 2),
        "tp1_distance_pct": round(tp1_dist, 2),
        "risk_reward": round(rr, 2),
        "mode": "sideways_mean_reversion",
    }
