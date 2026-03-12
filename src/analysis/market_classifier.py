"""
Market condition classifier - determines optimal operation type
"""
from typing import Any, Dict

from src.core.logger import get_logger

logger = get_logger(__name__)


def classify_market_condition(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Classifica as condições de mercado e recomenda o tipo de operação ideal.

    Retorna:
        - operation_type: SCALP, DAY_TRADE, SWING_TRADE, POSITION_TRADE
        - confidence: 1-10
        - reasoning: explicação
        - parameters: stops e targets sugeridos
    """
    try:
        volatility = analysis.get("volatility", {})
        trend = analysis.get("trend_analysis", {})
        volume = analysis.get("volume_flow", {})
        indicators = analysis.get("key_indicators", {})

        volatility_level = volatility.get("level", "MEDIUM")
        atr_pct = volatility.get("atr_percent", 2.0)
        adx_value = trend.get("adx_value", 20)
        trend_strength = trend.get("trend_strength_interpretation", "WEAK")
        primary_trend = trend.get("primary_trend", "NEUTRAL")
        confluence_score = trend.get("confluence_score", 0)
        rsi = indicators.get("rsi", {}).get("value", 50)
        volume_trend = volume.get("obv_trend", "neutral")

        scores = {"SCALP": 0, "DAY_TRADE": 0, "SWING_TRADE": 0, "POSITION_TRADE": 0}

        # Volatility rules
        if volatility_level == "HIGH" or atr_pct > 3.0:
            scores["SCALP"] += 3
            scores["DAY_TRADE"] += 2
        elif volatility_level == "MEDIUM" or 1.5 <= atr_pct <= 3.0:
            scores["DAY_TRADE"] += 3
            scores["SWING_TRADE"] += 2
        else:
            scores["SWING_TRADE"] += 3
            scores["POSITION_TRADE"] += 3

        # Trend rules
        if adx_value < 20 or trend_strength == "WEAK":
            scores["SCALP"] += 3
            scores["DAY_TRADE"] += 1
        elif 20 <= adx_value < 35 or trend_strength == "MODERATE":
            scores["DAY_TRADE"] += 3
            scores["SWING_TRADE"] += 2
        elif 35 <= adx_value < 50 or trend_strength == "STRONG":
            scores["SWING_TRADE"] += 4
            scores["DAY_TRADE"] += 1
        else:
            scores["POSITION_TRADE"] += 4
            scores["SWING_TRADE"] += 2

        # Confluence rules
        if confluence_score >= 4:
            scores["SWING_TRADE"] += 2
            scores["POSITION_TRADE"] += 2
        elif confluence_score <= 2:
            scores["SCALP"] += 2
            scores["DAY_TRADE"] += 1

        # Momentum rules
        if 40 <= rsi <= 60:
            scores["SCALP"] += 2
        elif rsi < 30 or rsi > 70:
            scores["SWING_TRADE"] += 2

        # Volume rules
        if volume_trend in ["increasing", "strong_increasing"]:
            scores["SWING_TRADE"] += 1
            scores["POSITION_TRADE"] += 1

        # Primary trend rules
        if primary_trend in ["NEUTRAL", "SIDEWAYS"]:
            scores["SCALP"] += 2
            scores["DAY_TRADE"] += 1
        elif primary_trend in ["BULLISH", "BEARISH"]:
            scores["SWING_TRADE"] += 1
        elif primary_trend in ["STRONG_BULLISH", "STRONG_BEARISH"]:
            scores["POSITION_TRADE"] += 2

        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]
        total_score = sum(scores.values())

        if total_score > 0:
            dominance = best_score / total_score
            confidence = min(10, int(dominance * 15) + 3)
        else:
            confidence = 5

        parameters = {
            "SCALP": {"stop_loss_pct": 0.4, "take_profit_1_pct": 0.6, "take_profit_2_pct": 1.2, "max_duration_hours": 0.5, "min_volume_multiplier": 1.5},
            "DAY_TRADE": {"stop_loss_pct": 1.2, "take_profit_1_pct": 2.0, "take_profit_2_pct": 3.5, "max_duration_hours": 8, "min_volume_multiplier": 1.2},
            "SWING_TRADE": {"stop_loss_pct": 2.5, "take_profit_1_pct": 4.0, "take_profit_2_pct": 7.0, "max_duration_hours": 168, "min_volume_multiplier": 1.0},
            "POSITION_TRADE": {"stop_loss_pct": 6.0, "take_profit_1_pct": 12.0, "take_profit_2_pct": 20.0, "max_duration_hours": 672, "min_volume_multiplier": 0.8}
        }

        reasoning_parts = []
        if volatility_level == "HIGH":
            reasoning_parts.append("Alta volatilidade favorece operações curtas")
        if adx_value > 35:
            reasoning_parts.append(f"ADX forte ({adx_value}) indica tendência estabelecida")
        if confluence_score >= 4:
            reasoning_parts.append("Alta confluência entre timeframes")
        if primary_trend not in ["NEUTRAL", "SIDEWAYS"]:
            reasoning_parts.append(f"Tendência {primary_trend} identificada")

        reasoning = ". ".join(reasoning_parts) if reasoning_parts else "Análise baseada em múltiplos fatores"

        return {
            "operation_type": best_type,
            "confidence": confidence,
            "reasoning": reasoning,
            "parameters": parameters[best_type],
            "scores": scores,
            "market_conditions": {
                "volatility": volatility_level,
                "trend_strength": trend_strength,
                "adx": adx_value,
                "confluence": confluence_score,
                "rsi": rsi
            }
        }

    except Exception as e:
        logger.exception(f"Erro ao classificar condições de mercado: {e}")
        return {
            "operation_type": "SWING_TRADE",
            "confidence": 5,
            "reasoning": "Fallback devido a erro na análise",
            "parameters": {"stop_loss_pct": 2.5, "take_profit_1_pct": 4.0, "take_profit_2_pct": 7.0, "max_duration_hours": 168, "min_volume_multiplier": 1.0},
            "scores": {},
            "market_conditions": {}
        }
