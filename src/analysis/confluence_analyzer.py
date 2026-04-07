"""
Confluence Analyzer - Central system combining 5 analysis modules.
Source: smart_trading_system
Weights:
1. Market Structure (25%)
2. Trend Analysis multi-TF (25%)
3. Leading Indicators (20%)
4. Strategy Signals (20%)
5. S/R Levels (10%)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

from src.core.logger import get_logger

logger = get_logger(__name__)


class ConfluenceLevel(Enum):
    VERY_HIGH = "VERY_HIGH"      # >= 0.8
    HIGH = "HIGH"                # >= 0.65
    MODERATE = "MODERATE"        # >= 0.5
    LOW = "LOW"                  # >= 0.35
    VERY_LOW = "VERY_LOW"       # < 0.35


class SignalDirection(Enum):
    STRONG_BULLISH = "STRONG_BULLISH"
    BULLISH = "BULLISH"
    NEUTRAL = "NEUTRAL"
    BEARISH = "BEARISH"
    STRONG_BEARISH = "STRONG_BEARISH"


@dataclass
class ConfluenceResult:
    direction: SignalDirection
    score: float  # 0-1
    level: ConfluenceLevel
    components: Dict[str, float]
    signals: List[str]
    conflicts: List[str]
    recommendation: str
    metadata: Dict = field(default_factory=dict)


class ConfluenceAnalyzer:
    """
    Central confluence system that combines multiple analysis modules
    with configurable weights to generate a unified trading signal.
    """

    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        self.weights = {
            "market_structure": config.get("weight_structure", 0.25),
            "trend_analysis": config.get("weight_trend", 0.25),
            "leading_indicators": config.get("weight_leading", 0.20),
            "strategy_signals": config.get("weight_strategy", 0.20),
            "sr_levels": config.get("weight_sr", 0.10),
        }
        self.min_score_for_signal = config.get("min_score", 0.5)
        self.conflict_penalty = config.get("conflict_penalty", 0.15)

    def analyze(
        self,
        structure_data: Optional[Dict] = None,
        trend_data: Optional[Dict] = None,
        leading_data: Optional[Dict] = None,
        strategy_data: Optional[Dict] = None,
        sr_data: Optional[Dict] = None,
    ) -> ConfluenceResult:
        """
        Run confluence analysis combining all available data sources.

        Args:
            structure_data: Market structure analysis (HH/HL/LH/LL, phase)
            trend_data: Multi-timeframe trend data
            leading_data: Leading indicators (order flow, volume profile, liquidity)
            strategy_data: Strategy signals (breakout, swing, trend, mean_reversion)
            sr_data: Support/Resistance level data
        """
        components = {}
        all_signals = []
        all_conflicts = []
        bullish_score = 0.0
        bearish_score = 0.0

        # 1. Market Structure (25%)
        if structure_data:
            struct_result = self._analyze_structure(structure_data)
            components["market_structure"] = struct_result
            w = self.weights["market_structure"]
            if struct_result["bias"] == "BULLISH":
                bullish_score += w * struct_result["score"]
            elif struct_result["bias"] == "BEARISH":
                bearish_score += w * struct_result["score"]
            all_signals.extend(struct_result.get("signals", []))

        # 2. Trend Analysis (25%)
        if trend_data:
            trend_result = self._analyze_trend(trend_data)
            components["trend_analysis"] = trend_result
            w = self.weights["trend_analysis"]
            if trend_result["bias"] == "BULLISH":
                bullish_score += w * trend_result["score"]
            elif trend_result["bias"] == "BEARISH":
                bearish_score += w * trend_result["score"]
            all_signals.extend(trend_result.get("signals", []))

        # 3. Leading Indicators (20%)
        if leading_data:
            lead_result = self._analyze_leading(leading_data)
            components["leading_indicators"] = lead_result
            w = self.weights["leading_indicators"]
            if lead_result["bias"] == "BULLISH":
                bullish_score += w * lead_result["score"]
            elif lead_result["bias"] == "BEARISH":
                bearish_score += w * lead_result["score"]
            all_signals.extend(lead_result.get("signals", []))

        # 4. Strategy Signals (20%)
        if strategy_data:
            strat_result = self._analyze_strategies(strategy_data)
            components["strategy_signals"] = strat_result
            w = self.weights["strategy_signals"]
            if strat_result["bias"] == "BULLISH":
                bullish_score += w * strat_result["score"]
            elif strat_result["bias"] == "BEARISH":
                bearish_score += w * strat_result["score"]
            all_signals.extend(strat_result.get("signals", []))

        # 5. S/R Levels (10%)
        if sr_data:
            sr_result = self._analyze_sr(sr_data)
            components["sr_levels"] = sr_result
            w = self.weights["sr_levels"]
            if sr_result["bias"] == "BULLISH":
                bullish_score += w * sr_result["score"]
            elif sr_result["bias"] == "BEARISH":
                bearish_score += w * sr_result["score"]
            all_signals.extend(sr_result.get("signals", []))

        # Detect conflicts
        all_conflicts = self._detect_conflicts(components)

        # Apply conflict penalty only to the stronger side
        if all_conflicts:
            penalty = self.conflict_penalty * len(all_conflicts)
            if bullish_score >= bearish_score:
                bullish_score = max(0, bullish_score - penalty)
            else:
                bearish_score = max(0, bearish_score - penalty)

        # Determine final direction and score
        direction, score = self._determine_direction(bullish_score, bearish_score)
        level = self._classify_level(score)
        recommendation = self._generate_recommendation(direction, level, all_conflicts)

        return ConfluenceResult(
            direction=direction,
            score=score,
            level=level,
            components=components,
            signals=all_signals,
            conflicts=all_conflicts,
            recommendation=recommendation,
            metadata={
                "bullish_score": bullish_score,
                "bearish_score": bearish_score,
                "num_sources": sum(1 for v in [structure_data, trend_data, leading_data, strategy_data, sr_data] if v),
                "weights": self.weights,
            },
        )

    def _analyze_structure(self, data: Dict) -> Dict:
        """Analyze market structure component."""
        phase = data.get("phase", "UNKNOWN")
        trend = data.get("trend", "NEUTRAL")
        confidence = data.get("confidence", 0.5)

        bias = "NEUTRAL"
        score = 0.5
        signals = []

        if trend in ("UPTREND", "BULLISH"):
            bias = "BULLISH"
            score = min(0.5 + confidence * 0.5, 1.0)
            signals.append(f"Structure: {phase} uptrend")
        elif trend in ("DOWNTREND", "BEARISH"):
            bias = "BEARISH"
            score = min(0.5 + confidence * 0.5, 1.0)
            signals.append(f"Structure: {phase} downtrend")

        if phase in ("Accumulation",):
            bias = "BULLISH"
            score = max(score, 0.7)
        elif phase in ("Distribution",):
            bias = "BEARISH"
            score = max(score, 0.7)

        return {"bias": bias, "score": score, "signals": signals}

    def _analyze_trend(self, data: Dict) -> Dict:
        """Analyze trend component."""
        direction = data.get("direction", "NEUTRAL")
        strength = data.get("strength", 0.5)
        alignment = data.get("alignment", False)

        bias = "NEUTRAL"
        score = 0.5
        signals = []

        if direction in ("BULLISH", "BULL"):
            bias = "BULLISH"
            score = 0.5 + strength * 0.5
            if alignment:
                score = min(score + 0.15, 1.0)
                signals.append("Trend: Multi-TF aligned bullish")
            else:
                signals.append("Trend: Bullish (partial alignment)")
        elif direction in ("BEARISH", "BEAR"):
            bias = "BEARISH"
            score = 0.5 + strength * 0.5
            if alignment:
                score = min(score + 0.15, 1.0)
                signals.append("Trend: Multi-TF aligned bearish")
            else:
                signals.append("Trend: Bearish (partial alignment)")

        return {"bias": bias, "score": score, "signals": signals}

    def _analyze_leading(self, data: Dict) -> Dict:
        """Analyze leading indicators component."""
        order_flow = data.get("order_flow", {})
        volume_profile = data.get("volume_profile", {})

        bias = "NEUTRAL"
        score = 0.5
        signals = []

        # Order flow
        flow_bias = order_flow.get("bias", "NEUTRAL")
        flow_score = order_flow.get("score", 0.5)

        # Volume profile
        volume_profile.get("poc_position", "MIDDLE")

        if flow_bias == "BULLISH" and flow_score > 0.6:
            bias = "BULLISH"
            score = flow_score
            signals.append(f"Order flow: Bullish ({flow_score:.0%})")
        elif flow_bias == "BEARISH" and flow_score > 0.6:
            bias = "BEARISH"
            score = flow_score
            signals.append(f"Order flow: Bearish ({flow_score:.0%})")

        return {"bias": bias, "score": score, "signals": signals}

    def _analyze_strategies(self, data: Dict) -> Dict:
        """Analyze strategy signals component."""
        signals_list = data.get("signals", [])

        buy_signals = sum(1 for s in signals_list if s.get("direction") == "BUY")
        sell_signals = sum(1 for s in signals_list if s.get("direction") == "SELL")
        total = buy_signals + sell_signals

        bias = "NEUTRAL"
        score = 0.5
        signals = []

        if total > 0:
            if buy_signals > sell_signals:
                bias = "BULLISH"
                score = buy_signals / total
                signals.append(f"Strategies: {buy_signals} BUY vs {sell_signals} SELL")
            elif sell_signals > buy_signals:
                bias = "BEARISH"
                score = sell_signals / total
                signals.append(f"Strategies: {sell_signals} SELL vs {buy_signals} BUY")

            # Average confluence from strategy signals
            avg_conf = np.mean([s.get("confluence", 0.5) for s in signals_list]) if signals_list else 0.5
            score = (score + avg_conf) / 2

        return {"bias": bias, "score": score, "signals": signals}

    def _analyze_sr(self, data: Dict) -> Dict:
        """Analyze support/resistance levels."""
        nearest_support = data.get("nearest_support", {})
        nearest_resistance = data.get("nearest_resistance", {})
        current_price = data.get("current_price", 0)

        bias = "NEUTRAL"
        score = 0.5
        signals = []

        if current_price > 0 and nearest_support and nearest_resistance:
            sup_price = nearest_support.get("price", 0)
            res_price = nearest_resistance.get("price", 0)

            if sup_price > 0 and res_price > 0:
                dist_to_support = (current_price - sup_price) / current_price
                dist_to_resistance = (res_price - current_price) / current_price

                if dist_to_support < 0.01:  # Very close to support
                    bias = "BULLISH"
                    score = 0.7 + nearest_support.get("strength", 1) * 0.05
                    signals.append("Near strong support level")
                elif dist_to_resistance < 0.01:  # Very close to resistance
                    bias = "BEARISH"
                    score = 0.7 + nearest_resistance.get("strength", 1) * 0.05
                    signals.append("Near strong resistance level")

        return {"bias": bias, "score": min(score, 1.0), "signals": signals}

    def _detect_conflicts(self, components: Dict) -> List[str]:
        """Detect conflicts between analysis components."""
        conflicts = []
        biases = {}

        for name, comp in components.items():
            if isinstance(comp, dict) and "bias" in comp:
                biases[name] = comp["bias"]

        bullish = [k for k, v in biases.items() if v == "BULLISH"]
        bearish = [k for k, v in biases.items() if v == "BEARISH"]

        if bullish and bearish:
            conflicts.append(
                f"Conflict: {', '.join(bullish)} say BULLISH vs {', '.join(bearish)} say BEARISH"
            )

        return conflicts

    def _determine_direction(self, bull: float, bear: float) -> tuple:
        """Determine final direction and score."""
        diff = bull - bear
        total = bull + bear

        if total < 0.1:
            return SignalDirection.NEUTRAL, 0.0

        score = abs(diff) / max(total, 0.01)

        if diff > 0.3:
            return SignalDirection.STRONG_BULLISH, min(score, 1.0)
        elif diff > 0.1:
            return SignalDirection.BULLISH, min(score, 1.0)
        elif diff < -0.3:
            return SignalDirection.STRONG_BEARISH, min(score, 1.0)
        elif diff < -0.1:
            return SignalDirection.BEARISH, min(score, 1.0)
        return SignalDirection.NEUTRAL, score

    def _classify_level(self, score: float) -> ConfluenceLevel:
        if score >= 0.8:
            return ConfluenceLevel.VERY_HIGH
        elif score >= 0.65:
            return ConfluenceLevel.HIGH
        elif score >= 0.5:
            return ConfluenceLevel.MODERATE
        elif score >= 0.35:
            return ConfluenceLevel.LOW
        return ConfluenceLevel.VERY_LOW

    def _generate_recommendation(
        self, direction: SignalDirection, level: ConfluenceLevel, conflicts: List[str]
    ) -> str:
        if level in (ConfluenceLevel.VERY_LOW, ConfluenceLevel.LOW):
            return "NO_TRADE - Insufficient confluence"
        if conflicts and level == ConfluenceLevel.MODERATE:
            return "WAIT - Conflicting signals, wait for clarity"
        if direction == SignalDirection.NEUTRAL:
            return "NO_TRADE - No clear direction"
        if level == ConfluenceLevel.VERY_HIGH:
            action = "BUY" if "BULLISH" in direction.value else "SELL"
            return f"STRONG_{action} - Very high confluence"
        if level == ConfluenceLevel.HIGH:
            action = "BUY" if "BULLISH" in direction.value else "SELL"
            return f"{action} - High confluence"
        action = "BUY" if "BULLISH" in direction.value else "SELL"
        return f"CAUTIOUS_{action} - Moderate confluence"
