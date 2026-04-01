"""
Local Signal Generator — Gerador de sinais puramente técnico (sem LLM).

Analisa os mesmos dados coletados pelo prepare_analysis_for_llm() e gera
um sinal BUY/SELL/NO_SIGNAL baseado APENAS em regras técnicas objetivas.

Roda em paralelo com o DeepSeek (shadow mode) para coletar dados de
acertividade e eventualmente substituir a LLM como fonte de sinais.
"""

from typing import Any, Dict, Optional

from src.core.logger import get_logger

logger = get_logger(__name__)


class LocalSignalGenerator:
    """
    Gerador de sinais 100% local usando indicadores técnicos.

    Regras de decisão:
    - Score ponderado de múltiplos indicadores (-1 a +1 cada)
    - Score > 0 = BUY, Score < 0 = SELL
    - |Score| determina confiança (1-10)
    - Filtros obrigatórios: tendência, regime, R:R
    """

    # Pesos de cada componente (somam ~1.0)
    WEIGHTS = {
        "trend": 0.20,       # EMA alignment + tendência primária
        "rsi": 0.12,         # RSI oversold/overbought
        "macd": 0.12,        # MACD histogram + crossover
        "orderbook": 0.12,   # Pressão de compra/venda
        "cvd": 0.10,         # Cumulative Volume Delta
        "bb": 0.08,          # Bollinger Bands position
        "mtf": 0.10,         # Multi-timeframe alignment
        "adx": 0.08,         # Força da tendência
        "structure": 0.08,   # Proximidade de suporte/resistência
    }

    def __init__(self):
        pass

    def generate_signal(
        self,
        analysis_data: Dict[str, Any],
        market_regime: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Gera sinal BUY/SELL/NO_SIGNAL a partir dos dados de análise.

        Args:
            analysis_data: Dados de prepare_analysis_for_llm()
            market_regime: Resultado do MarketRegimeDetectorFutures

        Returns:
            Dict com signal, confidence, reasoning, component_scores
        """
        if "error" in analysis_data:
            return {
                "signal": "NO_SIGNAL",
                "confidence": 0,
                "reasoning": f"Dados indisponíveis: {analysis_data.get('error')}",
                "component_scores": {},
            }

        indicators = analysis_data.get("key_indicators", {})
        trend_data = analysis_data.get("trend_analysis", {})
        volume_flow = analysis_data.get("volume_flow", {})
        mtf = analysis_data.get("multi_timeframe", {})
        symbol = analysis_data.get("symbol", "UNKNOWN")

        # Calcular score de cada componente (-1.0 a +1.0)
        # Positivo = bullish, Negativo = bearish
        scores = {}

        # 1. TREND (EMA alignment)
        scores["trend"] = self._score_trend(trend_data)

        # 2. RSI
        scores["rsi"] = self._score_rsi(indicators)

        # 3. MACD
        scores["macd"] = self._score_macd(indicators)

        # 4. ORDERBOOK
        scores["orderbook"] = self._score_orderbook(volume_flow)

        # 5. CVD
        scores["cvd"] = self._score_cvd(volume_flow)

        # 6. BB
        scores["bb"] = self._score_bb(indicators)

        # 7. MTF
        scores["mtf"] = self._score_mtf(mtf)

        # 8. ADX (modula a força, não a direção)
        scores["adx"] = self._score_adx(trend_data)

        # 9. Market Structure (proximidade S/R)
        scores["structure"] = self._score_structure(analysis_data)

        # Score ponderado final
        weighted_score = sum(
            scores[k] * self.WEIGHTS[k] for k in scores
        )

        # ADX modula: se ADX fraco, reduzir score (mercado sem tendência)
        adx_val = trend_data.get("trend_strength_adx", trend_data.get("adx", 25))
        if adx_val < 15:
            weighted_score *= 0.5  # Mercado muito lateral → cortar score pela metade
        elif adx_val < 20:
            weighted_score *= 0.7

        # Regime de mercado modula
        if market_regime:
            base_regime = market_regime.get("base_regime", "SIDEWAYS")
            if base_regime == "SIDEWAYS":
                weighted_score *= 0.6  # Penalizar sinais em lateral

        # Decisão: BUY/SELL/NO_SIGNAL
        # Threshold mínimo para gerar sinal
        MIN_SCORE = 0.10  # |score| > 0.10 para ter sinal

        if abs(weighted_score) < MIN_SCORE:
            signal = "NO_SIGNAL"
            confidence = 0
        elif weighted_score > 0:
            signal = "BUY"
            # Mapear score (0.10 a ~0.60) para confidence (3 a 10)
            confidence = min(10, max(3, int(3 + (weighted_score - MIN_SCORE) * 14)))
        else:
            signal = "SELL"
            confidence = min(10, max(3, int(3 + (abs(weighted_score) - MIN_SCORE) * 14)))

        # Construir reasoning
        reasoning_parts = []
        for name, sc in sorted(scores.items(), key=lambda x: abs(x[1]), reverse=True):
            if abs(sc) > 0.1:
                direction = "BULL" if sc > 0 else "BEAR"
                reasoning_parts.append(f"{name.upper()}={direction}({sc:+.2f})")

        reasoning = (
            f"Local Generator: score_final={weighted_score:+.3f}, "
            f"componentes=[{', '.join(reasoning_parts[:5])}]"
        )

        logger.info(
            f"[LOCAL_GEN] {symbol}: {signal} conf={confidence}/10 "
            f"score={weighted_score:+.3f} | {reasoning}"
        )

        return {
            "signal": signal,
            "confidence": confidence,
            "reasoning": reasoning,
            "weighted_score": round(weighted_score, 4),
            "component_scores": {k: round(v, 4) for k, v in scores.items()},
        }

    # ================================================================
    # SCORING FUNCTIONS — cada uma retorna -1.0 a +1.0
    # ================================================================

    def _score_trend(self, trend_data: Dict) -> float:
        """Score da tendência (EMA alignment + primary trend)."""
        primary = trend_data.get("primary_trend", "neutral")
        trend_map = {
            "strong_bullish": 1.0,
            "bullish": 0.6,
            "neutral": 0.0,
            "bearish": -0.6,
            "strong_bearish": -1.0,
        }
        return trend_map.get(primary, 0.0)

    def _score_rsi(self, indicators: Dict) -> float:
        """
        Score do RSI.
        < 25 = forte bull (+1.0), 25-35 = bull (+0.5)
        > 75 = forte bear (-1.0), 65-75 = bear (-0.5)
        35-65 = neutro (0.0), com leve bias pela posição
        """
        rsi = indicators.get("rsi", {}).get("value", 50)
        if rsi < 25:
            return 1.0
        elif rsi < 30:
            return 0.7
        elif rsi < 35:
            return 0.4
        elif rsi > 75:
            return -1.0
        elif rsi > 70:
            return -0.7
        elif rsi > 65:
            return -0.4
        elif rsi < 45:
            return 0.1  # Leve bias bull
        elif rsi > 55:
            return -0.1  # Leve bias bear
        return 0.0

    def _score_macd(self, indicators: Dict) -> float:
        """Score do MACD histogram."""
        hist = indicators.get("macd", {}).get("histogram", 0)
        price = indicators.get("close", indicators.get("price", 1))
        if not price or price == 0:
            price = 1

        # Normalizar histograma pelo preço
        norm_hist = hist / abs(price) * 10000  # Em "pontos base"

        # Zona morta
        if abs(norm_hist) < 1:
            return 0.0

        # Clamp entre -1 e 1
        score = max(-1.0, min(1.0, norm_hist / 20))
        return score

    def _score_orderbook(self, volume_flow: Dict) -> float:
        """Score do orderbook."""
        bias = volume_flow.get("orderbook_bias", "neutral")
        imbalance = volume_flow.get("orderbook_imbalance", 0)

        bias_map = {
            "strong_buy_pressure": 1.0,
            "buy_pressure": 0.6,
            "neutral": 0.0,
            "sell_pressure": -0.6,
            "strong_sell_pressure": -1.0,
        }
        base = bias_map.get(bias, 0.0)

        # Modular com imbalance (0-1)
        if abs(imbalance) > 0:
            if imbalance > 0.5:
                base = max(base, 0.3)  # Reforça bull
            elif imbalance < -0.5:
                base = min(base, -0.3)  # Reforça bear

        return base

    def _score_cvd(self, volume_flow: Dict) -> float:
        """Score do CVD (Cumulative Volume Delta)."""
        direction = volume_flow.get("cvd_direction", "neutral")
        if direction == "positive":
            return 0.7
        elif direction == "negative":
            return -0.7
        return 0.0

    def _score_bb(self, indicators: Dict) -> float:
        """
        Score das Bollinger Bands.
        Position < 0.15 = near lower (bull), > 0.85 = near upper (bear)
        """
        pos = indicators.get("bollinger", {}).get("position", 0.5)
        if pos < 0.10:
            return 0.8
        elif pos < 0.20:
            return 0.5
        elif pos < 0.30:
            return 0.2
        elif pos > 0.90:
            return -0.8
        elif pos > 0.80:
            return -0.5
        elif pos > 0.70:
            return -0.2
        return 0.0

    def _score_mtf(self, mtf: Dict) -> float:
        """Score multi-timeframe."""
        bull = mtf.get("bullish_count", 0)
        bear = mtf.get("bearish_count", 0)
        total = max(bull + bear, 1)

        if bull >= 4:
            return 1.0
        elif bull >= 3:
            return 0.6
        elif bear >= 4:
            return -1.0
        elif bear >= 3:
            return -0.6
        # Diferença normalizada
        return (bull - bear) / total * 0.4

    def _score_adx(self, trend_data: Dict) -> float:
        """
        ADX não dá direção — mas modula a confiança na tendência.
        Retorna score baseado na tendência primária modulado pelo ADX.
        ADX forte + trend bullish = bull. ADX fraco = 0 (não vale operar).
        """
        adx = trend_data.get("trend_strength_adx", trend_data.get("adx", 25))
        primary = trend_data.get("primary_trend", "neutral")

        if adx < 20:
            return 0.0  # Sem tendência → não contribui

        # Direção da tendência
        trend_dir = 0.0
        if primary in ("bullish", "strong_bullish"):
            trend_dir = 1.0
        elif primary in ("bearish", "strong_bearish"):
            trend_dir = -1.0

        # Força do ADX normalizada (20-50 range)
        adx_strength = min(1.0, (adx - 20) / 30)
        return trend_dir * adx_strength

    def _score_structure(self, analysis_data: Dict) -> float:
        """
        Score baseado em proximidade de suporte/resistência.
        Perto de suporte = bull, perto de resistência = bear.
        """
        # Tentar extrair de support/resistance data
        trend = analysis_data.get("trend_analysis", {})
        dist_support = trend.get("distance_to_support_pct", None)
        dist_resistance = trend.get("distance_to_resistance_pct", None)

        if dist_support is not None and dist_resistance is not None:
            # Perto de suporte (< 1%) = bull
            if dist_support < 1.0:
                return 0.6
            elif dist_support < 2.0:
                return 0.3
            # Perto de resistência (< 1%) = bear
            if dist_resistance < 1.0:
                return -0.6
            elif dist_resistance < 2.0:
                return -0.3

        return 0.0
