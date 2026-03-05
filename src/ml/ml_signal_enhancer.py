"""
ML Signal Enhancer - Adds ML prediction score to trading signals.
Source: sinais
Combines technical confidence with ML prediction using configurable weight.
"""

import logging
from typing import Dict, Optional

from src.core.logger import get_logger

logger = get_logger(__name__)

try:
    from src.ml.xgboost_predictor import OptimizedXGBoostPredictor, XGBOOST_AVAILABLE
except ImportError:
    XGBOOST_AVAILABLE = False


class MLSignalEnhancer:
    """
    Enhances trading signals with ML predictions.
    Blends technical confidence with ML confidence using configurable weight.
    """

    def __init__(self, enabled: bool = True, ml_weight: float = 0.25):
        self.enabled = enabled and XGBOOST_AVAILABLE
        self.ml_weight = ml_weight
        self.predictor: Optional[OptimizedXGBoostPredictor] = None

        if self.enabled:
            try:
                self.predictor = OptimizedXGBoostPredictor()
                if self.predictor.model is None:
                    self.enabled = False
                    logger.info("ML Signal Enhancer: no trained model found")
            except Exception as e:
                self.enabled = False
                logger.warning(f"ML Signal Enhancer disabled: {e}")

    def enhance_signal(
        self,
        signal_type: str,
        technical_confidence: float,
        features_df=None,
    ) -> Dict:
        """
        Enhance a signal with ML prediction.

        Args:
            signal_type: "BUY" or "SELL"
            technical_confidence: Original confidence from technical analysis (0-1)
            features_df: Prepared features DataFrame for ML prediction

        Returns:
            Dict with final_confidence and ML details
        """
        if not self.enabled or features_df is None:
            return {
                "final_confidence": technical_confidence,
                "ml_enabled": False,
            }

        try:
            ml_result = self.predictor.predict(features_df)
            if ml_result is None:
                return {
                    "final_confidence": technical_confidence,
                    "ml_enabled": True,
                    "ml_error": "prediction_failed",
                }

            # Check if ML agrees with signal direction
            ml_bullish = ml_result["prediction"] == "BULLISH"
            signal_bullish = "BUY" in signal_type

            agreement = ml_bullish == signal_bullish

            if agreement:
                # ML agrees -> boost confidence
                ml_contribution = ml_result["confidence"] * self.ml_weight
                final_confidence = technical_confidence * (1 - self.ml_weight) + ml_contribution
            else:
                # ML disagrees -> reduce confidence
                final_confidence = (
                    technical_confidence * (1 - self.ml_weight) +
                    (1 - ml_result["confidence"]) * self.ml_weight
                )

            final_confidence = max(0.0, min(1.0, final_confidence))

            return {
                "final_confidence": final_confidence,
                "ml_enabled": True,
                "ml_prediction": ml_result["prediction"],
                "ml_confidence": ml_result["confidence"],
                "ml_agrees": agreement,
                "technical_confidence": technical_confidence,
                "confidence_delta": final_confidence - technical_confidence,
            }

        except Exception as e:
            logger.error(f"ML enhancement error: {e}")
            return {
                "final_confidence": technical_confidence,
                "ml_enabled": True,
                "ml_error": str(e),
            }
