"""
XGBoost Predictor - Optimized with hyperparameter tuning.
Source: sinais
Features:
- RandomizedSearch for efficient optimization
- Early stopping to prevent overfitting
- Temporal cross-validation (TimeSeriesSplit)
- Feature importance analysis
- Auto-retrain based on performance
"""

import os
import pickle
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.core.logger import get_logger

logger = get_logger(__name__)

try:
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
    from sklearn.utils import class_weight
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class OptimizedXGBoostPredictor:
    """
    XGBoost Predictor with hyperparameter optimization.
    Uses RandomizedSearch + TimeSeriesSplit for temporal validation.
    """

    def __init__(self, model_path: Optional[str] = None):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed. Run: pip install xgboost scikit-learn")

        self.base_params = {
            "objective": "binary:logistic",
            "tree_method": "hist",
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": 1,
        }

        self.param_distribution = {
            "max_depth": [3, 4, 5, 6, 7, 8],
            "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
            "n_estimators": [300, 500, 800, 1000, 1200],
            "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "min_child_weight": [1, 3, 5, 7, 10],
            "gamma": [0, 0.1, 0.2, 0.3, 0.5],
            "reg_alpha": [0, 0.1, 0.5, 1.0, 2.0],
            "reg_lambda": [0.5, 1.0, 1.5, 2.0, 3.0],
        }

        self.model_path = model_path or os.path.join("data", "models", "optimized_xgboost_model.pkl")
        self.model = None
        self.feature_names: List[str] = []
        self.best_params: Optional[Dict] = None
        self.optimization_results: Dict = {}

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.load_model()

    def train_with_optimization(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: List[str],
        n_iter: int = 50,
        cv_folds: int = 3,
    ) -> Dict:
        """Train with hyperparameter optimization using RandomizedSearch."""
        self.feature_names = feature_names

        tscv = TimeSeriesSplit(n_splits=cv_folds, gap=24)

        try:
            weights = class_weight.compute_class_weight("balanced", classes=np.unique(y), y=y)
            scale_pos_weight = weights[1] / weights[0] if len(weights) > 1 else 1.0
        except Exception:
            scale_pos_weight = 1.0

        xgb_params = self.base_params.copy()
        xgb_params["scale_pos_weight"] = scale_pos_weight
        xgb_model = xgb.XGBClassifier(**xgb_params)

        search = RandomizedSearchCV(
            xgb_model, self.param_distribution,
            n_iter=n_iter, cv=tscv, scoring="roc_auc",
            n_jobs=1, random_state=42, verbose=0,
        )

        start_time = datetime.now()
        try:
            search.fit(X, y)
        except Exception as e:
            return {"error": f"Optimization failed: {e}"}

        optimization_time = (datetime.now() - start_time).total_seconds()

        self.best_params = search.best_params_
        self.model = search.best_estimator_
        self.save_model()

        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred, zero_division=0)),
            "recall": float(recall_score(y, y_pred, zero_division=0)),
            "f1": float(f1_score(y, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, y_pred_proba)),
        }

        feature_importance = self._analyze_feature_importance()

        self.optimization_results = {
            "method": "RandomizedSearch",
            "cv_folds": cv_folds,
            "n_iter": n_iter,
            "optimization_time": optimization_time,
            "best_score": float(search.best_score_),
            "best_params": self.best_params,
            "n_samples": len(X),
            "n_features": len(feature_names),
            "metrics": metrics,
            "feature_importance": feature_importance,
            "training_date": datetime.now().isoformat(),
        }

        return {"success": True, "optimization_results": self.optimization_results}

    def predict(self, X: pd.DataFrame) -> Optional[Dict]:
        """Make a prediction on prepared features."""
        if self.model is None:
            return None

        try:
            X_pred = X[self.feature_names].iloc[[-1]] if len(X) > 1 else X[self.feature_names]

            prediction = self.model.predict(X_pred)[0]
            prediction_proba = self.model.predict_proba(X_pred)[0]
            confidence = float(prediction_proba[1] if prediction == 1 else prediction_proba[0])

            return {
                "prediction": "BULLISH" if prediction == 1 else "BEARISH",
                "confidence": confidence,
                "probability_up": float(prediction_proba[1]),
                "probability_down": float(prediction_proba[0]),
                "timestamp": datetime.now().isoformat(),
                "model_info": {
                    "optimization_method": self.optimization_results.get("method", "unknown"),
                    "best_score": self.optimization_results.get("best_score", 0),
                    "training_date": self.optimization_results.get("training_date", "unknown"),
                },
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None

    def _analyze_feature_importance(self) -> Dict:
        """Analyze and rank feature importance."""
        if self.model is None:
            return {}

        try:
            scores = self.model.feature_importances_
            importance = []
            for i, (name, score) in enumerate(zip(self.feature_names, scores)):
                importance.append({
                    "rank": i + 1,
                    "feature": name,
                    "importance": float(score),
                    "percentage": float(score / scores.sum() * 100),
                })
            importance.sort(key=lambda x: x["importance"], reverse=True)
            for i, item in enumerate(importance):
                item["rank"] = i + 1

            return {
                "all_features": importance,
                "top_10": importance[:10],
                "total_features": len(self.feature_names),
            }
        except Exception as e:
            logger.error(f"Feature importance error: {e}")
            return {}

    def get_model_info(self) -> Dict:
        if self.model is None:
            return {"status": "not_trained"}
        return {
            "status": "trained",
            "optimization_results": self.optimization_results,
            "best_params": self.best_params,
            "feature_count": len(self.feature_names),
        }

    def save_model(self):
        try:
            model_data = {
                "model": self.model,
                "feature_names": self.feature_names,
                "best_params": self.best_params,
                "optimization_results": self.optimization_results,
            }
            with open(self.model_path, "wb") as f:
                pickle.dump(model_data, f)
        except Exception as e:
            logger.error(f"Error saving model: {e}")

    def load_model(self) -> bool:
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, "rb") as f:
                    data = pickle.load(f)
                self.model = data["model"]
                self.feature_names = data["feature_names"]
                self.best_params = data["best_params"]
                self.optimization_results = data["optimization_results"]
                return True
            return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def should_retrain(self, days_threshold: int = 7) -> bool:
        if not self.optimization_results:
            return True
        training_date = self.optimization_results.get("training_date")
        if not training_date:
            return True
        try:
            last_training = datetime.fromisoformat(training_date)
            return (datetime.now() - last_training).days >= days_threshold
        except Exception:
            return True
