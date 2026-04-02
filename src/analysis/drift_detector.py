"""
Drift Detector — Detecta mudanças na distribuição das features e performance do ML
==================================================================================

Monitora 3 tipos de drift:
1. Feature Drift — mudança na distribuição dos indicadores (PSI + KS-test)
2. Prediction Drift — viés nas predições do modelo (sempre bullish/bearish)
3. Performance Drift — degradação do accuracy ao longo do tempo

Quando drift severo é detectado:
- Recomenda retreinar o ML
- Pode pausar votos do ML até retreino
- Registra histórico para análise

Baseado no drift_detector.py do bot_trade_20260115.
"""

import json
import os
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.core.logger import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

DRIFT_DIR = Path("data/drift")
BASELINE_FILE = DRIFT_DIR / "drift_baseline.json"
HISTORY_FILE = DRIFT_DIR / "drift_history.json"


class DriftSeverity(Enum):
    NONE = "NONE"           # PSI < 0.1
    LOW = "LOW"             # 0.1 <= PSI < 0.2
    MODERATE = "MODERATE"   # 0.2 <= PSI < 0.25
    HIGH = "HIGH"           # PSI >= 0.25


@dataclass
class FeatureDriftResult:
    feature_name: str
    psi_score: float
    ks_statistic: float
    ks_pvalue: float
    severity: str
    baseline_mean: float
    current_mean: float
    baseline_std: float
    current_std: float
    drift_detected: bool


@dataclass
class DriftReport:
    timestamp: str
    overall_drift_detected: bool
    overall_severity: str
    feature_drift_count: int
    features_with_drift: List[str]
    feature_results: List[Dict]
    prediction_drift: Dict
    performance_drift: Dict
    recommendations: List[str]


# Features monitoradas (mesmas que alimentam ML/LSTM)
MONITORED_FEATURES = [
    "rsi", "macd_histogram", "adx", "atr", "bb_position",
    "confidence", "risk_distance_pct", "reward_distance_pct",
    "risk_reward_ratio",
]


class DriftDetector:
    """
    Detecta drift de features, predições e performance do modelo ML.
    """

    def __init__(self):
        DRIFT_DIR.mkdir(parents=True, exist_ok=True)
        self.baseline: Optional[Dict] = self._load_json(BASELINE_FILE)
        self.drift_history: List[Dict] = self._load_json(HISTORY_FILE) or []
        self.performance_buffer: List[Dict] = []
        self.performance_buffer_max = 200

    # ------------------------------------------------------------------
    # Persistência
    # ------------------------------------------------------------------
    @staticmethod
    def _load_json(path: Path) -> Any:
        if path.exists():
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    @staticmethod
    def _save_json(path: Path, data: Any):
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # PSI (Population Stability Index)
    # ------------------------------------------------------------------
    @staticmethod
    def calculate_psi(baseline: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        baseline = baseline[~np.isnan(baseline) & ~np.isinf(baseline)]
        current = current[~np.isnan(current) & ~np.isinf(current)]
        if len(baseline) < 10 or len(current) < 10:
            return 0.0
        try:
            _, bin_edges = np.histogram(baseline, bins=bins)
            b_counts, _ = np.histogram(baseline, bins=bin_edges)
            c_counts, _ = np.histogram(current, bins=bin_edges)
            b_pct = (b_counts + 0.0001) / len(baseline)
            c_pct = (c_counts + 0.0001) / len(current)
            return float(np.sum((c_pct - b_pct) * np.log(c_pct / b_pct)))
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # KS Test
    # ------------------------------------------------------------------
    @staticmethod
    def calculate_ks_test(baseline: np.ndarray, current: np.ndarray) -> Tuple[float, float]:
        baseline = baseline[~np.isnan(baseline) & ~np.isinf(baseline)]
        current = current[~np.isnan(current) & ~np.isinf(current)]
        if len(baseline) < 10 or len(current) < 10:
            return 0.0, 1.0
        try:
            from scipy import stats
            stat, pval = stats.ks_2samp(baseline, current)
            return float(stat), float(pval)
        except Exception:
            return 0.0, 1.0

    # ------------------------------------------------------------------
    # Severidade
    # ------------------------------------------------------------------
    @staticmethod
    def get_severity(psi: float) -> DriftSeverity:
        if psi < 0.1:
            return DriftSeverity.NONE
        elif psi < 0.2:
            return DriftSeverity.LOW
        elif psi < 0.25:
            return DriftSeverity.MODERATE
        return DriftSeverity.HIGH

    # ------------------------------------------------------------------
    # Baseline
    # ------------------------------------------------------------------
    def create_baseline_from_signals(self, signals: List[Dict]):
        """Cria baseline a partir de sinais históricos salvos."""
        if len(signals) < 20:
            logger.warning("[DRIFT] Sinais insuficientes para baseline")
            return

        import pandas as pd
        df = pd.DataFrame(signals)

        baseline = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "n_samples": len(df),
            "features": {},
        }

        for feat in MONITORED_FEATURES:
            if feat not in df.columns:
                continue
            values = df[feat].dropna().values.astype(float)
            if len(values) < 10:
                continue
            baseline["features"][feat] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "p25": float(np.percentile(values, 25)),
                "p50": float(np.percentile(values, 50)),
                "p75": float(np.percentile(values, 75)),
            }

        self.baseline = baseline
        self._save_json(BASELINE_FILE, baseline)
        logger.info(
            f"[DRIFT] Baseline criado: {len(df)} sinais, "
            f"{len(baseline['features'])} features"
        )

    # ------------------------------------------------------------------
    # Feature Drift
    # ------------------------------------------------------------------
    def analyze_feature_drift(self, current_signals: List[Dict]) -> List[FeatureDriftResult]:
        if not self.baseline:
            logger.warning("[DRIFT] Sem baseline — impossível detectar drift")
            return []

        import pandas as pd
        df = pd.DataFrame(current_signals)
        results = []

        for feat in MONITORED_FEATURES:
            if feat not in df.columns or feat not in self.baseline.get("features", {}):
                continue

            current_vals = df[feat].dropna().values.astype(float)
            if len(current_vals) < 10:
                continue

            bl = self.baseline["features"][feat]
            bl_mean, bl_std = bl["mean"], max(bl["std"], 0.001)

            # Simular distribuição baseline
            np.random.seed(42)
            baseline_vals = np.random.normal(bl_mean, bl_std, 1000)
            baseline_vals = np.clip(baseline_vals, bl["min"], bl["max"])

            psi = self.calculate_psi(baseline_vals, current_vals)
            ks_stat, ks_pval = self.calculate_ks_test(baseline_vals, current_vals)
            severity = self.get_severity(psi)

            results.append(FeatureDriftResult(
                feature_name=feat,
                psi_score=psi,
                ks_statistic=ks_stat,
                ks_pvalue=ks_pval,
                severity=severity.value,
                baseline_mean=bl_mean,
                current_mean=float(np.mean(current_vals)),
                baseline_std=bl_std,
                current_std=float(np.std(current_vals)),
                drift_detected=severity in (DriftSeverity.MODERATE, DriftSeverity.HIGH),
            ))

        return results

    # ------------------------------------------------------------------
    # Prediction Drift
    # ------------------------------------------------------------------
    @staticmethod
    def analyze_prediction_drift(predictions: List[float]) -> Dict:
        if len(predictions) < 10:
            return {"drift_detected": False, "severity": "NONE", "message": "Dados insuficientes"}

        preds = np.array(predictions)
        mean_pred = float(np.mean(preds))
        std_pred = float(np.std(preds))

        bias_score = abs(mean_pred - 0.5) + max(0, 0.2 - std_pred)

        if bias_score < 0.1:
            severity, drift = "NONE", False
        elif bias_score < 0.2:
            severity, drift = "LOW", False
        elif bias_score < 0.3:
            severity, drift = "MODERATE", True
        else:
            severity, drift = "HIGH", True

        if mean_pred > 0.6:
            bias_dir = "BULLISH"
        elif mean_pred < 0.4:
            bias_dir = "BEARISH"
        else:
            bias_dir = "NEUTRAL"

        return {
            "drift_detected": drift,
            "mean_prediction": mean_pred,
            "std_prediction": std_pred,
            "bias_score": float(bias_score),
            "severity": severity,
            "bias_direction": bias_dir,
            "positive_ratio": float(np.mean(preds > 0.5)),
            "n_predictions": len(predictions),
        }

    # ------------------------------------------------------------------
    # Performance Drift
    # ------------------------------------------------------------------
    def add_performance_sample(self, actual: int, predicted: int, probability: float):
        self.performance_buffer.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "actual": actual,
            "predicted": predicted,
            "probability": probability,
            "correct": actual == predicted,
        })
        if len(self.performance_buffer) > self.performance_buffer_max:
            self.performance_buffer = self.performance_buffer[-self.performance_buffer_max:]

    def analyze_performance_drift(self, window_size: int = 50) -> Dict:
        if len(self.performance_buffer) < window_size:
            return {
                "drift_detected": False,
                "severity": "NONE",
                "message": f"Dados insuficientes ({len(self.performance_buffer)}/{window_size})",
            }

        recent = self.performance_buffer[-window_size:]
        older = (
            self.performance_buffer[:-window_size]
            if len(self.performance_buffer) > window_size * 2
            else recent
        )

        recent_acc = np.mean([s["correct"] for s in recent])
        older_acc = np.mean([s["correct"] for s in older])
        change = recent_acc - older_acc

        if change < -0.10:
            severity, drift = "HIGH", True
        elif change < -0.05:
            severity, drift = "MODERATE", True
        elif change < -0.03:
            severity, drift = "LOW", True
        else:
            severity, drift = "NONE", False

        return {
            "drift_detected": drift,
            "severity": severity,
            "current_accuracy": float(recent_acc),
            "baseline_accuracy": float(older_acc),
            "accuracy_change_pct": float(change * 100),
            "n_recent": len(recent),
            "n_older": len(older),
        }

    # ------------------------------------------------------------------
    # Relatório completo
    # ------------------------------------------------------------------
    def generate_report(
        self,
        current_signals: List[Dict],
        predictions: Optional[List[float]] = None,
    ) -> DriftReport:
        feature_results = self.analyze_feature_drift(current_signals)
        features_with_drift = [r.feature_name for r in feature_results if r.drift_detected]

        prediction_drift = self.analyze_prediction_drift(predictions or [])
        performance_drift = self.analyze_performance_drift()

        # Severidade geral
        severities = [r.severity for r in feature_results if r.drift_detected]
        pred_sev = prediction_drift.get("severity", "NONE")
        perf_sev = performance_drift.get("severity", "NONE")

        if "HIGH" in severities or pred_sev == "HIGH" or perf_sev == "HIGH":
            overall = "HIGH"
        elif "MODERATE" in severities or pred_sev == "MODERATE" or perf_sev == "MODERATE":
            overall = "MODERATE"
        elif features_with_drift or prediction_drift.get("drift_detected"):
            overall = "LOW"
        else:
            overall = "NONE"

        recommendations = self._recommendations(
            feature_results, prediction_drift, performance_drift, overall
        )

        report = DriftReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            overall_drift_detected=overall != "NONE",
            overall_severity=overall,
            feature_drift_count=len(features_with_drift),
            features_with_drift=features_with_drift,
            feature_results=[asdict(r) for r in feature_results],
            prediction_drift=prediction_drift,
            performance_drift=performance_drift,
            recommendations=recommendations,
        )

        self.drift_history.append(asdict(report))
        if len(self.drift_history) > 100:
            self.drift_history = self.drift_history[-100:]
        self._save_json(HISTORY_FILE, self.drift_history)

        level = logger.warning if overall in ("HIGH", "MODERATE") else logger.info
        level(
            f"[DRIFT] Severidade={overall}, "
            f"features c/ drift={len(features_with_drift)}, "
            f"pred_drift={prediction_drift.get('drift_detected', False)}, "
            f"perf_drift={performance_drift.get('drift_detected', False)}"
        )

        return report

    @staticmethod
    def _recommendations(feat_results, pred_drift, perf_drift, overall) -> List[str]:
        recs = []
        if overall == "NONE":
            recs.append("Modelo estável — nenhuma ação necessária")
            return recs

        high = [r.feature_name for r in feat_results if r.severity == "HIGH"]
        if high:
            recs.append(
                f"URGENTE: Features com drift severo: {', '.join(high)}. "
                "Retreinar ML imediatamente."
            )

        moderate = [r.feature_name for r in feat_results if r.severity == "MODERATE"]
        if moderate:
            recs.append(
                f"Features com drift moderado: {', '.join(moderate)}. "
                "Planejar retreino em breve."
            )

        if pred_drift.get("drift_detected"):
            bias = pred_drift.get("bias_direction", "")
            recs.append(
                f"Modelo com viés {bias} "
                f"(média predições={pred_drift.get('mean_prediction', 0):.2f}). "
                "Balancear dataset."
            )

        if perf_drift.get("drift_detected"):
            change = perf_drift.get("accuracy_change_pct", 0)
            recs.append(f"Performance degradada: {change:.1f}% de queda. Retreino recomendado.")

        if overall == "HIGH":
            recs.append("CRÍTICO: Considerar pausar votos do ML até retreino.")

        return recs

    def should_pause_ml(self) -> bool:
        """Retorna True se drift é severo o suficiente para pausar ML."""
        if not self.drift_history:
            return False
        last = self.drift_history[-1]
        return last.get("overall_severity") == "HIGH"

    def get_last_report(self) -> Optional[Dict]:
        return self.drift_history[-1] if self.drift_history else None


# Instância global
_detector: Optional[DriftDetector] = None


def get_drift_detector() -> DriftDetector:
    global _detector
    if _detector is None:
        _detector = DriftDetector()
    return _detector
