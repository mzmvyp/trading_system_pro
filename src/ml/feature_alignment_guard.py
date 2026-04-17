"""
Runtime guard: blocks ML retrain if training/inference feature semantics diverge.

The April/2026 liquidations traced back to `risk_distance_pct`,
`reward_distance_pct` and `risk_reward_ratio` meaning different things in the
training path (distance-from-SL/TP) vs inference (ATR%, candle_body_pct,
volume_ratio). See deep_analysis_report.txt:179-193.

This module is called at the start of every retrain path
(`seed_from_evaluated_signals`, `manual_retrain`, drift-triggered, dashboard
buttons, CLI script). It inspects source files as text — same approach as
tests/test_ml_feature_alignment.py — so it has no heavy dependencies and
fails fast if someone regresses the fix.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


def check_feature_alignment() -> Tuple[bool, List[str]]:
    """Return (is_aligned, failure_reasons).

    `is_aligned=True` means both the training dataset generators and the
    inference path use ATR%, candle_body_pct and volume_ratio for the three
    features. Any regression (e.g. deriving them from SL/TP distance) makes
    this return False and populates `failure_reasons`.
    """
    failures: List[str] = []

    # 1. agent.py inference: ATR-based risk, candle_body for reward, volume_ratio for RR
    try:
        src = _read("src/trading/agent.py")
        idx = src.find("def _calc_risk_distance(")
        if idx == -1:
            failures.append("agent._calc_risk_distance não encontrado")
        else:
            body = src[idx : idx + 400]
            if "atr" not in body.lower():
                failures.append("agent._calc_risk_distance não usa ATR")
            if "stop_loss" in body:
                failures.append("agent._calc_risk_distance referencia stop_loss (data leakage)")

        idx = src.find("def _calc_reward_distance(")
        if idx == -1:
            failures.append("agent._calc_reward_distance não encontrado")
        else:
            body = src[idx : idx + 400]
            if "candle_body_pct" not in body:
                failures.append("agent._calc_reward_distance não usa candle_body_pct")
            if "take_profit" in body:
                failures.append("agent._calc_reward_distance referencia take_profit")

        idx = src.find("def _calc_risk_reward(")
        if idx == -1:
            failures.append("agent._calc_risk_reward não encontrado")
        else:
            body = src[idx : idx + 400]
            if "volume_ratio" not in body:
                failures.append("agent._calc_risk_reward não usa volume_ratio")
    except Exception as e:
        failures.append(f"erro lendo agent.py: {e}")

    # 2. dataset_generator: não pode usar SL/TP como base das features
    try:
        src = _read("src/ml/dataset_generator.py")
        bad_risk = "self.dataset['entry_price'] - self.dataset['stop_loss']"
        if bad_risk in src:
            failures.append("dataset_generator deriva risk_distance_pct de stop_loss")
        bad_reward = "self.dataset['take_profit_1'] - self.dataset['entry_price']"
        if bad_reward in src:
            failures.append("dataset_generator deriva reward_distance_pct de take_profit_1")
        if "self.dataset['atr'].abs()" not in src:
            failures.append("dataset_generator não deriva risk_distance_pct de ATR")
    except Exception as e:
        failures.append(f"erro lendo dataset_generator.py: {e}")

    # 3. online_learning: aliases explícitos
    try:
        src = _read("src/ml/online_learning.py")
        for needle in (
            "risk_distance_pct = atr_pct",
            "reward_distance_pct = candle_body_pct",
            "risk_reward_ratio = volume_ratio",
        ):
            if needle not in src:
                failures.append(f"online_learning não tem alias '{needle}'")
    except Exception as e:
        failures.append(f"erro lendo online_learning.py: {e}")

    # 4. train_from_signals: mesma semântica
    try:
        src = _read("src/ml/train_from_signals.py")
        if "atr_pct = (atr / close * 100)" not in src:
            failures.append("train_from_signals não deriva atr_pct da fórmula esperada")
        if "'reward_distance_pct': float(candle_body_pct)" not in src:
            failures.append("train_from_signals não mapeia reward_distance_pct ← candle_body_pct")
        if "'risk_reward_ratio': float(volume_ratio)" not in src:
            failures.append("train_from_signals não mapeia risk_reward_ratio ← volume_ratio")
    except Exception as e:
        failures.append(f"erro lendo train_from_signals.py: {e}")

    return (len(failures) == 0, failures)


def assert_features_aligned() -> None:
    """Raise RuntimeError if any retrain path is about to use misaligned features.

    Call at the start of every retrain entrypoint. This prevents the
    Apr/2026 inverted-ML bug from silently re-appearing after a refactor.
    """
    ok, failures = check_feature_alignment()
    if not ok:
        msg = (
            "[FEATURE ALIGNMENT GUARD] Retreino BLOQUEADO — features de treino e "
            "inferência divergem. Isto é a causa-raiz documentada no "
            "deep_analysis_report.txt:179-193 (correlação ML↔outcome r=-0.1257, "
            "p<0.001). Corrija antes de retreinar:\n  - "
            + "\n  - ".join(failures)
        )
        raise RuntimeError(msg)
