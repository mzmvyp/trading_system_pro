"""
Invariant tests: ML features MUST mean the same thing in training and inference.

Root cause of the April/2026 liquidations was a semantic mismatch in 3 features
(risk_distance_pct, reward_distance_pct, risk_reward_ratio) between the old
training path (dataset_generator.py) and the inference path (agent.py /
simple_validator.py). See deep_analysis_report.txt:179-193.

These tests lock the alignment so regressions can't silently reintroduce the bug.
They read source as text to avoid importing heavy ML deps.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _read(rel_path: str) -> str:
    return (REPO_ROOT / rel_path).read_text(encoding="utf-8")


def test_agent_inference_uses_atr_candle_volume_semantics():
    """agent._calc_* must map to ATR%, candle_body_pct, volume_ratio (not SL/TP)."""
    src = _read("src/trading/agent.py")

    # Locate each helper and check its body uses the aligned primitive
    # (we search small substrings to stay resilient to formatting changes).
    def _body_of(fn_name: str) -> str:
        idx = src.find(f"def {fn_name}(")
        assert idx != -1, f"{fn_name} not found in agent.py"
        # Take ~400 chars after definition to cover small bodies
        return src[idx : idx + 400]

    risk_body = _body_of("_calc_risk_distance")
    assert "atr" in risk_body.lower(), (
        "risk_distance_pct inference must use ATR, not SL distance "
        "(would recreate the inverted-ML bug)."
    )
    assert "stop_loss" not in risk_body, (
        "risk_distance_pct inference cannot reference stop_loss directly."
    )

    reward_body = _body_of("_calc_reward_distance")
    assert "candle_body_pct" in reward_body, (
        "reward_distance_pct inference must use candle_body_pct, not TP distance."
    )
    assert "take_profit" not in reward_body, (
        "reward_distance_pct inference cannot reference take_profit directly."
    )

    rr_body = _body_of("_calc_risk_reward")
    assert "volume_ratio" in rr_body, (
        "risk_reward_ratio inference must use volume_ratio in current semantics."
    )


def test_simple_validator_inference_uses_aligned_semantics():
    """SimpleSignalValidator inference must use the same feature semantics."""
    src = _read("src/ml/simple_validator.py")
    assert "'reward_distance_pct': deepseek_signal.get('candle_body_pct'" in src, (
        "simple_validator must map reward_distance_pct ← candle_body_pct."
    )
    assert "'risk_reward_ratio': deepseek_signal.get('volume_ratio'" in src, (
        "simple_validator must map risk_reward_ratio ← volume_ratio."
    )


def test_dataset_generator_does_not_use_sl_tp_for_features():
    """The old dataset_generator must NOT compute features from SL/TP distances."""
    src = _read("src/ml/dataset_generator.py")

    bad_risk = "self.dataset['entry_price'] - self.dataset['stop_loss']"
    assert bad_risk not in src, (
        "dataset_generator is computing risk_distance_pct from SL distance — "
        "this was the root cause of the inverted ML (deep_analysis_report.txt:179)."
    )

    bad_reward = "self.dataset['take_profit_1'] - self.dataset['entry_price']"
    assert bad_reward not in src, (
        "dataset_generator is computing reward_distance_pct from TP distance — "
        "same inverted ML bug."
    )


def test_dataset_generator_uses_aligned_semantics():
    """The old dataset_generator, if used, must produce aligned features."""
    src = _read("src/ml/dataset_generator.py")
    # After the fix, ATR-based risk_distance_pct and candle_body / volume_ratio
    # flows must exist.
    assert "self.dataset['atr'].abs()" in src, (
        "dataset_generator must derive risk_distance_pct from ATR."
    )
    assert "self.dataset['candle_body_pct']" in src, (
        "dataset_generator must source reward_distance_pct from candle_body_pct."
    )
    assert "self.dataset['volume_ratio']" in src, (
        "dataset_generator must source risk_reward_ratio from volume_ratio."
    )


def test_online_learning_uses_aligned_semantics():
    """Online learning data-point constructor must use the same semantics."""
    src = _read("src/ml/online_learning.py")
    assert "risk_distance_pct = atr_pct" in src, (
        "online_learning must alias risk_distance_pct ← atr_pct."
    )
    assert "reward_distance_pct = candle_body_pct" in src, (
        "online_learning must alias reward_distance_pct ← candle_body_pct."
    )
    assert "risk_reward_ratio = volume_ratio" in src, (
        "online_learning must alias risk_reward_ratio ← volume_ratio."
    )


def test_feature_alignment_guard_passes_on_current_tree():
    """Runtime guard must agree with the static tests on the current tree."""
    from src.ml.feature_alignment_guard import check_feature_alignment

    ok, failures = check_feature_alignment()
    assert ok, f"Guard detectou desalinhamento: {failures}"


def test_train_from_signals_uses_aligned_semantics():
    """Active training path must match inference."""
    src = _read("src/ml/train_from_signals.py")
    # train_from_signals computes atr_pct, candle_body_pct, volume_ratio
    # and stores them as the 3 feature columns.
    assert "atr_pct = (atr / close * 100)" in src, (
        "train_from_signals must derive risk_distance_pct from ATR%."
    )
    assert "'reward_distance_pct': float(candle_body_pct)" in src, (
        "train_from_signals must map reward_distance_pct ← candle_body_pct."
    )
    assert "'risk_reward_ratio': float(volume_ratio)" in src, (
        "train_from_signals must map risk_reward_ratio ← volume_ratio."
    )
