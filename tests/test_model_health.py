"""
Tests for the model health circuit breaker.

The thresholds were tightened on 17-Abr-2026 after observing the real
distribution of the last 100 signals (standalone_bilstm_last_signals_accuracy
output). The loose initial thresholds were missing the exact degeneracy they
were meant to catch — both ML (mean 0.39, ~15 clusters, stdev ≈ 0.05) and
LSTM (mean 0.46, range 0.36-0.54, stdev ≈ 0.035) sat one tick inside the
healthy band.

The regression tests below replay those distributions and assert the gate
fires, while a genuinely calibrated model (mean ≈ 0.5, stdev ≈ 0.15, many
distinct outputs) still passes.
"""
from __future__ import annotations

import random
from pathlib import Path


def test_warming_up_returns_healthy():
    """Less than MIN_SAMPLES samples → gate passes (no opinion)."""
    from src.ml import model_health

    model_health._reset_for_tests()
    for _ in range(5):
        allow, status = model_health.health_gate("ml", 0.436)
    assert allow, "should still allow voting while warming up"
    assert "warming_up" in status["reason"]


def test_near_constant_output_blocks_vote():
    """ML stuck at ~0.436 — the original April/2026 symptom — must block."""
    from src.ml import model_health

    model_health._reset_for_tests()
    for _ in range(60):
        allow, status = model_health.health_gate("ml", 0.436)
    assert not allow, "near-constant output must block the vote"
    assert (
        "near-constant" in status["reason"]
        or "distinct" in status["reason"]
        or "biased" in status["reason"]
    )


def test_ml_majority_class_regression_blocks_vote():
    """
    Regression for Apr-17: ML outputs clustered on ~8-15 values near 0.39
    (collapsed to majority-class predictor). Gate must fire on this.

    Replays the observed cluster set from the 100-signal audit.
    """
    from src.ml import model_health

    model_health._reset_for_tests()
    # Observed repeated values from the Apr-17 dump (weights approximate).
    clusters = [0.332, 0.371, 0.398, 0.400, 0.408, 0.409, 0.431, 0.432,
                0.442, 0.454, 0.503, 0.525, 0.560, 0.569]
    weights = [25, 10, 10, 2, 2, 2, 2, 30, 2, 3, 2, 2, 2, 2]
    samples = []
    for v, w in zip(clusters, weights):
        samples.extend([v] * w)
    random.seed(17)
    random.shuffle(samples)
    for p in samples[:100]:
        allow, status = model_health.health_gate("ml", p)
    assert not allow, f"Apr-17 ML distribution must block; status={status}"


def test_lstm_narrow_band_regression_blocks_vote():
    """
    Regression for Apr-17: LSTM outputs compressed in [0.36, 0.54],
    mean ≈ 0.46, stdev ≈ 0.035 — no informational range. Must block.
    """
    from src.ml import model_health

    model_health._reset_for_tests()
    random.seed(46)
    for _ in range(100):
        # Very narrow gaussian reproducing the observed spread.
        p = max(0.0, min(1.0, random.gauss(0.46, 0.035)))
        allow, status = model_health.health_gate("lstm", p)
    assert not allow, f"LSTM narrow-band distribution must block; status={status}"


def test_severely_biased_mean_blocks_vote():
    """Strong upward bias (mean > 0.58) must trip the gate."""
    from src.ml import model_health

    model_health._reset_for_tests()
    random.seed(7)
    for _ in range(80):
        p = max(0.0, min(1.0, random.gauss(0.72, 0.05)))
        allow, status = model_health.health_gate("lstm", p)
    assert not allow, f"mean 0.72 must block; status={status}"
    assert "biased UP" in status["reason"]


def test_healthy_distribution_allows_vote():
    """A real, calibrated model (mean ≈ 0.5, stdev ≈ 0.15) passes."""
    from src.ml import model_health

    model_health._reset_for_tests()
    random.seed(1)
    for _ in range(150):
        p = max(0.0, min(1.0, random.gauss(0.5, 0.15)))
        allow, status = model_health.health_gate("ml", p)
    assert allow, f"well-distributed output must pass; status={status}"
    assert status["stdev"] >= 0.10
    assert 0.42 <= status["mean"] <= 0.58


def test_invalid_input_does_not_crash():
    """None / NaN / out-of-range probabilities must be ignored, not raise."""
    from src.ml import model_health

    model_health._reset_for_tests()
    model_health.record_probability("ml", None)
    model_health.record_probability("ml", float("nan"))
    model_health.record_probability("ml", 1.5)
    model_health.record_probability("ml", -0.2)
    model_health.record_probability("ml", "not a number")  # type: ignore[arg-type]
    status = model_health.evaluate("ml")
    assert status["n"] == 0


def test_agent_imports_health_gate():
    """Agent voter paths must import the health gate."""
    repo = Path(__file__).resolve().parent.parent
    src = (repo / "src/trading/agent.py").read_text(encoding="utf-8")
    assert "from src.ml.model_health import health_gate" in src, (
        "agent.py deve importar health_gate para proteger voto ML/LSTM"
    )
    # Both ML and LSTM paths should use it — appears at least twice.
    assert src.count("health_gate(") >= 2, (
        "health_gate deve ser chamado em ambos os caminhos (ML e LSTM)"
    )
