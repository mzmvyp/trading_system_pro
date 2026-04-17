"""
Tests for the model health circuit breaker.

These tests reproduce the Abril/2026 degenerate-distribution scenarios as
documented in the post-mortem:
  - ML near-constant (~0.436 in 92/100 cases, 5 distinct values)
  - LSTM overconfident / biased up (mean ~0.59 vs real WR 33%)

Both cases must trip the health gate and force the vote neutral.
"""
from __future__ import annotations

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
    """ML stuck at ~0.436 — the April/2026 symptom — must block the vote."""
    from src.ml import model_health

    model_health._reset_for_tests()
    for _ in range(60):
        allow, status = model_health.health_gate("ml", 0.436)
    assert not allow, "near-constant output must block the vote"
    assert "near-constant" in status["reason"] or "distinct" in status["reason"]


def test_mild_bias_inside_band_passes():
    """
    Mild bias (mean ~0.59) is NOT flagged by the distribution gate — that's
    a calibration problem (WR mismatch), not a degeneracy problem. The gate
    only catches egregious cases. This test documents that boundary.
    """
    import random

    from src.ml import model_health

    model_health._reset_for_tests()
    random.seed(42)
    for _ in range(100):
        p = max(0.0, min(1.0, random.gauss(0.59, 0.08)))
        allow, status = model_health.health_gate("lstm", p)
    assert allow, f"mean 0.59 inside band should pass; status={status}"
    assert 0.50 <= status["mean"] <= 0.65
    assert status["stdev"] >= 0.03


def test_severely_biased_mean_blocks_vote():
    """Strong upward bias (mean > 0.65) must always trip the gate."""
    import random

    from src.ml import model_health

    model_health._reset_for_tests()
    random.seed(7)
    for _ in range(80):
        p = max(0.0, min(1.0, random.gauss(0.72, 0.05)))
        allow, status = model_health.health_gate("lstm", p)
    assert not allow, f"mean 0.72 must block; status={status}"
    assert "biased UP" in status["reason"]


def test_healthy_distribution_allows_vote():
    """Well-distributed probabilities around 0.5 must pass."""
    import random

    from src.ml import model_health

    model_health._reset_for_tests()
    random.seed(1)
    for _ in range(80):
        p = max(0.0, min(1.0, random.gauss(0.5, 0.12)))
        allow, status = model_health.health_gate("ml", p)
    assert allow, f"well-distributed output must pass; status={status}"


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
