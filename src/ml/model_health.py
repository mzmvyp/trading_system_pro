"""
Model Health Circuit Breaker
============================

Trading decisions were being driven by ML/LSTM outputs that were effectively
constant or severely biased:

  - ML (sklearn): probabilities clustered on ~0.436 in ~92/100 cases, only 5
    distinct values — the model is effectively outputting a constant. The 66%
    "accuracy" comes from aligning with the majority class (SL_HIT), not from
    real predictive power.
  - LSTM (Bi-LSTM): mean probability 0.59 vs actual win rate 33% — overconfident
    and biased toward FOR, converting into systematic false-positive votes.

Both failure modes are silent: the model keeps returning numbers, voter logs
keep accumulating, and nothing in the training pipeline flags the degeneracy.

This module tracks the last N probabilities emitted by each model and exposes a
single function `health_gate(...)` that returns whether the model's recent
output is trustworthy enough to vote. If not, the caller MUST force the vote to
0 (neutral). We do NOT try to rescue the vote — the correct behaviour is to
disable it until retraining fixes the distribution.

The ring buffer is in-memory and persists only for the life of the agent
process; that is intentional — a restarted process should get a fresh reading
(the previous process may have warmed up with inputs from a prior regime).
"""
from __future__ import annotations

import json
import os
import statistics
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Deque, Dict, Optional, Tuple


# Minimum samples before we judge the distribution. Below this we return
# "no opinion" and let the vote proceed normally.
_MIN_SAMPLES = 30

# Max samples kept per model (the rolling window).
_WINDOW = 200

# Degeneracy thresholds (tuned from the April/2026 incident data).
#   * Standard deviation below _FLAT_STD → model is near-constant.
#   * Mean outside [_BIAS_LOW, _BIAS_HIGH] → model has collapsed to one class.
#   * Distinct-value count below _MIN_DISTINCT → discretized output.
_FLAT_STD = 0.03
_BIAS_LOW = 0.35
_BIAS_HIGH = 0.65
_MIN_DISTINCT = 3


_buffers: Dict[str, Deque[float]] = {}
_lock = threading.Lock()

_STATE_PATH = os.path.join("ml_models", "model_health_state.json")


def record_probability(model: str, probability: float) -> None:
    """Append a probability to the rolling window for `model`."""
    if probability is None:
        return
    try:
        p = float(probability)
    except (TypeError, ValueError):
        return
    if not (0.0 <= p <= 1.0):
        return
    with _lock:
        buf = _buffers.get(model)
        if buf is None:
            buf = deque(maxlen=_WINDOW)
            _buffers[model] = buf
        buf.append(p)


def _stats(buf: Deque[float]) -> Tuple[float, float, int]:
    values = list(buf)
    mean = statistics.fmean(values)
    stdev = statistics.pstdev(values) if len(values) > 1 else 0.0
    distinct = len({round(v, 3) for v in values})
    return mean, stdev, distinct


def evaluate(model: str) -> Dict:
    """Return the current health status of `model` without changing state."""
    with _lock:
        buf = _buffers.get(model)
        n = len(buf) if buf is not None else 0
        if buf is None or n < _MIN_SAMPLES:
            return {
                "healthy": True,
                "reason": f"warming_up ({n}/{_MIN_SAMPLES})",
                "n": n,
                "mean": None,
                "stdev": None,
                "distinct": None,
            }
        mean, stdev, distinct = _stats(buf)

    problems = []
    if stdev < _FLAT_STD:
        problems.append(f"near-constant output (stdev={stdev:.3f} < {_FLAT_STD})")
    if distinct < _MIN_DISTINCT:
        problems.append(f"only {distinct} distinct values in last {n}")
    if mean < _BIAS_LOW:
        problems.append(f"mean={mean:.2f} biased DOWN (< {_BIAS_LOW})")
    elif mean > _BIAS_HIGH:
        problems.append(f"mean={mean:.2f} biased UP (> {_BIAS_HIGH})")

    healthy = not problems
    return {
        "healthy": healthy,
        "reason": "ok" if healthy else "; ".join(problems),
        "n": n,
        "mean": round(mean, 4),
        "stdev": round(stdev, 4),
        "distinct": distinct,
    }


def health_gate(model: str, probability: float) -> Tuple[bool, Dict]:
    """Record the probability and return `(allow_vote, status_dict)`.

    `allow_vote=False` means the caller MUST force the vote to 0 (neutral) and
    log the reason. The probability is still returned untouched for display.
    """
    record_probability(model, probability)
    status = evaluate(model)
    return status["healthy"], status


def snapshot() -> Dict[str, Dict]:
    """Return current stats for all tracked models (debug / dashboards)."""
    with _lock:
        models = list(_buffers.keys())
    return {m: evaluate(m) for m in models}


def persist_snapshot(path: Optional[str] = None) -> None:
    """Write the current snapshot to disk so dashboards can read it."""
    target = path or _STATE_PATH
    os.makedirs(os.path.dirname(target), exist_ok=True)
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "thresholds": {
            "min_samples": _MIN_SAMPLES,
            "window": _WINDOW,
            "flat_std": _FLAT_STD,
            "bias_low": _BIAS_LOW,
            "bias_high": _BIAS_HIGH,
            "min_distinct": _MIN_DISTINCT,
        },
        "models": snapshot(),
    }
    with open(target, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def _reset_for_tests() -> None:
    """Test helper — wipe all buffers."""
    with _lock:
        _buffers.clear()
