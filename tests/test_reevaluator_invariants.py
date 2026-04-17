"""
Invariant tests for the reevaluation layer.

Context: three reevaluation paths were running in parallel
(signal_reevaluator, position_reevaluator, position_monitor). They shared
state loosely and each had its own idea of "when to act":

- signal_reevaluator had an absolute breakeven/trailing (1.5% / 2.5%)
- position_reevaluator has a proportional breakeven/trailing (70% / 90% TP1)
- min_hours_open was hardcoded as 1.0 in position_reevaluator and read from
  settings.reevaluation_min_time_open_hours everywhere else

These tests lock in the consolidated behaviour so the duplication can't
silently come back.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


def test_signal_reevaluator_no_longer_moves_stops_on_absolute_pnl():
    """Absolute breakeven/trailing were moved out of signal_reevaluator."""
    src = _read("src/trading/signal_reevaluator.py")
    # The old code called _execute_adjust_stop from inside the auto-protect
    # block. That block was replaced with a deprecation comment. The action
    # strings 'AUTO_BREAKEVEN' / 'AUTO_TRAILING_STOP' must no longer be
    # emitted from this file.
    assert '"AUTO_BREAKEVEN"' not in src, (
        "signal_reevaluator não deve mais emitir AUTO_BREAKEVEN — "
        "position_reevaluator cuida disso de forma proporcional"
    )
    assert '"AUTO_TRAILING_STOP"' not in src, (
        "signal_reevaluator não deve mais emitir AUTO_TRAILING_STOP"
    )


def test_signal_reevaluator_has_price_change_gate():
    """should_reevaluate must skip DeepSeek when price barely moved."""
    src = _read("src/trading/signal_reevaluator.py")
    assert "reevaluation_min_price_change_pct" in src, (
        "signal_reevaluator deve ter gate de variação mínima de preço "
        "para não gastar quota DeepSeek em mercado lateral"
    )
    assert "last_reevaluation_price" in src, (
        "signal_reevaluator deve guardar preço da última reavaliação DeepSeek"
    )


def test_position_reevaluator_uses_settings_for_min_hours_open():
    """min_hours_open deve vir de settings — não mais hardcoded."""
    src = _read("src/trading/position_reevaluator.py")
    assert "settings.reevaluation_min_time_open_hours" in src, (
        "position_reevaluator deve derivar min_hours_open de settings "
        "(unificado com signal_reevaluator e position_monitor)"
    )
