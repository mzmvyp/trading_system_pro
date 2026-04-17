"""
Tests for chart pattern detectors (Double Top/Bottom, H&S / Inverse H&S).

Uses synthetic OHLC series so we don't depend on market data or heavy libs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.chart_patterns import (
    detect_all,
    detect_double_bottom,
    detect_double_top,
    detect_head_and_shoulders,
    detect_inverse_head_and_shoulders,
)


def _to_df(closes: np.ndarray) -> pd.DataFrame:
    highs = closes + 0.05
    lows = closes - 0.05
    opens = closes
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes})


def _bump(length: int, center: int, width: int, peak: float, base: float) -> np.ndarray:
    """Triangular bump with `peak` at `center`, falling to `base` over `width`."""
    x = np.full(length, base, dtype=float)
    for i in range(length):
        dist = abs(i - center)
        if dist <= width:
            frac = 1 - dist / width
            x[i] = base + frac * (peak - base)
    return x


def test_double_top_detected_on_clean_pattern():
    length = 60
    closes = np.full(length, 100.0)
    closes += _bump(length, 15, 7, 10.0, 0.0)  # first peak 110
    closes += _bump(length, 40, 7, 10.0, 0.0)  # second peak 110
    # Confirm break below neckline
    closes[50:] = 95.0

    df = _to_df(closes)
    pat = detect_double_top(df, order=3, tolerance_pct=2.0, min_separation=10)
    assert pat is not None, "Double top não detectado em padrão limpo"
    assert pat.direction == "BEARISH"
    assert pat.neckline < 110
    assert pat.target < pat.neckline
    assert pat.metadata["confirmed"] is True


def test_double_bottom_detected_on_clean_pattern():
    length = 60
    closes = np.full(length, 100.0)
    closes -= _bump(length, 15, 7, 10.0, 0.0)  # first trough 90
    closes -= _bump(length, 40, 7, 10.0, 0.0)  # second trough 90
    closes[50:] = 105.0  # break above neckline

    df = _to_df(closes)
    pat = detect_double_bottom(df, order=3, tolerance_pct=2.0, min_separation=10)
    assert pat is not None
    assert pat.direction == "BULLISH"
    assert pat.neckline > 90
    assert pat.target > pat.neckline
    assert pat.metadata["confirmed"] is True


def test_noise_produces_only_unconfirmed_hits():
    """Noise may produce loose pattern matches, but they must be unconfirmed.

    The `confirmed` metadata flag is what downstream callers use to gate
    execution — unconfirmed hits are expected to be ignored in practice.
    """
    rng = np.random.default_rng(42)
    closes = 100 + rng.normal(0, 0.3, 200)  # pure noise, no structural pattern
    df = _to_df(closes)
    hits = detect_all(df)
    for h in hits:
        assert h.metadata.get("confirmed") is False, (
            f"Noise produziu hit confirmado: {h.name} (não deveria fechar fora do "
            "neckline em ruído branco)"
        )


def test_head_and_shoulders_detected():
    length = 80
    closes = np.full(length, 100.0)
    closes += _bump(length, 15, 6, 8.0, 0.0)   # left shoulder 108
    closes += _bump(length, 35, 6, 15.0, 0.0)  # head 115
    closes += _bump(length, 55, 6, 8.0, 0.0)   # right shoulder 108
    closes[65:] = 95.0  # break below neckline

    df = _to_df(closes)
    pat = detect_head_and_shoulders(df, order=3, shoulder_tolerance_pct=3.0)
    assert pat is not None
    assert pat.direction == "BEARISH"
    assert pat.metadata["head"] > pat.metadata["left_shoulder"]
    assert pat.metadata["head"] > pat.metadata["right_shoulder"]
    assert pat.target < pat.neckline


def test_inverse_head_and_shoulders_detected():
    length = 80
    closes = np.full(length, 100.0)
    closes -= _bump(length, 15, 6, 8.0, 0.0)   # left shoulder 92
    closes -= _bump(length, 35, 6, 15.0, 0.0)  # head 85
    closes -= _bump(length, 55, 6, 8.0, 0.0)   # right shoulder 92
    closes[65:] = 105.0  # break above neckline

    df = _to_df(closes)
    pat = detect_inverse_head_and_shoulders(df, order=3, shoulder_tolerance_pct=3.0)
    assert pat is not None
    assert pat.direction == "BULLISH"
    assert pat.metadata["head"] < pat.metadata["left_shoulder"]
    assert pat.target > pat.neckline


def test_detect_all_returns_list():
    length = 60
    closes = np.full(length, 100.0)
    closes += _bump(length, 15, 7, 10.0, 0.0)
    closes += _bump(length, 40, 7, 10.0, 0.0)
    closes[50:] = 95.0
    df = _to_df(closes)

    hits = detect_all(df)
    assert isinstance(hits, list)
    assert any(h.name == "DOUBLE_TOP" for h in hits)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
