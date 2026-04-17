"""
Chart Pattern Detector — reversals on swing highs/lows.

Patterns:
- Double Top / Double Bottom (reversal after two matching peaks / troughs)
- Head & Shoulders (H&S) / Inverse H&S (a.k.a. OCO / OCOI)

Contract: given an OHLC DataFrame, return a list of detected patterns
with direction, neckline, target, and a confidence heuristic. Pure analytics —
no orders are placed from here. This module exists as Tier 3.14 of the
liquidation post-mortem: price topped at $85k+ on 15/abr with a clear double
top that no voter recognized. See deep_analysis_report.txt.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ChartPattern:
    name: str
    direction: str  # "BULLISH" (reversal up) or "BEARISH" (reversal down)
    neckline: float
    target: float
    reliability: float  # 0..1
    metadata: Dict = field(default_factory=dict)


def _find_swings(series: pd.Series, order: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Return (peak_idx, trough_idx) using a ±`order` window filter.

    A point is a peak/trough if it's strictly greater/less than every neighbor
    in its ±order window. This handles plateaus gracefully — flat regions
    simply don't produce swings.
    """
    vals = series.values
    n = len(vals)
    peaks: List[int] = []
    troughs: List[int] = []
    for i in range(order, n - order):
        left = vals[i - order : i]
        right = vals[i + 1 : i + order + 1]
        center = vals[i]
        if center > left.max() and center > right.max():
            peaks.append(i)
        elif center < left.min() and center < right.min():
            troughs.append(i)
    return np.array(peaks, dtype=int), np.array(troughs, dtype=int)


def detect_double_top(
    df: pd.DataFrame,
    *,
    order: int = 5,
    tolerance_pct: float = 1.0,
    min_separation: int = 5,
) -> Optional[ChartPattern]:
    """Double top: two peaks within tolerance_pct, confirmed trough in between.

    Target = neckline - (peak - neckline) (mirror projection).
    """
    if len(df) < order * 4:
        return None

    highs = df["high"]
    lows = df["low"]
    peaks, _ = _find_swings(highs, order=order)
    if len(peaks) < 2:
        return None

    p2, p1 = peaks[-1], peaks[-2]
    if p2 - p1 < min_separation:
        return None

    h1, h2 = float(highs.iloc[p1]), float(highs.iloc[p2])
    if max(h1, h2) == 0:
        return None
    diff_pct = abs(h1 - h2) / max(h1, h2) * 100
    if diff_pct > tolerance_pct:
        return None

    trough_range = lows.iloc[p1 : p2 + 1]
    if trough_range.empty:
        return None
    neckline = float(trough_range.min())
    peak = (h1 + h2) / 2
    if peak <= neckline:
        return None

    last_close = float(df["close"].iloc[-1])
    # Confirmação só é válida se o preço fechou abaixo do neckline
    confirmed = last_close < neckline
    reliability = 0.6 + (0.3 if confirmed else 0.0) + max(0.0, 0.1 - diff_pct / 10.0)
    target = neckline - (peak - neckline)

    return ChartPattern(
        name="DOUBLE_TOP",
        direction="BEARISH",
        neckline=neckline,
        target=target,
        reliability=min(reliability, 0.95),
        metadata={
            "peak_1_idx": int(p1),
            "peak_2_idx": int(p2),
            "peak_1_price": h1,
            "peak_2_price": h2,
            "peaks_diff_pct": diff_pct,
            "confirmed": confirmed,
        },
    )


def detect_double_bottom(
    df: pd.DataFrame,
    *,
    order: int = 5,
    tolerance_pct: float = 1.0,
    min_separation: int = 5,
) -> Optional[ChartPattern]:
    """Double bottom: two troughs within tolerance_pct, peak in between."""
    if len(df) < order * 4:
        return None

    highs = df["high"]
    lows = df["low"]
    _, troughs = _find_swings(lows, order=order)
    if len(troughs) < 2:
        return None

    t2, t1 = troughs[-1], troughs[-2]
    if t2 - t1 < min_separation:
        return None

    l1, l2 = float(lows.iloc[t1]), float(lows.iloc[t2])
    if max(l1, l2) == 0:
        return None
    diff_pct = abs(l1 - l2) / max(l1, l2) * 100
    if diff_pct > tolerance_pct:
        return None

    peak_range = highs.iloc[t1 : t2 + 1]
    if peak_range.empty:
        return None
    neckline = float(peak_range.max())
    trough = (l1 + l2) / 2
    if neckline <= trough:
        return None

    last_close = float(df["close"].iloc[-1])
    confirmed = last_close > neckline
    reliability = 0.6 + (0.3 if confirmed else 0.0) + max(0.0, 0.1 - diff_pct / 10.0)
    target = neckline + (neckline - trough)

    return ChartPattern(
        name="DOUBLE_BOTTOM",
        direction="BULLISH",
        neckline=neckline,
        target=target,
        reliability=min(reliability, 0.95),
        metadata={
            "trough_1_idx": int(t1),
            "trough_2_idx": int(t2),
            "trough_1_price": l1,
            "trough_2_price": l2,
            "troughs_diff_pct": diff_pct,
            "confirmed": confirmed,
        },
    )


def detect_head_and_shoulders(
    df: pd.DataFrame,
    *,
    order: int = 5,
    shoulder_tolerance_pct: float = 2.0,
) -> Optional[ChartPattern]:
    """Head & Shoulders (OCO): 3 peaks — middle higher, shoulders roughly equal.

    Neckline = average of the two troughs between peaks.
    Target = neckline - (head - neckline).
    """
    if len(df) < order * 6:
        return None

    highs = df["high"]
    lows = df["low"]
    peaks, _ = _find_swings(highs, order=order)
    _, low_troughs = _find_swings(lows, order=order)
    if len(peaks) < 3:
        return None

    ls_idx, head_idx, rs_idx = peaks[-3], peaks[-2], peaks[-1]
    ls, head, rs = (
        float(highs.iloc[ls_idx]),
        float(highs.iloc[head_idx]),
        float(highs.iloc[rs_idx]),
    )
    if not (head > ls and head > rs):
        return None
    if max(ls, rs) == 0:
        return None
    shoulder_diff_pct = abs(ls - rs) / max(ls, rs) * 100
    if shoulder_diff_pct > shoulder_tolerance_pct:
        return None

    # Find necklines: use lows troughs if detected, else use min in the range
    left_troughs = [t for t in low_troughs if ls_idx < t < head_idx]
    right_troughs = [t for t in low_troughs if head_idx < t < rs_idx]
    if left_troughs:
        nl_left = float(lows.iloc[left_troughs[-1]])
    else:
        left_slice = lows.iloc[ls_idx + 1 : head_idx]
        if left_slice.empty:
            return None
        nl_left = float(left_slice.min())
    if right_troughs:
        nl_right = float(lows.iloc[right_troughs[0]])
    else:
        right_slice = lows.iloc[head_idx + 1 : rs_idx]
        if right_slice.empty:
            return None
        nl_right = float(right_slice.min())
    neckline = (nl_left + nl_right) / 2
    if head <= neckline:
        return None

    last_close = float(df["close"].iloc[-1])
    confirmed = last_close < neckline
    reliability = 0.7 + (0.2 if confirmed else 0.0) + max(0.0, 0.1 - shoulder_diff_pct / 20.0)
    target = neckline - (head - neckline)

    return ChartPattern(
        name="HEAD_AND_SHOULDERS",
        direction="BEARISH",
        neckline=neckline,
        target=target,
        reliability=min(reliability, 0.95),
        metadata={
            "left_shoulder": ls,
            "head": head,
            "right_shoulder": rs,
            "shoulder_diff_pct": shoulder_diff_pct,
            "confirmed": confirmed,
        },
    )


def detect_inverse_head_and_shoulders(
    df: pd.DataFrame,
    *,
    order: int = 5,
    shoulder_tolerance_pct: float = 2.0,
) -> Optional[ChartPattern]:
    """Inverse H&S (OCOI): 3 troughs — middle lower, shoulders roughly equal."""
    if len(df) < order * 6:
        return None

    highs = df["high"]
    lows = df["low"]
    _, trough_idx = _find_swings(lows, order=order)
    peak_idx, _ = _find_swings(highs, order=order)
    if len(trough_idx) < 3:
        return None

    ls_idx, head_idx, rs_idx = trough_idx[-3], trough_idx[-2], trough_idx[-1]
    ls, head, rs = (
        float(lows.iloc[ls_idx]),
        float(lows.iloc[head_idx]),
        float(lows.iloc[rs_idx]),
    )
    if not (head < ls and head < rs):
        return None
    if max(ls, rs) == 0:
        return None
    shoulder_diff_pct = abs(ls - rs) / max(ls, rs) * 100
    if shoulder_diff_pct > shoulder_tolerance_pct:
        return None

    left_peaks = [p for p in peak_idx if ls_idx < p < head_idx]
    right_peaks = [p for p in peak_idx if head_idx < p < rs_idx]
    if left_peaks:
        nl_left = float(highs.iloc[left_peaks[-1]])
    else:
        left_slice = highs.iloc[ls_idx + 1 : head_idx]
        if left_slice.empty:
            return None
        nl_left = float(left_slice.max())
    if right_peaks:
        nl_right = float(highs.iloc[right_peaks[0]])
    else:
        right_slice = highs.iloc[head_idx + 1 : rs_idx]
        if right_slice.empty:
            return None
        nl_right = float(right_slice.max())
    neckline = (nl_left + nl_right) / 2
    if neckline <= head:
        return None

    last_close = float(df["close"].iloc[-1])
    confirmed = last_close > neckline
    reliability = 0.7 + (0.2 if confirmed else 0.0) + max(0.0, 0.1 - shoulder_diff_pct / 20.0)
    target = neckline + (neckline - head)

    return ChartPattern(
        name="INVERSE_HEAD_AND_SHOULDERS",
        direction="BULLISH",
        neckline=neckline,
        target=target,
        reliability=min(reliability, 0.95),
        metadata={
            "left_shoulder": ls,
            "head": head,
            "right_shoulder": rs,
            "shoulder_diff_pct": shoulder_diff_pct,
            "confirmed": confirmed,
        },
    )


def detect_all(df: pd.DataFrame) -> List[ChartPattern]:
    """Run every detector and return a list of hits (possibly empty)."""
    out: List[ChartPattern] = []
    for fn in (
        detect_double_top,
        detect_double_bottom,
        detect_head_and_shoulders,
        detect_inverse_head_and_shoulders,
    ):
        try:
            res = fn(df)
            if res is not None:
                out.append(res)
        except Exception as e:
            logger.debug(f"[CHART PATTERNS] {fn.__name__} falhou: {e}")
    return out
