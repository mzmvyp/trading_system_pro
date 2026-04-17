"""
Continuous Voter Backtester (Tier 3.15).

Aggregates `model_votes_log.jsonl` into **rolling windows** of N closed trades
and reports per-voter accuracy/WR inside each window. Catches voters whose
performance degrades over time — the static aggregate in signal_tracker.py
hides regime-dependent regressions (e.g. RSI votes-for worked great in
sideways markets but collapsed to 22% WR during the April rally).

Usage:
    python scripts/voter_backtester.py                       # window 200
    python scripts/voter_backtester.py --window 100 --step 50
    python scripts/voter_backtester.py --min-votes 20        # ignore sparse
    python scripts/voter_backtester.py --json                # machine output

Windowing: sliding window of `--window` trades, advancing by `--step` trades.
Reports per voter:
    - Accuracy = (correct votes) / (total non-neutral votes)
    - For-WR   = win rate when voter voted for the side actually taken
    - Against-Acc = accuracy when voting against (= trade lost)
    - N = sample size (non-neutral votes in window)

A voter whose For-WR drops below 35% in the latest window is flagged as
candidate for veto-only downgrade (see agent.py Tier 2 refactor).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterator, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.trading.signal_tracker import MODEL_VOTES_LOG  # noqa: E402

VOTER_NAMES = [
    "rsi",
    "macd",
    "trend",
    "adx",
    "bb",
    "orderbook",
    "mtf",
    "cvd",
    "ml",
    "lstm",
    "llm",
    "regime",
    "chart_patterns",
    "setup_validator",
]

WINNING_OUTCOMES = {"TP1_HIT", "TP2_HIT"}
LOSING_OUTCOMES = {"SL_HIT"}


def _iter_records(path: Path) -> Iterator[Dict]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _score_window(records: List[Dict]) -> Dict[str, Dict]:
    stats = {
        name: {
            "total": 0,
            "correct": 0,
            "voted_for": 0,
            "voted_against": 0,
            "for_wins": 0,
            "against_losses": 0,
        }
        for name in VOTER_NAMES
    }
    for rec in records:
        outcome = rec.get("outcome", "")
        if outcome not in WINNING_OUTCOMES and outcome not in LOSING_OUTCOMES:
            continue
        is_winner = outcome in WINNING_OUTCOMES
        votes = rec.get("voter_votes") or {}
        if not isinstance(votes, dict):
            continue
        for name in VOTER_NAMES:
            v = votes.get(name)
            if v is None or v == 0:
                continue
            s = stats[name]
            s["total"] += 1
            if v > 0:
                s["voted_for"] += 1
                if is_winner:
                    s["for_wins"] += 1
                    s["correct"] += 1
            else:
                s["voted_against"] += 1
                if not is_winner:
                    s["against_losses"] += 1
                    s["correct"] += 1

    out: Dict[str, Dict] = {}
    for name, s in stats.items():
        if s["total"] == 0:
            continue
        for_wr = (s["for_wins"] / s["voted_for"]) if s["voted_for"] > 0 else None
        against_acc = (
            s["against_losses"] / s["voted_against"] if s["voted_against"] > 0 else None
        )
        out[name] = {
            "n": s["total"],
            "accuracy": round(s["correct"] / s["total"], 4),
            "for_wr": round(for_wr, 4) if for_wr is not None else None,
            "against_acc": round(against_acc, 4) if against_acc is not None else None,
            "voted_for": s["voted_for"],
            "voted_against": s["voted_against"],
        }
    return out


def _fmt_pct(v) -> str:
    return f"{v * 100:5.1f}%" if isinstance(v, (int, float)) else "  n/a"


def _print_window(idx: int, start_ts: str, end_ts: str, stats: Dict, min_votes: int) -> None:
    print(
        f"\n── Janela {idx}  [{start_ts[:16]} → {end_ts[:16]}]  "
        f"(voters com N ≥ {min_votes})"
    )
    print(f"  {'voter':<10}{'N':>5}  {'acc':>7}  {'for-WR':>7}  {'against-acc':>12}  flags")
    for name in VOTER_NAMES:
        s = stats.get(name)
        if not s or s["n"] < min_votes:
            continue
        flags = []
        if s["for_wr"] is not None and s["for_wr"] < 0.35 and s["voted_for"] >= 10:
            flags.append("VETO-ONLY?")
        if s["against_acc"] is not None and s["against_acc"] >= 0.65 and s["voted_against"] >= 10:
            flags.append("strong-veto")
        print(
            f"  {name:<10}{s['n']:>5}  {_fmt_pct(s['accuracy']):>7}  "
            f"{_fmt_pct(s['for_wr']):>7}  {_fmt_pct(s['against_acc']):>12}  "
            f"{', '.join(flags)}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", default=str(REPO_ROOT / MODEL_VOTES_LOG))
    ap.add_argument("--window", type=int, default=200, help="trades per window")
    ap.add_argument("--step", type=int, default=100, help="slide step")
    ap.add_argument("--min-votes", type=int, default=10, help="min N to display a voter")
    ap.add_argument("--json", action="store_true", help="emit machine-readable output")
    args = ap.parse_args()

    records = [
        r
        for r in _iter_records(Path(args.log))
        if r.get("outcome") in WINNING_OUTCOMES | LOSING_OUTCOMES
    ]
    records.sort(key=lambda r: r.get("closed_at", ""))

    if len(records) < args.window:
        print(f"[INFO] Apenas {len(records)} trades fechados; janela mínima é {args.window}.")
        return 0

    windows: List[Dict] = []
    i = 0
    while i + args.window <= len(records):
        win = records[i : i + args.window]
        stats = _score_window(win)
        windows.append(
            {
                "window_index": len(windows) + 1,
                "start": win[0].get("closed_at", ""),
                "end": win[-1].get("closed_at", ""),
                "n_trades": len(win),
                "voters": stats,
            }
        )
        i += args.step

    if args.json:
        print(json.dumps({"windows": windows}, indent=2, default=str))
        return 0

    print(f"Total trades fechados: {len(records)} | janelas: {len(windows)} "
          f"(window={args.window}, step={args.step})")
    for w in windows:
        _print_window(
            w["window_index"], w["start"], w["end"], w["voters"], args.min_votes
        )

    # Veredicto final (última janela)
    latest = windows[-1]["voters"]
    flags = [
        n
        for n, s in latest.items()
        if s["for_wr"] is not None and s["for_wr"] < 0.35 and s["voted_for"] >= 10
    ]
    print("\n" + "=" * 70)
    if flags:
        print(f"[ALERTA] Última janela: voters {flags} têm for-WR < 35% com N≥10.")
        print("         Candidatos a veto-only (ver agent.py Tier 2).")
    else:
        print("[OK] Nenhum voter abaixo do piso na última janela.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
