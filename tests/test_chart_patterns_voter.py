"""
Invariant test: chart_patterns must be wired into the confluence voter flow.

We parse source as text so the check runs with no heavy deps, same approach as
test_ml_feature_alignment.py.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


def test_agent_defines_vote_chart_patterns():
    src = _read("src/trading/agent.py")
    assert "async def _vote_chart_patterns(" in src, (
        "agent deve definir _vote_chart_patterns para rodar Double Top/Bottom e H&S"
    )
    assert "from src.analysis.chart_patterns import detect_all" in src, (
        "chart_patterns.detect_all precisa ser importado no voter"
    )


def test_agent_calls_chart_patterns_after_confluence():
    src = _read("src/trading/agent.py")
    assert "await self._vote_chart_patterns(symbol, llm_signal_dir)" in src, (
        "agent deve chamar _vote_chart_patterns após _calculate_technical_confluence"
    )
    assert 'voter_votes", {})["chart_patterns"]' in src, (
        "voto de chart_patterns deve ser gravado em voter_votes para tracking"
    )


def test_signal_tracker_tracks_chart_patterns_voter():
    src = _read("src/trading/signal_tracker.py")
    assert '"chart_patterns"' in src, (
        "signal_tracker deve tracked chart_patterns em voter_names para accuracy"
    )


def test_voter_backtester_tracks_chart_patterns():
    src = _read("scripts/voter_backtester.py")
    assert '"chart_patterns"' in src, (
        "voter_backtester deve incluir chart_patterns em VOTER_NAMES"
    )
