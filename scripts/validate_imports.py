#!/usr/bin/env python3
"""Validate imports of src modules inside Docker. Exit 0 if all succeed."""
import importlib
import sys

# Submodules to import (excluding __init__); covers consolidated Phase 6 modules
MODULES = [
    "src.core.config",
    "src.core.logger",
    "src.core.constants",
    "src.core.exceptions",
    "src.exchange.client",
    "src.exchange.executor",
    "src.exchange.utils",
    "src.analysis.agno_tools",
    "src.analysis.indicators",
    "src.analysis.market_data",
    "src.analysis.multi_timeframe",
    "src.trading.agent",
    "src.trading.signal_reevaluator",
    "src.trading.stop_adjuster",
    "src.trading.paper_trading",
    "src.trading.portfolio",
    "src.trading.risk_manager",
    "src.trading.position_manager",
    "src.prompts.deepseek_client",
    "src.dashboard.streamlit_app",
    "src.dashboard.ml_dashboard",
    "src.ml.dataset_generator",
    "src.ml.simple_validator",
    "src.ml.online_learning",
    "src.strategies.base_strategy",
    "src.strategies.trend_following",
    "src.strategies.mean_reversion",
    "src.strategies.breakout",
    "src.sentiment.llm_analyzer",
    "src.filters.volatility_filter",
    "src.utils.helpers",
]

def main():
    failed = []
    for name in MODULES:
        try:
            importlib.import_module(name)
            print(f"  OK {name}")
        except Exception as e:
            print(f"  FAIL {name}: {e}", file=sys.stderr)
            failed.append((name, e))
    if failed:
        print(f"\n{len(failed)} module(s) failed.", file=sys.stderr)
        sys.exit(1)
    print(f"\nAll {len(MODULES)} module imports OK.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
