"""
One-shot post-fix retrain (Tier 3.13).

THIS SCRIPT IS A THIN CLI WRAPPER over the SAME functions used by:
- the drift-triggered auto-retrain (agent._trigger_drift_retrain)
- the dashboard retrain buttons (ml_dashboard.py)

It exists because Tier 3.12 fixed the training feature semantics, so every
existing model is stale and should be retrained once immediately — not wait
for the next drift event. The canonical feature-alignment guard
(src/ml/feature_alignment_guard.py) runs inside each of those functions and
will refuse to train if someone ever regresses the fix.

Entrypoints called:
  1. online_learning.seed_from_evaluated_signals(force_retrain=True)
     → classical ML (XGBoost/RF) via aligned features
  2. LSTMSequenceValidator.train_from_backtest()
     → Bi-LSTM via real signal sequences

Nothing is duplicated — this is just a scheduler so you don't have to click
three buttons in the dashboard.

Usage:
    python scripts/retrain_ml_after_feature_fix.py
    python scripts/retrain_ml_after_feature_fix.py --skip-lstm
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _run_classical_ml() -> bool:
    from src.ml.online_learning import seed_from_evaluated_signals

    print("\n[1] Classical ML — seed + retrain (mesmo entrypoint do drift e do dashboard)")
    result = seed_from_evaluated_signals(force_retrain=True)
    if not result.get("success"):
        print(f"  [ERRO] {result.get('error', 'unknown')}")
        return False
    rt = result.get("retrain_result") or {}
    if rt.get("success"):
        print(
            f"  [OK] accuracy={rt.get('new_accuracy', 0):.1%} "
            f"F1={rt.get('new_f1', 0):.3f} samples={rt.get('samples_used', 0)}"
        )
    else:
        print(f"  [INFO] sinais alimentados={result.get('signals_added', 0)} — sem retrain ainda")
    return True


def _run_lstm() -> bool:
    from src.ml.lstm_sequence_validator import LSTMSequenceValidator

    print("\n[2] Bi-LSTM — train_from_backtest (mesmo entrypoint do drift)")
    lstm = LSTMSequenceValidator()
    result = asyncio.run(lstm.train_from_backtest(epochs=50, batch_size=32))
    if result.get("success"):
        print(
            f"  [OK] accuracy={result.get('test_accuracy', 0):.1%} "
            f"F1={result.get('test_f1', 0):.3f} samples={result.get('total_samples', 0)} "
            f"source={result.get('data_source', '?')}"
        )
        return True
    print(f"  [ERRO] {result.get('reason', 'unknown')}")
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-lstm", action="store_true", help="apenas ML clássico")
    args = ap.parse_args()

    # Guard: se features estiverem desalinhadas, os próprios entrypoints abaixo
    # vão levantar RuntimeError. Rodamos uma vez aqui para reportar cedo.
    from src.ml.feature_alignment_guard import (
        assert_features_aligned,
        check_feature_alignment,
    )

    ok, failures = check_feature_alignment()
    if not ok:
        print("[ABORT] feature alignment falhou:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("[guard] feature alignment OK")
    assert_features_aligned()  # idempotente, só para garantir

    ml_ok = _run_classical_ml()
    lstm_ok = True if args.skip_lstm else _run_lstm()

    print("\n" + "=" * 60)
    print(f"ML clássico: {'OK' if ml_ok else 'FALHOU'}")
    print(f"Bi-LSTM    : {'OK' if lstm_ok else 'PULADO' if args.skip_lstm else 'FALHOU'}")
    print("=" * 60)
    print("Checklist pós-retreino (manual):")
    print("  [ ] Revisar val_auc/balanced_acc em ml_models/*.json (>= 0.55)")
    print("  [ ] Rodar backtester de voters (scripts/voter_backtester.py) para")
    print("      confirmar que o ML voltou a correlacionar positivamente")
    print("  [ ] Só então flippar em src/core/config.py:")
    print("        ml_validation_enabled = True")
    print("        lstm_validation_enabled = True")
    return 0 if (ml_ok and lstm_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
