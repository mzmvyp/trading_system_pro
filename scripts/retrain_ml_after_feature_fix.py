"""
Retrain ML/LSTM after the Tier 3.12 feature-alignment fix.

WHY THIS SCRIPT EXISTS
----------------------
The deep_analysis_report identified that `risk_distance_pct`,
`reward_distance_pct` and `risk_reward_ratio` had different semantics between
training and inference, producing an INVERTED ML (correlation
ML<->outcome = -0.1257, p<0.001). After the fix in dataset_generator.py,
train_from_signals.py and online_learning.py, **every existing model must be
retrained from scratch** — otherwise the stored weights still encode the old,
wrong mapping.

PIPELINE
--------
1. Regenerate the classical ML dataset (dataset_generator) with aligned features
2. Retrain the classical model via train_from_signals (XGBoost/RF)
3. Regenerate the backtest sequence dataset (backtest_dataset_generator)
4. Retrain the Bi-LSTM (lstm_sequence_validator)
5. Print a checklist of what to flip in config.py before re-enabling

USAGE
-----
    python scripts/retrain_ml_after_feature_fix.py

SAFE TO RUN
-----------
Models are written to ml_models/ as new files; old models are kept until the
user manually promotes the new ones. Re-enabling ml_validation_enabled /
lstm_validation_enabled in config.py is a SEPARATE manual step.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list[str], *, allow_fail: bool = False) -> int:
    print(f"\n$ {' '.join(cmd)}")
    rc = subprocess.call(cmd, cwd=REPO_ROOT)
    if rc != 0 and not allow_fail:
        print(f"[ERRO] comando falhou (rc={rc}): {' '.join(cmd)}")
        sys.exit(rc)
    return rc


def main() -> None:
    print("=" * 70)
    print("RETRAIN ML/LSTM APÓS FIX DE FEATURE ALIGNMENT (Tier 3.13)")
    print("=" * 70)
    print("Pré-requisito: testes tests/test_ml_feature_alignment.py PASSANDO")
    print("Se esses testes falharem, não continue — a semântica ainda está errada.")
    print("=" * 70)

    rc = _run(
        [sys.executable, "-m", "pytest", "tests/test_ml_feature_alignment.py", "-q"],
        allow_fail=True,
    )
    if rc != 0:
        print("\n[ABORT] Feature alignment tests falharam. Corrija antes de retreinar.")
        sys.exit(1)

    # 1. Dataset clássico (aligned features agora)
    print("\n[1/4] Regenerando dataset clássico (ML)…")
    _run([sys.executable, "-m", "src.ml.train_from_signals"])

    # 2. Dataset LSTM (sequência de candles + features estáticas)
    print("\n[2/4] Regenerando dataset LSTM (backtest generator)…")
    _run([sys.executable, "-m", "src.ml.backtest_dataset_generator"], allow_fail=True)

    # 3. Treinar Bi-LSTM
    print("\n[3/4] Treinando Bi-LSTM…")
    _run([sys.executable, "-m", "src.ml.lstm_sequence_validator"], allow_fail=True)

    # 4. Relatório
    print("\n[4/4] Relatório final")
    print("=" * 70)
    print("CHECKLIST PÓS-TREINO (manual):")
    print("  [ ] Revisar val_auc/balanced_accuracy dos novos modelos em")
    print("      ml_models/*.json — só promover se > 0.55")
    print("  [ ] Rodar backtest com ml_validation_enabled=True/False e comparar")
    print("      PnL. Expectativa após fix: ML ligado >= ML desligado.")
    print("  [ ] Verificar correlação ML↔outcome no revalidation_report.")
    print("      Se ainda estiver negativa, NÃO habilitar ML em produção.")
    print("  [ ] Só então flippar em src/core/config.py:")
    print("        ml_validation_enabled: bool = True")
    print("        lstm_validation_enabled: bool = True")
    print("=" * 70)


if __name__ == "__main__":
    main()
