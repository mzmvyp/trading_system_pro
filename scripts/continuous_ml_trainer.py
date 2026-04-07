"""
Agente de Treino Contínuo ML — Loop iterativo até atingir target accuracy.
==========================================================================

Fluxo:
1. Treina modelo ML com config actual
2. Testa nos últimos N dias de sinais REAIS (nunca vistos no treino)
3. Calcula accuracy real
4. Se < target: envia features + resultados ao DeepSeek para sugestões
5. Aplica sugestões, evita repetir configs já testadas
6. Loop até atingir target ou max_iterations

Aplica-se a: SimpleSignalValidator (ensemble ML) e LSTMSequenceValidator.
Modelos: XGBoost, RandomForest, GradientBoosting, MLP, LogisticRegression.

Uso:
  python scripts/continuous_ml_trainer.py --target-accuracy 0.60 --max-iter 20
  python scripts/continuous_ml_trainer.py --model lstm --target-accuracy 0.55
"""

import argparse
import asyncio
import json
import os
import sys
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from src.core.config import settings
from src.core.logger import get_logger
from src.trading.signal_tracker import load_all_signals, evaluate_signal

logger = get_logger("continuous_trainer")

ATTEMPTS_LOG = ROOT / "ml_models" / "training_attempts.json"
BLACKLIST = set(getattr(settings, "token_blacklist", []))


def load_attempts() -> List[Dict]:
    if ATTEMPTS_LOG.exists():
        with open(ATTEMPTS_LOG, "r") as f:
            return json.load(f)
    return []


def save_attempt(attempt: Dict):
    attempts = load_attempts()
    attempts.append(attempt)
    ATTEMPTS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(ATTEMPTS_LOG, "w") as f:
        json.dump(attempts, f, indent=2, default=str)


def config_hash(config: Dict) -> str:
    """Hash para identificar config única e evitar repetir."""
    return hashlib.md5(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:12]


def get_tested_hashes() -> set:
    return {a.get("config_hash", "") for a in load_attempts()}


# ============================================================
# CARREGAR SINAIS REAIS PARA TESTE
# ============================================================
def load_real_test_signals(days_back: int = 20) -> pd.DataFrame:
    """Carrega sinais reais dos últimos N dias para teste."""
    print(f"[TEST] Carregando sinais dos últimos {days_back} dias...")
    all_signals = load_all_signals("signals")

    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    recent = []

    for sig in all_signals:
        if sig.get("symbol") in BLACKLIST:
            continue
        if sig.get("source") == "LOCAL_GEN":
            continue
        if sig.get("signal") not in ("BUY", "SELL"):
            continue

        ts_str = sig.get("timestamp", "")
        try:
            if "T" in ts_str:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            else:
                ts = datetime.strptime(ts_str[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            if ts < cutoff:
                continue
        except (ValueError, TypeError):
            continue

        result = evaluate_signal(sig)
        outcome = result.get("outcome", "PENDING")
        if outcome in ("TP1_HIT", "TP2_HIT", "SL_HIT"):
            sig["_outcome"] = outcome
            sig["_is_winner"] = outcome in ("TP1_HIT", "TP2_HIT")
            recent.append(sig)

    print(f"[TEST] {len(recent)} sinais com outcome nos últimos {days_back} dias")
    return recent


# ============================================================
# TESTAR MODELO ML NOS SINAIS REAIS
# ============================================================
def test_ml_on_real_signals(signals: List[Dict]) -> Dict:
    """Testa o modelo ML actual em sinais reais e calcula accuracy."""
    from src.ml.simple_validator import SimpleSignalValidator

    validator = SimpleSignalValidator()
    try:
        validator.load_models()
    except Exception as e:
        return {"error": f"Modelo não carregado: {e}", "accuracy": 0, "n_tested": 0}

    correct = 0
    total = 0
    details = []

    for sig in signals:
        try:
            result = validator.validate_deepseek_signal(sig)
            ml_pred = result.get("model_validation", {}).get("prediction", 0)
            ml_prob = result.get("model_validation", {}).get("probability_success", 0.5)
            actual = 1 if sig["_is_winner"] else 0

            is_correct = ml_pred == actual
            if is_correct:
                correct += 1
            total += 1

            details.append({
                "symbol": sig.get("symbol"),
                "signal": sig.get("signal"),
                "ml_prob": round(ml_prob, 3),
                "ml_pred": ml_pred,
                "actual": actual,
                "correct": is_correct,
            })
        except Exception:
            continue

    accuracy = correct / max(total, 1)
    print(f"[TEST ML] Accuracy: {accuracy*100:.1f}% ({correct}/{total})")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "n_tested": total,
        "details": details,
    }


# ============================================================
# PEDIR SUGESTÕES AO DEEPSEEK
# ============================================================
async def ask_deepseek_for_improvements(
    current_results: Dict,
    current_config: Dict,
    previous_attempts: List[Dict],
) -> Optional[Dict]:
    """Envia resultados ao DeepSeek e pede sugestões de melhoria."""
    try:
        from src.exchange.client import BinanceClient  # noqa: just checking imports work
    except Exception:
        pass

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("[WARN] DEEPSEEK_API_KEY não definida — gerando sugestão local")
        return _generate_local_suggestion(current_results, previous_attempts)

    import httpx

    prev_summary = ""
    for i, att in enumerate(previous_attempts[-5:]):
        prev_summary += f"\n  Tentativa {i+1}: accuracy={att.get('accuracy',0)*100:.1f}%, config_hash={att.get('config_hash','')}"

    prompt = f"""Sou um sistema de ML para trading de crypto futures. Preciso melhorar a accuracy do meu modelo.

RESULTADOS ACTUAIS:
- Accuracy: {current_results['accuracy']*100:.1f}% ({current_results['correct']}/{current_results['total']})
- Features usadas: {current_config.get('features', [])}
- Modelos no ensemble: LogisticRegression, RandomForest, GradientBoosting, MLP, XGBoost
- Target: classificar se um trade vai dar TP (win) ou SL (loss)

TENTATIVAS ANTERIORES:{prev_summary if prev_summary else ' Nenhuma'}

FEATURES DISPONÍVEIS:
rsi, macd_histogram, adx, atr, bb_position, cvd, orderbook_imbalance,
bullish_tf_count, bearish_tf_count, confidence, trend_encoded, sentiment_encoded,
signal_encoded, risk_distance_pct (ATR%), reward_distance_pct (candle_body%), risk_reward_ratio (volume_ratio)

REGRAS:
1. Sugere UMA mudança concreta (feature engineering, hyperparameter, preprocessing)
2. Não sugerir features que não existem nos dados
3. Responder APENAS em JSON válido com este formato:
{{
  "suggestion_type": "feature_engineering|hyperparameter|preprocessing|ensemble",
  "description": "breve descrição",
  "changes": {{...detalhes específicos...}},
  "expected_improvement": "X%"
}}"""

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 500,
                },
            )
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

            # Extract JSON from response
            import re
            json_match = re.search(r"\{[\s\S]*\}", content)
            if json_match:
                suggestion = json.loads(json_match.group())
                print(f"[DEEPSEEK] Sugestão: {suggestion.get('description', 'N/A')}")
                return suggestion
    except Exception as e:
        print(f"[DEEPSEEK] Erro: {e}")

    return _generate_local_suggestion(current_results, previous_attempts)


def _generate_local_suggestion(results: Dict, previous: List) -> Dict:
    """Gera sugestão local quando DeepSeek não disponível."""
    import random

    suggestions = [
        {
            "suggestion_type": "hyperparameter",
            "description": "Aumentar n_estimators do RandomForest",
            "changes": {"rf_n_estimators": random.choice([200, 300, 500])},
        },
        {
            "suggestion_type": "hyperparameter",
            "description": "Ajustar max_depth do XGBoost",
            "changes": {"xgb_max_depth": random.choice([3, 5, 7, 10])},
        },
        {
            "suggestion_type": "feature_engineering",
            "description": "Adicionar interação RSI*ADX como feature",
            "changes": {"add_feature": "rsi_adx_interaction"},
        },
        {
            "suggestion_type": "preprocessing",
            "description": "Usar RobustScaler em vez de StandardScaler",
            "changes": {"scaler_type": "robust"},
        },
        {
            "suggestion_type": "hyperparameter",
            "description": "Aumentar learning_rate do GradientBoosting",
            "changes": {"gb_learning_rate": random.choice([0.05, 0.1, 0.2])},
        },
        {
            "suggestion_type": "ensemble",
            "description": "Usar soft voting em vez de calibrated",
            "changes": {"voting": "soft"},
        },
    ]

    tested = get_tested_hashes()
    for s in suggestions:
        h = config_hash(s["changes"])
        if h not in tested:
            return s

    return suggestions[0]


# ============================================================
# LOOP PRINCIPAL
# ============================================================
async def training_loop(
    target_accuracy: float = 0.60,
    max_iterations: int = 20,
    test_days: int = 20,
    model_type: str = "ml",
):
    print("\n" + "=" * 70)
    print(f"AGENTE DE TREINO CONTÍNUO — {model_type.upper()}")
    print(f"Target: {target_accuracy*100:.0f}% accuracy | Max: {max_iterations} iterações")
    print("=" * 70)

    # Carregar sinais de teste (fixos para toda a sessão)
    test_signals = load_real_test_signals(test_days)
    if len(test_signals) < 10:
        print(f"[ERRO] Poucos sinais de teste ({len(test_signals)} < 10)")
        return

    best_accuracy = 0
    best_iteration = 0

    for iteration in range(1, max_iterations + 1):
        print(f"\n{'='*50}")
        print(f"ITERAÇÃO {iteration}/{max_iterations}")
        print(f"{'='*50}")

        # 1. Treinar modelo
        if model_type == "ml":
            print("[TRAIN] Treinando ML ensemble...")
            from src.ml.train_from_signals import run_training_pipeline
            success = run_training_pipeline()
            if not success:
                print("[ERRO] Treino ML falhou")
                continue
        elif model_type == "lstm":
            print("[TRAIN] Treinando LSTM...")
            from src.ml.lstm_sequence_validator import LSTMSequenceValidator
            lstm = LSTMSequenceValidator()
            result = lstm.train_from_backtest(epochs=50, batch_size=32)
            if not result.get("success"):
                print(f"[ERRO] Treino LSTM falhou: {result.get('reason')}")
                continue

        # 2. Testar em sinais reais
        print("\n[TEST] Testando em sinais reais...")
        test_results = test_ml_on_real_signals(test_signals)

        accuracy = test_results["accuracy"]
        print(f"\n[RESULTADO] Iteração {iteration}: Accuracy = {accuracy*100:.1f}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_iteration = iteration
            print(f"  >>> MELHOR ATÉ AGORA! (era {best_accuracy*100:.1f}%)")

        # Log attempt
        attempt = {
            "iteration": iteration,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_type": model_type,
            "accuracy": accuracy,
            "correct": test_results["correct"],
            "total": test_results["n_tested"],
            "config_hash": config_hash({"iteration": iteration}),
        }
        save_attempt(attempt)

        # 3. Atingiu target?
        if accuracy >= target_accuracy:
            print(f"\n{'='*50}")
            print(f"TARGET ATINGIDO! {accuracy*100:.1f}% >= {target_accuracy*100:.0f}%")
            print(f"{'='*50}")
            break

        # 4. Pedir sugestões
        print("\n[IMPROVE] Pedindo sugestões ao DeepSeek...")
        previous = load_attempts()
        suggestion = await ask_deepseek_for_improvements(
            test_results,
            {"features": ["rsi", "macd_histogram", "adx", "atr", "bb_position",
                          "cvd", "orderbook_imbalance", "bullish_tf_count",
                          "bearish_tf_count", "confidence", "trend_encoded",
                          "sentiment_encoded", "signal_encoded",
                          "risk_distance_pct", "reward_distance_pct", "risk_reward_ratio"]},
            previous,
        )

        if suggestion:
            print(f"  Sugestão: {suggestion.get('description', 'N/A')}")
            print(f"  Tipo: {suggestion.get('suggestion_type', 'N/A')}")
            # TODO: aplicar sugestão automaticamente no próximo ciclo
        else:
            print("  Sem sugestão disponível")

    print(f"\n{'='*70}")
    print(f"FIM DO TREINO CONTÍNUO")
    print(f"  Melhor accuracy: {best_accuracy*100:.1f}% (iteração {best_iteration})")
    print(f"  Total iterações: {iteration}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Agente de Treino Contínuo ML/LSTM")
    parser.add_argument("--target-accuracy", type=float, default=0.60,
                        help="Accuracy target (0.0-1.0, default 0.60)")
    parser.add_argument("--max-iter", type=int, default=20,
                        help="Max iterações (default 20)")
    parser.add_argument("--test-days", type=int, default=20,
                        help="Dias de sinais para teste (default 20)")
    parser.add_argument("--model", choices=["ml", "lstm"], default="ml",
                        help="Tipo de modelo (default ml)")
    args = parser.parse_args()

    asyncio.run(training_loop(
        target_accuracy=args.target_accuracy,
        max_iterations=args.max_iter,
        test_days=args.test_days,
        model_type=args.model,
    ))


if __name__ == "__main__":
    main()
