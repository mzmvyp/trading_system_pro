"""
Backfill Voter Accuracy — Reprocessa sinais historicos para popular voter_votes.

Lê todos os sinais salvos em signals/*.json, reconstroi voter_votes a partir de
confluence_details (texto) e indicadores, avalia resultado (TP/SL) e alimenta
o model_votes_log.jsonl para o dashboard de Voter Accuracy.

Uso:
    python scripts/backfill_voter_accuracy.py                 # todos os sinais
    python scripts/backfill_voter_accuracy.py --last 100      # ultimos 100
    python scripts/backfill_voter_accuracy.py --days 7        # ultimos 7 dias
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Adicionar raiz ao path
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from src.trading.signal_tracker import (
    MODEL_VOTES_LOG,
    evaluate_signal,
    load_all_signals,
)

# Voters tecnicos que podemos reconstruir do confluence_details
VOTER_PATTERNS = {
    "rsi": {
        "for": [r"RSI oversold", r"RSI overbought"],
        "against": [r"RSI contra"],
    },
    "macd": {
        "for": [r"MACD aligned"],
        "against": [r"MACD contra"],
    },
    "trend": {
        "for": [r"Trend aligned"],
        "against": [r"Trend contra"],
    },
    "adx": {
        "for": [r"ADX strong trend"],
        "against": [],
    },
    "bb": {
        "for": [r"BB near lower band", r"BB near upper band"],
        "against": [],
    },
    "orderbook": {
        "for": [r"Orderbook aligned"],
        "against": [r"Orderbook contra"],
    },
    "mtf": {
        "for": [r"MTF aligned"],
        "against": [r"MTF contra"],
    },
    "cvd": {
        "for": [r"CVD aligned"],
        "against": [r"CVD contra"],
    },
}


def reconstruct_voter_votes(signal: dict) -> dict:
    """
    Reconstroi voter_votes a partir de confluence_details (texto)
    e dados do sinal (ml_probability, lstm_probability, confidence).
    """
    # Se ja tem voter_votes estruturado, usar direto
    if signal.get("voter_votes") and isinstance(signal["voter_votes"], dict):
        return signal["voter_votes"]

    details = signal.get("confluence_details", [])
    if not details:
        return {}

    details_text = " | ".join(details) if isinstance(details, list) else str(details)

    votes = {}

    # Reconstruir votos tecnicos a partir do texto
    for voter_name, patterns in VOTER_PATTERNS.items():
        vote = 0
        for p in patterns.get("for", []):
            if re.search(p, details_text, re.IGNORECASE):
                vote = 1
                break
        if vote == 0:
            for p in patterns.get("against", []):
                if re.search(p, details_text, re.IGNORECASE):
                    vote = -1
                    break
        votes[voter_name] = vote

    # ML vote
    ml_prob = signal.get("ml_probability")
    ml_pred = signal.get("ml_prediction")
    if ml_prob is not None:
        if ml_prob >= 0.75:
            votes["ml"] = 1
        elif ml_prob < 0.30:
            votes["ml"] = -2
        elif ml_prob < 0.50:
            votes["ml"] = -1
        else:
            votes["ml"] = 0
        votes["ml_prob"] = round(ml_prob, 4)
    elif ml_pred is not None:
        votes["ml"] = 1 if ml_pred == 1 else -1
        votes["ml_prob"] = 0.5
    else:
        votes["ml"] = 0
        votes["ml_prob"] = 0.5

    # LSTM vote
    lstm_prob = signal.get("lstm_probability")
    if lstm_prob is not None:
        if lstm_prob >= 0.75:
            votes["lstm"] = 1
        elif lstm_prob < 0.30:
            votes["lstm"] = -2
        elif lstm_prob < 0.50:
            votes["lstm"] = -1
        else:
            votes["lstm"] = 0
        votes["lstm_prob"] = round(lstm_prob, 4)
    else:
        votes["lstm"] = 0
        votes["lstm_prob"] = 0.5

    # LLM vote
    confidence = signal.get("confidence", 5)
    votes["llm"] = 1 if confidence >= 5 else 0

    return votes


def backfill(last_n: int = 0, days: int = 0):
    """Reprocessa sinais historicos e popula model_votes_log.jsonl."""
    print("=" * 60)
    print("BACKFILL VOTER ACCURACY")
    print("=" * 60)

    # Carregar sinais
    signals = load_all_signals("signals")
    print(f"[1/4] {len(signals)} sinais encontrados em signals/")

    if not signals:
        print("[ERRO] Nenhum sinal encontrado.")
        return

    # Filtrar por data ou quantidade
    if days > 0:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        filtered = []
        for s in signals:
            ts = s.get("timestamp", "")
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if dt >= cutoff:
                    filtered.append(s)
            except (ValueError, TypeError):
                pass
        signals = filtered
        print(f"[FILTRO] {len(signals)} sinais nos ultimos {days} dias")

    if last_n > 0:
        # Ordenar por timestamp (mais recente primeiro) e pegar ultimos N
        signals.sort(
            key=lambda s: s.get("timestamp", ""),
            reverse=True,
        )
        signals = signals[:last_n]
        print(f"[FILTRO] Pegando ultimos {last_n} sinais")

    # Filtrar apenas BUY/SELL (NO_SIGNAL nao tem outcome)
    signals = [s for s in signals if s.get("signal") in ("BUY", "SELL")]
    print(f"[2/4] {len(signals)} sinais BUY/SELL para processar")

    if not signals:
        print("[AVISO] Nenhum sinal BUY/SELL encontrado.")
        return

    # Avaliar cada sinal e reconstruir voter_votes
    records = []
    evaluated = 0
    skipped = 0

    for i, signal in enumerate(signals):
        if (i + 1) % 20 == 0:
            print(f"  Processando {i + 1}/{len(signals)}...")

        # Avaliar resultado (TP/SL)
        try:
            evaluation = evaluate_signal(signal)
        except Exception:
            skipped += 1
            continue

        outcome = evaluation.get("outcome", "")
        if outcome not in ("SL_HIT", "TP1_HIT", "TP2_HIT"):
            skipped += 1
            continue

        is_winner = evaluation.get("is_winner", False)

        # Reconstruir voter_votes
        voter_votes = reconstruct_voter_votes(signal)

        # ML correct
        ml_prob = signal.get("ml_probability")
        if ml_prob is not None:
            if ml_prob >= 0.5:
                ml_correct = is_winner
            else:
                ml_correct = not is_winner
        else:
            ml_correct = None

        # LSTM correct
        lstm_prob = signal.get("lstm_probability")
        if lstm_prob is not None:
            if lstm_prob > 0.6:
                lstm_vote = 1
            elif lstm_prob < 0.4:
                lstm_vote = -1
            else:
                lstm_vote = 0
        else:
            lstm_vote = None
            lstm_prob = None

        record = {
            "symbol": signal.get("symbol", ""),
            "timestamp": signal.get("timestamp", ""),
            "source": signal.get("source", ""),
            "signal": signal.get("signal", ""),
            "outcome": outcome,
            "is_winner": is_winner,
            "executed": signal.get("executed", False),
            "ml_prediction": signal.get("ml_prediction"),
            "ml_probability": ml_prob,
            "lstm_probability": lstm_prob,
            "lstm_vote": lstm_vote,
            "ml_correct": ml_correct,
            "ml_operational_correct": evaluation.get("ml_operational_correct"),
            "lstm_correct": evaluation.get("lstm_correct"),
            "confluence_score": signal.get("confluence_score"),
            "confluence_votes_for": signal.get("confluence_votes_for"),
            "voter_votes": voter_votes,
            "_backfilled": True,
        }

        records.append(record)
        evaluated += 1

    print(f"[3/4] {evaluated} sinais avaliados com sucesso, {skipped} ignorados")

    if not records:
        print("[AVISO] Nenhum registro para salvar.")
        return

    # Salvar no model_votes_log.jsonl (append ou criar)
    os.makedirs(os.path.dirname(MODEL_VOTES_LOG) or ".", exist_ok=True)

    # Ler registros existentes para evitar duplicatas
    existing_keys = set()
    if os.path.exists(MODEL_VOTES_LOG):
        with open(MODEL_VOTES_LOG, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        r = json.loads(line)
                        key = f"{r.get('symbol')}_{r.get('timestamp')}_{r.get('source')}"
                        existing_keys.add(key)
                    except json.JSONDecodeError:
                        pass

    new_count = 0
    with open(MODEL_VOTES_LOG, "a", encoding="utf-8") as f:
        for record in records:
            key = f"{record['symbol']}_{record['timestamp']}_{record['source']}"
            if key in existing_keys:
                continue
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            new_count += 1

    print(f"[4/4] {new_count} registros adicionados ao {MODEL_VOTES_LOG}")
    if new_count < len(records):
        print(f"  ({len(records) - new_count} duplicatas ignoradas)")

    # Mostrar resumo
    print("\n" + "=" * 60)
    print("RESUMO")
    print("=" * 60)

    # Computar voter accuracy com os dados
    from src.trading.signal_tracker import compute_voter_accuracy
    voter_data = compute_voter_accuracy()

    if voter_data:
        all_names = [
            "rsi", "macd", "trend", "adx", "bb",
            "orderbook", "mtf", "cvd", "ml", "lstm", "llm",
        ]
        print(f"\n{'Votante':<12} {'Votos':>6} {'Corretos':>9} {'Accuracy':>9}")
        print("-" * 40)
        for name in all_names:
            v = voter_data.get(name, {})
            total = v.get("total", 0)
            correct = v.get("correct", 0)
            acc = v.get("accuracy")
            acc_str = f"{acc * 100:.1f}%" if acc is not None else "N/A"
            print(f"{name.upper():<12} {total:>6} {correct:>9} {acc_str:>9}")

        ds = voter_data.get("deepseek", {})
        ds_wr = ds.get("win_rate")
        print(f"\nDeepSeek Win Rate: {ds_wr * 100:.1f}% ({ds.get('total', 0)} sinais)")

    print("\n[OK] Backfill concluido! Dashboard Voter Accuracy atualizado.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill voter accuracy data")
    parser.add_argument("--last", type=int, default=0, help="Ultimos N sinais")
    parser.add_argument("--days", type=int, default=0, help="Ultimos N dias")
    args = parser.parse_args()

    backfill(last_n=args.last, days=args.days)
