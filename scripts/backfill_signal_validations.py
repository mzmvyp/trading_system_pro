"""
Backfill Signal Validations - Reavalia todos os sinais e regista acertividade por processo.
============================================================================================

O que faz:
1. Carrega todos os sinais BUY/SELL armazenados em signals/
2. Para cada sinal, busca dados passados da Binance (klines desde o momento do sinal)
3. Se faltar opinião da ML ou do LSTM, "replaya": passa os dados no modelo e anota a indicação
4. Verifica o desfecho real (SL/TP/EXPIRED) com os dados da Binance
5. Regista no log (signals/model_votes_log.jsonl) quem acertou e quem errou
6. No fim, contabiliza acurácia por sistema (fonte, direção, ML, LSTM)

Uso:
    python -m scripts.backfill_signal_validations
    # ou
    python scripts/backfill_signal_validations.py

Requer: Dados da Binance (sem API key para klines públicos), modelos ML/LSTM opcionais.
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Garantir que o project root está no path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.trading.signal_tracker import (
    load_all_signals,
    evaluate_signal,
    _append_model_votes_log,
    get_system_accuracy_report,
    MODEL_VOTES_LOG,
)


def _parse_signal_time(timestamp_str: str) -> datetime:
    """Converte string do sinal para datetime UTC."""
    if not timestamp_str:
        return datetime.now(timezone.utc)
    try:
        if "T" in timestamp_str:
            t = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        else:
            t = datetime.strptime(timestamp_str[:19], "%Y-%m-%d %H:%M:%S")
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return t
    except (ValueError, TypeError):
        return datetime.now(timezone.utc)


def _build_ml_features_from_signal_and_candles(signal: dict, last_row: dict) -> dict:
    """Monta o dict de features esperado pelo SimpleSignalValidator a partir do sinal e da última vela."""
    entry = signal.get("entry_price") or 0
    stop = signal.get("stop_loss") or 0
    tp1 = signal.get("take_profit_1") or signal.get("take_profit") or 0

    def risk_distance():
        if entry > 0 and stop > 0:
            return abs(entry - stop) / entry * 100
        return 2.0

    def reward_distance():
        if entry > 0 and tp1 > 0:
            return abs(tp1 - entry) / entry * 100
        return 2.0

    def risk_reward():
        r, w = risk_distance(), reward_distance()
        return w / r if r > 0 else 1.0

    bb_upper = last_row.get("bb_upper") or last_row.get("close", 1)
    bb_lower = last_row.get("bb_lower") or last_row.get("close", 1)
    close = last_row.get("close", 1)
    bb_range = (bb_upper - bb_lower) if (bb_upper != bb_lower) else 1e-9
    bb_position = (close - bb_lower) / bb_range

    return {
        "rsi": last_row.get("rsi", 50),
        "macd_histogram": last_row.get("macd_hist", last_row.get("macd_histogram", 0)),
        "adx": last_row.get("adx", 25),
        "atr": last_row.get("atr", 0),
        "bb_position": float(bb_position) if bb_position == bb_position else 0.5,
        "cvd": signal.get("cvd", signal.get("order_flow", {}).get("cvd", 0)),
        "orderbook_imbalance": signal.get("orderbook_imbalance", signal.get("order_flow", {}).get("orderbook_imbalance", 0.5)),
        "bullish_tf_count": signal.get("bullish_tf_count", 0),
        "bearish_tf_count": signal.get("bearish_tf_count", 0),
        "confidence": signal.get("confidence", 5),
        "trend_encoded": {"strong_bullish": 2, "bullish": 1, "neutral": 0, "bearish": -1, "strong_bearish": -2}.get(
            (signal.get("trend") or "neutral").lower(), 0
        ),
        "sentiment_encoded": {"bullish": 1, "neutral": 0, "bearish": -1}.get(
            (signal.get("sentiment") or "neutral").lower(), 0
        ),
        "signal_encoded": 1 if signal.get("signal") == "BUY" else 0,
        "risk_distance_pct": risk_distance(),
        "reward_distance_pct": reward_distance(),
        "risk_reward_ratio": risk_reward(),
    }


async def _backfill_ml_and_lstm(signal: dict, ml_validator, lstm_validator, backtest_engine) -> dict:
    """
    Se o sinal não tiver ml_prediction ou lstm_probability, busca candles no passado,
    passa na ML e no LSTM e preenche no próprio dict (in-place).
    """
    sig_time = _parse_signal_time(signal.get("timestamp", ""))
    symbol = signal.get("symbol", "BTCUSDT")
    need_ml = signal.get("ml_prediction") is None
    need_lstm = signal.get("lstm_probability") is None
    if not need_ml and not need_lstm:
        return signal

    start = sig_time - timedelta(hours=80)  # ~3+ dias 1h para ter sequence_length + indicadores
    end = sig_time + timedelta(hours=2)

    try:
        df = await backtest_engine.fetch_data(symbol, "1h", start, end)
    except Exception as e:
        print(f"  [AVISO] Falha ao buscar candles para {symbol} @ {sig_time}: {e}")
        return signal

    if df is None or df.empty or len(df) < 60:
        return signal

    df = backtest_engine.calculate_indicators(df)
    last_row = df.iloc[-1].to_dict() if len(df) > 0 else {}

    if need_ml and ml_validator is not None:
        try:
            features = _build_ml_features_from_signal_and_candles(signal, last_row)
            res = ml_validator.predict_signal(features)
            if "error" not in res:
                signal["ml_prediction"] = res.get("prediction", 0)
                signal["ml_probability"] = res.get("probability_success", 0.5)
        except Exception as e:
            print(f"  [ML] Erro para {symbol}: {e}")

    if need_lstm and lstm_validator is not None:
        try:
            # LSTM espera as colunas do backtest (incl. volume_ratio)
            res = lstm_validator.predict_from_candles(df)
            if res.get("error"):
                pass
            else:
                signal["lstm_probability"] = res.get("probability", 0.5)
        except Exception as e:
            print(f"  [LSTM] Erro para {symbol}: {e}")

    return signal


async def run_backfill(signals_dir: str = "signals", limit: int = None):
    """
    Carrega todos os sinais, preenche ML/LSTM quando faltar, reavalia com Binance e regista no log.
    """
    print("Carregando sinais...")
    signals = load_all_signals(signals_dir)
    buy_sell = [s for s in signals if s.get("signal") in ("BUY", "SELL")]
    if limit:
        buy_sell = buy_sell[:limit]
    print(f"  Total BUY/SELL: {len(buy_sell)}")

    ml_validator = None
    lstm_validator = None
    backtest_engine = None

    try:
        from src.ml.simple_validator import SimpleSignalValidator
        ml_validator = SimpleSignalValidator()
        ml_validator.load_models()
        print("  ML (sklearn) carregado.")
    except Exception as e:
        print(f"  ML não disponível: {e}")

    try:
        from src.ml.lstm_sequence_validator import LSTMSequenceValidator
        lstm_validator = LSTMSequenceValidator()
        if lstm_validator.load_model():
            print("  Bi-LSTM carregado.")
        else:
            lstm_validator = None
    except Exception as e:
        print(f"  LSTM não disponível: {e}")

    try:
        from src.backtesting.backtest_engine import BacktestEngine
        backtest_engine = BacktestEngine()
    except Exception as e:
        print(f"  BacktestEngine não disponível: {e}")
        return

    processed = 0
    for i, sig in enumerate(buy_sell):
        symbol = sig.get("symbol", "?")
        ts = sig.get("timestamp", "")[:19]
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Processando {i+1}/{len(buy_sell)}: {symbol} @ {ts}")

        sig = await _backfill_ml_and_lstm(sig, ml_validator, lstm_validator, backtest_engine)
        evaluation = evaluate_signal(sig)
        if evaluation.get("outcome") in ("SL_HIT", "TP1_HIT", "TP2_HIT", "EXPIRED"):
            _append_model_votes_log(sig, evaluation)
            processed += 1

    print(f"\nRegistados no log: {processed} sinais finalizados.")
    report = get_system_accuracy_report()
    print("\n--- Acertividade por sistema ---")
    print(f"  Total registos (deduplicados): {report.get('total_records', 0)}")
    for src, d in report.get("by_source", {}).items():
        print(f"  {src}: {d.get('wins', 0)}/{d.get('total', 0)} acertos ({d.get('accuracy_pct', 0):.1f}%)")
    for direction, d in report.get("by_direction", {}).items():
        print(f"  {direction}: {d.get('wins', 0)}/{d.get('total', 0)} acertos ({d.get('accuracy_pct', 0):.1f}%)")
    ml_d = report.get("ml", {})
    lstm_d = report.get("lstm", {})
    print(f"  ML: {ml_d.get('correct', 0)}/{ml_d.get('total', 0)} ({(ml_d.get('accuracy_pct') or 0):.1f}%)")
    print(f"  LSTM: {lstm_d.get('correct', 0)}/{lstm_d.get('total', 0)} ({(lstm_d.get('accuracy_pct') or 0):.1f}%)")
    print(f"\nLog guardado em: {MODEL_VOTES_LOG}")


def main():
    import argparse
    p = argparse.ArgumentParser(description="Backfill: reavalia todos os sinais, regista ML/LSTM e acertividade.")
    p.add_argument("--signals-dir", default="signals", help="Pasta com ficheiros de sinal")
    p.add_argument("--limit", type=int, default=None, help="Máximo de sinais a processar (útil para teste)")
    args = p.parse_args()
    asyncio.run(run_backfill(signals_dir=args.signals_dir, limit=args.limit))


if __name__ == "__main__":
    main()
