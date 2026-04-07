"""
Real Signal Dataset Generator — Gera dados de treino para LSTM a partir de sinais REAIS.
========================================================================================

Em vez de usar trades sintéticos do backtest engine, usa os 6000+ sinais reais
armazenados em signals/ com outcomes verificados (TP/SL) contra o mercado.

Pipeline:
1. Carrega sinais reais de signals/ (excluindo blacklist)
2. Para cada sinal com outcome (TP1_HIT, TP2_HIT, SL_HIT):
   a. Busca os últimos N candles de 1h antes do timestamp do sinal
   b. Calcula indicadores técnicos
   c. Extrai sequência de features normalizada
   d. Label = 1 se TP hit, 0 se SL hit
3. Salva no mesmo formato que backtest_dataset_generator (compatível com LSTM)
"""

import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.core.config import settings
from src.core.logger import get_logger
from src.trading.signal_tracker import evaluate_signal, load_all_signals

logger = get_logger(__name__)

OUTPUT_DIR = Path("ml_dataset/backtest")
SIGNALS_DIR = "signals"


class RealSignalDatasetGenerator:
    """Gera dataset de treino para LSTM usando sinais reais com outcomes verificados."""

    def __init__(
        self,
        sequence_length: int = 60,
        max_signals: int = 5000,
    ):
        self.sequence_length = sequence_length
        self.max_signals = max_signals
        self.feature_columns = [
            "close", "high", "low", "open", "volume",
            "rsi", "ema_fast", "ema_slow", "macd", "macd_signal", "macd_hist",
            "bb_upper", "bb_middle", "bb_lower", "adx", "atr",
            "volume_ratio",
        ]
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    async def generate(self) -> Dict:
        """Pipeline completo: carrega sinais reais, busca klines, gera sequências."""
        from src.backtesting.backtest_engine import BacktestEngine

        blacklist = set(getattr(settings, 'token_blacklist', []))

        print("[REAL] Carregando sinais reais...")
        all_signals = load_all_signals(SIGNALS_DIR)

        valid_signals = [
            s for s in all_signals
            if s.get("signal") in ("BUY", "SELL")
            and s.get("symbol") not in blacklist
            and s.get("source") != "LOCAL_GEN"
            and s.get("entry_price", 0) > 0
            and s.get("stop_loss", 0) > 0
        ]

        print(f"[REAL] {len(valid_signals)} sinais válidos (de {len(all_signals)} total)")

        # Evaluate outcomes
        print("[REAL] Avaliando outcomes contra mercado real...")
        evaluated = []
        for i, sig in enumerate(valid_signals[:self.max_signals]):
            if i % 100 == 0 and i > 0:
                print(f"  ... {i}/{min(len(valid_signals), self.max_signals)} avaliados ({len(evaluated)} com outcome)")
            try:
                result = evaluate_signal(sig)
                outcome = result.get("outcome", "PENDING")
                if outcome in ("TP1_HIT", "TP2_HIT", "SL_HIT"):
                    sig["_outcome"] = outcome
                    sig["_is_winner"] = outcome in ("TP1_HIT", "TP2_HIT")
                    evaluated.append(sig)
            except Exception:
                continue

        print(f"[REAL] {len(evaluated)} sinais com outcome confirmado")
        if len(evaluated) < 50:
            return {"total_trades": len(evaluated), "error": "insufficient_signals"}

        # Sort by timestamp for temporal split
        evaluated.sort(key=lambda s: s.get("timestamp", ""))

        # Extract sequences
        print("[REAL] Extraindo sequências de candles...")
        engine = BacktestEngine()
        sequences = []
        errors = 0

        for i, sig in enumerate(evaluated):
            if i % 50 == 0 and i > 0:
                print(f"  ... {i}/{len(evaluated)} sequências ({len(sequences)} OK, {errors} erros)")
            try:
                seq = await self._extract_sequence_for_signal(sig, engine)
                if seq is not None:
                    sequences.append(seq)
                else:
                    errors += 1
            except Exception:
                errors += 1
                continue

        print(f"[REAL] {len(sequences)} sequências extraídas ({errors} erros)")

        if len(sequences) < 50:
            return {"total_trades": len(sequences), "error": "insufficient_sequences"}

        stats = {
            "total_trades": len(sequences),
            "winning_trades": sum(1 for s in sequences if s[1] == 1),
            "losing_trades": sum(1 for s in sequences if s[1] == 0),
            "data_source": "real_signals",
            "symbols_processed": list(set(s[2].get("symbol", "") for s in sequences)),
        }

        self._save_dataset(sequences, stats)
        return stats

    async def _extract_sequence_for_signal(
        self, signal: Dict, engine
    ) -> Optional[Tuple[np.ndarray, int, Dict]]:
        """Extrai sequência de N candles antes do timestamp do sinal."""
        symbol = signal.get("symbol", "")
        timestamp_str = signal.get("timestamp", "")

        if not symbol or not timestamp_str:
            return None

        try:
            if 'T' in timestamp_str:
                ts = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            else:
                ts = datetime.strptime(timestamp_str[:19], '%Y-%m-%d %H:%M:%S')
                ts = ts.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            return None

        hours_needed = self.sequence_length + 20
        start_time = ts - timedelta(hours=hours_needed)
        end_time = ts

        try:
            df = await engine.fetch_data(symbol, "1h", start_time, end_time)
        except Exception:
            return None

        if df.empty or len(df) < self.sequence_length:
            return None

        df = engine.calculate_indicators(df)

        # Get available features
        available = [f for f in self.feature_columns if f in df.columns]
        if len(available) < 10:
            return None

        # Take last sequence_length candles
        window = df.tail(self.sequence_length)
        if len(window) < self.sequence_length:
            return None

        seq_data = window[available].copy()

        # Normalize prices relative to first close
        first_close = seq_data["close"].iloc[0]
        if first_close > 0:
            for col in ["close", "high", "low", "open"]:
                if col in seq_data.columns:
                    seq_data[col] = (seq_data[col] / first_close - 1) * 100
            for col in ["ema_fast", "ema_slow", "bb_upper", "bb_middle", "bb_lower"]:
                if col in seq_data.columns:
                    seq_data[col] = (seq_data[col] / first_close - 1) * 100
            if "atr" in seq_data.columns:
                seq_data["atr"] = seq_data["atr"] / first_close * 100

        if "volume" in seq_data.columns:
            vol_mean = seq_data["volume"].mean()
            if vol_mean > 0:
                seq_data["volume"] = seq_data["volume"] / vol_mean

        sequence = seq_data.values.astype(np.float32)
        sequence = np.nan_to_num(sequence, nan=0.0)

        label = 1 if signal["_is_winner"] else 0

        metadata = {
            "symbol": symbol,
            "direction": signal.get("signal", ""),
            "entry_price": signal.get("entry_price", 0),
            "exit_reason": signal["_outcome"],
            "source": signal.get("source", ""),
            "entry_time": timestamp_str,
        }

        return (sequence, label, metadata)

    def _save_dataset(self, sequences: List[Tuple], stats: Dict):
        """Salva dataset no mesmo formato do backtest_dataset_generator."""
        import shutil

        expected_shape = sequences[0][0].shape
        valid = [(x, y, m) for x, y, m in sequences if x.shape == expected_shape]

        if len(valid) < len(sequences):
            logger.warning(f"[REAL] {len(sequences) - len(valid)} sequências descartadas (shape)")

        # Already sorted by timestamp above
        X = np.array([v[0] for v in valid], dtype=np.float32)
        y = np.array([v[1] for v in valid], dtype=np.int32)
        meta = [v[2] for v in valid]

        # 80/20 temporal split
        split_idx = int(len(X) * 0.8)
        X_train_raw, X_test = X[:split_idx], X[split_idx:]
        y_train_raw, y_test = y[:split_idx], y[split_idx:]

        # BALANCEAR treino: igualar wins e losses
        idx_win = np.where(y_train_raw == 1)[0]
        idx_loss = np.where(y_train_raw == 0)[0]
        n_min = min(len(idx_win), len(idx_loss))
        if n_min >= 10 and abs(len(idx_win) - len(idx_loss)) > n_min * 0.2:
            rng = np.random.RandomState(42)
            if len(idx_win) > n_min:
                idx_win = rng.choice(idx_win, size=n_min, replace=False)
            if len(idx_loss) > n_min:
                idx_loss = rng.choice(idx_loss, size=n_min, replace=False)
            balanced_idx = np.sort(np.concatenate([idx_win, idx_loss]))
            X_train = X_train_raw[balanced_idx]
            y_train = y_train_raw[balanced_idx]
            print(f"[BALANCE] Train: {len(idx_win)} wins + {len(idx_loss)} losses = {len(X_train)} balanceado")
        else:
            X_train = X_train_raw
            y_train = y_train_raw

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        np.save(OUTPUT_DIR / f"X_train_{timestamp}.npy", X_train)
        np.save(OUTPUT_DIR / f"X_test_{timestamp}.npy", X_test)
        np.save(OUTPUT_DIR / f"y_train_{timestamp}.npy", y_train)
        np.save(OUTPUT_DIR / f"y_test_{timestamp}.npy", y_test)

        for name in ["X_train", "X_test", "y_train", "y_test"]:
            src = OUTPUT_DIR / f"{name}_{timestamp}.npy"
            dst = OUTPUT_DIR / f"{name}_latest.npy"
            if dst.exists():
                dst.unlink()
            shutil.copy(src, dst)

        dataset_info = {
            "timestamp": timestamp,
            "data_source": "real_signals",
            "sequence_length": X.shape[1] if len(X.shape) == 3 else self.sequence_length,
            "n_features": X.shape[2] if len(X.shape) == 3 else 0,
            "feature_columns": self.feature_columns,
            "total_samples": len(X),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "win_rate_train": float(y_train.mean()) if len(y_train) > 0 else 0,
            "win_rate_test": float(y_test.mean()) if len(y_test) > 0 else 0,
            "shape": list(X.shape),
            "stats": stats,
        }

        with open(OUTPUT_DIR / f"dataset_info_{timestamp}.json", "w") as f:
            json.dump(dataset_info, f, indent=2, default=str)
        with open(OUTPUT_DIR / "dataset_info_latest.json", "w") as f:
            json.dump(dataset_info, f, indent=2, default=str)

        print(f"\n{'='*60}")
        print("DATASET DE SINAIS REAIS GERADO")
        print(f"{'='*60}")
        print(f"  Shape: {X.shape}")
        print(f"  Train: {len(X_train)} (WR: {y_train.mean()*100:.1f}%)")
        print(f"  Test:  {len(X_test)} (WR: {y_test.mean()*100:.1f}%)")
        print(f"  Dir:   {OUTPUT_DIR}")
        print(f"{'='*60}")


async def main():
    gen = RealSignalDatasetGenerator(sequence_length=60, max_signals=5000)
    stats = await gen.generate()
    print(f"\nResumo: {stats}")


if __name__ == "__main__":
    asyncio.run(main())
