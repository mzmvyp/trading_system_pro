"""
Backtest Dataset Generator - Gera dados de treino para LSTM a partir do backtest.
=================================================================================

Pipeline:
1. BacktestEngine roda sobre dados históricos → gera trades com resultado (win/loss)
2. Para cada trade, captura os últimos N candles (com indicadores) como sequência temporal
3. Label: 1 = trade vencedor (TP1/TP2), 0 = trade perdedor (SL/EXPIRED)
4. Salva sequências como dataset pronto para LSTM/Bi-LSTM

Vantagem: Gera milhares de trades rotulados sem esperar dados reais de produção.
"""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.backtesting.backtest_engine import BacktestEngine, BacktestParams, Trade
from src.core.logger import get_logger

logger = get_logger(__name__)

# Diretório de output
OUTPUT_DIR = Path("ml_dataset/backtest")


class BacktestDatasetGenerator:
    """
    Gera dataset de treino para LSTM usando trades do backtest.

    Cada amostra é uma sequência de N candles (com indicadores) antes do
    momento de entrada do trade, com label = win/loss.
    """

    def __init__(
        self,
        symbols: List[str] = None,
        interval: str = "1h",
        sequence_length: int = 60,
        days_back: int = 180,
        n_param_variations: int = 20,
    ):
        """
        Args:
            symbols: Lista de pares para gerar dados
            interval: Timeframe dos candles
            sequence_length: Quantos candles antes do trade compõem a sequência
            days_back: Quantos dias de histórico usar
            n_param_variations: Quantas variações de parâmetros testar
                               (mais variações = mais trades = mais dados)
        """
        self.symbols = symbols or ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        self.interval = interval
        self.sequence_length = sequence_length
        self.days_back = days_back
        self.n_param_variations = n_param_variations

        # Features que serão extraídas de cada candle na sequência
        self.feature_columns = [
            "close", "high", "low", "open", "volume",
            "rsi", "ema_fast", "ema_slow", "macd", "macd_signal", "macd_hist",
            "bb_upper", "bb_middle", "bb_lower", "adx", "atr",
            "volume_ratio",
        ]

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    async def generate(self) -> Dict:
        """
        Pipeline completo: buscar dados → rodar backtests → extrair sequências → salvar.

        Returns:
            Dict com estatísticas do dataset gerado
        """
        all_sequences = []  # (sequence_array, label, metadata)
        stats = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "symbols_processed": [],
            "param_variations": self.n_param_variations,
        }

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=self.days_back)

        for symbol in self.symbols:
            logger.info(f"[DATASET] Processando {symbol}...")
            print(f"[DATASET] Gerando trades para {symbol} ({self.days_back} dias)...")

            try:
                # 1. Buscar dados históricos uma vez
                engine = BacktestEngine()
                df = await engine.fetch_data(symbol, self.interval, start_time, end_time)
                if df.empty or len(df) < self.sequence_length + 50:
                    logger.warning(f"[DATASET] Dados insuficientes para {symbol}: {len(df)} candles")
                    continue

                # 2. Rodar backtest com múltiplas variações de parâmetros
                # Cada variação gera trades diferentes = mais dados de treino
                symbol_trades = 0
                for variation in range(self.n_param_variations):
                    params = self._random_params()
                    engine_var = BacktestEngine(params=params)

                    # Calcular indicadores e gerar sinais
                    df_with_indicators = engine_var.calculate_indicators(df.copy())
                    df_with_signals = engine_var.generate_signals(df_with_indicators)
                    trades = engine_var.simulate_trades(df_with_signals)

                    if not trades:
                        continue

                    # 3. Para cada trade, extrair sequência de candles antes da entrada
                    for trade in trades:
                        seq = self._extract_sequence(
                            df_with_indicators, trade, symbol
                        )
                        if seq is not None:
                            all_sequences.append(seq)
                            symbol_trades += 1

                            if trade.is_winner:
                                stats["winning_trades"] += 1
                            else:
                                stats["losing_trades"] += 1

                stats["total_trades"] += symbol_trades
                stats["symbols_processed"].append(symbol)
                print(f"  [OK] {symbol}: {symbol_trades} trades extraídos")

            except Exception as e:
                logger.error(f"[DATASET] Erro processando {symbol}: {e}")
                print(f"  [ERRO] {symbol}: {e}")

        if not all_sequences:
            print("[ERRO] Nenhuma sequência gerada!")
            return stats

        # 4. Montar arrays numpy e salvar
        self._save_dataset(all_sequences, stats)

        return stats

    def _random_params(self) -> BacktestParams:
        """Gera parâmetros aleatórios para diversificar os trades."""
        import random

        return BacktestParams(
            rsi_period=random.choice([10, 14, 21]),
            rsi_oversold=random.uniform(20, 35),
            rsi_overbought=random.uniform(65, 80),
            ema_fast=random.choice([9, 12, 20]),
            ema_slow=random.choice([26, 50, 100]),
            macd_fast=random.choice([8, 12, 16]),
            macd_slow=random.choice([21, 26, 30]),
            macd_signal=random.choice([7, 9, 12]),
            bb_period=random.choice([15, 20, 25]),
            bb_std=random.uniform(1.5, 2.5),
            adx_period=random.choice([10, 14, 20]),
            adx_min_strength=random.uniform(15, 30),
            volume_ma_period=random.choice([14, 20, 30]),
            volume_surge_multiplier=random.uniform(1.2, 2.0),
            min_confluence=random.choice([2, 3, 4]),
            sl_atr_multiplier=random.uniform(1.0, 2.0),
            tp1_atr_multiplier=random.uniform(2.0, 4.0),
            tp2_atr_multiplier=random.uniform(4.0, 6.0),
            tp1_close_pct=random.uniform(0.3, 0.7),
        )

    def _extract_sequence(
        self, df: pd.DataFrame, trade: Trade, symbol: str
    ) -> Optional[Tuple[np.ndarray, int, Dict]]:
        """
        Extrai sequência de N candles antes da entrada do trade.

        Returns:
            (sequence_array [seq_len, n_features], label, metadata) ou None
        """
        # Encontrar o índice da entrada do trade
        entry_idx = None
        if trade.entry_time is not None:
            # Buscar candle mais próximo do entry_time
            if "timestamp" in df.columns:
                time_diffs = abs(pd.to_datetime(df["timestamp"]) - pd.Timestamp(trade.entry_time))
                entry_idx = time_diffs.idxmin()
            else:
                # Fallback: buscar pelo preço de entrada
                price_diffs = abs(df["close"] - trade.entry_price)
                entry_idx = price_diffs.idxmin()

        if entry_idx is None:
            return None

        # Precisamos de seq_len candles ANTES da entrada
        start_idx = entry_idx - self.sequence_length
        if start_idx < 0:
            return None

        # Extrair janela de candles
        window = df.iloc[start_idx:entry_idx]
        if len(window) < self.sequence_length:
            return None

        # Extrair features disponíveis
        available_features = [f for f in self.feature_columns if f in window.columns]
        if len(available_features) < 10:
            return None

        # Normalizar: retornos percentuais para preço/volume, valores absolutos para indicadores
        seq_data = window[available_features].copy()

        # Normalizar preços como % mudança em relação ao close do primeiro candle
        first_close = seq_data["close"].iloc[0]
        if first_close > 0:
            for col in ["close", "high", "low", "open"]:
                if col in seq_data.columns:
                    seq_data[col] = (seq_data[col] / first_close - 1) * 100

            # Normalizar BB, EMA em relação ao primeiro close também
            for col in ["ema_fast", "ema_slow", "bb_upper", "bb_middle", "bb_lower"]:
                if col in seq_data.columns:
                    seq_data[col] = (seq_data[col] / first_close - 1) * 100

            # ATR como % do preço
            if "atr" in seq_data.columns:
                seq_data["atr"] = seq_data["atr"] / first_close * 100

        # Volume como ratio (já normalizado se volume_ratio existe)
        if "volume" in seq_data.columns:
            vol_mean = seq_data["volume"].mean()
            if vol_mean > 0:
                seq_data["volume"] = seq_data["volume"] / vol_mean

        # RSI já está 0-100, ADX já está 0-100, MACD/hist mantém escala relativa

        # Converter para numpy array
        sequence = seq_data.values.astype(np.float32)

        # Tratar NaN
        sequence = np.nan_to_num(sequence, nan=0.0)

        # Label: 1 = winner, 0 = loser
        label = 1 if trade.is_winner else 0

        # Metadata para debug/análise
        metadata = {
            "symbol": symbol,
            "direction": trade.direction,
            "entry_price": trade.entry_price,
            "exit_reason": trade.exit_reason,
            "pnl_pct": trade.pnl_pct,
            "entry_time": str(trade.entry_time),
        }

        return (sequence, label, metadata)

    def _save_dataset(self, sequences: List[Tuple], stats: Dict):
        """Salva dataset em formato numpy + metadata."""
        # Separar arrays, labels e metadata
        X_list = [s[0] for s in sequences]
        y_list = [s[1] for s in sequences]
        meta_list = [s[2] for s in sequences]

        # Verificar shapes consistentes
        expected_shape = X_list[0].shape
        valid = [(x, y, m) for x, y, m in zip(X_list, y_list, meta_list) if x.shape == expected_shape]

        if len(valid) < len(sequences):
            logger.warning(f"[DATASET] {len(sequences) - len(valid)} sequências descartadas por shape inconsistente")

        X = np.array([v[0] for v in valid], dtype=np.float32)
        y = np.array([v[1] for v in valid], dtype=np.int32)
        meta = [v[2] for v in valid]

        # Split temporal (80/20) - NÃO aleatório para evitar leakage
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        _ = (meta[:split_idx], meta[split_idx:])  # meta splits available if needed

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        # Salvar arrays
        np.save(OUTPUT_DIR / f"X_train_{timestamp}.npy", X_train)
        np.save(OUTPUT_DIR / f"X_test_{timestamp}.npy", X_test)
        np.save(OUTPUT_DIR / f"y_train_{timestamp}.npy", y_train)
        np.save(OUTPUT_DIR / f"y_test_{timestamp}.npy", y_test)

        # Symlinks para latest
        for name in ["X_train", "X_test", "y_train", "y_test"]:
            src = OUTPUT_DIR / f"{name}_{timestamp}.npy"
            dst = OUTPUT_DIR / f"{name}_latest.npy"
            if dst.exists():
                dst.unlink()
            import shutil
            shutil.copy(src, dst)

        # Metadata
        dataset_info = {
            "timestamp": timestamp,
            "sequence_length": self.sequence_length,
            "n_features": X.shape[2] if len(X.shape) == 3 else 0,
            "feature_columns": [f for f in self.feature_columns if f in self.feature_columns],
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
        print("DATASET GERADO COM SUCESSO")
        print(f"{'='*60}")
        print(f"  Shape: {X.shape} (samples, sequence_len, features)")
        print(f"  Train: {len(X_train)} amostras (win rate: {y_train.mean()*100:.1f}%)")
        print(f"  Test:  {len(X_test)} amostras (win rate: {y_test.mean()*100:.1f}%)")
        print(f"  Dir:   {OUTPUT_DIR}")
        print(f"{'='*60}")


async def main():
    """Gera dataset de treino a partir do backtest."""
    generator = BacktestDatasetGenerator(
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        interval="1h",
        sequence_length=60,
        days_back=180,
        n_param_variations=20,
    )
    stats = await generator.generate()
    print(f"\nResumo: {stats}")


if __name__ == "__main__":
    asyncio.run(main())
