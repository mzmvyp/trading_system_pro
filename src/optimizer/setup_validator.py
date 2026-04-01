"""
Setup Validator — Valida setups contra dados históricos reais
==============================================================

Inspirado no ContinuousBacktestValidator do bot_trade_20260115.

Diferente do optimizer (que busca MELHORES parâmetros), este módulo:
1. Gera MUITOS sinais em dados históricos (centenas por par)
2. Valida cada sinal contra as candles futuras reais
3. Classifica cada sinal por CONTEXTO (RSI zone, ADX strength, trend, etc.)
4. Calcula win rate POR SETUP/CONTEXTO
5. Na hora de executar um trade real, verifica: "este tipo de setup
   historicamente funciona ou não?"

Uso no sistema:
- Roda periodicamente (junto com o optimizer ou separado)
- Alimenta backtest_results/setup_statistics.json
- Agent consulta antes de executar: validate_signal_before_trade()
- Setup com win rate > 55% = voto FOR, < 40% = voto AGAINST
"""

import asyncio
import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.backtesting.backtest_engine import BacktestEngine, BacktestParams
from src.core.logger import get_logger

logger = get_logger(__name__)

RESULTS_DIR = Path("backtest_results")
STATS_FILE = RESULTS_DIR / "setup_statistics.json"
HISTORY_FILE = RESULTS_DIR / "backtest_history.json"

DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
]


@dataclass
class ValidatedSignal:
    """Um sinal gerado em dados históricos e validado contra o futuro."""
    symbol: str
    timestamp: str
    signal_type: str  # BUY ou SELL
    setup_key: str    # ex: "BUY_oversold_moderate_bullish_medium"

    # Indicadores no momento do sinal
    rsi: float
    macd_hist: float
    adx: float
    atr: float
    bb_position: float
    trend: str
    volatility: str

    # Preços
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float

    # Resultado
    result: str = "PENDING"  # TP1, TP2, SL, TIMEOUT
    exit_price: float = 0.0
    pnl_pct: float = 0.0
    hours_to_exit: float = 0.0


class SetupValidator:
    """
    Valida setups contra dados históricos.
    Gera centenas de sinais, valida cada um, e calcula stats por setup.
    """

    def __init__(self):
        self.statistics: Dict[str, Dict] = {}
        self.history: List[Dict] = []
        RESULTS_DIR.mkdir(exist_ok=True)
        self._load_statistics()

    def _load_statistics(self):
        if STATS_FILE.exists():
            try:
                with open(STATS_FILE, "r") as f:
                    self.statistics = json.load(f)
                logger.info(f"[SETUP_VALIDATOR] Estatísticas carregadas: {len(self.statistics)} setups")
            except Exception:
                self.statistics = {}

    def _save_statistics(self):
        with open(STATS_FILE, "w") as f:
            json.dump(self.statistics, f, indent=2, default=str)

    def _save_history(self):
        # Manter últimos 10000
        if len(self.history) > 10000:
            self.history = self.history[-10000:]
        with open(HISTORY_FILE, "w") as f:
            json.dump(self.history, f, indent=2, default=str)

    @staticmethod
    def classify_context(rsi: float, adx: float, trend: str, volatility_pct: float) -> Tuple[str, str, str, str]:
        """Classifica o contexto de mercado em categorias."""
        # RSI
        if rsi < 30:
            rsi_cat = "oversold"
        elif rsi < 40:
            rsi_cat = "low"
        elif rsi < 60:
            rsi_cat = "neutral"
        elif rsi < 70:
            rsi_cat = "high"
        else:
            rsi_cat = "overbought"

        # ADX
        if adx < 20:
            adx_cat = "weak"
        elif adx < 40:
            adx_cat = "moderate"
        else:
            adx_cat = "strong"

        # Trend normalizado
        t = trend.lower() if trend else "neutral"
        if "bullish" in t:
            trend_cat = "bullish"
        elif "bearish" in t:
            trend_cat = "bearish"
        else:
            trend_cat = "neutral"

        # Volatilidade
        if volatility_pct < 1.0:
            vol_cat = "low"
        elif volatility_pct < 2.5:
            vol_cat = "medium"
        else:
            vol_cat = "high"

        return rsi_cat, adx_cat, trend_cat, vol_cat

    @staticmethod
    def make_setup_key(signal_type: str, rsi_cat: str, adx_cat: str,
                       trend_cat: str, vol_cat: str) -> str:
        return f"{signal_type}_{rsi_cat}_{adx_cat}_{trend_cat}_{vol_cat}"

    def generate_and_validate(self, df: pd.DataFrame, symbol: str,
                              params: Optional[BacktestParams] = None) -> List[ValidatedSignal]:
        """
        Gera sinais em dados históricos e valida contra candles futuras.

        Diferente do optimizer: não busca MELHORES params,
        gera MUITOS sinais e classifica por CONTEXTO.
        """
        params = params or BacktestParams()
        bt = BacktestEngine(params=params)

        # Calcular indicadores
        df = bt.calculate_indicators(df.copy())
        if df.empty:
            return []

        # Gerar sinais
        df = bt.generate_signals(df)

        signals = []
        signal_rows = df[df["signal"].isin(["BUY", "SELL"])].copy()

        for idx, row in signal_rows.iterrows():
            i = df.index.get_loc(idx)

            # Precisamos de pelo menos 48 barras futuras para validar
            if i >= len(df) - 48:
                continue

            entry_price = row["close"]
            atr = row.get("atr", 0)
            if pd.isna(atr) or atr <= 0:
                continue

            signal_type = row["signal"]

            # Classificar contexto
            rsi = row.get("rsi", 50)
            adx = row.get("adx", 25)
            macd_hist = row.get("macd_hist", 0)

            # Trend
            if row["close"] > row.get("ema_fast", row["close"]) > row.get("ema_slow", row["close"]):
                trend = "bullish"
            elif row["close"] < row.get("ema_fast", row["close"]) < row.get("ema_slow", row["close"]):
                trend = "bearish"
            else:
                trend = "neutral"

            # Volatilidade
            vol_pct = (atr / entry_price * 100) if entry_price > 0 else 1

            # BB position
            bb_range = row.get("bb_upper", 0) - row.get("bb_lower", 0)
            bb_pos = (row["close"] - row.get("bb_lower", 0)) / bb_range if bb_range > 0 else 0.5

            rsi_cat, adx_cat, trend_cat, vol_cat = self.classify_context(rsi, adx, trend, vol_pct)
            setup_key = self.make_setup_key(signal_type, rsi_cat, adx_cat, trend_cat, vol_cat)

            # SL/TP
            if signal_type == "BUY":
                sl = entry_price - atr * params.sl_atr_multiplier
                tp1 = entry_price + atr * params.tp1_atr_multiplier
                tp2 = entry_price + atr * params.tp2_atr_multiplier
            else:
                sl = entry_price + atr * params.sl_atr_multiplier
                tp1 = entry_price - atr * params.tp1_atr_multiplier
                tp2 = entry_price - atr * params.tp2_atr_multiplier

            sig = ValidatedSignal(
                symbol=symbol,
                timestamp=str(row.get("timestamp", "")),
                signal_type=signal_type,
                setup_key=setup_key,
                rsi=float(rsi) if not pd.isna(rsi) else 50,
                macd_hist=float(macd_hist) if not pd.isna(macd_hist) else 0,
                adx=float(adx) if not pd.isna(adx) else 25,
                atr=float(atr),
                bb_position=float(bb_pos),
                trend=trend,
                volatility=vol_cat,
                entry_price=float(entry_price),
                stop_loss=float(sl),
                take_profit_1=float(tp1),
                take_profit_2=float(tp2),
            )

            # Validar contra dados futuros
            sig = self._validate_signal(df, i, sig)
            if sig.result != "PENDING":
                signals.append(sig)
                self._update_stats(sig)

        return signals

    def _validate_signal(self, df: pd.DataFrame, signal_idx: int,
                         sig: ValidatedSignal) -> ValidatedSignal:
        """Valida um sinal contra as próximas 48 barras."""
        future = df.iloc[signal_idx + 1: signal_idx + 49]
        if future.empty:
            return sig

        tp1_hit = False
        for j, (_, row) in enumerate(future.iterrows()):
            h, l = row["high"], row["low"]

            if sig.signal_type == "BUY":
                if l <= sig.stop_loss:
                    sig.result = "SL"
                    sig.exit_price = sig.stop_loss
                    sig.pnl_pct = (sig.stop_loss - sig.entry_price) / sig.entry_price * 100
                    sig.hours_to_exit = j + 1
                    return sig
                if h >= sig.take_profit_2:
                    sig.result = "TP2"
                    sig.exit_price = sig.take_profit_2
                    sig.pnl_pct = (sig.take_profit_2 - sig.entry_price) / sig.entry_price * 100
                    sig.hours_to_exit = j + 1
                    return sig
                if h >= sig.take_profit_1 and not tp1_hit:
                    tp1_hit = True
                    sig.result = "TP1"
                    sig.exit_price = sig.take_profit_1
                    sig.pnl_pct = (sig.take_profit_1 - sig.entry_price) / sig.entry_price * 100
                    sig.hours_to_exit = j + 1
            else:  # SELL
                if h >= sig.stop_loss:
                    sig.result = "SL"
                    sig.exit_price = sig.stop_loss
                    sig.pnl_pct = (sig.entry_price - sig.stop_loss) / sig.entry_price * 100
                    sig.hours_to_exit = j + 1
                    return sig
                if l <= sig.take_profit_2:
                    sig.result = "TP2"
                    sig.exit_price = sig.take_profit_2
                    sig.pnl_pct = (sig.entry_price - sig.take_profit_2) / sig.entry_price * 100
                    sig.hours_to_exit = j + 1
                    return sig
                if l <= sig.take_profit_1 and not tp1_hit:
                    tp1_hit = True
                    sig.result = "TP1"
                    sig.exit_price = sig.take_profit_1
                    sig.pnl_pct = (sig.entry_price - sig.take_profit_1) / sig.entry_price * 100
                    sig.hours_to_exit = j + 1

        if sig.result == "PENDING":
            sig.result = "TIMEOUT"
            last_close = future.iloc[-1]["close"]
            if sig.signal_type == "BUY":
                sig.pnl_pct = (last_close - sig.entry_price) / sig.entry_price * 100
            else:
                sig.pnl_pct = (sig.entry_price - last_close) / sig.entry_price * 100
            sig.exit_price = last_close
            sig.hours_to_exit = 48

        return sig

    def _update_stats(self, sig: ValidatedSignal):
        """Atualiza estatísticas do setup."""
        key = sig.setup_key
        if key not in self.statistics:
            self.statistics[key] = {
                "total": 0, "tp1": 0, "tp2": 0, "sl": 0, "timeout": 0,
                "total_pnl": 0.0, "win_rate": 0.0, "avg_pnl": 0.0,
                "avg_hours": 0.0, "last_updated": "",
            }

        s = self.statistics[key]
        s["total"] += 1
        s[sig.result.lower()] = s.get(sig.result.lower(), 0) + 1
        s["total_pnl"] += sig.pnl_pct
        s["avg_hours"] = (s["avg_hours"] * (s["total"] - 1) + sig.hours_to_exit) / s["total"]

        wins = s.get("tp1", 0) + s.get("tp2", 0)
        s["win_rate"] = wins / s["total"] * 100 if s["total"] > 0 else 0
        s["avg_pnl"] = s["total_pnl"] / s["total"]
        s["last_updated"] = datetime.now(timezone.utc).isoformat()

        self.history.append(asdict(sig))

    async def run_full_validation(
        self, symbols: List[str] = None, days_back: int = 90
    ) -> Dict[str, Any]:
        """
        Roda validação completa: gera sinais em todos os pares e valida.

        Diferente do optimizer: não busca parâmetros melhores.
        Gera MUITOS sinais e classifica quais CONTEXTOS funcionam.
        """
        symbols = symbols or DEFAULT_SYMBOLS
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days_back)

        engine = BacktestEngine()
        total_signals = 0
        results = {}

        for symbol in symbols:
            try:
                df = await engine.fetch_data(symbol, "1h", start_time, end_time)
                if df.empty or len(df) < 100:
                    continue

                signals = self.generate_and_validate(df, symbol)
                total_signals += len(signals)

                wins = sum(1 for s in signals if s.result in ("TP1", "TP2"))
                wr = wins / len(signals) * 100 if signals else 0

                results[symbol] = {
                    "signals": len(signals),
                    "wins": wins,
                    "win_rate": round(wr, 1),
                }
                logger.info(
                    f"[SETUP_VALIDATOR] {symbol}: {len(signals)} sinais, "
                    f"WR={wr:.1f}%"
                )
            except Exception as e:
                logger.error(f"[SETUP_VALIDATOR] {symbol}: {e}")

        self._save_statistics()
        self._save_history()

        logger.info(
            f"[SETUP_VALIDATOR] Validação completa: {total_signals} sinais, "
            f"{len(self.statistics)} setups rastreados"
        )

        return {
            "total_signals": total_signals,
            "total_setups": len(self.statistics),
            "per_symbol": results,
        }

    def validate_incoming_signal(self, signal_data: Dict) -> Dict:
        """
        Valida um sinal ANTES de executar, consultando estatísticas históricas.

        Chamado pelo agent.py como double-check.
        Retorna: recomendação + win rate histórico do setup.
        """
        rsi = signal_data.get("rsi", 50)
        adx = signal_data.get("adx", 25)
        trend = signal_data.get("trend", "neutral")
        atr = signal_data.get("atr", 0)
        entry = signal_data.get("entry_price", 1)
        vol_pct = (atr / entry * 100) if entry > 0 else 1
        signal_type = signal_data.get("signal", "BUY")

        rsi_cat, adx_cat, trend_cat, vol_cat = self.classify_context(rsi, adx, trend, vol_pct)
        setup_key = self.make_setup_key(signal_type, rsi_cat, adx_cat, trend_cat, vol_cat)

        if setup_key in self.statistics:
            stats = self.statistics[setup_key]
            total = stats.get("total", 0)
            wr = stats.get("win_rate", 0)

            if total < 15:
                vote = 0
                recommendation = "INSUFFICIENT_DATA"
            elif wr >= 60:
                vote = 1
                recommendation = "STRONG_APPROVE"
            elif wr >= 50:
                vote = 0
                recommendation = "APPROVE"
            elif wr >= 40:
                vote = 0
                recommendation = "NEUTRAL"
            else:
                vote = -1
                recommendation = "AVOID"

            return {
                "setup_key": setup_key,
                "recommendation": recommendation,
                "vote": vote,
                "historical_win_rate": wr,
                "historical_avg_pnl": stats.get("avg_pnl", 0),
                "total_samples": total,
                "avg_hours_to_exit": stats.get("avg_hours", 0),
            }

        return {
            "setup_key": setup_key,
            "recommendation": "NEW_SETUP",
            "vote": 0,
            "message": "Setup ainda não testado historicamente",
        }

    def get_best_setups(self, min_samples: int = 15) -> List[Dict]:
        """Retorna melhores setups ordenados por win_rate * avg_pnl."""
        best = []
        for key, stats in self.statistics.items():
            if stats.get("total", 0) >= min_samples:
                best.append({
                    "setup": key,
                    "win_rate": stats["win_rate"],
                    "avg_pnl": stats["avg_pnl"],
                    "total": stats["total"],
                    "avg_hours": stats.get("avg_hours", 0),
                })
        best.sort(key=lambda x: x["win_rate"] * max(0.01, x["avg_pnl"]), reverse=True)
        return best

    def get_worst_setups(self, min_samples: int = 15) -> List[Dict]:
        """Retorna piores setups (para evitar)."""
        worst = []
        for key, stats in self.statistics.items():
            if stats.get("total", 0) >= min_samples and stats["win_rate"] < 45:
                worst.append({
                    "setup": key,
                    "win_rate": stats["win_rate"],
                    "avg_pnl": stats["avg_pnl"],
                    "total": stats["total"],
                })
        worst.sort(key=lambda x: x["win_rate"])
        return worst


# Instância global
_validator: Optional[SetupValidator] = None


def get_setup_validator() -> SetupValidator:
    global _validator
    if _validator is None:
        _validator = SetupValidator()
    return _validator


def validate_signal_before_trade(signal_data: Dict) -> Dict:
    """Helper: valida sinal antes de executar."""
    return get_setup_validator().validate_incoming_signal(signal_data)
