"""
Continuous Backtest Validator - Sistema de Validacao Continua de Setups
=======================================================================

Este sistema roda continuamente em background e:
1. Coleta dados historicos de multiplos simbolos
2. Simula sinais baseados nos mesmos indicadores que o DeepSeek usa
3. Valida cada sinal contra dados reais do Binance
4. Calcula estatisticas de acerto por setup/condicao de mercado
5. Identifica os melhores setups para o momento atual
6. Serve como double-check antes de executar trades

Uso (na raiz do trading_system_pro):
    python continuous_backtest_validator.py --symbols BTCUSDT,ETHUSDT --interval 300
    python continuous_backtest_validator.py --all  # Todos os TOP10

Dados historicos: src.exchange.client.BinanceClient (Futures API publica).

Autor: Trading Bot
Data: 2026-01-19
"""

import os
import sys
import json
import asyncio
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict
import time

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Configuracoes (paths relativos a este repo)
_RESULTS = _ROOT / "backtest_results"
CONFIG = {
    "results_dir": str(_RESULTS),
    "stats_file": str(_RESULTS / "setup_statistics.json"),
    "history_file": str(_RESULTS / "backtest_history.json"),
    "default_interval_seconds": 300,  # 5 minutos
    "lookback_days": 30,  # Dias de dados historicos para analise
    "validation_hours": 48,  # Horas para validar se o sinal deu certo
    "min_samples_for_stats": 20,  # Minimo de amostras para estatisticas confiaveis
    "top_symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT",
                    "XRPUSDT", "DOTUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT"]
}


@dataclass
class SignalSetup:
    """Representa um setup de sinal com suas caracteristicas"""
    symbol: str
    timestamp: str
    signal_type: str  # BUY ou SELL

    # Indicadores
    rsi: float
    macd_histogram: float
    adx: float
    atr: float
    bb_position: float  # 0-1, posicao nas bandas de bollinger

    # Contexto de mercado
    trend: str  # bullish, bearish, neutral
    volatility: str  # low, medium, high

    # Precos
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float

    # Resultado (preenchido apos validacao)
    result: str = "PENDING"  # TP1, TP2, SL, TIMEOUT, PENDING
    exit_price: float = 0.0
    pnl_percent: float = 0.0
    hours_to_exit: float = 0.0


class TechnicalIndicators:
    """Calcula indicadores tecnicos a partir de dados OHLCV"""

    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calcula RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calcula MACD, Signal Line e Histogram"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calcula ADX"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        return adx

    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calcula ATR"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    @staticmethod
    def calculate_bollinger_position(close: pd.Series, period: int = 20, std_dev: int = 2) -> pd.Series:
        """Calcula posicao relativa nas Bandas de Bollinger (0 = lower, 1 = upper)"""
        sma = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        position = (close - lower) / (upper - lower)
        return position.clip(0, 1)


class ContinuousBacktestValidator:
    """
    Sistema de validacao continua de setups via backtest.
    """

    def __init__(self):
        self.indicators = TechnicalIndicators()
        self.setup_statistics: Dict[str, Dict] = {}
        self.backtest_history: List[Dict] = []
        self.is_running = False

        # Criar diretorio de resultados
        Path(CONFIG["results_dir"]).mkdir(exist_ok=True)

        # Carregar dados existentes
        self._load_statistics()
        self._load_history()

    def _load_statistics(self):
        """Carrega estatisticas de setups"""
        if os.path.exists(CONFIG["stats_file"]):
            try:
                with open(CONFIG["stats_file"], 'r') as f:
                    self.setup_statistics = json.load(f)
                print(f"[BACKTEST] Estatisticas carregadas: {len(self.setup_statistics)} setups")
            except Exception as e:
                print(f"[BACKTEST] Erro ao carregar estatisticas: {e}")
                self.setup_statistics = {}

    def _save_statistics(self):
        """Salva estatisticas de setups"""
        with open(CONFIG["stats_file"], 'w') as f:
            json.dump(self.setup_statistics, f, indent=2, default=str)

    def _load_history(self):
        """Carrega historico de backtests"""
        if os.path.exists(CONFIG["history_file"]):
            try:
                with open(CONFIG["history_file"], 'r') as f:
                    self.backtest_history = json.load(f)
                print(f"[BACKTEST] Historico carregado: {len(self.backtest_history)} sinais validados")
            except Exception as e:
                print(f"[BACKTEST] Erro ao carregar historico: {e}")
                self.backtest_history = []

    def _save_history(self):
        """Salva historico de backtests"""
        # Manter apenas os ultimos 10000 registros
        if len(self.backtest_history) > 10000:
            self.backtest_history = self.backtest_history[-10000:]
        with open(CONFIG["history_file"], 'w') as f:
            json.dump(self.backtest_history, f, indent=2, default=str)

    async def get_historical_data(self, symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
        """Obtem dados historicos do Binance"""
        try:
            from src.exchange.client import BinanceClient

            end_dt = datetime.now(timezone.utc)
            start_dt = end_dt - timedelta(days=days)

            async with BinanceClient() as client:
                df = await client.get_historical_klines(symbol, "1h", start_dt, end_dt)

            if df is None or df.empty:
                print(f"[BACKTEST] Sem dados para {symbol}")
                return None

            return df

        except Exception as e:
            print(f"[BACKTEST] Erro ao obter dados para {symbol}: {e}")
            return None

    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula todos os indicadores tecnicos"""
        df = df.copy()

        # RSI
        df['rsi'] = self.indicators.calculate_rsi(df['close'])

        # MACD
        macd_line, signal_line, histogram = self.indicators.calculate_macd(df['close'])
        df['macd'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_histogram'] = histogram

        # ADX
        df['adx'] = self.indicators.calculate_adx(df['high'], df['low'], df['close'])

        # ATR
        df['atr'] = self.indicators.calculate_atr(df['high'], df['low'], df['close'])

        # Bollinger Position
        df['bb_position'] = self.indicators.calculate_bollinger_position(df['close'])

        # SMA para trend
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()

        # Volatilidade relativa
        df['volatility'] = df['atr'] / df['close'] * 100

        # Remover linhas com NaN
        df = df.dropna()

        return df

    def classify_market_context(self, row: pd.Series) -> Tuple[str, str]:
        """Classifica o contexto de mercado baseado nos indicadores"""
        # Trend
        if row['close'] > row['sma_20'] > row['sma_50']:
            trend = "bullish"
        elif row['close'] < row['sma_20'] < row['sma_50']:
            trend = "bearish"
        else:
            trend = "neutral"

        # Volatility
        vol = row['volatility']
        if vol < 1.0:
            volatility = "low"
        elif vol < 2.5:
            volatility = "medium"
        else:
            volatility = "high"

        return trend, volatility

    def generate_signals_from_data(self, df: pd.DataFrame, symbol: str) -> List[SignalSetup]:
        """Gera sinais a partir dos dados usando criterios MAIS REALISTAS"""
        signals = []

        for i in range(50, len(df) - 48):  # Deixa 48h para validacao
            row = df.iloc[i]

            # Classificar contexto
            trend, volatility = self.classify_market_context(row)

            # CORRIGIDO: Criterios MAIS REALISTAS para BUY
            # Baseado nos sinais reais que o DeepSeek gera
            buy_signal = (
                (
                    # Setup 1: Reversao de oversold
                    (row['rsi'] < 35 and row['macd_histogram'] > row['macd_histogram'] * 0.8) or
                    # Setup 2: Continuacao de tendencia
                    (row['rsi'] > 40 and row['rsi'] < 55 and row['macd_histogram'] > 0 and trend == "bullish") or
                    # Setup 3: Breakout
                    (row['bb_position'] > 0.7 and row['adx'] > 30 and trend == "bullish")
                ) and
                row['adx'] >= 20  # Regime detection: skip mercado lateral
            )

            # Criterios para SELL
            sell_signal = (
                (
                    # Setup 1: Reversao de overbought
                    (row['rsi'] > 65 and row['macd_histogram'] < row['macd_histogram'] * 0.8) or
                    # Setup 2: Continuacao de tendencia de baixa
                    (row['rsi'] > 45 and row['rsi'] < 60 and row['macd_histogram'] < 0 and trend == "bearish") or
                    # Setup 3: Breakdown
                    (row['bb_position'] < 0.3 and row['adx'] > 30 and trend == "bearish")
                ) and
                row['adx'] >= 20  # Regime detection: skip mercado lateral
            )

            if buy_signal or sell_signal:
                signal_type = "BUY" if buy_signal else "SELL"
                entry_price = row['close']
                atr = row['atr']

                # Calcular SL/TP usando os MESMOS parâmetros do bot real
                # Scalp: SL 0.3%, TP1 0.4%, TP2 0.8%
                # Day Trade: SL 0.6%, TP1 0.8%, TP2 1.5%
                atr_pct = (atr / entry_price) * 100 if entry_price > 0 else 1
                if atr_pct > 0.15:  # Alta vol -> scalp
                    sl_pct, tp1_pct, tp2_pct = 0.003, 0.004, 0.008
                else:  # Day trade
                    sl_pct, tp1_pct, tp2_pct = 0.006, 0.008, 0.015

                if signal_type == "BUY":
                    stop_loss = entry_price * (1 - sl_pct)
                    take_profit_1 = entry_price * (1 + tp1_pct)
                    take_profit_2 = entry_price * (1 + tp2_pct)
                else:
                    stop_loss = entry_price * (1 + sl_pct)
                    take_profit_1 = entry_price * (1 - tp1_pct)
                    take_profit_2 = entry_price * (1 - tp2_pct)

                signal = SignalSetup(
                    symbol=symbol,
                    timestamp=df.index[i].isoformat() if hasattr(df.index[i], 'isoformat') else str(df.index[i]),
                    signal_type=signal_type,
                    rsi=row['rsi'],
                    macd_histogram=row['macd_histogram'],
                    adx=row['adx'],
                    atr=atr,
                    bb_position=row['bb_position'],
                    trend=trend,
                    volatility=volatility,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit_1=take_profit_1,
                    take_profit_2=take_profit_2
                )

                signals.append((i, signal))

        return signals

    def validate_signal(self, df: pd.DataFrame, signal_idx: int, signal: SignalSetup) -> SignalSetup:
        """Valida um sinal contra dados futuros"""
        # Pegar proximas 48 horas de dados
        future_data = df.iloc[signal_idx + 1:signal_idx + 49]

        if future_data.empty:
            signal.result = "NO_DATA"
            return signal

        entry_price = signal.entry_price
        stop_loss = signal.stop_loss
        tp1 = signal.take_profit_1
        tp2 = signal.take_profit_2

        for j, (idx, row) in enumerate(future_data.iterrows()):
            high = row['high']
            low = row['low']

            if signal.signal_type == "BUY":
                # Verificar SL primeiro
                if low <= stop_loss:
                    signal.result = "SL"
                    signal.exit_price = stop_loss
                    signal.pnl_percent = (stop_loss - entry_price) / entry_price * 100
                    signal.hours_to_exit = j + 1
                    return signal

                # Verificar TP2
                if high >= tp2:
                    signal.result = "TP2"
                    signal.exit_price = tp2
                    signal.pnl_percent = (tp2 - entry_price) / entry_price * 100
                    signal.hours_to_exit = j + 1
                    return signal

                # Verificar TP1
                if high >= tp1 and signal.result != "TP1":
                    signal.result = "TP1"
                    signal.exit_price = tp1
                    signal.pnl_percent = (tp1 - entry_price) / entry_price * 100
                    signal.hours_to_exit = j + 1
                    # Continua para ver se atinge TP2

            else:  # SELL
                # Verificar SL primeiro
                if high >= stop_loss:
                    signal.result = "SL"
                    signal.exit_price = stop_loss
                    signal.pnl_percent = (entry_price - stop_loss) / entry_price * 100
                    signal.hours_to_exit = j + 1
                    return signal

                # Verificar TP2
                if low <= tp2:
                    signal.result = "TP2"
                    signal.exit_price = tp2
                    signal.pnl_percent = (entry_price - tp2) / entry_price * 100
                    signal.hours_to_exit = j + 1
                    return signal

                # Verificar TP1
                if low <= tp1 and signal.result != "TP1":
                    signal.result = "TP1"
                    signal.exit_price = tp1
                    signal.pnl_percent = (entry_price - tp1) / entry_price * 100
                    signal.hours_to_exit = j + 1

        # Se nao atingiu nenhum target
        if signal.result == "PENDING":
            signal.result = "TIMEOUT"
            signal.exit_price = future_data.iloc[-1]['close']
            if signal.signal_type == "BUY":
                signal.pnl_percent = (signal.exit_price - entry_price) / entry_price * 100
            else:
                signal.pnl_percent = (entry_price - signal.exit_price) / entry_price * 100
            signal.hours_to_exit = 48

        return signal

    def get_setup_key(self, signal: SignalSetup) -> str:
        """Gera chave unica para identificar um tipo de setup"""
        # Categorizar RSI
        if signal.rsi < 30:
            rsi_cat = "oversold"
        elif signal.rsi < 40:
            rsi_cat = "low"
        elif signal.rsi < 60:
            rsi_cat = "neutral"
        elif signal.rsi < 70:
            rsi_cat = "high"
        else:
            rsi_cat = "overbought"

        # Categorizar ADX
        if signal.adx < 20:
            adx_cat = "weak"
        elif signal.adx < 40:
            adx_cat = "moderate"
        else:
            adx_cat = "strong"

        # CORRIGIDO: Normalizar trend para 3 categorias (bullish/bearish/neutral)
        trend = signal.trend.lower() if signal.trend else "neutral"
        if "bullish" in trend:
            trend = "bullish"
        elif "bearish" in trend:
            trend = "bearish"
        else:
            trend = "neutral"

        # CORRIGIDO: Normalizar volatility para 3 categorias (low/medium/high)
        volatility = signal.volatility.lower() if signal.volatility else "medium"
        if volatility not in ("low", "medium", "high"):
            volatility = "medium"

        return f"{signal.signal_type}_{rsi_cat}_{adx_cat}_{trend}_{volatility}"

    def update_statistics(self, signal: SignalSetup):
        """Atualiza estatisticas do setup"""
        key = self.get_setup_key(signal)

        if key not in self.setup_statistics:
            self.setup_statistics[key] = {
                "total": 0,
                "tp1": 0,
                "tp2": 0,
                "sl": 0,
                "timeout": 0,
                "total_pnl": 0.0,
                "avg_hours_to_exit": 0.0,
                "win_rate": 0.0,
                "avg_pnl": 0.0,
                "last_updated": ""
            }

        stats = self.setup_statistics[key]
        stats["total"] += 1

        if signal.result == "TP1":
            stats["tp1"] += 1
        elif signal.result == "TP2":
            stats["tp2"] += 1
        elif signal.result == "SL":
            stats["sl"] += 1
        elif signal.result == "TIMEOUT":
            stats["timeout"] += 1

        stats["total_pnl"] += signal.pnl_percent
        stats["avg_hours_to_exit"] = (
            (stats["avg_hours_to_exit"] * (stats["total"] - 1) + signal.hours_to_exit) / stats["total"]
        )

        wins = stats["tp1"] + stats["tp2"]
        stats["win_rate"] = wins / stats["total"] * 100 if stats["total"] > 0 else 0
        stats["avg_pnl"] = stats["total_pnl"] / stats["total"] if stats["total"] > 0 else 0
        stats["last_updated"] = datetime.now().isoformat()

        self._save_statistics()

    async def run_backtest_for_symbol(self, symbol: str) -> Dict:
        """Executa backtest completo para um simbolo"""
        print(f"\n[BACKTEST] Processando {symbol}...")

        # Obter dados historicos
        df = await self.get_historical_data(symbol, days=CONFIG["lookback_days"])

        if df is None or len(df) < 100:
            return {"symbol": symbol, "error": "Dados insuficientes"}

        # Calcular indicadores
        df = self.calculate_all_indicators(df)

        # Gerar sinais
        signals_with_idx = self.generate_signals_from_data(df, symbol)
        print(f"[BACKTEST] {symbol}: {len(signals_with_idx)} sinais gerados")

        # Validar cada sinal
        results = {
            "symbol": symbol,
            "total_signals": len(signals_with_idx),
            "tp1": 0,
            "tp2": 0,
            "sl": 0,
            "timeout": 0,
            "total_pnl": 0.0,
            "signals": []
        }

        for idx, signal in signals_with_idx:
            validated_signal = self.validate_signal(df, idx, signal)

            # Atualizar contadores
            if validated_signal.result == "TP1":
                results["tp1"] += 1
            elif validated_signal.result == "TP2":
                results["tp2"] += 1
            elif validated_signal.result == "SL":
                results["sl"] += 1
            elif validated_signal.result == "TIMEOUT":
                results["timeout"] += 1

            results["total_pnl"] += validated_signal.pnl_percent

            # Atualizar estatisticas globais
            self.update_statistics(validated_signal)

            # Adicionar ao historico
            self.backtest_history.append(asdict(validated_signal))

            results["signals"].append(asdict(validated_signal))

        # Calcular metricas
        if results["total_signals"] > 0:
            results["win_rate"] = (results["tp1"] + results["tp2"]) / results["total_signals"] * 100
            results["avg_pnl"] = results["total_pnl"] / results["total_signals"]
        else:
            results["win_rate"] = 0
            results["avg_pnl"] = 0

        return results

    def get_best_setups(self, min_samples: int = None) -> List[Dict]:
        """Retorna os melhores setups baseado nas estatisticas"""
        min_samples = min_samples or CONFIG["min_samples_for_stats"]

        best_setups = []
        for key, stats in self.setup_statistics.items():
            if stats["total"] >= min_samples:
                best_setups.append({
                    "setup": key,
                    "win_rate": stats["win_rate"],
                    "avg_pnl": stats["avg_pnl"],
                    "total_samples": stats["total"],
                    "avg_hours_to_exit": stats["avg_hours_to_exit"]
                })

        # Ordenar por win_rate * avg_pnl (score combinado)
        best_setups.sort(key=lambda x: x["win_rate"] * max(0.01, x["avg_pnl"]), reverse=True)

        return best_setups

    def validate_incoming_signal(self, signal_data: Dict) -> Dict:
        """
        Valida um sinal recebido contra as estatisticas historicas.
        Serve como DOUBLE-CHECK antes de executar um trade.

        Returns:
            Dict com recomendacao e estatisticas do setup
        """
        # Criar SignalSetup a partir dos dados recebidos
        indicators = signal_data.get('indicators', {})

        signal = SignalSetup(
            symbol=signal_data.get('symbol', ''),
            timestamp=datetime.now().isoformat(),
            signal_type=signal_data.get('signal', 'BUY'),
            rsi=signal_data.get('rsi', indicators.get('rsi', 50)),
            macd_histogram=signal_data.get('macd_histogram', indicators.get('macd_histogram', 0)),
            adx=signal_data.get('adx', indicators.get('adx', 25)),
            atr=signal_data.get('atr', indicators.get('atr', 0)),
            bb_position=signal_data.get('bb_position', indicators.get('bb_position', 0.5)),
            trend=signal_data.get('trend', 'neutral'),
            volatility=signal_data.get('volatility', 'medium'),
            entry_price=signal_data.get('entry_price', 0),
            stop_loss=signal_data.get('stop_loss', 0),
            take_profit_1=signal_data.get('take_profit_1', 0),
            take_profit_2=signal_data.get('take_profit_2', 0)
        )

        setup_key = self.get_setup_key(signal)

        if setup_key in self.setup_statistics:
            stats = self.setup_statistics[setup_key]

            # Determinar recomendacao
            if stats["total"] < CONFIG["min_samples_for_stats"]:
                recommendation = "INSUFFICIENT_DATA"
                confidence = 0.5
            elif stats["win_rate"] >= 65 and stats["avg_pnl"] > 0:
                recommendation = "STRONG_BUY"
                confidence = min(0.95, stats["win_rate"] / 100)
            elif stats["win_rate"] >= 55 and stats["avg_pnl"] > 0:
                recommendation = "BUY"
                confidence = stats["win_rate"] / 100
            elif stats["win_rate"] >= 45:
                recommendation = "NEUTRAL"
                confidence = 0.5
            elif stats["win_rate"] >= 35:
                recommendation = "WEAK"
                confidence = 1 - (stats["win_rate"] / 100)
            else:
                recommendation = "AVOID"
                confidence = 1 - (stats["win_rate"] / 100)

            return {
                "setup_key": setup_key,
                "recommendation": recommendation,
                "confidence": confidence,
                "historical_win_rate": stats["win_rate"],
                "historical_avg_pnl": stats["avg_pnl"],
                "total_samples": stats["total"],
                "avg_hours_to_exit": stats["avg_hours_to_exit"],
                "tp1_rate": stats["tp1"] / stats["total"] * 100 if stats["total"] > 0 else 0,
                "tp2_rate": stats["tp2"] / stats["total"] * 100 if stats["total"] > 0 else 0,
                "sl_rate": stats["sl"] / stats["total"] * 100 if stats["total"] > 0 else 0
            }
        else:
            return {
                "setup_key": setup_key,
                "recommendation": "NEW_SETUP",
                "confidence": 0.5,
                "message": "Este setup ainda nao foi testado historicamente"
            }

    async def run_continuous(self, symbols: List[str], interval: int = None):
        """Executa backtest continuo em loop"""
        interval = interval or CONFIG["default_interval_seconds"]
        self.is_running = True

        print("\n" + "=" * 70)
        print("CONTINUOUS BACKTEST VALIDATOR - INICIANDO")
        print("=" * 70)
        print(f"Simbolos: {symbols}")
        print(f"Intervalo: {interval} segundos")
        print("=" * 70)

        cycle = 0
        while self.is_running:
            cycle += 1
            print(f"\n[CICLO {cycle}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            try:
                # Processar cada simbolo
                all_results = []
                for symbol in symbols:
                    result = await self.run_backtest_for_symbol(symbol)
                    all_results.append(result)

                # Salvar historico
                self._save_history()

                # Mostrar resumo
                self._print_cycle_summary(all_results)

                # Mostrar melhores setups
                self._print_best_setups()

            except Exception as e:
                print(f"[ERRO] Erro no ciclo {cycle}: {e}")
                import traceback
                traceback.print_exc()

            # Aguardar proximo ciclo
            print(f"\n[AGUARDANDO] Proximo ciclo em {interval} segundos...")
            await asyncio.sleep(interval)

    def _print_cycle_summary(self, results: List[Dict]):
        """Imprime resumo do ciclo"""
        print("\n" + "-" * 50)
        print("RESUMO DO CICLO")
        print("-" * 50)

        for r in results:
            if "error" in r:
                print(f"{r['symbol']}: {r['error']}")
            else:
                print(f"{r['symbol']}: {r['total_signals']} sinais | "
                      f"Win: {r.get('win_rate', 0):.1f}% | "
                      f"PnL: {r.get('avg_pnl', 0):+.2f}%")

    def _print_best_setups(self):
        """Imprime os melhores setups"""
        best = self.get_best_setups()

        if not best:
            print("\n[INFO] Ainda nao ha setups suficientes para ranking")
            return

        print("\n" + "-" * 50)
        print("TOP 5 MELHORES SETUPS (baseado em dados historicos)")
        print("-" * 50)

        for i, setup in enumerate(best[:5], 1):
            print(f"{i}. {setup['setup']}")
            print(f"   Win Rate: {setup['win_rate']:.1f}% | "
                  f"Avg PnL: {setup['avg_pnl']:+.2f}% | "
                  f"Amostras: {setup['total_samples']}")

    def stop(self):
        """Para a execucao continua"""
        self.is_running = False
        print("[BACKTEST] Parando validador continuo...")

    def get_statistics_summary(self) -> Dict:
        """Retorna resumo das estatisticas"""
        total_setups = len(self.setup_statistics)
        total_samples = sum(s["total"] for s in self.setup_statistics.values())

        avg_win_rate = np.mean([s["win_rate"] for s in self.setup_statistics.values()]) if self.setup_statistics else 0
        avg_pnl = np.mean([s["avg_pnl"] for s in self.setup_statistics.values()]) if self.setup_statistics else 0

        return {
            "total_setups_tracked": total_setups,
            "total_signals_validated": total_samples,
            "average_win_rate": avg_win_rate,
            "average_pnl": avg_pnl,
            "best_setups": self.get_best_setups()[:10],
            "history_size": len(self.backtest_history)
        }


# Instancia global para uso no bot
backtest_validator = ContinuousBacktestValidator()


async def initialize_backtest_data_if_needed(symbols: List[str] = None) -> Dict:
    """
    NOVO: Inicializa dados de backtest se nao existirem.
    Chamado automaticamente quando o bot inicia.

    Args:
        symbols: Lista de simbolos para backtest. Se None, usa top 5.

    Returns:
        Dict com status da inicializacao
    """
    symbols = symbols or CONFIG["top_symbols"][:5]

    # Verificar se ja tem dados
    stats = backtest_validator.get_statistics_summary()

    if stats["total_signals_validated"] >= 100:
        print(f"[BACKTEST] Dados existentes: {stats['total_signals_validated']} sinais validados")
        return {"status": "already_initialized", "signals": stats["total_signals_validated"]}

    print("[BACKTEST] Inicializando dados de backtest (primeira execucao)...")
    print(f"[BACKTEST] Simbolos: {symbols}")

    total_signals = 0
    for symbol in symbols:
        try:
            result = await backtest_validator.run_backtest_for_symbol(symbol)
            total_signals += result.get("total_signals", 0)
            print(f"[BACKTEST] {symbol}: {result.get('total_signals', 0)} sinais | "
                  f"Win: {result.get('win_rate', 0):.1f}%")
        except Exception as e:
            print(f"[BACKTEST] Erro ao processar {symbol}: {e}")

    backtest_validator._save_history()

    return {
        "status": "initialized",
        "signals": total_signals,
        "symbols_processed": len(symbols)
    }


def validate_signal_before_trade(signal_data: Dict) -> Dict:
    """
    Funcao helper para validar um sinal antes de executar o trade.
    Chamada pelo bot como double-check.
    """
    return backtest_validator.validate_incoming_signal(signal_data)


def get_backtest_statistics() -> Dict:
    """Retorna estatisticas do backtest"""
    return backtest_validator.get_statistics_summary()


async def main():
    """Funcao principal"""
    os.chdir(_ROOT)

    parser = argparse.ArgumentParser(description="Continuous Backtest Validator")
    parser.add_argument("--symbols", type=str, help="Simbolos separados por virgula (ex: BTCUSDT,ETHUSDT)")
    parser.add_argument("--all", action="store_true", help="Usar todos os TOP10 simbolos")
    parser.add_argument("--interval", type=int, default=300, help="Intervalo entre ciclos em segundos")
    parser.add_argument("--once", action="store_true", help="Executar apenas uma vez (sem loop)")
    parser.add_argument("--stats", action="store_true", help="Mostrar apenas estatisticas atuais")

    args = parser.parse_args()

    validator = ContinuousBacktestValidator()

    # Modo estatisticas
    if args.stats:
        summary = validator.get_statistics_summary()
        print("\n" + "=" * 70)
        print("ESTATISTICAS DO BACKTEST VALIDATOR")
        print("=" * 70)
        print(f"Total de setups rastreados: {summary['total_setups_tracked']}")
        print(f"Total de sinais validados: {summary['total_signals_validated']}")
        print(f"Win rate medio: {summary['average_win_rate']:.1f}%")
        print(f"PnL medio: {summary['average_pnl']:+.2f}%")

        print("\n" + "-" * 50)
        print("TOP 10 MELHORES SETUPS")
        print("-" * 50)
        for i, setup in enumerate(summary['best_setups'], 1):
            print(f"{i}. {setup['setup']}")
            print(f"   Win: {setup['win_rate']:.1f}% | PnL: {setup['avg_pnl']:+.2f}% | N={setup['total_samples']}")
        return

    # Definir simbolos
    if args.all:
        symbols = CONFIG["top_symbols"]
    elif args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        symbols = CONFIG["top_symbols"][:3]  # Default: top 3

    # Modo unico ou continuo
    if args.once:
        print("\n[BACKTEST] Executando uma vez...")
        for symbol in symbols:
            result = await validator.run_backtest_for_symbol(symbol)
            print(f"\n{symbol}: {result.get('total_signals', 0)} sinais | "
                  f"Win: {result.get('win_rate', 0):.1f}% | "
                  f"PnL: {result.get('avg_pnl', 0):+.2f}%")
        validator._save_history()
        validator._print_best_setups()
    else:
        try:
            await validator.run_continuous(symbols, args.interval)
        except KeyboardInterrupt:
            validator.stop()
            print("\n[BACKTEST] Finalizado pelo usuario")


if __name__ == "__main__":
    asyncio.run(main())
