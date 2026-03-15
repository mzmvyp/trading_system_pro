"""
Backtest Engine - Simulação de trades em dados históricos
=========================================================

Origem: Baseado na especificação do repo sinais/backtesting + smart_trading_system/reports.
Usa dados via BinanceClient (get_historical_klines) e indicadores TA-Lib.

Responsabilidades:
- Buscar dados históricos da Binance
- Calcular indicadores técnicos (RSI, MACD, EMA, BB, ADX)
- Gerar sinais baseados em parâmetros configuráveis
- Simular execução de trades (entry, SL, TP1, TP2)
- Calcular métricas: win rate, return, Sharpe, max drawdown
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import talib

from src.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestParams:
    """Parâmetros configuráveis para o backtest (espaço de otimização)."""

    # RSI
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0

    # EMA
    ema_fast: int = 20
    ema_slow: int = 50

    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0

    # ADX (força de tendência)
    adx_period: int = 14
    adx_min_strength: float = 20.0

    # Volume
    volume_ma_period: int = 20
    volume_surge_multiplier: float = 1.5

    # Confiança mínima (número de condições que devem alinhar)
    min_confluence: int = 3

    # Risk management
    sl_atr_multiplier: float = 1.5
    tp1_atr_multiplier: float = 3.0
    tp2_atr_multiplier: float = 5.0
    tp1_close_pct: float = 0.5  # fechar 50% no TP1


@dataclass
class Trade:
    """Representa um trade simulado."""
    entry_time: datetime
    exit_time: Optional[datetime] = None
    direction: str = ""  # BUY ou SELL
    entry_price: float = 0.0
    exit_price: float = 0.0
    stop_loss: float = 0.0
    take_profit_1: float = 0.0
    take_profit_2: float = 0.0
    exit_reason: str = ""  # SL_HIT, TP1_HIT, TP2_HIT, EXPIRED
    pnl_pct: float = 0.0
    is_winner: bool = False


@dataclass
class BacktestMetrics:
    """Métricas calculadas do backtest."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_return_pct: float = 0.0
    avg_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    profit_factor: float = 0.0
    avg_winner_pct: float = 0.0
    avg_loser_pct: float = 0.0
    max_consecutive_losses: int = 0
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)


class BacktestEngine:
    """
    Motor de backtesting que simula trades em dados históricos.

    Usa indicadores TA-Lib para gerar sinais e simula execução
    barra-a-barra com SL/TP/expiração.
    """

    def __init__(self, params: Optional[BacktestParams] = None):
        self.params = params or BacktestParams()

    async def fetch_data(self, symbol: str, interval: str,
                         start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Busca dados históricos da Binance."""
        from src.exchange.client import BinanceClient

        async with BinanceClient() as client:
            df = await client.get_historical_klines(symbol, interval, start_time, end_time)

        if df.empty:
            logger.warning(f"No data fetched for {symbol} {interval} {start_time}-{end_time}")
            return df

        df = df.reset_index()
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        logger.info(f"Fetched {len(df)} candles for {symbol} @ {interval}")
        return df

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula todos os indicadores técnicos necessários."""
        close = np.asarray(df['close'].values, dtype=np.float64)
        high = np.asarray(df['high'].values, dtype=np.float64)
        low = np.asarray(df['low'].values, dtype=np.float64)
        volume = np.asarray(df['volume'].values, dtype=np.float64)
        p = self.params

        # RSI
        df['rsi'] = talib.RSI(close, timeperiod=p.rsi_period)

        # EMAs
        df['ema_fast'] = talib.EMA(close, timeperiod=p.ema_fast)
        df['ema_slow'] = talib.EMA(close, timeperiod=p.ema_slow)

        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
            close, fastperiod=p.macd_fast, slowperiod=p.macd_slow, signalperiod=p.macd_signal
        )

        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
            close, timeperiod=p.bb_period, nbdevup=p.bb_std, nbdevdn=p.bb_std
        )

        # ADX
        df['adx'] = talib.ADX(high, low, close, timeperiod=p.adx_period)

        # ATR (para SL/TP)
        df['atr'] = talib.ATR(high, low, close, timeperiod=14)

        # Volume MA
        df['volume_ma'] = talib.SMA(volume, timeperiod=p.volume_ma_period)
        df['volume_ratio'] = df['volume'] / df['volume_ma'].replace(0, np.nan)

        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Gera sinais BUY/SELL baseado nos indicadores e parâmetros.
        Usa sistema de confluência: conta quantas condições se alinham.
        """
        p = self.params
        df['signal'] = 'NONE'
        df['confluence_score'] = 0

        for i in range(len(df)):
            if pd.isna(df.loc[i, 'rsi']) or pd.isna(df.loc[i, 'ema_fast']):
                continue

            bull_score = 0
            bear_score = 0

            # 1. RSI
            rsi = df.loc[i, 'rsi']
            if rsi < p.rsi_oversold + 10:  # zona oversold → bullish
                bull_score += 1
            elif rsi > p.rsi_overbought - 10:  # zona overbought → bearish
                bear_score += 1

            # 2. EMA trend
            if df.loc[i, 'close'] > df.loc[i, 'ema_fast'] > df.loc[i, 'ema_slow']:
                bull_score += 1
            elif df.loc[i, 'close'] < df.loc[i, 'ema_fast'] < df.loc[i, 'ema_slow']:
                bear_score += 1

            # 3. MACD
            macd_hist = df.loc[i, 'macd_hist']
            if not pd.isna(macd_hist):
                if macd_hist > 0:
                    bull_score += 1
                elif macd_hist < 0:
                    bear_score += 1

            # 4. Bollinger Bands position
            bb_range = df.loc[i, 'bb_upper'] - df.loc[i, 'bb_lower']
            if bb_range > 0:
                bb_pos = (df.loc[i, 'close'] - df.loc[i, 'bb_lower']) / bb_range
                if bb_pos < 0.3:  # perto da banda inferior
                    bull_score += 1
                elif bb_pos > 0.7:  # perto da banda superior
                    bear_score += 1

            # 5. Volume surge
            vol_ratio = df.loc[i, 'volume_ratio']
            if not pd.isna(vol_ratio) and vol_ratio >= p.volume_surge_multiplier:
                # Volume confirma a direção dominante
                if bull_score > bear_score:
                    bull_score += 1
                elif bear_score > bull_score:
                    bear_score += 1

            # 6. ADX (trend strength)
            adx = df.loc[i, 'adx']
            has_trend = not pd.isna(adx) and adx >= p.adx_min_strength

            # Gerar sinal se confluência suficiente E tendência presente
            if has_trend and bull_score >= p.min_confluence:
                df.loc[i, 'signal'] = 'BUY'
                df.loc[i, 'confluence_score'] = bull_score
            elif has_trend and bear_score >= p.min_confluence:
                df.loc[i, 'signal'] = 'SELL'
                df.loc[i, 'confluence_score'] = bear_score

        return df

    def simulate_trades(self, df: pd.DataFrame, max_hold_bars: int = 48) -> List[Trade]:
        """
        Simula a execução dos sinais barra-a-barra.
        Verifica SL, TP1, TP2 e expiração por tempo.
        """
        p = self.params
        trades: List[Trade] = []
        in_trade = False
        current_trade: Optional[Trade] = None
        tp1_hit = False

        for i in range(len(df)):
            if in_trade and current_trade:
                row = df.iloc[i]
                bars_held = i - entry_bar

                if current_trade.direction == 'BUY':
                    # Check SL
                    if row['low'] <= current_trade.stop_loss:
                        pnl = (current_trade.stop_loss - current_trade.entry_price) / current_trade.entry_price * 100
                        if tp1_hit:
                            pnl = pnl * (1 - p.tp1_close_pct) + tp1_pnl * p.tp1_close_pct
                        current_trade.exit_price = current_trade.stop_loss
                        current_trade.exit_reason = 'SL_HIT'
                        current_trade.pnl_pct = pnl
                        current_trade.exit_time = row.get('timestamp', None)
                        current_trade.is_winner = pnl > 0
                        trades.append(current_trade)
                        in_trade = False
                        continue

                    # Check TP1
                    if not tp1_hit and row['high'] >= current_trade.take_profit_1:
                        tp1_pnl = (current_trade.take_profit_1 - current_trade.entry_price) / current_trade.entry_price * 100
                        tp1_hit = True
                        # Move SL to breakeven
                        current_trade.stop_loss = current_trade.entry_price

                    # Check TP2
                    if row['high'] >= current_trade.take_profit_2:
                        pnl_tp2 = (current_trade.take_profit_2 - current_trade.entry_price) / current_trade.entry_price * 100
                        if tp1_hit:
                            pnl = tp1_pnl * p.tp1_close_pct + pnl_tp2 * (1 - p.tp1_close_pct)
                        else:
                            pnl = pnl_tp2
                        current_trade.exit_price = current_trade.take_profit_2
                        current_trade.exit_reason = 'TP2_HIT'
                        current_trade.pnl_pct = pnl
                        current_trade.exit_time = row.get('timestamp', None)
                        current_trade.is_winner = True
                        trades.append(current_trade)
                        in_trade = False
                        continue

                elif current_trade.direction == 'SELL':
                    # Check SL
                    if row['high'] >= current_trade.stop_loss:
                        pnl = (current_trade.entry_price - current_trade.stop_loss) / current_trade.entry_price * 100
                        if tp1_hit:
                            pnl = pnl * (1 - p.tp1_close_pct) + tp1_pnl * p.tp1_close_pct
                        current_trade.exit_price = current_trade.stop_loss
                        current_trade.exit_reason = 'SL_HIT'
                        current_trade.pnl_pct = pnl
                        current_trade.exit_time = row.get('timestamp', None)
                        current_trade.is_winner = pnl > 0
                        trades.append(current_trade)
                        in_trade = False
                        continue

                    # Check TP1
                    if not tp1_hit and row['low'] <= current_trade.take_profit_1:
                        tp1_pnl = (current_trade.entry_price - current_trade.take_profit_1) / current_trade.entry_price * 100
                        tp1_hit = True
                        current_trade.stop_loss = current_trade.entry_price

                    # Check TP2
                    if row['low'] <= current_trade.take_profit_2:
                        pnl_tp2 = (current_trade.entry_price - current_trade.take_profit_2) / current_trade.entry_price * 100
                        if tp1_hit:
                            pnl = tp1_pnl * p.tp1_close_pct + pnl_tp2 * (1 - p.tp1_close_pct)
                        else:
                            pnl = pnl_tp2
                        current_trade.exit_price = current_trade.take_profit_2
                        current_trade.exit_reason = 'TP2_HIT'
                        current_trade.pnl_pct = pnl
                        current_trade.exit_time = row.get('timestamp', None)
                        current_trade.is_winner = True
                        trades.append(current_trade)
                        in_trade = False
                        continue

                # Expiração por tempo
                if bars_held >= max_hold_bars:
                    close_price = row['close']
                    if current_trade.direction == 'BUY':
                        pnl = (close_price - current_trade.entry_price) / current_trade.entry_price * 100
                    else:
                        pnl = (current_trade.entry_price - close_price) / current_trade.entry_price * 100
                    if tp1_hit:
                        pnl = pnl * (1 - p.tp1_close_pct) + tp1_pnl * p.tp1_close_pct
                    current_trade.exit_price = close_price
                    current_trade.exit_reason = 'EXPIRED'
                    current_trade.pnl_pct = pnl
                    current_trade.exit_time = row.get('timestamp', None)
                    current_trade.is_winner = pnl > 0
                    trades.append(current_trade)
                    in_trade = False
                    continue

            # Abrir trade novo se não estamos em um
            if not in_trade and df.iloc[i].get('signal', 'NONE') != 'NONE':
                row = df.iloc[i]
                atr = row.get('atr', 0)
                if pd.isna(atr) or atr <= 0:
                    continue

                direction = row['signal']
                entry_price = row['close']
                entry_bar = i
                tp1_hit = False
                tp1_pnl = 0.0

                if direction == 'BUY':
                    sl = entry_price - atr * p.sl_atr_multiplier
                    tp1 = entry_price + atr * p.tp1_atr_multiplier
                    tp2 = entry_price + atr * p.tp2_atr_multiplier
                else:  # SELL
                    sl = entry_price + atr * p.sl_atr_multiplier
                    tp1 = entry_price - atr * p.tp1_atr_multiplier
                    tp2 = entry_price - atr * p.tp2_atr_multiplier

                current_trade = Trade(
                    entry_time=row.get('timestamp', datetime.now(timezone.utc)),
                    direction=direction,
                    entry_price=entry_price,
                    stop_loss=sl,
                    take_profit_1=tp1,
                    take_profit_2=tp2,
                )
                in_trade = True

        # Fechar trade aberto no final
        if in_trade and current_trade:
            last_row = df.iloc[-1]
            close_price = last_row['close']
            if current_trade.direction == 'BUY':
                pnl = (close_price - current_trade.entry_price) / current_trade.entry_price * 100
            else:
                pnl = (current_trade.entry_price - close_price) / current_trade.entry_price * 100
            current_trade.exit_price = close_price
            current_trade.exit_reason = 'END_OF_DATA'
            current_trade.pnl_pct = pnl
            current_trade.exit_time = last_row.get('timestamp', None)
            current_trade.is_winner = pnl > 0
            trades.append(current_trade)

        return trades

    def calculate_metrics(self, trades: List[Trade]) -> BacktestMetrics:
        """Calcula métricas de performance a partir dos trades."""
        if not trades:
            return BacktestMetrics()

        total = len(trades)
        winners = [t for t in trades if t.is_winner]
        losers = [t for t in trades if not t.is_winner]
        pnls = [t.pnl_pct for t in trades]

        win_rate = len(winners) / total * 100 if total > 0 else 0
        total_return = sum(pnls)
        avg_return = np.mean(pnls)
        avg_winner = np.mean([t.pnl_pct for t in winners]) if winners else 0
        avg_loser = np.mean([t.pnl_pct for t in losers]) if losers else 0

        # Equity curve e max drawdown
        equity = [100.0]  # começa com 100%
        for pnl in pnls:
            equity.append(equity[-1] * (1 + pnl / 100))

        peak = equity[0]
        max_dd = 0.0
        for val in equity:
            if val > peak:
                peak = val
            dd = (peak - val) / peak * 100
            if dd > max_dd:
                max_dd = dd

        # Sharpe ratio (anualizado, assumindo ~365 trades/ano como proxy)
        if len(pnls) >= 2 and np.std(pnls) > 0:
            sharpe = np.mean(pnls) / np.std(pnls, ddof=1) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Sortino ratio
        downside = [p for p in pnls if p < 0]
        if downside and np.std(downside, ddof=1) > 0:
            sortino = np.mean(pnls) / np.std(downside, ddof=1) * np.sqrt(252)
        else:
            sortino = 0.0

        # Profit factor
        total_gains = sum(t.pnl_pct for t in winners)
        total_losses = abs(sum(t.pnl_pct for t in losers))
        profit_factor = total_gains / total_losses if total_losses > 0 else float('inf')

        # Max consecutive losses
        max_consec_loss = 0
        current_streak = 0
        for t in trades:
            if not t.is_winner:
                current_streak += 1
                max_consec_loss = max(max_consec_loss, current_streak)
            else:
                current_streak = 0

        return BacktestMetrics(
            total_trades=total,
            winning_trades=len(winners),
            losing_trades=len(losers),
            win_rate=win_rate,
            total_return_pct=total_return,
            avg_return_pct=avg_return,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            profit_factor=profit_factor,
            avg_winner_pct=avg_winner,
            avg_loser_pct=avg_loser,
            max_consecutive_losses=max_consec_loss,
            trades=trades,
            equity_curve=equity,
        )

    async def run(self, symbol: str, interval: str,
                  start_time: datetime, end_time: datetime,
                  max_hold_bars: int = 48) -> BacktestMetrics:
        """
        Executa o backtest completo: fetch → indicadores → sinais → trades → métricas.
        """
        df = await self.fetch_data(symbol, interval, start_time, end_time)
        if df.empty or len(df) < 50:
            logger.warning(f"Insufficient data for backtest ({len(df)} candles)")
            return BacktestMetrics()

        df = self.calculate_indicators(df)
        df = self.generate_signals(df)
        trades = self.simulate_trades(df, max_hold_bars=max_hold_bars)
        metrics = self.calculate_metrics(trades)

        signal_count = len(df[df['signal'] != 'NONE'])
        logger.info(
            f"Backtest {symbol} {interval}: {len(df)} candles, "
            f"{signal_count} signals, {metrics.total_trades} trades, "
            f"WR={metrics.win_rate:.1f}%, Return={metrics.total_return_pct:.2f}%"
        )
        return metrics

    def run_on_dataframe(self, df: pd.DataFrame, max_hold_bars: int = 48) -> BacktestMetrics:
        """
        Executa backtest em um DataFrame já carregado (para walk-forward).
        """
        if df.empty or len(df) < 50:
            return BacktestMetrics()

        df = df.copy().reset_index(drop=True)
        df = self.calculate_indicators(df)
        df = self.generate_signals(df)
        trades = self.simulate_trades(df, max_hold_bars=max_hold_bars)
        return self.calculate_metrics(trades)
