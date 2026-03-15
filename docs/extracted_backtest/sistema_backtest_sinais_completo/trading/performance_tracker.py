# -*- coding: utf-8 -*-
"""
Performance Tracker - Rastreamento de Performance do Paper Trading
Calcula métricas avançadas de trading
"""
import sqlite3
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np

from config.settings import settings


@dataclass
class TradeStats:
    """Estatísticas de uma série de trades"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    profit_factor: float
    avg_duration_minutes: float


class PerformanceTracker:
    """
    Rastreador de Performance
    
    Métricas calculadas:
    - Win rate, profit factor
    - Sharpe ratio, Sortino ratio
    - Maximum drawdown
    - Expectancy
    - Consecutive wins/losses
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.db_path = settings.database.signals_db_path
        self.logger.info("✅ Performance Tracker inicializado")
    
    def _get_connection(self):
        """Conexão com banco de dados"""
        return sqlite3.connect(self.db_path, timeout=10)
    
    def get_closed_trades(self, days: int = 7) -> List[Dict]:
        """Busca trades fechados dos últimos N dias"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        query = """
        SELECT id, symbol, side, entry_price, exit_price, quantity,
               entry_time, exit_time, pnl, pnl_percentage, fees_paid,
               slippage_cost, exit_reason, duration_minutes, metadata
        FROM paper_trades
        WHERE datetime(created_at) >= ?
        ORDER BY exit_time DESC
        """
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (cutoff_date.isoformat(),))
                rows = cursor.fetchall()
                
                trades = []
                for row in rows:
                    trades.append({
                        'id': row[0],
                        'symbol': row[1],
                        'side': row[2],
                        'entry_price': row[3],
                        'exit_price': row[4],
                        'quantity': row[5],
                        'entry_time': row[6],
                        'exit_time': row[7],
                        'pnl': row[8],
                        'pnl_percentage': row[9],
                        'fees_paid': row[10],
                        'slippage_cost': row[11],
                        'exit_reason': row[12],
                        'duration_minutes': row[13]
                    })
                
                return trades
        except Exception as e:
            self.logger.error(f"❌ Erro ao buscar trades: {e}")
            return []
    
    def calculate_trade_stats(self, trades: List[Dict]) -> TradeStats:
        """Calcula estatísticas básicas dos trades"""
        if not trades:
            return TradeStats(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] < 0]
        
        total = len(trades)
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = (win_count / total) * 100 if total > 0 else 0
        
        avg_win = sum(t['pnl'] for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t['pnl'] for t in losses) / len(losses) if losses else 0
        
        largest_win = max((t['pnl'] for t in wins), default=0)
        largest_loss = min((t['pnl'] for t in losses), default=0)
        
        total_gains = sum(t['pnl'] for t in wins)
        total_losses = abs(sum(t['pnl'] for t in losses))
        profit_factor = total_gains / total_losses if total_losses > 0 else float('inf')
        
        avg_duration = sum(t['duration_minutes'] for t in trades) / len(trades)
        
        return TradeStats(
            total_trades=total,
            winning_trades=win_count,
            losing_trades=loss_count,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            profit_factor=profit_factor,
            avg_duration_minutes=avg_duration
        )
    
    def calculate_sharpe_ratio(self, trades: List[Dict], risk_free_rate: float = 0.02) -> float:
        """
        Calcula Sharpe Ratio
        
        Args:
            trades: Lista de trades
            risk_free_rate: Taxa livre de risco anual (padrão: 2%)
        """
        if not trades:
            return 0.0
        
        returns = [t['pnl_percentage'] / 100 for t in trades]
        
        if len(returns) < 2:
            return 0.0
        
        avg_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        if std_return == 0:
            return 0.0
        
        # Ajusta risk-free rate para período do trade
        avg_duration_days = np.mean([t['duration_minutes'] / (60 * 24) for t in trades])
        period_risk_free = risk_free_rate * (avg_duration_days / 365)
        
        sharpe = (avg_return - period_risk_free) / std_return
        
        # Anualiza o Sharpe Ratio
        trades_per_year = 365 / avg_duration_days
        annualized_sharpe = sharpe * np.sqrt(trades_per_year)
        
        return annualized_sharpe
    
    def calculate_sortino_ratio(self, trades: List[Dict], risk_free_rate: float = 0.02) -> float:
        """
        Calcula Sortino Ratio (considera apenas downside risk)
        """
        if not trades:
            return 0.0
        
        returns = [t['pnl_percentage'] / 100 for t in trades]
        
        if len(returns) < 2:
            return 0.0
        
        avg_return = np.mean(returns)
        
        # Apenas retornos negativos para downside deviation
        downside_returns = [r for r in returns if r < 0]
        
        if not downside_returns:
            return float('inf')  # Sem perdas
        
        downside_std = np.std(downside_returns, ddof=1)
        
        if downside_std == 0:
            return 0.0
        
        avg_duration_days = np.mean([t['duration_minutes'] / (60 * 24) for t in trades])
        period_risk_free = risk_free_rate * (avg_duration_days / 365)
        
        sortino = (avg_return - period_risk_free) / downside_std
        
        trades_per_year = 365 / avg_duration_days
        annualized_sortino = sortino * np.sqrt(trades_per_year)
        
        return annualized_sortino
    
    def calculate_max_drawdown(self, days: int = 30) -> Dict:
        """Calcula Maximum Drawdown"""
        query = """
        SELECT timestamp, total_capital
        FROM paper_performance
        WHERE datetime(timestamp) >= ?
        ORDER BY timestamp ASC
        """
        
        cutoff = datetime.now() - timedelta(days=days)
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (cutoff.isoformat(),))
                rows = cursor.fetchall()
                
                if not rows:
                    return {'max_drawdown': 0.0, 'max_drawdown_pct': 0.0}
                
                capitals = [row[1] for row in rows]
                
                # Calcula drawdown
                peak = capitals[0]
                max_dd = 0.0
                max_dd_pct = 0.0
                
                for capital in capitals:
                    if capital > peak:
                        peak = capital
                    
                    dd = peak - capital
                    dd_pct = (dd / peak) * 100 if peak > 0 else 0
                    
                    if dd > max_dd:
                        max_dd = dd
                        max_dd_pct = dd_pct
                
                return {
                    'max_drawdown': max_dd,
                    'max_drawdown_pct': max_dd_pct,
                    'current_capital': capitals[-1],
                    'peak_capital': peak,
                    'is_in_drawdown': capitals[-1] < peak
                }
        except Exception as e:
            self.logger.error(f"❌ Erro ao calcular drawdown: {e}")
            return {'max_drawdown': 0.0, 'max_drawdown_pct': 0.0}
    
    def calculate_expectancy(self, trades: List[Dict]) -> float:
        """
        Calcula Expectancy (valor esperado por trade)
        """
        if not trades:
            return 0.0
        
        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] < 0]
        
        total = len(trades)
        win_rate = len(wins) / total if total > 0 else 0
        loss_rate = 1 - win_rate
        
        avg_win = sum(t['pnl'] for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t['pnl'] for t in losses) / len(losses) if losses else 0
        
        expectancy = (win_rate * avg_win) + (loss_rate * avg_loss)
        
        return expectancy
    
    def get_performance_by_symbol(self, days: int = 7) -> List[Dict]:
        """Performance agrupada por símbolo"""
        trades = self.get_closed_trades(days)
        
        if not trades:
            return []
        
        by_symbol = {}
        for trade in trades:
            symbol = trade['symbol']
            if symbol not in by_symbol:
                by_symbol[symbol] = []
            by_symbol[symbol].append(trade)
        
        results = []
        for symbol, symbol_trades in by_symbol.items():
            stats = self.calculate_trade_stats(symbol_trades)
            total_pnl = sum(t['pnl'] for t in symbol_trades)
            
            results.append({
                'symbol': symbol,
                'total_trades': stats.total_trades,
                'win_rate': stats.win_rate,
                'total_pnl': total_pnl,
                'profit_factor': stats.profit_factor,
                'avg_duration_minutes': stats.avg_duration_minutes
            })
        
        return sorted(results, key=lambda x: x['total_pnl'], reverse=True)
    
    def get_performance_by_timeframe(self, days: int = 7) -> List[Dict]:
        """Performance agrupada por timeframe"""
        query = """
        SELECT pt.*, ps.metadata
        FROM paper_trades pt
        LEFT JOIN paper_positions ps ON pt.position_id = ps.id
        WHERE datetime(pt.created_at) >= ?
        """
        
        cutoff = datetime.now() - timedelta(days=days)
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (cutoff.isoformat(),))
                rows = cursor.fetchall()
                
                by_timeframe = {}
                
                for row in rows:
                    import json
                    metadata = json.loads(row[15]) if row[15] else {}
                    timeframe = metadata.get('timeframe', 'unknown')
                    
                    if timeframe not in by_timeframe:
                        by_timeframe[timeframe] = []
                    
                    by_timeframe[timeframe].append({
                        'pnl': row[9],
                        'pnl_percentage': row[10],
                        'duration_minutes': row[14]
                    })
                
                results = []
                for tf, trades in by_timeframe.items():
                    wins = [t for t in trades if t['pnl'] > 0]
                    win_rate = (len(wins) / len(trades)) * 100 if trades else 0
                    total_pnl = sum(t['pnl'] for t in trades)
                    
                    results.append({
                        'timeframe': tf,
                        'total_trades': len(trades),
                        'win_rate': win_rate,
                        'total_pnl': total_pnl,
                        'avg_pnl': total_pnl / len(trades) if trades else 0
                    })
                
                return sorted(results, key=lambda x: x['total_pnl'], reverse=True)
                
        except Exception as e:
            self.logger.error(f"❌ Erro ao calcular performance por timeframe: {e}")
            return []
    
    def get_comprehensive_report(self, days: int = 7) -> Dict:
        """Relatório completo de performance"""
        trades = self.get_closed_trades(days)
        stats = self.calculate_trade_stats(trades)
        drawdown = self.calculate_max_drawdown(days)
        
        return {
            'period_days': days,
            'timestamp': datetime.now().isoformat(),
            'basic_stats': {
                'total_trades': stats.total_trades,
                'winning_trades': stats.winning_trades,
                'losing_trades': stats.losing_trades,
                'win_rate': stats.win_rate,
                'profit_factor': stats.profit_factor,
                'avg_win': stats.avg_win,
                'avg_loss': stats.avg_loss,
                'largest_win': stats.largest_win,
                'largest_loss': stats.largest_loss,
                'avg_duration_minutes': stats.avg_duration_minutes
            },
            'advanced_metrics': {
                'sharpe_ratio': self.calculate_sharpe_ratio(trades),
                'sortino_ratio': self.calculate_sortino_ratio(trades),
                'expectancy': self.calculate_expectancy(trades),
                'max_drawdown': drawdown['max_drawdown'],
                'max_drawdown_pct': drawdown['max_drawdown_pct']
            },
            'by_symbol': self.get_performance_by_symbol(days),
            'by_timeframe': self.get_performance_by_timeframe(days)
        }

