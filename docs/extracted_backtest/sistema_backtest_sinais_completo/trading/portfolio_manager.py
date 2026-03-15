# -*- coding: utf-8 -*-
"""
Portfolio Manager - Gestão de Portfolio do Paper Trading
Gerencia alocação, risco e exposição
"""
import logging
from typing import Dict, List
from datetime import datetime
from dataclasses import dataclass

from trading.paper_trader import PaperTrader, PaperPosition


@dataclass
class PortfolioAllocation:
    """Alocação do portfolio"""
    symbol: str
    allocated_capital: float
    percentage: float
    positions_count: int
    avg_entry_price: float
    current_pnl: float


class PortfolioManager:
    """
    Gerenciador de Portfolio
    
    Features:
    - Diversificação automática
    - Gestão de risco por símbolo
    - Exposição máxima configurável
    - Rebalanceamento automático
    """
    
    def __init__(self, paper_trader: PaperTrader):
        self.logger = logging.getLogger(__name__)
        self.trader = paper_trader
        
        # Configurações de risco
        self.max_exposure_per_symbol = 0.20  # 20% por símbolo
        self.max_correlated_exposure = 0.40  # 40% em ativos correlacionados
        self.min_positions_for_diversification = 3
        
        self.logger.info("✅ Portfolio Manager inicializado")
    
    def get_portfolio_allocation(self) -> List[PortfolioAllocation]:
        """Retorna alocação atual do portfolio"""
        allocations = {}
        
        for position in self.trader.open_positions.values():
            if position.symbol not in allocations:
                allocations[position.symbol] = {
                    'allocated_capital': 0.0,
                    'positions_count': 0,
                    'total_entry_value': 0.0,
                    'current_pnl': 0.0
                }
            
            position_value = position.entry_price * position.quantity
            allocations[position.symbol]['allocated_capital'] += position_value
            allocations[position.symbol]['positions_count'] += 1
            allocations[position.symbol]['total_entry_value'] += position_value
            allocations[position.symbol]['current_pnl'] += position.pnl
        
        result = []
        for symbol, data in allocations.items():
            percentage = (data['allocated_capital'] / self.trader.current_balance) * 100
            avg_entry = data['total_entry_value'] / data['positions_count']
            
            result.append(PortfolioAllocation(
                symbol=symbol,
                allocated_capital=data['allocated_capital'],
                percentage=percentage,
                positions_count=data['positions_count'],
                avg_entry_price=avg_entry,
                current_pnl=data['current_pnl']
            ))
        
        return sorted(result, key=lambda x: x.allocated_capital, reverse=True)
    
    def check_symbol_exposure(self, symbol: str, additional_capital: float) -> tuple[bool, str]:
        """Verifica se pode adicionar mais exposição a um símbolo"""
        current_allocation = sum(
            pos.entry_price * pos.quantity 
            for pos in self.trader.open_positions.values() 
            if pos.symbol == symbol
        )
        
        total_exposure = current_allocation + additional_capital
        exposure_pct = (total_exposure / self.trader.current_balance) * 100
        
        if exposure_pct > self.max_exposure_per_symbol * 100:
            return False, f"Exposição em {symbol} muito alta: {exposure_pct:.1f}% (máx: {self.max_exposure_per_symbol*100}%)"
        
        return True, "OK"
    
    def check_diversification(self) -> Dict:
        """Verifica nível de diversificação"""
        allocations = self.get_portfolio_allocation()
        
        total_symbols = len(allocations)
        is_diversified = total_symbols >= self.min_positions_for_diversification
        
        # Calcula concentração (índice Herfindahl)
        concentration = sum(a.percentage ** 2 for a in allocations) / 100
        
        return {
            'total_symbols': total_symbols,
            'is_diversified': is_diversified,
            'concentration_index': concentration,
            'top_3_exposure': sum(a.percentage for a in allocations[:3]),
            'recommendations': self._get_diversification_recommendations(allocations)
        }
    
    def _get_diversification_recommendations(self, allocations: List[PortfolioAllocation]) -> List[str]:
        """Gera recomendações de diversificação"""
        recommendations = []
        
        if len(allocations) < self.min_positions_for_diversification:
            recommendations.append(f"Portfolio pouco diversificado: {len(allocations)} símbolos (recomendado: {self.min_positions_for_diversification}+)")
        
        for allocation in allocations:
            if allocation.percentage > self.max_exposure_per_symbol * 100:
                recommendations.append(f"Reduzir exposição em {allocation.symbol}: {allocation.percentage:.1f}% (máx: {self.max_exposure_per_symbol*100}%)")
        
        return recommendations
    
    def get_risk_metrics(self) -> Dict:
        """Calcula métricas de risco do portfolio"""
        if not self.trader.open_positions:
            return {
                'total_risk_amount': 0.0,
                'total_risk_percentage': 0.0,
                'largest_single_risk': 0.0,
                'average_risk_per_position': 0.0
            }
        
        total_risk = 0.0
        risks = []
        
        for position in self.trader.open_positions.values():
            # Risco = distância até stop loss
            if position.side == 'LONG':
                risk_per_unit = position.entry_price - position.stop_loss
            else:
                risk_per_unit = position.stop_loss - position.entry_price
            
            position_risk = risk_per_unit * position.quantity
            total_risk += position_risk
            risks.append(position_risk)
        
        return {
            'total_risk_amount': total_risk,
            'total_risk_percentage': (total_risk / self.trader.current_balance) * 100,
            'largest_single_risk': max(risks) if risks else 0.0,
            'average_risk_per_position': sum(risks) / len(risks) if risks else 0.0,
            'risk_reward_ratio': self._calculate_avg_risk_reward()
        }
    
    def _calculate_avg_risk_reward(self) -> float:
        """Calcula risk/reward médio das posições abertas"""
        ratios = []
        
        for position in self.trader.open_positions.values():
            if position.side == 'LONG':
                risk = position.entry_price - position.stop_loss
                reward = position.targets[0] - position.entry_price if position.targets else 0
            else:
                risk = position.stop_loss - position.entry_price
                reward = position.entry_price - position.targets[0] if position.targets else 0
            
            if risk > 0:
                ratios.append(reward / risk)
        
        return sum(ratios) / len(ratios) if ratios else 0.0
    
    def get_portfolio_summary(self) -> Dict:
        """Resumo completo do portfolio"""
        allocations = self.get_portfolio_allocation()
        diversification = self.check_diversification()
        risk_metrics = self.get_risk_metrics()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_value': self.trader.current_balance,
            'available_cash': self.trader.available_balance,
            'cash_percentage': (self.trader.available_balance / self.trader.current_balance) * 100,
            'invested_capital': self.trader.total_invested,
            'invested_percentage': (self.trader.total_invested / self.trader.current_balance) * 100,
            'allocations': [
                {
                    'symbol': a.symbol,
                    'value': a.allocated_capital,
                    'percentage': a.percentage,
                    'positions': a.positions_count,
                    'pnl': a.current_pnl
                }
                for a in allocations
            ],
            'diversification': diversification,
            'risk_metrics': risk_metrics
        }

