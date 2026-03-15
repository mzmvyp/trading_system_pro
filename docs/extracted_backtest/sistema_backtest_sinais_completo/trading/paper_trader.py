# -*- coding: utf-8 -*-
"""
Paper Trading Engine - Sistema de Simulação de Trades
Simula trades baseados nos sinais do sistema existente
"""
import sqlite3
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json

from config.settings import settings


@dataclass
class PaperPosition:
    """Posição aberta no paper trading"""
    id: str
    signal_id: str
    symbol: str
    side: str  # 'LONG' ou 'SHORT'
    entry_price: float
    quantity: float
    entry_time: datetime
    stop_loss: float
    targets: List[float]
    targets_hit: List[bool]
    status: str = "OPEN"  # OPEN, TARGET_1_HIT, TARGET_2_HIT, STOPPED, EXPIRED
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: float = 0.0
    pnl_percentage: float = 0.0
    fees_paid: float = 0.0
    slippage_cost: float = 0.0
    metadata: Dict = field(default_factory=dict)


class PaperTrader:
    """
    Paper Trading Engine - Simula trades baseados nos sinais reais
    
    Features:
    - Capital inicial configurável
    - Fees realistas da Binance (0.1%)
    - Slippage simulado (0.05%)
    - Tracking completo de P&L
    - Integração com sinais do signal_writer.py
    """
    
    def __init__(self, initial_capital: float = 10000.0):
        self.logger = logging.getLogger(__name__)
        self.db_path = settings.database.signals_db_path
        
        # Configurações de trading
        self.initial_capital = initial_capital
        self.current_balance = initial_capital
        self.available_balance = initial_capital
        self.total_invested = 0.0
        
        # Fees e slippage
        self.fee_percentage = 0.001  # 0.1% Binance
        self.slippage_percentage = 0.0005  # 0.05%
        
        # Gestão de risco
        self.max_position_size_pct = 0.10  # 10% do capital por trade
        self.max_open_positions = 5
        
        # Posições e histórico
        self.open_positions: Dict[str, PaperPosition] = {}
        self.closed_positions: List[PaperPosition] = []
        
        # Estatísticas
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_fees = 0.0
        
        # Inicializa tabelas
        self._ensure_tables_exist()
        self._load_open_positions()
        
        self.logger.info(f"✅ Paper Trader inicializado | Capital: ${self.initial_capital:,.2f}")
    
    def get_status(self) -> Dict:
        """Retorna status atual do paper trader"""
        total_pnl = self.current_balance - self.initial_capital
        pnl_percentage = (total_pnl / self.initial_capital) * 100 if self.initial_capital > 0 else 0
        
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        return {
            'initial_capital': self.initial_capital,
            'current_balance': self.current_balance,
            'available_balance': self.available_balance,
            'total_pnl': total_pnl,
            'pnl_percentage': pnl_percentage,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'open_positions': len(self.open_positions),
            'total_fees': self.total_fees,
            'max_position_size_pct': self.max_position_size_pct,
            'max_open_positions': self.max_open_positions
        }
    
    def _get_connection(self):
        """Conexão com banco de dados"""
        return sqlite3.connect(self.db_path, timeout=10)
    
    def _ensure_tables_exist(self):
        """Cria tabelas para paper trading"""
        create_positions_table = """
        CREATE TABLE IF NOT EXISTS paper_positions (
            id TEXT PRIMARY KEY,
            signal_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            entry_price REAL NOT NULL,
            quantity REAL NOT NULL,
            entry_time TEXT NOT NULL,
            stop_loss REAL NOT NULL,
            targets TEXT NOT NULL,
            targets_hit TEXT NOT NULL,
            status TEXT NOT NULL,
            exit_price REAL,
            exit_time TEXT,
            pnl REAL DEFAULT 0.0,
            pnl_percentage REAL DEFAULT 0.0,
            fees_paid REAL DEFAULT 0.0,
            slippage_cost REAL DEFAULT 0.0,
            metadata TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
        
        create_trades_table = """
        CREATE TABLE IF NOT EXISTS paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            position_id TEXT NOT NULL,
            signal_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            entry_price REAL NOT NULL,
            exit_price REAL NOT NULL,
            quantity REAL NOT NULL,
            entry_time TEXT NOT NULL,
            exit_time TEXT NOT NULL,
            pnl REAL NOT NULL,
            pnl_percentage REAL NOT NULL,
            fees_paid REAL NOT NULL,
            slippage_cost REAL NOT NULL,
            exit_reason TEXT NOT NULL,
            duration_minutes REAL,
            metadata TEXT,
            created_at TEXT NOT NULL
        )
        """
        
        create_performance_table = """
        CREATE TABLE IF NOT EXISTS paper_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            total_capital REAL NOT NULL,
            available_balance REAL NOT NULL,
            total_invested REAL NOT NULL,
            total_pnl REAL NOT NULL,
            total_pnl_percentage REAL NOT NULL,
            total_trades INTEGER NOT NULL,
            winning_trades INTEGER NOT NULL,
            losing_trades INTEGER NOT NULL,
            win_rate REAL NOT NULL,
            total_fees REAL NOT NULL,
            open_positions INTEGER NOT NULL,
            metadata TEXT
        )
        """
        
        try:
            with self._get_connection() as conn:
                conn.execute(create_positions_table)
                conn.execute(create_trades_table)
                conn.execute(create_performance_table)
                conn.commit()
                self.logger.debug("✅ Tabelas de paper trading verificadas/criadas")
        except Exception as e:
            self.logger.error(f"❌ Erro ao criar tabelas: {e}")
    
    def _load_open_positions(self):
        """Carrega posições abertas do banco"""
        try:
            query = """
            SELECT id, signal_id, symbol, side, entry_price, quantity, entry_time,
                   stop_loss, targets, targets_hit, status, exit_price, exit_time,
                   pnl, pnl_percentage, fees_paid, slippage_cost, metadata
            FROM paper_positions
            WHERE status IN ('OPEN', 'TARGET_1_HIT')
            """
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                rows = cursor.fetchall()
                
                for row in rows:
                    position = PaperPosition(
                        id=row[0],
                        signal_id=row[1],
                        symbol=row[2],
                        side=row[3],
                        entry_price=row[4],
                        quantity=row[5],
                        entry_time=datetime.fromisoformat(row[6]),
                        stop_loss=row[7],
                        targets=json.loads(row[8]),
                        targets_hit=json.loads(row[9]),
                        status=row[10],
                        exit_price=row[11],
                        exit_time=datetime.fromisoformat(row[12]) if row[12] else None,
                        pnl=row[13],
                        pnl_percentage=row[14],
                        fees_paid=row[15],
                        slippage_cost=row[16],
                        metadata=json.loads(row[17]) if row[17] else {}
                    )
                    
                    self.open_positions[position.id] = position
                    self.available_balance -= (position.entry_price * position.quantity)
                    self.total_invested += (position.entry_price * position.quantity)
                
                self.logger.info(f"📊 Carregadas {len(self.open_positions)} posições abertas")
        except Exception as e:
            self.logger.error(f"❌ Erro ao carregar posições: {e}")
    
    def can_open_position(self, capital_needed: float) -> Tuple[bool, str]:
        """Verifica se pode abrir nova posição"""
        # Verifica limite de posições
        if len(self.open_positions) >= self.max_open_positions:
            return False, f"Máximo de {self.max_open_positions} posições atingido"
        
        # Verifica capital disponível
        if capital_needed > self.available_balance:
            return False, f"Capital insuficiente: precisa ${capital_needed:.2f}, disponível ${self.available_balance:.2f}"
        
        # Verifica tamanho máximo de posição
        max_allowed = self.current_balance * self.max_position_size_pct
        if capital_needed > max_allowed:
            return False, f"Posição muito grande: máximo ${max_allowed:.2f} ({self.max_position_size_pct*100}%)"
        
        return True, "OK"
    
    def execute_signal(self, signal: Dict) -> Optional[PaperPosition]:
        """
        Executa um sinal criando uma posição paper trading
        
        Args:
            signal: Dict com dados do sinal (do trading_signals_v2)
        
        Returns:
            PaperPosition se executado, None caso contrário
        """
        try:
            symbol = signal['symbol']
            signal_id = signal['id']
            entry_price = float(signal['entry_price'])
            stop_loss = float(signal['stop_loss'])
            targets = json.loads(signal['targets']) if isinstance(signal['targets'], str) else signal['targets']
            signal_type = signal['signal_type']
            
            # Determina side
            side = 'LONG' if 'BUY' in signal_type else 'SHORT'
            
            # Calcula tamanho da posição (10% do capital)
            position_value = self.current_balance * self.max_position_size_pct
            quantity = position_value / entry_price
            
            # Calcula fees e slippage
            entry_fee = position_value * self.fee_percentage
            slippage = position_value * self.slippage_percentage
            total_cost = position_value + entry_fee + slippage
            
            # Verifica se pode abrir
            can_open, reason = self.can_open_position(total_cost)
            if not can_open:
                self.logger.warning(f"🚫 Não pode abrir posição para {symbol}: {reason}")
                return None
            
            # Aplica slippage ao preço de entrada
            if side == 'LONG':
                adjusted_entry = entry_price * (1 + self.slippage_percentage)
            else:
                adjusted_entry = entry_price * (1 - self.slippage_percentage)
            
            # Cria posição
            position_id = f"PAPER_{symbol}_{int(datetime.now().timestamp())}"
            position = PaperPosition(
                id=position_id,
                signal_id=signal_id,
                symbol=symbol,
                side=side,
                entry_price=adjusted_entry,
                quantity=quantity,
                entry_time=datetime.now(),
                stop_loss=stop_loss,
                targets=targets,
                targets_hit=[False] * len(targets),
                status="OPEN",
                fees_paid=entry_fee,
                slippage_cost=slippage,
                metadata={
                    'original_entry': entry_price,
                    'signal_confidence': signal.get('confidence', 0.0),
                    'signal_detector': signal.get('detector_name', 'unknown'),
                    'timeframe': signal.get('timeframe', 'unknown')
                }
            )
            
            # Atualiza balanços
            self.available_balance -= total_cost
            self.total_invested += position_value
            self.total_fees += entry_fee
            
            # Salva posição
            self.open_positions[position_id] = position
            self._save_position(position)
            
            self.logger.info(
                f"✅ PAPER TRADE ABERTO: {symbol} {side} | "
                f"Entry: ${adjusted_entry:.4f} | Qty: {quantity:.6f} | "
                f"Value: ${position_value:.2f} | Stop: ${stop_loss:.4f} | "
                f"Targets: [{', '.join([f'${t:.4f}' for t in targets])}] | "
                f"Fees: ${entry_fee:.2f}"
            )
            
            return position
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao executar sinal: {e}")
            return None
    
    def _save_position(self, position: PaperPosition):
        """Salva ou atualiza posição no banco"""
        sql = """
        INSERT OR REPLACE INTO paper_positions (
            id, signal_id, symbol, side, entry_price, quantity, entry_time,
            stop_loss, targets, targets_hit, status, exit_price, exit_time,
            pnl, pnl_percentage, fees_paid, slippage_cost, metadata,
            created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        try:
            with self._get_connection() as conn:
                conn.execute(sql, (
                    position.id,
                    position.signal_id,
                    position.symbol,
                    position.side,
                    position.entry_price,
                    position.quantity,
                    position.entry_time.isoformat(),
                    position.stop_loss,
                    json.dumps(position.targets),
                    json.dumps(position.targets_hit),
                    position.status,
                    position.exit_price,
                    position.exit_time.isoformat() if position.exit_time else None,
                    position.pnl,
                    position.pnl_percentage,
                    position.fees_paid,
                    position.slippage_cost,
                    json.dumps(position.metadata),
                    position.entry_time.isoformat(),
                    datetime.now().isoformat()
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"❌ Erro ao salvar posição: {e}")
    
    def update_positions(self, current_prices: Dict[str, float]):
        """
        Atualiza posições abertas com preços atuais
        Verifica stops e targets
        
        Args:
            current_prices: Dict {symbol: price}
        """
        closed_count = 0
        
        for position_id, position in list(self.open_positions.items()):
            if position.symbol not in current_prices:
                continue
            
            current_price = current_prices[position.symbol]
            
            # Verifica stop loss
            if self._check_stop_loss(position, current_price):
                self._close_position(position, current_price, "STOP_HIT")
                closed_count += 1
                continue
            
            # Verifica targets
            target_hit = self._check_targets(position, current_price)
            if target_hit:
                # Se atingiu último target, fecha posição
                if all(position.targets_hit):
                    self._close_position(position, current_price, "ALL_TARGETS_HIT")
                    closed_count += 1
                else:
                    # Atualiza status mas não fecha
                    target_num = sum(position.targets_hit)
                    position.status = f"TARGET_{target_num}_HIT"
                    self._save_position(position)
        
        if closed_count > 0:
            self.logger.info(f"📊 {closed_count} posições fechadas nesta atualização")
    
    def _check_stop_loss(self, position: PaperPosition, current_price: float) -> bool:
        """Verifica se stop loss foi atingido"""
        if position.side == 'LONG':
            return current_price <= position.stop_loss
        else:  # SHORT
            return current_price >= position.stop_loss
    
    def _check_targets(self, position: PaperPosition, current_price: float) -> bool:
        """Verifica se algum target foi atingido"""
        hit_any = False
        
        for i, (target, already_hit) in enumerate(zip(position.targets, position.targets_hit)):
            if already_hit:
                continue
            
            if position.side == 'LONG':
                if current_price >= target:
                    position.targets_hit[i] = True
                    hit_any = True
                    self.logger.info(f"🎯 Target {i+1} atingido para {position.symbol} | ${target:.4f}")
            else:  # SHORT
                if current_price <= target:
                    position.targets_hit[i] = True
                    hit_any = True
                    self.logger.info(f"🎯 Target {i+1} atingido para {position.symbol} | ${target:.4f}")
        
        return hit_any
    
    def _close_position(self, position: PaperPosition, exit_price: float, reason: str):
        """Fecha uma posição e calcula P&L"""
        # Calcula P&L bruto
        if position.side == 'LONG':
            gross_pnl = (exit_price - position.entry_price) * position.quantity
        else:  # SHORT
            gross_pnl = (position.entry_price - exit_price) * position.quantity
        
        # Calcula fees de saída
        exit_value = exit_price * position.quantity
        exit_fee = exit_value * self.fee_percentage
        exit_slippage = exit_value * self.slippage_percentage
        
        # P&L líquido
        net_pnl = gross_pnl - exit_fee - exit_slippage
        pnl_pct = (net_pnl / (position.entry_price * position.quantity)) * 100
        
        # Atualiza posição
        position.exit_price = exit_price
        position.exit_time = datetime.now()
        position.pnl = net_pnl
        position.pnl_percentage = pnl_pct
        position.fees_paid += exit_fee
        position.slippage_cost += exit_slippage
        position.status = reason
        
        # Atualiza balanços
        position_value = position.entry_price * position.quantity
        self.total_invested -= position_value
        self.available_balance += (position_value + net_pnl)
        self.current_balance += net_pnl
        self.total_fees += exit_fee
        
        # Atualiza estatísticas
        self.total_trades += 1
        if net_pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Salva trade fechado
        self._save_closed_trade(position)
        self._save_position(position)
        
        # Remove das posições abertas
        del self.open_positions[position.id]
        self.closed_positions.append(position)
        
        duration = (position.exit_time - position.entry_time).total_seconds() / 60
        
        result_emoji = "✅" if net_pnl > 0 else "❌"
        self.logger.info(
            f"{result_emoji} PAPER TRADE FECHADO: {position.symbol} {position.side} | "
            f"Entry: ${position.entry_price:.4f} → Exit: ${exit_price:.4f} | "
            f"P&L: ${net_pnl:.2f} ({pnl_pct:+.2f}%) | "
            f"Razão: {reason} | Duração: {duration:.1f}min | "
            f"Fees: ${position.fees_paid:.2f}"
        )
    
    def _save_closed_trade(self, position: PaperPosition):
        """Salva trade fechado no histórico"""
        sql = """
        INSERT INTO paper_trades (
            position_id, signal_id, symbol, side, entry_price, exit_price,
            quantity, entry_time, exit_time, pnl, pnl_percentage,
            fees_paid, slippage_cost, exit_reason, duration_minutes,
            metadata, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        duration = (position.exit_time - position.entry_time).total_seconds() / 60
        
        try:
            with self._get_connection() as conn:
                conn.execute(sql, (
                    position.id,
                    position.signal_id,
                    position.symbol,
                    position.side,
                    position.entry_price,
                    position.exit_price,
                    position.quantity,
                    position.entry_time.isoformat(),
                    position.exit_time.isoformat(),
                    position.pnl,
                    position.pnl_percentage,
                    position.fees_paid,
                    position.slippage_cost,
                    position.status,
                    duration,
                    json.dumps(position.metadata),
                    datetime.now().isoformat()
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"❌ Erro ao salvar trade fechado: {e}")
    
    def get_performance_summary(self) -> Dict:
        """Retorna resumo de performance"""
        total_pnl = self.current_balance - self.initial_capital
        total_pnl_pct = (total_pnl / self.initial_capital) * 100
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        return {
            'initial_capital': self.initial_capital,
            'current_balance': self.current_balance,
            'available_balance': self.available_balance,
            'total_invested': self.total_invested,
            'total_pnl': total_pnl,
            'total_pnl_percentage': total_pnl_pct,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'total_fees': self.total_fees,
            'open_positions': len(self.open_positions),
            'closed_positions': len(self.closed_positions),
            'timestamp': datetime.now().isoformat()
        }
    
    def save_performance_snapshot(self):
        """Salva snapshot de performance no banco"""
        summary = self.get_performance_summary()
        
        sql = """
        INSERT INTO paper_performance (
            timestamp, total_capital, available_balance, total_invested,
            total_pnl, total_pnl_percentage, total_trades, winning_trades,
            losing_trades, win_rate, total_fees, open_positions, metadata
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        try:
            with self._get_connection() as conn:
                conn.execute(sql, (
                    summary['timestamp'],
                    summary['current_balance'],
                    summary['available_balance'],
                    summary['total_invested'],
                    summary['total_pnl'],
                    summary['total_pnl_percentage'],
                    summary['total_trades'],
                    summary['winning_trades'],
                    summary['losing_trades'],
                    summary['win_rate'],
                    summary['total_fees'],
                    summary['open_positions'],
                    json.dumps({})
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"❌ Erro ao salvar snapshot: {e}")

