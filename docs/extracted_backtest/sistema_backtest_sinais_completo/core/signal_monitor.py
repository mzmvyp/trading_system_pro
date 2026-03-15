# signal_monitor.py - MONITORAMENTO SEM LOCKS DE BANCO

"""
Signal Monitor otimizado para evitar locks:
1. Queries mais rápidas com timeouts
2. Usa dados de 5m ao invés de 1m
3. Batch updates para reduzir conexões
4. Sem transações longas
"""

import sqlite3
import json
import time
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging

from config.settings import settings

class SignalStatusMonitor:
    """Monitor de status de sinais SEM LOCKS"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.db_path = settings.database.signals_db_path
        self.signals_table = settings.database.signals_table
        
        # Estados que BLOQUEIAM novos sinais
        self.BLOCKING_STATES = ['ACTIVE', 'TARGET_1_HIT']
        
        # Estados FINALIZADOS (não bloqueiam)
        self.COMPLETED_STATES = ['TARGET_2_HIT', 'STOP_HIT', 'EXPIRED', 'MANUALLY_CLOSED']
        
        # CONFIGURAÇÕES ANTI-LOCK
        self.CONNECTION_TIMEOUT = 2  # Máximo 2s para conectar
        self.QUERY_TIMEOUT = 3       # Máximo 3s para query
        self.MAX_SIGNALS_PER_BATCH = 20  # Processa em lotes pequenos
        
        self.logger.info("SignalStatusMonitor SEM LOCKS inicializado")
    
    def _get_fast_connection(self):
        """Conexão otimizada para monitoramento"""
        try:
            conn = sqlite3.connect(
                self.db_path, 
                timeout=self.CONNECTION_TIMEOUT,
                check_same_thread=False,
                isolation_level=None  # Autocommit
            )
            
            # CONFIGURAÇÕES ANTI-LOCK
            conn.execute("PRAGMA read_uncommitted = true")
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA cache_size = 5000")
            conn.execute("PRAGMA temp_store = memory")
            
            return conn
            
        except Exception as e:
            self.logger.error(f"Erro na conexão de monitoramento: {e}")
            raise
    
    def check_active_signals(self, update_status: bool = True) -> Dict[str, Any]:
        """
        Verifica sinais ativos SEM LOCK - usa dados de 5m
        """
        start_time = time.time()
        
        try:
            # QUERY OTIMIZADA: busca apenas sinais que podem mudar
            query = f"""
            SELECT id, symbol, timeframe, signal_type, entry_price, stop_loss, 
                   targets, targets_hit, status, created_at, detector_name
            FROM {self.signals_table}
            WHERE status IN ('ACTIVE', 'TARGET_1_HIT')
            ORDER BY created_at DESC
            LIMIT {self.MAX_SIGNALS_PER_BATCH}
            """
            
            with self._get_fast_connection() as conn:
                conn.execute(f"PRAGMA busy_timeout = {self.QUERY_TIMEOUT * 1000}")
                
                cursor = conn.cursor()
                cursor.execute(query)
                signals_data = cursor.fetchall()
            
            if not signals_data:
                return {
                    'signals_checked': 0,
                    'signals_updated': 0,
                    'total_active_signals': 0,
                    'signals': [],
                    'execution_time': time.time() - start_time
                }
            
            self.logger.info(f"🔍 Verificando {len(signals_data)} sinais ativos")
            
            # Processa sinais em paralelo (sem DB)
            signals_to_update = []
            current_signals = []
            
            for signal_row in signals_data:
                try:
                    signal_info = self._process_signal_row(signal_row)
                    current_signals.append(signal_info)
                    
                    if signal_info.get('needs_update') and update_status:
                        signals_to_update.append(signal_info)
                        
                except Exception as e:
                    self.logger.warning(f"Erro ao processar sinal {signal_row[0]}: {e}")
                    continue
            
            # Batch update para reduzir locks
            updates_made = 0
            if signals_to_update:
                updates_made = self._batch_update_signals(signals_to_update)
            
            execution_time = time.time() - start_time
            
            result = {
                'signals_checked': len(signals_data),
                'signals_updated': updates_made,
                'total_active_signals': len([s for s in current_signals if s['status'] in self.BLOCKING_STATES]),
                'signals': current_signals,
                'execution_time': execution_time
            }
            
            if updates_made > 0:
                self.logger.info(f"✅ Monitoramento: {len(signals_data)} verificados | {updates_made} atualizados | {execution_time:.2f}s")
            else:
                self.logger.debug(f"✅ Monitoramento: {len(signals_data)} verificados | sem mudanças | {execution_time:.2f}s")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Erro no monitoramento em {execution_time:.2f}s: {e}")
            return {
                'error': str(e),
                'signals_checked': 0,
                'signals_updated': 0,
                'execution_time': execution_time
            }
    
    def _process_signal_row(self, signal_row: tuple) -> Dict[str, Any]:
        """Processa uma linha de sinal SEM ACESSAR DB"""
        
        signal_id, symbol, timeframe, signal_type, entry_price, stop_loss, targets_json, targets_hit_json, current_status, created_at, detector_name = signal_row
        
        # Parse JSON data
        try:
            targets = json.loads(targets_json) if targets_json else [entry_price * 1.02, entry_price * 1.04]
            targets_hit = json.loads(targets_hit_json) if targets_hit_json else [False, False]
        except:
            targets = [entry_price * 1.02, entry_price * 1.04]
            targets_hit = [False, False]
        
        signal_info = {
            'id': signal_id,
            'symbol': symbol,
            'timeframe': timeframe,
            'signal_type': signal_type,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'targets': targets,
            'targets_hit': targets_hit,
            'status': current_status,
            'created_at': created_at,
            'detector_name': detector_name,
            'needs_update': False,
            'calculated_status': {}
        }
        
        # Busca preço atual SEM LOCK (usa cache ou 5m)
        current_price = self._get_current_price_fast(symbol)
        
        if current_price:
            signal_info['current_price'] = current_price
            
            # Calcula novo status
            new_status, reason = self._calculate_new_status(
                current_price, entry_price, stop_loss, targets, 
                targets_hit, current_status, signal_type
            )
            
            signal_info['calculated_status'] = {
                'new_status': new_status,
                'reason': reason,
                'price_checked': current_price
            }
            
            if new_status != current_status:
                signal_info['needs_update'] = True
                signal_info['new_status'] = new_status
                signal_info['update_reason'] = reason
        else:
            signal_info['calculated_status'] = {
                'error': 'Could not fetch current price',
                'new_status': current_status
            }
        
        return signal_info
    
    def _get_current_price_fast(self, symbol: str) -> Optional[float]:
        """
        Busca preço atual SEM LOCK - usa dados de 5m
        """
        try:
            # Importa aqui para evitar circular import
            from core.data_reader import DataReader
            
            data_reader = DataReader()
            current_price = data_reader.get_current_price(symbol)
            
            return current_price
            
        except Exception as e:
            self.logger.debug(f"Erro ao buscar preço de {symbol}: {e}")
            return None
    
    def _calculate_new_status(self, current_price: float, entry_price: float, stop_loss: float, 
                            targets: List[float], targets_hit: List[bool], current_status: str, 
                            signal_type: str) -> Tuple[str, str]:
        """Calcula novo status baseado no preço atual"""
        
        try:
            # Verifica stop loss primeiro
            if signal_type == 'BUY_LONG':
                if current_price <= stop_loss:
                    return 'STOP_HIT', f'Price {current_price} <= stop {stop_loss}'
            else:  # SELL_SHORT
                if current_price >= stop_loss:
                    return 'STOP_HIT', f'Price {current_price} >= stop {stop_loss}'
            
            # Verifica targets
            new_targets_hit = targets_hit.copy()
            
            if signal_type == 'BUY_LONG':
                # Target 1
                if len(targets) > 0 and current_price >= targets[0] and not targets_hit[0]:
                    new_targets_hit[0] = True
                    if current_status == 'ACTIVE':
                        return 'TARGET_1_HIT', f'Target 1 hit: {current_price} >= {targets[0]}'
                
                # Target 2
                if len(targets) > 1 and current_price >= targets[1] and not targets_hit[1]:
                    new_targets_hit[1] = True
                    return 'TARGET_2_HIT', f'Target 2 hit: {current_price} >= {targets[1]}'
            
            else:  # SELL_SHORT
                # Target 1
                if len(targets) > 0 and current_price <= targets[0] and not targets_hit[0]:
                    new_targets_hit[0] = True
                    if current_status == 'ACTIVE':
                        return 'TARGET_1_HIT', f'Target 1 hit: {current_price} <= {targets[0]}'
                
                # Target 2
                if len(targets) > 1 and current_price <= targets[1] and not targets_hit[1]:
                    new_targets_hit[1] = True
                    return 'TARGET_2_HIT', f'Target 2 hit: {current_price} <= {targets[1]}'
            
            # Sem mudanças
            return current_status, 'No changes detected'
            
        except Exception as e:
            return current_status, f'Calculation error: {e}'
    
    def _batch_update_signals(self, signals_to_update: List[Dict]) -> int:
        """Atualiza sinais em lote para reduzir locks"""
        
        if not signals_to_update:
            return 0
        
        updated_count = 0
        
        try:
            with self._get_fast_connection() as conn:
                conn.execute(f"PRAGMA busy_timeout = {self.QUERY_TIMEOUT * 1000}")
                
                for signal in signals_to_update:
                    try:
                        update_query = f"""
                        UPDATE {self.signals_table}
                        SET status = ?, updated_at = ?
                        WHERE id = ?
                        """
                        
                        conn.execute(update_query, (
                            signal['new_status'],
                            datetime.now().isoformat(),
                            signal['id']
                        ))
                        
                        updated_count += 1
                        
                        # Log da transição
                        self.logger.info(
                            f"🔄 {signal['symbol']}: {signal['status']} → {signal['new_status']} | "
                            f"Motivo: {signal['update_reason']}"
                        )
                        
                    except Exception as e:
                        self.logger.warning(f"Erro ao atualizar sinal {signal['id']}: {e}")
                        continue
                
                # Commit em lote
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Erro no batch update: {e}")
        
        return updated_count
    
    def get_truly_active_signals(self) -> Dict[str, Any]:
        """Retorna sinais que REALMENTE bloqueiam novos sinais"""
        
        try:
            query = f"""
            SELECT symbol, timeframe, status, COUNT(*) as count
            FROM {self.signals_table}
            WHERE status IN ('ACTIVE', 'TARGET_1_HIT')
            GROUP BY symbol, timeframe, status
            ORDER BY symbol, timeframe
            """
            
            with self._get_fast_connection() as conn:
                conn.execute("PRAGMA busy_timeout = 1000")
                
                cursor = conn.cursor()
                cursor.execute(query)
                results = cursor.fetchall()
            
            blocking_combinations = []
            total_blocking = 0
            
            for symbol, timeframe, status, count in results:
                blocking_combinations.append({
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'status': status,
                    'count': count
                })
                total_blocking += count
            
            return {
                'total_blocking_signals': total_blocking,
                'blocking_combinations': blocking_combinations,
                'blocking_states': self.BLOCKING_STATES,
                'completed_states': self.COMPLETED_STATES
            }
            
        except Exception as e:
            self.logger.error(f"Erro ao buscar sinais bloqueadores: {e}")
            return {
                'error': str(e),
                'total_blocking_signals': 0,
                'blocking_combinations': []
            }
    
    def get_monitoring_statistics(self) -> Dict[str, Any]:
        """Estatísticas rápidas de monitoramento"""
        
        try:
            # Query simples e rápida
            summary_query = f"""
            SELECT status, COUNT(*) as count
            FROM {self.signals_table}
            GROUP BY status
            """
            
            with self._get_fast_connection() as conn:
                conn.execute("PRAGMA busy_timeout = 500")
                
                cursor = conn.cursor()
                cursor.execute(summary_query)
                status_counts = cursor.fetchall()
            
            summary = {
                'total_signals': 0,
                'active_blocking': 0,
                'completed': 0,
                'status_distribution': {}
            }
            
            for status, count in status_counts:
                summary['status_distribution'][status] = count
                summary['total_signals'] += count
                
                if status in self.BLOCKING_STATES:
                    summary['active_blocking'] += count
                elif status in self.COMPLETED_STATES:
                    summary['completed'] += count
            
            # Calcula win rate simples se houver dados
            total_finished = summary['completed']
            if total_finished > 0:
                wins = summary['status_distribution'].get('TARGET_2_HIT', 0)
                summary['overall_win_rate'] = (wins / total_finished) * 100
            else:
                summary['overall_win_rate'] = 0
            
            return {
                'summary': summary,
                'blocking_states': self.BLOCKING_STATES,
                'completed_states': self.COMPLETED_STATES
            }
            
        except Exception as e:
            self.logger.error(f"Erro nas estatísticas: {e}")
            return {
                'error': str(e),
                'summary': {
                    'total_signals': 0,
                    'active_blocking': 0,
                    'completed': 0
                }
            }

def print_signal_monitoring_report():
    """Função utilitária para relatório de monitoramento"""
    
    print("🔍 RELATÓRIO DE MONITORAMENTO DE SINAIS")
    print("=" * 50)
    
    try:
        monitor = SignalStatusMonitor()
        
        # Executa verificação
        start_time = time.time()
        results = monitor.check_active_signals(update_status=False)
        execution_time = time.time() - start_time
        
        if 'error' in results:
            print(f"❌ Erro: {results['error']}")
            return
        
        print(f"⏱️ Tempo de execução: {execution_time:.2f}s")
        print(f"🔍 Sinais verificados: {results['signals_checked']}")
        print(f"📊 Total ativos (bloqueando): {results['total_active_signals']}")
        
        # Sinais que precisam atualização
        signals_needing_update = [s for s in results.get('signals', []) if s.get('needs_update')]
        if signals_needing_update:
            print(f"\n🔄 SINAIS PRECISANDO ATUALIZAÇÃO ({len(signals_needing_update)}):")
            for signal in signals_needing_update:
                print(f"   🔄 {signal['symbol']}: {signal['status']} → {signal['new_status']}")
                print(f"      Motivo: {signal['update_reason']}")
        else:
            print("\n✅ Todos os sinais estão atualizados")
        
        # Estatísticas gerais
        stats = monitor.get_monitoring_statistics()
        if 'error' not in stats:
            summary = stats['summary']
            print(f"\n📊 ESTATÍSTICAS GERAIS:")
            print(f"   Total de sinais: {summary['total_signals']}")
            print(f"   Ativos (bloqueando): {summary['active_blocking']}")
            print(f"   Finalizados: {summary['completed']}")
            
            if summary['overall_win_rate'] > 0:
                print(f"   Win Rate: {summary['overall_win_rate']:.1f}%")
        
    except Exception as e:
        print(f"❌ Erro no relatório: {e}")

if __name__ == "__main__":
    print_signal_monitoring_report()