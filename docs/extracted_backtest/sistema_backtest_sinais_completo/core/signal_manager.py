# signal_manager.py - UTILITÁRIO CORRIGIDO - ESTADOS DE BLOQUEIO ATUALIZADOS

"""
Utilitário para gerenciar sinais ativos no sistema corrigido
Estados que bloqueiam: ACTIVE, TARGET_1_HIT
Estados finalizados: TARGET_2_HIT, STOP_HIT, EXPIRED, MANUALLY_CLOSED
"""

import json
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

from config.settings import settings
from core.signal_writer import EnhancedSignalWriter

class SignalManager:
    """Gerenciador de sinais ativos - LÓGICA CORRIGIDA PARA TARGETS"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.signal_writer = EnhancedSignalWriter()
        self.db_path = settings.database.signals_db_path
        self.signals_table = settings.database.signals_table
        self.backup_table = settings.database.backup_table
        
        # 🚨 ESTADOS CORRIGIDOS - agora TARGET_1_HIT depende do número de targets
        self.ALWAYS_BLOCKING_STATES = ['ACTIVE']  # Sempre bloqueiam
        self.CONDITIONAL_BLOCKING_STATES = ['TARGET_1_HIT']  # Bloqueiam apenas se tem 2+ targets
        self.COMPLETED_STATES = ['TARGET_2_HIT', 'STOP_HIT', 'EXPIRED', 'MANUALLY_CLOSED']  # Nunca bloqueiam
        
        self.logger.info("SignalManager inicializado com lógica inteligente de targets")
    
    def _get_connection(self):
        return sqlite3.connect(self.db_path, timeout=10)
    
    def get_truly_blocking_signals(self, symbol: str = None) -> Dict:
        """
        📊 Retorna apenas sinais que REALMENTE bloqueiam baseado na lógica de targets
        """
        base_query = """
        SELECT 
            symbol, timeframe, detector_name, signal_type, confidence, 
            entry_price, created_at, status, targets, stop_loss, id
        FROM {table}
        WHERE status IN ('ACTIVE', 'TARGET_1_HIT')
        {symbol_filter}
        ORDER BY symbol, timeframe, created_at DESC
        """.format(
            table=self.signals_table,
            symbol_filter="AND symbol = ?" if symbol else ""
        )
        
        try:
            params = [symbol] if symbol else []
            
            with self._get_connection() as conn:
                df = pd.read_sql_query(base_query, conn, params=params)
            
            if df.empty:
                return {
                    'total_blocking': 0,
                    'symbols_blocked': 0,
                    'by_symbol': {},
                    'signals': [],
                    'logic_explanation': 'ACTIVE sempre bloqueia | TARGET_1_HIT bloqueia apenas se tem 2+ targets'
                }
            
            # Analisa cada sinal para ver se realmente bloqueia
            truly_blocking = []
            
            for _, row in df.iterrows():
                try:
                    targets = json.loads(row['targets']) if row['targets'] else []
                    targets_count = len(targets)
                except:
                    targets_count = 0
                
                is_blocking = False
                blocking_reason = ""
                
                if row['status'] == 'ACTIVE':
                    is_blocking = True
                    blocking_reason = "Ativo - aguardando resultado"
                    
                elif row['status'] == 'TARGET_1_HIT':
                    if targets_count >= 2:
                        is_blocking = True
                        blocking_reason = f"Target 1/2 atingido - target 2 pendente"
                    else:
                        blocking_reason = f"Target único atingido - NÃO bloqueia"
                
                if is_blocking:
                    signal_data = row.to_dict()
                    signal_data['blocking_reason'] = blocking_reason
                    signal_data['targets_count'] = targets_count
                    truly_blocking.append(signal_data)
            
            # Agrupa por symbol
            by_symbol = {}
            for signal in truly_blocking:
                symbol_name = signal['symbol']
                if symbol_name not in by_symbol:
                    by_symbol[symbol_name] = {
                        'total': 0,
                        'timeframes': [],
                        'detectors': [],
                        'statuses': [],
                        'blocking_reasons': []
                    }
                
                by_symbol[symbol_name]['total'] += 1
                by_symbol[symbol_name]['timeframes'].append(signal['timeframe'])
                by_symbol[symbol_name]['detectors'].append(signal['detector_name'])
                by_symbol[symbol_name]['statuses'].append(signal['status'])
                by_symbol[symbol_name]['blocking_reasons'].append(signal['blocking_reason'])
            
            return {
                'total_blocking': len(truly_blocking),
                'symbols_blocked': len(by_symbol),
                'by_symbol': by_symbol,
                'signals': truly_blocking,
                'logic_explanation': 'ACTIVE sempre bloqueia | TARGET_1_HIT bloqueia apenas se tem 2+ targets',
                'blocking_states_logic': {
                    'ACTIVE': 'Sempre bloqueia - ainda não atingiu nenhum target',
                    'TARGET_1_HIT_with_2_targets': 'Bloqueia - target 2 ainda pendente',
                    'TARGET_1_HIT_with_1_target': 'NÃO bloqueia - objetivo já atingido',
                    'TARGET_2_HIT': 'NÃO bloqueia - objetivo completo',
                    'STOP_HIT': 'NÃO bloqueia - sinal finalizado',
                    'EXPIRED': 'NÃO bloqueia - sinal expirado'
                }
            }
            
        except Exception as e:
            self.logger.error(f"Erro ao obter sinais bloqueadores: {e}")
            return {'error': str(e)}
    
    def get_active_signals_overview(self) -> Dict:
        """
        📊 Visão geral dos sinais com lógica inteligente de bloqueio
        """
        return self.get_truly_blocking_signals()
    
    def get_signals_by_symbol(self, symbol: str) -> Dict:
        """
        🔍 Detalhes dos sinais de um symbol específico com lógica inteligente
        """
        result = self.get_truly_blocking_signals(symbol)
        
        if 'error' in result:
            return result
        
        # Calcula timeframes disponíveis baseado nos que realmente bloqueiam
        enabled_timeframes = settings.get_enabled_timeframes()
        blocked_timeframes = []
        
        for signal in result['signals']:
            blocked_timeframes.append(signal['timeframe'])
        
        available_timeframes = [tf for tf in enabled_timeframes if tf not in blocked_timeframes]
        
        # Adiciona análise detalhada de cada sinal
        detailed_signals = []
        for signal in result['signals']:
            signal['progress'] = self._calculate_signal_progress_intelligent(signal)
            detailed_signals.append(signal)
        
        return {
            'symbol': symbol,
            'blocking_signals': result['total_blocking'],
            'signals': detailed_signals,
            'blocked_timeframes': list(set(blocked_timeframes)),
            'available_timeframes': available_timeframes,
            'logic_explanation': result['logic_explanation']
        }
    
    def _calculate_signal_progress_intelligent(self, signal: Dict) -> str:
        """Calcula progresso do sinal com lógica inteligente de targets"""
        status = signal['status']
        targets_count = signal.get('targets_count', 0)
        blocking_reason = signal.get('blocking_reason', '')
        
        if status == 'TARGET_2_HIT':
            return "TARGET 2/2 ATINGIDO - FINALIZADO"
        elif status == 'TARGET_1_HIT':
            if targets_count >= 2:
                return f"TARGET 1/2 ATINGIDO - AGUARDANDO TARGET 2 (BLOQUEIA)"
            else:
                return f"TARGET ÚNICO ATINGIDO - FINALIZADO (NÃO BLOQUEIA)"
        elif status == 'ACTIVE':
            return f"ATIVO - AGUARDANDO RESULTADO (BLOQUEIA)"
        elif status == 'STOP_HIT':
            return "STOP LOSS ATINGIDO - FINALIZADO"
        else:
            return f"{status} - {blocking_reason}"
    
    def deactivate_signal_by_id(self, signal_id: str, reason: str = "manual_admin") -> bool:
        """
        🔴 Desativa um sinal específico pelo ID (apenas se estiver bloqueando)
        """
        sql = f"""
        UPDATE {self.signals_table}
        SET status = 'MANUALLY_CLOSED', updated_at = ?
        WHERE id = ? AND status IN ('ACTIVE', 'TARGET_1_HIT')
        """
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, [datetime.now().isoformat(), signal_id])
                affected_rows = cursor.rowcount
                conn.commit()
                
                if affected_rows > 0:
                    self.logger.info(f"🔴 SINAL DESATIVADO POR ID: {signal_id} | Motivo: {reason}")
                    return True
                else:
                    self.logger.warning(f"⚠️ Sinal não encontrado ou já finalizado: {signal_id}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Erro ao desativar sinal {signal_id}: {e}")
            return False
    
    def deactivate_signals_by_symbol(self, symbol: str, timeframe: Optional[str] = None, reason: str = "manual_admin") -> int:
        """
        🔴 Desativa sinais que estão REALMENTE bloqueando um symbol
        """
        if timeframe:
            sql = f"""
            UPDATE {self.signals_table}
            SET status = 'MANUALLY_CLOSED', updated_at = ?
            WHERE symbol = ? AND timeframe = ? AND status IN ('ACTIVE', 'TARGET_1_HIT')
            """
            params = [datetime.now().isoformat(), symbol, timeframe]
            action_desc = f"{symbol} {timeframe}"
        else:
            sql = f"""
            UPDATE {self.signals_table}
            SET status = 'MANUALLY_CLOSED', updated_at = ?
            WHERE symbol = ? AND status IN ('ACTIVE', 'TARGET_1_HIT')
            """
            params = [datetime.now().isoformat(), symbol]
            action_desc = f"{symbol} (todos timeframes)"
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, params)
                affected_rows = cursor.rowcount
                conn.commit()
                
                if affected_rows > 0:
                    self.logger.info(f"🔴 {affected_rows} SINAIS DESATIVADOS: {action_desc} | Motivo: {reason}")
                    
                    # Log adicional: verifica quais eram realmente bloqueadores
                    truly_blocking = self.get_truly_blocking_signals(symbol)
                    remaining_blocking = truly_blocking.get('total_blocking', 0)
                    
                    if remaining_blocking == 0:
                        self.logger.info(f"✅ {symbol} agora está completamente desbloqueado")
                    else:
                        self.logger.warning(f"⚠️ {symbol} ainda tem {remaining_blocking} sinais bloqueadores")
                    
                    return affected_rows
                else:
                    self.logger.warning(f"⚠️ Nenhum sinal bloqueador encontrado para: {action_desc}")
                    return 0
                    
        except Exception as e:
            self.logger.error(f"Erro ao desativar sinais de {action_desc}: {e}")
            return 0
    
    def deactivate_old_signals(self, hours_old: int = 24, reason: str = "auto_cleanup_old") -> int:
        """
        🔴 Desativa sinais ACTIVE mais antigos que X horas
        """
        cutoff_time = datetime.now() - timedelta(hours=hours_old)
        
        sql = f"""
        UPDATE {self.signals_table}
        SET status = 'EXPIRED', updated_at = ?
        WHERE status = 'ACTIVE' AND datetime(created_at) < ?
        """
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, (datetime.now().isoformat(), cutoff_time.isoformat()))
                affected_rows = cursor.rowcount
                conn.commit()
                
                if affected_rows > 0:
                    self.logger.info(f"🔴 {affected_rows} SINAIS ANTIGOS MARCADOS COMO EXPIRED (>{hours_old}h) | Motivo: {reason}")
                    return affected_rows
                else:
                    self.logger.info(f"✅ Nenhum sinal ACTIVE antigo encontrado (>{hours_old}h)")
                    return 0
                    
        except Exception as e:
            self.logger.error(f"Erro ao marcar sinais antigos como expired: {e}")
            return 0
    
    def get_backup_signals_stats(self, days: int = 1) -> Dict:
        """
        📦 Estatísticas dos sinais enviados para backup
        """
        start_date = datetime.now() - timedelta(days=days)
        
        sql = f"""
        SELECT 
            symbol, timeframe, detector_name, backup_reason,
            COUNT(*) as count,
            AVG(confidence) as avg_confidence
        FROM {self.backup_table}
        WHERE datetime(backup_timestamp) >= ?
        GROUP BY symbol, timeframe, detector_name, backup_reason
        ORDER BY count DESC
        """
        
        try:
            with self._get_connection() as conn:
                df = pd.read_sql_query(sql, conn, params=(start_date.isoformat(),))
            
            if df.empty:
                return {
                    'period_days': days,
                    'total_backups': 0,
                    'by_reason': {},
                    'by_symbol': {},
                    'details': []
                }
            
            # Agrupa por motivo
            df['reason_category'] = df['backup_reason'].str.split(':').str[0]
            by_reason = df.groupby('reason_category')['count'].sum().to_dict()
            
            # Agrupa por symbol
            by_symbol = df.groupby('symbol')['count'].sum().to_dict()
            
            return {
                'period_days': days,
                'total_backups': df['count'].sum(),
                'by_reason': by_reason,
                'by_symbol': by_symbol,
                'details': df.to_dict('records')
            }
            
        except Exception as e:
            self.logger.error(f"Erro ao obter estatísticas de backup: {e}")
            return {'error': str(e)}
    
    def force_clear_all_blocking_signals(self, confirmation_code: str = None) -> Dict:
        """
        ⚠️ FUNÇÃO PERIGOSA: Limpa TODOS os sinais que estão REALMENTE bloqueando
        """
        expected_code = "CLEAR_ALL_BLOCKING_CONFIRMED"
        
        if confirmation_code != expected_code:
            return {
                'status': 'error',
                'message': f'Código de confirmação incorreto. Use: {expected_code}',
                'signals_cleared': 0
            }
        
        sql = f"""
        UPDATE {self.signals_table}
        SET status = 'MANUALLY_CLOSED', updated_at = ?
        WHERE status IN ('ACTIVE', 'TARGET_1_HIT')
        """
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, [datetime.now().isoformat()])
                affected_rows = cursor.rowcount
                conn.commit()
                
                self.logger.warning(f"⚠️ LIMPEZA TOTAL EXECUTADA: {affected_rows} sinais fechados | Confirmação: {confirmation_code}")
                
                return {
                    'status': 'success',
                    'message': 'Todos os sinais potencialmente bloqueadores foram fechados',
                    'signals_cleared': affected_rows,
                    'note': 'Sinais TARGET_1_HIT com 1 target já não bloqueavam',
                    'states_cleared': ['ACTIVE', 'TARGET_1_HIT'],
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Erro na limpeza total: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'signals_cleared': 0
            }

def print_blocking_analysis(symbol: str = None):
    """
    🖨️ Função utilitária para analisar sinais com lógica inteligente de bloqueio
    """
    manager = SignalManager()
    
    if symbol:
        data = manager.get_truly_blocking_signals(symbol.upper())
        if 'error' in data:
            print(f"❌ Erro: {data['error']}")
            return
        
        print(f"\n🧠 ANÁLISE INTELIGENTE DE BLOQUEIO PARA {symbol.upper()}")
        print("=" * 80)
        print(f"💡 {data['logic_explanation']}")
        print("=" * 80)
        
        if data['total_blocking'] == 0:
            print("✅ Nenhum sinal realmente bloqueando - Symbol disponível para novos sinais")
            return
        
        print(f"🚫 Total de sinais REALMENTE bloqueando: {data['total_blocking']}")
        print("\nDetalhes dos sinais bloqueadores:")
        print("-" * 80)
        
        for signal in data['signals']:
            progress = signal.get('blocking_reason', 'Unknown')
            targets_count = signal.get('targets_count', 0)
            
            status_icon = {
                'ACTIVE': '🔴',
                'TARGET_1_HIT': '🟡' if targets_count >= 2 else '🟢'
            }.get(signal['status'], '🔴')
            
            print(f"{status_icon} {signal['timeframe']} | {signal['detector_name']} | {signal['signal_type']} | "
                  f"Targets: {targets_count} | Status: {signal['status']} | "
                  f"Razão: {progress} | Criado: {signal['created_at'][:19]}")
    
    else:
        data = manager.get_truly_blocking_signals()
        if 'error' in data:
            print(f"❌ Erro: {data['error']}")
            return
        
        print(f"\n🧠 ANÁLISE INTELIGENTE DE BLOQUEIO GERAL")
        print("=" * 80)
        print(f"💡 {data['logic_explanation']}")
        print("=" * 80)
        
        if data['total_blocking'] == 0:
            print("✅ Nenhum sinal realmente bloqueando no sistema - Todos os symbols disponíveis")
            return
        
        print(f"🚫 Total de sinais REALMENTE bloqueando: {data['total_blocking']}")
        print(f"Symbols com sinais bloqueadores: {data['symbols_blocked']}")
        
        print(f"\nLógica de estados:")
        for state, explanation in data['blocking_states_logic'].items():
            print(f"   • {state}: {explanation}")
        
        print("\nPor symbol:")
        print("-" * 80)
        
        for symbol, info in data['by_symbol'].items():
            reasons = list(set(info['blocking_reasons']))
            reasons_str = '; '.join(reasons)
            
            print(f"• {symbol:8} | {info['total']} sinais bloqueando | "
                  f"TF: {', '.join(set(info['timeframes']))} | "
                  f"Razões: {reasons_str}")

def print_active_signals_table(symbol: str = None):
    """
    🖨️ Função utilitária para imprimir tabela dos sinais QUE BLOQUEIAM
    """
    manager = SignalManager()
    
    if symbol:
        data = manager.get_signals_by_symbol(symbol.upper())
        if 'error' in data:
            print(f"❌ Erro: {data['error']}")
            return
        
        print(f"\n📊 SINAIS BLOQUEADORES PARA {symbol.upper()}")
        print("=" * 80)
        
        if data['blocking_signals'] == 0:
            print("✅ Nenhum sinal bloqueando - Symbol disponível para novos sinais")
            print(f"Timeframes disponíveis: {data['available_timeframes']}")
            return
        
        print(f"Total de sinais bloqueando: {data['blocking_signals']}")
        print(f"Timeframes bloqueados: {data['blocked_timeframes']}")
        print(f"Timeframes disponíveis: {data['available_timeframes']}")
        print("\nDetalhes dos sinais bloqueadores:")
        print("-" * 80)
        
        for signal in data['signals']:
            progress = signal.get('progress', 'ATIVO')
            targets_info = ""
            if signal.get('targets_hit'):
                hits = sum(signal['targets_hit'])
                total_targets = len(signal['targets_hit'])
                targets_info = f" | Targets: {hits}/{total_targets}"
            
            status_icon = {
                'ACTIVE': '🔴',
                'TARGET_1_HIT': '🟡'
            }.get(signal['status'], '🔴')
            
            print(f"{status_icon} {signal['timeframe']} | {signal['detector_name']} | {signal['signal_type']} | "
                  f"Conf: {signal['confidence']:.3f} | Entry: ${signal['entry_price']:,.4f} | "
                  f"Stop: ${signal['stop_loss']:,.4f}{targets_info} | "
                  f"Status: {progress} | Criado: {signal['created_at'][:19]}")
    
    else:
        data = manager.get_active_signals_overview()
        if 'error' in data:
            print(f"❌ Erro: {data['error']}")
            return
        
        print(f"\n📊 VISÃO GERAL DOS SINAIS BLOQUEADORES")
        print("=" * 80)
        
        if data['total_blocking'] == 0:
            print("✅ Nenhum sinal bloqueando no sistema - Todos os symbols disponíveis")
            return
        
        print(f"Total de sinais bloqueando: {data['total_blocking']}")
        print(f"Symbols com sinais bloqueadores: {data['symbols_blocked']}")
        print(f"Estados que bloqueiam: {data['blocking_states']}")
        print(f"Estados finalizados: {data['completed_states']}")
        
        if 'by_status' in data and data['by_status']:
            print(f"\nPor status: {data['by_status']}")
        
        print(f"Por timeframe: {data['by_timeframe']}")
        print("\nPor symbol:")
        print("-" * 80)
        
        for symbol, info in data['by_symbol'].items():
            timeframes_str = ', '.join(info['timeframes'])
            status_summary = ""
            if 'statuses' in info:
                status_counts = {}
                for status in info['statuses']:
                    status_counts[status] = status_counts.get(status, 0) + 1
                status_summary = f" | Status: {status_counts}"
            
            print(f"• {symbol:8} | {info['total']} sinais | TF: {timeframes_str} | "
                  f"Conf média: {info['avg_confidence']:.3f}{status_summary}")

def clear_symbol_signals(symbol: str, timeframe: str = None):
    """
    🔴 Função utilitária para limpar sinais bloqueadores de um symbol
    """
    manager = SignalManager()
    
    if timeframe:
        cleared = manager.deactivate_signals_by_symbol(symbol.upper(), timeframe, "manual_utility_clear")
        print(f"🔴 {cleared} sinais bloqueadores desativados para {symbol.upper()} {timeframe}")
    else:
        cleared = manager.deactivate_signals_by_symbol(symbol.upper(), None, "manual_utility_clear")
        print(f"🔴 {cleared} sinais bloqueadores desativados para {symbol.upper()} (todos timeframes)")

if __name__ == "__main__":
    # Exemplo de uso standalone
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "overview":
            print_active_signals_table()
        
        elif command == "symbol" and len(sys.argv) > 2:
            symbol = sys.argv[2]
            print_active_signals_table(symbol)
        
        elif command == "clear" and len(sys.argv) > 2:
            symbol = sys.argv[2]
            timeframe = sys.argv[3] if len(sys.argv) > 3 else None
            clear_symbol_signals(symbol, timeframe)
        
        else:
            print("Uso:")
            print("  python signal_manager.py overview           # Visão geral")
            print("  python signal_manager.py symbol BTC         # Sinais do BTC") 
            print("  python signal_manager.py clear BTC          # Limpa sinais do BTC")
            print("  python signal_manager.py clear BTC 5m       # Limpa sinal BTC 5m")
    
    else:
        print_active_signals_table()