# -*- coding: utf-8 -*-
"""
Signal Integration - Integração entre sinais reais e paper trading
Monitora novos sinais e executa automaticamente no paper trader
"""
import sqlite3
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from config.settings import settings
from trading.paper_trader import PaperTrader
from core.data_reader import DataReader


class SignalToPaperTrader:
    """
    Ponte entre sistema de sinais e paper trading
    
    Funcionalidades:
    - Monitora novos sinais ativos
    - Executa automaticamente no paper trader
    - Atualiza posições com preços do stream
    - Sincroniza status entre sistema e paper trader
    """
    
    def __init__(self, paper_trader: Optional[PaperTrader] = None):
        self.logger = logging.getLogger(__name__)
        self.db_path = settings.database.signals_db_path
        self.stream_db_path = settings.database.stream_db_path
        
        # Paper trader
        self.paper_trader = paper_trader or PaperTrader()
        
        # Data reader para preços atuais
        self.data_reader = DataReader()
        
        # Controle de sinais já processados
        self.processed_signals = set()
        self._load_processed_signals()
        
        self.logger.info("✅ Signal Integration inicializada")
    
    def _get_signals_connection(self):
        """Conexão com banco de sinais"""
        return sqlite3.connect(self.db_path, timeout=10)
    
    def _get_stream_connection(self):
        """Conexão com banco de stream"""
        return sqlite3.connect(self.stream_db_path, timeout=10)
    
    def _load_processed_signals(self):
        """Carrega sinais já processados do paper trader"""
        try:
            query = """
            SELECT DISTINCT signal_id 
            FROM paper_positions
            """
            
            with self._get_signals_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                rows = cursor.fetchall()
                
                self.processed_signals = {row[0] for row in rows}
                
                self.logger.info(f"📊 {len(self.processed_signals)} sinais já processados")
        except Exception as e:
            self.logger.error(f"❌ Erro ao carregar sinais processados: {e}")
    
    def get_new_signals(self, since_minutes: int = 5) -> List[Dict]:
        """
        Busca novos sinais ACTIVE que ainda não foram processados
        
        Args:
            since_minutes: Buscar sinais dos últimos N minutos
        """
        cutoff_time = datetime.now() - timedelta(minutes=since_minutes)
        
        query = f"""
        SELECT id, symbol, signal_type, entry_price, stop_loss, targets,
               confidence, timeframe, detector_name, detector_type, created_at
        FROM {settings.database.signals_table}
        WHERE status = 'ACTIVE'
        AND datetime(created_at) >= ?
        ORDER BY created_at DESC
        """
        
        try:
            with self._get_signals_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (cutoff_time.isoformat(),))
                rows = cursor.fetchall()
                
                new_signals = []
                for row in rows:
                    signal_id = row[0]
                    
                    # Ignora se já foi processado
                    if signal_id in self.processed_signals:
                        continue
                    
                    new_signals.append({
                        'id': signal_id,
                        'symbol': row[1],
                        'signal_type': row[2],
                        'entry_price': row[3],
                        'stop_loss': row[4],
                        'targets': row[5],
                        'confidence': row[6],
                        'timeframe': row[7],
                        'detector_name': row[8],
                        'detector_type': row[9],
                        'created_at': row[10]
                    })
                
                return new_signals
        except Exception as e:
            self.logger.error(f"❌ Erro ao buscar novos sinais: {e}")
            return []
    
    def process_new_signals(self):
        """Processa novos sinais executando no paper trader"""
        new_signals = self.get_new_signals()
        
        if not new_signals:
            self.logger.debug("Nenhum novo sinal para processar")
            return
        
        self.logger.info(f"🔔 {len(new_signals)} novos sinais encontrados")
        
        for signal in new_signals:
            try:
                position = self.paper_trader.execute_signal(signal)
                
                if position:
                    self.processed_signals.add(signal['id'])
                    self.logger.info(f"✅ Sinal {signal['id']} executado: {signal['symbol']} {signal['signal_type']}")
                else:
                    # Marca como processado mesmo se não executou (para não tentar novamente)
                    self.processed_signals.add(signal['id'])
                    self.logger.warning(f"⚠️ Sinal {signal['id']} não executado: {signal['symbol']}")
                    
            except Exception as e:
                self.logger.error(f"❌ Erro ao processar sinal {signal['id']}: {e}")
    
    def get_current_prices(self) -> Dict[str, float]:
        """Busca preços atuais do stream database"""
        symbols = [pos.symbol for pos in self.paper_trader.open_positions.values()]
        
        if not symbols:
            return {}
        
        # Remove duplicatas
        unique_symbols = list(set(symbols))
        
        prices = {}
        
        for symbol in unique_symbols:
            try:
                # Tenta buscar do data_reader primeiro (usa cache)
                market_data = self.data_reader.get_latest_data(symbol, "5m")
                
                if market_data and market_data.is_sufficient_data:
                    prices[symbol] = float(market_data.data.iloc[-1]['close_price'])
                    self.logger.debug(f"Preço de {symbol}: ${prices[symbol]:.4f}")
                else:
                    # Fallback: busca direto do banco
                    price = self._get_price_from_stream(symbol)
                    if price:
                        prices[symbol] = price
                        
            except Exception as e:
                self.logger.error(f"❌ Erro ao buscar preço de {symbol}: {e}")
        
        return prices
    
    def _get_price_from_stream(self, symbol: str) -> Optional[float]:
        """Busca preço direto do stream database"""
        query = """
        SELECT close_price 
        FROM crypto_ohlc
        WHERE symbol = ?
        AND timeframe = '1m'
        ORDER BY timestamp DESC
        LIMIT 1
        """
        
        try:
            with self._get_stream_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (symbol,))
                row = cursor.fetchone()
                
                if row:
                    return float(row[0])
                return None
        except Exception as e:
            self.logger.error(f"❌ Erro ao buscar preço do stream para {symbol}: {e}")
            return None
    
    def update_positions(self):
        """Atualiza posições abertas com preços atuais"""
        if not self.paper_trader.open_positions:
            self.logger.debug("Nenhuma posição aberta para atualizar")
            return
        
        # Busca preços atuais
        current_prices = self.get_current_prices()
        
        if not current_prices:
            self.logger.warning("⚠️ Não foi possível obter preços atuais")
            return
        
        # Atualiza posições
        self.paper_trader.update_positions(current_prices)
    
    def sync_with_real_signals(self):
        """
        Sincroniza status entre sinais reais e posições paper
        Se um sinal foi manualmente fechado, fecha também no paper trader
        """
        query = f"""
        SELECT id, status
        FROM {settings.database.signals_table}
        WHERE status IN ('TARGET_2_HIT', 'STOP_HIT', 'EXPIRED', 'MANUALLY_CLOSED')
        AND datetime(updated_at) >= ?
        """
        
        cutoff = datetime.now() - timedelta(hours=1)
        
        try:
            with self._get_signals_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (cutoff.isoformat(),))
                rows = cursor.fetchall()
                
                for signal_id, status in rows:
                    # Procura posição correspondente
                    for position in list(self.paper_trader.open_positions.values()):
                        if position.signal_id == signal_id and position.status == "OPEN":
                            # Fecha posição
                            current_price = self.get_current_prices().get(position.symbol)
                            
                            if current_price:
                                self.paper_trader._close_position(
                                    position,
                                    current_price,
                                    f"SYNC_{status}"
                                )
                                self.logger.info(f"🔄 Posição {position.id} sincronizada com sinal {signal_id}: {status}")
                                
        except Exception as e:
            self.logger.error(f"❌ Erro ao sincronizar sinais: {e}")
    
    def run_continuous_monitoring(self, check_interval: int = 30):
        """
        Monitora continuamente novos sinais e atualiza posições
        
        Args:
            check_interval: Intervalo em segundos entre verificações
        """
        self.logger.info(f"🚀 Iniciando monitoramento contínuo (intervalo: {check_interval}s)")
        
        try:
            while True:
                # Processa novos sinais
                self.process_new_signals()
                
                # Atualiza posições abertas
                self.update_positions()
                
                # Sincroniza com sinais reais
                self.sync_with_real_signals()
                
                # Salva snapshot de performance a cada 10 ciclos
                if hasattr(self, '_cycle_count'):
                    self._cycle_count += 1
                else:
                    self._cycle_count = 1
                
                if self._cycle_count % 10 == 0:
                    self.paper_trader.save_performance_snapshot()
                    
                    # Log de status
                    summary = self.paper_trader.get_performance_summary()
                    self.logger.info(
                        f"💰 Paper Trading Status: "
                        f"Capital: ${summary['current_balance']:,.2f} | "
                        f"P&L: ${summary['total_pnl']:+,.2f} ({summary['total_pnl_percentage']:+.2f}%) | "
                        f"Trades: {summary['total_trades']} | "
                        f"Win Rate: {summary['win_rate']:.1f}% | "
                        f"Posições Abertas: {summary['open_positions']}"
                    )
                
                # Aguarda próximo ciclo
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            self.logger.info("\n🛑 Monitoramento interrompido pelo usuário")
        except Exception as e:
            self.logger.error(f"❌ Erro no monitoramento contínuo: {e}")
            raise
    
    def get_integration_status(self) -> Dict:
        """Retorna status da integração"""
        return {
            'processed_signals': len(self.processed_signals),
            'open_positions': len(self.paper_trader.open_positions),
            'paper_trader_summary': self.paper_trader.get_performance_summary(),
            'timestamp': datetime.now().isoformat()
        }


# Função auxiliar para uso standalone
def start_paper_trading_monitor(initial_capital: float = 10000.0, check_interval: int = 30):
    """
    Inicia monitoramento paper trading standalone
    
    Args:
        initial_capital: Capital inicial do paper trader
        check_interval: Intervalo de verificação em segundos
    """
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"🚀 Iniciando Paper Trading Monitor")
    logger.info(f"💰 Capital Inicial: ${initial_capital:,.2f}")
    logger.info(f"⏱️ Intervalo de verificação: {check_interval}s")
    
    # Cria paper trader
    paper_trader = PaperTrader(initial_capital=initial_capital)
    
    # Cria integração
    integration = SignalToPaperTrader(paper_trader)
    
    # Inicia monitoramento
    integration.run_continuous_monitoring(check_interval)


if __name__ == "__main__":
    start_paper_trading_monitor()

