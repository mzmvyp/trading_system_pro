# data_reader.py - CORRIGIDO PARA EVITAR TRAVAMENTOS

"""
Data Reader otimizado ANTI-TRAVAMENTO:
1. Timeouts muito agressivos
2. Validação prévia do banco
3. Fallback robusto
4. Debug detalhado
"""

import sqlite3
import pandas as pd
import logging
import time
import os
import gc
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from config.settings import settings

@dataclass
class MarketData:
    """Estrutura COMPLETA para dados de mercado"""
    symbol: str
    timeframe: str
    data: pd.DataFrame
    last_update: datetime
    
    @property
    def is_sufficient_data(self) -> bool:
        """Verifica se há dados suficientes para análise"""
        return len(self.data) >= 50
    
    @property
    def latest_price(self) -> float:
        """Preço mais recente (compatibilidade com TechnicalAnalyzer)"""
        if len(self.data) > 0:
            return float(self.data.iloc[-1]['close_price'])
        return 0.0
    
    @property
    def current_price(self) -> float:
        """Alias para latest_price"""
        return self.latest_price

class DataReader:
    """Data Reader ANTI-TRAVAMENTO - SUPER ROBUSTO"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # CONFIGURAÇÃO COMPATÍVEL - detecta automaticamente a configuração correta
        self.db_path, self.table_name = self._detect_database_config()
        
        # CONFIGURAÇÕES ANTI-TRAVAMENTO
        self.CONNECTION_TIMEOUT = 1  # Máximo 1s para conectar
        self.QUERY_TIMEOUT = 2       # Máximo 2s para query
        self.MAX_RETRIES = 1         # Máximo 1 tentativa (rápido)
        
        # Valida banco ANTES de usar
        self.database_validated = self._validate_database_setup()
        
        self.logger.info(f"DataReader ANTI-TRAVAMENTO inicializado:")
        self.logger.info(f"  • DB: {self.db_path}")
        self.logger.info(f"  • Tabela: {self.table_name}")
        self.logger.info(f"  • Validado: {self.database_validated}")
        self.logger.info(f"  • Timeouts: {self.CONNECTION_TIMEOUT}s conexão, {self.QUERY_TIMEOUT}s query")
    
    # ADICIONAR método genérico que funciona para ambos:
    
    def get_live_candle(self, symbol: str, timeframe: str) -> Dict:
        """Versão SIMPLIFICADA mais confiável"""
        
        try:
            timeout = getattr(self, 'CONNECTION_TIMEOUT', 1.0)
            
            with sqlite3.connect(self.db_path, timeout=timeout) as conn:
                cursor = conn.cursor()
                
                # Query SIMPLIFICADA (sem subqueries complexas que podem dar erro)
                live_candle_query = """
                SELECT 
                    symbol,
                    max(kline_close_time) as close_time,
                    min(open_price) as open_price,
                    max(high_price) as high_price,
                    min(low_price) as low_price,
                    max(close_price) as close_price,
                    sum(volume) as volume
                FROM kline_microstructure_1m 
                WHERE symbol = ?
                AND kline_close_time > (
                    SELECT max(kline_close_time) 
                    FROM crypto_ohlc 
                    WHERE symbol = ? AND timeframe = ?
                )
                GROUP BY symbol
                """
                
                cursor.execute(live_candle_query, [symbol, symbol, timeframe])
                result = cursor.fetchone()
                
                if result:
                    columns = [desc[0] for desc in cursor.description]
                    live_data = dict(zip(columns, result))
                    self.logger.debug(f"📡 {symbol} {timeframe}: Candle live encontrado (${live_data.get('close_price', 0):.4f})")
                    return live_data
                else:
                    self.logger.debug(f"📭 {symbol} {timeframe}: Nenhum candle live")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Erro na busca de candle live {symbol} {timeframe}: {e}")
            return None
    
    def get_enhanced_data(self, symbol: str, timeframe: str) -> MarketData:
        """Dados históricos + candle atual em formação - VERSÃO DEFINITIVA CORRIGIDA"""
        
        try:
            # 1. Busca dados históricos normais
            historical_data = self.get_latest_data(symbol, timeframe)
            
            if not historical_data or not historical_data.is_sufficient_data:
                self.logger.debug(f"❌ {symbol} {timeframe}: Dados históricos insuficientes")
                return historical_data
            
            # 2. Apenas para 1h, 4h e 1d
            if timeframe not in ["1h"]:
                self.logger.debug(f"⚪ {symbol} {timeframe}: Não usa dados live, retornando históricos")
                return historical_data
            
            # 3. Busca candle em formação
            live_candle = self.get_live_candle(symbol, timeframe)
            
            if live_candle and all(live_candle.get(k) is not None for k in ['open_price', 'high_price', 'low_price', 'close_price']):
                try:
                    # Cria linha do candle live
                    live_row = pd.DataFrame([{
                        'timestamp': pd.to_datetime(live_candle['close_time']),
                        'open_price': float(live_candle['open_price']),
                        'high_price': float(live_candle['high_price']),
                        'low_price': float(live_candle['low_price']),
                        'close_price': float(live_candle['close_price']),
                        'volume': float(live_candle['volume']) if live_candle.get('volume') else 0.0
                    }])
                    
                    # Combina dados históricos + candle live
                    combined_data = pd.concat([historical_data.data, live_row], ignore_index=True)
                    
                    # CORREÇÃO: Usa apenas parâmetros aceitos pelo construtor
                    enhanced_data = MarketData(
                        symbol=symbol,
                        timeframe=timeframe,
                        data=combined_data,
                        last_update=datetime.now()
                    )
                    
                    self.logger.debug(f"📡 {symbol} {timeframe}: Candle LIVE adicionado (${live_candle['close_price']:.4f}, vol: {live_candle.get('volume', 0):.0f})")
                    return enhanced_data
                    
                except Exception as parse_error:
                    self.logger.warning(f"⚠️ {symbol} {timeframe}: Erro ao processar candle live: {parse_error}")
                    return historical_data
            else:
                # Se não conseguiu dados live válidos, retorna dados históricos
                self.logger.debug(f"⚪ {symbol} {timeframe}: Sem dados live válidos, usando históricos")
                return historical_data
                
        except Exception as e:
            self.logger.error(f"Erro na busca de dados enhanced {symbol} {timeframe}: {e}")
            # Em caso de erro, retorna dados históricos normais
            return self.get_latest_data(symbol, timeframe)
    
    def _detect_database_config(self) -> tuple[str, str]:
        """Detecta configuração do banco COM DEBUG"""
        try:
            # Primeira prioridade: stream_db_path
            if hasattr(settings.database, 'stream_db_path'):
                db_path = settings.database.stream_db_path
                table_name = getattr(settings.database, 'stream_table', 'crypto_ohlc')
                self.logger.info(f"✅ Configuração STREAM detectada: {db_path}")
                return db_path, table_name
            
            # Segunda prioridade: market_data_db_path
            elif hasattr(settings.database, 'market_data_db_path'):
                db_path = settings.database.market_data_db_path
                table_name = settings.database.market_data_table
                self.logger.info(f"✅ Configuração MARKET_DATA detectada: {db_path}")
                return db_path, table_name
            
            # Terceira prioridade: configuração alternativa
            elif hasattr(settings.database, 'market_data_path'):
                db_path = settings.database.market_data_path
                table_name = getattr(settings.database, 'market_data_table', 'market_data')
                self.logger.info(f"✅ Configuração ALT detectada: {db_path}")
                return db_path, table_name
            
            # Quarta prioridade: configuração básica
            elif hasattr(settings.database, 'path'):
                db_path = settings.database.path
                table_name = getattr(settings.database, 'table', 'market_data')
                self.logger.info(f"✅ Configuração BÁSICA detectada: {db_path}")
                return db_path, table_name
            
            # Fallback para configuração padrão
            else:
                db_path = "data/market_data.db"
                table_name = "market_data"
                self.logger.warning(f"⚠️ Usando configuração PADRÃO: {db_path}")
                return db_path, table_name
                
        except Exception as e:
            # Fallback absoluto
            db_path = "data/market_data.db" 
            table_name = "market_data"
            self.logger.error(f"❌ Erro na detecção de configuração DB, usando padrão: {e}")
            return db_path, table_name
    
    def _validate_database_setup(self) -> bool:
        """Valida se o banco existe e está acessível COM TIMEOUT AGRESSIVO"""
        try:
            self.logger.info(f"🔍 Validando banco: {self.db_path}")
            
            # Verifica se arquivo existe
            if not os.path.exists(self.db_path):
                self.logger.error(f"❌ Arquivo do banco não existe: {self.db_path}")
                return False
            
            # Verifica se é acessível
            if not os.access(self.db_path, os.R_OK):
                self.logger.error(f"❌ Sem permissão de leitura: {self.db_path}")
                return False
            
            # Testa conexão RÁPIDA
            start_time = time.time()
            try:
                conn = sqlite3.connect(
                    self.db_path, 
                    timeout=0.5,  # TIMEOUT SUPER AGRESSIVO
                    check_same_thread=False
                )
                
                conn.execute("PRAGMA busy_timeout = 200")  # 0.2s
                cursor = conn.cursor()
                
                # Verifica se tabela existe
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (self.table_name,))
                table_exists = cursor.fetchone() is not None
                
                if not table_exists:
                    self.logger.error(f"❌ Tabela '{self.table_name}' não existe")
                    conn.close()
                    return False
                
                # Testa uma query simples
                cursor.execute(f"SELECT COUNT(*) FROM {self.table_name} LIMIT 1")
                count = cursor.fetchone()[0]
                
                conn.close()
                
                validation_time = time.time() - start_time
                self.logger.info(f"✅ Banco validado: {count} registros, {validation_time:.2f}s")
                
                if count == 0:
                    self.logger.warning(f"⚠️ Banco vazio: {count} registros")
                    return False
                
                return True
                
            except sqlite3.OperationalError as e:
                validation_time = time.time() - start_time
                self.logger.error(f"❌ Erro de acesso ao banco em {validation_time:.2f}s: {e}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Erro na validação do banco: {e}")
            return False
    
    def _get_optimized_connection(self):
        """Conexão SUPER otimizada anti-travamento"""
        try:
            conn = sqlite3.connect(
                self.db_path, 
                timeout=self.CONNECTION_TIMEOUT,
                check_same_thread=False,
                isolation_level=None  # Autocommit mode
            )
            
            # CONFIGURAÇÕES ANTI-TRAVAMENTO
            conn.execute("PRAGMA read_uncommitted = true")  # Não bloqueia leituras
            conn.execute("PRAGMA journal_mode = WAL")       # Write-Ahead Logging
            conn.execute("PRAGMA synchronous = OFF")        # Sem sincronização (mais rápido)
            conn.execute("PRAGMA cache_size = 2000")        # Cache menor
            conn.execute("PRAGMA temp_store = memory")      # Temp em memória
            conn.execute(f"PRAGMA busy_timeout = {self.QUERY_TIMEOUT * 1000}")  # Timeout global
            
            return conn
            
        except Exception as e:
            self.logger.error(f"❌ Erro na conexão otimizada: {e}")
            raise
    
    def get_valid_symbols_for_analysis(self) -> List[str]:
        """
        Retorna símbolos válidos COM PROTEÇÃO ANTI-TRAVAMENTO
        """
        self.logger.info("🔍 Iniciando busca de símbolos válidos...")
        
        # Se banco não foi validado, usa fallback
        if not self.database_validated:
            self.logger.warning("❌ Banco não validado, usando símbolos fallback")
            return self._get_fallback_symbols()
        
        start_time = time.time()
        
        try:
            # Query SUPER rápida com LIMIT baixo
            query = f"""
            SELECT DISTINCT symbol
            FROM {self.table_name}
            WHERE timeframe = '5m'
            AND timestamp > datetime('now', '-2 days')
            LIMIT 8
            """
            
            self.logger.debug(f"Query: {query}")
            
            symbols = []
            try:
                with self._get_optimized_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(query)
                    results = cursor.fetchall()
                    symbols = [row[0] for row in results]
                    
            except sqlite3.OperationalError as e:
                query_time = time.time() - start_time
                self.logger.error(f"❌ Query falhou em {query_time:.2f}s: {e}")
                return self._get_fallback_symbols()
            
            query_time = time.time() - start_time
            self.logger.info(f"✅ Query executada em {query_time:.2f}s, {len(symbols)} símbolos encontrados")
            
            # Se não achou símbolos, usa fallback
            if not symbols:
                self.logger.warning("⚠️ Nenhum símbolo encontrado no banco, usando fallback")
                return self._get_fallback_symbols()
            
            # Validação RÁPIDA (apenas 3 símbolos para ser rápido)
            valid_symbols = []
            for symbol in symbols[:8]:  # Limita a 3 para ser rápido
                if self._quick_symbol_validation(symbol):
                    valid_symbols.append(symbol)
                
                # Se já tem 3 válidos, para
                if len(valid_symbols) >= 8:
                    break
            
            total_time = time.time() - start_time
            
            if valid_symbols:
                self.logger.info(f"✅ {len(valid_symbols)} símbolos válidos em {total_time:.2f}s: {valid_symbols}")
                return valid_symbols
            else:
                self.logger.warning(f"⚠️ Nenhum símbolo válido em {total_time:.2f}s, usando fallback")
                return self._get_fallback_symbols()
            
        except Exception as e:
            total_time = time.time() - start_time
            self.logger.error(f"❌ Erro na busca de símbolos em {total_time:.2f}s: {e}")
            return self._get_fallback_symbols()
    
    def _quick_symbol_validation(self, symbol: str) -> bool:
        """Validação SUPER rápida de um símbolo"""
        try:
            start_time = time.time()
            
            quick_check = f"""
            SELECT COUNT(*) 
            FROM {self.table_name} 
            WHERE symbol = ? AND timeframe = '5m' 
            AND timestamp > datetime('now', '-1 days')
            LIMIT 1
            """
            
            with self._get_optimized_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(quick_check, (symbol,))
                count = cursor.fetchone()[0]
            
            validation_time = time.time() - start_time
            is_valid = count >= 50  # Pelo menos 50 registros recentes
            
            self.logger.debug(f"  {symbol}: {count} registros, válido: {is_valid}, {validation_time:.2f}s")
            return is_valid
                
        except Exception as e:
            validation_time = time.time() - start_time
            self.logger.debug(f"  {symbol}: Erro na validação em {validation_time:.2f}s: {e}")
            return False
    
    def _get_fallback_symbols(self) -> List[str]:
        """Símbolos de fallback quando banco não está disponível"""
        fallback_symbols = ["BTC", "ETH", "BNB"]
        self.logger.info(f"📋 Usando símbolos fallback: {fallback_symbols}")
        return fallback_symbols
    
    def _has_real_data(self, symbol: str, timeframe: str) -> bool:
        """Verifica se há dados reais da Binance no banco"""
        try:
            with sqlite3.connect(self.db_path, timeout=1) as conn:
                cursor = conn.cursor()
                
                # Verificar se há dados para o símbolo/timeframe
                cursor.execute(f"""
                    SELECT COUNT(*) FROM {self.table_name}
                    WHERE symbol = ? AND timeframe = ?
                """, (symbol, timeframe))
                
                count = cursor.fetchone()[0]
                
                if count == 0:
                    self.logger.warning(f"⚠️ Nenhum dado encontrado para {symbol} {timeframe}")
                    return False
                
                # Verificar se os dados são recentes (últimas 24h)
                cursor.execute(f"""
                    SELECT MAX(timestamp) FROM {self.table_name}
                    WHERE symbol = ? AND timeframe = ?
                """, (symbol, timeframe))
                
                last_timestamp = cursor.fetchone()[0]
                if last_timestamp:
                    last_update = datetime.fromisoformat(last_timestamp.replace('Z', '+00:00'))
                    now = datetime.now(timezone.utc)
                    
                    if now - last_update > timedelta(hours=24):
                        self.logger.warning(f"⚠️ Dados antigos para {symbol} {timeframe} - última atualização: {last_update}")
                        return False
                
                self.logger.info(f"✅ Dados reais encontrados para {symbol} {timeframe} - {count} registros")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Erro ao verificar dados reais para {symbol} {timeframe}: {e}")
            return False
    
    def get_latest_data(self, symbol: str, timeframe: str, limit: int = None) -> Optional[MarketData]:
        """Busca dados otimizados por timeframe - APENAS DADOS REAIS DA BINANCE"""
        
        if limit is None:
            # Limites otimizados por timeframe
            if timeframe == '5m':
                limit = 2000   # ~7 dias para ML
            elif timeframe == '15m':
                limit = 1000  # ~10 dias  
            elif timeframe == '1h':
                limit = 500  # ~20 dias
            else:
                limit = 1000   # fallback para ML
                
        if not self.database_validated:
            self.logger.warning(f"❌ Banco não validado, não é possível buscar dados para {symbol}")
            return None
        
        # VALIDACAO CRITICA: Verificar se há dados reais
        if not self._has_real_data(symbol, timeframe):
            self.logger.error(f"❌ SEM DADOS REAIS para {symbol} {timeframe} - Execute binance_data_collector.py primeiro!")
            return None
        
        start_time = time.time()
        
        try:
            # Query otimizada com LIMIT baixo
            query = f"""
            SELECT timestamp, open_price, high_price, low_price, close_price, volume
            FROM {self.table_name}
            WHERE symbol = ? AND timeframe = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """
            
            df = None
            try:
                with self._get_optimized_connection() as conn:
                    df = pd.read_sql_query(
                        query, 
                        conn, 
                        params=(symbol, timeframe, limit)
                    )
            except sqlite3.OperationalError as e:
                execution_time = time.time() - start_time
                self.logger.error(f"❌ Query de dados falhou para {symbol} {timeframe} em {execution_time:.2f}s: {e}")
                return None
            
            execution_time = time.time() - start_time
            
            if df.empty:
                self.logger.warning(f"⚠️ Nenhum dado para {symbol} {timeframe} em {execution_time:.2f}s")
                return None
            
            # Processamento rápido
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Converte tipos numéricos
            numeric_columns = ['open_price', 'high_price', 'low_price', 'close_price', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove linhas com dados inválidos
            df = df.dropna().reset_index(drop=True)
            
            self.logger.debug(f"✅ {symbol} {timeframe}: {len(df)} registros em {execution_time:.2f}s")
            
            result = MarketData(
                symbol=symbol,
                timeframe=timeframe,
                data=df,
                last_update=datetime.now()
            )
            
            # Limpeza de memória para evitar memory leak
            self._cleanup_memory()
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Erro ao buscar {symbol} {timeframe} em {execution_time:.2f}s: {e}")
            return None
        finally:
            # Força garbage collection para liberar memória
            gc.collect()
    
    def _cleanup_memory(self):
        """Limpeza de memória para evitar memory leak"""
        try:
            # Força garbage collection
            gc.collect()
            
            # Ajusta thresholds do garbage collector para ser mais agressivo
            gc.set_threshold(700, 10, 10)
            
            self.logger.debug("Memória limpa com sucesso")
        except Exception as e:
            self.logger.warning(f"Erro na limpeza de memória: {e}")
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Busca preço atual COM TIMEOUT SUPER AGRESSIVO
        """
        if not self.database_validated:
            return None
        
        try:
            # Query super rápida
            query_5m = f"""
            SELECT close_price
            FROM {self.table_name}
            WHERE symbol = ? AND timeframe = '5m'
            ORDER BY timestamp DESC
            LIMIT 1
            """
            
            with self._get_optimized_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query_5m, (symbol,))
                result = cursor.fetchone()
            
            return float(result[0]) if result else None
                
        except Exception as e:
            self.logger.debug(f"Erro ao buscar preço de {symbol}: {e}")
            return None
    
    def test_connection(self) -> Dict[str, Any]:
        """Testa conexão de forma SUPER RÁPIDA"""
        try:
            start_time = time.time()
            
            if not self.database_validated:
                return {
                    'status': 'error',
                    'database_path': self.db_path,
                    'main_table': self.table_name,
                    'error': 'Database validation failed during initialization',
                    'anti_lock_enabled': True
                }
            
            with self._get_optimized_connection() as conn:
                cursor = conn.cursor()
                
                # Teste simples
                cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
                table_count = cursor.fetchone()[0]
                
                # Conta registros rapidamente
                cursor.execute(f"SELECT COUNT(*) FROM {self.table_name} LIMIT 1")
                record_count = cursor.fetchone()[0]
            
            execution_time = time.time() - start_time
            
            return {
                'status': 'success',
                'database_path': self.db_path,
                'main_table': self.table_name,
                'main_table_exists': True,
                'total_tables': table_count,
                'sample_record_count': record_count,
                'connection_time': execution_time,
                'anti_lock_enabled': True,
                'validation_passed': True
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'database_path': self.db_path,
                'main_table': self.table_name,
                'error': str(e),
                'anti_lock_enabled': True,
                'validation_passed': False
            }
    
    # Métodos desabilitados para evitar travamentos
    def get_microstructure_for_validation(self, symbol: str, start_time: datetime, duration_minutes: int) -> Optional[pd.DataFrame]:
        """DESABILITADO: Microestrutura pode causar locks"""
        self.logger.debug(f"Microestrutura desabilitada para {symbol} (evita travamentos)")
        return None
    
    def test_microstructure_connection(self) -> Dict[str, Any]:
        """DESABILITADO: Testa microestrutura"""
        return {
            'table_exists': False,
            'has_data': False,
            'sample_data_count': 0,
            'status': 'disabled_to_avoid_locks'
        }
    
    def get_price_at_time(self, symbol: str, target_time: datetime, tolerance_minutes: int = 5) -> Optional[float]:
        """Busca preço histórico COM TIMEOUT"""
        if not self.database_validated:
            return None
        
        try:
            start_time = target_time - timedelta(minutes=tolerance_minutes)
            end_time = target_time + timedelta(minutes=tolerance_minutes)
            
            query = f"""
            SELECT close_price
            FROM {self.table_name}
            WHERE symbol = ? AND timeframe = '5m'
            AND timestamp BETWEEN ? AND ?
            ORDER BY ABS(julianday(timestamp) - julianday(?))
            LIMIT 1
            """
            
            with self._get_optimized_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (symbol, start_time.isoformat(), end_time.isoformat(), target_time.isoformat()))
                result = cursor.fetchone()
            
            return float(result[0]) if result else None
            
        except Exception as e:
            self.logger.debug(f"Erro ao buscar preço histórico de {symbol}: {e}")
            return None