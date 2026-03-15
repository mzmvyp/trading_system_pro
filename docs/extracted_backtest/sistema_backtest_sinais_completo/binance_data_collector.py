#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coletor de dados reais da Binance - SEM DADOS FICTÍCIOS
Sistema de produção que coleta dados reais via WebSocket da Binance
"""

import asyncio
import json
import logging
import sqlite3
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional
import websockets
from binance import AsyncClient
from binance.exceptions import BinanceAPIException

class BinanceDataCollector:
    """Coletor de dados reais da Binance via WebSocket"""
    
    def __init__(self, db_path: str = "data/crypto_stream.db"):
        self.db_path = db_path
        self.symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT", "UNIUSDT", "SOLUSDT", "MATICUSDT"]
        self.timeframes = ["1h", "4h", "1d"]
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Controle de execução
        self.running = False
        self.websocket = None
        
        # Estatísticas
        self.total_candles_received = 0
        self.total_candles_saved = 0
        self.last_update = None
        
        self.logger.info("=== COLETOR DE DADOS REAIS DA BINANCE INICIADO ===")
        self.logger.info("ATENCAO: APENAS DADOS REAIS - NENHUM DADO FICTICIO")
    
    def _setup_logging(self):
        """Configuração de logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('binance_collector.log'),
                logging.StreamHandler()
            ]
        )
    
    def _create_database(self):
        """Cria estrutura do banco de dados"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Tabela para dados OHLCV
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS crypto_ohlc (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    open_price REAL NOT NULL,
                    high_price REAL NOT NULL,
                    low_price REAL NOT NULL,
                    close_price REAL NOT NULL,
                    volume REAL NOT NULL,
                    inserted_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timeframe, timestamp)
                )
            """)
            
            # Tabela para microestrutura 1m
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS kline_microstructure_1m (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    kline_close_time INTEGER NOT NULL,
                    open_price REAL NOT NULL,
                    high_price REAL NOT NULL,
                    low_price REAL NOT NULL,
                    close_price REAL NOT NULL,
                    volume REAL NOT NULL,
                    inserted_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, kline_close_time)
                )
            """)
            
            conn.commit()
            conn.close()
            
            self.logger.info("Estrutura do banco de dados criada/verificada")
            
        except Exception as e:
            self.logger.error(f"Erro ao criar banco de dados: {e}")
            raise
    
    def _save_candle_data(self, symbol: str, timeframe: str, candle_data: Dict):
        """Salva dados de candle no banco"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Converter timestamp
            timestamp = datetime.fromtimestamp(candle_data['close_time'] / 1000, tz=timezone.utc)
            
            # Inserir dados OHLCV
            cursor.execute("""
                INSERT OR REPLACE INTO crypto_ohlc 
                (symbol, timeframe, timestamp, open_price, high_price, low_price, close_price, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol, timeframe, timestamp,
                float(candle_data['open']),
                float(candle_data['high']),
                float(candle_data['low']),
                float(candle_data['close']),
                float(candle_data['volume'])
            ))
            
            # Se for timeframe 1m, salvar também na tabela de microestrutura
            if timeframe == "1m":
                cursor.execute("""
                    INSERT OR REPLACE INTO kline_microstructure_1m 
                    (symbol, kline_close_time, open_price, high_price, low_price, close_price, volume, inserted_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol, candle_data['close_time'],
                    float(candle_data['open']),
                    float(candle_data['high']),
                    float(candle_data['low']),
                    float(candle_data['close']),
                    float(candle_data['volume']),
                    timestamp.isoformat()
                ))
            
            conn.commit()
            conn.close()
            
            self.total_candles_saved += 1
            
            # NOVO: Limpeza automática de dados antigos (manter apenas 4 anos)
            if self.total_candles_saved % 100 == 0:  # A cada 100 candles salvos
                self._cleanup_old_data()
            
        except Exception as e:
            self.logger.error(f"Erro ao salvar dados de {symbol} {timeframe}: {e}")
    
    def _cleanup_old_data(self):
        """Remove dados mais antigos que 4 anos para manter apenas 4 anos de dados"""
        try:
            from datetime import timedelta
            
            # Calcula data limite (4 anos atrás)
            cutoff_date = datetime.now() - timedelta(days=4*365)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Remove dados antigos da tabela principal
            cursor.execute("""
                DELETE FROM crypto_ohlc 
                WHERE timestamp < ?
            """, (cutoff_date,))
            
            deleted_main = cursor.rowcount
            
            # Remove dados antigos da tabela de microestrutura
            cutoff_timestamp = int(cutoff_date.timestamp() * 1000)
            cursor.execute("""
                DELETE FROM kline_microstructure_1m 
                WHERE kline_close_time < ?
            """, (cutoff_timestamp,))
            
            deleted_micro = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            if deleted_main > 0 or deleted_micro > 0:
                self.logger.info(f"🧹 Limpeza automática: {deleted_main} registros principais + {deleted_micro} microestrutura removidos (mantendo 4 anos)")
            
        except Exception as e:
            self.logger.error(f"Erro na limpeza automática: {e}")
    
    async def _get_historical_data(self, symbol: str, timeframe: str, limit: int = 1000):
        """Obtém dados históricos da Binance"""
        try:
            client = AsyncClient()
            
            self.logger.info(f"Obtendo dados históricos: {symbol} {timeframe}")
            
            # Obter dados históricos
            klines = await client.get_klines(
                symbol=symbol,
                interval=timeframe,
                limit=limit
            )
            
            await client.close_connection()
            
            # Processar e salvar dados
            for kline in klines:
                candle_data = {
                    'open_time': kline[0],
                    'open': kline[1],
                    'high': kline[2],
                    'low': kline[3],
                    'close': kline[4],
                    'volume': kline[5],
                    'close_time': kline[6],
                    'quote_asset_volume': kline[7],
                    'number_of_trades': kline[8],
                    'taker_buy_base_asset_volume': kline[9],
                    'taker_buy_quote_asset_volume': kline[10]
                }
                
                self._save_candle_data(symbol, timeframe, candle_data)
                self.total_candles_received += 1
            
            self.logger.info(f"Dados históricos obtidos: {symbol} {timeframe} - {len(klines)} candles")
            
        except Exception as e:
            self.logger.error(f"Erro ao obter dados históricos {symbol} {timeframe}: {e}")
    
    async def _websocket_handler(self, websocket, path):
        """Handler para WebSocket da Binance"""
        self.logger.info("WebSocket conectado - recebendo dados em tempo real")
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    
                    if 'stream' in data and 'data' in data:
                        stream = data['stream']
                        kline_data = data['data']
                        
                        # Extrair símbolo e timeframe do stream
                        symbol = kline_data['s']
                        timeframe = stream.split('@')[1].split('_')[0]
                        
                        # Verificar se é um candle fechado
                        if kline_data['x']:  # x indica se o candle está fechado
                            candle_data = {
                                'open_time': kline_data['t'],
                                'open': kline_data['o'],
                                'high': kline_data['h'],
                                'low': kline_data['l'],
                                'close': kline_data['c'],
                                'volume': kline_data['v'],
                                'close_time': kline_data['T']
                            }
                            
                            self._save_candle_data(symbol, timeframe, candle_data)
                            self.total_candles_received += 1
                            self.last_update = datetime.now()
                            
                            if self.total_candles_received % 100 == 0:
                                self.logger.info(f"Dados recebidos: {self.total_candles_received} candles salvos")
                
                except json.JSONDecodeError as e:
                    self.logger.error(f"Erro ao decodificar JSON: {e}")
                except Exception as e:
                    self.logger.error(f"Erro no processamento WebSocket: {e}")
        
        except websockets.exceptions.ConnectionClosed:
            self.logger.warning("Conexão WebSocket fechada")
        except Exception as e:
            self.logger.error(f"Erro no WebSocket: {e}")
    
    async def start_collection(self):
        """Inicia coleta de dados"""
        try:
            # Criar banco de dados
            self._create_database()
            
            # Obter dados históricos primeiro
            self.logger.info("=== OBTENDO DADOS HISTORICOS ===")
            for symbol in self.symbols:
                for timeframe in self.timeframes:
                    await self._get_historical_data(symbol, timeframe, 1000)
                    await asyncio.sleep(0.1)  # Rate limiting
            
            self.logger.info(f"=== DADOS HISTORICOS CONCLUIDOS ===")
            self.logger.info(f"Total candles salvos: {self.total_candles_saved}")
            
            # Iniciar WebSocket para dados em tempo real
            self.logger.info("=== INICIANDO WEBSOCKET TEMPO REAL ===")
            
            # Construir streams para WebSocket
            streams = []
            for symbol in self.symbols:
                for timeframe in self.timeframes:
                    streams.append(f"{symbol.lower()}@kline_{timeframe}")
            
            # URL do WebSocket da Binance
            ws_url = f"wss://stream.binance.com:9443/stream?streams={','.join(streams)}"
            
            self.running = True
            async with websockets.connect(ws_url) as websocket:
                self.websocket = websocket
                self.logger.info("WebSocket conectado - dados em tempo real ativos")
                
                async for message in websocket:
                    if not self.running:
                        break
                    
                    try:
                        data = json.loads(message)
                        
                        if 'stream' in data and 'data' in data:
                            stream = data['stream']
                            kline_data = data['data']
                            
                            symbol = kline_data['s']
                            timeframe = stream.split('@')[1].split('_')[0]
                            
                            if kline_data['x']:  # Candle fechado
                                candle_data = {
                                    'open_time': kline_data['t'],
                                    'open': kline_data['o'],
                                    'high': kline_data['h'],
                                    'low': kline_data['l'],
                                    'close': kline_data['c'],
                                    'volume': kline_data['v'],
                                    'close_time': kline_data['T']
                                }
                                
                                self._save_candle_data(symbol, timeframe, candle_data)
                                self.total_candles_received += 1
                                self.last_update = datetime.now()
                                
                                if self.total_candles_received % 50 == 0:
                                    self.logger.info(f"Tempo real: {self.total_candles_received} candles | Última atualização: {self.last_update}")
                    
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Erro JSON: {e}")
                    except Exception as e:
                        self.logger.error(f"Erro processamento: {e}")
        
        except Exception as e:
            self.logger.error(f"Erro na coleta de dados: {e}")
        finally:
            self.running = False
            self.logger.info("=== COLETOR FINALIZADO ===")

    async def start_collection_with_reconnect(self):
        """Inicia coleta com reconexão automática"""
        max_retries = 10
        retry_count = 0
        
        while self.running and retry_count < max_retries:
            try:
                await self.start_collection()
                retry_count = 0  # Reset on success
                self.logger.info("WebSocket reconectado com sucesso")
            except websockets.exceptions.ConnectionClosed:
                retry_count += 1
                wait_time = min(5 * retry_count, 60)  # Exponential backoff
                self.logger.warning(f"WebSocket desconectado, reconectando em {wait_time}s... (tentativa {retry_count})")
                await asyncio.sleep(wait_time)
            except Exception as e:
                retry_count += 1
                wait_time = min(10 * retry_count, 120)
                self.logger.error(f"Erro crítico: {e}, reconectando em {wait_time}s... (tentativa {retry_count})")
                await asyncio.sleep(wait_time)
        
        if retry_count >= max_retries:
            self.logger.error("Máximo de tentativas de reconexão atingido")
    
    def stop_collection(self):
        """Para a coleta de dados"""
        self.running = False
        if self.websocket:
            self.websocket.close()
        self.logger.info("Parando coleta de dados...")
    
    def get_stats(self):
        """Retorna estatísticas da coleta"""
        return {
            'total_candles_received': self.total_candles_received,
            'total_candles_saved': self.total_candles_saved,
            'last_update': self.last_update,
            'running': self.running
        }

async def main():
    """Função principal"""
    collector = BinanceDataCollector()
    
    try:
        await collector.start_collection_with_reconnect()
    except KeyboardInterrupt:
        collector.stop_collection()
        print("Coleta interrompida pelo usuário")

if __name__ == "__main__":
    asyncio.run(main())
