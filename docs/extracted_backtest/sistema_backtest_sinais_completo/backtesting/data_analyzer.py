# -*- coding: utf-8 -*-
"""
Data Analyzer - Sistema de análise e visualização de dados do stream
Verifica qualidade dos dados, gaps, e performance do sistema de coleta
"""

import logging
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import os
from dataclasses import dataclass, asdict

from core.data_reader import DataReader


@dataclass
class DataQualityReport:
    """Relatório de qualidade dos dados"""
    symbol: str
    timeframe: str
    total_records: int
    date_range: Tuple[str, str]
    missing_periods: int
    duplicate_records: int
    invalid_prices: int
    volume_anomalies: int
    data_quality_score: float
    gaps_details: List[Dict]
    created_at: str


@dataclass
class StreamPerformanceReport:
    """Relatório de performance do stream"""
    symbol: str
    timeframe: str
    collection_period: str
    expected_records: int
    actual_records: int
    collection_rate: float
    avg_delay_minutes: float
    max_gap_hours: float
    reliability_score: float
    created_at: str


class DataAnalyzer:
    """
    Analisador de dados do stream para verificar qualidade e performance
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.data_reader = DataReader()
        
        # Configurações
        self.timeframe_minutes = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '1h': 60,
            '4h': 240,
            '1d': 1440
        }
        
        self.logger.info("Data Analyzer inicializado")
    
    def analyze_data_quality(
        self,
        symbol: str = "BTCUSDT",
        timeframe: str = "1h",
        days: int = 7
    ) -> DataQualityReport:
        """
        Analisa qualidade dos dados para um símbolo/timeframe
        """
        
        self.logger.info(f"🔍 Analisando qualidade dos dados: {symbol} {timeframe}")
        
        # 1. Busca dados
        data = self._get_data_for_analysis(symbol, timeframe, days)
        
        if data is None or len(data) == 0:
            return DataQualityReport(
                symbol=symbol,
                timeframe=timeframe,
                total_records=0,
                date_range=("", ""),
                missing_periods=0,
                duplicate_records=0,
                invalid_prices=0,
                volume_anomalies=0,
                data_quality_score=0.0,
                gaps_details=[],
                created_at=datetime.now().isoformat()
            )
        
        # 2. Análises de qualidade
        total_records = len(data)
        date_range = (data.index[0].isoformat(), data.index[-1].isoformat())
        
        # Detecta períodos faltantes
        missing_periods, gaps_details = self._detect_missing_periods(data, timeframe)
        
        # Detecta duplicatas
        duplicate_records = self._detect_duplicates(data)
        
        # Detecta preços inválidos
        invalid_prices = self._detect_invalid_prices(data)
        
        # Detecta anomalias de volume
        volume_anomalies = self._detect_volume_anomalies(data)
        
        # Calcula score de qualidade (0-100)
        quality_score = self._calculate_quality_score(
            total_records, missing_periods, duplicate_records, 
            invalid_prices, volume_anomalies
        )
        
        return DataQualityReport(
            symbol=symbol,
            timeframe=timeframe,
            total_records=total_records,
            date_range=date_range,
            missing_periods=missing_periods,
            duplicate_records=duplicate_records,
            invalid_prices=invalid_prices,
            volume_anomalies=volume_anomalies,
            data_quality_score=quality_score,
            gaps_details=gaps_details,
            created_at=datetime.now().isoformat()
        )
    
    def analyze_stream_performance(
        self,
        symbol: str = "BTCUSDT",
        timeframe: str = "1h",
        hours: int = 24
    ) -> StreamPerformanceReport:
        """
        Analisa performance do stream de dados
        """
        
        self.logger.info(f"📊 Analisando performance do stream: {symbol} {timeframe}")
        
        # 1. Busca dados recentes
        data = self._get_data_for_analysis(symbol, timeframe, hours/24)
        
        if data is None or len(data) == 0:
            return StreamPerformanceReport(
                symbol=symbol,
                timeframe=timeframe,
                collection_period=f"{hours}h",
                expected_records=0,
                actual_records=0,
                collection_rate=0.0,
                avg_delay_minutes=0.0,
                max_gap_hours=0.0,
                reliability_score=0.0,
                created_at=datetime.now().isoformat()
            )
        
        # 2. Calcula métricas de performance
        timeframe_minutes = self.timeframe_minutes.get(timeframe, 60)
        expected_records = int(hours * 60 / timeframe_minutes)
        actual_records = len(data)
        collection_rate = (actual_records / expected_records) * 100 if expected_records > 0 else 0
        
        # Calcula atraso médio
        avg_delay = self._calculate_avg_delay(data, timeframe)
        
        # Calcula maior gap
        max_gap = self._calculate_max_gap(data, timeframe)
        
        # Calcula score de confiabilidade
        reliability_score = self._calculate_reliability_score(
            collection_rate, avg_delay, max_gap
        )
        
        return StreamPerformanceReport(
            symbol=symbol,
            timeframe=timeframe,
            collection_period=f"{hours}h",
            expected_records=expected_records,
            actual_records=actual_records,
            collection_rate=collection_rate,
            avg_delay_minutes=avg_delay,
            max_gap_hours=max_gap,
            reliability_score=reliability_score,
            created_at=datetime.now().isoformat()
        )
    
    def get_stream_overview(self) -> Dict:
        """
        Retorna visão geral do stream de dados
        """
        
        try:
            conn = sqlite3.connect(self.data_reader.db_path)
            cursor = conn.cursor()
            
            # Estatísticas gerais
            cursor.execute("SELECT COUNT(*) FROM crypto_ohlc")
            total_records = cursor.fetchone()[0]
            
            # Símbolos únicos
            cursor.execute("SELECT DISTINCT symbol FROM crypto_ohlc")
            symbols = [row[0] for row in cursor.fetchall()]
            
            # Timeframes únicos
            cursor.execute("SELECT DISTINCT timeframe FROM crypto_ohlc")
            timeframes = [row[0] for row in cursor.fetchall()]
            
            # Dados mais recentes
            cursor.execute("""
                SELECT symbol, timeframe, MAX(timestamp) as last_update, COUNT(*) as records
                FROM crypto_ohlc 
                GROUP BY symbol, timeframe
                ORDER BY last_update DESC
            """)
            
            latest_data = []
            for row in cursor.fetchall():
                latest_data.append({
                    'symbol': row[0],
                    'timeframe': row[1],
                    'last_update': row[2],
                    'records': row[3]
                })
            
            # Estatísticas de volume
            cursor.execute("""
                SELECT AVG(volume) as avg_volume, MAX(volume) as max_volume, MIN(volume) as min_volume
                FROM crypto_ohlc 
                WHERE timestamp > datetime('now', '-24 hours')
            """)
            
            volume_stats = cursor.fetchone()
            
            conn.close()
            
            return {
                'total_records': total_records,
                'symbols': symbols,
                'timeframes': timeframes,
                'latest_data': latest_data,
                'volume_stats': {
                    'avg_volume': volume_stats[0] if volume_stats[0] else 0,
                    'max_volume': volume_stats[1] if volume_stats[1] else 0,
                    'min_volume': volume_stats[2] if volume_stats[2] else 0
                },
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao obter visão geral: {e}")
            return {
                'total_records': 0,
                'symbols': [],
                'timeframes': [],
                'latest_data': [],
                'volume_stats': {},
                'error': str(e)
            }
    
    def _get_data_for_analysis(
        self,
        symbol: str,
        timeframe: str,
        days: float
    ) -> Optional[pd.DataFrame]:
        """Busca dados para análise"""
        
        try:
            conn = sqlite3.connect(self.data_reader.db_path)
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            query = """
                SELECT timestamp, open_price, high_price, low_price, close_price, volume
                FROM crypto_ohlc 
                WHERE symbol = ? AND timeframe = ? 
                AND timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp ASC
            """
            
            df = pd.read_sql_query(
                query, conn,
                params=(symbol, timeframe, start_date.isoformat(), end_date.isoformat())
            )
            
            conn.close()
            
            if len(df) == 0:
                return None
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao buscar dados: {e}")
            return None
    
    def _detect_missing_periods(
        self,
        data: pd.DataFrame,
        timeframe: str
    ) -> Tuple[int, List[Dict]]:
        """Detecta períodos faltantes nos dados"""
        
        try:
            timeframe_minutes = self.timeframe_minutes.get(timeframe, 60)
            
            # Cria série temporal esperada
            start_time = data.index[0]
            end_time = data.index[-1]
            expected_times = pd.date_range(
                start=start_time,
                end=end_time,
                freq=f'{timeframe_minutes}min'
            )
            
            # Encontra períodos faltantes
            actual_times = set(data.index)
            missing_times = [t for t in expected_times if t not in actual_times]
            
            # Agrupa gaps consecutivos
            gaps_details = []
            if missing_times:
                current_gap_start = missing_times[0]
                current_gap_end = missing_times[0]
                
                for i in range(1, len(missing_times)):
                    if (missing_times[i] - missing_times[i-1]).total_seconds() <= timeframe_minutes * 60:
                        current_gap_end = missing_times[i]
                    else:
                        # Gap terminou
                        gaps_details.append({
                            'start': current_gap_start.isoformat(),
                            'end': current_gap_end.isoformat(),
                            'duration_minutes': (current_gap_end - current_gap_start).total_seconds() / 60 + timeframe_minutes,
                            'missing_periods': int((current_gap_end - current_gap_start).total_seconds() / (timeframe_minutes * 60)) + 1
                        })
                        current_gap_start = missing_times[i]
                        current_gap_end = missing_times[i]
                
                # Adiciona último gap
                gaps_details.append({
                    'start': current_gap_start.isoformat(),
                    'end': current_gap_end.isoformat(),
                    'duration_minutes': (current_gap_end - current_gap_start).total_seconds() / 60 + timeframe_minutes,
                    'missing_periods': int((current_gap_end - current_gap_start).total_seconds() / (timeframe_minutes * 60)) + 1
                })
            
            return len(missing_times), gaps_details
            
        except Exception as e:
            self.logger.error(f"❌ Erro detectando períodos faltantes: {e}")
            return 0, []
    
    def _detect_duplicates(self, data: pd.DataFrame) -> int:
        """Detecta registros duplicados"""
        
        try:
            # Verifica duplicatas por timestamp
            duplicates = data.index.duplicated()
            return duplicates.sum()
            
        except Exception as e:
            self.logger.error(f"❌ Erro detectando duplicatas: {e}")
            return 0
    
    def _detect_invalid_prices(self, data: pd.DataFrame) -> int:
        """Detecta preços inválidos"""
        
        try:
            invalid_count = 0
            
            # Preços negativos ou zero
            invalid_count += (data['open_price'] <= 0).sum()
            invalid_count += (data['high_price'] <= 0).sum()
            invalid_count += (data['low_price'] <= 0).sum()
            invalid_count += (data['close_price'] <= 0).sum()
            
            # High < Low
            invalid_count += (data['high_price'] < data['low_price']).sum()
            
            # Preços fora dos limites OHLC
            invalid_count += (data['high_price'] < data['open_price']).sum()
            invalid_count += (data['high_price'] < data['close_price']).sum()
            invalid_count += (data['low_price'] > data['open_price']).sum()
            invalid_count += (data['low_price'] > data['close_price']).sum()
            
            return invalid_count
            
        except Exception as e:
            self.logger.error(f"❌ Erro detectando preços inválidos: {e}")
            return 0
    
    def _detect_volume_anomalies(self, data: pd.DataFrame) -> int:
        """Detecta anomalias de volume"""
        
        try:
            # Volume negativo
            negative_volume = (data['volume'] < 0).sum()
            
            # Volume zero (pode ser normal em alguns casos)
            zero_volume = (data['volume'] == 0).sum()
            
            # Volume extremamente alto (outliers)
            volume_mean = data['volume'].mean()
            volume_std = data['volume'].std()
            high_threshold = volume_mean + (5 * volume_std)  # 5 desvios padrão
            high_volume = (data['volume'] > high_threshold).sum()
            
            return negative_volume + high_volume  # Não conta zero volume como anomalia
            
        except Exception as e:
            self.logger.error(f"❌ Erro detectando anomalias de volume: {e}")
            return 0
    
    def _calculate_quality_score(
        self,
        total_records: int,
        missing_periods: int,
        duplicate_records: int,
        invalid_prices: int,
        volume_anomalies: int
    ) -> float:
        """Calcula score de qualidade (0-100)"""
        
        if total_records == 0:
            return 0.0
        
        # Penalidades
        missing_penalty = (missing_periods / total_records) * 50
        duplicate_penalty = (duplicate_records / total_records) * 30
        price_penalty = (invalid_prices / total_records) * 40
        volume_penalty = (volume_anomalies / total_records) * 20
        
        # Score base
        base_score = 100.0
        
        # Aplica penalidades
        final_score = base_score - missing_penalty - duplicate_penalty - price_penalty - volume_penalty
        
        return max(0.0, min(100.0, final_score))
    
    def _calculate_avg_delay(self, data: pd.DataFrame, timeframe: str) -> float:
        """Calcula atraso médio dos dados"""
        
        try:
            timeframe_minutes = self.timeframe_minutes.get(timeframe, 60)
            
            # Calcula diferença entre timestamp esperado e real
            delays = []
            
            for timestamp in data.index:
                # Timestamp esperado (início da hora)
                expected_time = timestamp.replace(minute=0, second=0, microsecond=0)
                
                # Calcula delay em minutos
                delay_minutes = (timestamp - expected_time).total_seconds() / 60
                delays.append(delay_minutes)
            
            return np.mean(delays) if delays else 0.0
            
        except Exception as e:
            self.logger.error(f"❌ Erro calculando atraso médio: {e}")
            return 0.0
    
    def _calculate_max_gap(self, data: pd.DataFrame, timeframe: str) -> float:
        """Calcula maior gap entre dados"""
        
        try:
            timeframe_minutes = self.timeframe_minutes.get(timeframe, 60)
            
            if len(data) < 2:
                return 0.0
            
            # Calcula diferenças entre timestamps consecutivos
            time_diffs = data.index.to_series().diff().dt.total_seconds() / 60
            
            # Remove primeira diferença (NaN)
            time_diffs = time_diffs.dropna()
            
            # Encontra maior gap
            max_gap_minutes = time_diffs.max()
            max_gap_hours = max_gap_minutes / 60
            
            return max_gap_hours
            
        except Exception as e:
            self.logger.error(f"❌ Erro calculando maior gap: {e}")
            return 0.0
    
    def _calculate_reliability_score(
        self,
        collection_rate: float,
        avg_delay: float,
        max_gap: float
    ) -> float:
        """Calcula score de confiabilidade (0-100)"""
        
        # Score baseado na taxa de coleta
        collection_score = min(100.0, collection_rate)
        
        # Penaliza atrasos
        delay_penalty = min(20.0, avg_delay * 2)  # Máximo 20 pontos de penalidade
        
        # Penaliza gaps
        gap_penalty = min(30.0, max_gap * 5)  # Máximo 30 pontos de penalidade
        
        final_score = collection_score - delay_penalty - gap_penalty
        
        return max(0.0, min(100.0, final_score))


def main():
    """Função principal para análise de dados"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Data Analyzer")
    parser.add_argument('--symbol', default='BTCUSDT', help='Símbolo para analisar')
    parser.add_argument('--timeframe', default='1h', help='Timeframe')
    parser.add_argument('--days', type=int, default=7, help='Dias para análise de qualidade')
    parser.add_argument('--hours', type=int, default=24, help='Horas para análise de performance')
    parser.add_argument('--overview', action='store_true', help='Mostrar visão geral')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Executa análise
    analyzer = DataAnalyzer()
    
    if args.overview:
        print("📊 VISÃO GERAL DO STREAM:")
        overview = analyzer.get_stream_overview()
        print(f"   Total de registros: {overview['total_records']}")
        print(f"   Símbolos: {overview['symbols']}")
        print(f"   Timeframes: {overview['timeframes']}")
        print(f"   Últimos dados: {len(overview['latest_data'])} combinações")
    else:
        print(f"🔍 ANÁLISE DE QUALIDADE: {args.symbol} {args.timeframe}")
        
        quality_report = analyzer.analyze_data_quality(
            symbol=args.symbol,
            timeframe=args.timeframe,
            days=args.days
        )
        
        print(f"   Registros: {quality_report.total_records}")
        print(f"   Períodos faltantes: {quality_report.missing_periods}")
        print(f"   Duplicatas: {quality_report.duplicate_records}")
        print(f"   Preços inválidos: {quality_report.invalid_prices}")
        print(f"   Anomalias de volume: {quality_report.volume_anomalies}")
        print(f"   Score de qualidade: {quality_report.data_quality_score:.1f}/100")
        
        print(f"\n📈 ANÁLISE DE PERFORMANCE: {args.symbol} {args.timeframe}")
        
        performance_report = analyzer.analyze_stream_performance(
            symbol=args.symbol,
            timeframe=args.timeframe,
            hours=args.hours
        )
        
        print(f"   Taxa de coleta: {performance_report.collection_rate:.1f}%")
        print(f"   Atraso médio: {performance_report.avg_delay_minutes:.1f} min")
        print(f"   Maior gap: {performance_report.max_gap_hours:.1f} h")
        print(f"   Score de confiabilidade: {performance_report.reliability_score:.1f}/100")


if __name__ == "__main__":
    main()
