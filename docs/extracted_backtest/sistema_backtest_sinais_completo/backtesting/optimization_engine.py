# -*- coding: utf-8 -*-
"""
Optimization Engine - Sistema de otimização contínua de indicadores
Testa diferentes configurações para encontrar o melhor setup
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
from itertools import product
import threading
import time

from core.data_reader import DataReader
from indicators.technical import TechnicalAnalyzer
from ml.ml_integration import create_ml_enhancer
from llm.llm_integration import create_llm_enhancer
from config.settings import settings


@dataclass
class OptimizationConfig:
    """Configuração de indicadores para teste"""
    rsi_period: int = 14
    rsi_overbought: float = 70
    rsi_oversold: float = 30
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    volume_ma_period: int = 20
    min_volume_ratio: float = 1.2
    confidence_threshold: float = 0.6


@dataclass
class OptimizationResult:
    """Resultado de uma configuração testada"""
    config: OptimizationConfig
    score: float
    win_rate: float
    avg_return: float
    sharpe_ratio: float
    total_trades: int
    max_drawdown: float
    test_date: str
    symbol: str
    timeframe: str


class OptimizationEngine:
    """
    Engine de otimização que testa diferentes configurações de indicadores
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.data_reader = DataReader()
        
        # Configurações de otimização
        self.optimization_ranges = {
            'rsi_period': [10, 14, 18, 22],
            'rsi_overbought': [65, 70, 75, 80],
            'rsi_oversold': [20, 25, 30, 35],
            'macd_fast': [8, 10, 12, 15],
            'macd_slow': [20, 24, 26, 30],
            'macd_signal': [7, 9, 11, 13],
            'bollinger_period': [15, 20, 25, 30],
            'bollinger_std': [1.5, 2.0, 2.5, 3.0],
            'confidence_threshold': [0.55, 0.6, 0.65, 0.7, 0.75]
        }
        
        # Resultados
        self.results: List[OptimizationResult] = []
        self.best_config: Optional[OptimizationConfig] = None
        self.is_running = False
        self.optimization_thread: Optional[threading.Thread] = None
        
        # Configurações do teste
        self.test_period_days = 7  # Testa últimos 7 dias
        self.min_trades_for_valid = 10  # Mínimo de trades para considerar válido
        
        # Status para dashboard
        self.optimization_status = {
            'is_running': False,
            'total_tests': 0,
            'current_test': 0,
            'best_score': 0.0,
            'last_update': None,
            'current_symbol': None,
            'current_timeframe': None,
            'configs_tested': [],
            'errors': []
        }
        
        self.logger.info("Optimization Engine inicializado")
    
    def start_continuous_optimization(
        self,
        symbol: str = "BTCUSDT",
        timeframe: str = "1h",
        interval_hours: int = 6
    ):
        """Inicia otimização contínua em background"""
        
        if self.is_running:
            self.logger.warning("⚠️ Otimização já está rodando")
            return
        
        self.is_running = True
        
        def optimization_loop():
            while self.is_running:
                try:
                    self.logger.info(f"🔄 Iniciando ciclo de otimização: {symbol} {timeframe}")
                    
                    # Atualiza status
                    self.optimization_status.update({
                        'is_running': True,
                        'current_symbol': symbol,
                        'current_timeframe': timeframe,
                        'total_tests': 0,
                        'current_test': 0,
                        'last_update': datetime.now().isoformat()
                    })
                    
                    # Executa otimização com callback de progresso
                    def progress_callback(current, total, best_score):
                        self.optimization_status.update({
                            'current_test': current,
                            'total_tests': total,
                            'best_score': best_score or 0.0,
                            'last_update': datetime.now().isoformat()
                        })
                    
                    result = self.run_optimization_with_progress(
                        symbol=symbol,
                        timeframe=timeframe,
                        max_configs=50,  # Testa até 50 configurações por ciclo
                        recent_days=self.test_period_days,
                        progress_callback=progress_callback
                    )
                    
                    if result['success']:
                        self.logger.info(f"✅ Otimização concluída: {result['best_score']:.3f}")
                        
                        # Atualiza melhor configuração
                        if result['best_config']:
                            self.best_config = result['best_config']
                            self._save_best_config(symbol, timeframe)
                            
                        # Adiciona aos resultados testados
                        if result['results']:
                            self.optimization_status['configs_tested'].extend([
                                {
                                    'score': r.score,
                                    'win_rate': r.win_rate,
                                    'config': r.config.__dict__
                                } for r in result['results'][:5]  # Top 5
                            ])
                    
                    # Aguarda próximo ciclo
                    self.logger.info(f"⏰ Próximo ciclo em {interval_hours}h")
                    time.sleep(interval_hours * 3600)
                    
                except Exception as e:
                    error_msg = f"Erro no ciclo de otimização: {e}"
                    self.logger.error(f"❌ {error_msg}")
                    self.optimization_status['errors'].append({
                        'timestamp': datetime.now().isoformat(),
                        'error': error_msg
                    })
                    time.sleep(3600)  # Aguarda 1h em caso de erro
        
        # Inicia thread
        self.optimization_thread = threading.Thread(target=optimization_loop, daemon=True)
        self.optimization_thread.start()
        
        self.logger.info(f"🚀 Otimização contínua iniciada: {interval_hours}h de intervalo")
    
    def stop_continuous_optimization(self):
        """Para otimização contínua"""
        self.is_running = False
        self.optimization_status['is_running'] = False
        if self.optimization_thread:
            self.optimization_thread.join(timeout=10)
        self.logger.info("🛑 Otimização contínua parada")
    
    def get_optimization_status(self) -> Dict:
        """Retorna status atual da otimização para o dashboard"""
        return self.optimization_status.copy()
    
    def run_optimization(
        self,
        symbol: str = "BTCUSDT",
        timeframe: str = "1h",
        max_configs: int = 100,
        recent_days: int = 7
    ) -> Dict:
        """Executa otimização sem callback de progresso (compatibilidade)"""
        return self.run_optimization_with_progress(
            symbol, timeframe, max_configs, recent_days, None
        )
    
    def run_optimization_with_progress(
        self,
        symbol: str = "BTCUSDT",
        timeframe: str = "1h",
        max_configs: int = 100,
        recent_days: int = 7,
        progress_callback: Optional[callable] = None
    ) -> Dict:
        """
        Executa otimização de indicadores
        
        Args:
            symbol: Símbolo para otimizar
            timeframe: Timeframe
            max_configs: Máximo de configurações para testar
            recent_days: Dias recentes para testar
        """
        
        self.logger.info(f"🔍 Iniciando otimização: {symbol} {timeframe}")
        self.logger.info(f"📊 Configurações: {max_configs}, Período: {recent_days} dias")
        
        # 1. Busca dados recentes
        historical_data = self._get_recent_data(symbol, timeframe, recent_days)
        
        if historical_data is None or len(historical_data) < 100:
            return {
                'success': False,
                'error': f'Dados insuficientes: {len(historical_data) if historical_data is not None else 0} registros'
            }
        
        self.logger.info(f"✅ Dados carregados: {len(historical_data)} candles")
        
        # 2. Gera configurações para testar
        configs_to_test = self._generate_configs(max_configs)
        
        # 3. Testa cada configuração
        results = []
        successful_tests = 0
        best_score = 0
        
        for i, config in enumerate(configs_to_test):
            try:
                result = self._test_configuration(
                    config, symbol, timeframe, historical_data
                )
                
                if result and result.total_trades >= self.min_trades_for_valid:
                    results.append(result)
                    successful_tests += 1
                    
                    # Atualiza melhor score
                    if result.score > best_score:
                        best_score = result.score
                
                # Chama callback de progresso
                if progress_callback:
                    progress_callback(i + 1, len(configs_to_test), best_score)
                
                if (i + 1) % 10 == 0:
                    self.logger.info(f"📈 Configurações testadas: {i + 1}/{len(configs_to_test)}")
                
            except Exception as e:
                self.logger.error(f"❌ Erro testando configuração {i}: {e}")
                # Ainda chama callback mesmo em caso de erro
                if progress_callback:
                    progress_callback(i + 1, len(configs_to_test), best_score)
        
        # 4. Encontra melhor configuração
        best_result = None
        if results:
            best_result = max(results, key=lambda x: x.score)
            self.results.extend(results)
        
        # 5. Salva resultados
        self._save_optimization_results(symbol, timeframe, results)
        
        self.logger.info(f"✅ Otimização concluída: {successful_tests}/{len(configs_to_test)} configurações válidas")
        
        return {
            'success': True,
            'total_configs': len(configs_to_test),
            'successful_configs': successful_tests,
            'best_config': best_result.config if best_result else None,
            'best_score': best_result.score if best_result else 0,
            'results': results[:10]  # Apenas primeiros 10 para resposta
        }
    
    def _get_recent_data(
        self,
        symbol: str,
        timeframe: str,
        days: int
    ) -> Optional[pd.DataFrame]:
        """Busca dados recentes"""
        
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
            self.logger.error(f"❌ Erro ao buscar dados recentes: {e}")
            return None
    
    def _generate_configs(self, max_configs: int) -> List[OptimizationConfig]:
        """Gera configurações aleatórias para testar"""
        
        configs = []
        
        # Gera configurações aleatórias
        for _ in range(max_configs):
            config = OptimizationConfig(
                rsi_period=np.random.choice(self.optimization_ranges['rsi_period']),
                rsi_overbought=np.random.choice(self.optimization_ranges['rsi_overbought']),
                rsi_oversold=np.random.choice(self.optimization_ranges['rsi_oversold']),
                macd_fast=np.random.choice(self.optimization_ranges['macd_fast']),
                macd_slow=np.random.choice(self.optimization_ranges['macd_slow']),
                macd_signal=np.random.choice(self.optimization_ranges['macd_signal']),
                bollinger_period=np.random.choice(self.optimization_ranges['bollinger_period']),
                bollinger_std=np.random.choice(self.optimization_ranges['bollinger_std']),
                confidence_threshold=np.random.choice(self.optimization_ranges['confidence_threshold'])
            )
            
            # Validação básica
            if (config.macd_fast < config.macd_slow and 
                config.rsi_oversold < config.rsi_overbought):
                configs.append(config)
        
        return configs
    
    def _test_configuration(
        self,
        config: OptimizationConfig,
        symbol: str,
        timeframe: str,
        data: pd.DataFrame
    ) -> Optional[OptimizationResult]:
        """Testa uma configuração específica"""
        
        try:
            # Cria analyzer com configuração personalizada
            analyzer = TechnicalAnalyzer()
            
            # Aplica configuração (simplificado - em produção seria mais complexo)
            # Por agora, usa a configuração padrão mas com threshold personalizado
            
            # Simula trades usando a configuração
            trades = self._simulate_trades_with_config(config, data)
            
            if len(trades) < self.min_trades_for_valid:
                return None
            
            # Calcula métricas
            returns = [trade['return_pct'] for trade in trades]
            wins = [r for r in returns if r > 0]
            
            win_rate = len(wins) / len(returns) if returns else 0
            avg_return = np.mean(returns) if returns else 0
            total_return = np.sum(returns) if returns else 0
            max_drawdown = self._calculate_max_drawdown(returns)
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            
            # Score composto (combina várias métricas)
            score = (
                win_rate * 0.3 +  # 30% peso no win rate
                (avg_return / 100) * 0.3 +  # 30% peso no retorno médio
                sharpe_ratio * 0.2 +  # 20% peso no sharpe
                (1 - max_drawdown / 100) * 0.2  # 20% peso na redução de drawdown
            )
            
            return OptimizationResult(
                config=config,
                score=score,
                win_rate=win_rate,
                avg_return=avg_return,
                sharpe_ratio=sharpe_ratio,
                total_trades=len(trades),
                max_drawdown=max_drawdown,
                test_date=datetime.now().isoformat(),
                symbol=symbol,
                timeframe=timeframe
            )
            
        except Exception as e:
            self.logger.error(f"❌ Erro testando configuração: {e}")
            return None
    
    def _simulate_trades_with_config(
        self,
        config: OptimizationConfig,
        data: pd.DataFrame
    ) -> List[Dict]:
        """Simula trades usando uma configuração específica"""
        
        trades = []
        
        try:
            # Implementação simplificada - em produção seria mais sofisticada
            # Por agora, simula trades baseado em movimentos de preço
            
            for i in range(24, len(data) - 24):  # Deixa margem para análise e validação
                
                # Dados para análise
                analysis_data = data.iloc[i-24:i]
                current_price = data.iloc[i]['close_price']
                
                # Simula sinal baseado em RSI simples
                prices = analysis_data['close_price']
                rsi = self._calculate_rsi(prices, config.rsi_period)
                
                if len(rsi) < 2:
                    continue
                
                current_rsi = rsi.iloc[-1]
                
                # Condições de entrada (simplificadas)
                if (current_rsi < config.rsi_oversold and 
                    current_price > analysis_data['close_price'].iloc[-2]):  # BUY
                    
                    # Simula trade
                    entry_price = current_price
                    target_1 = entry_price * (1 + 0.01)  # 1% target
                    target_2 = entry_price * (1 + 0.02)  # 2% target
                    stop_loss = entry_price * (1 - 0.015)  # 1.5% stop
                    
                    # Procura por exit
                    exit_price, exit_reason = self._find_exit_price(
                        data.iloc[i:i+24], target_1, target_2, stop_loss, 'BUY'
                    )
                    
                    if exit_price:
                        return_pct = (exit_price - entry_price) / entry_price * 100
                        
                        trades.append({
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'return_pct': return_pct,
                            'exit_reason': exit_reason,
                            'confidence': 0.6  # Simulado
                        })
                
                elif (current_rsi > config.rsi_overbought and 
                      current_price < analysis_data['close_price'].iloc[-2]):  # SELL
                    
                    entry_price = current_price
                    target_1 = entry_price * (1 - 0.01)
                    target_2 = entry_price * (1 - 0.02)
                    stop_loss = entry_price * (1 + 0.015)
                    
                    exit_price, exit_reason = self._find_exit_price(
                        data.iloc[i:i+24], target_1, target_2, stop_loss, 'SELL'
                    )
                    
                    if exit_price:
                        return_pct = (entry_price - exit_price) / entry_price * 100
                        
                        trades.append({
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'return_pct': return_pct,
                            'exit_reason': exit_reason,
                            'confidence': 0.6
                        })
            
            return trades
            
        except Exception as e:
            self.logger.error(f"❌ Erro simulando trades: {e}")
            return []
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calcula RSI"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except:
            return pd.Series()
    
    def _find_exit_price(
        self,
        data: pd.DataFrame,
        target_1: float,
        target_2: float,
        stop_loss: float,
        signal_type: str
    ) -> Tuple[Optional[float], str]:
        """Encontra preço de saída"""
        
        try:
            for _, row in data.iterrows():
                high = row['high_price']
                low = row['low_price']
                
                if signal_type == 'BUY':
                    if high >= target_2:
                        return target_2, 'TARGET_2_HIT'
                    elif high >= target_1:
                        return target_1, 'TARGET_1_HIT'
                    elif low <= stop_loss:
                        return stop_loss, 'STOP_HIT'
                else:  # SELL
                    if low <= target_2:
                        return target_2, 'TARGET_2_HIT'
                    elif low <= target_1:
                        return target_1, 'TARGET_1_HIT'
                    elif high >= stop_loss:
                        return stop_loss, 'STOP_HIT'
            
            # Se não houve hit, usa último preço
            return data['close_price'].iloc[-1], 'EXPIRED'
            
        except Exception as e:
            self.logger.error(f"❌ Erro encontrando exit price: {e}")
            return None, 'ERROR'
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calcula máximo drawdown"""
        if not returns:
            return 0.0
        
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        
        return abs(np.min(drawdown)) if len(drawdown) > 0 else 0.0
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calcula Sharpe ratio"""
        if not returns or len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        excess_returns = returns_array - 0.02  # Assume 2% risk-free rate
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        return np.mean(excess_returns) / np.std(excess_returns)
    
    def _save_optimization_results(
        self,
        symbol: str,
        timeframe: str,
        results: List[OptimizationResult]
    ):
        """Salva resultados da otimização"""
        
        try:
            os.makedirs('backtesting/optimization', exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"backtesting/optimization/optimization_{symbol}_{timeframe}_{timestamp}.json"
            
            save_data = {
                'metadata': {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'timestamp': timestamp,
                    'total_configs': len(results)
                },
                'results': [asdict(result) for result in results]
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"💾 Resultados salvos em: {filename}")
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao salvar resultados: {e}")
    
    def _save_best_config(self, symbol: str, timeframe: str):
        """Salva melhor configuração"""
        
        try:
            if not self.best_config:
                return
            
            os.makedirs('backtesting/optimization', exist_ok=True)
            
            filename = f"backtesting/optimization/best_config_{symbol}_{timeframe}.json"
            
            save_data = {
                'symbol': symbol,
                'timeframe': timeframe,
                'config': asdict(self.best_config),
                'updated_at': datetime.now().isoformat()
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"💾 Melhor configuração salva: {filename}")
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao salvar melhor configuração: {e}")
    
    def get_current_best_config(self) -> Optional[OptimizationConfig]:
        """Retorna melhor configuração atual"""
        return self.best_config
    
    def get_optimization_status(self) -> Dict:
        """Retorna status da otimização"""
        return {
            'is_running': self.is_running,
            'total_tests': len(self.results),
            'best_config': asdict(self.best_config) if self.best_config else None,
            'last_update': self.results[-1].test_date if self.results else None
        }


def main():
    """Função principal para executar otimização"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimization Engine")
    parser.add_argument('--symbol', default='BTCUSDT', help='Símbolo para otimizar')
    parser.add_argument('--timeframe', default='1h', help='Timeframe')
    parser.add_argument('--configs', type=int, default=50, help='Número de configurações')
    parser.add_argument('--days', type=int, default=7, help='Dias de dados')
    parser.add_argument('--continuous', action='store_true', help='Otimização contínua')
    parser.add_argument('--interval', type=int, default=6, help='Intervalo em horas para contínua')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Executa otimização
    engine = OptimizationEngine()
    
    if args.continuous:
        print("🚀 Iniciando otimização contínua...")
        engine.start_continuous_optimization(
            symbol=args.symbol,
            timeframe=args.timeframe,
            interval_hours=args.interval
        )
        
        try:
            while True:
                time.sleep(60)  # Aguarda 1 minuto
        except KeyboardInterrupt:
            print("\n🛑 Parando otimização contínua...")
            engine.stop_continuous_optimization()
    else:
        result = engine.run_optimization(
            symbol=args.symbol,
            timeframe=args.timeframe,
            max_configs=args.configs,
            recent_days=args.days
        )
        
        if result['success']:
            print(f"\n📊 RESULTADOS DA OTIMIZAÇÃO:")
            print(f"   Configurações testadas: {result['total_configs']}")
            print(f"   Configurações válidas: {result['successful_configs']}")
            print(f"   Melhor score: {result['best_score']:.3f}")
        else:
            print(f"❌ Erro na otimização: {result.get('error', 'Unknown')}")


if __name__ == "__main__":
    main()
