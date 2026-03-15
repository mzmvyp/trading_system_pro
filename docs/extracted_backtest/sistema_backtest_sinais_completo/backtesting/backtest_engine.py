# -*- coding: utf-8 -*-
"""
Backtest Engine - Sistema de backtesting com dados reais
Testa a performance do sistema completo (Técnico + ML + LLM)
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
from core.analyzer import MultiTimeframeAnalyzer
from indicators.technical import TechnicalAnalyzer
from ml.ml_integration import MLSignalEnhancer, create_ml_enhancer
from llm.llm_integration import LLMSignalEnhancer, create_llm_enhancer
from config.settings import settings


@dataclass
class BacktestResult:
    """Resultado de um teste de backtest"""
    test_id: str
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    total_signals: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    avg_trade_duration_hours: float
    signals_details: List[Dict]
    created_at: str


@dataclass
class TradeResult:
    """Resultado de uma operação individual"""
    signal_id: str
    entry_time: str
    entry_price: float
    exit_time: str
    exit_price: float
    signal_type: str  # BUY/SELL
    exit_reason: str  # TARGET_1_HIT, TARGET_2_HIT, STOP_HIT, EXPIRED
    return_pct: float
    duration_hours: float
    confidence: float
    ml_prediction: Optional[str] = None
    llm_sentiment: Optional[float] = None


class BacktestEngine:
    """
    Engine de backtesting que simula o sistema completo
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.data_reader = DataReader()
        
        # Inicializa componentes do sistema
        self.technical_analyzer = TechnicalAnalyzer()
        
        # ML e LLM enhancers
        self.ml_enhancer = create_ml_enhancer(enabled=True, ml_weight=0.25)
        self.llm_enhancer = create_llm_enhancer(enabled=True, llm_weight=0.20)
        
        # Configurações do backtest
        self.default_lookback_days = 30  # Período para teste
        self.min_data_points = 50  # Mínimo de candles necessários (reduzido de 100)
        
        self.logger.info("Backtest Engine inicializado")
    
    def run_backtest(
        self,
        symbol: str = "BTCUSDT",
        timeframe: str = "1h",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        lookback_days: int = 30,
        test_count: int = 1000
    ) -> Dict:
        """Executa backtest sem callback de progresso (compatibilidade)"""
        return self.run_backtest_with_progress(
            symbol, timeframe, test_count, lookback_days, start_date, end_date, None
        )
    
    def run_backtest_with_progress(
        self,
        symbol: str = "BTCUSDT",
        timeframe: str = "1h",
        test_count: int = 1000,
        lookback_days: int = 30,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        progress_callback: Optional[callable] = None
    ) -> Dict:
        """
        Executa backtest completo
        
        Args:
            symbol: Símbolo para testar
            timeframe: Timeframe (1h)
            start_date: Data início (opcional)
            end_date: Data fim (opcional)
            lookback_days: Dias para buscar dados
            test_count: Número de testes para executar
        """
        
        self.logger.info(f"🚀 Iniciando backtest: {symbol} {timeframe}")
        self.logger.info(f"📊 Testes: {test_count}, Período: {lookback_days} dias")
        
        # 1. Busca dados históricos
        historical_data = self._get_historical_data(
            symbol, timeframe, lookback_days, start_date, end_date
        )
        
        if historical_data is None or len(historical_data) < self.min_data_points:
            return {
                'success': False,
                'error': f'Dados insuficientes: {len(historical_data) if historical_data is not None else 0} registros'
            }
        
        self.logger.info(f"✅ Dados carregados: {len(historical_data)} candles")
        
        # 2. Executa testes
        test_results = []
        successful_tests = 0
        
        for i in range(test_count):
            try:
                result = self._run_single_test(
                    symbol, timeframe, historical_data, i
                )
                
                if result:
                    test_results.append(result)
                    successful_tests += 1
                
                # Chama callback de progresso
                if progress_callback:
                    # Calcula estatísticas parciais
                    partial_stats = self._calculate_partial_statistics(test_results)
                    progress_callback(i + 1, test_count, partial_stats)
                
                # Log a cada 100 testes
                if (i + 1) % 100 == 0:
                    self.logger.info(f"📈 Testes concluídos: {i + 1}/{test_count}")
                
            except Exception as e:
                self.logger.error(f"❌ Erro no teste {i}: {e}")
                # Ainda chama callback mesmo em caso de erro
                if progress_callback:
                    partial_stats = self._calculate_partial_statistics(test_results)
                    progress_callback(i + 1, test_count, partial_stats)
        
        # 3. Calcula estatísticas finais
        final_stats = self._calculate_final_statistics(test_results)
        
        # 4. Salva resultados
        self._save_backtest_results(symbol, timeframe, test_results, final_stats)
        
        self.logger.info(f"✅ Backtest concluído: {successful_tests}/{test_count} testes bem-sucedidos")
        
        return {
            'success': True,
            'total_tests': test_count,
            'successful_tests': successful_tests,
            'statistics': final_stats,
            'results': test_results[:10]  # Apenas primeiros 10 para resposta
        }
    
    def _get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        lookback_days: int,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """Busca dados históricos do banco"""
        
        try:
            conn = sqlite3.connect(self.data_reader.db_path)
            
            # Calcula datas se não fornecidas
            if not end_date:
                end_date = datetime.now()
            elif isinstance(end_date, str):
                end_date = datetime.fromisoformat(end_date)
            
            if not start_date:
                start_date = end_date - timedelta(days=lookback_days)
            elif isinstance(start_date, str):
                start_date = datetime.fromisoformat(start_date)
            
            # Ajusta para buscar dados disponíveis (não exatas)
            # Busca dados dos últimos N dias disponíveis
            query_count = """
                SELECT COUNT(*) FROM crypto_ohlc 
                WHERE symbol = ? AND timeframe = ?
                AND timestamp >= ?
            """
            
            cursor = conn.cursor()
            start_date_str = start_date.isoformat() if hasattr(start_date, 'isoformat') else str(start_date)
            cursor.execute(query_count, (symbol, timeframe, start_date_str))
            available_count = cursor.fetchone()[0]
            
            # Se não há dados suficientes, busca mais para trás
            if available_count < int(lookback_days) * 20:  # 20 candles por dia em média
                # Busca dados dos últimos 30 dias disponíveis
                query_recent = """
                    SELECT timestamp FROM crypto_ohlc 
                    WHERE symbol = ? AND timeframe = ?
                    ORDER BY timestamp DESC LIMIT ?
                """
                cursor.execute(query_recent, (symbol, timeframe, int(lookback_days) * 24))
                recent_data = cursor.fetchall()
                
                if recent_data:
                    end_date = datetime.fromisoformat(recent_data[0][0])
                    start_date = datetime.fromisoformat(recent_data[-1][0])
                    self.logger.info(f"📊 Ajustando período para dados disponíveis: {start_date} a {end_date}")
            
            # Query para buscar dados
            query = """
                SELECT timestamp, open_price, high_price, low_price, close_price, volume
                FROM crypto_ohlc 
                WHERE symbol = ? AND timeframe = ? 
                AND timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp ASC
            """
            
            start_date_str = start_date.isoformat() if hasattr(start_date, 'isoformat') else str(start_date)
            end_date_str = end_date.isoformat() if hasattr(end_date, 'isoformat') else str(end_date)
            df = pd.read_sql_query(
                query, conn,
                params=(symbol, timeframe, start_date_str, end_date_str)
            )
            
            conn.close()
            
            if len(df) == 0:
                self.logger.warning(f"⚠️ Nenhum dado encontrado para {symbol} {timeframe}")
                return None
            
            # Converte timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao buscar dados históricos: {e}")
            return None
    
    def _run_single_test(
        self,
        symbol: str,
        timeframe: str,
        historical_data: pd.DataFrame,
        test_index: int
    ) -> Optional[BacktestResult]:
        """Executa um único teste de backtest"""
        
        try:
            # Seleciona ponto de início aleatório (últimos 80% dos dados)
            data_length = len(historical_data)
            
            # Para datasets pequenos, ajusta a lógica
            if data_length < 100:
                # Para datasets pequenos, usa pelo menos 30 registros para análise
                min_start = max(30, int(data_length * 0.3))
                max_start = int(data_length * 0.7)
            else:
                # Para datasets grandes, usa a lógica original
                min_start = max(50, int(data_length * 0.2))
                max_start = int(data_length * 0.8)
            
            # Garante que há pelo menos 12 horas para validação (reduzido de 24)
            if data_length - max_start < 12:
                max_start = data_length - 12
            
            # Garante que há pelo menos 30 registros para análise (reduzido de 50)
            if max_start < 30:
                max_start = 30
            
            if min_start >= max_start:
                return None
                
            start_index = np.random.randint(min_start, max_start)
            
            # Dados para análise (antes do ponto de teste)
            analysis_data = historical_data.iloc[:start_index]
            
            # Dados para validação (após o ponto de teste)
            validation_data = historical_data.iloc[start_index:]
            
            if len(validation_data) < 24:  # Pelo menos 24 horas para validar
                return None
            
            # 1. Analisa dados históricos para gerar sinal
            signal = self._generate_signal_from_data(
                symbol, timeframe, analysis_data
            )
            
            if not signal:
                return None
            
            # 2. Simula trade com dados de validação
            trade_result = self._simulate_trade(
                signal, validation_data
            )
            
            if not trade_result:
                return None
            
            # 3. Cria resultado do teste
            return BacktestResult(
                test_id=f"test_{test_index}_{int(datetime.now().timestamp())}",
                symbol=symbol,
                timeframe=timeframe,
                start_date=validation_data.index[0].isoformat(),
                end_date=validation_data.index[-1].isoformat(),
                total_signals=1,
                winning_trades=1 if trade_result.return_pct > 0 else 0,
                losing_trades=1 if trade_result.return_pct <= 0 else 0,
                win_rate=1.0 if trade_result.return_pct > 0 else 0.0,
                total_return=trade_result.return_pct,
                max_drawdown=abs(trade_result.return_pct) if trade_result.return_pct < 0 else 0,
                sharpe_ratio=self._calculate_sharpe_ratio([trade_result.return_pct]),
                avg_trade_duration_hours=trade_result.duration_hours,
                signals_details=[asdict(trade_result)],
                created_at=datetime.now().isoformat()
            )
            
        except Exception as e:
            self.logger.error(f"❌ Erro no teste individual: {e}")
            return None
    
    def _generate_signal_from_data(
        self,
        symbol: str,
        timeframe: str,
        data: pd.DataFrame
    ) -> Optional[Dict]:
        """Gera sinal usando dados históricos"""
        
        try:
            if len(data) < 30:  # Mínimo para análise técnica (reduzido de 50)
                return None
            
            # 1. Análise técnica
            from core.data_reader import MarketData
            market_data = MarketData(
                symbol=symbol,
                timeframe=timeframe,
                data=data,
                last_update=datetime.now()
            )
            analysis_results = self.technical_analyzer.analyze_all(market_data, timeframe)
            
            # Extrai sinais dos resultados
            technical_signals = []
            for indicator_name, result in analysis_results.items():
                for signal_data in result.signals:
                    technical_signals.append(signal_data)
            
            if not technical_signals:
                return None
            
            # Filtra sinais conflitantes e pega o mais forte
            buy_signals = [s for s in technical_signals if 'BUY' in s.get('signal_type', '')]
            sell_signals = [s for s in technical_signals if 'SELL' in s.get('signal_type', '')]
            
            # Se há sinais conflitantes, pega o com maior confiança
            if buy_signals and sell_signals:
                best_buy = max(buy_signals, key=lambda x: x.get('confidence', 0))
                best_sell = max(sell_signals, key=lambda x: x.get('confidence', 0))
                
                if best_buy.get('confidence', 0) > best_sell.get('confidence', 0):
                    best_signal = best_buy
                else:
                    best_signal = best_sell
            elif buy_signals:
                best_signal = max(buy_signals, key=lambda x: x.get('confidence', 0))
            elif sell_signals:
                best_signal = max(sell_signals, key=lambda x: x.get('confidence', 0))
            else:
                best_signal = max(technical_signals, key=lambda x: x.get('confidence', 0))
            
            # 2. Aplica ML enhancement
            if self.ml_enhancer and self.ml_enhancer.enabled:
                ml_result = self.ml_enhancer.enhance_signal(
                    symbol=symbol,
                    signal_type=best_signal.get('signal_type', 'BUY'),
                    technical_confidence=best_signal.get('confidence', 0.5),
                    timeframe=timeframe
                )
                
                if ml_result.get('ml_enabled'):
                    best_signal['ml_prediction'] = ml_result.get('ml_prediction')
                    best_signal['ml_confidence'] = ml_result.get('ml_confidence')
                    best_signal['confidence'] = ml_result.get('final_confidence', best_signal.get('confidence', 0.5))
            
            # 3. Aplica LLM enhancement (simulado para backtest)
            if self.llm_enhancer and self.llm_enhancer.enabled:
                # Para backtest, simula sentiment baseado no movimento recente
                recent_return = (data['close_price'].iloc[-1] - data['close_price'].iloc[-24]) / data['close_price'].iloc[-24]
                sentiment_score = np.clip(recent_return * 100, -100, 100)
                
                best_signal['llm_sentiment'] = sentiment_score
                # Ajusta confiança baseado no sentiment
                sentiment_factor = 1 + (sentiment_score / 200)  # -0.5 a +1.5
                best_signal['confidence'] *= sentiment_factor
            
            # Filtra por confiança mínima
            if best_signal.get('confidence', 0) < 0.6:  # Threshold mínimo
                return None
            
            return best_signal
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao gerar sinal: {e}")
            return None
    
    def _simulate_trade(
        self,
        signal: Dict,
        validation_data: pd.DataFrame
    ) -> Optional[TradeResult]:
        """Simula trade usando dados de validação"""
        
        try:
            entry_price = signal.get('entry_price', validation_data['close_price'].iloc[0])
            signal_type = signal.get('signal_type', 'BUY')
            confidence = signal.get('confidence', 0.5)
            
            # Calcula targets e stop loss baseado na confiança
            if signal_type == 'BUY':
                target_1 = entry_price * (1 + 0.01 * confidence)  # 1% * confiança
                target_2 = entry_price * (1 + 0.02 * confidence)  # 2% * confiança
                stop_loss = entry_price * (1 - 0.015 * confidence)  # 1.5% * confiança
            else:
                target_1 = entry_price * (1 - 0.01 * confidence)
                target_2 = entry_price * (1 - 0.02 * confidence)
                stop_loss = entry_price * (1 + 0.015 * confidence)
            
            # Simula trade
            entry_time = validation_data.index[0]
            exit_time = None
            exit_price = None
            exit_reason = "EXPIRED"
            
            # Procura por hits de targets/stop
            for i, (timestamp, row) in enumerate(validation_data.iterrows()):
                high = row['high_price']
                low = row['low_price']
                
                if signal_type == 'BUY':
                    # Verifica target 2 primeiro (mais alto)
                    if high >= target_2:
                        exit_time = timestamp
                        exit_price = target_2
                        exit_reason = "TARGET_2_HIT"
                        break
                    # Verifica target 1
                    elif high >= target_1:
                        exit_time = timestamp
                        exit_price = target_1
                        exit_reason = "TARGET_1_HIT"
                        break
                    # Verifica stop loss
                    elif low <= stop_loss:
                        exit_time = timestamp
                        exit_price = stop_loss
                        exit_reason = "STOP_HIT"
                        break
                else:  # SELL
                    if low <= target_2:
                        exit_time = timestamp
                        exit_price = target_2
                        exit_reason = "TARGET_2_HIT"
                        break
                    elif low <= target_1:
                        exit_time = timestamp
                        exit_price = target_1
                        exit_reason = "TARGET_1_HIT"
                        break
                    elif high >= stop_loss:
                        exit_time = timestamp
                        exit_price = stop_loss
                        exit_reason = "STOP_HIT"
                        break
            
            # Se não houve hit, usa último preço
            if exit_time is None:
                exit_time = validation_data.index[-1]
                exit_price = validation_data['close_price'].iloc[-1]
                exit_reason = "EXPIRED"
            
            # Calcula retorno
            if signal_type == 'BUY':
                return_pct = (exit_price - entry_price) / entry_price * 100
            else:
                return_pct = (entry_price - exit_price) / entry_price * 100
            
            # Duração em horas (mínimo 1 hora para evitar 0.0h)
            duration_seconds = (exit_time - entry_time).total_seconds()
            duration_hours = max(duration_seconds / 3600, 1.0)  # Mínimo 1 hora
            
            return TradeResult(
                signal_id=f"signal_{int(datetime.now().timestamp())}",
                entry_time=entry_time.isoformat(),
                entry_price=entry_price,
                exit_time=exit_time.isoformat(),
                exit_price=exit_price,
                signal_type=signal_type,
                exit_reason=exit_reason,
                return_pct=return_pct,
                duration_hours=duration_hours,
                confidence=confidence,
                ml_prediction=signal.get('ml_prediction'),
                llm_sentiment=signal.get('llm_sentiment')
            )
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao simular trade: {e}")
            return None
    
    def _calculate_partial_statistics(self, test_results: List[BacktestResult]) -> Dict:
        """Calcula estatísticas parciais durante o backtest"""
        if not test_results:
            return {
                'valid_tests': 0,
                'win_rate': 0,
                'avg_return': 0,
                'total_return': 0,
                'max_drawdown': 0
            }
        
        total_tests = len(test_results)
        total_wins = sum(r.winning_trades for r in test_results)
        returns = [r.total_return for r in test_results]
        
        win_rate = total_wins / total_tests if total_tests > 0 else 0
        avg_return = np.mean(returns) if returns else 0
        total_return = np.sum(returns) if returns else 0
        max_drawdown = max(r.max_drawdown for r in test_results) if test_results else 0
        
        return {
            'valid_tests': total_tests,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'total_return': total_return,
            'max_drawdown': max_drawdown
        }
    
    def _calculate_final_statistics(self, test_results: List[BacktestResult]) -> Dict:
        """Calcula estatísticas finais do backtest"""
        
        if not test_results:
            return {
                'total_tests': 0,
                'win_rate': 0,
                'avg_return': 0,
                'total_return': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'avg_duration': 0
            }
        
        # Estatísticas básicas
        total_tests = len(test_results)
        total_wins = sum(r.winning_trades for r in test_results)
        total_losses = sum(r.losing_trades for r in test_results)
        
        returns = [r.total_return for r in test_results]
        durations = [r.avg_trade_duration_hours for r in test_results]
        
        win_rate = total_wins / total_tests if total_tests > 0 else 0
        avg_return = np.mean(returns)
        total_return = np.sum(returns)
        max_drawdown = max(r.max_drawdown for r in test_results)
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        avg_duration = np.mean(durations)
        
        # Estatísticas de saída
        exit_reasons = {}
        for result in test_results:
            for detail in result.signals_details:
                reason = detail.get('exit_reason', 'UNKNOWN')
                exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        
        return {
            'total_tests': total_tests,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'avg_duration': avg_duration,
            'exit_reasons': exit_reasons,
            'returns_distribution': {
                'min': min(returns),
                'max': max(returns),
                'std': np.std(returns),
                'median': np.median(returns)
            }
        }
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calcula Sharpe ratio"""
        if not returns or len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        excess_returns = returns_array - 0.02  # Assume 2% risk-free rate
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        return np.mean(excess_returns) / np.std(excess_returns)
    
    def _save_backtest_results(
        self,
        symbol: str,
        timeframe: str,
        results: List[BacktestResult],
        statistics: Dict
    ):
        """Salva resultados do backtest"""
        
        try:
            # Cria diretório se não existir
            os.makedirs('backtesting/results', exist_ok=True)
            
            # Arquivo de resultados
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"backtesting/results/backtest_{symbol}_{timeframe}_{timestamp}.json"
            
            # Prepara dados para salvar
            save_data = {
                'metadata': {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'timestamp': timestamp,
                    'total_tests': len(results),
                    'statistics': statistics
                },
                'results': [asdict(result) for result in results]
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"💾 Resultados salvos em: {filename}")
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao salvar resultados: {e}")


def main():
    """Função principal para executar backtest"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Backtest Engine")
    parser.add_argument('--symbol', default='BTCUSDT', help='Símbolo para testar')
    parser.add_argument('--timeframe', default='1h', help='Timeframe')
    parser.add_argument('--tests', type=int, default=1000, help='Número de testes')
    parser.add_argument('--days', type=int, default=30, help='Dias de dados')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Executa backtest
    engine = BacktestEngine()
    result = engine.run_backtest(
        symbol=args.symbol,
        timeframe=args.timeframe,
        test_count=args.tests,
        lookback_days=args.days
    )
    
    if result['success']:
        stats = result['statistics']
        print(f"\n📊 RESULTADOS DO BACKTEST:")
        print(f"   Testes: {stats['total_tests']}")
        print(f"   Win Rate: {stats['win_rate']:.2%}")
        print(f"   Retorno Médio: {stats['avg_return']:.2f}%")
        print(f"   Retorno Total: {stats['total_return']:.2f}%")
        print(f"   Max Drawdown: {stats['max_drawdown']:.2f}%")
        print(f"   Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
        print(f"   Duração Média: {stats['avg_duration']:.1f}h")
    else:
        print(f"❌ Erro no backtest: {result.get('error', 'Unknown')}")


if __name__ == "__main__":
    main()
