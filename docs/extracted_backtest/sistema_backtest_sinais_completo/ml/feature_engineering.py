# -*- coding: utf-8 -*-
"""
Feature Engineering - Preparação de features para ML
Extrai features dos indicadores técnicos já calculados pelo sistema
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from core.data_reader import DataReader
from indicators.technical import TechnicalAnalyzer


class FeatureEngineer:
    """
    Engenheiro de Features para Machine Learning
    
    Features geradas:
    - Indicadores técnicos (RSI, MACD, Bollinger)
    - Lag features (preços anteriores)
    - Rolling statistics (média, std, min, max)
    - Volume profile
    - Padrões de preço
    - Features temporais (hora, dia da semana)
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.data_reader = DataReader()
        self.technical_analyzer = TechnicalAnalyzer()
        
        self.logger.info("✅ Feature Engineer inicializado")
    
    def prepare_features(self, symbol: str, timeframe: str = "5m", lookback: int = 200) -> Optional[pd.DataFrame]:
        """
        Prepara features para um símbolo
        
        Args:
            symbol: Símbolo da crypto
            timeframe: Timeframe dos dados
            lookback: Número de candles para buscar
        
        Returns:
            DataFrame com features ou None se não houver dados suficientes
        """
        try:
            # Busca dados do mercado
            market_data = self.data_reader.get_latest_data(symbol, timeframe)
            
            if not market_data or not market_data.is_sufficient_data:
                self.logger.warning(f"⚠️ Dados insuficientes para {symbol}")
                self.logger.debug(f"   Market data: {market_data}")
                if market_data:
                    self.logger.debug(f"   Data length: {len(market_data.data) if market_data.data is not None else 0}")
                return None
            
            df = market_data.data.copy()
            
            if len(df) < lookback:
                self.logger.warning(f"Apenas {len(df)} candles disponíveis para {symbol} (necessário: {lookback})")
                return None
            
            # Usa últimos N candles
            df = df.tail(lookback).copy()
            
            # Calcula indicadores técnicos
            df = self._add_technical_indicators(df, timeframe)
            
            # Adiciona lag features
            df = self._add_lag_features(df)
            
            # Adiciona rolling statistics
            df = self._add_rolling_features(df)
            
            # Adiciona volume features
            df = self._add_volume_features(df)
            
            # Adiciona price patterns
            df = self._add_price_patterns(df)
            
            # Adiciona features temporais
            df = self._add_temporal_features(df)
            
            # Remove NaN
            df = df.dropna()
            
            if len(df) < 50:
                self.logger.warning(f"Apenas {len(df)} amostras após limpeza para {symbol}")
                return None
            
            self.logger.info(f"✅ {len(df)} amostras preparadas para {symbol} com {len(df.columns)} features")
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao preparar features para {symbol}: {e}")
            return None
    
    def _add_technical_indicators(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Adiciona indicadores técnicos"""
        # RSI
        delta = df['close_price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_fast = df['close_price'].ewm(span=12, adjust=False).mean()
        ema_slow = df['close_price'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close_price'].rolling(window=20).mean()
        df['bb_std'] = df['close_price'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (2 * df['bb_std'])
        df['bb_lower'] = df['bb_middle'] - (2 * df['bb_std'])
        df['bb_position'] = (df['close_price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        
        # ATR (Average True Range)
        df['prev_close'] = df['close_price'].shift(1)
        df['tr1'] = df['high_price'] - df['low_price']
        df['tr2'] = abs(df['high_price'] - df['prev_close'])
        df['tr3'] = abs(df['low_price'] - df['prev_close'])
        df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr'] = df['true_range'].ewm(span=14, adjust=False).mean()
        
        # EMA múltiplas
        for period in [9, 21, 50]:
            df[f'ema_{period}'] = df['close_price'].ewm(span=period, adjust=False).mean()
        
        # Remove colunas temporárias
        df.drop(['prev_close', 'tr1', 'tr2', 'tr3', 'true_range', 'bb_std'], axis=1, inplace=True, errors='ignore')
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features de lag (valores anteriores)"""
        # Preços anteriores
        for lag in [1, 2, 3, 5, 10]:
            df[f'close_lag_{lag}'] = df['close_price'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        
        # Retornos anteriores
        for lag in [1, 2, 3, 5]:
            df[f'return_lag_{lag}'] = df['close_price'].pct_change(lag).shift(1)
        
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona estatísticas rolling"""
        windows = [5, 10, 20, 50]
        
        for window in windows:
            # Rolling mean
            df[f'close_mean_{window}'] = df['close_price'].rolling(window=window).mean()
            
            # Rolling std
            df[f'close_std_{window}'] = df['close_price'].rolling(window=window).std()
            
            # Rolling min/max
            df[f'close_min_{window}'] = df['close_price'].rolling(window=window).min()
            df[f'close_max_{window}'] = df['close_price'].rolling(window=window).max()
            
            # Posição relativa no range
            df[f'close_position_{window}'] = (
                (df['close_price'] - df[f'close_min_{window}']) / 
                (df[f'close_max_{window}'] - df[f'close_min_{window}'] + 1e-10)
            )
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features de volume"""
        # Volume relativo
        df['volume_mean_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_mean_20'] + 1e-10)
        
        # VWAP
        typical_price = (df['high_price'] + df['low_price'] + df['close_price']) / 3
        df['vwap'] = (typical_price * df['volume']).rolling(window=20).sum() / (df['volume'].rolling(window=20).sum() + 1e-10)
        df['vwap_distance'] = (df['close_price'] - df['vwap']) / (df['vwap'] + 1e-10)
        
        # Volume Weighted Moving Average
        for window in [10, 20]:
            df[f'vwma_{window}'] = (df['close_price'] * df['volume']).rolling(window=window).sum() / (df['volume'].rolling(window=window).sum() + 1e-10)
        
        return df
    
    def _add_price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features de padrões de preço"""
        # Body size (tamanho do corpo do candle)
        df['body_size'] = abs(df['close_price'] - df['open_price'])
        df['body_size_pct'] = df['body_size'] / (df['open_price'] + 1e-10)
        
        # Upper/Lower shadow
        df['upper_shadow'] = df['high_price'] - df[['open_price', 'close_price']].max(axis=1)
        df['lower_shadow'] = df[['open_price', 'close_price']].min(axis=1) - df['low_price']
        
        # Direção do candle
        df['is_bullish'] = (df['close_price'] > df['open_price']).astype(int)
        
        # Range do candle
        df['candle_range'] = df['high_price'] - df['low_price']
        df['candle_range_pct'] = df['candle_range'] / (df['open_price'] + 1e-10)
        
        # Consecutivos
        df['consecutive_up'] = 0
        df['consecutive_down'] = 0
        
        for i in range(1, len(df)):
            if df.iloc[i]['is_bullish'] == 1:
                df.iloc[i, df.columns.get_loc('consecutive_up')] = df.iloc[i-1]['consecutive_up'] + 1
            else:
                df.iloc[i, df.columns.get_loc('consecutive_down')] = df.iloc[i-1]['consecutive_down'] + 1
        
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features temporais"""
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        
        # Cyclical encoding (para capturar periodicidade)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def create_target(self, df: pd.DataFrame, prediction_horizon: int = 12) -> pd.DataFrame:
        """
        Cria target para treinamento (movimento de preço futuro)
        
        Args:
            df: DataFrame com features
            prediction_horizon: Quantos períodos à frente prever (padrão: 12 = 1h em 5m)
        
        Returns:
            DataFrame com coluna 'target' (1 = alta, 0 = baixa)
        """
        # Calcula retorno futuro
        df['future_return'] = df['close_price'].shift(-prediction_horizon) / df['close_price'] - 1
        
        # Define threshold para movimento significativo (1.0% - mais realista)
        threshold = 0.01
        
        # Target binário: 1 se movimento positivo > threshold, 0 caso contrário
        df['target'] = (df['future_return'] > threshold).astype(int)
        
        # Remove linhas sem target (últimas N linhas)
        df = df[~df['target'].isna()].copy()
        
        # Remove future_return (vazamento de informação)
        df.drop('future_return', axis=1, inplace=True)
        
        return df
    
    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """Retorna nomes das features (excluindo target e metadados)"""
        exclude_cols = [
            'timestamp', 'symbol', 'timeframe', 'target',
            'open_price', 'high_price', 'low_price', 'close_price', 'volume'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        return feature_cols
    
    def prepare_training_data(
        self, 
        symbols: List[str], 
        timeframe: str = "5m",
        lookback: int = 1000,
        prediction_horizon: int = 12
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], List[str]]:
        """
        Prepara dados de treinamento para múltiplos símbolos
        
        Args:
            symbols: Lista de símbolos
            timeframe: Timeframe dos dados
            lookback: Candles para buscar por símbolo
            prediction_horizon: Períodos à frente para prever
        
        Returns:
            (X, y, feature_names) ou (None, None, []) se erro
        """
        all_data = []
        
        for symbol in symbols:
            self.logger.info(f"Preparando features para {symbol}...")
            
            df = self.prepare_features(symbol, timeframe, lookback)
            
            if df is not None and len(df) > 0:
                # Adiciona target
                df = self.create_target(df, prediction_horizon)
                
                if len(df) > 0:
                    df['symbol'] = symbol
                    all_data.append(df)
                    self.logger.info(f"✅ {len(df)} amostras de {symbol}")
            else:
                self.logger.warning(f"⚠️ Pulando {symbol} - sem dados suficientes")
        
        if not all_data:
            self.logger.error("❌ Nenhum dado preparado")
            return None, None, []
        
        # Combina todos os símbolos
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Separa features e target
        feature_names = self.get_feature_names(combined_df)
        X = combined_df[feature_names]
        y = combined_df['target']
        
        self.logger.info(f"✅ Dataset final: {len(X)} amostras, {len(feature_names)} features")
        self.logger.info(f"📊 Distribuição de classes: {y.value_counts().to_dict()}")
        
        return X, y, feature_names

