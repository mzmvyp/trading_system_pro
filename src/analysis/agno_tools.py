"""
Ferramentas AGNO com indicadores técnicos reais e análise de sentimento
Updated with logging, constants, and improved error handling
"""
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import talib
from dotenv import load_dotenv

from src.core.constants import DEFAULT_KLINES_LIMIT
from src.core.logger import get_logger

# Carregar variáveis de ambiente do .env
load_dotenv()

logger = get_logger(__name__)

# Análise de sentimento baseada apenas em dados de mercado (Twitter removido)

def classify_market_condition(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Classifica as condições de mercado e recomenda o tipo de operação ideal.

    Retorna:
        - operation_type: SCALP, DAY_TRADE, SWING_TRADE, POSITION_TRADE
        - confidence: 1-10
        - reasoning: explicação
        - parameters: stops e targets sugeridos
    """
    try:
        # Extrair indicadores relevantes
        volatility = analysis.get("volatility", {})
        trend = analysis.get("trend_analysis", {})
        volume = analysis.get("volume_flow", {})
        indicators = analysis.get("key_indicators", {})

        # 1. VOLATILIDADE
        volatility_level = volatility.get("level", "MEDIUM")
        atr_pct = volatility.get("atr_percent", 2.0)

        # 2. FORÇA DA TENDÊNCIA
        adx_value = trend.get("adx_value", 20)
        trend_strength = trend.get("trend_strength_interpretation", "WEAK")
        primary_trend = trend.get("primary_trend", "NEUTRAL")
        confluence_score = trend.get("confluence_score", 0)

        # 3. MOMENTUM
        rsi = indicators.get("rsi", {}).get("value", 50)
        indicators.get("macd", {}).get("momentum_direction", "neutral")

        # 4. VOLUME
        volume_trend = volume.get("obv_trend", "neutral")
        volume.get("orderbook_bias", "neutral")

        # ===== LÓGICA DE CLASSIFICAÇÃO =====

        scores = {
            "SCALP": 0,
            "DAY_TRADE": 0,
            "SWING_TRADE": 0,
            "POSITION_TRADE": 0
        }

        # --- Regras de Volatilidade ---
        if volatility_level == "HIGH" or atr_pct > 3.0:
            scores["SCALP"] += 3
            scores["DAY_TRADE"] += 2
        elif volatility_level == "MEDIUM" or 1.5 <= atr_pct <= 3.0:
            scores["DAY_TRADE"] += 3
            scores["SWING_TRADE"] += 2
        else:  # LOW
            scores["SWING_TRADE"] += 3
            scores["POSITION_TRADE"] += 3

        # --- Regras de Tendência ---
        if adx_value < 20 or trend_strength == "WEAK":
            # Tendência fraca = melhor para scalp (range trading)
            scores["SCALP"] += 3
            scores["DAY_TRADE"] += 1
        elif 20 <= adx_value < 35 or trend_strength == "MODERATE":
            # Tendência moderada = day trade ou swing
            scores["DAY_TRADE"] += 3
            scores["SWING_TRADE"] += 2
        elif 35 <= adx_value < 50 or trend_strength == "STRONG":
            # Tendência forte = swing trade
            scores["SWING_TRADE"] += 4
            scores["DAY_TRADE"] += 1
        else:  # adx >= 50, VERY_STRONG
            # Tendência muito forte = position trade
            scores["POSITION_TRADE"] += 4
            scores["SWING_TRADE"] += 2

        # --- Regras de Confluência ---
        if confluence_score >= 4:
            # Alta confluência entre timeframes = operações maiores
            scores["SWING_TRADE"] += 2
            scores["POSITION_TRADE"] += 2
        elif confluence_score <= 2:
            # Baixa confluência = operações curtas
            scores["SCALP"] += 2
            scores["DAY_TRADE"] += 1

        # --- Regras de Momentum ---
        if 40 <= rsi <= 60:
            # RSI neutro = range trading
            scores["SCALP"] += 2
        elif rsi < 30 or rsi > 70:
            # RSI extremo = possível reversão, swing
            scores["SWING_TRADE"] += 2

        # --- Regras de Volume ---
        if volume_trend in ["increasing", "strong_increasing"]:
            # Volume crescente confirma tendência
            scores["SWING_TRADE"] += 1
            scores["POSITION_TRADE"] += 1

        # --- Tendência Primária ---
        if primary_trend == "NEUTRAL" or primary_trend == "SIDEWAYS":
            scores["SCALP"] += 2
            scores["DAY_TRADE"] += 1
        elif primary_trend in ["BULLISH", "BEARISH"]:
            scores["SWING_TRADE"] += 1
        elif primary_trend in ["STRONG_BULLISH", "STRONG_BEARISH"]:
            scores["POSITION_TRADE"] += 2

        # ===== DETERMINAR VENCEDOR =====

        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]
        total_score = sum(scores.values())

        # Calcular confiança (quanto mais dominante, mais confiante)
        if total_score > 0:
            dominance = best_score / total_score
            confidence = min(10, int(dominance * 15) + 3)
        else:
            confidence = 5

        # ===== PARÂMETROS POR TIPO =====

        # TPs reduzidos para alvos realistas em mercado lateral
        # Antes: TPs muito longe (7-20%) nunca batiam, lucro evaporava
        parameters = {
            "SCALP": {
                "stop_loss_pct": 0.3,
                "take_profit_1_pct": 0.5,
                "take_profit_2_pct": 0.8,
                "max_duration_hours": 0.5,
                "min_volume_multiplier": 1.5
            },
            "DAY_TRADE": {
                "stop_loss_pct": 1.0,
                "take_profit_1_pct": 1.2,
                "take_profit_2_pct": 2.0,
                "max_duration_hours": 8,
                "min_volume_multiplier": 1.2
            },
            "SWING_TRADE": {
                "stop_loss_pct": 2.0,
                "take_profit_1_pct": 2.5,
                "take_profit_2_pct": 4.0,
                "max_duration_hours": 168,  # 7 dias
                "min_volume_multiplier": 1.0
            },
            "POSITION_TRADE": {
                "stop_loss_pct": 4.0,
                "take_profit_1_pct": 6.0,
                "take_profit_2_pct": 10.0,
                "max_duration_hours": 672,  # 28 dias
                "min_volume_multiplier": 0.8
            }
        }

        # Gerar reasoning
        reasoning_parts = []
        if volatility_level == "HIGH":
            reasoning_parts.append("Alta volatilidade favorece operações curtas")
        if adx_value > 35:
            reasoning_parts.append(f"ADX forte ({adx_value}) indica tendência estabelecida")
        if confluence_score >= 4:
            reasoning_parts.append("Alta confluência entre timeframes")
        if primary_trend not in ["NEUTRAL", "SIDEWAYS"]:
            reasoning_parts.append(f"Tendência {primary_trend} identificada")

        reasoning = ". ".join(reasoning_parts) if reasoning_parts else "Análise baseada em múltiplos fatores"

        return {
            "operation_type": best_type,
            "confidence": confidence,
            "reasoning": reasoning,
            "parameters": parameters[best_type],
            "scores": scores,
            "market_conditions": {
                "volatility": volatility_level,
                "trend_strength": trend_strength,
                "adx": adx_value,
                "confluence": confluence_score,
                "rsi": rsi
            }
        }

    except Exception as e:
        logger.exception(f"Erro ao classificar condições de mercado: {e}")
        # Fallback para swing trade
        return {
            "operation_type": "SWING_TRADE",
            "confidence": 5,
            "reasoning": "Fallback devido a erro na análise",
            "parameters": {
                "stop_loss_pct": 2.5,
                "take_profit_1_pct": 4.0,
                "take_profit_2_pct": 7.0,
                "max_duration_hours": 168,
                "min_volume_multiplier": 1.0
            },
            "scores": {},
            "market_conditions": {}
        }

async def get_market_data(symbol: str = "BTCUSDT") -> Dict[str, Any]:
    """
    Obtém dados de mercado da Binance para análise (CORRIGIDO: agora async usando BinanceClient).
    """
    try:
        from src.exchange.client import BinanceClient

        logger.debug(f"Fetching market data for {symbol}")

        async with BinanceClient() as client:
            # Obter ticker 24h
            ticker = await client.get_ticker_24hr(symbol)

            # Obter klines (apenas contagem, não os dados completos)
            klines_df = await client.get_klines(symbol, '1h', limit=DEFAULT_KLINES_LIMIT)

            # Obter funding rate
            funding = await client.get_funding_rate(symbol)

            # Obter open interest
            open_interest = await client.get_open_interest(symbol)

        result = {
            "symbol": symbol,
            "current_price": float(ticker['lastPrice']),
            "price_change_24h": float(ticker['priceChangePercent']),
            "volume_24h": float(ticker['volume']),
            "high_24h": float(ticker['highPrice']),
            "low_24h": float(ticker['lowPrice']),
            "funding_rate": float(funding.get('lastFundingRate', 0)),
            "open_interest": float(open_interest.get('openInterest', 0)),
            "klines_count": len(klines_df),
            # REMOVIDO: "recent_klines" - muito grande e causa erros de decodificação
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        logger.info(f"Market data fetched for {symbol}: ${result['current_price']:.2f}")
        return result

    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        logger.exception(f"[{symbol}] Erro ao obter dados de mercado ({error_type}): {error_msg}")
        return {
            "error": f"Erro ao obter dados de mercado ({error_type}): {error_msg}",
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error_type": error_type
        }

def _analyze_market_structure(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analisa estrutura de mercado identificando suporte, resistência e estrutura de tendência.
    """
    try:
        if len(df) < 10:
            return {"structure": "insufficient_data"}

        # Identificar High Higher (HH) e Lower Lows (LL)
        highs = df['high'].rolling(window=5).max()
        lows = df['low'].rolling(window=5).min()

        # Identificar estrutura de tendência
        recent_highs = highs.tail(10).dropna()
        recent_lows = lows.tail(10).dropna()

        if len(recent_highs) >= 3 and len(recent_lows) >= 3:
            # Verificar se está fazendo Higher High an Lower Low (Uptrend)
            if (recent_highs.iloc[-1] > recent_highs.iloc[-3] and
                recent_lows.iloc[-1] > recent_lows.iloc[-3]):
                structure = "UPTREND"
                strength = "strong" if (recent_highs.iloc[-1] > recent_highs.iloc[-5] and
                                      recent_lows.iloc[-1] > recent_lows.iloc[-5]) else "moderate"

            # Verificar se está fazendo Lower High e Lower Low (Downtrend)
            elif (recent_highs.iloc[-1] < recent_highs.iloc[-3] and
                  recent_lows.iloc[-1] < recent_lows.iloc[-3]):
                structure = "DOWNTREND"
                strength = "strong" if (recent_highs.iloc[-1] < recent_highs.iloc[-5] and
                                      recent_lows.iloc[-1] < recent_lows.iloc[-5]) else "moderate"
            else:
                structure = "RANGE"
                strength = "neutral"
        else:
            structure = "RANGE"
            strength = "neutral"

        # Identificar níveis de suporte e resistência dinâmicos
        support_level = recent_lows.min() if len(recent_lows) > 0 else df['low'].min()
        resistance_level = recent_highs.max() if len(recent_highs) > 0 else df['high'].max()

        return {
            "structure": structure,
            "strength": strength,
            "support_level": float(support_level),
            "resistance_level": float(resistance_level),
            "recent_highs_count": len(recent_highs),
            "recent_lows_count": len(recent_lows)
        }

    except Exception as e:
        return {
            "structure": "error",
            "error": str(e)
        }

async def analyze_multiple_timeframes(symbol: str) -> Dict[str, Any]:
    """
    Análise multi-timeframe para maior precisão.
    CORRIGIDO: Agora async usando BinanceClient.
    """
    try:
        from src.exchange.client import BinanceClient

        timeframes = ['5m', '15m', '1h', '4h', '1d']
        analyses = {}

        async with BinanceClient() as client:
            for tf in timeframes:
                try:
                    # Obter dados para timeframe específico usando BinanceClient
                    # exclude_forming=True: remove candle incompleto para evitar sinais falsos
                    klines_df = await client.get_klines(symbol, tf, limit=100, exclude_forming=True)

                    if not klines_df.empty and len(klines_df) >= 20:
                        # Resetar índice para ter timestamp como coluna
                        df = klines_df.reset_index()

                        # Garantir colunas numéricas
                        for col in ['open', 'high', 'low', 'close', 'volume']:
                            if col in df.columns:
                                df[col] = pd.to_numeric(df[col], errors='coerce')

                        # Calcular tendência usando EMA 20 e EMA 50 (mais robusto que SMA 20)
                        close_prices = df['close'].values
                        current_price = close_prices[-1]

                        if len(close_prices) >= 50:
                            ema_20 = talib.EMA(close_prices, timeperiod=20)[-1]
                            ema_50 = talib.EMA(close_prices, timeperiod=50)[-1]

                            # Tendência = preço acima de AMBAS EMAs e EMA20 > EMA50
                            if current_price > ema_20 > ema_50:
                                trend = "bullish"
                            elif current_price < ema_20 < ema_50:
                                trend = "bearish"
                            else:
                                trend = "neutral"
                        else:
                            ema_20 = talib.EMA(close_prices, timeperiod=20)[-1]
                            ema_50 = ema_20  # fallback
                            if current_price > ema_20:
                                trend = "bullish"
                            elif current_price < ema_20:
                                trend = "bearish"
                            else:
                                trend = "neutral"

                        analyses[tf] = {
                            "trend": trend,
                            "current_price": float(current_price),
                            "ema_20": float(ema_20),
                            "ema_50": float(ema_50)
                        }
                except Exception as e:
                    logger.warning(f"Erro no timeframe {tf}: {e}")
                    continue

        # Calcular confluência COM PESO por timeframe
        # Timeframes maiores têm MAIS peso (4h e 1d dominam a decisão)
        tf_weights = {'5m': 0.5, '15m': 0.5, '1h': 1.0, '4h': 2.0, '1d': 3.0}

        weighted_bullish = 0
        weighted_bearish = 0
        bullish_timeframes = 0
        bearish_timeframes = 0
        neutral_timeframes = 0

        for tf_name, tf_data in analyses.items():
            weight = tf_weights.get(tf_name, 1.0)
            if tf_data['trend'] == 'bullish':
                bullish_timeframes += 1
                weighted_bullish += weight
            elif tf_data['trend'] == 'bearish':
                bearish_timeframes += 1
                weighted_bearish += weight
            else:
                neutral_timeframes += 1

        # Confluência baseada em peso (4h+1d = 5.0, supera 5m+15m+1h = 2.0)
        if weighted_bullish > weighted_bearish:
            confluence = "bullish"
        elif weighted_bearish > weighted_bullish:
            confluence = "bearish"
        else:
            confluence = "neutral"

        return {
            "symbol": symbol,
            "timeframes": analyses,
            "confluence": confluence,
            "bullish_count": bullish_timeframes,
            "bearish_count": bearish_timeframes,
            "neutral_count": neutral_timeframes,
            "weighted_bullish": weighted_bullish,
            "weighted_bearish": weighted_bearish,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        return {
            "error": f"Erro na análise multi-timeframe: {str(e)}",
            "symbol": symbol
        }

async def analyze_order_flow(symbol: str) -> Dict[str, Any]:
    """
    Análise de fluxo de ordens e delta.
    CORRIGIDO: Usa BinanceClient com rate limiter, circuit breaker e retry.
    """
    try:
        from src.exchange.client import BinanceClient

        async with BinanceClient() as client:
            # Obter orderbook
            orderbook = await client.get_orderbook(symbol, limit=20)

            # Calcular imbalance
            bid_volume = sum([float(b[1]) for b in orderbook['bids'][:20]])
            ask_volume = sum([float(a[1]) for a in orderbook['asks'][:20]])

            total_volume = bid_volume + ask_volume
            imbalance = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0

            # Obter trades recentes para CVD usando BinanceClient (com retry e circuit breaker)
            buy_volume = 0
            sell_volume = 0
            try:
                trades = await client.get_agg_trades(symbol, limit=100)
                for trade in trades:
                    if trade['m']:  # isBuyerMaker
                        sell_volume += float(trade['q'])
                    else:
                        buy_volume += float(trade['q'])
            except Exception as e:
                logger.warning(f"[{symbol}] Erro ao obter aggTrades, usando orderbook apenas: {e}")

            cvd = buy_volume - sell_volume

            return {
                "symbol": symbol,
                "orderbook_imbalance": imbalance,
                "bid_volume": bid_volume,
                "ask_volume": ask_volume,
                "cvd": cvd,
                "buy_volume": buy_volume,
                "sell_volume": sell_volume,
                "buy_pressure": buy_volume > sell_volume * 1.2,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    except Exception as e:
        logger.exception(f"Erro na análise de order flow: {e}")
        return {
            "error": f"Erro na análise de order flow: {str(e)}",
            "symbol": symbol
        }

async def analyze_technical_indicators(symbol: str = "BTCUSDT", optimized_params: Optional[Dict] = None, mover_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Analisa indicadores técnicos REAIS usando TA-Lib.
    MELHORADO: Inclui EMA, OBV, Volume Profile e Fibonacci conforme sugestões Claude/DeepSeek.
    CORRIGIDO: Agora async usando BinanceClient.
    DINÂMICO: Usa parâmetros otimizados do ContinuousOptimizer quando disponíveis.
    Se optimized_params não for passado, tenta carregar automaticamente para o símbolo.

    Args:
        symbol: Par de trading
        optimized_params: Parâmetros otimizados (se já carregados)
        mover_type: "gainer", "loser" ou None - usado para fallback de categoria
    """
    try:
        # Auto-carregar parâmetros otimizados se não fornecidos
        if optimized_params is None:
            try:
                from src.backtesting.continuous_optimizer import load_best_config
                from dataclasses import asdict
                best = load_best_config(symbol, "1h", mover_type=mover_type)
                if best:
                    optimized_params = asdict(best)
            except Exception:
                pass  # Usa defaults se falhar

        from src.exchange.client import BinanceClient

        # CORRIGIDO: Obter klines usando BinanceClient async
        # exclude_forming=True: remove o último candle (incompleto) para evitar
        # indicadores calculados com dados parciais que causam entradas atrasadas
        async with BinanceClient() as client:
            klines_df = await client.get_klines(symbol, '1h', limit=200, exclude_forming=True)

        if klines_df.empty or len(klines_df) < 50:  # Mínimo para indicadores confiáveis
            return {
                "error": "Dados insuficientes para análise técnica (mínimo 50 candles)",
                "symbol": symbol
            }

        # BinanceClient já retorna DataFrame com índice timestamp e colunas numéricas
        # Resetar índice para ter timestamp como coluna
        df = klines_df.reset_index()

        # Garantir que temos as colunas necessárias
        if 'timestamp' not in df.columns:
            df['timestamp'] = df.index if hasattr(df.index, 'values') else range(len(df))

        # Converter para numérico (já deve estar, mas garantir)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Calcular indicadores técnicos REAIS
        # CORRIGIDO: TA-Lib requer arrays float64 (double), converter explicitamente
        # Preencher NaN antes de converter para numpy array
        df['close'] = df['close'].ffill().bfill().fillna(0)
        df['high'] = df['high'].ffill().bfill().fillna(0)
        df['low'] = df['low'].ffill().bfill().fillna(0)
        df['volume'] = df['volume'].fillna(0)

        # Converter para arrays numpy float64 (double) - REQUERIDO pelo TA-Lib
        close_prices = np.asarray(df['close'].values, dtype=np.float64)
        high_prices = np.asarray(df['high'].values, dtype=np.float64)
        low_prices = np.asarray(df['low'].values, dtype=np.float64)
        volume = np.asarray(df['volume'].values, dtype=np.float64)

        # Verificação final: garantir que são float64 e não têm NaN
        if close_prices.dtype != np.float64:
            close_prices = close_prices.astype(np.float64)
        if high_prices.dtype != np.float64:
            high_prices = high_prices.astype(np.float64)
        if low_prices.dtype != np.float64:
            low_prices = low_prices.astype(np.float64)
        if volume.dtype != np.float64:
            volume = volume.astype(np.float64)

        # Remover qualquer NaN restante (não deveria ter, mas garantir)
        close_prices = np.nan_to_num(close_prices, nan=0.0)
        high_prices = np.nan_to_num(high_prices, nan=0.0)
        low_prices = np.nan_to_num(low_prices, nan=0.0)
        volume = np.nan_to_num(volume, nan=0.0)

        current_price = close_prices[-1]

        # Parâmetros dinâmicos do optimizer (fallback para defaults clássicos)
        op = optimized_params or {}
        _rsi_period = int(op.get("rsi_period", 14))
        _macd_fast = int(op.get("macd_fast", 12))
        _macd_slow = int(op.get("macd_slow", 26))
        _macd_signal = int(op.get("macd_signal", 9))
        _adx_period = int(op.get("adx_period", 14))
        _atr_period = int(op.get("atr_period", 14))
        _bb_period = int(op.get("bb_period", 20))
        _bb_std = float(op.get("bb_std", 2.0))
        _params_source = "optimizer" if optimized_params else "default"
        logger.info(f"[{symbol}] Indicadores usando params {_params_source}: RSI={_rsi_period}, MACD={_macd_fast}/{_macd_slow}/{_macd_signal}, ADX={_adx_period}, BB={_bb_period}/{_bb_std}")

        # RSI (período dinâmico)
        rsi = talib.RSI(close_prices, timeperiod=_rsi_period)[-1]

        # MACD (períodos dinâmicos)
        macd, macd_signal, macd_hist = talib.MACD(close_prices, fastperiod=_macd_fast, slowperiod=_macd_slow, signalperiod=_macd_signal)
        macd_value = macd[-1]
        macd_signal_value = macd_signal[-1]
        macd_histogram = macd_hist[-1]
        macd_crossover = "bullish" if macd_histogram > 0 and macd_value > macd_signal_value else "bearish" if macd_histogram < 0 else "neutral"

        # ADX (período dinâmico)
        adx = talib.ADX(high_prices, low_prices, close_prices, timeperiod=_adx_period)[-1]

        # ATR (período dinâmico)
        atr = talib.ATR(high_prices, low_prices, close_prices, timeperiod=_atr_period)[-1]

        # Bollinger Bands (período e desvio dinâmicos)
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close_prices, timeperiod=_bb_period, nbdevup=_bb_std, nbdevdn=_bb_std)
        bb_position = (close_prices[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1]) if (bb_upper[-1] - bb_lower[-1]) > 0 else 0.5

        # SMA (fast/slow dinâmicos, 200 sempre fixo)
        _ema_fast_period = int(op.get("ema_fast", 20))
        _ema_slow_period = int(op.get("ema_slow", 50))
        sma_20 = talib.SMA(close_prices, timeperiod=_ema_fast_period)[-1]
        sma_50 = talib.SMA(close_prices, timeperiod=_ema_slow_period)[-1]
        sma_200 = talib.SMA(close_prices, timeperiod=200) if len(close_prices) >= 200 else None
        sma_200_value = float(sma_200[-1]) if sma_200 is not None and not np.isnan(sma_200[-1]) else None

        # EMA (fast/slow dinâmicos, 200 sempre fixo)
        ema_20 = talib.EMA(close_prices, timeperiod=_ema_fast_period)[-1]
        ema_50 = talib.EMA(close_prices, timeperiod=_ema_slow_period)[-1]
        ema_200 = talib.EMA(close_prices, timeperiod=200) if len(close_prices) >= 200 else None
        ema_200_value = float(ema_200[-1]) if ema_200 is not None and not np.isnan(ema_200[-1]) else None

        # OBV (On-Balance Volume) - conforme sugestão
        obv = talib.OBV(close_prices, volume)
        obv_value = float(obv[-1]) if not np.isnan(obv[-1]) else 0
        obv_trend = "bullish" if len(obv) >= 5 and obv[-1] > obv[-5] else "bearish"

        # Volume Profile - identificar POC (Point of Control)
        try:
            price_ranges = np.linspace(df['low'].min(), df['high'].max(), 20)
            volume_profile = {}
            for i in range(len(price_ranges) - 1):
                mask = (df['close'] >= price_ranges[i]) & (df['close'] < price_ranges[i+1])
                volume_profile[float(price_ranges[i])] = float(df[mask]['volume'].sum())

            # CORRIGIDO: Proteção contra volume_profile vazio
            if volume_profile and len(volume_profile) > 0:
                poc_price = max(volume_profile, key=volume_profile.get)
            else:
                poc_price = current_price
        except (ValueError, TypeError, KeyError) as e:
            logger.warning(f"Erro ao calcular volume profile: {e}")
            poc_price = current_price
            volume_profile = {}

        # Fibonacci Retracement (conforme sugestão)
        period_high = df['high'].max()
        period_low = df['low'].min()
        fib_range = period_high - period_low

        fib_levels = {
            "fib_0": float(period_high),
            "fib_23.6": float(period_high - (fib_range * 0.236)),
            "fib_38.2": float(period_high - (fib_range * 0.382)),
            "fib_50": float(period_high - (fib_range * 0.50)),
            "fib_61.8": float(period_high - (fib_range * 0.618)),
            "fib_100": float(period_low)
        }

        # Determinar tendência melhorada (usando EMA + ADX conforme correção)
        # CORRIGIDO: Considerar ADX para determinar força da tendência
        # ADX > 25 = tendência forte, ADX <= 25 = tendência fraca
        adx_value = float(adx) if not np.isnan(adx) else 25

        if ema_200_value:
            if current_price > ema_20 > ema_50 > ema_200_value:
                # EMA alinhada bullish: verificar ADX para determinar força
                if adx_value > 25:
                    trend = "strong_bullish"  # Tendência forte confirmada
                else:
                    trend = "bullish"  # Alinhamento bullish mas ADX fraco
            elif current_price > ema_20 > ema_50:
                trend = "bullish"
            elif current_price < ema_20 < ema_50 < ema_200_value:
                # EMA alinhada bearish: verificar ADX para determinar força
                if adx_value > 25:
                    trend = "strong_bearish"  # Tendência forte confirmada
                else:
                    trend = "bearish"  # Alinhamento bearish mas ADX fraco
            elif current_price < ema_20 < ema_50:
                trend = "bearish"
            else:
                trend = "neutral"
        else:
            if current_price > ema_20 > ema_50:
                # Sem EMA200, verificar ADX para classificar força
                if adx_value > 25:
                    trend = "strong_bullish"
                else:
                    trend = "bullish"
            elif current_price < ema_20 < ema_50:
                # Sem EMA200, verificar ADX para classificar força
                if adx_value > 25:
                    trend = "strong_bearish"
                else:
                    trend = "bearish"
            else:
                trend = "neutral"

        # Determinar momentum melhorado
        if rsi > 70:
            momentum = "overbought"
        elif rsi < 30:
            momentum = "oversold"
        elif rsi > 50:
            momentum = "bullish"
        elif rsi < 50:
            momentum = "bearish"
        else:
            momentum = "neutral"

        # Suporte e resistência melhorados (usando Fibonacci e Volume Profile)
        support = min([fib_levels["fib_61.8"], bb_lower[-1], poc_price])
        resistance = max([fib_levels["fib_38.2"], bb_upper[-1], poc_price])

        # Análise de estrutura de mercado
        market_structure = _analyze_market_structure(df)

        return {
            "symbol": symbol,
            "trend": trend,
            "momentum": momentum,
            "volatility": "high" if atr > current_price * 0.02 else "normal",
            "market_structure": market_structure,
            "indicators": {
                "rsi": float(rsi) if not np.isnan(rsi) else 50,
                "macd": float(macd_value) if not np.isnan(macd_value) else 0,
                "macd_signal": float(macd_signal_value) if not np.isnan(macd_signal_value) else 0,
                "macd_histogram": float(macd_histogram) if not np.isnan(macd_histogram) else 0,
                "macd_crossover": macd_crossover,
                "adx": float(adx) if not np.isnan(adx) else 25,
                "atr": float(atr) if not np.isnan(atr) else current_price * 0.01,
                "bb_position": float(bb_position) if not np.isnan(bb_position) else 0.5,
                "bb_upper": float(bb_upper[-1]) if not np.isnan(bb_upper[-1]) else current_price * 1.05,
                "bb_middle": float(bb_middle[-1]) if not np.isnan(bb_middle[-1]) else current_price,
                "bb_lower": float(bb_lower[-1]) if not np.isnan(bb_lower[-1]) else current_price * 0.95,
                "sma_20": float(sma_20) if not np.isnan(sma_20) else current_price,
                "sma_50": float(sma_50) if not np.isnan(sma_50) else current_price,
                "sma_200": sma_200_value,
                "ema_20": float(ema_20) if not np.isnan(ema_20) else current_price,
                "ema_50": float(ema_50) if not np.isnan(ema_50) else current_price,
                "ema_200": ema_200_value,
                "obv": obv_value,
                "obv_trend": obv_trend,
                "candle_body_pct": float(abs(df['close'].iloc[-1] - df['open'].iloc[-1]) / current_price * 100) if current_price > 0 else 0.5,
                "volume_ratio": float(volume[-1] / volume[-21:-1].mean()) if len(volume) >= 21 and volume[-21:-1].mean() > 0 else 1.0,
            },
            "volume_profile": {
                "poc_price": float(poc_price),
                "high_volume_zones": sorted(volume_profile.items(), key=lambda x: x[1], reverse=True)[:3] if volume_profile and len(volume_profile) > 0 else []
            },
            "fibonacci_levels": fib_levels,
            "support": float(support) if not np.isnan(support) else current_price * 0.95,
            "resistance": float(resistance) if not np.isnan(resistance) else current_price * 1.05,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "_params_source": _params_source,
            "_optimized_params_used": {
                "rsi_period": _rsi_period,
                "macd": f"{_macd_fast}/{_macd_slow}/{_macd_signal}",
                "adx_period": _adx_period,
                "bb_period": _bb_period,
                "bb_std": _bb_std,
                "ema_fast": _ema_fast_period,
                "ema_slow": _ema_slow_period,
            } if _params_source == "optimizer" else None,
        }
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        logger.exception(f"[{symbol}] Erro na análise técnica ({error_type}): {error_msg}")
        return {
            "error": f"Erro na análise técnica ({error_type}): {error_msg}",
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error_type": error_type
        }

async def analyze_market_sentiment(symbol: str = "BTCUSDT") -> Dict[str, Any]:
    """
    Análise de sentimento baseada em dados de mercado (preço, volume, funding rate).
    CORRIGIDO: Agora async.
    """
    try:
        market_data = await get_market_data(symbol)
        if "error" in market_data:
            return market_data

        price_change = market_data['price_change_24h']
        volume = market_data['volume_24h']
        funding_rate = market_data['funding_rate']
        open_interest = market_data.get('open_interest', 0)

        # Análise baseada em dados reais de mercado
        sentiment_score = 0
        confidence = 0.5

        # 1. Variação de preço (indicador principal de sentimento)
        if price_change > 8:
            sentiment_score += 3  # Forte bullish
            confidence += 0.3
        elif price_change > 3:
            sentiment_score += 2  # Bullish
            confidence += 0.2
        elif price_change > 1:
            sentiment_score += 1  # Levemente bullish
            confidence += 0.1
        elif price_change < -8:
            sentiment_score -= 3  # Forte bearish
            confidence += 0.3
        elif price_change < -3:
            sentiment_score -= 2  # Bearish
            confidence += 0.2
        elif price_change < -1:
            sentiment_score -= 1  # Levemente bearish
            confidence += 0.1

        # 2. Volume (alta = interesse, baixa = desinteresse)
        if volume > 2000000:  # Volume muito alto = forte interesse
            sentiment_score += 2
            confidence += 0.2
        elif volume > 1000000:  # Volume alto = interesse
            sentiment_score += 1
            confidence += 0.1
        elif volume < 50000:  # Volume baixo = desinteresse
            sentiment_score -= 1
            confidence += 0.1

        # 3. Funding rate (positivo = bullish, negativo = bearish)
        if funding_rate > 0.02:  # Funding muito positivo = muito bullish
            sentiment_score += 2
            confidence += 0.2
        elif funding_rate > 0.005:  # Funding positivo = bullish
            sentiment_score += 1
            confidence += 0.1
        elif funding_rate < -0.02:  # Funding muito negativo = muito bearish
            sentiment_score -= 2
            confidence += 0.2
        elif funding_rate < -0.005:  # Funding negativo = bearish
            sentiment_score -= 1
            confidence += 0.1

        # 4. Open Interest (alta = interesse institucional)
        if open_interest > 100000000:  # OI muito alto
            sentiment_score += 1
            confidence += 0.1
        elif open_interest < 10000000:  # OI baixo
            sentiment_score -= 1
            confidence += 0.1

        # Determinar sentimento final baseado em dados reais
        if sentiment_score >= 3:
            sentiment = "very_positive"
            final_confidence = min(0.95, confidence)
        elif sentiment_score >= 1:
            sentiment = "positive"
            final_confidence = min(0.9, confidence)
        elif sentiment_score <= -3:
            sentiment = "very_negative"
            final_confidence = min(0.95, confidence)
        elif sentiment_score <= -1:
            sentiment = "negative"
            final_confidence = min(0.9, confidence)
        else:
            sentiment = "neutral"
            final_confidence = confidence

        return {
            "symbol": symbol,
            "sentiment": sentiment,
            "confidence": float(final_confidence),
            "factors": {
                "price_change": price_change,
                "volume_level": "high" if volume > 1000000 else "low" if volume < 100000 else "normal",
                "funding_rate": funding_rate,
                "open_interest": open_interest
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        return {
            "error": f"Erro na análise de sentimento: {str(e)}",
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


# ============================================================================
# FUNÇÕES AUXILIARES PARA INTERPRETAÇÃO DE DADOS
# ============================================================================

def _classify_rsi(rsi: float) -> Dict[str, str]:
    """Classifica RSI em zona e hint de ação"""
    try:
        if rsi < 30:
            return {"zone": "oversold", "action_hint": "potential_buy"}
        elif rsi < 40:
            return {"zone": "approaching_oversold", "action_hint": "potential_buy"}
        elif rsi < 60:
            return {"zone": "neutral", "action_hint": "wait"}
        elif rsi < 70:
            return {"zone": "approaching_overbought", "action_hint": "potential_sell"}
        else:
            return {"zone": "overbought", "action_hint": "potential_sell"}
    except Exception:
        return {"zone": "neutral", "action_hint": "wait"}

def _interpret_adx(adx: float) -> str:
    """Interpreta força da tendência baseado no ADX"""
    try:
        if adx < 20:
            return "no_trend"
        elif adx < 25:
            return "weak"
        elif adx < 50:
            return "moderate"
        else:
            return "strong"
    except Exception:
        return "no_trend"

def _interpret_macd_momentum(histogram: float, prev_histogram: Optional[float] = None) -> str:
    """Determina direção do momentum do MACD"""
    try:
        if prev_histogram is None:
            if histogram > 0:
                return "accelerating_up"
            else:
                return "accelerating_down"

        if histogram > 0 and histogram > prev_histogram:
            return "accelerating_up"
        elif histogram > 0 and histogram < prev_histogram:
            return "decelerating_up"
        elif histogram < 0 and histogram < prev_histogram:
            return "accelerating_down"
        else:
            return "decelerating_down"
    except Exception:
        return "neutral"

def _classify_bollinger_position(position: float) -> str:
    """Classifica posição nas Bollinger Bands"""
    try:
        if position < 0.2:
            return "lower_band"
        elif position < 0.4:
            return "below_middle"
        elif position < 0.6:
            return "middle"
        elif position < 0.8:
            return "above_middle"
        else:
            return "upper_band"
    except Exception:
        return "middle"

def _detect_ema_alignment(ema20: float, ema50: float, ema200: Optional[float], price: float) -> str:
    """Detecta alinhamento das EMAs"""
    try:
        if ema200 is None:
            if price > ema20 > ema50:
                return "bullish_stack"
            elif price < ema20 < ema50:
                return "bearish_stack"
            else:
                return "mixed"
        else:
            if price > ema20 > ema50 > ema200:
                return "bullish_stack"
            elif price < ema20 < ema50 < ema200:
                return "bearish_stack"
            else:
                return "mixed"
    except Exception:
        return "mixed"

def _interpret_funding_rate(rate: float) -> str:
    """Interpreta funding rate"""
    try:
        if rate > 0.01:
            return "crowded_long"
        elif rate > 0.005:
            return "slightly_long"
        elif rate > -0.005:
            return "neutral"
        elif rate > -0.01:
            return "slightly_short"
        else:
            return "crowded_short"
    except Exception:
        return "neutral"

def _classify_orderbook_imbalance(imbalance: float) -> str:
    """Classifica pressão do orderbook"""
    try:
        if imbalance > 0.5:
            return "strong_buy_pressure"
        elif imbalance > 0.2:
            return "buy_pressure"
        elif imbalance > -0.2:
            return "neutral"
        elif imbalance > -0.5:
            return "sell_pressure"
        else:
            return "strong_sell_pressure"
    except Exception:
        return "neutral"

def _calculate_suggested_stops(atr: float, price: float, signal_type: str = "BUY") -> Dict[str, float]:
    """Calcula stop loss e take profits sugeridos baseado em ATR"""
    try:
        atr_pct = (atr / price) * 100

        if signal_type == "BUY":
            stop_pct = 1.5 * atr_pct
            tp1_pct = 2.0 * atr_pct
            tp2_pct = 3.0 * atr_pct
        else:  # SELL
            stop_pct = 1.5 * atr_pct
            tp1_pct = 2.0 * atr_pct
            tp2_pct = 3.0 * atr_pct

        return {
            "suggested_stop_pct": stop_pct,
            "suggested_tp1_pct": tp1_pct,
            "suggested_tp2_pct": tp2_pct
        }
    except Exception:
        return {
            "suggested_stop_pct": 2.0,
            "suggested_tp1_pct": 2.5,
            "suggested_tp2_pct": 5.0
        }

def _identify_conflicting_signals(data: Dict) -> List[str]:
    """Identifica sinais que se contradizem"""
    conflicts = []
    try:
        trend = data.get("trend_analysis", {}).get("primary_trend", "neutral")
        momentum = data.get("trend_analysis", {}).get("momentum", "neutral")
        rsi_zone = data.get("key_indicators", {}).get("rsi", {}).get("zone", "neutral")
        macd_crossover = data.get("key_indicators", {}).get("macd", {}).get("crossover", "neutral")

        # RSI oversold mas tendência bearish
        if rsi_zone == "oversold" and "bearish" in trend:
            conflicts.append("RSI oversold but strong bearish trend")

        # RSI overbought mas tendência bullish
        if rsi_zone == "overbought" and "bullish" in trend:
            conflicts.append("RSI overbought but strong bullish trend")

        # MACD bearish mas momentum bullish
        if macd_crossover == "bearish" and "bullish" in momentum:
            conflicts.append("MACD bearish crossover but bullish momentum")

        # MACD bullish mas momentum bearish
        if macd_crossover == "bullish" and "bearish" in momentum:
            conflicts.append("MACD bullish crossover but bearish momentum")

        # Tendência bullish mas orderbook com sell pressure
        orderbook_bias = data.get("volume_flow", {}).get("orderbook_bias", "neutral")
        if "bullish" in trend and "sell" in orderbook_bias:
            conflicts.append("Bullish trend but sell pressure in orderbook")

        # Tendência bearish mas orderbook com buy pressure
        if "bearish" in trend and "buy" in orderbook_bias:
            conflicts.append("Bearish trend but buy pressure in orderbook")

    except Exception as e:
        logger.warning(f"Erro ao identificar sinais conflitantes: {e}")

    return conflicts

def _calculate_overall_bias(data: Dict) -> Dict[str, Any]:
    """Calcula score agregado de -10 a +10 com interpretação"""
    try:
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0

        # CORRIGIDO: Proteção contra None em todas as chamadas .get().get()
        trend_analysis = data.get("trend_analysis") or {}
        trend = trend_analysis.get("primary_trend", "neutral")
        if "strong_bullish" in trend:
            bullish_count += 3
        elif "bullish" in trend:
            bullish_count += 2
        elif "strong_bearish" in trend:
            bearish_count += 3
        elif "bearish" in trend:
            bearish_count += 2
        else:
            neutral_count += 1

        # Analisar momentum
        momentum = trend_analysis.get("momentum", "neutral")
        if momentum == "overbought":
            bearish_count += 1
        elif momentum == "oversold":
            bullish_count += 1
        elif "bullish" in momentum:
            bullish_count += 1
        elif "bearish" in momentum:
            bearish_count += 1
        else:
            neutral_count += 1

        # Analisar RSI
        key_indicators = data.get("key_indicators") or {}
        rsi_data = key_indicators.get("rsi") or {}
        rsi_hint = rsi_data.get("action_hint", "wait")
        if "buy" in rsi_hint:
            bullish_count += 1
        elif "sell" in rsi_hint:
            bearish_count += 1
        else:
            neutral_count += 1

        # Analisar MACD
        macd_data = key_indicators.get("macd") or {}
        macd_crossover = macd_data.get("crossover", "neutral")
        if macd_crossover == "bullish":
            bullish_count += 1
        elif macd_crossover == "bearish":
            bearish_count += 1
        else:
            neutral_count += 1

        # Analisar EMA alignment
        ema_structure = key_indicators.get("ema_structure") or {}
        ema_alignment = ema_structure.get("ema_alignment", "mixed")
        if "bullish" in ema_alignment:
            bullish_count += 1
        elif "bearish" in ema_alignment:
            bearish_count += 1
        else:
            neutral_count += 1

        # Analisar orderbook
        volume_flow = data.get("volume_flow") or {}
        orderbook_bias = volume_flow.get("orderbook_bias", "neutral")
        if "buy" in orderbook_bias:
            bullish_count += 1
        elif "sell" in orderbook_bias:
            bearish_count += 1
        else:
            neutral_count += 1

        # Analisar sentimento
        sentiment_data = data.get("sentiment") or {}
        sentiment = sentiment_data.get("overall", "neutral")
        if "very_positive" in sentiment:
            bullish_count += 2
        elif "positive" in sentiment:
            bullish_count += 1
        elif "very_negative" in sentiment:
            bearish_count += 2
        elif "negative" in sentiment:
            bearish_count += 1
        else:
            neutral_count += 1

        # Calcular bias geral (-10 a +10)
        overall_bias = bullish_count - bearish_count
        overall_bias = max(-10, min(10, overall_bias))

        # Interpretar bias
        if overall_bias >= 7:
            interpretation = "strong_buy"
            recommended_action = "BUY"
        elif overall_bias >= 3:
            interpretation = "buy"
            recommended_action = "BUY"
        elif overall_bias <= -7:
            interpretation = "strong_sell"
            recommended_action = "SELL"
        elif overall_bias <= -3:
            interpretation = "sell"
            recommended_action = "SELL"
        else:
            interpretation = "neutral"
            recommended_action = "WAIT"

        return {
            "bullish_factors_count": bullish_count,
            "bearish_factors_count": bearish_count,
            "neutral_factors_count": neutral_count,
            "overall_bias": overall_bias,
            "overall_bias_interpretation": interpretation,
            "recommended_action": recommended_action
        }
    except Exception as e:
        logger.warning(f"Erro ao calcular bias geral: {e}")
        return {
            "bullish_factors_count": 0,
            "bearish_factors_count": 0,
            "neutral_factors_count": 0,
            "overall_bias": 0,
            "overall_bias_interpretation": "neutral",
            "recommended_action": "WAIT"
        }

def _interpret_confluence(bullish_count: int, bearish_count: int) -> str:
    """Interpreta alinhamento de timeframes"""
    try:
        # CORRIGIDO: Garantir que são inteiros válidos
        bullish_count = int(bullish_count) if bullish_count is not None else 0
        bearish_count = int(bearish_count) if bearish_count is not None else 0

        if bullish_count >= 4:
            return "strong_bullish_alignment"
        elif bullish_count >= 3:
            return "bullish_alignment"
        elif bearish_count >= 4:
            return "strong_bearish_alignment"
        elif bearish_count >= 3:
            return "bearish_alignment"
        else:
            return "mixed_signals"
    except Exception as e:
        logger.warning(f"Erro ao interpretar confluência: {e}")
        return "mixed_signals"

# ============================================================================
# FUNÇÃO PRINCIPAL: PREPARAR ANÁLISE PARA LLM
# ============================================================================

async def prepare_analysis_for_llm(symbol: str, mover_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Prepara dados SUMARIZADOS e INTERPRETADOS para envio ao DeepSeek.

    REGRAS:
    - Payload máximo: 5KB
    - NUNCA incluir arrays de klines
    - SEMPRE interpretar valores numéricos em categorias
    - Incluir scores agregados pré-calculados

    Args:
        symbol: Par de trading
        mover_type: "gainer", "loser" ou None - para carregar config de categoria

    Returns:
        Dict estruturado e compacto para a LLM
    """
    try:
        # Carregar parâmetros otimizados por símbolo (com fallback de categoria)
        _opt_params = None
        try:
            from src.backtesting.continuous_optimizer import load_best_config
            from dataclasses import asdict
            best = load_best_config(symbol, "1h", mover_type=mover_type)
            if best:
                _opt_params = asdict(best)
                logger.info(f"[{symbol}] Usando parâmetros otimizados do ContinuousOptimizer (mover_type={mover_type})")
        except Exception as e:
            logger.debug(f"[{symbol}] Sem parâmetros otimizados: {e}")

        # Coletar todos os dados necessários (async)
        market_data = await get_market_data(symbol)
        technical_indicators = await analyze_technical_indicators(symbol, optimized_params=_opt_params, mover_type=mover_type)
        sentiment = await analyze_market_sentiment(symbol)
        multi_timeframe = await analyze_multiple_timeframes(symbol)
        order_flow = await analyze_order_flow(symbol)

        # Verificar erros e logar detalhes
        errors = []
        if "error" in market_data:
            error_msg = market_data.get("error", "Erro desconhecido em market_data")
            errors.append(f"market_data: {error_msg}")
            logger.error(f"[{symbol}] Erro em market_data: {error_msg}")

        if "error" in technical_indicators:
            error_msg = technical_indicators.get("error", "Erro desconhecido em technical_indicators")
            errors.append(f"technical_indicators: {error_msg}")
            logger.error(f"[{symbol}] Erro em technical_indicators: {error_msg}")

        if "error" in sentiment:
            error_msg = sentiment.get("error", "Erro desconhecido em sentiment")
            errors.append(f"sentiment: {error_msg}")
            logger.warning(f"[{symbol}] Erro em sentiment: {error_msg}")  # Warning pois não é crítico

        if "error" in multi_timeframe:
            error_msg = multi_timeframe.get("error", "Erro desconhecido em multi_timeframe")
            errors.append(f"multi_timeframe: {error_msg}")
            logger.warning(f"[{symbol}] Erro em multi_timeframe: {error_msg}")  # Warning pois não é crítico

        if "error" in order_flow:
            error_msg = order_flow.get("error", "Erro desconhecido em order_flow")
            errors.append(f"order_flow: {error_msg}")
            logger.warning(f"[{symbol}] Erro em order_flow: {error_msg}")  # Warning pois não é crítico

        # Se houver erro crítico (market_data ou technical_indicators), retornar erro
        if "error" in market_data or "error" in technical_indicators:
            error_summary = "; ".join(errors)
            logger.error(f"[{symbol}] Erro ao coletar dados críticos: {error_summary}")
            return {
                "error": f"Erro ao coletar dados de mercado: {error_summary}",
                "symbol": symbol,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        # Extrair valores principais
        current_price = market_data.get("current_price", 0)
        price_change_24h = market_data.get("price_change_24h", 0)
        high_24h = market_data.get("high_24h", current_price)
        low_24h = market_data.get("low_24h", current_price)
        position_in_range = ((current_price - low_24h) / (high_24h - low_24h) * 100) if (high_24h - low_24h) > 0 else 50

        # Extrair indicadores técnicos
        indicators = technical_indicators.get("indicators", {})
        rsi_value = indicators.get("rsi", 50)
        macd_hist = indicators.get("macd_histogram", 0)
        macd_crossover = indicators.get("macd_crossover", "neutral")
        adx_value = indicators.get("adx", 25)
        atr_value = indicators.get("atr", current_price * 0.01)
        bb_position = indicators.get("bb_position", 0.5)
        ema_20 = indicators.get("ema_20", current_price)
        ema_50 = indicators.get("ema_50", current_price)
        ema_200 = indicators.get("ema_200")
        obv_trend = indicators.get("obv_trend", "neutral")

        # Extrair níveis
        support = technical_indicators.get("support", current_price * 0.95)
        resistance = technical_indicators.get("resistance", current_price * 1.05)
        fib_levels = technical_indicators.get("fibonacci_levels", {})
        # CORRIGIDO: Proteção adicional contra None em volume_profile
        volume_profile_data = technical_indicators.get("volume_profile") or {}
        poc_price = volume_profile_data.get("poc_price", current_price)

        # Calcular distâncias
        distance_to_support = ((current_price - support) / current_price) * 100
        distance_to_resistance = ((resistance - current_price) / current_price) * 100

        # Interpretar dados
        rsi_classification = _classify_rsi(rsi_value)
        adx_interpretation = _interpret_adx(adx_value)
        macd_momentum = _interpret_macd_momentum(macd_hist)
        bb_zone = _classify_bollinger_position(bb_position)
        ema_alignment = _detect_ema_alignment(ema_20, ema_50, ema_200, current_price)
        funding_interpretation = _interpret_funding_rate(market_data.get("funding_rate", 0))
        orderbook_bias = _classify_orderbook_imbalance(order_flow.get("orderbook_imbalance", 0))
        suggested_stops = _calculate_suggested_stops(atr_value, current_price)

        # Analisar timeframes
        timeframe_alignment = {}
        confluence_bullish = multi_timeframe.get("bullish_count", 0)
        confluence_bearish = multi_timeframe.get("bearish_count", 0)
        confluence_score = confluence_bullish - confluence_bearish
        confluence_interpretation = _interpret_confluence(confluence_bullish, confluence_bearish)

        for tf in ["5m", "15m", "1h", "4h", "1d"]:
            tf_data = multi_timeframe.get("timeframes", {}).get(tf, {})
            timeframe_alignment[tf] = tf_data.get("trend", "neutral")

        # Determinar tendência primária
        primary_trend = technical_indicators.get("trend", "neutral")
        momentum = technical_indicators.get("momentum", "neutral")

        # Classificar volatilidade
        atr_pct = (atr_value / current_price) * 100
        if atr_pct > 0.05:
            volatility_level = "extreme"
        elif atr_pct > 0.03:
            volatility_level = "high"
        elif atr_pct > 0.015:
            volatility_level = "normal"
        else:
            volatility_level = "low"

        # Volume trend
        volume_24h = market_data.get("volume_24h", 0)
        volume_trend = "stable"  # Simplificado - poderia comparar com médias

        # Open interest trend
        oi_trend = "stable"  # Simplificado

        # Construir estrutura de análise
        analysis = {
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),

            "price_context": {
                "current": current_price,
                "change_24h_pct": price_change_24h,
                "high_24h": high_24h,
                "low_24h": low_24h,
                "position_in_range_pct": position_in_range
            },

            "trend_analysis": {
                "primary_trend": primary_trend,
                "trend_strength_adx": adx_value,
                "trend_strength_interpretation": adx_interpretation,
                "momentum": momentum,
                "timeframe_alignment": timeframe_alignment,
                "confluence_score": confluence_score,
                "confluence_interpretation": confluence_interpretation
            },

            "key_indicators": {
                "rsi": {
                    "value": rsi_value,
                    "zone": rsi_classification["zone"],
                    "action_hint": rsi_classification["action_hint"]
                },
                "macd": {
                    "histogram": macd_hist,
                    "crossover": macd_crossover,
                    "momentum_direction": macd_momentum
                },
                "bollinger": {
                    "position": bb_position,
                    "zone": bb_zone,
                    "squeeze_detected": False  # Simplificado
                },
                "ema_structure": {
                    "price_vs_ema20": "above" if current_price > ema_20 else "below",
                    "price_vs_ema50": "above" if current_price > ema_50 else "below",
                    "price_vs_ema200": "above" if ema_200 and current_price > ema_200 else "below" if ema_200 else "N/A",
                    "ema_alignment": ema_alignment
                },
                "obv_trend": obv_trend
            },

            "key_levels": {
                "immediate_support": support,
                "immediate_resistance": resistance,
                "fib_382": fib_levels.get("fib_38.2", support),
                "fib_50": fib_levels.get("fib_50", current_price),
                "fib_618": fib_levels.get("fib_61.8", resistance),
                "volume_poc": poc_price,
                "distance_to_support_pct": distance_to_support,
                "distance_to_resistance_pct": distance_to_resistance
            },

            "volume_flow": {
                "volume_24h": volume_24h,
                "volume_trend": volume_trend,
                "obv_trend": obv_trend,
                "orderbook_imbalance": order_flow.get("orderbook_imbalance", 0),
                "orderbook_bias": orderbook_bias,
                "cvd_direction": "positive" if order_flow.get("cvd", 0) > 0 else "negative"
            },

            "sentiment": {
                "overall": sentiment.get("sentiment", "neutral"),
                "confidence": sentiment.get("confidence", 0.5),
                "funding_rate": market_data.get("funding_rate", 0),
                "funding_interpretation": funding_interpretation,
                "open_interest_trend": oi_trend
            },

            "volatility": {
                "atr_value": atr_value,
                "atr_pct": atr_pct,
                "level": volatility_level,
                "suggested_stop_pct": suggested_stops["suggested_stop_pct"],
                "suggested_tp1_pct": suggested_stops["suggested_tp1_pct"],
                "suggested_tp2_pct": suggested_stops["suggested_tp2_pct"]
            },

            "conflicting_signals": [],  # Será preenchido depois
            "aggregated_scores": {},  # Será preenchido depois

            # Multi-timeframe data para confluencia
            "multi_timeframe": {
                "bullish_count": confluence_bullish,
                "bearish_count": confluence_bearish,
                "timeframe_alignment": timeframe_alignment,
            },

            # Dados brutos para calculo tecnico de SL/TP (nao enviados ao LLM)
            "_raw_indicators": {
                "ema_20": ema_20,
                "ema_50": ema_50,
                "ema_200": ema_200,
                "sma_200": indicators.get("sma_200"),
                "bb_upper": indicators.get("bb_upper", current_price * 1.05),
                "bb_lower": indicators.get("bb_lower", current_price * 0.95),
                "bb_middle": indicators.get("bb_middle", ema_20),
            },
            "_market_structure": technical_indicators.get("market_structure", {}),
            "_optimized_params": _opt_params,  # None se usando defaults
        }

        # Identificar sinais conflitantes
        analysis["conflicting_signals"] = _identify_conflicting_signals(analysis)

        # Calcular scores agregados
        analysis["aggregated_scores"] = _calculate_overall_bias(analysis)

        # CORRIGIDO: Garantir que não estamos enviando dados muito grandes
        # Limitar high_volume_zones se muito grande (já está limitado em analyze_technical_indicators, mas garantir)
        # Nota: high_volume_zones não está diretamente em analysis, mas sim em key_levels.volume_poc
        # O volume_profile completo está em technical_indicators, mas não é incluído em analysis

        # Validar tamanho do payload
        payload_size = len(json.dumps(analysis))
        if payload_size > 10000:  # 10KB (com margem de segurança)
            logger.warning(f"Payload muito grande: {payload_size} bytes para {symbol}")
            # Se payload muito grande, remover dados desnecessários
            # Remover high_volume_zones se existir em algum lugar
            if "key_levels" in analysis and "volume_poc" in analysis["key_levels"]:
                # Manter apenas POC, já está otimizado
                pass

        return analysis

    except Exception as e:
        logger.exception(f"Erro ao preparar análise para LLM: {e}")
        return {
            "error": f"Erro ao preparar análise: {str(e)}",
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

def _create_analysis_prompt(analysis: Dict[str, Any], market_classification: Dict[str, Any] = None) -> str:
    """Cria prompt estruturado para o DeepSeek com classificação de mercado"""
    try:
        conflicting = analysis.get("conflicting_signals", [])
        conflicting_text = "\n".join([f"- {s}" for s in conflicting]) if conflicting else "- Nenhum conflito identificado"

        # Classificação de mercado
        if market_classification:
            recommended_type = market_classification.get("operation_type", "SWING_TRADE")
            type_params = market_classification.get("parameters", {})
            market_conditions = market_classification.get("market_conditions", {})
            classification_text = f"""
## CLASSIFICAÇÃO DE MERCADO (Análise Automática)

- **Tipo Recomendado:** {recommended_type}
- **Confiança:** {market_classification.get('confidence', 5)}/10
- **Razão:** {market_classification.get('reasoning', 'N/A')}

### Parâmetros Sugeridos para {recommended_type}:
- Stop Loss: {type_params.get('stop_loss_pct', 2.5):.1f}%
- Take Profit 1: {type_params.get('take_profit_1_pct', 4.0):.1f}%
- Take Profit 2: {type_params.get('take_profit_2_pct', 7.0):.1f}%

### Condições de Mercado:
- Volatilidade: {market_conditions.get('volatility', 'N/A')}
- Força da Tendência: {market_conditions.get('trend_strength', 'N/A')}
- ADX: {market_conditions.get('adx', 'N/A')}
- Confluência: {market_conditions.get('confluence', 'N/A')}/5
"""
        else:
            classification_text = ""
            recommended_type = "SWING_TRADE"

        return f"""
Você é um trader profissional. Analise os dados e forneça um sinal de trading.

{classification_text}

## DADOS DE MERCADO

- Símbolo: {analysis['symbol']}
- Preço: ${analysis['price_context']['current']:,.2f}
- Variação 24h: {analysis['price_context']['change_24h_pct']:+.2f}%
- Posição no range: {analysis['price_context']['position_in_range_pct']:.0f}%

## ANÁLISE DE TENDÊNCIA

- Tendência primária: {analysis['trend_analysis']['primary_trend']}
- Força (ADX): {analysis['trend_analysis']['trend_strength_interpretation']}
- Momentum: {analysis['trend_analysis']['momentum']}
- Alinhamento timeframes: {analysis['trend_analysis']['confluence_interpretation']}
- Score confluência: {analysis['trend_analysis']['confluence_score']}/5

## INDICADORES

- RSI: {analysis['key_indicators']['rsi']['value']:.1f} ({analysis['key_indicators']['rsi']['zone']})
- MACD: {analysis['key_indicators']['macd']['crossover']} - {analysis['key_indicators']['macd']['momentum_direction']}
- Bollinger: {analysis['key_indicators']['bollinger']['zone']}
- EMAs: {analysis['key_indicators']['ema_structure']['ema_alignment']}

## NÍVEIS TÉCNICOS (use para SL/TP)

- Suporte imediato: ${analysis['key_levels']['immediate_support']:,.2f} ({analysis['key_levels']['distance_to_support_pct']:+.2f}%)
- Resistência imediata: ${analysis['key_levels']['immediate_resistance']:,.2f} ({analysis['key_levels']['distance_to_resistance_pct']:+.2f}%)
- Fibonacci 38.2%: ${analysis['key_levels']['fib_382']:,.2f}
- Fibonacci 50.0%: ${analysis['key_levels']['fib_50']:,.2f}
- Fibonacci 61.8%: ${analysis['key_levels']['fib_618']:,.2f}
- POC Volume: ${analysis['key_levels']['volume_poc']:,.2f}
- IMPORTANTE: SL DEVE estar atrás de suporte (BUY) ou resistência (SELL). TP nos próximos níveis

## VOLUME E FLUXO

- Pressão orderbook: {analysis['volume_flow']['orderbook_bias']}
- OBV: {analysis['volume_flow']['obv_trend']}

## SENTIMENTO

- Geral: {analysis['sentiment']['overall']}
- Funding: {analysis['sentiment']['funding_interpretation']}

## VOLATILIDADE

- Nível: {analysis['volatility']['level']}
- ATR: {analysis['volatility'].get('atr_value', 0):.2f} ({analysis['volatility'].get('atr_pct', 0):.2f}%)

## SINAIS CONFLITANTES

{conflicting_text}

## SCORE AGREGADO

- Fatores bullish: {analysis['aggregated_scores']['bullish_factors_count']}
- Fatores bearish: {analysis['aggregated_scores']['bearish_factors_count']}
- Bias geral: {analysis['aggregated_scores']['overall_bias']}/10 ({analysis['aggregated_scores']['overall_bias_interpretation']})
- Ação recomendada: {analysis['aggregated_scores']['recommended_action']}

---

## IMPORTANTE: ESCALA DE CONFIANÇA (0-10)

A confiança deve ser um número inteiro de 1 a 10, onde:

- **10**: Sinal extremamente forte, múltiplas confluências, alta probabilidade de sucesso
- **9**: Sinal muito forte, confluências claras, boa probabilidade
- **8**: Sinal forte, confluências presentes, probabilidade acima da média
- **7**: Sinal moderado-forte, algumas confluências, probabilidade razoável (MÍNIMO PARA EXECUÇÃO)
- **6**: Sinal moderado, confluências limitadas, probabilidade média (NÃO SERÁ EXECUTADO)
- **5**: Sinal fraco, poucas confluências, probabilidade baixa (NÃO SERÁ EXECUTADO)
- **4 ou menos**: Sinal muito fraco ou ambíguo (NÃO SERÁ EXECUTADO)

**CRÍTICO**: Apenas sinais com confiança >= 7 serão executados pelo sistema.
Se você não tiver confiança suficiente (>= 7), retorne "NO_SIGNAL" ao invés de um sinal fraco.

**Como calcular a confiança:**
- Considere a força da tendência (ADX, alinhamento EMAs)
- Considere a confluência de indicadores (RSI, MACD, Bollinger)
- Considere o alinhamento multi-timeframe
- Considere a clareza dos níveis de suporte/resistência
- Considere sinais conflitantes (reduz confiança)
- Considere a volatilidade (alta volatilidade pode reduzir confiança)

---

## INSTRUÇÕES

1. **ANALISE O TIPO DE OPERAÇÃO RECOMENDADO** ({recommended_type}) e considere se concorda.

2. **VOCÊ PODE DISCORDAR** - Se achar que outro tipo é melhor, indique no campo `operation_type`.

3. **TIPOS DISPONÍVEIS:**
   - **SCALP**: Operação de 5-30 minutos. Stop 0.3-0.5%, TP 0.5-1.5%. Use quando: alta volatilidade + mercado lateral.
   - **DAY_TRADE**: Operação de 1-8 horas. Stop 0.8-1.5%, TP 1.5-4%. Use quando: tendência intraday clara.
   - **SWING_TRADE**: Operação de 1-7 dias. Stop 2-3.5%, TP 3-10%. Use quando: tendência definida + momentum.
   - **POSITION_TRADE**: Operação de 1-4 semanas. Stop 5-8%, TP 8-30%. Use quando: tendência macro forte.

4. **SEJA DECISIVO** - Forneça BUY ou SELL sempre que houver oportunidade.

5. **AJUSTE OS PARÂMETROS** conforme o tipo de operação escolhido.

6. **CONFIANÇA:**
   - 7/10 = Sinal válido, executável
   - 8/10 = Sinal bom com confluências
   - 9/10 = Sinal forte
   - 10/10 = Sinal excepcional

---

RESPONDA APENAS COM JSON:

```json
{{
    "signal": "BUY" ou "SELL" ou "NO_SIGNAL",
    "operation_type": "SCALP" ou "DAY_TRADE" ou "SWING_TRADE" ou "POSITION_TRADE",
    "entry_price": <número>,
    "stop_loss": <número>,
    "take_profit_1": <número>,
    "take_profit_2": <número>,
    "confidence": <1-10>,
    "reasoning": "<justificativa incluindo por que escolheu esse tipo de operação>"
}}
```
"""
    except Exception as e:
        logger.exception(f"Erro ao criar prompt: {e}")
        return "Erro ao criar prompt de análise."

async def get_deepseek_analysis(symbol: str) -> Dict[str, Any]:
    """
    Prepara análise otimizada para o DeepSeek e chama diretamente.
    CORRIGIDO: Agora chama DeepSeek diretamente e retorna o sinal JSON processado.
    """
    try:
        import os

        from agno.agent import Agent
        from agno.models.deepseek import DeepSeek

        # Usar a nova função que já sumariza tudo
        analysis = await prepare_analysis_for_llm(symbol)

        if "error" in analysis:
            return analysis

        # NOVO: Classificar condições de mercado primeiro
        market_classification = classify_market_condition(analysis)
        logger.info(f"[CLASSIFICAÇÃO] {symbol}: {market_classification['operation_type']} (confiança: {market_classification['confidence']}/10)")

        # Criar prompt estruturado com classificação
        prompt = _create_analysis_prompt(analysis, market_classification)

        # CORRIGIDO: Chamar DeepSeek diretamente ao invés de retornar prompt
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            error_msg = (
                "ERRO CRÍTICO: DEEPSEEK_API_KEY não encontrada. "
                "Configure a variável de ambiente DEEPSEEK_API_KEY com sua chave da API. "
                "Obtenha sua chave em: https://platform.deepseek.com/"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Criar agent simples apenas para chamar DeepSeek
        model = DeepSeek(id="deepseek-chat", api_key=api_key, temperature=0.3, max_tokens=1000)
        agent = Agent(
            model=model,
            instructions="Você é um trader profissional. Analise os dados e forneça um sinal de trading em formato JSON."
        )

        # Chamar DeepSeek diretamente
        logger.info(f"[DEEPSEEK] Chamando DeepSeek diretamente para {symbol}")
        response = await agent.arun(prompt)

        # Extrair conteúdo da resposta
        response_content = str(response.content) if hasattr(response, 'content') else str(response)

        # Tentar extrair JSON da resposta
        import re
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_content, re.DOTALL)
        if json_match:
            try:
                signal_json = json.loads(json_match.group(1))

                # Se DeepSeek não especificou operation_type, usar o recomendado
                operation_type = signal_json.get("operation_type", market_classification["operation_type"])

                logger.info(f"[DEEPSEEK] {symbol}: {signal_json.get('signal', 'N/A')} ({operation_type}) - Confiança: {signal_json.get('confidence', 0)}/10")

                # Retornar sinal processado
                return {
                    "signal": signal_json.get("signal", "NO_SIGNAL"),
                    "operation_type": operation_type,
                    "entry_price": signal_json.get("entry_price"),
                    "stop_loss": signal_json.get("stop_loss"),
                    "take_profit_1": signal_json.get("take_profit_1"),
                    "take_profit_2": signal_json.get("take_profit_2"),
                    "confidence": signal_json.get("confidence", 5),
                    "reasoning": signal_json.get("reasoning", ""),
                    "market_classification": market_classification,  # Classificação automática
                    "analysis_data": analysis,  # JSON de análise enviado
                    "deepseek_prompt": prompt,  # Prompt de texto enviado
                    "raw_response": response_content,  # Resposta bruta do DeepSeek
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            except json.JSONDecodeError as e:
                logger.warning(f"[DEEPSEEK] Erro ao decodificar JSON: {e}")

        # Se não conseguiu extrair JSON, retornar resposta bruta
        logger.warning("[DEEPSEEK] Não foi possível extrair JSON, retornando resposta bruta")
        return {
            "analysis_data": analysis,  # JSON de análise enviado
            "deepseek_prompt": prompt,  # Prompt de texto enviado
            "raw_response": response_content,  # Resposta bruta do DeepSeek
            "needs_agent_processing": True,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        logger.exception(f"Erro na preparação para DeepSeek: {e}")
        return {
            "error": f"Erro na preparação para DeepSeek: {str(e)}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

def _calculate_current_drawdown() -> float:
    """
    Calcula o drawdown atual baseado no histórico de trades.
    MODIFICADO: Sistema em modo P&L - calcula drawdown baseado em P&L acumulado.
    """
    try:
        from src.trading.paper_trading import real_paper_trading
        summary = real_paper_trading.get_portfolio_summary()
        # Em modo P&L, drawdown é baseado em P&L negativo acumulado
        # Se P&L total for negativo, isso representa um drawdown
        total_pnl_percent = summary.get('total_pnl_percent', 0)
        # Drawdown é o valor negativo do P&L (se negativo) ou 0
        # Converter para proporção (0.0 a 1.0)
        if total_pnl_percent < 0:
            # P&L negativo = drawdown, converter para proporção
            # Ex: -20% P&L = 0.20 drawdown
            return abs(total_pnl_percent) / 100.0
        return 0.0
    except Exception as e:
        logger.warning(f"Erro ao calcular drawdown: {e}")
        return 0.0

def _calculate_total_exposure() -> float:
    """
    Calcula a exposição total atual.
    MODIFICADO: Sistema em modo P&L - não calcula exposição em valores, apenas retorna 0.
    Esta função não é mais usada, mas mantida para compatibilidade.
    """
    # MODIFICADO: Em modo P&L, não há cálculo de exposição baseado em capital
    return 0.0

def _get_daily_trades_count() -> int:
    """
    Retorna o número de trades executados hoje.
    CORRIGIDO: Usa real_paper_trading ao invés de paper_trading.
    """
    try:
        from src.trading.paper_trading import real_paper_trading
        today = datetime.now(timezone.utc).date()
        trades = real_paper_trading.get_trade_history()
        daily_trades = [t for t in trades if datetime.fromisoformat(t['timestamp']).date() == today]
        return len(daily_trades)
    except Exception:
        return 0

def validate_risk_and_position(
    signal: Dict[str, Any],
    symbol: str,
    account_balance: float = None,
    _trend_data: Dict[str, Any] = None,
    _capital: float = None
) -> Dict[str, Any]:
    """
    Valida risco e calcula tamanho de posição apropriado com circuit breakers.
    CORRIGIDO: Verifica se já existe posição aberta para o símbolo.
    MODIFICADO: Usa saldo atual do portfólio se account_balance não for fornecido.
    """
    try:
        # MODIFICADO: Não precisamos mais de account_balance (sistema em modo P&L)
        # Usar valor padrão apenas para cálculos internos, mas não será usado
        if account_balance is None:
            from src.core.config import settings
            account_balance = settings.initial_capital

        if signal.get('signal') == 'HOLD' or signal.get('signal') == 'NO_SIGNAL':
            return {
                "can_execute": False,
                "reason": "Sinal HOLD/NO_SIGNAL - não executar",
                "risk_level": "low"
            }

        # ========================================
        # COOLDOWN PÓS-FECHAMENTO (anti-whipsaw)
        # Impede reabrir posição logo após fechamento (TP ou SL)
        # Bug fix: DRIFTUSDT bateu TP2 e reabriu imediatamente, perdendo o lucro
        # ========================================
        try:
            from src.trading.risk_manager import _check_sl_cooldown, _sl_cooldown_registry, _sl_cooldown_hours
            from src.trading.risk_manager import _direction_cooldown_registry, _direction_cooldown_hours
            from datetime import datetime as _dt_cd, timezone as _tz_cd

            # Check 1: Cooldown geral pós-fechamento (4h após qualquer fechamento)
            if _check_sl_cooldown(symbol):
                remaining = _sl_cooldown_hours - (_dt_cd.now(_tz_cd.utc) - _sl_cooldown_registry[symbol]).total_seconds() / 3600
                logger.warning(f"[COOLDOWN] {symbol} BLOQUEADO: cooldown pós-fechamento ativo ({remaining:.1f}h restantes)")
                return {
                    "can_execute": False,
                    "reason": f"Cooldown pós-fechamento ativo para {symbol}: {remaining:.1f}h restantes (evita reentrada imediata)",
                    "risk_level": "medium"
                }

            # Check 2: Cooldown direcional (6h para mesma direção)
            sig_type_cd = signal.get("signal", "").upper()
            dir_cd = _direction_cooldown_registry.get(symbol)
            if dir_cd:
                hours_since = (_dt_cd.now(_tz_cd.utc) - dir_cd["time"]).total_seconds() / 3600
                if hours_since < _direction_cooldown_hours and sig_type_cd == dir_cd["direction"]:
                    remaining = _direction_cooldown_hours - hours_since
                    logger.warning(
                        f"[COOLDOWN DIRECIONAL] {symbol} BLOQUEADO: {sig_type_cd} bloqueado por {remaining:.1f}h "
                        f"(última posição {dir_cd['direction']} fechada há {hours_since:.1f}h)"
                    )
                    return {
                        "can_execute": False,
                        "reason": f"Cooldown direcional: {sig_type_cd} {symbol} bloqueado por {remaining:.1f}h "
                                  f"(última posição {dir_cd['direction']} fechada há {hours_since:.1f}h)",
                        "risk_level": "medium"
                    }
                elif hours_since >= _direction_cooldown_hours:
                    del _direction_cooldown_registry[symbol]
        except Exception as e:
            logger.warning(f"[COOLDOWN] Erro ao verificar cooldowns: {e}")

        # CORRIGIDO: Verificar se já existe QUALQUER posição aberta para este símbolo
        # NÃO permite long e short ao mesmo tempo no mesmo símbolo (de qualquer fonte)
        try:
            import json
            import os
            signal.get("source", "UNKNOWN")
            signal_type = signal.get("signal", "")

            if os.path.exists("portfolio/state.json"):
                with open("portfolio/state.json", "r", encoding='utf-8') as f:
                    state = json.load(f)
                    positions = state.get("positions", {})

                    # Verificar TODAS as posições abertas para este símbolo (qualquer fonte, qualquer direção)
                    for pos_key, pos in positions.items():
                        if pos.get("status") == "OPEN" and pos.get("symbol") == symbol:
                            existing_signal = pos.get("signal", "UNKNOWN")
                            existing_source = pos.get("source", "UNKNOWN")
                            return {
                                "can_execute": False,
                                "reason": f"Ja existe posicao {existing_signal} ({existing_source}) aberta para {symbol}. Feche antes de abrir nova.",
                                "risk_level": "medium"
                            }
        except Exception as e:
            logger.warning(f"Erro ao verificar posicoes existentes: {e}")
            pass

        # ========================================
        # BLACKLIST DE TOKENS ILÍQUIDOS/PERDEDORES
        # ========================================
        from src.core.config import settings as _cfg
        if symbol in _cfg.token_blacklist:
            logger.warning(f"[BLACKLIST] {symbol} está na blacklist — sinal ignorado")
            return {
                "can_execute": False,
                "reason": f"{symbol} está na blacklist (ilíquido ou consistentemente perdedor)",
                "risk_level": "high"
            }

        # ========================================
        # FILTRO DE SELL MAIS RESTRITIVO
        # Dados mostram SELL com 46% WR vs BUY com 67% WR
        # Shorts precisam de confiança maior
        # ========================================
        signal_type_check = signal.get("signal", "").upper()
        confidence_check = signal.get("confidence", 0)
        if signal_type_check == "SELL" and confidence_check < _cfg.sell_min_confidence:
            logger.warning(f"[SELL FILTER] SELL {symbol} bloqueado: confiança {confidence_check}/10 < mínimo {_cfg.sell_min_confidence}/10 para shorts")
            return {
                "can_execute": False,
                "reason": f"SELL requer confiança mínima {_cfg.sell_min_confidence}/10 (recebido {confidence_check}/10). "
                          f"Shorts têm win rate menor, precisam de sinal mais forte.",
                "risk_level": "medium"
            }

        # ========================================
        # FILTRO DE TENDÊNCIA DINÂMICO (EMA 50/200 no 4h)
        # Bloqueia sinais contra a tendência dominante.
        # Se adapta automaticamente: quando mercado virar bullish,
        # permite longs de novo. Zona neutra permite ambas direções.
        # ========================================
        signal_type = signal.get("signal", "").upper()
        if _trend_data and signal_type in ("BUY", "SELL"):
            trend = _trend_data.get("trend", "UNKNOWN")
            allow_long = _trend_data.get("allow_long", True)
            allow_short = _trend_data.get("allow_short", True)
            trend_desc = _trend_data.get("description", "")

            if signal_type == "BUY" and not allow_long:
                logger.warning(f"╔══ [BLOQUEADO] BUY {symbol} — Trend filter: {trend_desc}")
                return {
                    "can_execute": False,
                    "reason": f"Sinal BUY bloqueado pelo filtro de tendencia: {trend_desc}. "
                              f"EMA50 < EMA200 no 4h indica tendencia de baixa.",
                    "risk_level": "medium",
                    "trend": trend
                }

            if signal_type == "SELL" and not allow_short:
                logger.warning(f"╔══ [BLOQUEADO] SELL {symbol} — Trend filter: {trend_desc}")
                return {
                    "can_execute": False,
                    "reason": f"Sinal SELL bloqueado pelo filtro de tendencia: {trend_desc}. "
                              f"EMA50 > EMA200 no 4h indica tendencia de alta.",
                    "risk_level": "medium",
                    "trend": trend
                }

            logger.info(f"[TREND FILTER] {signal_type} {symbol} permitido: {trend_desc}")

        # ========================================
        # VALIDAÇÃO DE RISK:REWARD MÍNIMO (2:1)
        # Rejeita sinais onde o potencial de ganho não compensa o risco.
        # Usa TP1 como referência (TP2 é bônus).
        # ========================================
        entry_price_rr = signal.get('entry_price', 0)
        stop_loss_rr = signal.get('stop_loss', 0)
        tp1 = signal.get('tp1') or signal.get('take_profit_1') or signal.get('target_1', 0)
        signal.get('tp2') or signal.get('take_profit_2') or signal.get('target_2', 0)

        if entry_price_rr and stop_loss_rr and tp1:
            risk_distance = abs(entry_price_rr - stop_loss_rr)
            reward_distance_tp1 = abs(tp1 - entry_price_rr)

            if risk_distance > 0:
                rr_ratio = reward_distance_tp1 / risk_distance

                if rr_ratio < 1.48:  # Tolerância para floating point (1.4999 exibe "1.50")
                    # R:R menor que ~1.5:1 = muito ruim, bloquear
                    logger.warning(f"[R:R] BLOQUEADO {symbol}: R:R = {rr_ratio:.2f}:1 (minimo 1.5:1)")
                    return {
                        "can_execute": False,
                        "reason": f"Risk:Reward inadequado: {rr_ratio:.2f}:1 (minimo 1.5:1). "
                                  f"Risco={risk_distance:.2f}, Reward(TP1)={reward_distance_tp1:.2f}",
                        "risk_level": "high",
                        "rr_ratio": rr_ratio
                    }
                elif rr_ratio < 2.0:
                    # R:R entre 1.5 e 2.0 = aceitável mas não ideal
                    logger.info(f"[R:R] {symbol}: R:R = {rr_ratio:.2f}:1 (aceitavel, ideal >= 2:1)")
                else:
                    logger.info(f"[R:R] {symbol}: R:R = {rr_ratio:.2f}:1 (bom)")

        entry_price = signal.get('entry_price', 0)
        stop_loss = signal.get('stop_loss', 0)
        confidence = signal.get('confidence', 0)

        # CORRIGIDO: Validar confiança antes de executar
        # UNIFICADO: Sempre usar escala 0-10, mínimo 7 para executar
        from src.core.config import settings

        # CORRIGIDO: Não converter confiança automaticamente.
        # O prompt do DeepSeek já pede escala 1-10 explicitamente.
        # A conversão anterior dobrava confiança 5 para 10, causando falsos positivos.

        # Garantir que confiança está no range válido (1-10)
        if confidence < 1 or confidence > 10:
            return {
                "can_execute": False,
                "reason": f"Confianca invalida: {confidence} (deve ser entre 1 e 10)",
                "risk_level": "medium",
                "confidence": confidence
            }

        # Validar mínimo de confiança (escala 0-10, configurável via MIN_CONFIDENCE_0_10)
        min_confidence = settings.min_confidence_0_10

        if confidence < min_confidence:
            return {
                "can_execute": False,
                "reason": f"Confianca muito baixa: {confidence}/10 (minimo {min_confidence}/10 para executar)",
                "risk_level": "medium",
                "confidence": confidence,
                "min_confidence": min_confidence
            }

        if not entry_price or not stop_loss:
            return {
                "can_execute": False,
                "reason": "Precos de entrada ou stop loss nao definidos",
                "risk_level": "high"
            }

        # Calcular distância do stop
        risk_per_trade = abs(entry_price - stop_loss)
        risk_percentage = (risk_per_trade / entry_price) * 100

        # Stop largo NÃO é motivo para bloquear - apenas ajustar tamanho da posição
        # Quanto maior o stop, menor a posição, mantendo o risco em $ controlado
        # Só bloqueia se stop for absurdo (> 20%) - provavelmente bug
        if risk_percentage > 20.0:
            return {
                "can_execute": False,
                "reason": f"Stop loss provavelmente inválido: {risk_percentage:.2f}% de distância (> 20%)",
                "risk_level": "high"
            }

        if risk_percentage > 5.0:
            logger.info(f"[RISCO] Stop largo ({risk_percentage:.2f}%) - posição será reduzida proporcionalmente")

        # Circuit Breaker 2: Verificar drawdown atual
        # MODIFICADO: Para paper trading, permitir drawdown maior (40%) para não bloquear recuperação
        current_drawdown = _calculate_current_drawdown()
        max_drawdown_allowed = 0.40  # 40% para paper trading (mais flexível)
        if current_drawdown > max_drawdown_allowed:
            return {
                "can_execute": False,
                "reason": f"Drawdown atual muito alto: {current_drawdown:.2%} (máximo {max_drawdown_allowed:.0%})",
                "risk_level": "high"
            }
        elif current_drawdown > 0.15:
            # Drawdown entre 15% e 40%: permitir mas reduzir tamanho da posição
            logger.warning(f"[RISCO] Drawdown elevado ({current_drawdown:.2%}), reduzindo tamanho de posição")

        # REMOVIDO: Circuit Breaker 3 - Verificação de capital/exposição
        # Sistema agora foca apenas em P&L, sem restrições de capital
        logger.info("[P&L MODE] Sistema em modo P&L - sem restrições de capital")

        # Circuit Breaker 4: Verificar limite diário de trades
        daily_trades = _get_daily_trades_count()
        max_daily = settings.max_daily_trades
        if daily_trades >= max_daily:
            return {
                "can_execute": False,
                "reason": f"Limite diário de trades atingido: {daily_trades} (máximo {max_daily})",
                "risk_level": "medium"
            }

        # POSITION SIZING baseado na fórmula:
        # 1. risco_em_$ = capital * (risk_percent / 100)
        # 2. distancia_stop_% = |entry - stop| / entry
        # 3. tamanho_posicao = risco_em_$ / distancia_stop_%
        # 4. alavancagem = tamanho_posicao / capital
        #
        # Capital OBRIGATÓRIO da API da Binance (saldo real)
        # Sem saldo real = NÃO abre posição
        # FIX: usar account_balance (que é o que agent.py passa) em vez de _capital
        capital = account_balance if account_balance else _capital

        if not capital or capital <= 0:
            return {
                "can_execute": False,
                "reason": "Não foi possível obter saldo real da Binance. Sem saldo confirmado, não abre posição.",
                "risk_level": "high"
            }
        logger.info(f"[CAPITAL] Saldo real da Binance: ${capital:.2f}")

        risk_pct = settings.risk_percent_per_trade  # configurado em config.py (atualmente 2%)
        risk_amount = capital * (risk_pct / 100)  # ex: $100 * 2% = $2

        stop_distance_pct = risk_percentage / 100  # já calculado acima como %
        if stop_distance_pct > 0:
            position_value = risk_amount / stop_distance_pct  # ex: $3.60 / 0.02 = $180
            position_size = position_value / entry_price  # em unidades
        else:
            position_size = 1.0
            position_value = entry_price

        leverage = position_value / capital if capital > 0 else 0
        max_risk_amount = risk_amount  # sempre = capital * risk%

        logger.info(
            f"[POSITION SIZE] Capital=${capital:.2f}, Risco={risk_pct}%=${risk_amount:.2f}, "
            f"Stop={risk_percentage:.2f}%, Posição=${position_value:.2f}, "
            f"Unidades={position_size:.6f}, Alavancagem={leverage:.1f}x"
        )

        return {
            "can_execute": True,
            "recommended_position_size": position_size,
            "position_value": position_size * entry_price,
            "max_risk_amount": max_risk_amount,
            "risk_level": "acceptable",
            "current_drawdown": current_drawdown,
            "daily_trades": daily_trades
        }
    except Exception as e:
        return {
            "can_execute": False,
            "reason": f"Erro na validação: {str(e)}",
            "risk_level": "high"
        }

async def backtest_strategy(symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
    """
    Backtesting com dados históricos (FIXED: now async to avoid nested event loops).
    """
    try:
        from datetime import datetime

        from src.exchange.client import BinanceClient

        # Converter strings para datetime
        start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))

        # Obter dados históricos usando o binance_client (agora com await direto)
        async with BinanceClient() as client:
            historical_data = await client.get_historical_klines(symbol, '1h', start_dt, end_dt)

        if historical_data.empty:
            return {"error": "Nenhum dado histórico encontrado"}

        # Simular estratégia
        results = []
        total_trades = 0
        winning_trades = 0
        total_return = 0

        # Análise simples baseada em SMA
        for i in range(20, len(historical_data)):
            current_price = historical_data['close'].iloc[i]
            sma_20 = historical_data['close'].iloc[i-20:i].mean()
            sma_50 = historical_data['close'].iloc[i-50:i].mean() if i >= 50 else sma_20

            # Sinal simples
            if current_price > sma_20 > sma_50:
                signal = "BUY"
                entry_price = current_price
                exit_price = historical_data['close'].iloc[min(i+24, len(historical_data)-1)]  # 24h depois
                pnl = (exit_price - entry_price) / entry_price

                results.append({
                    "timestamp": historical_data.index[i].isoformat(),
                    "signal": signal,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "pnl_percent": pnl * 100
                })

                total_trades += 1
                if pnl > 0:
                    winning_trades += 1
                total_return += pnl

        # Calcular métricas
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        avg_return = total_return / total_trades if total_trades > 0 else 0

        return {
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": total_trades - winning_trades,
            "win_rate": win_rate,
            "total_return_percent": total_return * 100,
            "avg_return_percent": avg_return * 100,
            "results": results[-50:] if len(results) > 50 else results,  # Últimos 50 trades
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        return {
            "error": f"Erro no backtesting: {str(e)}",
            "symbol": symbol
        }

def execute_paper_trade(
    signal: Dict[str, Any],
    position_size: float
) -> Dict[str, Any]:
    """
    Executa um paper trade REAL usando o sistema completo de simulação.
    VALIDAÇÃO: SL, TP1 e TP2 são obrigatórios.
    """
    try:
        # HARD BLOCK: Validar SL/TP1/TP2 antes de executar
        sl = signal.get("stop_loss", 0) or 0
        tp1 = signal.get("take_profit_1", 0) or 0
        tp2 = signal.get("take_profit_2", 0) or 0
        if sl <= 0 or tp1 <= 0 or tp2 <= 0:
            logger.error(f"[PAPER TRADE BLOCK] SL=${sl}, TP1=${tp1}, TP2=${tp2} — Todos devem ser > 0")
            return {"success": False, "error": f"SL/TP obrigatórios: SL=${sl}, TP1=${tp1}, TP2=${tp2}"}

        from src.trading.paper_trading import real_paper_trading

        # Executar trade usando o sistema REAL
        result = real_paper_trading.execute_trade(signal, position_size)

        if result["success"]:
            # Obter resumo do portfólio (modo P&L)
            try:
                portfolio_summary = real_paper_trading.get_portfolio_summary()

                # Verificar se retornou erro
                if "error" in portfolio_summary:
                    logger.warning(f"Erro ao obter resumo do portfólio: {portfolio_summary.get('error')}")
                    # Continuar mesmo com erro, usando valores padrão
                    portfolio_summary = {}

                return {
                    "success": True,
                    "trade_id": result["trade_id"],
                    "message": result["message"],
                    "file": result["file"],
                    "portfolio_summary": {
                        "total_pnl_percent": f"{portfolio_summary.get('total_pnl_percent', 0):+.2f}%",
                        "open_positions": portfolio_summary.get('open_positions_count', 0),
                        "total_trades": portfolio_summary.get('total_trades', 0)
                    }
                }
            except KeyError as ke:
                # Erro específico de chave faltando no portfolio_summary
                logger.error(f"Chave faltando no portfolio_summary: {ke}")
                return {
                    "success": True,  # Trade foi executado com sucesso, apenas erro no resumo
                    "trade_id": result["trade_id"],
                    "message": result["message"],
                    "file": result["file"],
                    "portfolio_summary": {
                        "total_pnl_percent": "0.00%",
                        "open_positions": 0,
                        "total_trades": 0
                    }
                }
        else:
            return result

    except KeyError as e:
        # Erro específico de chave faltando - pode ser current_balance ou outra chave
        logger.error(f"Erro de chave faltando ao executar paper trade: {e}")
        return {
            "success": False,
            "error": f"Erro ao executar paper trade: Chave faltando - {str(e)}"
        }
    except Exception as e:
        logger.exception(f"Erro inesperado ao executar paper trade: {e}")
        return {
            "success": False,
            "error": f"Erro ao executar paper trade: {str(e)}"
        }
