"""
DeepSeek prompt preparation and analysis
Handles LLM interaction for trading signal generation
"""
import json
import os
import re
from typing import Dict, Any, List
from datetime import datetime

from src.core.logger import get_logger
from src.analysis.market_data import get_market_data
from src.analysis.indicators import (
    analyze_technical_indicators,
    _classify_rsi, _interpret_adx, _interpret_macd_momentum,
    _classify_bollinger_position, _detect_ema_alignment,
    _interpret_funding_rate, _classify_orderbook_imbalance,
    _calculate_suggested_stops
)
from src.analysis.sentiment import analyze_market_sentiment
from src.analysis.multi_timeframe import analyze_multiple_timeframes
from src.analysis.order_flow import analyze_order_flow
from src.analysis.market_classifier import classify_market_condition
from src.trading.risk_manager import validate_risk_and_position

logger = get_logger(__name__)


def _identify_conflicting_signals(data: Dict) -> List[str]:
    """Identifica sinais que se contradizem"""
    conflicts = []
    try:
        trend = data.get("trend_analysis", {}).get("primary_trend", "neutral")
        momentum = data.get("trend_analysis", {}).get("momentum", "neutral")
        rsi_zone = data.get("key_indicators", {}).get("rsi", {}).get("zone", "neutral")
        macd_crossover = data.get("key_indicators", {}).get("macd", {}).get("crossover", "neutral")

        if rsi_zone == "oversold" and "bearish" in trend:
            conflicts.append("RSI oversold but strong bearish trend")
        if rsi_zone == "overbought" and "bullish" in trend:
            conflicts.append("RSI overbought but strong bullish trend")
        if macd_crossover == "bearish" and "bullish" in momentum:
            conflicts.append("MACD bearish crossover but bullish momentum")
        if macd_crossover == "bullish" and "bearish" in momentum:
            conflicts.append("MACD bullish crossover but bearish momentum")

        orderbook_bias = data.get("volume_flow", {}).get("orderbook_bias", "neutral")
        if "bullish" in trend and "sell" in orderbook_bias:
            conflicts.append("Bullish trend but sell pressure in orderbook")
        if "bearish" in trend and "buy" in orderbook_bias:
            conflicts.append("Bearish trend but buy pressure in orderbook")
    except Exception as e:
        logger.warning(f"Erro ao identificar sinais conflitantes: {e}")

    return conflicts


def _interpret_confluence(bullish_count: int, bearish_count: int) -> str:
    """Interpreta alinhamento de timeframes"""
    try:
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


def _calculate_overall_bias(data: Dict) -> Dict[str, Any]:
    """Calcula score agregado de -10 a +10 com interpretação"""
    try:
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0

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

        key_indicators = data.get("key_indicators") or {}
        rsi_data = key_indicators.get("rsi") or {}
        rsi_hint = rsi_data.get("action_hint", "wait")
        if "buy" in rsi_hint:
            bullish_count += 1
        elif "sell" in rsi_hint:
            bearish_count += 1
        else:
            neutral_count += 1

        macd_data = key_indicators.get("macd") or {}
        macd_crossover = macd_data.get("crossover", "neutral")
        if macd_crossover == "bullish":
            bullish_count += 1
        elif macd_crossover == "bearish":
            bearish_count += 1
        else:
            neutral_count += 1

        ema_structure = key_indicators.get("ema_structure") or {}
        ema_alignment = ema_structure.get("ema_alignment", "mixed")
        if "bullish" in ema_alignment:
            bullish_count += 1
        elif "bearish" in ema_alignment:
            bearish_count += 1
        else:
            neutral_count += 1

        volume_flow = data.get("volume_flow") or {}
        orderbook_bias = volume_flow.get("orderbook_bias", "neutral")
        if "buy" in orderbook_bias:
            bullish_count += 1
        elif "sell" in orderbook_bias:
            bearish_count += 1
        else:
            neutral_count += 1

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

        overall_bias = max(-10, min(10, bullish_count - bearish_count))

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
            "bullish_factors_count": 0, "bearish_factors_count": 0,
            "neutral_factors_count": 0, "overall_bias": 0,
            "overall_bias_interpretation": "neutral", "recommended_action": "WAIT"
        }


async def prepare_analysis_for_llm(symbol: str) -> Dict[str, Any]:
    """Prepara dados SUMARIZADOS e INTERPRETADOS para envio ao DeepSeek."""
    try:
        market_data = await get_market_data(symbol)
        technical_indicators = await analyze_technical_indicators(symbol)
        sentiment = await analyze_market_sentiment(symbol)
        multi_timeframe = await analyze_multiple_timeframes(symbol)
        order_flow = await analyze_order_flow(symbol)

        errors = []
        if "error" in market_data:
            errors.append(f"market_data: {market_data.get('error')}")
            logger.error(f"[{symbol}] Erro em market_data: {market_data.get('error')}")
        if "error" in technical_indicators:
            errors.append(f"technical_indicators: {technical_indicators.get('error')}")
            logger.error(f"[{symbol}] Erro em technical_indicators: {technical_indicators.get('error')}")
        if "error" in sentiment:
            logger.warning(f"[{symbol}] Erro em sentiment: {sentiment.get('error')}")
        if "error" in multi_timeframe:
            logger.warning(f"[{symbol}] Erro em multi_timeframe: {multi_timeframe.get('error')}")
        if "error" in order_flow:
            logger.warning(f"[{symbol}] Erro em order_flow: {order_flow.get('error')}")

        if "error" in market_data or "error" in technical_indicators:
            error_summary = "; ".join(errors)
            return {"error": f"Erro ao coletar dados de mercado: {error_summary}", "symbol": symbol, "timestamp": datetime.now().isoformat()}

        current_price = market_data.get("current_price", 0)
        price_change_24h = market_data.get("price_change_24h", 0)
        high_24h = market_data.get("high_24h", current_price)
        low_24h = market_data.get("low_24h", current_price)
        position_in_range = ((current_price - low_24h) / (high_24h - low_24h) * 100) if (high_24h - low_24h) > 0 else 50

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

        support = technical_indicators.get("support", current_price * 0.95)
        resistance = technical_indicators.get("resistance", current_price * 1.05)
        fib_levels = technical_indicators.get("fibonacci_levels", {})
        volume_profile_data = technical_indicators.get("volume_profile") or {}
        poc_price = volume_profile_data.get("poc_price", current_price)

        distance_to_support = ((current_price - support) / current_price) * 100
        distance_to_resistance = ((resistance - current_price) / current_price) * 100

        rsi_classification = _classify_rsi(rsi_value)
        adx_interpretation = _interpret_adx(adx_value)
        macd_momentum = _interpret_macd_momentum(macd_hist)
        bb_zone = _classify_bollinger_position(bb_position)
        ema_alignment = _detect_ema_alignment(ema_20, ema_50, ema_200, current_price)
        funding_interpretation = _interpret_funding_rate(market_data.get("funding_rate", 0))
        orderbook_bias = _classify_orderbook_imbalance(order_flow.get("orderbook_imbalance", 0))
        suggested_stops = _calculate_suggested_stops(atr_value, current_price)

        confluence_bullish = multi_timeframe.get("bullish_count", 0)
        confluence_bearish = multi_timeframe.get("bearish_count", 0)
        confluence_score = confluence_bullish - confluence_bearish
        confluence_interpretation = _interpret_confluence(confluence_bullish, confluence_bearish)

        timeframe_alignment = {}
        for tf in ["5m", "15m", "1h", "4h", "1d"]:
            tf_data = multi_timeframe.get("timeframes", {}).get(tf, {})
            timeframe_alignment[tf] = tf_data.get("trend", "neutral")

        primary_trend = technical_indicators.get("trend", "neutral")
        momentum = technical_indicators.get("momentum", "neutral")

        atr_pct = (atr_value / current_price) * 100
        if atr_pct > 0.05:
            volatility_level = "extreme"
        elif atr_pct > 0.03:
            volatility_level = "high"
        elif atr_pct > 0.015:
            volatility_level = "normal"
        else:
            volatility_level = "low"

        analysis = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "price_context": {"current": current_price, "change_24h_pct": price_change_24h, "high_24h": high_24h, "low_24h": low_24h, "position_in_range_pct": position_in_range},
            "trend_analysis": {"primary_trend": primary_trend, "trend_strength_adx": adx_value, "trend_strength_interpretation": adx_interpretation, "momentum": momentum, "timeframe_alignment": timeframe_alignment, "confluence_score": confluence_score, "confluence_interpretation": confluence_interpretation},
            "key_indicators": {
                "rsi": {"value": rsi_value, "zone": rsi_classification["zone"], "action_hint": rsi_classification["action_hint"]},
                "macd": {"histogram": macd_hist, "crossover": macd_crossover, "momentum_direction": macd_momentum},
                "bollinger": {"position": bb_position, "zone": bb_zone, "squeeze_detected": False},
                "ema_structure": {"price_vs_ema20": "above" if current_price > ema_20 else "below", "price_vs_ema50": "above" if current_price > ema_50 else "below", "price_vs_ema200": "above" if ema_200 and current_price > ema_200 else "below" if ema_200 else "N/A", "ema_alignment": ema_alignment},
                "obv_trend": obv_trend
            },
            "key_levels": {"immediate_support": support, "immediate_resistance": resistance, "fib_382": fib_levels.get("fib_38.2", support), "fib_50": fib_levels.get("fib_50", current_price), "fib_618": fib_levels.get("fib_61.8", resistance), "volume_poc": poc_price, "distance_to_support_pct": distance_to_support, "distance_to_resistance_pct": distance_to_resistance},
            "volume_flow": {"volume_24h": market_data.get("volume_24h", 0), "volume_trend": "stable", "obv_trend": obv_trend, "orderbook_imbalance": order_flow.get("orderbook_imbalance", 0), "orderbook_bias": orderbook_bias, "cvd_direction": "positive" if order_flow.get("cvd", 0) > 0 else "negative"},
            "sentiment": {"overall": sentiment.get("sentiment", "neutral"), "confidence": sentiment.get("confidence", 0.5), "funding_rate": market_data.get("funding_rate", 0), "funding_interpretation": funding_interpretation, "open_interest_trend": "stable"},
            "volatility": {"atr_value": atr_value, "atr_pct": atr_pct, "level": volatility_level, "suggested_stop_pct": suggested_stops["suggested_stop_pct"], "suggested_tp1_pct": suggested_stops["suggested_tp1_pct"], "suggested_tp2_pct": suggested_stops["suggested_tp2_pct"]},
            "conflicting_signals": [],
            "aggregated_scores": {}
        }

        analysis["conflicting_signals"] = _identify_conflicting_signals(analysis)
        analysis["aggregated_scores"] = _calculate_overall_bias(analysis)

        payload_size = len(json.dumps(analysis))
        if payload_size > 10000:
            logger.warning(f"Payload muito grande: {payload_size} bytes para {symbol}")

        return analysis

    except Exception as e:
        logger.exception(f"Erro ao preparar análise para LLM: {e}")
        return {"error": f"Erro ao preparar análise: {str(e)}", "symbol": symbol, "timestamp": datetime.now().isoformat()}


def _create_analysis_prompt(analysis: Dict[str, Any], market_classification: Dict[str, Any] = None) -> str:
    """Cria prompt estruturado para o DeepSeek com classificação de mercado"""
    try:
        conflicting = analysis.get("conflicting_signals", [])
        conflicting_text = "\n".join([f"- {s}" for s in conflicting]) if conflicting else "- Nenhum conflito identificado"

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

## NÍVEIS CHAVE

- Suporte: ${analysis['key_levels']['immediate_support']:,.2f} ({analysis['key_levels']['distance_to_support_pct']:+.2f}%)
- Resistência: ${analysis['key_levels']['immediate_resistance']:,.2f} ({analysis['key_levels']['distance_to_resistance_pct']:+.2f}%)
- POC Volume: ${analysis['key_levels']['volume_poc']:,.2f}

## VOLUME E FLUXO

- Pressão orderbook: {analysis['volume_flow']['orderbook_bias']}
- OBV: {analysis['volume_flow']['obv_trend']}

## SENTIMENTO

- Geral: {analysis['sentiment']['overall']}
- Funding: {analysis['sentiment']['funding_interpretation']}

## VOLATILIDADE

- Nível: {analysis['volatility']['level']}
- Stop sugerido: {analysis['volatility']['suggested_stop_pct']:.2f}%
- TP1 sugerido: {analysis['volatility']['suggested_tp1_pct']:.2f}%
- TP2 sugerido: {analysis['volatility']['suggested_tp2_pct']:.2f}%

## SINAIS CONFLITANTES

{conflicting_text}

## SCORE AGREGADO

- Fatores bullish: {analysis['aggregated_scores']['bullish_factors_count']}
- Fatores bearish: {analysis['aggregated_scores']['bearish_factors_count']}
- Bias geral: {analysis['aggregated_scores']['overall_bias']}/10 ({analysis['aggregated_scores']['overall_bias_interpretation']})
- Ação recomendada: {analysis['aggregated_scores']['recommended_action']}

---

## IMPORTANTE: ESCALA DE CONFIANÇA (0-10)

A confiança deve ser um número inteiro de 1 a 10.
**CRÍTICO**: Apenas sinais com confiança >= 7 serão executados pelo sistema.

---

## INSTRUÇÕES

1. **ANALISE O TIPO DE OPERAÇÃO RECOMENDADO** ({recommended_type}) e considere se concorda.
2. **SEJA DECISIVO** - Forneça BUY ou SELL sempre que houver oportunidade.
3. **AJUSTE OS PARÂMETROS** conforme o tipo de operação escolhido.

---

RESPONDA APENAS COM JSON:

```json
{{{{
    "signal": "BUY" ou "SELL" ou "NO_SIGNAL",
    "operation_type": "SCALP" ou "DAY_TRADE" ou "SWING_TRADE" ou "POSITION_TRADE",
    "entry_price": <número>,
    "stop_loss": <número>,
    "take_profit_1": <número>,
    "take_profit_2": <número>,
    "confidence": <1-10>,
    "reasoning": "<justificativa>"
}}}}
```
"""
    except Exception as e:
        logger.exception(f"Erro ao criar prompt: {e}")
        return "Erro ao criar prompt de análise."


async def get_deepseek_analysis(symbol: str) -> Dict[str, Any]:
    """Prepara análise otimizada para o DeepSeek e chama diretamente."""
    try:
        from agno.models.deepseek import DeepSeek
        from agno.agent import Agent

        analysis = await prepare_analysis_for_llm(symbol)
        if "error" in analysis:
            return analysis

        market_classification = classify_market_condition(analysis)
        logger.info(f"[CLASSIFICAÇÃO] {symbol}: {market_classification['operation_type']} (confiança: {market_classification['confidence']}/10)")

        prompt = _create_analysis_prompt(analysis, market_classification)

        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY não encontrada.")

        model = DeepSeek(id="deepseek-chat", api_key=api_key, temperature=0.3, max_tokens=1000)
        agent = Agent(
            model=model,
            instructions="Você é um trader profissional. Analise os dados e forneça um sinal de trading em formato JSON."
        )

        logger.info(f"[DEEPSEEK] Chamando DeepSeek diretamente para {symbol}")
        response = await agent.arun(prompt)

        response_content = str(response.content) if hasattr(response, 'content') else str(response)

        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_content, re.DOTALL)
        if json_match:
            try:
                signal_json = json.loads(json_match.group(1))
                operation_type = signal_json.get("operation_type", market_classification["operation_type"])
                logger.info(f"[DEEPSEEK] {symbol}: {signal_json.get('signal', 'N/A')} ({operation_type}) - Confiança: {signal_json.get('confidence', 0)}/10")

                return {
                    "signal": signal_json.get("signal", "NO_SIGNAL"),
                    "operation_type": operation_type,
                    "entry_price": signal_json.get("entry_price"),
                    "stop_loss": signal_json.get("stop_loss"),
                    "take_profit_1": signal_json.get("take_profit_1"),
                    "take_profit_2": signal_json.get("take_profit_2"),
                    "confidence": signal_json.get("confidence", 5),
                    "reasoning": signal_json.get("reasoning", ""),
                    "market_classification": market_classification,
                    "analysis_data": analysis,
                    "deepseek_prompt": prompt,
                    "raw_response": response_content,
                    "timestamp": datetime.now().isoformat()
                }
            except json.JSONDecodeError as e:
                logger.warning(f"[DEEPSEEK] Erro ao decodificar JSON: {e}")

        logger.warning(f"[DEEPSEEK] Não foi possível extrair JSON, retornando resposta bruta")
        return {
            "analysis_data": analysis,
            "deepseek_prompt": prompt,
            "raw_response": response_content,
            "needs_agent_processing": True,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.exception(f"Erro na preparação para DeepSeek: {e}")
        return {"error": f"Erro na preparação para DeepSeek: {str(e)}", "timestamp": datetime.now().isoformat()}


async def backtest_strategy(symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
    """Backtesting com dados históricos."""
    try:
        from datetime import timedelta
        from src.exchange.client import BinanceClient

        start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))

        async with BinanceClient() as client:
            historical_data = await client.get_historical_klines(symbol, '1h', start_dt, end_dt)

        if historical_data.empty:
            return {"error": "Nenhum dado histórico encontrado"}

        results = []
        total_trades = 0
        winning_trades = 0
        total_return = 0

        for i in range(20, len(historical_data)):
            current_price = historical_data['close'].iloc[i]
            sma_20 = historical_data['close'].iloc[i-20:i].mean()
            sma_50 = historical_data['close'].iloc[i-50:i].mean() if i >= 50 else sma_20

            if current_price > sma_20 > sma_50:
                entry_price = current_price
                exit_price = historical_data['close'].iloc[min(i+24, len(historical_data)-1)]
                pnl = (exit_price - entry_price) / entry_price
                results.append({
                    "timestamp": historical_data.index[i].isoformat(),
                    "signal": "BUY", "entry_price": entry_price,
                    "exit_price": exit_price, "pnl": pnl, "pnl_percent": pnl * 100
                })
                total_trades += 1
                if pnl > 0:
                    winning_trades += 1
                total_return += pnl

        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        avg_return = total_return / total_trades if total_trades > 0 else 0

        return {
            "symbol": symbol, "start_date": start_date, "end_date": end_date,
            "total_trades": total_trades, "winning_trades": winning_trades,
            "losing_trades": total_trades - winning_trades, "win_rate": win_rate,
            "total_return_percent": total_return * 100, "avg_return_percent": avg_return * 100,
            "results": results[-50:] if len(results) > 50 else results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": f"Erro no backtesting: {str(e)}", "symbol": symbol}


def execute_paper_trade(signal: Dict[str, Any], position_size: float) -> Dict[str, Any]:
    """Executa um paper trade REAL usando o sistema completo de simulação."""
    try:
        from src.trading.paper_trading import real_paper_trading

        result = real_paper_trading.execute_trade(signal, position_size)

        if result["success"]:
            try:
                portfolio_summary = real_paper_trading.get_portfolio_summary()
                if "error" in portfolio_summary:
                    logger.warning(f"Erro ao obter resumo do portfólio: {portfolio_summary.get('error')}")
                    portfolio_summary = {}
                return {
                    "success": True, "trade_id": result["trade_id"],
                    "message": result["message"], "file": result["file"],
                    "portfolio_summary": {
                        "total_pnl_percent": f"{portfolio_summary.get('total_pnl_percent', 0):+.2f}%",
                        "open_positions": portfolio_summary.get('open_positions_count', 0),
                        "total_trades": portfolio_summary.get('total_trades', 0)
                    }
                }
            except KeyError as ke:
                logger.error(f"Chave faltando no portfolio_summary: {ke}")
                return {
                    "success": True, "trade_id": result["trade_id"],
                    "message": result["message"], "file": result["file"],
                    "portfolio_summary": {"total_pnl_percent": "0.00%", "open_positions": 0, "total_trades": 0}
                }
        else:
            return result

    except KeyError as e:
        logger.error(f"Erro de chave faltando ao executar paper trade: {e}")
        return {"success": False, "error": f"Erro ao executar paper trade: Chave faltando - {str(e)}"}
    except Exception as e:
        logger.exception(f"Erro inesperado ao executar paper trade: {e}")
        return {"success": False, "error": f"Erro ao executar paper trade: {str(e)}"}
