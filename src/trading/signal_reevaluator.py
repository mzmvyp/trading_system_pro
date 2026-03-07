"""
Sistema de Reavaliacao de Sinais Abertos
Monitora posicoes abertas e usa DeepSeek para reavaliar e sugerir acoes

Acoes possiveis:
- HOLD: Manter posicao sem alteracoes
- CLOSE: Fechar posicao imediatamente
- CLOSE_PARTIAL: Fechar parte da posicao (ex: 50%)
- MOVE_STOP_BREAKEVEN: Mover stop loss para ponto de entrada (break-even)
- TRAILING_STOP: Ajustar stop loss para travar lucros (trailing)
- ADJUST_STOP: Ajustar stop loss para novo nivel
- ADJUST_TP: Ajustar take profit para novo nivel
- ADD_TO_POSITION: Adicionar a posicao (se houver oportunidade)

NOVO: Ajuste automatico de stop apos TP1 atingido usando DeepSeek
NOVO: Trailing Stop Dinamico baseado em ATR
NOVO: Time-Based Exit (saida por tempo)
"""

import json
import os
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path

from src.core.logger import get_logger
from src.core.config import settings

logger = get_logger(__name__)

# Importar position_manager para funcionalidades avancadas
try:
    from position_manager import position_manager
    POSITION_MANAGER_AVAILABLE = True
except ImportError:
    POSITION_MANAGER_AVAILABLE = False
    logger.warning("[REEVALUATOR] position_manager nao disponivel")

# Importar stop_adjuster para ajuste apos TP1
try:
    from stop_adjuster import adjust_stop_on_tp1_hit
    STOP_ADJUSTER_AVAILABLE = True
except ImportError:
    STOP_ADJUSTER_AVAILABLE = False
    logger.warning("[REEVALUATOR] stop_adjuster nao disponivel")


class SignalReevaluator:
    """
    Sistema de reavaliacao de sinais/posicoes abertas.
    Coleta dados de mercado atualizados e consulta DeepSeek para sugestoes.
    """

    def __init__(self):
        """Inicializa o sistema de reavaliacao"""
        self.last_reevaluation = {}  # {position_key: datetime}
        self.reevaluation_history = []  # Historico de reavaliacoes
        self.tp1_stop_adjusted = {}  # {position_key: datetime} - posicoes que ja tiveram stop ajustado apos TP1

        # Criar diretorio para logs de reavaliacao
        Path("reevaluation_logs").mkdir(exist_ok=True)

        logger.info("[REEVALUATOR] Sistema de reavaliacao inicializado")
        if STOP_ADJUSTER_AVAILABLE:
            logger.info("[REEVALUATOR] Ajuste de stop apos TP1 disponivel")

    async def get_open_positions(self) -> List[Dict[str, Any]]:
        """
        Busca todas as posicoes abertas.
        Em modo paper: le do arquivo state.json
        Em modo real: consulta Binance Futures API

        Returns:
            Lista de posicoes abertas com detalhes
        """
        positions = []

        try:
            if settings.trading_mode == "paper":
                # Modo Paper: ler do state.json
                positions = await self._get_paper_positions()
            else:
                # Modo Real: consultar Binance + state.json para contexto
                positions = await self._get_real_positions()

            logger.info(f"[REEVALUATOR] {len(positions)} posicoes abertas encontradas")
            return positions

        except Exception as e:
            logger.exception(f"[REEVALUATOR] Erro ao buscar posicoes abertas: {e}")
            return []

    async def _get_paper_positions(self) -> List[Dict[str, Any]]:
        """Busca posicoes do paper trading (state.json)"""
        positions = []

        try:
            state_file = Path("portfolio/state.json")
            if not state_file.exists():
                return []

            with open(state_file, "r", encoding="utf-8") as f:
                state = json.load(f)

            for pos_key, pos in state.get("positions", {}).items():
                if pos.get("status") == "OPEN":
                    # Adicionar chave da posicao para referencia
                    pos["position_key"] = pos_key
                    positions.append(pos)

            return positions

        except Exception as e:
            logger.error(f"[REEVALUATOR] Erro ao ler posicoes paper: {e}")
            return []

    async def _get_real_positions(self) -> List[Dict[str, Any]]:
        """
        Busca posicoes reais da Binance Futures.
        Combina dados da Binance com contexto do state.json
        """
        positions = []

        try:
            from src.exchange.executor import BinanceFuturesExecutor

            async with BinanceFuturesExecutor() as executor:
                # Buscar todas as posicoes abertas
                account_info = await executor.get_account_info()

                for position in account_info.get("positions", []):
                    # Filtrar apenas posicoes com quantidade != 0
                    position_amt = float(position.get("positionAmt", 0))
                    if position_amt != 0:
                        symbol = position.get("symbol")

                        # Buscar contexto do state.json se existir
                        context = await self._get_position_context(symbol)

                        # Se nao tem timestamp no context, assumir que foi aberta ha tempo suficiente
                        # (posicao pode ter sido aberta manualmente ou por outro sistema)
                        context_timestamp = context.get("timestamp")
                        if not context_timestamp:
                            # Assumir que foi aberta ha pelo menos o tempo minimo + intervalo
                            min_required_hours = settings.reevaluation_min_time_open_hours + settings.reevaluation_interval_hours
                            assumed_open_time = datetime.now() - timedelta(hours=min_required_hours)
                            context_timestamp = assumed_open_time.isoformat()
                            logger.info(f"[REEVALUATOR] {symbol}: Sem timestamp no state.json, assumindo aberta ha {min_required_hours}h")
                        
                        pos_data = {
                            "position_key": f"{symbol}_BINANCE",
                            "symbol": symbol,
                            "source": "BINANCE",
                            "signal": "BUY" if position_amt > 0 else "SELL",
                            "entry_price": float(position.get("entryPrice", 0)),
                            "position_size": abs(position_amt),
                            "unrealized_pnl": float(position.get("unrealizedProfit", 0)),
                            "leverage": int(position.get("leverage", 1)),
                            "margin_type": position.get("marginType", "cross"),
                            "liquidation_price": float(position.get("liquidationPrice", 0)),
                            "status": "OPEN",
                            # Contexto adicional do state.json
                            "stop_loss": context.get("stop_loss"),
                            "take_profit_1": context.get("take_profit_1"),
                            "take_profit_2": context.get("take_profit_2"),
                            "operation_type": context.get("operation_type", "SWING_TRADE"),
                            "timestamp": context_timestamp
                        }
                        positions.append(pos_data)

            return positions

        except ImportError:
            logger.warning("[REEVALUATOR] BinanceFuturesExecutor nao disponivel, usando apenas paper positions")
            return await self._get_paper_positions()
        except Exception as e:
            logger.error(f"[REEVALUATOR] Erro ao buscar posicoes reais: {e}")
            return await self._get_paper_positions()

    async def _get_position_context(self, symbol: str) -> Dict[str, Any]:
        """Busca contexto adicional de uma posicao do state.json"""
        try:
            state_file = Path("portfolio/state.json")
            if not state_file.exists():
                return {}

            with open(state_file, "r", encoding="utf-8") as f:
                state = json.load(f)

            # Procurar posicao por simbolo
            for pos_key, pos in state.get("positions", {}).items():
                if pos.get("symbol") == symbol and pos.get("status") == "OPEN":
                    return pos

            return {}

        except Exception as e:
            logger.debug(f"[REEVALUATOR] Sem contexto para {symbol}: {e}")
            return {}

    def should_reevaluate(self, position: Dict[str, Any]) -> bool:
        """
        Verifica se uma posicao deve ser reavaliada baseado em:
        - Tempo desde ultima reavaliacao
        - Tempo minimo que a posicao esta aberta
        - Configuracoes do sistema

        Args:
            position: Dados da posicao

        Returns:
            True se deve reavaliar, False caso contrario
        """
        if not settings.reevaluation_enabled:
            return False

        position_key = position.get("position_key", "unknown")

        # Verificar tempo que a posicao esta aberta
        try:
            open_time = datetime.fromisoformat(position.get("timestamp", datetime.now().isoformat()))
            hours_open = (datetime.now() - open_time).total_seconds() / 3600

            if hours_open < settings.reevaluation_min_time_open_hours:
                logger.info(f"[REEVALUATOR] {position_key}: Posicao aberta ha apenas {hours_open:.2f}h (min: {settings.reevaluation_min_time_open_hours}h) - PULANDO")
                return False
        except Exception as e:
            logger.warning(f"[REEVALUATOR] Erro ao verificar tempo aberto: {e}")

        # Verificar tempo desde ultima reavaliacao
        last_eval = self.last_reevaluation.get(position_key)
        if last_eval:
            hours_since_eval = (datetime.now() - last_eval).total_seconds() / 3600
            if hours_since_eval < settings.reevaluation_interval_hours:
                logger.info(f"[REEVALUATOR] {position_key}: Ultima reavaliacao ha {hours_since_eval:.2f}h (intervalo: {settings.reevaluation_interval_hours}h) - PULANDO")
                return False

        logger.info(f"[REEVALUATOR] {position_key}: OK para reavaliar (aberta ha {hours_open:.2f}h)")
        return True

    async def get_current_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        Coleta dados atualizados do mercado para um simbolo.
        Usa as mesmas funcoes do sistema de geracao de sinais.

        Args:
            symbol: Par de trading (ex: BTCUSDT)

        Returns:
            Dados de mercado atualizados
        """
        try:
            from src.analysis.agno_tools import (
                get_market_data,
                analyze_technical_indicators,
                analyze_market_sentiment,
                analyze_multiple_timeframes,
                analyze_order_flow,
                classify_market_condition
            )

            logger.info(f"[REEVALUATOR] Coletando dados de mercado para {symbol}...")

            # Coletar todos os dados em paralelo
            market_data, technical, sentiment, mtf, order_flow = await asyncio.gather(
                get_market_data(symbol),
                analyze_technical_indicators(symbol),
                analyze_market_sentiment(symbol),
                analyze_multiple_timeframes(symbol),
                analyze_order_flow(symbol),
                return_exceptions=True
            )

            # Tratar excecoes
            if isinstance(market_data, Exception):
                market_data = {"error": str(market_data)}
            if isinstance(technical, Exception):
                technical = {"error": str(technical)}
            if isinstance(sentiment, Exception):
                sentiment = {"error": str(sentiment)}
            if isinstance(mtf, Exception):
                mtf = {"error": str(mtf)}
            if isinstance(order_flow, Exception):
                order_flow = {"error": str(order_flow)}

            # Classificar condicoes de mercado
            market_classification = {}
            if "error" not in technical:
                try:
                    # Preparar dados para classificacao
                    analysis_for_classification = {
                        "volatility": {
                            "level": "MEDIUM",
                            "atr_percent": (technical.get("indicators", {}).get("atr", 0) /
                                          market_data.get("current_price", 1)) * 100 if market_data.get("current_price") else 2.0
                        },
                        "trend_analysis": {
                            "adx_value": technical.get("indicators", {}).get("adx", 25),
                            "trend_strength_interpretation": "MODERATE",
                            "primary_trend": technical.get("trend", "NEUTRAL"),
                            "confluence_score": mtf.get("bullish_count", 0) - mtf.get("bearish_count", 0) if isinstance(mtf, dict) else 0
                        },
                        "key_indicators": {
                            "rsi": {"value": technical.get("indicators", {}).get("rsi", 50)},
                            "macd": {"momentum_direction": technical.get("indicators", {}).get("macd_crossover", "neutral")}
                        },
                        "volume_flow": {
                            "obv_trend": technical.get("indicators", {}).get("obv_trend", "neutral"),
                            "orderbook_bias": order_flow.get("orderbook_imbalance", 0) if isinstance(order_flow, dict) else 0
                        }
                    }
                    market_classification = classify_market_condition(analysis_for_classification)
                except Exception as e:
                    logger.warning(f"[REEVALUATOR] Erro ao classificar mercado: {e}")

            result = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "market_data": market_data,
                "technical_indicators": technical,
                "sentiment": sentiment,
                "multi_timeframe": mtf,
                "order_flow": order_flow,
                "market_classification": market_classification,
                "current_price": market_data.get("current_price", 0) if isinstance(market_data, dict) else 0
            }

            logger.info(f"[REEVALUATOR] Dados coletados para {symbol}: preco=${result['current_price']:.2f}")
            return result

        except Exception as e:
            logger.exception(f"[REEVALUATOR] Erro ao coletar dados de mercado: {e}")
            return {"error": str(e), "symbol": symbol}

    def _create_reevaluation_prompt(
        self,
        position: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> str:
        """
        Cria prompt especializado para DeepSeek reavaliar uma posicao.

        Args:
            position: Dados da posicao aberta
            market_data: Dados atuais do mercado

        Returns:
            Prompt formatado para o DeepSeek
        """
        # Calcular metricas da posicao
        entry_price = position.get("entry_price", 0) or 0
        current_price = market_data.get("current_price", entry_price) or entry_price
        signal_type = position.get("signal", "BUY")
        stop_loss = position.get("stop_loss") or 0  # None vira 0
        take_profit_1 = position.get("take_profit_1") or 0  # None vira 0
        take_profit_2 = position.get("take_profit_2") or 0  # None vira 0
        operation_type = position.get("operation_type", "SWING_TRADE")

        # Calcular P&L atual
        if signal_type == "BUY":
            pnl_percent = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
            distance_to_sl = ((current_price - stop_loss) / current_price) * 100 if stop_loss and stop_loss > 0 else 0
            distance_to_tp1 = ((take_profit_1 - current_price) / current_price) * 100 if take_profit_1 and take_profit_1 > 0 else 0
        else:  # SELL
            pnl_percent = ((entry_price - current_price) / entry_price) * 100 if entry_price > 0 else 0
            distance_to_sl = ((stop_loss - current_price) / current_price) * 100 if stop_loss and stop_loss > 0 else 0
            distance_to_tp1 = ((current_price - take_profit_1) / current_price) * 100 if take_profit_1 and take_profit_1 > 0 else 0

        # Calcular tempo aberto
        try:
            open_time = datetime.fromisoformat(position.get("timestamp", datetime.now().isoformat()))
            hours_open = (datetime.now() - open_time).total_seconds() / 3600
        except:
            hours_open = 0

        # Extrair indicadores tecnicos
        tech = market_data.get("technical_indicators", {})
        indicators = tech.get("indicators", {})
        sentiment = market_data.get("sentiment", {})
        mtf = market_data.get("multi_timeframe", {})
        order_flow = market_data.get("order_flow", {})
        market_class = market_data.get("market_classification", {})

        # Determinar se esta perto de niveis importantes
        price_context = ""
        if take_profit_1 and distance_to_tp1 < 1.0:
            price_context += f"\n- ALERTA: Preco MUITO PROXIMO do TP1 ({distance_to_tp1:.2f}%)"
        if stop_loss and distance_to_sl < 1.0:
            price_context += f"\n- ALERTA: Preco MUITO PROXIMO do Stop Loss ({distance_to_sl:.2f}%)"
        if pnl_percent > 3.0:
            price_context += f"\n- LUCRO SIGNIFICATIVO: {pnl_percent:.2f}% - Considere proteger lucros"
        if pnl_percent < -2.0:
            price_context += f"\n- PREJUIZO: {pnl_percent:.2f}% - Avaliar se vale manter"

        prompt = f"""
Voce e um trader profissional especializado em gestao de posicoes abertas.
Analise a posicao aberta e os dados de mercado atuais para determinar a MELHOR ACAO.

## POSICAO ATUAL

- **Simbolo:** {position.get('symbol')}
- **Tipo:** {signal_type} ({operation_type})
- **Entrada:** ${entry_price:.2f}
- **Preco Atual:** ${current_price:.2f}
- **P&L Atual:** {pnl_percent:+.2f}%
- **Tempo Aberto:** {hours_open:.1f} horas
- **Stop Loss:** ${stop_loss:.2f} (distancia: {distance_to_sl:.2f}%)
- **Take Profit 1:** ${take_profit_1:.2f} (distancia: {distance_to_tp1:.2f}%)
- **Take Profit 2:** ${take_profit_2:.2f}
{price_context}

## DADOS DE MERCADO ATUAIS

### Preco e Volume
- Preco: ${current_price:.2f}
- Variacao 24h: {market_data.get('market_data', {}).get('price_change_24h', 0):.2f}%
- Volume 24h: ${market_data.get('market_data', {}).get('volume_24h', 0):,.0f}

### Indicadores Tecnicos
- RSI: {indicators.get('rsi', 50):.1f}
- MACD: {indicators.get('macd_crossover', 'N/A')} / Histograma: {indicators.get('macd_histogram', 0):.4f}
- ADX: {indicators.get('adx', 25):.1f}
- ATR: ${indicators.get('atr', 0):.2f}
- Tendencia: {tech.get('trend', 'N/A')}
- Momentum: {tech.get('momentum', 'N/A')}

### Multi-Timeframe
- Confluencia Bullish: {mtf.get('bullish_count', 0)}/5
- Confluencia Bearish: {mtf.get('bearish_count', 0)}/5
- Direcao Geral: {mtf.get('confluence', 'neutral')}

### Order Flow
- Imbalance Orderbook: {order_flow.get('orderbook_imbalance', 0):.3f}
- Pressao de Compra: {'Sim' if order_flow.get('buy_pressure', False) else 'Nao'}
- CVD: {order_flow.get('cvd', 0):.2f}

### Sentimento
- Geral: {sentiment.get('sentiment', 'neutral')}
- Funding Rate: {market_data.get('market_data', {}).get('funding_rate', 0):.6f}

### Classificacao de Mercado
- Tipo Sugerido: {market_class.get('operation_type', 'N/A')}
- Volatilidade: {market_class.get('market_conditions', {}).get('volatility', 'N/A')}

---

## ACOES DISPONIVEIS

1. **HOLD** - Manter posicao sem alteracoes
   - Use quando: Tendencia continua favoravel, indicadores alinhados

2. **CLOSE** - Fechar posicao IMEDIATAMENTE
   - Use quando: Reversao iminente, risco alto, prejuizo crescente sem recuperacao

3. **CLOSE_PARTIAL** - Fechar 50% da posicao
   - Use quando: Lucro bom mas incerteza, proteger ganhos parciais

4. **MOVE_STOP_BREAKEVEN** - Mover stop loss para preco de entrada
   - Use quando: Posicao em lucro e quer eliminar risco de perda

5. **TRAILING_STOP** - Ajustar stop para travar lucros (novo_stop = preco_atual - ATR*1.5)
   - Use quando: Lucro significativo, tendencia forte, quer travar ganhos

6. **ADJUST_STOP** - Mover stop para nivel especifico
   - Use quando: Novo nivel de suporte/resistencia identificado

7. **ADJUST_TP** - Ajustar take profit para nivel especifico
   - Use quando: Resistencia/suporte mais forte identificado

---

## REGRAS IMPORTANTES

1. **SE P&L > 3%**: Considere FORTEMENTE proteger lucros (TRAILING_STOP ou CLOSE_PARTIAL)
2. **SE P&L < -3%**: Avalie se vale manter ou CLOSE para limitar perdas
3. **SE Tendencia inverteu**: Considere CLOSE ou ADJUST_STOP agressivo
4. **SE proximo ao TP1 (<1%)**: Pode manter ou CLOSE para garantir
5. **SE proximo ao SL (<1%)**: Avalie se stop esta bem posicionado ou se deve CLOSE

---

## INSTRUCOES

Analise cuidadosamente todos os dados e responda APENAS com JSON:

```json
{{
    "action": "HOLD" | "CLOSE" | "CLOSE_PARTIAL" | "MOVE_STOP_BREAKEVEN" | "TRAILING_STOP" | "ADJUST_STOP" | "ADJUST_TP",
    "confidence": <1-10>,
    "new_stop_loss": <numero se ADJUST_STOP ou TRAILING_STOP, null caso contrario>,
    "new_take_profit": <numero se ADJUST_TP, null caso contrario>,
    "reasoning": "<explicacao detalhada da decisao>",
    "urgency": "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",
    "risk_assessment": "<avaliacao do risco atual da posicao>"
}}
```

**IMPORTANTE:**
- Seja conservador com posicoes em lucro (proteja ganhos)
- Seja objetivo com posicoes em prejuizo (limite perdas)
- Considere o tipo de operacao ({operation_type}) ao sugerir acoes
- A confianca deve refletir certeza na recomendacao (>=7 para agir)
"""

        return prompt

    async def reevaluate_position(self, position: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reavalia uma posicao especifica usando DeepSeek.

        Args:
            position: Dados da posicao a reavaliar

        Returns:
            Resultado da reavaliacao com acao recomendada
        """
        position_key = position.get("position_key", "unknown")
        symbol = position.get("symbol", "UNKNOWN")

        # VERIFICAR TP1 ANTES DE GASTAR TOKENS DO DEEPSEEK (se configurado)
        # Se reevaluation_require_tp1_hit = True, só reavalia após TP1 atingido
        if settings.reevaluation_require_tp1_hit:
            tp1_hit = await self._check_tp1_hit(position)
            
            if not tp1_hit:
                logger.info(f"[REEVALUATOR] {position_key}: TP1 nao atingido - pulando reavaliacao (economia de tokens)")
                return {
                    "position_key": position_key,
                    "action": "HOLD",
                    "confidence": 5,
                    "reasoning": "TP1 ainda nao foi atingido - aguardando",
                    "tp1_pending": True,
                    "timestamp": datetime.now().isoformat()
                }

        logger.info(f"[REEVALUATOR] Iniciando reavaliacao de {position_key} (TP1 atingido)...")

        try:
            # 1. Coletar dados atuais do mercado
            market_data = await self.get_current_market_data(symbol)

            if "error" in market_data and not market_data.get("current_price"):
                return {
                    "position_key": position_key,
                    "action": "HOLD",
                    "confidence": 0,
                    "error": f"Erro ao coletar dados de mercado: {market_data.get('error')}",
                    "timestamp": datetime.now().isoformat()
                }

            # 2. Criar prompt de reavaliacao
            prompt = self._create_reevaluation_prompt(position, market_data)

            # 3. Chamar DeepSeek
            result = await self._call_deepseek_reevaluation(prompt, position_key)

            # 4. Adicionar contexto ao resultado
            result["position_key"] = position_key
            result["symbol"] = symbol
            result["entry_price"] = position.get("entry_price")
            result["current_price"] = market_data.get("current_price")
            result["pnl_percent"] = self._calculate_pnl_percent(position, market_data.get("current_price", 0))
            result["timestamp"] = datetime.now().isoformat()

            # 5. Atualizar timestamp de ultima reavaliacao
            self.last_reevaluation[position_key] = datetime.now()

            # 6. Salvar no historico
            self.reevaluation_history.append(result)
            self._save_reevaluation_log(result)

            logger.info(f"[REEVALUATOR] {position_key}: Acao={result.get('action', 'N/A')} | Confianca={result.get('confidence', 0)}/10")

            return result

        except Exception as e:
            logger.exception(f"[REEVALUATOR] Erro na reavaliacao de {position_key}: {e}")
            return {
                "position_key": position_key,
                "action": "HOLD",
                "confidence": 0,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _calculate_pnl_percent(self, position: Dict[str, Any], current_price: float) -> float:
        """Calcula P&L percentual da posicao"""
        entry_price = position.get("entry_price", 0)
        signal_type = position.get("signal", "BUY")

        if entry_price <= 0 or current_price <= 0:
            return 0.0

        if signal_type == "BUY":
            return ((current_price - entry_price) / entry_price) * 100
        else:
            return ((entry_price - current_price) / entry_price) * 100

    async def _call_deepseek_reevaluation(self, prompt: str, position_key: str) -> Dict[str, Any]:
        """
        Chama DeepSeek para reavaliar a posicao.

        Args:
            prompt: Prompt formatado
            position_key: Identificador da posicao

        Returns:
            Resultado da reavaliacao parseado
        """
        import re

        try:
            from agno.models.deepseek import DeepSeek
            from agno.agent import Agent

            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                return {
                    "action": "HOLD",
                    "confidence": 0,
                    "error": "DEEPSEEK_API_KEY nao configurada",
                    "reasoning": "Sem API key, mantendo posicao"
                }

            # Criar agent para chamada
            model = DeepSeek(id="deepseek-chat", api_key=api_key, temperature=0.2, max_tokens=800)
            agent = Agent(
                model=model,
                instructions="Voce e um trader profissional especializado em gestao de posicoes. Responda apenas com JSON valido."
            )

            logger.debug(f"[REEVALUATOR] Chamando DeepSeek para {position_key}")
            response = await agent.arun(prompt)

            # Extrair conteudo da resposta
            response_content = str(response.content) if hasattr(response, 'content') else str(response)

            # Tentar extrair JSON
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(1))
                result["raw_response"] = response_content
                return result

            # Tentar parsear resposta inteira como JSON
            try:
                result = json.loads(response_content)
                result["raw_response"] = response_content
                return result
            except json.JSONDecodeError:
                pass

            # Fallback
            logger.warning(f"[REEVALUATOR] Nao foi possivel extrair JSON da resposta")
            return {
                "action": "HOLD",
                "confidence": 3,
                "reasoning": "Nao foi possivel interpretar resposta do DeepSeek",
                "raw_response": response_content
            }

        except Exception as e:
            logger.exception(f"[REEVALUATOR] Erro ao chamar DeepSeek: {e}")
            return {
                "action": "HOLD",
                "confidence": 0,
                "error": str(e),
                "reasoning": "Erro na chamada ao DeepSeek"
            }

    def _save_reevaluation_log(self, result: Dict[str, Any]):
        """Salva log de reavaliacao"""
        try:
            date_str = datetime.now().strftime("%Y%m%d")
            log_file = Path(f"reevaluation_logs/reevaluation_{date_str}.json")

            # Carregar logs existentes
            logs = []
            if log_file.exists():
                with open(log_file, "r", encoding="utf-8") as f:
                    logs = json.load(f)

            # Adicionar novo log
            logs.append(result)

            # Salvar
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(logs, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.warning(f"[REEVALUATOR] Erro ao salvar log: {e}")

    async def execute_reevaluation_action(
        self,
        position: Dict[str, Any],
        reevaluation: Dict[str, Any],
        auto_execute: bool = False
    ) -> Dict[str, Any]:
        """
        Executa a acao recomendada pela reavaliacao.

        Args:
            position: Dados da posicao
            reevaluation: Resultado da reavaliacao
            auto_execute: Se True, executa automaticamente. Se False, apenas sugere.

        Returns:
            Resultado da execucao
        """
        action = reevaluation.get("action", "HOLD")
        confidence = reevaluation.get("confidence", 0)
        position_key = position.get("position_key", "unknown")

        # Verificar confianca minima para executar
        if confidence < settings.reevaluation_min_confidence:
            logger.info(f"[REEVALUATOR] {position_key}: Confianca {confidence}/10 abaixo do minimo ({settings.reevaluation_min_confidence}). Mantendo posicao.")
            return {
                "executed": False,
                "reason": f"Confianca {confidence}/10 abaixo do minimo",
                "action": action,
                "position_key": position_key
            }

        if not auto_execute:
            logger.info(f"[REEVALUATOR] {position_key}: Acao {action} sugerida (auto_execute=False)")
            return {
                "executed": False,
                "reason": "Modo sugestao - acao nao executada automaticamente",
                "action": action,
                "suggested": True,
                "position_key": position_key,
                "details": reevaluation
            }

        # Executar acao baseado no tipo
        try:
            if action == "HOLD":
                return {"executed": True, "action": "HOLD", "message": "Posicao mantida"}

            elif action == "CLOSE":
                return await self._execute_close(position)

            elif action == "CLOSE_PARTIAL":
                return await self._execute_close_partial(position, 0.5)

            elif action in ["MOVE_STOP_BREAKEVEN", "TRAILING_STOP", "ADJUST_STOP"]:
                # OPCIONAL: So ajustar stop se TP1 ja foi atingido (se configurado)
                if settings.reevaluation_require_tp1_hit:
                    tp1_hit = await self._check_tp1_hit(position)
                    
                    if not tp1_hit:
                        logger.info(f"[REEVALUATOR] {position_key}: Ajuste de stop BLOQUEADO - TP1 ainda nao foi atingido")
                        return {
                            "executed": False,
                            "reason": "TP1 ainda nao foi atingido - ajuste de stop so permitido apos TP1",
                            "action": action,
                            "position_key": position_key
                        }
                
                # Pode ajustar stop
                if action == "MOVE_STOP_BREAKEVEN":
                    return await self._execute_adjust_stop(position, position.get("entry_price"))
                elif action == "TRAILING_STOP":
                    new_stop = reevaluation.get("new_stop_loss")
                    if new_stop:
                        return await self._execute_adjust_stop(position, new_stop)
                    else:
                        return await self._execute_trailing_stop(position)
                else:  # ADJUST_STOP
                    new_stop = reevaluation.get("new_stop_loss")
                    if new_stop:
                        return await self._execute_adjust_stop(position, new_stop)
                    return {"executed": False, "error": "new_stop_loss nao especificado"}

            elif action == "ADJUST_TP":
                new_tp = reevaluation.get("new_take_profit")
                if new_tp:
                    return await self._execute_adjust_tp(position, new_tp)
                return {"executed": False, "error": "new_take_profit nao especificado"}

            else:
                return {"executed": False, "error": f"Acao desconhecida: {action}"}

        except Exception as e:
            logger.exception(f"[REEVALUATOR] Erro ao executar acao {action}: {e}")
            return {"executed": False, "error": str(e)}

    async def _execute_close(self, position: Dict[str, Any]) -> Dict[str, Any]:
        """Executa fechamento completo da posicao"""
        from src.trading.paper_trading import real_paper_trading

        position_key = position.get("position_key", "unknown")
        symbol = position.get("symbol")

        try:
            # Obter preco atual
            current_price = await self._get_current_price(symbol)

            if settings.trading_mode == "paper":
                # Fechar no paper trading
                result = await real_paper_trading.close_position_manual(position_key, current_price)
                return {
                    "executed": True,
                    "action": "CLOSE",
                    "position_key": position_key,
                    "close_price": current_price,
                    "result": result
                }
            else:
                # Fechar na Binance real
                from src.exchange.executor import BinanceFuturesExecutor

                async with BinanceFuturesExecutor() as executor:
                    side = "SELL" if position.get("signal") == "BUY" else "BUY"
                    result = await executor.place_market_order(
                        symbol=symbol,
                        side=side,
                        quantity=position.get("position_size"),
                        reduce_only=True
                    )
                    return {
                        "executed": True,
                        "action": "CLOSE",
                        "position_key": position_key,
                        "close_price": current_price,
                        "result": result
                    }

        except Exception as e:
            logger.exception(f"[REEVALUATOR] Erro ao fechar posicao: {e}")
            return {"executed": False, "error": str(e)}

    async def _execute_close_partial(self, position: Dict[str, Any], percent: float) -> Dict[str, Any]:
        """Executa fechamento parcial da posicao"""
        # Para paper trading, usamos o sistema existente de fechamento parcial
        # Para real, enviamos ordem proporcional

        position_key = position.get("position_key", "unknown")
        symbol = position.get("symbol")

        try:
            current_price = await self._get_current_price(symbol)
            close_size = position.get("position_size", 0) * percent

            if settings.trading_mode == "paper":
                from src.trading.paper_trading import real_paper_trading
                await real_paper_trading._close_position_partial(
                    position_key, current_price, "REEVALUATION_PARTIAL", percent
                )
                return {
                    "executed": True,
                    "action": "CLOSE_PARTIAL",
                    "percent_closed": percent * 100,
                    "position_key": position_key
                }
            else:
                from src.exchange.executor import BinanceFuturesExecutor

                async with BinanceFuturesExecutor() as executor:
                    side = "SELL" if position.get("signal") == "BUY" else "BUY"
                    result = await executor.place_market_order(
                        symbol=symbol,
                        side=side,
                        quantity=close_size,
                        reduce_only=True
                    )
                    return {
                        "executed": True,
                        "action": "CLOSE_PARTIAL",
                        "percent_closed": percent * 100,
                        "result": result
                    }

        except Exception as e:
            logger.exception(f"[REEVALUATOR] Erro ao fechar parcialmente: {e}")
            return {"executed": False, "error": str(e)}

    async def _execute_adjust_stop(self, position: Dict[str, Any], new_stop: float) -> Dict[str, Any]:
        """Ajusta stop loss da posicao"""
        position_key = position.get("position_key", "unknown")

        try:
            if settings.trading_mode == "paper":
                # Atualizar no state.json
                state_file = Path("portfolio/state.json")
                with open(state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)

                if position_key in state.get("positions", {}):
                    old_stop = state["positions"][position_key].get("stop_loss")
                    state["positions"][position_key]["stop_loss"] = new_stop
                    state["positions"][position_key]["stop_adjusted_at"] = datetime.now().isoformat()
                    state["positions"][position_key]["stop_adjusted_reason"] = "REEVALUATION"

                    with open(state_file, "w", encoding="utf-8") as f:
                        json.dump(state, f, indent=2)

                    logger.info(f"[REEVALUATOR] {position_key}: Stop ajustado de ${old_stop:.2f} para ${new_stop:.2f}")
                    return {
                        "executed": True,
                        "action": "ADJUST_STOP",
                        "old_stop": old_stop,
                        "new_stop": new_stop,
                        "position_key": position_key
                    }
            else:
                # Ajustar na Binance (cancelar SL antigo e criar novo)
                from src.exchange.executor import BinanceFuturesExecutor

                symbol = position.get("symbol")
                async with BinanceFuturesExecutor() as executor:
                    # Cancelar apenas ordens de Stop Loss existentes (preserva Take Profits)
                    cancel_result = await executor.cancel_stop_loss_orders(symbol)
                    logger.info(f"[REEVALUATOR] {position_key}: Canceladas {cancel_result.get('count', 0)} ordens de SL antigas")

                    # Criar novo SL
                    side = "SELL" if position.get("signal") == "BUY" else "BUY"
                    sl_result = await executor.place_stop_loss(
                        symbol=symbol,
                        side=side,
                        quantity=position.get("position_size"),
                        stop_price=new_stop
                    )

                    logger.info(f"[REEVALUATOR] {position_key}: Novo SL criado em ${new_stop:.2f}")
                    
                    return {
                        "executed": True,
                        "action": "ADJUST_STOP",
                        "new_stop": new_stop,
                        "position_key": position_key,
                        "sl_order": sl_result
                    }

        except Exception as e:
            logger.exception(f"[REEVALUATOR] Erro ao ajustar stop: {e}")
            return {"executed": False, "error": str(e)}

        return {"executed": False, "error": "Posicao nao encontrada"}

    async def _check_tp1_hit(self, position: Dict[str, Any]) -> bool:
        """
        Verifica se o TP1 (Take Profit 1) ja foi atingido.
        
        LOGICA PRINCIPAL: O sistema sempre cria 2 ordens de Take Profit (TP1 e TP2).
        Se existe apenas 1 ordem de TP aberta, significa que TP1 foi executado.
        Se existem 0 ordens de TP, ambos foram executados ou cancelados.
        
        Args:
            position: Dados da posicao
            
        Returns:
            True se TP1 foi atingido, False caso contrario
        """
        position_key = position.get("position_key", "unknown")
        symbol = position.get("symbol")
        
        try:
            # METODO PRINCIPAL: Contar ordens de Take Profit abertas
            # Sistema sempre cria 2 TPs. Se tem apenas 1, TP1 foi batido.
            from src.exchange.executor import BinanceFuturesExecutor
            
            async with BinanceFuturesExecutor() as executor:
                # Buscar ordens abertas do simbolo
                params = {"symbol": symbol}
                open_orders = await executor._request("GET", "/fapi/v1/openOrders", params, signed=True)
                
                if isinstance(open_orders, list):
                    # Contar apenas ordens de Take Profit
                    tp_orders = [o for o in open_orders if o.get("type") == "TAKE_PROFIT_MARKET"]
                    tp_count = len(tp_orders)
                    
                    if tp_count == 1:
                        # Exatamente 1 TP = TP1 foi batido, TP2 ainda aberto
                        logger.info(f"[REEVALUATOR] {position_key}: TP1 ATINGIDO - apenas 1 ordem de TP aberta")
                        return True
                    elif tp_count == 0:
                        # 0 TPs = posicao fechada ou problema, nao ajusta nada
                        logger.debug(f"[REEVALUATOR] {position_key}: Nenhuma ordem de TP aberta - posicao pode estar fechada")
                        return False
                    else:
                        # 2 TPs ainda abertos = TP1 nao foi batido
                        logger.debug(f"[REEVALUATOR] {position_key}: TP1 pendente - {tp_count} ordens de TP abertas")
                        return False
            
            # Fallback: Verificar flag de TP1 atingido (se existir)
            if position.get("tp1_hit") or position.get("tp1_executed"):
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"[REEVALUATOR] Erro ao verificar TP1 para {position_key}: {e}")
            # Em caso de erro, assume que TP1 nao foi atingido (mais seguro)
            return False

    async def _execute_trailing_stop(self, position: Dict[str, Any]) -> Dict[str, Any]:
        """Executa trailing stop baseado em ATR"""
        symbol = position.get("symbol")

        try:
            # Obter ATR atual
            from src.analysis.agno_tools import analyze_technical_indicators
            tech = await analyze_technical_indicators(symbol)
            atr = tech.get("indicators", {}).get("atr", 0)

            current_price = await self._get_current_price(symbol)

            if atr > 0:
                # Trailing stop = preco atual - 1.5 * ATR (para BUY)
                if position.get("signal") == "BUY":
                    new_stop = current_price - (atr * 1.5)
                else:
                    new_stop = current_price + (atr * 1.5)

                return await self._execute_adjust_stop(position, new_stop)
            else:
                return {"executed": False, "error": "Nao foi possivel calcular ATR"}

        except Exception as e:
            logger.exception(f"[REEVALUATOR] Erro no trailing stop: {e}")
            return {"executed": False, "error": str(e)}

    async def _execute_adjust_tp(self, position: Dict[str, Any], new_tp: float) -> Dict[str, Any]:
        """Ajusta take profit da posicao"""
        position_key = position.get("position_key", "unknown")

        try:
            if settings.trading_mode == "paper":
                state_file = Path("portfolio/state.json")
                with open(state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)

                if position_key in state.get("positions", {}):
                    old_tp = state["positions"][position_key].get("take_profit_1")
                    state["positions"][position_key]["take_profit_1"] = new_tp
                    state["positions"][position_key]["tp_adjusted_at"] = datetime.now().isoformat()

                    with open(state_file, "w", encoding="utf-8") as f:
                        json.dump(state, f, indent=2)

                    return {
                        "executed": True,
                        "action": "ADJUST_TP",
                        "old_tp": old_tp,
                        "new_tp": new_tp,
                        "position_key": position_key
                    }

            # Para Binance real, similar ao adjust_stop
            return {"executed": False, "error": "Nao implementado para modo real ainda"}

        except Exception as e:
            logger.exception(f"[REEVALUATOR] Erro ao ajustar TP: {e}")
            return {"executed": False, "error": str(e)}

    async def _get_current_price(self, symbol: str) -> float:
        """Obtem preco atual de um simbolo"""
        try:
            from src.exchange.client import BinanceClient

            async with BinanceClient() as client:
                ticker = await client.get_ticker_24hr(symbol)
                return float(ticker.get("lastPrice", 0))
        except Exception as e:
            logger.error(f"[REEVALUATOR] Erro ao obter preco de {symbol}: {e}")
            return 0.0

    async def check_and_adjust_stop_after_tp1(self, position: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Verifica se TP1 foi atingido e ajusta o stop loss usando DeepSeek.

        Detecta TP1 atingido por:
        1. Preço atual passou do TP1 (para LONG: current > TP1, para SHORT: current < TP1)
        2. Posição não teve stop ajustado ainda após TP1

        Args:
            position: Dados da posição

        Returns:
            Resultado do ajuste ou None se não aplicável
        """
        if not STOP_ADJUSTER_AVAILABLE:
            return None

        position_key = position.get("position_key", "unknown")
        symbol = position.get("symbol")
        signal_type = position.get("signal", "BUY")
        entry_price = position.get("entry_price", 0)
        stop_loss = position.get("stop_loss", 0)
        take_profit_1 = position.get("take_profit_1", 0)
        position_size = position.get("position_size", 0)

        # Verificar se já ajustou stop após TP1
        if position_key in self.tp1_stop_adjusted:
            logger.debug(f"[TP1 CHECK] {position_key}: Stop ja ajustado apos TP1")
            return None

        # Verificar se tem TP1 definido
        if not take_profit_1 or take_profit_1 == 0:
            return None

        # Obter preço atual
        try:
            current_price = await self._get_current_price(symbol)
            if current_price <= 0:
                return None
        except Exception as e:
            logger.error(f"[TP1 CHECK] Erro ao obter preco de {symbol}: {e}")
            return None

        # Verificar se TP1 foi atingido
        tp1_hit = False
        if signal_type == "BUY":
            # Para LONG: TP1 atingido se preço atual >= TP1
            if current_price >= take_profit_1:
                tp1_hit = True
                logger.info(f"[TP1 HIT] {position_key}: LONG - Preco ${current_price:.4f} >= TP1 ${take_profit_1:.4f}")
        else:
            # Para SHORT: TP1 atingido se preço atual <= TP1
            if current_price <= take_profit_1:
                tp1_hit = True
                logger.info(f"[TP1 HIT] {position_key}: SHORT - Preco ${current_price:.4f} <= TP1 ${take_profit_1:.4f}")

        if not tp1_hit:
            return None

        # TP1 foi atingido! Ajustar stop usando DeepSeek
        logger.info(f"[TP1 AJUSTE] {position_key}: Iniciando ajuste de stop apos TP1...")

        try:
            # Chamar stop_adjuster
            result = await adjust_stop_on_tp1_hit(
                symbol=symbol,
                side=signal_type,  # "BUY" para LONG, "SELL" para SHORT
                entry_price=entry_price,
                current_price=current_price,
                original_stop=stop_loss,
                tp1_price=take_profit_1,
                position_size=position_size
            )

            if result.get("success"):
                # Marcar como ajustado
                self.tp1_stop_adjusted[position_key] = datetime.now()
                logger.info(f"[TP1 AJUSTE] {position_key}: Stop ajustado de ${result.get('old_stop', 0):.4f} para ${result.get('new_stop', 0):.4f}")

                # Atualizar state.json se em paper mode
                if settings.trading_mode == "paper":
                    await self._update_stop_in_state(position_key, result.get("new_stop"))

                return result
            else:
                logger.error(f"[TP1 AJUSTE] {position_key}: Falha no ajuste - {result.get('error', 'Unknown')}")
                return result

        except Exception as e:
            logger.exception(f"[TP1 AJUSTE] Erro ao ajustar stop de {position_key}: {e}")
            return {"success": False, "error": str(e)}

    async def _update_stop_in_state(self, position_key: str, new_stop: float):
        """Atualiza stop loss no state.json (paper trading)"""
        try:
            state_file = Path("portfolio/state.json")
            if not state_file.exists():
                return

            with open(state_file, "r", encoding="utf-8") as f:
                state = json.load(f)

            if position_key in state.get("positions", {}):
                state["positions"][position_key]["stop_loss"] = new_stop
                state["positions"][position_key]["stop_adjusted_at"] = datetime.now().isoformat()
                state["positions"][position_key]["stop_adjusted_reason"] = "TP1_HIT_DEEPSEEK"

                with open(state_file, "w", encoding="utf-8") as f:
                    json.dump(state, f, indent=2)

                logger.info(f"[STATE] Stop atualizado para {position_key}: ${new_stop:.4f}")

        except Exception as e:
            logger.error(f"[STATE] Erro ao atualizar stop: {e}")

    async def reevaluate_all_positions(self, auto_execute: bool = False) -> List[Dict[str, Any]]:
        """
        Reavalia todas as posicoes abertas.

        Args:
            auto_execute: Se True, executa acoes automaticamente

        Returns:
            Lista de resultados de reavaliacao
        """
        results = []

        # Buscar posicoes abertas
        positions = await self.get_open_positions()

        if not positions:
            logger.info("[REEVALUATOR] Nenhuma posicao aberta para reavaliar")
            return []

        logger.info(f"[REEVALUATOR] Iniciando reavaliacao de {len(positions)} posicoes...")

        for position in positions:
            position_key = position.get("position_key", "unknown")
            symbol = position.get("symbol", "UNKNOWN")
            
            # ============================================
            # 1. TIME-BASED EXIT (saida por tempo)
            # ============================================
            if POSITION_MANAGER_AVAILABLE:
                try:
                    time_action = position_manager.check_time_based_exit(position)
                    if time_action:
                        logger.warning(f"[TIME EXIT] {position_key}: Acao recomendada = {time_action}")
                        
                        if auto_execute and time_action == "BREAKEVEN":
                            # Mover para breakeven
                            be_result = await self._execute_adjust_stop(position, position.get("entry_price"))
                            results.append({
                                "position_key": position_key,
                                "action": "TIME_EXIT_BREAKEVEN",
                                "executed": be_result.get("executed", False),
                                "pnl_percent": self._calculate_pnl_percent(position, await self._get_current_price(symbol))
                            })
                            continue
                        elif auto_execute and time_action == "CLOSE":
                            close_result = await self._execute_close(position)
                            results.append({
                                "position_key": position_key,
                                "action": "TIME_EXIT_CLOSE",
                                "executed": close_result.get("executed", False),
                                "pnl_percent": self._calculate_pnl_percent(position, await self._get_current_price(symbol))
                            })
                            continue
                except Exception as e:
                    logger.debug(f"[TIME EXIT] Erro: {e}")
            
            # ============================================
            # 2. TRAILING STOP DINAMICO (baseado em ATR)
            # ============================================
            if POSITION_MANAGER_AVAILABLE and settings.trailing_stop_enabled:
                try:
                    current_price = await self._get_current_price(symbol)
                    
                    # Obter ATR atual
                    from src.analysis.agno_tools import analyze_technical_indicators
                    tech = await analyze_technical_indicators(symbol)
                    atr = tech.get("indicators", {}).get("atr", 0)
                    
                    if atr > 0 and current_price > 0:
                        new_trailing = await position_manager.calculate_trailing_stop(
                            position, current_price, atr
                        )
                        
                        if new_trailing:
                            logger.info(f"[TRAILING] {position_key}: Movendo stop para ${new_trailing:.2f}")
                            
                            if auto_execute:
                                trailing_result = await self._execute_adjust_stop(position, new_trailing)
                                results.append({
                                    "position_key": position_key,
                                    "action": "TRAILING_STOP_DYNAMIC",
                                    "new_stop": new_trailing,
                                    "executed": trailing_result.get("executed", False),
                                    "pnl_percent": self._calculate_pnl_percent(position, current_price)
                                })
                                # Nao precisa reavaliar via DeepSeek se trailing foi executado
                                continue
                except Exception as e:
                    logger.debug(f"[TRAILING] Erro ao calcular trailing: {e}")
            
            # ============================================
            # 3. VERIFICAR TP1 E AJUSTAR STOP
            # ============================================
            if STOP_ADJUSTER_AVAILABLE:
                try:
                    tp1_result = await self.check_and_adjust_stop_after_tp1(position)
                    if tp1_result:
                        # Adicionar resultado do ajuste de TP1 aos resultados
                        tp1_result["position_key"] = position_key
                        tp1_result["action"] = "STOP_ADJUSTED_TP1"
                        tp1_result["pnl_percent"] = self._calculate_pnl_percent(
                            position,
                            await self._get_current_price(symbol)
                        )
                        results.append(tp1_result)
                        logger.info(f"[TP1] Stop ajustado para {position_key}")
                except Exception as e:
                    logger.error(f"[TP1] Erro ao verificar TP1 para {position_key}: {e}")

            # ============================================
            # 4. REAVALIACAO VIA DEEPSEEK (normal)
            # ============================================
            # Verificar se deve reavaliar (reavaliacao normal via DeepSeek)
            if not self.should_reevaluate(position):
                continue

            # Reavaliar
            reevaluation = await self.reevaluate_position(position)

            # Executar acao se aplicavel
            if reevaluation.get("action") != "HOLD":
                execution = await self.execute_reevaluation_action(
                    position, reevaluation, auto_execute
                )
                reevaluation["execution"] = execution

            results.append(reevaluation)

            # Pequena pausa entre reavaliacoes para evitar rate limiting
            await asyncio.sleep(2)

        logger.info(f"[REEVALUATOR] Reavaliacao completa: {len(results)} posicoes avaliadas")
        return results


# Instancia global do reevaluator
signal_reevaluator = SignalReevaluator()
