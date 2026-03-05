"""
Trading Agent usando AGNO para orquestração
"""
from agno.agent import Agent
from agno.models.deepseek import DeepSeek
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from pathlib import Path
import os
import re
from dotenv import load_dotenv

# CORREÇÃO: Importar logger
from src.core.logger import get_logger

# Carregar variáveis de ambiente
load_dotenv()

# Importar ferramentas dos módulos corretos
from src.analysis.market_data import get_market_data
from src.analysis.indicators import analyze_technical_indicators
from src.analysis.sentiment import analyze_market_sentiment
from src.analysis.multi_timeframe import analyze_multiple_timeframes
from src.analysis.order_flow import analyze_order_flow
from src.prompts.deepseek_prompt import get_deepseek_analysis, execute_paper_trade, backtest_strategy
from src.trading.risk_manager import validate_risk_and_position

# CORREÇÃO: Criar instância do logger
logger = get_logger(__name__)

class AgnoTradingAgent:
    """
    Agent de trading que usa AGNO para orquestrar análises
    """
    
    def __init__(self, paper_trading: bool = True):
        """
        Inicializa o agent de trading.
        
        Args:
            paper_trading: Se True, apenas simula trades
        """
        self.paper_trading = paper_trading
        
        # Carregar validador ML (se disponível)
        self.ml_validator = None
        self._load_ml_validator()
        
        # Obter API key - OBRIGATÓRIA
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError(
                "ERRO CRÍTICO: DEEPSEEK_API_KEY não encontrada. "
                "Configure a variável de ambiente DEEPSEEK_API_KEY com sua chave da API. "
                "Obtenha sua chave em: https://platform.deepseek.com/"
            )
        
        # Aplicar decorator @tool nas ferramentas
        from agno.tools import tool
        
        # Configurar o Agent AGNO com otimizações de velocidade
        # CORREÇÃO: Remover get_deepseek_analysis das tools para evitar chamada duplicada
        # O DeepSeek já é chamado diretamente antes do AGNO processar
        self.agent = Agent(
            model=DeepSeek(id="deepseek-chat", api_key=api_key, temperature=0.3, max_tokens=1000),
            tools=[
                tool(get_market_data),
                tool(analyze_technical_indicators),
                tool(analyze_market_sentiment),
                # REMOVIDO: tool(get_deepseek_analysis) - evita chamada duplicada
                tool(validate_risk_and_position),
                tool(execute_paper_trade),
                tool(analyze_multiple_timeframes),
                tool(analyze_order_flow),
                tool(backtest_strategy)
            ],
            instructions=self._get_instructions()
        )
        
        # Criar pastas necessárias
        Path("signals").mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
        Path("paper_trades").mkdir(exist_ok=True)
        
        # Criar estrutura de diretórios para respostas do DeepSeek (ano/mês/dia)
        today = datetime.now()
        deepseek_logs_dir = Path(f"deepseek_logs/{today.year}/{today.month:02d}/{today.day:02d}")
        deepseek_logs_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_ml_validator(self):
        """Carrega o validador ML se disponível"""
        try:
            from src.ml.simple_validator import SimpleSignalValidator
            validator = SimpleSignalValidator()
            validator.load_models()
            self.ml_validator = validator
            logger.info("[ML] Validador ML carregado com sucesso")
            print("[ML] Validador de sinais ML carregado - confluencia habilitada!")
        except Exception as e:
            logger.warning(f"[ML] Validador ML nao disponivel: {e}")
            print(f"[ML] Validador ML nao disponivel (execute simple_signal_validator.py para treinar)")
            self.ml_validator = None
    
    def _validate_with_ml_model(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valida um sinal usando o modelo ML treinado.
        
        Args:
            signal: Sinal gerado pelo DeepSeek/AGNO
            
        Returns:
            Dict com resultado da validacao ML
        """
        # Se validador nao disponivel, permitir sinal
        if self.ml_validator is None:
            return {
                "skip_signal": False,
                "has_confluence": False,
                "probability": 0.5,
                "reason": "Validador ML nao disponivel"
            }
        
        try:
            # Extrair features do sinal
            features = {
                'rsi': signal.get('rsi', signal.get('indicators', {}).get('rsi', 50)),
                'macd_histogram': signal.get('macd_histogram', signal.get('indicators', {}).get('macd_histogram', 0)),
                'adx': signal.get('adx', signal.get('indicators', {}).get('adx', 25)),
                'atr': signal.get('atr', signal.get('indicators', {}).get('atr', 0)),
                'bb_position': signal.get('bb_position', signal.get('indicators', {}).get('bb_position', 0.5)),
                'cvd': signal.get('cvd', signal.get('order_flow', {}).get('cvd', 0)),
                'orderbook_imbalance': signal.get('orderbook_imbalance', signal.get('order_flow', {}).get('orderbook_imbalance', 0.5)),
                'bullish_tf_count': signal.get('bullish_tf_count', 0),
                'bearish_tf_count': signal.get('bearish_tf_count', 0),
                'confidence': signal.get('confidence', 5),
                'trend_encoded': self._encode_trend(signal.get('trend', 'neutral')),
                'sentiment_encoded': self._encode_sentiment(signal.get('sentiment', 'neutral')),
                'signal_encoded': 1 if signal.get('signal') == 'BUY' else 0,
                'risk_distance_pct': self._calc_risk_distance(signal),
                'reward_distance_pct': self._calc_reward_distance(signal),
                'risk_reward_ratio': self._calc_risk_reward(signal),
            }
            
            # Fazer predicao
            result = self.ml_validator.predict_signal(features)
            
            probability = result.get('probability_success', 0.5)
            prediction = result.get('prediction', 0)
            
            # Configuracao: threshold de probabilidade para aceitar sinal
            from src.core.config import settings
            ml_threshold = getattr(settings, 'ml_validation_threshold', 0.65)
            ml_required = getattr(settings, 'ml_validation_required', False)
            ml_enabled = getattr(settings, 'ml_validation_enabled', True)

            # CORRIGIDO: Lógica de confluência mais clara
            # has_confluence = True se modelo prevê sucesso E probabilidade > threshold
            has_confluence = prediction == 1 and probability >= ml_threshold

            # CORRIGIDO: Lógica de skip mais clara
            # skip_signal = True APENAS se:
            # - ml_required=True (ML é obrigatório) E não tem confluência, OU
            # - ml_enabled=True E não tem confluência E probabilidade é muito baixa (< 0.4)
            if ml_required:
                # Se ML é obrigatório, só executa com confluência
                skip_signal = not has_confluence
            elif ml_enabled:
                # Se ML está habilitado mas não obrigatório, só bloqueia se prob muito baixa
                skip_signal = not has_confluence and probability < 0.4
            else:
                # Se ML desabilitado, não bloqueia
                skip_signal = False

            return {
                "skip_signal": skip_signal,
                "has_confluence": has_confluence,
                "probability": probability,
                "prediction": prediction,
                "reason": f"ML prob: {probability:.1%}, threshold: {ml_threshold:.1%}"
            }
            
        except Exception as e:
            logger.warning(f"[ML] Erro na validacao ML: {e}")
            return {
                "skip_signal": False,
                "has_confluence": False,
                "probability": 0.5,
                "reason": f"Erro: {e}"
            }
    
    def _encode_trend(self, trend: str) -> int:
        """Codifica tendencia para valor numerico"""
        trend_map = {'strong_bullish': 2, 'bullish': 1, 'neutral': 0, 'bearish': -1, 'strong_bearish': -2}
        return trend_map.get(trend.lower() if trend else 'neutral', 0)
    
    def _encode_sentiment(self, sentiment: str) -> int:
        """Codifica sentimento para valor numerico"""
        sentiment_map = {'bullish': 1, 'neutral': 0, 'bearish': -1}
        return sentiment_map.get(sentiment.lower() if sentiment else 'neutral', 0)
    
    def _calc_risk_distance(self, signal: Dict) -> float:
        """Calcula distancia do stop em %"""
        entry = signal.get('entry_price', 0)
        stop = signal.get('stop_loss', 0)
        if entry > 0 and stop > 0:
            return abs(entry - stop) / entry * 100
        return 2.0  # Default 2%
    
    def _calc_reward_distance(self, signal: Dict) -> float:
        """Calcula distancia do TP1 em %"""
        entry = signal.get('entry_price', 0)
        tp1 = signal.get('take_profit_1', signal.get('take_profit', 0))
        if entry > 0 and tp1 > 0:
            return abs(tp1 - entry) / entry * 100
        return 2.0  # Default 2%
    
    def _calc_risk_reward(self, signal: Dict) -> float:
        """Calcula risk/reward ratio"""
        risk = self._calc_risk_distance(signal)
        reward = self._calc_reward_distance(signal)
        if risk > 0:
            return reward / risk
        return 1.0

    # REFATORADO: Constante de preços padrão para evitar duplicação
    DEFAULT_PRICES = {
        "BTCUSDT": 90000, "ETHUSDT": 3000, "SOLUSDT": 140,
        "BNBUSDT": 600, "ADAUSDT": 0.5, "XRPUSDT": 2.0,
        "DOGEUSDT": 0.15, "AVAXUSDT": 40, "DOTUSDT": 7,
        "LINKUSDT": 20, "PAXGUSDT": 2700
    }

    def _extract_price_from_text(self, text: str, min_price: float = 0.01, max_price: float = 1000000) -> Optional[float]:
        """
        REFATORADO: Função helper para extrair preço de texto.
        Elimina código duplicado de extração de preço.

        Args:
            text: Texto para buscar preço
            min_price: Preço mínimo válido
            max_price: Preço máximo válido

        Returns:
            Preço extraído ou None se não encontrado
        """
        if not text:
            return None

        price_patterns = [
            r"\$([0-9,]+\.?[0-9]+)",  # $90,563.50
            r"([0-9]{1,3}(?:[,.][0-9]{1,2})?)\s*(?:USD|USDT)",  # 90,563.50 USD
            r"preço[^0-9]*([0-9,]+\.?[0-9]+)",  # preço 90,563.50
            r"preco[^0-9]*([0-9,]+\.?[0-9]+)",  # preco 90,563.50
            r"entrada[^0-9]*\$?([0-9,]+\.?[0-9]*)",
            r"entry[^0-9]*\$?([0-9,]+\.?[0-9]*)",
            r"entry_price[^0-9]*[:=]\s*\$?([0-9,]+\.?[0-9]*)",
            r"current[^0-9]*price[^0-9]*\$?([0-9,]+\.?[0-9]*)"
        ]

        for pattern in price_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    price_str = match.group(1).replace(",", "")
                    price = float(price_str)
                    if min_price <= price <= max_price:
                        return price
                except ValueError:
                    continue

        return None

    def _get_default_price(self, symbol: str) -> float:
        """Retorna preço padrão para um símbolo"""
        return self.DEFAULT_PRICES.get(symbol, 100)

    def _calculate_stop_loss(self, entry_price: float, signal_type: str, distance_pct: float = 0.02) -> float:
        """
        REFATORADO: Calcula stop loss baseado no tipo de sinal.

        Args:
            entry_price: Preço de entrada
            signal_type: BUY ou SELL
            distance_pct: Distância percentual do stop (default 2%)

        Returns:
            Preço do stop loss
        """
        if signal_type == "BUY":
            return entry_price * (1 - distance_pct)
        else:  # SELL
            return entry_price * (1 + distance_pct)
    
    def _get_instructions(self) -> str:
        """Retorna as instruções para o agent"""
        return """
        Você é um trader profissional especializado em análise técnica e gestão de risco.
        
        PROCESSO DE ANÁLISE:
        1. Colete dados de mercado usando get_market_data()
        2. Analise indicadores técnicos com analyze_technical_indicators()
        3. Capture sentimento com analyze_market_sentiment()
        4. Analise multi-timeframe com analyze_multiple_timeframes()
        5. Analise order flow com analyze_order_flow()
        6. Baseado nos dados coletados, tome sua própria decisão de trading
        7. Valide risco com validate_risk_and_position()
        8. Execute paper trade se apropriado com execute_paper_trade()
        9. Para backtesting, use backtest_strategy() com datas específicas
        
        IMPORTANTE: Você está gerando um sinal AGNO independente. Analise os dados coletados
        e tome sua própria decisão, não dependa de análises externas.
        
        REGRAS DE TRADING:
        - SEMPRE forneça um sinal: BUY ou SELL (seja decisivo)
        - NÃO use HOLD ou NÃO OPERAR
        - Para BUY/SELL, defina OBRIGATORIAMENTE:
          * Entrada: preço específico
          * Stop Loss: preço específico
          * Take Profit 1: preço específico
          * Take Profit 2: preço específico
          * Confiança: 1-10
        
        GESTÃO DE RISCO:
        - Confiança mínima 7/10 para executar (sinais com confiança < 7 serão rejeitados)
        - Escala de confiança: 1-10 (sempre use esta escala)
        - Se confiança < 7, retorne "NO_SIGNAL" ao invés de um sinal fraco
        - Respeite circuit breakers automáticos
        - Analise estrutura de mercado (suporte/resistência)
        - Considere múltiplos timeframes
        
        FORMATO DE RESPOSTA OBRIGATÓRIO:
        Sempre termine sua análise com um bloco JSON estruturado:
        
        ```json
        {
            "signal": "BUY" ou "SELL" ou "NO_SIGNAL",
            "entry_price": 95000.00,
            "stop_loss": 93000.00,
            "take_profit_1": 97000.00,
            "take_profit_2": 99000.00,
            "confidence": 7 (escala 1-10, mínimo 7 para executar)
        }
        ```
        
        Seja detalhado na análise mas objetivo na decisão.
        """
    
    async def analyze(self, symbol: str = "BTCUSDT") -> Dict[str, Any]:
        """
        Executa análise completa usando o AGNO Agent.
        CORRIGIDO: Verifica posição existente antes de analisar.
        
        Args:
            symbol: Símbolo para analisar
            
        Returns:
            Sinal de trading estruturado
        """
        print(f"\n[AGNO] AGNO Agent iniciando analise de {symbol}")
        print("="*60)
        
        # Verificar posição e limpar ordens órfãs antes de analisar
        from src.core.config import settings
        if settings.trading_mode == "real":
            try:
                from src.exchange.executor import BinanceFuturesExecutor
                executor = BinanceFuturesExecutor()
                
                # 1. Verificar posição existente
                existing_position = await executor.get_position(symbol)
                has_position = existing_position and "position_amt" in existing_position
                
                # 2. Se NÃO tem posição, verificar se tem ordens órfãs para cancelar
                if not has_position:
                    open_orders = await executor.get_open_orders(symbol)
                    if open_orders and len(open_orders) > 0:
                        logger.info(f"[LIMPEZA] {symbol}: {len(open_orders)} ordens orfas encontradas. Cancelando...")
                        print(f"[LIMPEZA] {symbol}: Cancelando {len(open_orders)} ordens orfas")
                        await executor.cancel_all_orders(symbol)
                
                # 3. Se TEM posição, pular análise
                if has_position:
                    side = existing_position.get("side", "UNKNOWN")
                    amt = abs(existing_position.get("position_amt", 0))
                    logger.warning(f"[BINANCE] Ja existe posicao {side} aberta para {symbol} ({amt} unidades). Pulando analise.")
                    print(f"[BINANCE] Ja existe posicao {side} aberta para {symbol}. Pulando analise.")
                    return {
                        "symbol": symbol,
                        "signal": "NO_SIGNAL",
                        "confidence": 0,
                        "reason": f"Ja existe posicao {side} aberta na Binance para {symbol}",
                        "timestamp": datetime.now().isoformat(),
                        "source": "AGNO"
                    }
            except Exception as e:
                # Se erro ao verificar, continuar com análise (log warning)
                logger.warning(f"Erro ao verificar posicao na Binance: {e}")
        
        # A verificação de posição existente no paper trading será feita em validate_risk_and_position()
        
        # CORRIGIDO: Verificar última análise (1 hora) antes de enviar para DeepSeek
        try:
            import json
            import os
            from src.core.config import settings
            
            # Verificar última análise do símbolo
            last_analysis_file = f"signals/agno_{symbol}_last_analysis.json"
            if os.path.exists(last_analysis_file):
                with open(last_analysis_file, "r", encoding='utf-8') as f:
                    last_analysis = json.load(f)
                    last_timestamp_str = last_analysis.get("timestamp")
                    if last_timestamp_str:
                        # Parse timestamp (suporta com e sem timezone)
                        try:
                            last_timestamp = datetime.fromisoformat(last_timestamp_str.replace('Z', '+00:00'))
                        except ValueError:
                            # Tentar sem timezone
                            last_timestamp = datetime.fromisoformat(last_timestamp_str)
                        
                        # Se não tem timezone, assumir local
                        if last_timestamp.tzinfo is None:
                            last_timestamp = last_timestamp.replace(tzinfo=datetime.now().astimezone().tzinfo)
                        
                        now = datetime.now(last_timestamp.tzinfo)
                        time_since_last = now - last_timestamp
                        hours_since_last = time_since_last.total_seconds() / 3600
                        min_interval = settings.min_analysis_interval_hours
                        
                        if hours_since_last < min_interval:
                            remaining_minutes = int((min_interval - hours_since_last) * 60)
                            print(f"[AVISO] Ultima analise de {symbol} foi ha {int(hours_since_last*60)} minutos. Aguardando {min_interval}h (restam {remaining_minutes} minutos).")
                            return {
                                "symbol": symbol,
                                "signal": "NO_SIGNAL",
                                "confidence": 0,
                                "reason": f"Ultima analise ha {int(hours_since_last*60)} minutos (minimo {min_interval}h)",
                                "timestamp": datetime.now().isoformat()
                            }
        except Exception as e:
            # Se houver erro, continuar com análise (não bloquear)
            logger.warning(f"Erro ao verificar ultima analise: {e}")
        
        # Prompt para o agent
        # CORRIGIDO: get_deepseek_analysis() agora retorna o sinal JSON diretamente
        # Obter dados sumarizados para o AGNO agent analisar
        # Nota: get_deepseek_analysis() agora retorna sinal direto, mas podemos chamar prepare_analysis_for_llm() diretamente
        # Por enquanto, vamos usar o prompt simples e deixar o AGNO analisar
        prompt = f"""
         Analise os dados de mercado para {symbol} e forneça um sinal de trading.
         
         NOTA: O sistema gera DOIS sinais separados para comparação:
         - Um sinal DEEPSEEK (direto) - já foi gerado automaticamente
         - Um sinal AGNO (este) - você deve analisar independentemente
         
         Processo:
         1. Use get_market_data("{symbol}") para obter dados de mercado
         2. Use analyze_technical_indicators("{symbol}") para indicadores técnicos
         3. Use analyze_market_sentiment("{symbol}") para sentimento
         4. Analise os dados e decida: BUY, SELL ou NO_SIGNAL
         5. Retorne APENAS o JSON estruturado no final:
         
         ```json
         {{
             "signal": "BUY" ou "SELL" ou "NO_SIGNAL",
             "entry_price": <número>,
             "stop_loss": <número>,
             "take_profit_1": <número>,
             "take_profit_2": <número>,
             "confidence": <1-10>,
             "reasoning": "<justificativa>"
         }}
         ```
         
         IMPORTANTE: 
         - Você está gerando o sinal AGNO (independente do sinal DeepSeek)
         - O sistema executará ambos os sinais separadamente se ambos forem válidos
         - Forneça sua própria análise baseada nos dados coletados
         """
        
        try:
            # Verificar configuração de sinais
            from src.core.config import settings
            
            # OTIMIZAÇÃO: Só gerar sinal DeepSeek se estiver habilitado
            # Isso evita chamadas duplicadas à API quando só queremos sinais AGNO
            deepseek_signal = None
            
            if settings.accept_deepseek_signals:
                # 1. SINAL DEEPSEEK DIRETO (só se habilitado)
                logger.info(f"[DEEPSEEK] Gerando sinal DeepSeek para {symbol}...")
                deepseek_result = await get_deepseek_analysis(symbol)
                
                if isinstance(deepseek_result, dict) and "signal" in deepseek_result:
                    # DeepSeek já retornou sinal JSON processado
                    logger.info(f"[SINAL DEEPSEEK] Sinal direto: {deepseek_result.get('signal', 'N/A')}")
                    deepseek_signal = {
                        "symbol": symbol,
                        "source": "DEEPSEEK",
                        "timestamp": datetime.now().isoformat(),
                        "signal": deepseek_result.get("signal", "NO_SIGNAL"),
                        "entry_price": deepseek_result.get("entry_price"),
                        "stop_loss": deepseek_result.get("stop_loss"),
                        "take_profit_1": deepseek_result.get("take_profit_1"),
                        "take_profit_2": deepseek_result.get("take_profit_2"),
                        "confidence": deepseek_result.get("confidence", 5),
                        "reasoning": deepseek_result.get("reasoning", ""),
                        "raw_response": deepseek_result.get("raw_response", "")
                    }
                    
                    # Salvar resposta bruta do DeepSeek para auditoria
                    self._save_deepseek_response(
                        symbol, 
                        deepseek_result.get("deepseek_prompt", ""), 
                        deepseek_result.get("raw_response", ""),
                        deepseek_result.get("analysis_data", {})
                    )
                    
                    # Salvar sinal DeepSeek
                    self._save_signal(deepseek_signal)
                    
                    # Executar se for sinal válido
                    if deepseek_signal.get("signal") in ["BUY", "SELL"]:
                        validation = validate_risk_and_position(deepseek_signal, symbol)
                        if validation.get("can_execute"):
                            logger.info(f"[DEEPSEEK] Executando sinal {deepseek_signal.get('signal')} para {symbol}")
                            position_size = validation.get("recommended_position_size", validation.get("position_size"))
                            execution_result = execute_paper_trade(deepseek_signal, position_size)
                            if execution_result.get("success"):
                                logger.info(f"[DEEPSEEK] Trade executado: {execution_result.get('message', '')}")
                            else:
                                logger.warning(f"[DEEPSEEK] Falha: {execution_result.get('error', '')}")
                        else:
                            logger.info(f"[DEEPSEEK] Não executado: {validation.get('reason', '')}")
            else:
                logger.info(f"[DEEPSEEK] Sinais DEEPSEEK desabilitados. Pulando analise DeepSeek direta.")
            
            # 2. SINAL AGNO PROCESSADO
            # Executar agent - ELE VAI ORQUESTRAR TUDO!
            response = await self.agent.arun(prompt)
            
            # Salvar resposta bruta do AGNO para auditoria
            self._save_deepseek_response(symbol, prompt, response, {})
            
            # Processar resposta do AGNO (agora async)
            agno_signal = await self._process_agent_response(response, symbol)
            agno_signal["source"] = "AGNO"  # Identificador da fonte
            
            # IMPORTANTE: Adicionar indicadores técnicos ao sinal para validação ML
            try:
                from src.analysis.indicators import analyze_technical_indicators
                from src.analysis.order_flow import analyze_order_flow
                from src.analysis.multi_timeframe import analyze_multiple_timeframes
                
                # Coletar indicadores técnicos
                tech_data = await analyze_technical_indicators(symbol)
                if tech_data and "indicators" in tech_data:
                    indicators = tech_data["indicators"]
                    agno_signal["rsi"] = indicators.get("rsi", 50)
                    agno_signal["macd_histogram"] = indicators.get("macd_histogram", 0)
                    agno_signal["adx"] = indicators.get("adx", 25)
                    agno_signal["atr"] = indicators.get("atr", 0)
                    agno_signal["bb_position"] = indicators.get("bb_position", 0.5)
                    agno_signal["trend"] = indicators.get("trend", "neutral")
                    agno_signal["indicators"] = indicators
                    
                # Coletar order flow
                order_flow = await analyze_order_flow(symbol)
                if order_flow and "error" not in order_flow:
                    agno_signal["cvd"] = order_flow.get("cvd", 0)
                    agno_signal["orderbook_imbalance"] = order_flow.get("orderbook_imbalance", 0.5)
                    agno_signal["order_flow"] = order_flow
                    
                # Coletar multi-timeframe
                mtf_data = await analyze_multiple_timeframes(symbol)
                if mtf_data and "error" not in mtf_data:
                    agno_signal["bullish_tf_count"] = mtf_data.get("bullish_tf_count", 0)
                    agno_signal["bearish_tf_count"] = mtf_data.get("bearish_tf_count", 0)
                    
                logger.debug(f"[ML] Indicadores adicionados ao sinal: RSI={agno_signal.get('rsi')}, ADX={agno_signal.get('adx')}")
            except Exception as e:
                logger.warning(f"[ML] Erro ao coletar indicadores para ML: {e}")
            
            # CORRIGIDO: Se não tem entry_price, obter do mercado (async)
            if agno_signal.get("signal") in ["BUY", "SELL"] and not agno_signal.get("entry_price"):
                try:
                    market_data = await get_market_data(symbol)
                    if market_data and "current_price" in market_data:
                        agno_signal["entry_price"] = market_data["current_price"]
                        # Calcular stop loss se não tiver
                        if not agno_signal.get("stop_loss"):
                            if agno_signal["signal"] == "BUY":
                                agno_signal["stop_loss"] = agno_signal["entry_price"] * 0.98
                            else:  # SELL
                                agno_signal["stop_loss"] = agno_signal["entry_price"] * 1.02
                        logger.info(f"[AGNO] Preço atual obtido: Entry=${agno_signal['entry_price']}, SL=${agno_signal.get('stop_loss')}")
                except Exception as e:
                    logger.error(f"[AGNO] Erro ao obter preço atual: {e}")
            
            # Salvar sinal AGNO
            self._save_signal(agno_signal)
            
            # FILTRO DE SINAIS: Verificar se sinais AGNO estão habilitados
            if not settings.accept_agno_signals:
                logger.info(f"[AGNO] Sinais AGNO desabilitados (accept_agno_signals=False). Sinal ignorado.")
            elif agno_signal.get("signal") in ["BUY", "SELL"]:
                # VALIDAÇÃO ML: Usar modelo treinado para validar confluência
                ml_validation = self._validate_with_ml_model(agno_signal)
                ml_prob = ml_validation.get('probability', 0)
                ml_pred = ml_validation.get('prediction', 0)
                
                # SEMPRE mostrar resultado da validação ML
                print(f"[ML] Validacao: prob={ml_prob:.1%}, predicao={'SUCESSO' if ml_pred == 1 else 'FALHA'}")
                logger.info(f"[ML] Validacao ML: prob={ml_prob:.1%}, predicao={ml_pred}")
                
                if ml_validation.get("skip_signal"):
                    logger.warning(f"[ML] Sinal REJEITADO pelo modelo ML: prob={ml_prob:.1%}, predicao={'SUCESSO' if ml_pred == 1 else 'FALHA'}")
                    print(f"[ML] ❌ Sinal {agno_signal.get('signal')} REJEITADO - Modelo ML: FALHA (prob: {ml_prob:.1%})")
                    logger.info(f"[ML] Sinal bloqueado - Sem confluencia entre DeepSeek e ML")
                else:
                    if ml_validation.get("has_confluence"):
                        logger.info(f"[ML] Confluencia confirmada! Prob sucesso: {ml_prob:.1%}")
                        print(f"[ML] ✅ CONFLUENCIA! DeepSeek + ML concordam (prob: {ml_prob:.1%})")
                    else:
                        # Isso não deveria acontecer se skip_signal está correto, mas mantém para segurança
                        logger.warning(f"[ML] ATENCAO: Executando sem confluencia (prob: {ml_prob:.1%})")
                        print(f"[ML] ⚠️ Sem confluencia (prob: {ml_prob:.1%}) - Verifique configuracao ML")
                    
                    validation = validate_risk_and_position(agno_signal, symbol)
                    if validation.get("can_execute"):
                        logger.info(f"[AGNO] Validando e executando sinal {agno_signal.get('signal')} para {symbol}")
                        position_size = validation.get("recommended_position_size", validation.get("position_size"))

                        # VERIFICAR MODO DE TRADING: paper ou real
                        if settings.trading_mode == "real":
                            # MODO REAL: Executar na Binance Futures
                            # IMPORTANTE: Passar position_size=None para que o executor calcule
                            # baseado no saldo REAL disponível na Binance
                            from src.exchange.executor import BinanceFuturesExecutor
                            executor = BinanceFuturesExecutor()
                            execution_result = await executor.execute_signal(agno_signal, position_size=None)
                            if execution_result.get("success"):
                                logger.info(f"[AGNO REAL] Trade REAL executado com sucesso: {execution_result.get('message', '')}")
                            else:
                                logger.warning(f"[AGNO REAL] Falha ao executar trade REAL: {execution_result.get('error', '')}")
                        else:
                            # MODO PAPER: Simulação
                            execution_result = execute_paper_trade(agno_signal, position_size)
                            if execution_result.get("success"):
                                logger.info(f"[AGNO PAPER] Trade PAPER executado com sucesso: {execution_result.get('message', '')}")
                            else:
                                logger.warning(f"[AGNO PAPER] Falha ao executar trade PAPER: {execution_result.get('error', '')}")
                    else:
                        logger.info(f"[AGNO] Sinal nao executado: {validation.get('reason', '')}")
                
            # Retornar o sinal AGNO como principal (para compatibilidade)
            signal = agno_signal
            
            # CORRIGIDO: Salvar timestamp da última análise
            try:
                import json
                import os
                last_analysis_file = f"signals/agno_{symbol}_last_analysis.json"
                with open(last_analysis_file, "w", encoding='utf-8') as f:
                    json.dump({
                        "symbol": symbol,
                        "timestamp": datetime.now().isoformat(),
                        "signal": signal.get("signal", "NO_SIGNAL"),
                        "confidence": signal.get("confidence", 0)
                    }, f, indent=2)
            except Exception as e:
                logger.warning(f"Erro ao salvar ultima analise: {e}")
            
            # Imprimir resumo
            self._print_summary(signal)
            
            return signal
            
        except Exception as e:
            print(f"[ERRO] Erro na analise: {e}")
            return self._create_error_signal(symbol, str(e))
    
    def _extract_balanced_json(self, text: str) -> Optional[str]:
        """
        Extrai JSON balanceado corretamente, mesmo com objetos aninhados.
        CORREÇÃO: Resolve problema de regex que para no primeiro }
        """
        # Procurar por ```json ... ```
        json_block_match = re.search(r'```json\s*(\{.*?)\s*```', text, re.DOTALL)
        if json_block_match:
            start_pos = json_block_match.start(1)
            json_start = text.find('{', start_pos)
            if json_start == -1:
                return None
        else:
            # Procurar por { sem o bloco de código
            json_start = text.find('{')
            if json_start == -1:
                return None
        
        # Balancear chaves para encontrar o final correto do JSON
        count = 0
        in_string = False
        escape_next = False
        
        for i in range(json_start, len(text)):
            char = text[i]
            
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\':
                escape_next = True
                continue
            
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            
            if not in_string:
                if char == '{':
                    count += 1
                elif char == '}':
                    count -= 1
                    if count == 0:
                        # Encontrou o final balanceado
                        return text[json_start:i+1]
        
        return None
    
    async def _process_agent_response(self, response: Any, symbol: str) -> Dict[str, Any]:
        """Processa resposta do agent em formato estruturado (CORRIGIDO: agora é async)"""
        
        # CORRIGIDO: Extrair conteúdo real do RunOutput do AGNO
        # O AGNO retorna um objeto RunOutput, o conteúdo está em response.content
        response_text = None
        if hasattr(response, 'content'):
            # RunOutput do AGNO - conteúdo direto
            response_text = str(response.content) if response.content else None
            logger.debug(f"[AGNO] Conteúdo extraído de response.content: {response_text[:200] if response_text else 'None'}...")
        elif hasattr(response, 'output'):
            response_text = str(response.output)
        elif hasattr(response, 'messages') and len(response.messages) > 0:
            # Se for uma lista de mensagens, pegar a última
            last_message = response.messages[-1]
            if hasattr(last_message, 'content'):
                response_text = str(last_message.content)
            else:
                response_text = str(last_message)
        elif isinstance(response, dict):
            # Se já for dict, pode ser sinal direto
            if "signal" in response:
                logger.info(f"[SINAL DIRETO] Usando sinal do dict: {response.get('signal', 'N/A')}")
                return {
                    "symbol": symbol,
                    "timestamp": datetime.now().isoformat(),
                    "signal": response.get("signal", "NO_SIGNAL"),
                    "entry_price": response.get("entry_price"),
                    "stop_loss": response.get("stop_loss"),
                    "take_profit_1": response.get("take_profit_1"),
                    "take_profit_2": response.get("take_profit_2"),
                    "confidence": response.get("confidence", 5),
                    "reasoning": response.get("reasoning", ""),
                    "agent_response": str(response)
                }
            else:
                response_text = str(response)
        else:
            # Fallback: tentar str() mas logar aviso
            response_text = str(response)
            logger.warning(f"[AGNO] Resposta não é RunOutput conhecido, usando str(): {type(response)}")
        
        # Extrair informações da resposta
        signal = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "agent_response": response_text[:500] if response_text else "N/A",  # Limitar tamanho
        }
        
        if not response_text:
            logger.error(f"[ERRO] Não foi possível extrair conteúdo da resposta do AGNO")
            return {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "signal": "NO_SIGNAL",
                "confidence": 0,
                "reason": "Erro ao extrair resposta do AGNO"
            }
        
        # MELHORIA: Tentar extrair JSON estruturado primeiro (mais confiável)
        # CORREÇÃO: Usar função para balancear chaves e capturar JSON aninhado corretamente
        json_text = self._extract_balanced_json(response_text)
        if json_text:
            try:
                structured = json.loads(json_text)
                logger.info(f"[JSON ESTRUTURADO] Sinal extraído via JSON: {structured.get('signal', 'N/A')}")
                # Validar campos obrigatórios
                if structured.get("signal") in ["BUY", "SELL", "NO_SIGNAL"]:
                    signal.update({
                        "signal": structured.get("signal", "NO_SIGNAL"),
                        "entry_price": structured.get("entry_price"),
                        "stop_loss": structured.get("stop_loss"),
                        "take_profit_1": structured.get("take_profit_1"),
                        "take_profit_2": structured.get("take_profit_2"),
                        "confidence": structured.get("confidence", 5)
                    })
                    # Validar se tem entrada para BUY/SELL
                    if signal["signal"] in ["BUY", "SELL"] and not signal.get("entry_price"):
                        logger.warning("[JSON] Sinal BUY/SELL sem entry_price, usando fallback regex")
                        # Continuar para extração regex
                    else:
                        return signal
            except json.JSONDecodeError as e:
                logger.warning(f"[JSON] Erro ao decodificar JSON: {e}, usando fallback regex")
        
        # CORRIGIDO: Procurar pelo sinal FINAL (não o primeiro encontrado)
        # Priorizar "SINAL FINAL:" ou "SINAL:" que aparecem no final da análise
        signal["signal"] = "NO_SIGNAL"
        
        # CRÍTICO: Procurar primeiro por "SINAL FINAL" que é o mais importante
        # O DeepSeek sempre envia "SINAL FINAL: BUY" ou "SINAL FINAL: SELL"
        final_signal_patterns = [
            r"SINAL\s+FINAL[:\s]+\*?\*?(BUY|SELL)\*?\*?",  # Prioridade máxima: "SINAL FINAL: **SELL**"
            r"SINAL\s+FINAL[:\s]+(BUY|SELL)",              # "SINAL FINAL: SELL"
            r"###\s*\*\*SINAL\s+FINAL[:\s]+\*\*(BUY|SELL)", # "### **SINAL FINAL:** SELL"
            r"##\s+SINAL\s+FINAL[:\s]+(BUY|SELL)",          # "## SINAL FINAL: SELL"
            r"RESUMO[^:]*Sinal\s+(BUY|SELL)",               # "RESUMO: Sinal SELL"
            r"Conclusão[^:]*:\s*(BUY|SELL)",                 # "Conclusão: SELL"
            r"Recomendação[^:]*:\s*(BUY|SELL)"              # "Recomendação: SELL"
        ]
        
        # Procurar do final para o início (sinal mais recente)
        for pattern in final_signal_patterns:
            matches = list(re.finditer(pattern, response_text, re.IGNORECASE | re.MULTILINE))
            if matches:
                # Pegar o ÚLTIMO match (mais recente)
                last_match = matches[-1]
                signal_type = last_match.group(1).upper()
                if signal_type in ["BUY", "SELL"]:
                    signal["signal"] = signal_type
                    logger.info(f"[SINAL EXTRAIDO] Encontrado '{signal_type}' via padrão: {pattern[:50]}")
                    break
        
        # Se não encontrou padrão específico, procurar por qualquer BUY/SELL
        # mas APENAS se não encontrou "SINAL FINAL" antes
        if signal["signal"] == "NO_SIGNAL":
            # Procurar todas as ocorrências de BUY e SELL
            buy_matches = list(re.finditer(r'\bBUY\b', response_text, re.IGNORECASE))
            sell_matches = list(re.finditer(r'\bSELL\b', response_text, re.IGNORECASE))
            
            # Pegar a última ocorrência de cada
            last_buy_pos = buy_matches[-1].start() if buy_matches else -1
            last_sell_pos = sell_matches[-1].start() if sell_matches else -1
            
            # Escolher o que aparece mais próximo do final
            if last_buy_pos > last_sell_pos and last_buy_pos >= 0:
                signal["signal"] = "BUY"
                logger.warning(f"[SINAL FALLBACK] Usando BUY (última ocorrência na posição {last_buy_pos})")
            elif last_sell_pos > last_buy_pos and last_sell_pos >= 0:
                signal["signal"] = "SELL"
                logger.warning(f"[SINAL FALLBACK] Usando SELL (última ocorrência na posição {last_sell_pos})")
            elif last_buy_pos >= 0:
                signal["signal"] = "BUY"
            elif last_sell_pos >= 0:
                signal["signal"] = "SELL"
        
        # Para NO_SIGNAL, não deve ter entrada, stop ou targets
        if signal["signal"] == "NO_SIGNAL":
            # NO_SIGNAL = não executar
            signal["entry_price"] = None
            signal["stop_loss"] = None
            signal["take_profit_1"] = None
            signal["take_profit_2"] = None
        else:
            # VALIDAÇÃO FINAL: Garantir que entry_price e stop_loss existem antes de retornar
            if not signal.get("entry_price") or not signal.get("stop_loss"):
                logger.error(f"[ERRO CRITICO] Sinal {signal['signal']} sem entry_price ou stop_loss definidos!")
                logger.error(f"Entry: {signal.get('entry_price')}, Stop: {signal.get('stop_loss')}")
                logger.error(f"Response preview: {response_text[:500]}...")
                # Tentar extrair preço do texto novamente com padrões mais flexíveis
                if not signal.get("entry_price") and response_text:
                    # Procurar por qualquer número que pareça um preço (mais flexível)
                    price_patterns = [
                        r"\$([0-9,]+\.?[0-9]+)",  # $90,563.50
                        r"([0-9]{1,3}(?:[,.][0-9]{1,2})?)\s*(?:USD|USDT)",  # 90,563.50 USD
                        r"preço[^0-9]*([0-9,]+\.?[0-9]+)",  # preço 90,563.50
                    ]
                    for pattern in price_patterns:
                        match = re.search(pattern, response_text, re.IGNORECASE)
                        if match:
                            try:
                                price_str = match.group(1).replace(",", "")
                                price = float(price_str)
                                # Validar se é um preço razoável
                                if 0.01 <= price <= 1000000:
                                    signal["entry_price"] = price
                                    logger.warning(f"[FALLBACK] Preço extraído do texto: ${price}")
                                    break
                            except ValueError:
                                continue
                
                # Se ainda não tem entry_price, usar valores padrão baseados no símbolo
                if not signal.get("entry_price"):
                    default_price = self._get_default_price(symbol)
                    signal["entry_price"] = default_price
                    logger.warning(f"[FALLBACK] Usando preço padrão para {symbol}: ${default_price}")

                # Calcular stop loss se não tiver
                if not signal.get("stop_loss") and signal.get("entry_price"):
                    if signal["signal"] == "BUY":
                        signal["stop_loss"] = signal["entry_price"] * 0.98
                    else:  # SELL
                        signal["stop_loss"] = signal["entry_price"] * 1.02
                    logger.warning(f"[FALLBACK] Stop loss calculado: ${signal['stop_loss']}")
            # Para BUY/SELL, OBRIGATÓRIO ter entrada, stop e targets
            entry_patterns = [
                r"entrada[^0-9]*\$?([0-9,]+\.?[0-9]*)",
                r"entry[^0-9]*\$?([0-9,]+\.?[0-9]*)",
                r"entry_price[^0-9]*[:=]\s*\$?([0-9,]+\.?[0-9]*)",
                r"preço[^0-9]*\$?([0-9,]+\.?[0-9]*)",
                r"preco[^0-9]*\$?([0-9,]+\.?[0-9]*)",
                r"current[^0-9]*price[^0-9]*\$?([0-9,]+\.?[0-9]*)"
            ]
            
            # Se já tem entry_price do JSON, não precisa extrair
            if not signal.get("entry_price"):
                signal["entry_price"] = None
                for pattern in entry_patterns:
                    entry_match = re.search(pattern, response_text, re.IGNORECASE)
                    if entry_match:
                        try:
                            price = float(entry_match.group(1).replace(",", ""))
                            # CORRIGIDO: Validar se o preço é realista (suporta todas as moedas)
                            # BTC: 90k+, ETH: 3k+, SOL: 100+, ADA: 0.4+, DOGE: 0.1+, etc.
                            if 0.01 <= price <= 1000000:
                                signal["entry_price"] = price
                                logger.info(f"[PRECO EXTRAIDO] Entry price encontrado via regex: ${price}")
                                break
                        except ValueError:
                            continue
                
                # FALLBACK: Se não encontrou entry_price, usar valores padrão
                # CORREÇÃO: Não usar asyncio.run() aqui pois já estamos em um event loop
                # O preço será obtido no método analyze() antes de chamar _process_agent_response
                if not signal.get("entry_price"):
                    signal["entry_price"] = self._get_default_price(symbol)
                    logger.warning(f"[FALLBACK] Usando preço padrão para {symbol}: ${signal['entry_price']}")
            
            # CORRIGIDO: Stop Loss - melhor extração com validação
            if signal["signal"] == "BUY":
                # Para BUY, stop loss deve ser ABAIXO da entrada
                stop_patterns = [
                    r"stop[^0-9]*loss[^0-9]*\$?([0-9,]+\.?[0-9]*)",
                    r"stop[^0-9]*\$?([0-9,]+\.?[0-9]*)",
                    r"sl[^0-9]*\$?([0-9,]+\.?[0-9]*)"
                ]
                
                signal["stop_loss"] = None
                for pattern in stop_patterns:
                    stop_match = re.search(pattern, response_text, re.IGNORECASE)
                    if stop_match:
                        try:
                            stop_price = float(stop_match.group(1).replace(",", ""))
                            # CORRIGIDO: Validar stop loss (suporta todas as moedas)
                            # Para BUY, stop loss deve ser menor que entrada
                            if signal["entry_price"] and 0.01 <= stop_price < signal["entry_price"]:
                                signal["stop_loss"] = stop_price
                                logger.info(f"[STOP LOSS EXTRAIDO] Stop loss encontrado via regex: ${stop_price}")
                                break
                        except ValueError:
                            continue
                
                # Se não encontrou, calcular baseado em 2% abaixo da entrada
                if not signal["stop_loss"] and signal["entry_price"]:
                    signal["stop_loss"] = signal["entry_price"] * 0.98
                    
            elif signal["signal"] == "SELL":
                # Para SELL, stop loss deve ser ACIMA da entrada
                stop_patterns = [
                    r"stop[^0-9]*loss[^0-9]*\$?([0-9,]+\.?[0-9]*)",
                    r"stop[^0-9]*\$?([0-9,]+\.?[0-9]*)",
                    r"sl[^0-9]*\$?([0-9,]+\.?[0-9]*)"
                ]
                
                signal["stop_loss"] = None
                for pattern in stop_patterns:
                    stop_match = re.search(pattern, response_text, re.IGNORECASE)
                    if stop_match:
                        try:
                            stop_price = float(stop_match.group(1).replace(",", ""))
                            # CORRIGIDO: Validar stop loss (suporta todas as moedas)
                            # Para SELL, stop loss deve ser maior que entrada
                            if signal["entry_price"] and stop_price > signal["entry_price"] and stop_price <= 1000000:
                                signal["stop_loss"] = stop_price
                                logger.info(f"[STOP LOSS EXTRAIDO] Stop loss encontrado via regex: ${stop_price}")
                                break
                        except ValueError:
                            continue
                
                # Se não encontrou, calcular baseado em 2% acima da entrada
                if not signal["stop_loss"] and signal["entry_price"]:
                    signal["stop_loss"] = signal["entry_price"] * 1.02
            
            # CORRIGIDO: Take Profit 1 - melhor extração
            if signal["signal"] == "BUY":
                # Para BUY, TP deve ser ACIMA da entrada
                tp1_patterns = [
                    r"take[^0-9]*profit[^0-9]*1[^0-9]*\$?([0-9,]+\.?[0-9]*)",
                    r"tp1[^0-9]*\$?([0-9,]+\.?[0-9]*)",
                    r"alvo[^0-9]*1[^0-9]*\$?([0-9,]+\.?[0-9]*)",
                    r"target[^0-9]*1[^0-9]*\$?([0-9,]+\.?[0-9]*)"
                ]
                
                signal["take_profit_1"] = None
                for pattern in tp1_patterns:
                    tp1_match = re.search(pattern, response_text, re.IGNORECASE)
                    if tp1_match:
                        try:
                            price = float(tp1_match.group(1).replace(",", ""))
                            if signal["entry_price"] and price > signal["entry_price"] and price <= 1000000:
                                signal["take_profit_1"] = price
                                break
                        except ValueError:
                            continue
                
                # Se não encontrou, calcular 2% acima
                if not signal["take_profit_1"] and signal["entry_price"]:
                    signal["take_profit_1"] = signal["entry_price"] * 1.02
                    
            elif signal["signal"] == "SELL":
                # Para SELL, TP deve ser ABAIXO da entrada
                tp1_patterns = [
                    r"take[^0-9]*profit[^0-9]*1[^0-9]*\$?([0-9,]+\.?[0-9]*)",
                    r"tp1[^0-9]*\$?([0-9,]+\.?[0-9]*)",
                    r"alvo[^0-9]*1[^0-9]*\$?([0-9,]+\.?[0-9]*)",
                    r"target[^0-9]*1[^0-9]*\$?([0-9,]+\.?[0-9]*)"
                ]

                signal["take_profit_1"] = None
                for pattern in tp1_patterns:
                    tp1_match = re.search(pattern, response_text, re.IGNORECASE)
                    if tp1_match:
                        try:
                            price = float(tp1_match.group(1).replace(",", ""))
                            # CORRIGIDO: price >= 0.01 ao invés de price >= 1000
                            # Isso permite TP para moedas baratas como DOGE, ADA, etc.
                            if signal["entry_price"] and price < signal["entry_price"] and price >= 0.01:
                                signal["take_profit_1"] = price
                                break
                        except ValueError:
                            continue

                # Se não encontrou, calcular 2% abaixo
                if not signal["take_profit_1"] and signal["entry_price"]:
                    signal["take_profit_1"] = signal["entry_price"] * 0.98
            
            # CORRIGIDO: Take Profit 2 - melhor extração
            if signal["signal"] == "BUY":
                tp2_patterns = [
                    r"take[^0-9]*profit[^0-9]*2[^0-9]*\$?([0-9,]+\.?[0-9]*)",
                    r"tp2[^0-9]*\$?([0-9,]+\.?[0-9]*)",
                    r"alvo[^0-9]*2[^0-9]*\$?([0-9,]+\.?[0-9]*)",
                    r"target[^0-9]*2[^0-9]*\$?([0-9,]+\.?[0-9]*)"
                ]
                
                signal["take_profit_2"] = None
                for pattern in tp2_patterns:
                    tp2_match = re.search(pattern, response_text, re.IGNORECASE)
                    if tp2_match:
                        try:
                            price = float(tp2_match.group(1).replace(",", ""))
                            if signal["entry_price"] and price > signal["take_profit_1"] and price <= 1000000:
                                signal["take_profit_2"] = price
                                break
                        except ValueError:
                            continue
                
                # Se não encontrou, calcular 5% acima
                if not signal["take_profit_2"] and signal["entry_price"]:
                    signal["take_profit_2"] = signal["entry_price"] * 1.05
                    
            elif signal["signal"] == "SELL":
                tp2_patterns = [
                    r"take[^0-9]*profit[^0-9]*2[^0-9]*\$?([0-9,]+\.?[0-9]*)",
                    r"tp2[^0-9]*\$?([0-9,]+\.?[0-9]*)",
                    r"alvo[^0-9]*2[^0-9]*\$?([0-9,]+\.?[0-9]*)",
                    r"target[^0-9]*2[^0-9]*\$?([0-9,]+\.?[0-9]*)"
                ]

                signal["take_profit_2"] = None
                for pattern in tp2_patterns:
                    tp2_match = re.search(pattern, response_text, re.IGNORECASE)
                    if tp2_match:
                        try:
                            price = float(tp2_match.group(1).replace(",", ""))
                            # CORRIGIDO: price >= 0.01 ao invés de price >= 1000
                            # Isso permite TP para moedas baratas como DOGE, ADA, etc.
                            if signal["entry_price"] and signal["take_profit_1"] and price < signal["take_profit_1"] and price >= 0.01:
                                signal["take_profit_2"] = price
                                break
                        except ValueError:
                            continue

                # Se não encontrou, calcular 5% abaixo
                if not signal["take_profit_2"] and signal["entry_price"]:
                    signal["take_profit_2"] = signal["entry_price"] * 0.95
        
        # VALIDAÇÃO FINAL: Garantir que entry_price e stop_loss existem antes de retornar
        # NOTA: Não podemos usar asyncio.run() aqui pois já estamos em um event loop
        # Vamos usar o preço atual que já foi coletado anteriormente ou calcular baseado em valores padrão
        if signal["signal"] in ["BUY", "SELL"]:
            if not signal.get("entry_price") or not signal.get("stop_loss"):
                logger.error(f"[ERRO CRITICO] Sinal {signal['signal']} sem entry_price ou stop_loss definidos!")
                logger.error(f"Entry: {signal.get('entry_price')}, Stop: {signal.get('stop_loss')}")
                logger.error(f"Response preview: {response_text[:500] if response_text else 'N/A'}...")
                
                # Tentar extrair preço do texto novamente com padrões mais flexíveis
                if not signal.get("entry_price") and response_text:
                    # Procurar por qualquer número que pareça um preço (mais flexível)
                    price_patterns = [
                        r"\$([0-9,]+\.?[0-9]+)",  # $90,563.50
                        r"([0-9]{1,3}(?:[,.][0-9]{1,2})?)\s*(?:USD|USDT)",  # 90,563.50 USD
                        r"preço[^0-9]*([0-9,]+\.?[0-9]+)",  # preço 90,563.50
                    ]
                    for pattern in price_patterns:
                        match = re.search(pattern, response_text, re.IGNORECASE)
                        if match:
                            try:
                                price_str = match.group(1).replace(",", "")
                                # Se o padrão capturou com ponto, manter o ponto; se não, assumir decimal
                                if "." in price_str:
                                    price = float(price_str)
                                else:
                                    # Se não tem ponto, pode ser um número inteiro ou precisamos adicionar ponto decimal
                                    price = float(price_str)
                                # Validar se é um preço razoável
                                if 0.01 <= price <= 1000000:
                                    signal["entry_price"] = price
                                    logger.warning(f"[FALLBACK] Preço extraído do texto: ${price}")
                                    break
                            except ValueError:
                                continue
                
                # Se ainda não tem entry_price, usar valores padrão baseados no símbolo
                if not signal.get("entry_price"):
                    # Valores padrão aproximados (será substituído quando o sistema coletar preço real)
                    default_price = self._get_default_price(symbol)
                    signal["entry_price"] = default_price
                    logger.warning(f"[FALLBACK] Usando preço padrão para {symbol}: ${default_price}")
                
                # Calcular stop loss se não tiver
                if not signal.get("stop_loss") and signal.get("entry_price"):
                    if signal["signal"] == "BUY":
                        signal["stop_loss"] = signal["entry_price"] * 0.98
                    else:  # SELL
                        signal["stop_loss"] = signal["entry_price"] * 1.02
                    logger.warning(f"[FALLBACK] Stop loss calculado: ${signal['stop_loss']}")
        
        # Extrair confiança - corrigir regex para capturar corretamente
        conf_patterns = [
            r"confiança[^0-9]*([0-9]+)/10",
            r"confiança[^0-9]*([0-9]+)",
            r"confidence[^0-9]*([0-9]+)/10",
            r"confidence[^0-9]*([0-9]+)"
        ]
        
        signal["confidence"] = 5  # Default
        for pattern in conf_patterns:
            conf_match = re.search(pattern, response_text, re.IGNORECASE)
            if conf_match:
                signal["confidence"] = int(conf_match.group(1))
                break
        
        return signal
    
    def _save_signal(self, signal: Dict[str, Any]):
        """Salva sinal em arquivo JSON"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"signals/agno_{signal['symbol']}_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(signal, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"[SALVO] Sinal salvo: {filename}")
    
    def _save_deepseek_response(self, symbol: str, prompt: str, response: Any, analysis_data: Dict[str, Any] = None):
        """
        Salva prompt e resposta do DeepSeek em diretório organizado por data (ano/mês/dia)
        para auditoria e verificação de sinais gerados.
        
        Args:
            symbol: Símbolo analisado
            prompt: Prompt de texto enviado ao DeepSeek
            response: Resposta recebida do DeepSeek
            analysis_data: JSON de análise enviado (dados sumarizados)
        """
        try:
            now = datetime.now()
            # Criar diretório: deepseek_logs/YYYY/MM/DD
            log_dir = Path(f"deepseek_logs/{now.year}/{now.month:02d}/{now.day:02d}")
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Nome do arquivo: symbol_timestamp.json
            timestamp = now.strftime('%Y%m%d_%H%M%S')
            filename = log_dir / f"{symbol}_{timestamp}.json"
            
            # Preparar dados para salvar
            response_data = {
                "symbol": symbol,
                "timestamp": now.isoformat(),
                "prompt_sent": prompt,  # Prompt de texto enviado
                "analysis_data_sent": analysis_data if analysis_data else {},  # JSON de análise enviado
                "response_received": str(response),  # Resposta bruta do DeepSeek
                "response_type": type(response).__name__
            }
            
            # Salvar arquivo
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(response_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"[DEEPSEEK LOG] Prompt e resposta salvos: {filename}")
            print(f"[DEEPSEEK LOG] Resposta salva em: {filename}")
            
        except Exception as e:
            logger.error(f"Erro ao salvar resposta do DeepSeek: {e}")
            # Não bloquear o fluxo se houver erro ao salvar
    
    def _print_summary(self, signal: Dict[str, Any]):
        """Imprime resumo do sinal"""
        print("\n" + "="*60)
        print("RESULTADO DA ANALISE")
        print("="*60)
        print(f"Sinal: {signal.get('signal', 'N/A')}")
        print(f"Confianca: {signal.get('confidence', 0)}/10")
        if signal.get('entry_price'):
            print(f"Entrada: ${signal['entry_price']:,.2f}")
        if signal.get('stop_loss'):
            print(f"Stop Loss: ${signal['stop_loss']:,.2f}")
        print("="*60)
    
    def _create_error_signal(self, symbol: str, error: str) -> Dict[str, Any]:
        """Cria sinal de erro"""
        return {
            "symbol": symbol,
            "signal": "HOLD",
            "confidence": 0,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
    
    async def monitor_continuous(self, symbols: List[str], interval: int = 300):
        """
        Monitora múltiplos símbolos continuamente.
        
        Args:
            symbols: Lista de símbolos
            interval: Intervalo em segundos
        """
        print(f"[MONITOR] Monitoramento continuo de {symbols}")
        print(f"Intervalo: {interval}s")
        
        while True:
            for symbol in symbols:
                try:
                    await self.analyze(symbol)
                except Exception as e:
                    print(f"[ERRO] Erro em {symbol}: {e}")
                
                await asyncio.sleep(10)  # Pausa entre símbolos
            
            print(f"[AGUARDANDO] Aguardando {interval}s...")
            await asyncio.sleep(interval)