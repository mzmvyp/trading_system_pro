"""
Signal parsing from LLM responses
Extracts structured trading signals from text using JSON parsing and regex fallbacks
"""
import json
import re
from datetime import datetime
from typing import Any, Dict, Optional

from src.core.logger import get_logger

logger = get_logger(__name__)

# Default prices for fallback when market data unavailable
# NOTE: These are rough fallback values only used when all API calls fail.
# The system should always prefer live prices from Binance.
DEFAULT_PRICES = {
    "BTCUSDT": 95000, "ETHUSDT": 2500, "SOLUSDT": 150,
    "BNBUSDT": 650, "ADAUSDT": 0.70, "XRPUSDT": 2.50,
    "DOGEUSDT": 0.25, "AVAXUSDT": 35, "DOTUSDT": 7,
    "LINKUSDT": 18, "PAXGUSDT": 5100
}


def get_default_price(symbol: str) -> float:
    """Returns default price for a symbol"""
    return DEFAULT_PRICES.get(symbol, 100)


def extract_balanced_json(text: str) -> Optional[str]:
    """
    Extrai JSON balanceado corretamente, mesmo com objetos aninhados.
    """
    json_block_match = re.search(r'```json\s*(\{.*?)\s*```', text, re.DOTALL)
    if json_block_match:
        start_pos = json_block_match.start(1)
        json_start = text.find('{', start_pos)
        if json_start == -1:
            return None
    else:
        json_start = text.find('{')
        if json_start == -1:
            return None

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
                    return text[json_start:i+1]

    return None


def extract_price_from_text(text: str, min_price: float = 0.01, max_price: float = 1000000) -> Optional[float]:
    """
    Extrai preço de texto usando múltiplos padrões regex.
    """
    if not text:
        return None

    price_patterns = [
        r"\$([0-9,]+\.?[0-9]+)",
        r"([0-9]{1,3}(?:[,.][0-9]{1,2})?)\s*(?:USD|USDT)",
        r"preço[^0-9]*([0-9,]+\.?[0-9]+)",
        r"preco[^0-9]*([0-9,]+\.?[0-9]+)",
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


async def process_agent_response(response: Any, symbol: str) -> Dict[str, Any]:
    """Processa resposta do agent em formato estruturado"""

    response_text = None
    if hasattr(response, 'content'):
        response_text = str(response.content) if response.content else None
        logger.debug(f"[AGNO] Conteúdo extraído de response.content: {response_text[:200] if response_text else 'None'}...")
    elif hasattr(response, 'output'):
        response_text = str(response.output)
    elif hasattr(response, 'messages') and len(response.messages) > 0:
        last_message = response.messages[-1]
        response_text = str(last_message.content) if hasattr(last_message, 'content') else str(last_message)
    elif isinstance(response, dict):
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
        response_text = str(response)
        logger.warning(f"[AGNO] Resposta não é RunOutput conhecido, usando str(): {type(response)}")

    signal = {
        "symbol": symbol,
        "timestamp": datetime.now().isoformat(),
        "agent_response": response_text[:500] if response_text else "N/A",
    }

    if not response_text:
        logger.error("[ERRO] Não foi possível extrair conteúdo da resposta do AGNO")
        return {
            "symbol": symbol, "timestamp": datetime.now().isoformat(),
            "signal": "NO_SIGNAL", "confidence": 0,
            "reason": "Erro ao extrair resposta do AGNO"
        }

    # Try JSON first
    json_text = extract_balanced_json(response_text)
    if json_text:
        try:
            structured = json.loads(json_text)
            logger.info(f"[JSON ESTRUTURADO] Sinal extraído via JSON: {structured.get('signal', 'N/A')}")
            if structured.get("signal") in ["BUY", "SELL", "NO_SIGNAL"]:
                signal.update({
                    "signal": structured.get("signal", "NO_SIGNAL"),
                    "entry_price": structured.get("entry_price"),
                    "stop_loss": structured.get("stop_loss"),
                    "take_profit_1": structured.get("take_profit_1"),
                    "take_profit_2": structured.get("take_profit_2"),
                    "confidence": structured.get("confidence", 5)
                })
                if signal["signal"] in ["BUY", "SELL"] and not signal.get("entry_price"):
                    logger.warning("[JSON] Sinal BUY/SELL sem entry_price, usando fallback regex")
                else:
                    return signal
        except json.JSONDecodeError as e:
            logger.warning(f"[JSON] Erro ao decodificar JSON: {e}, usando fallback regex")

    # Regex fallback for signal type
    signal["signal"] = "NO_SIGNAL"

    final_signal_patterns = [
        r"SINAL\s+FINAL[:\s]+\*?\*?(BUY|SELL)\*?\*?",
        r"SINAL\s+FINAL[:\s]+(BUY|SELL)",
        r"###\s*\*\*SINAL\s+FINAL[:\s]+\*\*(BUY|SELL)",
        r"##\s+SINAL\s+FINAL[:\s]+(BUY|SELL)",
        r"RESUMO[^:]*Sinal\s+(BUY|SELL)",
        r"Conclusão[^:]*:\s*(BUY|SELL)",
        r"Recomendação[^:]*:\s*(BUY|SELL)"
    ]

    for pattern in final_signal_patterns:
        matches = list(re.finditer(pattern, response_text, re.IGNORECASE | re.MULTILINE))
        if matches:
            last_match = matches[-1]
            signal_type = last_match.group(1).upper()
            if signal_type in ["BUY", "SELL"]:
                signal["signal"] = signal_type
                logger.info(f"[SINAL EXTRAIDO] Encontrado '{signal_type}' via padrão: {pattern[:50]}")
                break

    if signal["signal"] == "NO_SIGNAL":
        buy_matches = list(re.finditer(r'\bBUY\b', response_text, re.IGNORECASE))
        sell_matches = list(re.finditer(r'\bSELL\b', response_text, re.IGNORECASE))
        last_buy_pos = buy_matches[-1].start() if buy_matches else -1
        last_sell_pos = sell_matches[-1].start() if sell_matches else -1
        if last_buy_pos > last_sell_pos and last_buy_pos >= 0:
            signal["signal"] = "BUY"
        elif last_sell_pos > last_buy_pos and last_sell_pos >= 0:
            signal["signal"] = "SELL"
        elif last_buy_pos >= 0:
            signal["signal"] = "BUY"
        elif last_sell_pos >= 0:
            signal["signal"] = "SELL"

    if signal["signal"] == "NO_SIGNAL":
        signal["entry_price"] = None
        signal["stop_loss"] = None
        signal["take_profit_1"] = None
        signal["take_profit_2"] = None
    else:
        # Extract prices using regex if not from JSON
        _extract_trade_prices(signal, response_text, symbol)

    # Extract confidence
    conf_patterns = [
        r"confiança[^0-9]*([0-9]+)/10",
        r"confiança[^0-9]*([0-9]+)",
        r"confidence[^0-9]*([0-9]+)/10",
        r"confidence[^0-9]*([0-9]+)"
    ]
    if "confidence" not in signal or signal.get("confidence") == 5:
        signal["confidence"] = 5
        for pattern in conf_patterns:
            conf_match = re.search(pattern, response_text, re.IGNORECASE)
            if conf_match:
                signal["confidence"] = int(conf_match.group(1))
                break

    return signal


def _extract_trade_prices(signal: Dict, response_text: str, symbol: str):
    """Extract entry, stop loss, and take profit prices from text"""
    entry_patterns = [
        r"entrada[^0-9]*\$?([0-9,]+\.?[0-9]*)",
        r"entry[^0-9]*\$?([0-9,]+\.?[0-9]*)",
        r"entry_price[^0-9]*[:=]\s*\$?([0-9,]+\.?[0-9]*)",
        r"preço[^0-9]*\$?([0-9,]+\.?[0-9]*)",
        r"preco[^0-9]*\$?([0-9,]+\.?[0-9]*)",
        r"current[^0-9]*price[^0-9]*\$?([0-9,]+\.?[0-9]*)"
    ]

    # Entry price
    if not signal.get("entry_price"):
        signal["entry_price"] = None
        for pattern in entry_patterns:
            entry_match = re.search(pattern, response_text, re.IGNORECASE)
            if entry_match:
                try:
                    price = float(entry_match.group(1).replace(",", ""))
                    if 0.01 <= price <= 1000000:
                        signal["entry_price"] = price
                        logger.info(f"[PRECO EXTRAIDO] Entry price encontrado via regex: ${price}")
                        break
                except ValueError:
                    continue

        if not signal.get("entry_price"):
            signal["entry_price"] = get_default_price(symbol)
            logger.warning(f"[FALLBACK] Usando preço padrão para {symbol}: ${signal['entry_price']}")

    # Stop loss
    stop_patterns = [
        r"stop[^0-9]*loss[^0-9]*\$?([0-9,]+\.?[0-9]*)",
        r"stop[^0-9]*\$?([0-9,]+\.?[0-9]*)",
        r"sl[^0-9]*\$?([0-9,]+\.?[0-9]*)"
    ]

    if signal["signal"] == "BUY":
        signal["stop_loss"] = None
        for pattern in stop_patterns:
            stop_match = re.search(pattern, response_text, re.IGNORECASE)
            if stop_match:
                try:
                    stop_price = float(stop_match.group(1).replace(",", ""))
                    if signal["entry_price"] and 0.01 <= stop_price < signal["entry_price"]:
                        signal["stop_loss"] = stop_price
                        break
                except ValueError:
                    continue
        if not signal["stop_loss"] and signal["entry_price"]:
            signal["stop_loss"] = signal["entry_price"] * 0.98
    elif signal["signal"] == "SELL":
        signal["stop_loss"] = None
        for pattern in stop_patterns:
            stop_match = re.search(pattern, response_text, re.IGNORECASE)
            if stop_match:
                try:
                    stop_price = float(stop_match.group(1).replace(",", ""))
                    if signal["entry_price"] and stop_price > signal["entry_price"] and stop_price <= 1000000:
                        signal["stop_loss"] = stop_price
                        break
                except ValueError:
                    continue
        if not signal["stop_loss"] and signal["entry_price"]:
            signal["stop_loss"] = signal["entry_price"] * 1.02

    # Take profit patterns
    tp1_patterns = [
        r"take[^0-9]*profit[^0-9]*1[^0-9]*\$?([0-9,]+\.?[0-9]*)",
        r"tp1[^0-9]*\$?([0-9,]+\.?[0-9]*)",
        r"alvo[^0-9]*1[^0-9]*\$?([0-9,]+\.?[0-9]*)",
        r"target[^0-9]*1[^0-9]*\$?([0-9,]+\.?[0-9]*)"
    ]
    tp2_patterns = [
        r"take[^0-9]*profit[^0-9]*2[^0-9]*\$?([0-9,]+\.?[0-9]*)",
        r"tp2[^0-9]*\$?([0-9,]+\.?[0-9]*)",
        r"alvo[^0-9]*2[^0-9]*\$?([0-9,]+\.?[0-9]*)",
        r"target[^0-9]*2[^0-9]*\$?([0-9,]+\.?[0-9]*)"
    ]

    _extract_take_profits(signal, response_text, tp1_patterns, tp2_patterns)


def _extract_take_profits(signal: Dict, response_text: str, tp1_patterns: list, tp2_patterns: list):
    """Extract take profit levels from text"""
    if signal["signal"] == "BUY":
        signal["take_profit_1"] = None
        for pattern in tp1_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                try:
                    price = float(match.group(1).replace(",", ""))
                    if signal["entry_price"] and price > signal["entry_price"] and price <= 1000000:
                        signal["take_profit_1"] = price
                        break
                except ValueError:
                    continue
        if not signal["take_profit_1"] and signal["entry_price"]:
            signal["take_profit_1"] = signal["entry_price"] * 1.02

        signal["take_profit_2"] = None
        for pattern in tp2_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                try:
                    price = float(match.group(1).replace(",", ""))
                    if signal["entry_price"] and price > signal.get("take_profit_1", 0) and price <= 1000000:
                        signal["take_profit_2"] = price
                        break
                except ValueError:
                    continue
        if not signal["take_profit_2"] and signal["entry_price"]:
            signal["take_profit_2"] = signal["entry_price"] * 1.05

    elif signal["signal"] == "SELL":
        signal["take_profit_1"] = None
        for pattern in tp1_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                try:
                    price = float(match.group(1).replace(",", ""))
                    if signal["entry_price"] and price < signal["entry_price"] and price >= 0.01:
                        signal["take_profit_1"] = price
                        break
                except ValueError:
                    continue
        if not signal["take_profit_1"] and signal["entry_price"]:
            signal["take_profit_1"] = signal["entry_price"] * 0.98

        signal["take_profit_2"] = None
        for pattern in tp2_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                try:
                    price = float(match.group(1).replace(",", ""))
                    if signal["entry_price"] and signal["take_profit_1"] and price < signal["take_profit_1"] and price >= 0.01:
                        signal["take_profit_2"] = price
                        break
                except ValueError:
                    continue
        if not signal["take_profit_2"] and signal["entry_price"]:
            signal["take_profit_2"] = signal["entry_price"] * 0.95
