"""
Executor de Ordens para Binance Futures
Executa ordens reais na Binance Futures USDT-M

AVISO: Este modulo executa ordens REAIS com dinheiro REAL.
Use com extrema cautela. Teste sempre em modo paper primeiro.
"""

import asyncio
import hashlib
import hmac
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import aiohttp
from dotenv import load_dotenv

# Carregar variaveis de ambiente
load_dotenv()

from src.core.config import settings  # noqa: E402
from src.core.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


class BinanceFuturesExecutor:
    """
    Executor de ordens para Binance Futures USDT-M.

    Recursos:
    - Ordens de mercado e limite
    - Stop Loss e Take Profit automáticos
    - Gestão de alavancagem
    - Verificação de margem disponível
    - Log completo de todas as operações
    - Suporte a Testnet para testes
    """

    # URLs da Binance Futures
    PRODUCTION_URL = "https://fapi.binance.com"
    TESTNET_URL = "https://testnet.binancefuture.com"

    # Class-level timestamp offset (persists across instances)
    _server_time_offset = None  # offset = server_time - local_time

    def __init__(self):
        """
        Inicializa o executor de Binance Futures.

        Requer:
        - BINANCE_API_KEY: Chave da API Binance
        - BINANCE_SECRET_KEY: Chave secreta da API Binance
        """
        # Verificar se deve usar testnet
        self.use_testnet = os.getenv("BINANCE_TESTNET", "false").lower() == "true"
        self.BASE_URL = self.TESTNET_URL if self.use_testnet else self.PRODUCTION_URL

        self.api_key = os.getenv("BINANCE_API_KEY") or settings.binance_api_key
        self.api_secret = os.getenv("BINANCE_SECRET_KEY") or settings.binance_secret_key

        if not self.api_key or not self.api_secret:
            raise ValueError(
                "BINANCE_API_KEY e BINANCE_SECRET_KEY sao necessarios para modo REAL. "
                "Configure no arquivo .env ou nas variaveis de ambiente."
            )

        # Diretório para logs de ordens reais
        self.orders_dir = Path("real_orders")
        self.orders_dir.mkdir(exist_ok=True)

        # Alavancagem padrão (conservadora)
        self.default_leverage = 5

        # Log do modo
        if self.use_testnet:
            logger.info("="*60)
            logger.info("[BINANCE FUTURES] MODO TESTNET ATIVO")
            logger.info(f"[BINANCE FUTURES] URL: {self.BASE_URL}")
            if settings.trading_mode == "real":
                logger.warning("[BINANCE FUTURES] ATENCAO: TRADING_MODE=real com TESTNET=true. Ordens serao executadas na TESTNET.")
            logger.info("="*60)
        else:
            logger.warning("="*60)
            logger.warning("[BINANCE FUTURES] MODO PRODUCAO ATIVO - ORDENS REAIS!")
            logger.warning(f"[BINANCE FUTURES] URL: {self.BASE_URL}")
            logger.warning("="*60)

    def _generate_signature(self, params: Dict[str, Any]) -> str:
        """Gera assinatura HMAC SHA256 para autenticação"""
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Dict[str, Any] = None,
        signed: bool = False,
        retry_timestamp: bool = True
    ) -> Dict[str, Any]:
        """
        Faz requisição à API Binance Futures.

        Args:
            method: GET, POST, DELETE
            endpoint: Endpoint da API (ex: /fapi/v1/order)
            params: Parâmetros da requisição
            signed: Se True, adiciona timestamp e assinatura
            retry_timestamp: Se True, faz retry com timestamp ajustado em caso de erro -1021

        Returns:
            Resposta da API como dict
        """
        url = f"{self.BASE_URL}{endpoint}"
        params = params or {}

        # Fazer cópia dos params para não modificar o original
        request_params = params.copy()

        headers = {
            "X-MBX-APIKEY": self.api_key
        }

        if signed:
            # Usar offset persistido se disponível, senão subtrair 1000ms como fallback
            local_ms = int(time.time() * 1000)
            if BinanceFuturesExecutor._server_time_offset is not None:
                request_params["timestamp"] = local_ms + BinanceFuturesExecutor._server_time_offset
            else:
                request_params["timestamp"] = local_ms - 1000
            request_params["recvWindow"] = 10000  # 10s window para evitar erro -1021
            request_params["signature"] = self._generate_signature(request_params)

        try:
            async with aiohttp.ClientSession() as session:
                if method == "GET":
                    async with session.get(url, params=request_params, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as response:
                        data = await response.json()
                elif method == "POST":
                    async with session.post(url, params=request_params, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as response:
                        data = await response.json()
                elif method == "DELETE":
                    async with session.delete(url, params=request_params, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as response:
                        data = await response.json()
                else:
                    raise ValueError(f"Metodo HTTP nao suportado: {method}")

                # Verificar erro da API (code pode vir como int ou str da Binance)
                try:
                    code_val = int(data["code"]) if isinstance(data, dict) and "code" in data else 0
                except (TypeError, ValueError):
                    code_val = 0
                if isinstance(data, dict) and "code" in data and code_val < 0:
                    error_msg = data.get("msg", "Erro desconhecido")
                    error_code = code_val

                    # Erro -1021: Timestamp ahead of server - fazer retry com server time sync
                    if error_code == -1021 and signed and retry_timestamp:
                        logger.warning("[BINANCE API] Erro -1021 detectado. Sincronizando com server time...")
                        # Buscar server time da Binance para calcular offset e persistir
                        try:
                            async with aiohttp.ClientSession() as time_session:
                                async with time_session.get(f"{self.BASE_URL}/fapi/v1/time", timeout=aiohttp.ClientTimeout(total=5)) as time_resp:
                                    time_data = await time_resp.json()
                                    server_time = time_data.get("serverTime", int(time.time() * 1000) - 3000)
                                    # Persist offset for future requests
                                    BinanceFuturesExecutor._server_time_offset = server_time - int(time.time() * 1000)
                                    logger.info(f"[BINANCE API] Offset persistido: {BinanceFuturesExecutor._server_time_offset}ms")
                                    request_params["timestamp"] = server_time
                        except Exception:
                            # Fallback: subtrair 3000ms se nao conseguir buscar server time
                            request_params["timestamp"] = int(time.time() * 1000) - 3000
                        request_params.pop("signature", None)
                        request_params["signature"] = self._generate_signature(request_params)

                        # Retry uma vez com timestamp ajustado (fazer requisição diretamente)
                        await asyncio.sleep(0.1)  # Pequeno delay antes do retry
                        try:
                            async with aiohttp.ClientSession() as retry_session:
                                if method == "GET":
                                    async with retry_session.get(url, params=request_params, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as retry_response:
                                        retry_data = await retry_response.json()
                                elif method == "POST":
                                    async with retry_session.post(url, params=request_params, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as retry_response:
                                        retry_data = await retry_response.json()
                                elif method == "DELETE":
                                    async with retry_session.delete(url, params=request_params, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as retry_response:
                                        retry_data = await retry_response.json()

                                # Verificar se o retry foi bem-sucedido (code pode ser int ou str)
                                try:
                                    retry_code = int(retry_data["code"]) if isinstance(retry_data, dict) and "code" in retry_data else 0
                                except (TypeError, ValueError):
                                    retry_code = 0
                                if isinstance(retry_data, dict) and "code" in retry_data and retry_code < 0:
                                    error_msg = retry_data.get("msg", "Erro desconhecido")
                                    error_code = retry_code
                                    if error_code == -4046:
                                        logger.info(f"[BINANCE API] Codigo: {error_code}, Mensagem: {error_msg}")
                                    else:
                                        logger.error(f"[BINANCE API ERROR] Codigo: {error_code}, Mensagem: {error_msg}")
                                    return {"error": error_msg, "code": error_code}

                                logger.info("[BINANCE API] Retry com timestamp ajustado foi bem-sucedido")
                                return retry_data
                        except Exception as retry_e:
                            logger.error(f"[BINANCE] Erro no retry: {retry_e}")
                            return {"error": error_msg, "code": error_code}

                    # Erro -4046 é apenas um aviso (margem já está no tipo correto)
                    if error_code == -4046:
                        logger.info(f"[BINANCE API] Codigo: {error_code}, Mensagem: {error_msg}")
                    else:
                        logger.error(f"[BINANCE API ERROR] Codigo: {error_code}, Mensagem: {error_msg}")

                    return {"error": error_msg, "code": error_code}

                return data

        except aiohttp.ClientError as e:
            logger.error(f"[BINANCE] Erro de conexão: {e}")
            return {"error": f"Erro de conexão: {str(e)}"}
        except asyncio.TimeoutError:
            logger.error("[BINANCE] Timeout na requisição")
            return {"error": "Timeout na requisição"}
        except Exception as e:
            logger.exception(f"[BINANCE] Erro inesperado: {e}")
            return {"error": f"Erro inesperado: {str(e)}"}

    async def get_account_info(self) -> Dict[str, Any]:
        """Obtém informações da conta Futures"""
        return await self._request("GET", "/fapi/v2/account", signed=True)

    async def get_balance(self) -> Dict[str, Any]:
        """Obtém saldo disponível em USDT"""
        account = await self.get_account_info()
        if "error" in account:
            return account

        # Encontrar saldo USDT
        for asset in account.get("assets", []):
            if asset["asset"] == "USDT":
                wallet_balance = float(asset["walletBalance"])
                return {
                    "available_balance": float(asset["availableBalance"]),
                    "wallet_balance": wallet_balance,
                    "total_balance": wallet_balance,  # Alias para clareza
                    "unrealized_pnl": float(asset["unrealizedProfit"]),
                    "margin_balance": float(asset["marginBalance"])
                }

        return {"error": "USDT nao encontrado na conta"}

    async def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Obtém posição aberta para um símbolo"""
        positions = await self._request("GET", "/fapi/v2/positionRisk", {"symbol": symbol}, signed=True)

        if isinstance(positions, dict) and "error" in positions:
            return positions

        for pos in positions:
            if pos["symbol"] == symbol and float(pos["positionAmt"]) != 0:
                return {
                    "symbol": symbol,
                    "side": "LONG" if float(pos["positionAmt"]) > 0 else "SHORT",
                    "position_amt": float(pos["positionAmt"]),
                    "entry_price": float(pos["entryPrice"]),
                    "unrealized_pnl": float(pos["unRealizedProfit"]),
                    "leverage": int(pos["leverage"]),
                    "liquidation_price": float(pos["liquidationPrice"])
                }

        return None  # Sem posição aberta

    async def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """Define alavancagem para um símbolo"""
        params = {
            "symbol": symbol,
            "leverage": leverage
        }
        return await self._request("POST", "/fapi/v1/leverage", params, signed=True)

    async def set_margin_type(self, symbol: str, margin_type: str = "ISOLATED") -> Dict[str, Any]:
        """Define tipo de margem (ISOLATED ou CROSSED)"""
        params = {
            "symbol": symbol,
            "marginType": margin_type
        }
        result = await self._request("POST", "/fapi/v1/marginType", params, signed=True)

        # Erro -4046 significa que já está no tipo de margem correto
        if isinstance(result, dict) and result.get("code") == -4046:
            return {"success": True, "message": f"Margem ja esta em {margin_type}"}

        return result

    async def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Obtém informações do símbolo (precisão, filtros, etc.)"""
        exchange_info = await self._request("GET", "/fapi/v1/exchangeInfo")

        if "error" in exchange_info:
            return exchange_info

        for sym in exchange_info.get("symbols", []):
            if sym["symbol"] == symbol:
                # Extrair precisão de quantidade e preço
                quantity_precision = sym["quantityPrecision"]
                price_precision = sym["pricePrecision"]

                # Extrair filtros
                min_qty = None
                min_notional = None
                for filter in sym.get("filters", []):
                    if filter["filterType"] == "LOT_SIZE":
                        min_qty = float(filter["minQty"])
                    elif filter["filterType"] == "MIN_NOTIONAL":
                        min_notional = float(filter.get("notional", 0))

                # Extrair leverage brackets (máximo permitido pela Binance para este par)
                # Binance retorna leverageBrackets ou pode ser consultado via /fapi/v1/leverageBracket
                # No exchangeInfo, o campo é "requiredMarginPercent" ou podemos inferir de "brackets"
                # Fallback seguro: usar limites conservadores por tipo de ativo
                max_leverage = 125  # Default máximo Binance
                # Binance Futures exchangeInfo não retém max leverage diretamente,
                # mas podemos inferir de "maintMarginPercent" se disponível
                maint_margin_pct = float(sym.get("maintMarginPercent", 0))
                if maint_margin_pct > 0:
                    # maintMarginPercent indica margem mínima, inverso ~= max leverage teórico
                    inferred_max = int(100 / maint_margin_pct)
                    max_leverage = min(max_leverage, inferred_max)

                return {
                    "symbol": symbol,
                    "quantity_precision": quantity_precision,
                    "price_precision": price_precision,
                    "min_qty": min_qty,
                    "min_notional": min_notional,
                    "max_leverage": max_leverage
                }

        return {"error": f"Simbolo {symbol} nao encontrado"}

    def _round_quantity(self, quantity: float, precision: int) -> float:
        """Arredonda quantidade para a precisão correta"""
        return round(quantity, precision)

    def _round_price(self, price: float, precision: int) -> float:
        """Arredonda preço para a precisão correta"""
        return round(price, precision)

    async def place_market_order(
        self,
        symbol: str,
        side: str,  # BUY ou SELL
        quantity: float,
        reduce_only: bool = False
    ) -> Dict[str, Any]:
        """
        Coloca ordem de mercado.

        Args:
            symbol: Símbolo (ex: BTCUSDT)
            side: BUY ou SELL
            quantity: Quantidade em unidades base
            reduce_only: Se True, apenas reduz posição

        Returns:
            Resultado da ordem
        """
        # Obter informações do símbolo
        symbol_info = await self.get_symbol_info(symbol)
        if symbol_info and "quantity_precision" in symbol_info:
            quantity = self._round_quantity(quantity, symbol_info["quantity_precision"])

        params = {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": quantity
        }

        if reduce_only:
            params["reduceOnly"] = "true"

        logger.warning(f"[ORDEM MERCADO] {side} {quantity} {symbol}")
        result = await self._request("POST", "/fapi/v1/order", params, signed=True)

        # Log da ordem
        self._log_order(symbol, "MARKET", side, quantity, result)

        return result

    async def place_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        reduce_only: bool = False
    ) -> Dict[str, Any]:
        """
        Coloca ordem limite.

        Args:
            symbol: Símbolo (ex: BTCUSDT)
            side: BUY ou SELL
            quantity: Quantidade em unidades base
            price: Preço limite
            reduce_only: Se True, apenas reduz posição

        Returns:
            Resultado da ordem
        """
        # Obter informações do símbolo
        symbol_info = await self.get_symbol_info(symbol)
        if symbol_info:
            quantity = self._round_quantity(quantity, symbol_info.get("quantity_precision", 3))
            price = self._round_price(price, symbol_info.get("price_precision", 2))

        params = {
            "symbol": symbol,
            "side": side,
            "type": "LIMIT",
            "quantity": quantity,
            "price": price,
            "timeInForce": "GTC"  # Good Till Cancel
        }

        if reduce_only:
            params["reduceOnly"] = "true"

        logger.warning(f"[ORDEM LIMITE] {side} {quantity} {symbol} @ ${price}")
        result = await self._request("POST", "/fapi/v1/order", params, signed=True)

        # Log da ordem
        self._log_order(symbol, "LIMIT", side, quantity, result, price)

        return result

    async def place_stop_loss(
        self,
        symbol: str,
        side: str,  # Lado oposto da posição
        quantity: float,
        stop_price: float,
        entry_price: float = None
    ) -> Dict[str, Any]:
        """
        Coloca Stop Loss via Algo Order API (/fapi/v1/algoOrder).
        Binance exige endpoint de Algo para STOP_MARKET (erro -4120 no endpoint antigo).
        Inclui validação para evitar erro -2021 'Order would immediately trigger'.
        """
        symbol_info = await self.get_symbol_info(symbol)
        if symbol_info:
            quantity = self._round_quantity(quantity, symbol_info.get("quantity_precision", 3))
            stop_price = self._round_price(stop_price, symbol_info.get("price_precision", 2))

        # Validação: SL não pode disparar imediatamente
        # Para SELL SL (posição LONG): stop_price deve ser ABAIXO do entry
        # Para BUY SL (posição SHORT): stop_price deve ser ACIMA do entry
        if entry_price and entry_price > 0:
            if side == "SELL" and stop_price >= entry_price:
                # SL de posição LONG está acima do entry — dispararia imediatamente
                corrected_sl = entry_price * 0.985  # 1.5% abaixo do entry
                logger.warning(
                    f"[SL FIX] {symbol} SL SELL ${stop_price:.4f} >= entry ${entry_price:.4f} "
                    f"(dispararia imediatamente). Corrigido para ${corrected_sl:.4f}"
                )
                stop_price = self._round_price(corrected_sl, symbol_info.get("price_precision", 2)) if symbol_info else round(corrected_sl, 2)
            elif side == "BUY" and stop_price <= entry_price:
                # SL de posição SHORT está abaixo do entry — dispararia imediatamente
                corrected_sl = entry_price * 1.015  # 1.5% acima do entry
                logger.warning(
                    f"[SL FIX] {symbol} SL BUY ${stop_price:.4f} <= entry ${entry_price:.4f} "
                    f"(dispararia imediatamente). Corrigido para ${corrected_sl:.4f}"
                )
                stop_price = self._round_price(corrected_sl, symbol_info.get("price_precision", 2)) if symbol_info else round(corrected_sl, 2)

        params = {
            "algoType": "CONDITIONAL",
            "symbol": symbol,
            "side": side,
            "type": "STOP_MARKET",
            "quantity": quantity,
            "triggerPrice": stop_price,
            "reduceOnly": "true",
            "workingType": "CONTRACT_PRICE"
        }

        logger.warning(f"[STOP LOSS] {side} {quantity} {symbol} @ trigger ${stop_price}")
        result = await self._request("POST", "/fapi/v1/algoOrder", params, signed=True)

        self._log_order(symbol, "STOP_LOSS", side, quantity, result, stop_price)

        return result

    async def place_take_profit(
        self,
        symbol: str,
        side: str,
        quantity: float,
        take_profit_price: float
    ) -> Dict[str, Any]:
        """
        Coloca Take Profit via Algo Order API (/fapi/v1/algoOrder).
        Binance exige endpoint de Algo para TAKE_PROFIT_MARKET (erro -4120 no endpoint antigo).
        """
        symbol_info = await self.get_symbol_info(symbol)
        if symbol_info:
            quantity = self._round_quantity(quantity, symbol_info.get("quantity_precision", 3))
            take_profit_price = self._round_price(take_profit_price, symbol_info.get("price_precision", 2))

        params = {
            "algoType": "CONDITIONAL",
            "symbol": symbol,
            "side": side,
            "type": "TAKE_PROFIT_MARKET",
            "quantity": quantity,
            "triggerPrice": take_profit_price,
            "reduceOnly": "true",
            "workingType": "CONTRACT_PRICE"
        }

        logger.warning(f"[TAKE PROFIT] {side} {quantity} {symbol} @ trigger ${take_profit_price}")
        result = await self._request("POST", "/fapi/v1/algoOrder", params, signed=True)

        self._log_order(symbol, "TAKE_PROFIT", side, quantity, result, take_profit_price)

        return result

    async def _get_algo_orders(self, symbol: str) -> List[Dict]:
        """Lista ordens algo (SL/TP) ativas para um símbolo."""
        result = await self._request("GET", "/fapi/v1/allAlgoOrders", {"symbol": symbol, "limit": 100}, signed=True)
        if isinstance(result, list):
            return [o for o in result if o.get("algoStatus") == "NEW"]
        return []

    async def cancel_all_orders(self, symbol: str) -> Dict[str, Any]:
        """Cancela todas as ordens abertas de um símbolo (limit, market + SL/TP via Algo API)."""
        # 1) Ordens regulares (limit, etc.)
        result = await self._request("DELETE", "/fapi/v1/allOpenOrders", {"symbol": symbol}, signed=True)
        if isinstance(result, dict) and "error" in result:
            return result
        # 2) Ordens algo (Stop Loss / Take Profit)
        algo_orders = await self._get_algo_orders(symbol)
        for order in algo_orders:
            algo_id = order.get("algoId")
            if algo_id is not None:
                await self._request("DELETE", "/fapi/v1/algoOrder", {"algoId": algo_id}, signed=True)
        if not (isinstance(result, dict) and "error" in result):
            logger.info(f"[CANCEL] Todas as ordens de {symbol} canceladas (incl. algo SL/TP)")
        return result

    async def close_position(self, symbol: str) -> Dict[str, Any]:
        """
        Fecha posição aberta e cancela todas as ordens pendentes (SL/TP).

        Args:
            symbol: Símbolo para fechar

        Returns:
            Resultado do fechamento
        """
        # Obter posição atual
        position = await self.get_position(symbol)

        if not position:
            return {"success": False, "error": "Nenhuma posição aberta para fechar"}

        if "error" in position:
            return position

        # Determinar lado oposto para fechar
        close_side = "SELL" if position["side"] == "LONG" else "BUY"
        quantity = abs(position["position_amt"])

        # Fechar com ordem de mercado
        result = await self.place_market_order(symbol, close_side, quantity, reduce_only=True)

        # CORRIGIDO: Cancelar TODAS as ordens pendentes (SL, TP1, TP2) após fechar posição
        try:
            await self.cancel_all_orders(symbol)
            logger.info(f"[CLOSE] Ordens pendentes de {symbol} canceladas apos fechamento")
        except Exception as e:
            logger.warning(f"[CLOSE] Erro ao cancelar ordens de {symbol}: {e}")

        return result

    async def execute_signal(self, signal: Dict[str, Any], position_size: float = None) -> Dict[str, Any]:
        """
        Executa um sinal de trading completo.

        Processo:
        1. Verifica saldo disponível
        2. Configura alavancagem
        3. Abre posição com ordem de mercado
        4. Coloca Stop Loss
        5. Coloca Take Profit (50% em TP1, 50% em TP2)

        Args:
            signal: Sinal de trading com entry_price, stop_loss, take_profit_1, take_profit_2
            position_size: Tamanho da posição (quantidade)

        Returns:
            Resultado da execução
        """
        symbol = signal.get("symbol")
        signal_type = signal.get("signal")  # BUY ou SELL
        entry_price = signal.get("entry_price")
        stop_loss = signal.get("stop_loss")
        take_profit_1 = signal.get("take_profit_1")
        take_profit_2 = signal.get("take_profit_2")
        source = signal.get("source", "UNKNOWN")

        logger.warning("="*60)
        logger.warning(f"[EXECUTANDO SINAL REAL] {signal_type} {symbol}")
        logger.warning(f"Fonte: {source}")
        logger.warning(f"Entry: ${entry_price:.2f}")
        # HARD BLOCK: SL, TP1 e TP2 são OBRIGATÓRIOS para execução real
        if not stop_loss or stop_loss <= 0:
            return {"success": False, "error": f"Stop Loss obrigatório. Valor recebido: {stop_loss}"}
        if not take_profit_1 or take_profit_1 <= 0:
            return {"success": False, "error": f"Take Profit 1 obrigatório. Valor recebido: {take_profit_1}"}
        if not take_profit_2 or take_profit_2 <= 0:
            return {"success": False, "error": f"Take Profit 2 obrigatório. Valor recebido: {take_profit_2}"}

        logger.warning(f"Stop Loss: ${stop_loss:.2f}")
        logger.warning(f"Take Profit 1: ${take_profit_1:.2f}")
        logger.warning(f"Take Profit 2: ${take_profit_2:.2f}")
        logger.warning("="*60)

        # PROTEÇÃO: Garantir distância mínima do SL (1.5% do entry)
        # SL muito apertado → alavancagem absurda → liquidação por ruído
        sl_distance_pct = abs(entry_price - stop_loss) / entry_price * 100
        MIN_SL_DISTANCE_PCT = 1.5  # Mínimo 1.5% de distância

        if sl_distance_pct < MIN_SL_DISTANCE_PCT:
            old_sl = stop_loss
            if signal_type == "BUY":
                stop_loss = entry_price * (1 - MIN_SL_DISTANCE_PCT / 100)
            else:  # SELL
                stop_loss = entry_price * (1 + MIN_SL_DISTANCE_PCT / 100)
            logger.warning(
                f"[SL AJUSTADO] Distância original {sl_distance_pct:.2f}% muito apertada "
                f"(mín {MIN_SL_DISTANCE_PCT}%). SL: ${old_sl:.4f} → ${stop_loss:.4f}"
            )
            # Atualizar no sinal para consistência
            signal["stop_loss"] = stop_loss

        try:
            # 1. Verificar saldo
            balance = await self.get_balance()
            if "error" in balance:
                return {"success": False, "error": f"Erro ao obter saldo: {balance['error']}"}

            available = balance.get("available_balance", 0)
            total_balance = balance.get("total_balance", available)
            logger.info(f"[SALDO] Total: ${total_balance:.2f} | Disponivel: ${available:.2f} USDT")

            # 2. Verificar se já existe posição para este símbolo
            existing_position = await self.get_position(symbol)
            if existing_position and "position_amt" in existing_position:
                return {
                    "success": False,
                    "error": f"Ja existe posicao aberta para {symbol}: {existing_position['side']} {abs(existing_position['position_amt'])}"
                }

            # 2a. GUARDA DIRECIONAL: máximo 2 posições na mesma direção
            # Evita stop em massa quando mercado reverte contra posições correlacionadas
            all_positions = await self.get_all_positions()
            if all_positions:
                signal_side = signal.get("signal", "").upper()
                # Na Binance, side SHORT = positionAmt negativo, LONG = positivo
                if signal_side == "SELL":
                    same_dir = [p for p in all_positions if float(p.get("positionAmt", p.get("position_amt", 0))) < 0]
                elif signal_side == "BUY":
                    same_dir = [p for p in all_positions if float(p.get("positionAmt", p.get("position_amt", 0))) > 0]
                else:
                    same_dir = []
                max_same_direction = 2
                if len(same_dir) >= max_same_direction:
                    dir_label = "SHORT" if signal_side == "SELL" else "LONG"
                    same_symbols = [p.get("symbol", "?") for p in same_dir]
                    logger.warning(
                        f"[GUARDA DIRECIONAL] {dir_label} {symbol} BLOQUEADO: já existem {len(same_dir)} "
                        f"{dir_label}s abertos ({', '.join(same_symbols)}). "
                        f"Máximo {max_same_direction} na mesma direção para evitar stop em massa."
                    )
                    return {
                        "success": False,
                        "error": f"Guarda direcional: já existem {len(same_dir)} {dir_label}s abertos "
                                 f"({', '.join(same_symbols)}). Máximo {max_same_direction} na mesma direção."
                    }

            # 2b. Cancelar TODAS as ordens pendentes (regulares + algo SL/TP) antes de abrir nova posição
            # Evita acúmulo/duplicação de ordens; get_open_orders só retorna regulares, então sempre cancelar
            try:
                await self.cancel_all_orders(symbol)
                logger.info(f"[LIMPEZA PRE-EXECUCAO] {symbol}: ordens limpas antes de abrir nova posicao")
            except Exception as e:
                logger.warning(f"[LIMPEZA PRE-EXECUCAO] Erro ao limpar ordens de {symbol}: {e}")

            # 3. Obter informações do símbolo
            symbol_info = await self.get_symbol_info(symbol)
            if not symbol_info or "error" in symbol_info:
                return {"success": False, "error": f"Erro ao obter info do simbolo: {symbol_info}"}

            # 4. Calcular risco baseado no SALDO DISPONIVEL
            # Usa o disponivel para evitar "Margin insufficient" da Binance
            # Se tem pouca margem livre, reduz o risco proporcionalmente
            capital_base = available  # Usar saldo DISPONIVEL, nao o total
            risk_percent = settings.risk_percent_per_trade / 100.0
            risk_amount = capital_base * risk_percent

            # Garantir risco minimo viavel (evitar posicoes microscopicas)
            if risk_amount < 1.0:
                logger.warning(f"[RISCO] Risco calculado muito baixo: ${risk_amount:.2f} (disponivel: ${available:.2f})")
                return {"success": False, "error": f"Saldo disponivel muito baixo: ${available:.2f}"}

            if position_size is None:
                if stop_loss and stop_loss != entry_price:
                    # Distancia do stop loss em $
                    stop_distance = abs(entry_price - stop_loss)

                    # Tamanho da posicao (em unidades do ativo)
                    # Se stop for atingido, perda = position_size * stop_distance = risk_amount
                    position_size = risk_amount / stop_distance

                    # Calcular valor total da posicao
                    position_value_calc = position_size * entry_price

                    # IMPORTANTE: A margem necessária em modo ISOLATED é apenas o risco (valor do stop)
                    # Não é o valor_posicao / alavancagem, é o próprio risk_amount + buffer
                    margin_required = risk_amount * 1.2  # 20% de buffer para taxas e slippage

                    logger.info(f"[RISCO] Disponivel: ${available:.2f} | Total: ${total_balance:.2f} | Risco: {settings.risk_percent_per_trade}% = ${risk_amount:.2f}")
                    logger.info(f"[POSICAO] Entry: ${entry_price:.4f} | Stop: ${stop_loss:.4f} | Distancia: ${stop_distance:.4f}")
                    logger.info(f"[POSICAO] Tamanho: {position_size:.4f} unidades | Valor: ${position_value_calc:.2f}")
                    logger.info(f"[MARGEM] Margem isolada necessaria: ${margin_required:.2f} (risco + buffer)")

                    # Verificar se tem margem disponível
                    if margin_required > available:
                        logger.warning(f"[MARGEM INSUFICIENTE] Necessario: ${margin_required:.2f} > Disponivel: ${available:.2f}")
                        return {
                            "success": False,
                            "error": f"Margem insuficiente. Disponivel: ${available:.2f}, Necessario: ${margin_required:.2f}"
                        }
                else:
                    # Fallback: usar 1% do saldo se nao tiver stop loss definido
                    position_value = available * 0.01
                    position_size = position_value / entry_price
                    logger.warning(f"[POSICAO FALLBACK] Sem stop loss valido, usando 1% do saldo: {position_size:.6f} unidades")

            # Arredondar para a precisão correta
            position_size = self._round_quantity(position_size, symbol_info.get("quantity_precision", 3))

            # Verificar quantidade mínima
            min_qty = symbol_info.get("min_qty", 0)
            if position_size < min_qty:
                return {
                    "success": False,
                    "error": f"Quantidade {position_size} menor que minimo {min_qty}"
                }

            # Verificar notional mínimo
            min_notional = symbol_info.get("min_notional", 0)
            notional = position_size * entry_price
            if notional < min_notional:
                return {
                    "success": False,
                    "error": f"Valor nocional ${notional:.2f} menor que minimo ${min_notional}"
                }

            # 5. Calcular e configurar alavancagem DINAMICA
            # Conceito: margem isolada = valor do risco (5% do capital)
            # alavancagem = valor_posicao / margem_desejada
            # A alavancagem é CONSEQUÊNCIA do cálculo, não um input.
            # O SL é a proteção real — a margem só precisa cobrir o risco.
            position_value = position_size * entry_price
            desired_margin = risk_amount * 1.2  # Margem = risco + 20% buffer para taxas/slippage

            if desired_margin > 0:
                calculated_leverage = int(position_value / desired_margin)
                # Cap de segurança: usar limite real do par (via exchangeInfo) ao invés de 125x fixo
                symbol_max_leverage = symbol_info.get("max_leverage", 20) if symbol_info else 20
                # Limite adicional de segurança: nunca passar de 50x independente do que Binance permite
                safe_max_leverage = min(symbol_max_leverage, 50)
                calculated_leverage = max(1, min(calculated_leverage, safe_max_leverage))
            else:
                calculated_leverage = self.default_leverage

            logger.info(f"[ALAVANCAGEM] Valor posicao: ${position_value:.2f} / Margem: ${desired_margin:.2f} = {calculated_leverage}x (max_par={symbol_info.get('max_leverage', '?') if symbol_info else '?'})")

            leverage_result = await self.set_leverage(symbol, calculated_leverage)
            if "error" in leverage_result:
                logger.warning(f"Erro ao configurar alavancagem: {leverage_result}")

            # 6. Configurar tipo de margem (ISOLATED para limitar perdas)
            margin_result = await self.set_margin_type(symbol, "ISOLATED")
            if isinstance(margin_result, dict) and "error" in margin_result:
                logger.warning(f"Erro ao configurar margem: {margin_result}")

            # 7. Abrir posição com ordem de mercado
            order_side = signal_type  # BUY ou SELL
            main_order = await self.place_market_order(symbol, order_side, position_size)

            if "error" in main_order or (isinstance(main_order, dict) and "orderId" not in main_order):
                return {
                    "success": False,
                    "error": f"Erro ao abrir posicao: {main_order}"
                }

            logger.info(f"[ORDEM PRINCIPAL] Executada: Order ID {main_order.get('orderId')}")

            # 8. Determinar lado oposto para Stop Loss e Take Profit
            close_side = "SELL" if signal_type == "BUY" else "BUY"

            def _order_id(r: dict) -> Optional[Any]:
                return r.get("orderId")

            # 9. Colocar Stop Loss (com entry_price para validação anti-trigger)
            sl_order = await self.place_stop_loss(symbol, close_side, position_size, stop_loss, entry_price=entry_price)
            if "error" in sl_order:
                logger.error(f"Erro ao colocar Stop Loss: {sl_order}")
            else:
                logger.info(f"[STOP LOSS] Colocado: ID {_order_id(sl_order)}")

            # 10. Colocar Take Profit 1 (50% da posição)
            tp1_size = self._round_quantity(position_size * 0.5, symbol_info.get("quantity_precision", 3))
            tp1_order = await self.place_take_profit(symbol, close_side, tp1_size, take_profit_1)
            if "error" in tp1_order:
                logger.error(f"Erro ao colocar Take Profit 1: {tp1_order}")
            else:
                logger.info(f"[TAKE PROFIT 1] Colocado: ID {_order_id(tp1_order)}")

            # 11. Colocar Take Profit 2 (50% restante)
            tp2_size = self._round_quantity(position_size - tp1_size, symbol_info.get("quantity_precision", 3))
            tp2_order = await self.place_take_profit(symbol, close_side, tp2_size, take_profit_2)
            if "error" in tp2_order:
                logger.error(f"Erro ao colocar Take Profit 2: {tp2_order}")
            else:
                logger.info(f"[TAKE PROFIT 2] Colocado: ID {_order_id(tp2_order)}")

            # 12. Registrar execução completa
            execution_record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "source": source,
                "signal": signal_type,
                "entry_price": entry_price,
                "position_size": position_size,
                "leverage": self.default_leverage,
                "stop_loss": stop_loss,
                "take_profit_1": take_profit_1,
                "take_profit_2": take_profit_2,
                "main_order_id": main_order.get("orderId"),
                "sl_order_id": _order_id(sl_order),
                "tp1_order_id": _order_id(tp1_order),
                "tp2_order_id": _order_id(tp2_order),
                "status": "OPEN"
            }

            # Salvar registro
            record_file = self.orders_dir / f"execution_{symbol}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
            with open(record_file, "w") as f:
                json.dump(execution_record, f, indent=2)

            return {
                "success": True,
                "message": f"Posicao {signal_type} aberta para {symbol} com SL e TP configurados",
                "order_id": main_order.get("orderId"),
                "position_size": position_size,
                "entry_price": float(main_order.get("avgPrice", entry_price)),
                "record_file": str(record_file)
            }

        except Exception as e:
            logger.exception(f"Erro ao executar sinal: {e}")
            return {
                "success": False,
                "error": f"Erro inesperado: {str(e)}"
            }

    def _log_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        quantity: float,
        result: Dict[str, Any],
        price: float = None
    ):
        """Registra ordem em arquivo de log"""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "order_type": order_type,
            "side": side,
            "quantity": quantity,
            "price": price,
            "result": result
        }

        log_file = self.orders_dir / f"orders_{datetime.now(timezone.utc).strftime('%Y%m%d')}.json"

        try:
            if log_file.exists():
                with open(log_file, "r") as f:
                    logs = json.load(f)
            else:
                logs = []

            logs.append(log_entry)

            with open(log_file, "w") as f:
                json.dump(logs, f, indent=2)

        except Exception as e:
            logger.error(f"Erro ao salvar log de ordem: {e}")

    async def get_open_orders(self, symbol: str = None) -> List[Dict[str, Any]]:
        """Obtém ordens abertas regulares (limit, etc.). Para SL/TP use get_open_algo_orders."""
        params = {}
        if symbol:
            params["symbol"] = symbol

        result = await self._request("GET", "/fapi/v1/openOrders", params, signed=True)
        if isinstance(result, dict) and "error" in result:
            return []
        return result

    async def get_open_algo_orders(self, symbol: str) -> List[Dict[str, Any]]:
        """Obtém ordens algo (SL/TP) ativas para um símbolo, em formato compatível com get_open_orders (campo type)."""
        raw = await self._request("GET", "/fapi/v1/allAlgoOrders", {"symbol": symbol, "limit": 100}, signed=True)
        if not isinstance(raw, list):
            return []
        out = []
        for o in raw:
            if o.get("algoStatus") != "NEW":
                continue
            out.append({
                "symbol": o.get("symbol", symbol),
                "type": o.get("orderType", ""),  # STOP_MARKET, TAKE_PROFIT_MARKET, etc.
                "side": o.get("side", ""),
                "origQty": o.get("quantity", 0),
                "stopPrice": o.get("triggerPrice", ""),
                "algoId": o.get("algoId"),  # para cancelamento via Algo API
                "time": o.get("createTime", 0),
            })
        return out

    async def cancel_algo_order(self, algo_id: int) -> Dict[str, Any]:
        """Cancela uma ordem algo (SL/TP) pelo algoId."""
        result = await self._request(
            "DELETE", "/fapi/v1/algoOrder",
            {"algoId": algo_id},
            signed=True
        )
        return result

    async def get_all_positions(self) -> List[Dict[str, Any]]:
        """Obtém todas as posições abertas"""
        positions = await self._request("GET", "/fapi/v2/positionRisk", signed=True)

        if isinstance(positions, dict) and "error" in positions:
            return [positions]

        # Filtrar apenas posições com quantidade != 0
        open_positions = []
        for pos in positions:
            if float(pos.get("positionAmt", 0)) != 0:
                open_positions.append({
                    "symbol": pos["symbol"],
                    "side": "LONG" if float(pos["positionAmt"]) > 0 else "SHORT",
                    "position_amt": float(pos["positionAmt"]),
                    "entry_price": float(pos["entryPrice"]),
                    "unrealized_pnl": float(pos["unRealizedProfit"]),
                    "leverage": int(pos["leverage"]),
                    "liquidation_price": float(pos["liquidationPrice"])
                })

        return open_positions


# Instância singleton (inicializada apenas quando necessário)
_executor_instance = None

def get_executor() -> BinanceFuturesExecutor:
    """Retorna instância singleton do executor"""
    global _executor_instance
    if _executor_instance is None:
        _executor_instance = BinanceFuturesExecutor()
    return _executor_instance
