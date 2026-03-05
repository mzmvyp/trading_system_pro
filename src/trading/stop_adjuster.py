"""
Sistema de Ajuste de Stop Loss após TP1
Quando o Take Profit 1 é atingido, pede ao DeepSeek um novo stop loss baseado em análise técnica.

Funcionalidades:
- Monitora posições que atingiram TP1
- Solicita ao DeepSeek um novo nível de stop baseado em suportes/resistências
- Move o stop loss para o novo nível recomendado
"""

import json
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

from src.core.logger import get_logger
from src.core.config import settings

logger = get_logger(__name__)


class StopAdjuster:
    """
    Ajusta stop loss após TP1 usando análise técnica do DeepSeek.
    """

    def __init__(self):
        self.adjusted_positions = {}  # {symbol: last_adjustment_time}
        Path("stop_adjustment_logs").mkdir(exist_ok=True)
        logger.info("[STOP ADJUSTER] Inicializado")

    async def get_new_stop_from_deepseek(
        self,
        symbol: str,
        side: str,  # "LONG" ou "SHORT"
        entry_price: float,
        current_price: float,
        original_stop: float,
        tp1_price: float
    ) -> Dict[str, Any]:
        """
        Solicita ao DeepSeek um novo nível de stop loss após TP1 ser atingido.

        Args:
            symbol: Par de trading (ex: BTCUSDT)
            side: Direção da posição (LONG ou SHORT)
            entry_price: Preço de entrada original
            current_price: Preço atual do mercado
            original_stop: Stop loss original
            tp1_price: Preço do TP1 que foi atingido

        Returns:
            Dict com novo stop recomendado e razão
        """
        try:
            from src.exchange.client import BinanceClient

            # Buscar dados de mercado atuais
            async with BinanceClient() as client:
                klines_1h = await client.get_klines(symbol, "1h", limit=50)
                klines_4h = await client.get_klines(symbol, "4h", limit=30)

            if klines_1h.empty:
                return {"error": "Dados de mercado não disponíveis"}

            # Calcular suportes/resistências simples
            recent_lows_1h = klines_1h['low'].tail(20).tolist()
            recent_highs_1h = klines_1h['high'].tail(20).tolist()

            prompt = f"""## AJUSTE DE STOP LOSS APÓS TP1

**Situação**: O Take Profit 1 foi atingido. Preciso de um NOVO STOP LOSS para proteger lucros.

### Dados da Posição
- **Par**: {symbol}
- **Direção**: {side}
- **Entrada**: ${entry_price:.4f}
- **Preço Atual**: ${current_price:.4f}
- **Stop Original**: ${original_stop:.4f}
- **TP1 Atingido**: ${tp1_price:.4f}
- **Lucro Atual**: {((current_price - entry_price) / entry_price * 100) if side == "LONG" else ((entry_price - current_price) / entry_price * 100):.2f}%

### Dados Técnicos (1H)
- Últimas mínimas: {[f"${x:.4f}" for x in recent_lows_1h[-5:]]}
- Últimas máximas: {[f"${x:.4f}" for x in recent_highs_1h[-5:]]}

### Regras para Novo Stop
1. **DEVE** estar acima do preço de entrada (para LONG) ou abaixo (para SHORT) - proteger lucro
2. **DEVE** considerar suportes/resistências recentes
3. **NÃO PODE** ser muito apertado (dar espaço para volatilidade normal)
4. **OBJETIVO**: Proteger a maior parte do lucro sem ser stopado por ruído

### Responda APENAS com JSON:
```json
{{
    "new_stop_loss": <preço do novo stop>,
    "reasoning": "<explicação técnica breve>",
    "support_level": <nível de suporte identificado>,
    "confidence": <1-10>
}}
```
"""

            # Chamar DeepSeek
            from src.prompts.deepseek_client import DeepSeekClient

            async with DeepSeekClient() as ds_client:
                response = await ds_client.chat(prompt)

            # Extrair JSON da resposta
            result = self._extract_json(response)

            if result and "new_stop_loss" in result:
                new_stop = float(result["new_stop_loss"])

                # Validar o novo stop
                if side == "LONG":
                    # Para LONG: novo stop deve estar acima da entrada
                    if new_stop <= entry_price:
                        # Se DeepSeek sugeriu abaixo da entrada, usar entrada + pequena margem
                        new_stop = entry_price * 1.001  # 0.1% acima da entrada
                        result["reasoning"] = f"Ajustado para breakeven+ (DeepSeek sugeriu ${result['new_stop_loss']:.4f} abaixo da entrada)"
                        result["new_stop_loss"] = new_stop
                else:
                    # Para SHORT: novo stop deve estar abaixo da entrada
                    if new_stop >= entry_price:
                        new_stop = entry_price * 0.999
                        result["reasoning"] = f"Ajustado para breakeven+ (DeepSeek sugeriu ${result['new_stop_loss']:.4f} acima da entrada)"
                        result["new_stop_loss"] = new_stop

                result["success"] = True
                return result
            else:
                return {"error": "Resposta do DeepSeek não contém stop válido", "raw_response": response[:500]}

        except Exception as e:
            logger.exception(f"[STOP ADJUSTER] Erro ao consultar DeepSeek: {e}")
            return {"error": str(e)}

    def _extract_json(self, text: str) -> Optional[Dict]:
        """Extrai JSON de uma string de texto"""
        try:
            # Tentar encontrar JSON no texto
            import re
            json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return None
        except:
            return None

    async def adjust_stop_after_tp1(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        current_price: float,
        original_stop: float,
        tp1_price: float,
        position_size: float
    ) -> Dict[str, Any]:
        """
        Executa o ajuste de stop após TP1 ser atingido.

        Returns:
            Dict com resultado da operação
        """
        logger.info(f"[STOP ADJUSTER] Iniciando ajuste de stop para {symbol} após TP1...")

        # 1. Obter recomendação do DeepSeek
        recommendation = await self.get_new_stop_from_deepseek(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            current_price=current_price,
            original_stop=original_stop,
            tp1_price=tp1_price
        )

        if "error" in recommendation:
            logger.error(f"[STOP ADJUSTER] Erro ao obter recomendação: {recommendation['error']}")
            # Fallback: usar breakeven + 0.5%
            if side == "LONG":
                new_stop = entry_price * 1.005
            else:
                new_stop = entry_price * 0.995
            recommendation = {
                "new_stop_loss": new_stop,
                "reasoning": "Fallback para breakeven+0.5% devido a erro no DeepSeek",
                "confidence": 5
            }

        new_stop = recommendation.get("new_stop_loss")

        # 2. Mover o stop loss
        try:
            from src.exchange.executor import BinanceFuturesExecutor

            async with BinanceFuturesExecutor() as executor:
                # Primeiro, cancelar ordens de stop existentes
                await executor.cancel_all_orders(symbol)

                # Colocar novo stop
                close_side = "SELL" if side == "LONG" else "BUY"
                result = await executor.place_stop_loss(
                    symbol=symbol,
                    side=close_side,
                    quantity=position_size,
                    stop_price=new_stop
                )

                if "error" not in result:
                    logger.info(f"[STOP ADJUSTER] ✅ Stop movido para ${new_stop:.4f} | Razão: {recommendation.get('reasoning', 'N/A')}")

                    # Salvar log
                    self._save_adjustment_log(symbol, recommendation, result)

                    return {
                        "success": True,
                        "symbol": symbol,
                        "old_stop": original_stop,
                        "new_stop": new_stop,
                        "reasoning": recommendation.get("reasoning"),
                        "confidence": recommendation.get("confidence")
                    }
                else:
                    logger.error(f"[STOP ADJUSTER] Erro ao mover stop: {result}")
                    return {"success": False, "error": result.get("error")}

        except Exception as e:
            logger.exception(f"[STOP ADJUSTER] Erro ao executar ajuste: {e}")
            return {"success": False, "error": str(e)}

    def _save_adjustment_log(self, symbol: str, recommendation: Dict, result: Dict):
        """Salva log do ajuste"""
        try:
            log_file = Path("stop_adjustment_logs") / f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            log_data = {
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "recommendation": recommendation,
                "result": result
            }
            with open(log_file, "w") as f:
                json.dump(log_data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"[STOP ADJUSTER] Erro ao salvar log: {e}")


# Instância global
stop_adjuster = StopAdjuster()


async def adjust_stop_on_tp1_hit(
    symbol: str,
    side: str,
    entry_price: float,
    current_price: float,
    original_stop: float,
    tp1_price: float,
    position_size: float
) -> Dict[str, Any]:
    """
    Função helper para ajustar stop quando TP1 é atingido.

    Uso:
        result = await adjust_stop_on_tp1_hit(
            symbol="BTCUSDT",
            side="LONG",
            entry_price=95000,
            current_price=96500,
            original_stop=93000,
            tp1_price=96000,
            position_size=0.01
        )
    """
    return await stop_adjuster.adjust_stop_after_tp1(
        symbol=symbol,
        side=side,
        entry_price=entry_price,
        current_price=current_price,
        original_stop=original_stop,
        tp1_price=tp1_price,
        position_size=position_size
    )
