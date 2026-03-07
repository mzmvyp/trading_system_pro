"""
Orphan Order Cleaner - Sistema de Limpeza de Ordens Órfãs
==========================================================

Detecta e cancela ordens abertas que não têm posição correspondente.
Isso acontece quando:
- Uma posição é fechada manualmente
- O TP1 é atingido mas TP2 e SL continuam abertos
- Erros de execução deixam ordens pendentes

Autor: Trading Bot
Data: 2026-01-14
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Any, Set
from src.core.logger import get_logger

logger = get_logger(__name__)


class OrphanOrderCleaner:
    """Limpa ordens órfãs (ordens sem posição correspondente)"""
    
    def __init__(self):
        self.last_cleanup = None
        self.cleanup_count = 0
        self.orders_cancelled = 0
        
    async def cleanup_orphan_orders(self, executor=None) -> Dict[str, Any]:
        """
        Limpa ordens órfãs da conta.
        
        Args:
            executor: BinanceFuturesExecutor instance (opcional, cria um novo se não fornecido)
            
        Returns:
            Dict com resultado da limpeza
        """
        print("\n" + "="*60)
        print("[LIMPEZA] Verificando ordens orfas...")
        print("="*60)
        
        try:
            # Criar executor se não fornecido
            if executor is None:
                from src.exchange.executor import BinanceFuturesExecutor
                executor = BinanceFuturesExecutor()
            
            # 1. Obter todas as posições abertas
            positions = await executor.get_all_positions()
            
            if isinstance(positions, list) and len(positions) > 0 and "error" in positions[0]:
                logger.error(f"[LIMPEZA] Erro ao obter posicoes: {positions[0].get('error')}")
                return {"success": False, "error": positions[0].get("error")}
            
            # Criar set de símbolos com posição aberta
            symbols_with_position: Set[str] = set()
            for pos in positions:
                if isinstance(pos, dict) and pos.get("symbol"):
                    symbols_with_position.add(pos["symbol"])
                    
            print(f"[LIMPEZA] Posicoes abertas: {len(symbols_with_position)}")
            for sym in symbols_with_position:
                pos = next((p for p in positions if p.get("symbol") == sym), {})
                print(f"   - {sym}: {pos.get('side', 'N/A')} {abs(pos.get('position_amt', 0)):.4f}")
            
            # 2. Obter todas as ordens abertas
            all_orders = await executor.get_open_orders()
            
            if isinstance(all_orders, dict) and "error" in all_orders:
                logger.error(f"[LIMPEZA] Erro ao obter ordens: {all_orders.get('error')}")
                return {"success": False, "error": all_orders.get("error")}
            
            print(f"[LIMPEZA] Ordens abertas: {len(all_orders)}")
            
            # 3. Identificar ordens órfãs (ordens de símbolos sem posição)
            orphan_orders = []
            valid_orders = []
            
            for order in all_orders:
                order_symbol = order.get("symbol", "")
                order_type = order.get("type", "")
                order_side = order.get("side", "")
                order_id = order.get("orderId", "")
                
                if order_symbol not in symbols_with_position:
                    # Órfã: ordem para símbolo sem posição
                    orphan_orders.append(order)
                    print(f"   [ORFA] {order_symbol} {order_type} {order_side} (ID: {order_id})")
                else:
                    valid_orders.append(order)
                    
            print(f"\n[LIMPEZA] Ordens orfas encontradas: {len(orphan_orders)}")
            print(f"[LIMPEZA] Ordens validas: {len(valid_orders)}")
            
            # 4. Cancelar ordens órfãs
            cancelled_orders = []
            failed_cancellations = []
            
            if orphan_orders:
                # Agrupar por símbolo para cancelamento em lote
                symbols_to_cancel = set(o.get("symbol") for o in orphan_orders)
                
                for symbol in symbols_to_cancel:
                    try:
                        result = await executor.cancel_all_orders(symbol)
                        
                        if isinstance(result, dict) and "error" in result:
                            failed_cancellations.append({
                                "symbol": symbol,
                                "error": result.get("error")
                            })
                            print(f"   [ERRO] Falha ao cancelar ordens de {symbol}: {result.get('error')}")
                        else:
                            # Contar quantas ordens foram canceladas deste símbolo
                            symbol_orders = [o for o in orphan_orders if o.get("symbol") == symbol]
                            cancelled_orders.extend(symbol_orders)
                            print(f"   [OK] {len(symbol_orders)} ordem(s) cancelada(s) para {symbol}")
                            
                    except Exception as e:
                        failed_cancellations.append({
                            "symbol": symbol,
                            "error": str(e)
                        })
                        logger.error(f"[LIMPEZA] Erro ao cancelar ordens de {symbol}: {e}")
                        
            # 5. Atualizar estatísticas
            self.last_cleanup = datetime.now()
            self.cleanup_count += 1
            self.orders_cancelled += len(cancelled_orders)
            
            # 6. Resultado
            result = {
                "success": True,
                "timestamp": self.last_cleanup.isoformat(),
                "positions_found": len(symbols_with_position),
                "orders_found": len(all_orders),
                "orphan_orders_found": len(orphan_orders),
                "orders_cancelled": len(cancelled_orders),
                "failed_cancellations": len(failed_cancellations),
                "total_cleanups": self.cleanup_count,
                "total_orders_cancelled": self.orders_cancelled
            }
            
            print("\n" + "-"*60)
            if cancelled_orders:
                print(f"[LIMPEZA] CONCLUIDO: {len(cancelled_orders)} ordens orfas canceladas!")
            else:
                print(f"[LIMPEZA] CONCLUIDO: Nenhuma ordem orfa encontrada.")
            print("-"*60 + "\n")
            
            logger.info(f"[LIMPEZA] Resultado: {len(cancelled_orders)} canceladas, {len(failed_cancellations)} falhas")
            
            return result
            
        except Exception as e:
            logger.exception(f"[LIMPEZA] Erro durante limpeza: {e}")
            return {"success": False, "error": str(e)}
            
    async def cleanup_specific_symbol(self, symbol: str, executor=None) -> Dict[str, Any]:
        """
        Cancela todas as ordens de um símbolo específico.
        
        Args:
            symbol: Símbolo para limpar ordens
            executor: BinanceFuturesExecutor instance
            
        Returns:
            Dict com resultado
        """
        try:
            if executor is None:
                from src.exchange.executor import BinanceFuturesExecutor
                executor = BinanceFuturesExecutor()
                
            result = await executor.cancel_all_orders(symbol)
            
            if isinstance(result, dict) and "error" in result:
                return {"success": False, "symbol": symbol, "error": result.get("error")}
                
            logger.info(f"[LIMPEZA] Ordens de {symbol} canceladas")
            return {"success": True, "symbol": symbol, "message": "Ordens canceladas"}
            
        except Exception as e:
            return {"success": False, "symbol": symbol, "error": str(e)}
            
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de limpeza"""
        return {
            "last_cleanup": self.last_cleanup.isoformat() if self.last_cleanup else None,
            "total_cleanups": self.cleanup_count,
            "total_orders_cancelled": self.orders_cancelled
        }


# Instância global
_cleaner_instance = None

def get_cleaner() -> OrphanOrderCleaner:
    """Retorna instância singleton do cleaner"""
    global _cleaner_instance
    if _cleaner_instance is None:
        _cleaner_instance = OrphanOrderCleaner()
    return _cleaner_instance


async def cleanup_orphan_orders() -> Dict[str, Any]:
    """Função helper para limpeza de ordens órfãs"""
    cleaner = get_cleaner()
    return await cleaner.cleanup_orphan_orders()


async def cleanup_symbol_orders(symbol: str) -> Dict[str, Any]:
    """Função helper para limpar ordens de um símbolo"""
    cleaner = get_cleaner()
    return await cleaner.cleanup_specific_symbol(symbol)


if __name__ == "__main__":
    # Teste direto
    print("\n" + "="*60)
    print("ORPHAN ORDER CLEANER - TESTE")
    print("="*60)
    
    async def test():
        result = await cleanup_orphan_orders()
        print(f"\nResultado: {result}")
        
    asyncio.run(test())

