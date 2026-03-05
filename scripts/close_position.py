#!/usr/bin/env python3
"""
Fecha posição aberta na Binance Futures (testnet ou produção).
Uso: python scripts/close_position.py [SYMBOL]
Lê variáveis do .env (BINANCE_API_KEY, BINANCE_SECRET_KEY, BINANCE_TESTNET).
"""
import asyncio
import os
import sys
from pathlib import Path

# Garantir que o projeto está no path
project_root = Path(__file__).resolve().parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")


async def main():
    symbol = (sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT").strip().upper()
    print(f"[CLOSE] Conectando na Binance (testnet={os.getenv('BINANCE_TESTNET', 'false')})...")
    print(f"[CLOSE] Símbolo: {symbol}")

    from src.exchange.executor import BinanceFuturesExecutor
    executor = BinanceFuturesExecutor()

    position = await executor.get_position(symbol)
    if not position or "error" in position:
        if isinstance(position, dict) and "error" in position:
            print(f"[CLOSE] Erro: {position['error']}")
        else:
            print(f"[CLOSE] Nenhuma posição aberta para {symbol}.")
        return

    side = position.get("side", "")
    amt = abs(position.get("position_amt", 0))
    print(f"[CLOSE] Posição encontrada: {side} {amt} {symbol}. Fechando...")

    result = await executor.close_position(symbol)
    if isinstance(result, dict) and result.get("error"):
        print(f"[CLOSE] Falha: {result['error']}")
        sys.exit(1)
    print(f"[CLOSE] Posição fechada com sucesso.")
    if isinstance(result, dict) and result.get("orderId"):
        print(f"[CLOSE] Order ID: {result['orderId']}")


if __name__ == "__main__":
    asyncio.run(main())
