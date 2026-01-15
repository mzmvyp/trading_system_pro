"""
Sistema de Trading com AGNO Agent
Updated with logging and improved error handling
"""
import asyncio
import argparse
import sys
import json
import os
from pathlib import Path
from trading_agent_agno import AgnoTradingAgent
from logger import get_logger

logger = get_logger(__name__)

def get_active_positions():
    """Retorna símbolos com posições ativas (CORRIGIDO: verifica BUY e SELL)"""
    active_symbols = []

    try:
        if os.path.exists("portfolio/state.json"):
            with open("portfolio/state.json", "r", encoding='utf-8') as f:
                state = json.load(f)
                positions = state.get("positions", {})
                
                # MODIFICADO: Verificar posições considerando novas chaves (SYMBOL_DEEPSEEK, SYMBOL_AGNO, etc.)
                for key, pos in positions.items():
                    if pos.get("status") == "OPEN":
                        symbol = pos.get("symbol")
                        if symbol:
                            # Extrair símbolo base da chave (pode ser SYMBOL_DEEPSEEK, SYMBOL_AGNO, SYMBOL_DEEPSEEK_SHORT, etc.)
                            base_symbol = symbol  # O símbolo já está limpo na posição
                            if base_symbol not in active_symbols:
                                active_symbols.append(base_symbol)
                
                logger.debug(f"Loaded {len(active_symbols)} active positions")
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Erro ao carregar posicoes ativas: {e}")
    except Exception as e:
        logger.exception(f"Erro inesperado ao carregar posicoes ativas: {e}")

    return active_symbols

async def main():
    parser = argparse.ArgumentParser(
        description='Sistema de Trading de Criptomoedas com AGNO Agent'
    )
    parser.add_argument(
        '--symbol', 
        default='BTCUSDT',
        help='Símbolo para trading (ex: BTCUSDT)'
    )
    parser.add_argument(
        '--mode',
        choices=['single', 'monitor', 'top5', 'top10'],
        default='single',
        help='Modo de operação'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=300,
        help='Intervalo para monitoramento em segundos'
    )
    parser.add_argument(
        '--paper',
        action='store_true',
        default=True,
        help='Usar paper trading (simulado)'
    )
    
    args = parser.parse_args()

    # Banner
    print("\n" + "="*60)
    print("SISTEMA DE TRADING COM AGNO AGENT")
    print("="*60)
    print(f"Simbolo: {args.symbol}")
    print(f"Modo: {args.mode}")
    print(f"Paper Trading: {'Sim' if args.paper else 'Nao'}")
    print("="*60)

    logger.info(f"Starting trading system - Mode: {args.mode}, Symbol: {args.symbol}, Paper: {args.paper}")

    # Criar agent
    agent = AgnoTradingAgent(paper_trading=args.paper)
    
    try:
        if args.mode == 'single':
            # Análise única
            signal = await agent.analyze(args.symbol)
            
            if signal.get('signal') in ['BUY', 'SELL'] and signal.get('confidence', 0) >= 7:
                print("\n[ALERTA] Sinal forte detectado!")
                print("Considere executar o trade com cautela.")
        
        elif args.mode == 'monitor':
            # Monitoramento contínuo do Top 10 - SEM dependência do real_paper_trading
            from config import settings
            
            symbols = settings.top_crypto_pairs  # Todos os pares configurados
            
            print(f"\n[MONITOR] Monitoramento continuo dos pares configurados")
            print(f"Pares: {symbols}")
            print(f"Intervalo: {args.interval}s")
            print(f"Modo: {settings.trading_mode.upper()}")
            print("="*60)
            
            # Rastrear posições anteriores para detectar fechamentos
            previous_positions = set()
            
            while True:
                try:
                    # Verificar posições ativas (via arquivo state.json)
                    active_positions = get_active_positions()
                    active_positions_set = set(active_positions)
                    
                    # Detectar posições fechadas (estavam ativas antes, mas não estão mais)
                    closed_positions = previous_positions - active_positions_set
                    if closed_positions:
                        print(f"\n[POSICAO FECHADA] Detectado fechamento: {closed_positions}")
                        print("[GERANDO NOVO SINAL] Analisando para gerar novo sinal...")
                    
                    print(f"\n[POSICOES] Posicoes ativas: {active_positions if active_positions else 'Nenhuma'}")
                    print(f"[MODO] Trading Mode: {settings.trading_mode.upper()}")
                    
                    # Filtrar apenas símbolos sem posição ativa
                    symbols_to_analyze = [s for s in symbols if s not in active_positions]
                    
                    if not symbols_to_analyze:
                        print("\n[OK] Todos os pares tem posicoes ativas. Aguardando...")
                    else:
                        print(f"\n[ANALISE] Analisando {len(symbols_to_analyze)} pares sem posicoes ativas...")
                        
                        for symbol in symbols_to_analyze:
                            try:
                                await agent.analyze(symbol)
                            except Exception as e:
                                logger.error(f"Erro ao analisar {symbol}: {e}")
                                print(f"[ERRO] Erro em {symbol}: {e}")

                            await asyncio.sleep(3)  # Pausa entre análises
                    
                    # Atualizar rastreamento de posições
                    previous_positions = active_positions_set
                    
                    print(f"\n[AGUARDANDO] Aguardando {args.interval}s...")
                    await asyncio.sleep(args.interval)
                    
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    logger.exception(f"Erro no ciclo de monitoramento: {e}")
                    print(f"[ERRO] Erro no ciclo de monitoramento: {e}")
                    await asyncio.sleep(30)
        
        elif args.mode == 'top5':
            # Todos os pares configurados
            from config import settings
            symbols = settings.top_crypto_pairs  # Todos os pares configurados
            
            print(f"\n[TOP] Analisando {len(symbols)} pares configurados...")
            print("Pares configurados em config.py")
            print("="*60)
            
            # Verificar posições ativas
            active_positions = get_active_positions()
            print(f"\n[POSICOES] Posicoes ativas: {active_positions if active_positions else 'Nenhuma'}")
            
            # Filtrar apenas símbolos sem posição ativa
            symbols_to_analyze = [s for s in symbols if s not in active_positions]
            
            if not symbols_to_analyze:
                print("\n[OK] Todos os pares tem posicoes ativas. Aguardando fechamento...")
            else:
                print(f"\n[ANALISE] Analisando {len(symbols_to_analyze)} pares sem posicoes ativas...")
                
                for i, symbol in enumerate(symbols_to_analyze, 1):
                    print(f"\n[{i}/{len(symbols_to_analyze)}] Analisando {symbol}...")
                    print("-" * 40)
                    signal = await agent.analyze(symbol)
                    
                    # Mostrar resumo rápido
                    if signal.get('signal') in ['BUY', 'SELL']:
                        print(f"[ALERTA] {signal.get('signal')} com confianca {signal.get('confidence', 0)}/10")
                    else:
                        print(f"[SINAL] {signal.get('signal')} - Confianca: {signal.get('confidence', 0)}/10")
                    
                    await asyncio.sleep(3)  # Pausa entre análises
        
        elif args.mode == 'top10':
            # Todos os pares configurados
            from config import settings
            symbols = settings.top_crypto_pairs
            
            print(f"\n[TOP] Analisando {len(symbols)} pares configurados...")
            for i, symbol in enumerate(symbols, 1):
                print(f"\n[{i}/{len(symbols)}] Analisando {symbol}...")
                await agent.analyze(symbol)
                await asyncio.sleep(5)  # Pausa entre análises
    
    except KeyboardInterrupt:
        logger.info("Sistema interrompido pelo usuário")
        print("\n\n[PARADO] Sistema interrompido pelo usuario")
    except Exception as e:
        logger.exception(f"Erro fatal: {e}")
        print(f"\n[ERRO FATAL] Erro fatal: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())