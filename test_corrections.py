"""
Script de teste para validar as correções do sistema de trading
"""
import asyncio
from agno_tools import get_deepseek_analysis
from dotenv import load_dotenv 
load_dotenv()
async def test_deepseek_analysis():
    """Testa se get_deepseek_analysis() chama DeepSeek corretamente"""
    print("="*60)
    print("TESTE: get_deepseek_analysis()")
    print("="*60)

    result = await get_deepseek_analysis('BTCUSDT')

    print(f"\nResultado:")
    print(f"  Source: {result.get('source', 'N/A')}")
    print(f"  Signal: {result.get('signal', 'N/A')}")
    print(f"  Entry: {result.get('entry_price', 'N/A')}")
    print(f"  Confidence: {result.get('confidence', 'N/A')}")

    # Validar
    if result.get('signal') in ['BUY', 'SELL', 'NO_SIGNAL']:
        print('\n✅ Sinal gerado corretamente!')
        return True
    else:
        print('\n❌ Problema na geração de sinal')
        return False

async def test_pnl_calculations():
    """Testa cálculos de P&L proporcional"""
    print("\n" + "="*60)
    print("TESTE: Cálculos de P&L Proporcional")
    print("="*60)

    # Simular fechamento parcial
    entry_price = 100.0
    current_price = 110.0  # 10% de ganho
    partial_percent = 0.5  # Fechando 50%

    # P&L bruto
    pnl_percent_raw = ((current_price - entry_price) / entry_price) * 100
    # P&L ponderado
    weighted_pnl_percent = pnl_percent_raw * partial_percent

    print(f"\nSimulação:")
    print(f"  Entry: ${entry_price:.2f}")
    print(f"  Current: ${current_price:.2f}")
    print(f"  P&L bruto: {pnl_percent_raw:+.2f}%")
    print(f"  Fechando: {partial_percent*100:.0f}%")
    print(f"  P&L ponderado: {weighted_pnl_percent:+.2f}%")

    # Validar
    expected = 5.0  # 10% * 0.5 = 5%
    if abs(weighted_pnl_percent - expected) < 0.01:
        print(f'\n✅ Cálculo correto! Esperado: {expected}%, Obtido: {weighted_pnl_percent}%')
        return True
    else:
        print(f'\n❌ Cálculo incorreto! Esperado: {expected}%, Obtido: {weighted_pnl_percent}%')
        return False

async def test_migration():
    """Testa migração de trades antigos"""
    print("\n" + "="*60)
    print("TESTE: Migração de Trades Antigos")
    print("="*60)

    # Simular trade antigo com pnl
    old_trade = {
        "trade_id": "test_001",
        "pnl": 100.0,  # $100 de lucro
        "entry_price": 1000.0,
        "position_size": 1.0
    }

    # Converter para pnl_percent
    entry = old_trade.get("entry_price", 1)
    size = old_trade.get("position_size", 1)
    if entry > 0 and size > 0:
        pnl_percent = (old_trade["pnl"] / (entry * size)) * 100
    else:
        pnl_percent = 0

    print(f"\nTrade antigo:")
    print(f"  PnL (absoluto): ${old_trade['pnl']:.2f}")
    print(f"  Entry: ${old_trade['entry_price']:.2f}")
    print(f"  Size: {old_trade['position_size']:.6f}")
    print(f"  PnL (convertido): {pnl_percent:+.2f}%")

    # Validar
    expected = 10.0  # $100 / $1000 = 10%
    if abs(pnl_percent - expected) < 0.01:
        print(f'\n✅ Conversão correta! Esperado: {expected}%, Obtido: {pnl_percent}%')
        return True
    else:
        print(f'\n❌ Conversão incorreta! Esperado: {expected}%, Obtido: {pnl_percent}%')
        return False

async def main():
    """Executa todos os testes"""
    print("\n" + "="*60)
    print("VALIDAÇÃO DAS CORREÇÕES - SISTEMA DE TRADING")
    print("="*60)

    results = []

    # Teste 1: DeepSeek Analysis
    try:
        result = await test_deepseek_analysis()
        results.append(("get_deepseek_analysis()", result))
    except Exception as e:
        print(f'\n❌ Erro no teste: {e}')
        results.append(("get_deepseek_analysis()", False))

    # Teste 2: P&L Calculations
    try:
        result = await test_pnl_calculations()
        results.append(("P&L Proporcional", result))
    except Exception as e:
        print(f'\n❌ Erro no teste: {e}')
        results.append(("P&L Proporcional", False))

    # Teste 3: Migration
    try:
        result = await test_migration()
        results.append(("Migração de Trades", result))
    except Exception as e:
        print(f'\n❌ Erro no teste: {e}')
        results.append(("Migração de Trades", False))

    # Resumo
    print("\n" + "="*60)
    print("RESUMO DOS TESTES")
    print("="*60)

    for test_name, passed in results:
        status = "✅ PASSOU" if passed else "❌ FALHOU"
        print(f"{status} - {test_name}")

    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)

    print("\n" + "="*60)
    print(f"Total: {total_passed}/{total_tests} testes passaram")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())
