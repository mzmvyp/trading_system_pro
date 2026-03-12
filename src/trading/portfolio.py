"""
Gerenciador de Portfólio - Interface para o Sistema de Paper Trading
CORRIGIDO: Usa real_paper_trading ao invés de paper_trading
"""

import argparse

from src.trading.paper_trading import real_paper_trading as paper_trading


def show_portfolio():
    """Mostra resumo do portfólio"""
    print("\n" + "="*60)
    print("📊 RESUMO DO PORTFÓLIO")
    print("="*60)

    summary = paper_trading.get_portfolio_summary()

    print(f"💰 Saldo Inicial: ${summary['initial_balance']:,.2f}")
    print(f"💵 Saldo Atual: ${summary['current_balance']:,.2f}")
    print(f"📈 Valor Total: ${summary['total_portfolio_value']:,.2f}")
    print(f"📊 Retorno Total: {summary['total_return_percent']:+.2f}%")
    print(f"💼 P&L Total: ${summary['total_pnl']:+,.2f}")
    print(f"🔢 Posições Abertas: {summary['open_positions_count']}")
    print(f"📋 Total de Trades: {summary['total_trades']}")
    print(f"✅ Trades Vencedores: {summary['winning_trades']}")
    print(f"❌ Trades Perdedores: {summary['losing_trades']}")

    # Calcular taxa de acerto
    if summary['total_trades'] > 0:
        win_rate = (summary['winning_trades'] / summary['total_trades']) * 100
        print(f"🎯 Taxa de Acerto: {win_rate:.1f}%")

    print("="*60)

def show_positions():
    """Mostra posições abertas"""
    positions = paper_trading.get_open_positions()

    if not positions:
        print("\n📭 Nenhuma posição aberta")
        return

    print("\n" + "="*60)
    print("📈 POSIÇÕES ABERTAS")
    print("="*60)

    for pos in positions:
        print(f"🔸 {pos['symbol']} - {pos['signal']}")
        print(f"   💰 Entrada: ${pos['entry_price']:.4f}")
        print(f"   📊 Tamanho: {pos['position_size']:.2f} unidades")
        print(f"   💵 Valor: ${pos['position_value']:,.2f}")
        if pos.get('stop_loss'):
            print(f"   🛑 Stop Loss: ${pos['stop_loss']:.4f}")
        if pos.get('take_profit_1'):
            print(f"   🎯 Target 1: ${pos['take_profit_1']:.4f}")
        if pos.get('take_profit_2'):
            print(f"   🎯 Target 2: ${pos['take_profit_2']:.4f}")
        print(f"   ⏰ Aberto em: {pos['timestamp']}")
        print("-" * 40)

def show_history(limit=10):
    """Mostra histórico de trades"""
    history = paper_trading.get_trade_history(limit)

    if not history:
        print("\n📭 Nenhum trade no histórico")
        return

    print("\n" + "="*60)
    print(f"📋 HISTÓRICO DE TRADES (Últimos {len(history)})")
    print("="*60)

    for trade in reversed(history):  # Mostrar mais recentes primeiro
        status_emoji = "✅" if trade.get('status') == 'CLOSED' else "🔄"
        pnl_emoji = "📈" if trade.get('pnl', 0) > 0 else "📉" if trade.get('pnl', 0) < 0 else "➖"

        print(f"{status_emoji} {trade['symbol']} - {trade['signal']}")
        print(f"   💰 Entrada: ${trade['entry_price']:.4f}")
        if trade.get('close_price'):
            print(f"   💰 Fechamento: ${trade['close_price']:.4f}")
        print(f"   📊 Tamanho: {trade['position_size']:.2f}")
        if trade.get('pnl') is not None:
            print(f"   {pnl_emoji} P&L: ${trade['pnl']:+,.2f}")
        print(f"   ⏰ {trade['timestamp']}")
        print("-" * 40)

def close_position(symbol, current_price):
    """Fecha uma posição"""
    result = paper_trading.close_position(symbol, current_price)

    if result["success"]:
        print(f"\n✅ Posição {symbol} fechada!")
        print(f"💰 P&L: ${result['pnl']:+,.2f}")
    else:
        print(f"\n❌ Erro ao fechar posição: {result['error']}")

def export_report():
    """Exporta relatório de performance"""
    filename = paper_trading.export_performance_report()
    print(f"\n📄 Relatório exportado: {filename}")

def reset_portfolio():
    """Reseta o portfólio"""
    confirm = input("\n⚠️ Tem certeza que deseja resetar o portfólio? (sim/não): ")
    if confirm.lower() in ['sim', 's', 'yes', 'y']:
        paper_trading.reset_portfolio()
        print("🔄 Portfólio resetado!")
    else:
        print("❌ Operação cancelada")

def main():
    parser = argparse.ArgumentParser(description='Gerenciador de Portfólio')
    parser.add_argument('--action', choices=['portfolio', 'positions', 'history', 'close', 'export', 'reset'],
                       default='portfolio', help='Ação a executar')
    parser.add_argument('--symbol', help='Símbolo para fechar posição')
    parser.add_argument('--price', type=float, help='Preço atual para fechar posição')
    parser.add_argument('--limit', type=int, default=10, help='Limite de trades no histórico')

    args = parser.parse_args()

    if args.action == 'portfolio':
        show_portfolio()
    elif args.action == 'positions':
        show_positions()
    elif args.action == 'history':
        show_history(args.limit)
    elif args.action == 'close':
        if not args.symbol or not args.price:
            print("❌ Use --symbol e --price para fechar posição")
            return
        close_position(args.symbol, args.price)
    elif args.action == 'export':
        export_report()
    elif args.action == 'reset':
        reset_portfolio()

if __name__ == "__main__":
    main()
