#!/usr/bin/env python3
"""
Script para extrair TODOS os dados de analytics do sistema de trading.
Gera um arquivo JSON consolidado para análise externa.

Uso: python scripts/extract_analytics.py
Saída: analytics_report.json
"""

import json
import os
import glob
from datetime import datetime


def safe_load_json(path):
    """Carrega JSON com tratamento de erro"""
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        return {"_error": str(e)}
    return None


def load_all_signals():
    """Carrega todos os sinais gerados"""
    signals = []
    for filepath in sorted(glob.glob("signals/agno_*_*.json")):
        if "_last_analysis" in filepath:
            continue
        data = safe_load_json(filepath)
        if data and data.get("signal") in ("BUY", "SELL"):
            signals.append({
                "file": os.path.basename(filepath),
                "symbol": data.get("symbol"),
                "signal": data.get("signal"),
                "entry_price": data.get("entry_price"),
                "stop_loss": data.get("stop_loss"),
                "take_profit_1": data.get("take_profit_1"),
                "take_profit_2": data.get("take_profit_2"),
                "confidence": data.get("confidence"),
                "source": data.get("source"),
                "timestamp": data.get("timestamp"),
                "trend": data.get("trend"),
                "indicators": data.get("indicators", {}),
                "bullish_tf_count": data.get("bullish_tf_count"),
                "bearish_tf_count": data.get("bearish_tf_count"),
            })
    return signals


def load_evaluations():
    """Avalia sinais contra dados reais da Binance"""
    try:
        from src.trading.signal_tracker import evaluate_all_signals, get_performance_summary
        evals = evaluate_all_signals()
        if evals:
            summary = get_performance_summary(evals)

            # Performance por fonte
            by_source = {}
            for source in set(e.get("source", "UNKNOWN") for e in evals):
                src_evals = [e for e in evals if e.get("source") == source]
                by_source[source] = get_performance_summary(src_evals)

            # Performance por simbolo
            by_symbol = {}
            for sym in set(e.get("symbol", "") for e in evals):
                sym_evals = [e for e in evals if e.get("symbol") == sym]
                by_symbol[sym] = get_performance_summary(sym_evals)

            # Performance LONG vs SHORT
            by_direction = {}
            for sig, label in [("BUY", "LONG"), ("SELL", "SHORT")]:
                dir_evals = [e for e in evals if e.get("signal") == sig]
                if dir_evals:
                    by_direction[label] = get_performance_summary(dir_evals)

            # Detalhes de cada avaliacao
            details = []
            for ev in evals:
                details.append({
                    "symbol": ev.get("symbol"),
                    "signal": ev.get("signal"),
                    "source": ev.get("source"),
                    "timestamp": ev.get("timestamp"),
                    "entry_price": ev.get("entry_price"),
                    "stop_loss": ev.get("stop_loss"),
                    "take_profit_1": ev.get("take_profit_1"),
                    "take_profit_2": ev.get("take_profit_2"),
                    "confidence": ev.get("confidence"),
                    "outcome": ev.get("outcome"),
                    "exit_price": ev.get("exit_price"),
                    "pnl_percent": ev.get("pnl_percent"),
                    "duration_hours": ev.get("duration_hours"),
                    "mfe": ev.get("mfe"),
                    "mae": ev.get("mae"),
                })

            return {
                "summary": summary,
                "by_source": by_source,
                "by_symbol": by_symbol,
                "by_direction": by_direction,
                "evaluations": details,
            }
    except Exception as e:
        return {"_error": str(e)}
    return None


def load_portfolio():
    """Carrega estado do portfolio"""
    return safe_load_json("portfolio/state.json")


def load_ml_data():
    """Carrega dados do ML"""
    return {
        "model_info": safe_load_json("ml_models/model_info_simple.json"),
        "performance_history": safe_load_json("ml_models/model_performance.json"),
        "prediction_log": safe_load_json("ml_models/prediction_log.json"),
        "online_learning_buffer_size": len(safe_load_json("ml_models/online_learning_buffer.json") or []),
    }


def load_real_orders():
    """Carrega ordens reais executadas"""
    orders = []
    for filepath in sorted(glob.glob("real_orders/execution_*.json")):
        data = safe_load_json(filepath)
        if data:
            orders.append(data)
    return orders


def count_files():
    """Conta arquivos em cada diretorio"""
    dirs = ["signals", "real_orders", "portfolio", "ml_models", "ml_dataset",
            "logs", "paper_trades", "simulation_logs", "reevaluation_logs",
            "stop_adjustment_logs", "deepseek_logs", "data/backups"]
    counts = {}
    for d in dirs:
        if os.path.isdir(d):
            files = glob.glob(f"{d}/**/*", recursive=True)
            counts[d] = len([f for f in files if os.path.isfile(f)])
        else:
            counts[d] = 0
    return counts


def main():
    print("=" * 60)
    print("  TRADING SYSTEM PRO - ANALYTICS EXTRACTION")
    print("=" * 60)
    print()

    report = {
        "_generated_at": datetime.now().isoformat(),
        "_version": "1.0",
    }

    # 1. File counts
    print("[1/6] Contando arquivos...")
    report["file_counts"] = count_files()

    # 2. Sinais
    print("[2/6] Carregando sinais...")
    signals = load_all_signals()
    report["signals"] = {
        "total": len(signals),
        "by_symbol": {},
        "by_source": {},
        "list": signals,
    }
    for s in signals:
        sym = s.get("symbol", "UNKNOWN")
        src = s.get("source", "UNKNOWN")
        report["signals"]["by_symbol"][sym] = report["signals"]["by_symbol"].get(sym, 0) + 1
        report["signals"]["by_source"][src] = report["signals"]["by_source"].get(src, 0) + 1

    # 3. Avaliacoes contra Binance
    print("[3/6] Avaliando sinais contra Binance (pode demorar)...")
    evals = load_evaluations()
    report["evaluations"] = evals

    # 4. Portfolio
    print("[4/6] Carregando portfolio...")
    portfolio = load_portfolio()
    if portfolio:
        report["portfolio"] = {
            "initial_balance": portfolio.get("initial_balance"),
            "current_balance": portfolio.get("current_balance"),
            "positions_count": len(portfolio.get("positions", {})),
            "trade_history_count": len(portfolio.get("trade_history", [])),
            "positions": portfolio.get("positions", {}),
            "trade_history": portfolio.get("trade_history", []),
        }
    else:
        report["portfolio"] = None

    # 5. ML
    print("[5/6] Carregando dados ML...")
    report["ml"] = load_ml_data()

    # 6. Real orders
    print("[6/6] Carregando ordens reais...")
    real_orders = load_real_orders()
    report["real_orders"] = {
        "total": len(real_orders),
        "list": real_orders,
    }

    # Salvar
    output_path = "analytics_report.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    print()
    print("=" * 60)
    print(f"  RELATORIO SALVO EM: {output_path}")
    print("=" * 60)

    # Resumo rápido
    if evals and isinstance(evals, dict) and "summary" in evals:
        s = evals["summary"]
        print()
        print("  RESUMO RAPIDO:")
        print(f"  Total sinais: {s.get('total_signals', 0)}")
        print(f"  Fechados: {s.get('closed', 0)}")
        print(f"  Ativos: {s.get('active', 0)}")
        print(f"  Win Rate: {s.get('win_rate', 0):.1f}%")
        print(f"  P&L Total: {s.get('total_pnl', 0):+.2f}%")
        print(f"  Profit Factor: {s.get('profit_factor', 0):.2f}")
        print(f"  Expectancy: {s.get('expectancy', 0):+.2f}%")
        print(f"  SL: {s.get('sl_hits', 0)} | TP1: {s.get('tp1_hits', 0)} | TP2: {s.get('tp2_hits', 0)}")
        print()

    print(f"\nCole o conteudo de '{output_path}' para analise completa.")


if __name__ == "__main__":
    main()
