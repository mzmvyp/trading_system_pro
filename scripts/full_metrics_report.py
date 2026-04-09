"""
Relatório Completo de Métricas — Análise de todos os sinais armazenados.
Exclui JCTUSDT (dados corrompidos) e gera análise minuciosa.
"""
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

SIGNALS_DIR = ROOT / "signals"
EXCLUDE_SYMBOLS = {"JCTUSDT", "4USDT"}

# ============================================================
# 1. LOAD ALL SIGNALS
# ============================================================
def load_all_signals():
    signals = []
    for f in SIGNALS_DIR.glob("agno_*.json"):
        if "last_analysis" in f.name:
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            sym = data.get("symbol", "")
            if sym in EXCLUDE_SYMBOLS:
                continue
            data["_file"] = str(f.name)
            signals.append(data)
        except (json.JSONDecodeError, IOError):
            continue
    return signals


# ============================================================
# 2. EVALUATE OUTCOMES VIA SIGNAL TRACKER
# ============================================================
def evaluate_signals(signals):
    from src.trading.signal_tracker import evaluate_signal
    results = []
    total = len(signals)
    for i, sig in enumerate(signals):
        if (i + 1) % 200 == 0:
            print(f"  Avaliando {i+1}/{total}...", flush=True)
        try:
            r = evaluate_signal(sig)
            r["_original"] = sig
            results.append(r)
        except Exception:
            continue
    return results


# ============================================================
# 3. BUILD DATAFRAME
# ============================================================
def build_dataframe(results):
    rows = []
    for r in results:
        sig = r.get("_original", {})
        ts_str = sig.get("timestamp", "")
        try:
            if "+" in ts_str or ts_str.endswith("Z"):
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            elif "T" in ts_str:
                ts = datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc)
            else:
                ts = datetime.strptime(ts_str[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        except Exception:
            ts = None

        voter_votes = sig.get("voter_votes", {})

        rows.append({
            "symbol": sig.get("symbol", ""),
            "timestamp": ts,
            "date": ts.date() if ts else None,
            "week": ts.isocalendar()[1] if ts else None,
            "hour": ts.hour if ts else None,
            "signal": sig.get("signal", "NO_SIGNAL"),
            "source": sig.get("source", ""),
            "confidence": sig.get("confidence", 0),
            "entry_price": sig.get("entry_price", 0),
            "stop_loss": sig.get("stop_loss", 0),
            "tp1": sig.get("take_profit_1", 0),
            "tp2": sig.get("take_profit_2", 0),
            "outcome": r.get("outcome", ""),
            "pnl_pct": r.get("pnl_percent", 0),
            "exit_price": r.get("exit_price", 0),
            "is_winner": r.get("is_winner", False),
            "executed": sig.get("executed", False),
            "execution_mode": sig.get("execution_mode"),
            "ml_probability": sig.get("ml_probability") or voter_votes.get("ml_prob"),
            "ml_vote": sig.get("ml_vote") or voter_votes.get("ml", 0),
            "lstm_probability": sig.get("lstm_probability") or voter_votes.get("lstm_prob"),
            "lstm_vote": voter_votes.get("lstm", 0),
            "confluence_score": sig.get("confluence_score", 0),
            "votes_for": sig.get("confluence_votes_for", 0),
            "votes_against": sig.get("confluence_votes_against", 0),
            "market_regime": sig.get("market_regime", ""),
            "market_regime_base": sig.get("market_regime_base", ""),
            "rsi": sig.get("rsi", 0),
            "adx": sig.get("adx", 0),
            "atr": sig.get("atr", 0),
            "bb_position": sig.get("bb_position", 0),
            "trend": sig.get("trend", ""),
            "macd_histogram": sig.get("macd_histogram", 0),
            "cvd": sig.get("cvd", 0),
            "orderbook_imbalance": sig.get("orderbook_imbalance", 0),
            "operation_type": sig.get("operation_type", ""),
            "block_reason": sig.get("block_reason", ""),
            "voter_rsi": voter_votes.get("rsi", 0),
            "voter_macd": voter_votes.get("macd", 0),
            "voter_trend": voter_votes.get("trend", 0),
            "voter_adx": voter_votes.get("adx", 0),
            "voter_bb": voter_votes.get("bb", 0),
            "voter_orderbook": voter_votes.get("orderbook", 0),
            "voter_cvd": voter_votes.get("cvd", 0),
            "voter_regime": voter_votes.get("regime", 0),
            "voter_setup_validator": voter_votes.get("setup_validator", 0),
            "voter_llm": voter_votes.get("llm", 0),
            # risk metrics
            "risk_distance_pct": 0,
            "reward_distance_pct": 0,
            "risk_reward": 0,
        })

    df = pd.DataFrame(rows)

    # Calculate risk metrics
    mask = (df["entry_price"] > 0) & (df["stop_loss"] > 0) & (df["tp1"] > 0)
    df.loc[mask, "risk_distance_pct"] = (abs(df["entry_price"] - df["stop_loss"]) / df["entry_price"] * 100).where(mask)
    df.loc[mask, "reward_distance_pct"] = (abs(df["tp1"] - df["entry_price"]) / df["entry_price"] * 100).where(mask)
    df.loc[mask & (df["risk_distance_pct"] > 0), "risk_reward"] = df["reward_distance_pct"] / df["risk_distance_pct"]

    return df


# ============================================================
# 4. REPORT GENERATION
# ============================================================
def generate_report(df):
    lines = []
    def p(s=""):
        lines.append(s)

    p("=" * 90)
    p("RELATÓRIO COMPLETO DE MÉTRICAS — TRADING SYSTEM PRO")
    p(f"Gerado em: {datetime.now(timezone.utc).isoformat()}")
    p(f"Período: {df['date'].min()} → {df['date'].max()}")
    p(f"Total sinais (excl. JCTUSDT): {len(df)}")
    p("=" * 90)

    # ---- OVERVIEW ----
    finalized = df[df["outcome"].isin(["SL_HIT", "TP1_HIT", "TP2_HIT", "EXPIRED"])]
    active = df[df["outcome"] == "ACTIVE"]
    no_signal = df[df["signal"] == "NO_SIGNAL"]
    has_signal = df[df["signal"].isin(["BUY", "SELL"])]
    winners = finalized[finalized["is_winner"] == True]

    p("\n" + "=" * 90)
    p("1. RESUMO GERAL")
    p("=" * 90)
    p(f"  Total sinais carregados:   {len(df)}")
    p(f"  Com sinal (BUY/SELL):      {len(has_signal)} ({len(has_signal)/len(df)*100:.1f}%)")
    p(f"  NO_SIGNAL (bloqueados):    {len(no_signal)} ({len(no_signal)/len(df)*100:.1f}%)")
    p(f"  Finalizados:               {len(finalized)}")
    p(f"  Ativos:                    {len(active)}")
    p(f"  Win Rate (finalizados):    {len(winners)/max(len(finalized),1)*100:.1f}%")
    p(f"  PnL Total:                 {finalized['pnl_pct'].sum():.2f}%")
    p(f"  PnL Médio:                 {finalized['pnl_pct'].mean():.2f}%")
    p(f"  Avg Win:                   {winners['pnl_pct'].mean():.2f}%")
    losers = finalized[finalized["is_winner"] == False]
    p(f"  Avg Loss:                  {losers['pnl_pct'].mean():.2f}%")
    p(f"  Melhor Trade:              {finalized['pnl_pct'].max():.2f}%")
    p(f"  Pior Trade:                {finalized['pnl_pct'].min():.2f}%")

    # Outcomes breakdown
    p(f"\n  Breakdown de outcomes:")
    for oc in ["TP1_HIT", "TP2_HIT", "SL_HIT", "EXPIRED", "ACTIVE"]:
        cnt = len(df[df["outcome"] == oc])
        p(f"    {oc:15s}: {cnt:5d} ({cnt/max(len(df),1)*100:.1f}%)")

    # Profit Factor
    gross_profit = winners["pnl_pct"].sum() if len(winners) > 0 else 0
    gross_loss = abs(losers["pnl_pct"].sum()) if len(losers) > 0 else 1
    pf = gross_profit / max(gross_loss, 0.01)
    p(f"\n  Profit Factor:             {pf:.2f}")

    # Expectancy
    wr = len(winners) / max(len(finalized), 1)
    avg_w = winners["pnl_pct"].mean() if len(winners) > 0 else 0
    avg_l = abs(losers["pnl_pct"].mean()) if len(losers) > 0 else 0
    expectancy = wr * avg_w - (1 - wr) * avg_l
    p(f"  Expectancy:                {expectancy:.2f}%")

    # ---- EXECUTED vs NOT ----
    p("\n" + "=" * 90)
    p("2. EXECUTADOS vs NÃO EXECUTADOS")
    p("=" * 90)
    for label, mask_val in [("EXECUTADOS", True), ("NÃO EXECUTADOS", False)]:
        sub = finalized[finalized["executed"] == mask_val]
        w = sub[sub["is_winner"] == True]
        l = sub[sub["is_winner"] == False]
        wr_sub = len(w) / max(len(sub), 1) * 100
        gp = w["pnl_pct"].sum() if len(w) > 0 else 0
        gl = abs(l["pnl_pct"].sum()) if len(l) > 0 else 0.01
        p(f"\n  {label}: {len(sub)} trades finalizados")
        p(f"    Win Rate:     {wr_sub:.1f}%")
        p(f"    PnL Total:    {sub['pnl_pct'].sum():.2f}%")
        p(f"    Avg Win:      {w['pnl_pct'].mean():.2f}%" if len(w) > 0 else "    Avg Win:      N/A")
        p(f"    Avg Loss:     {l['pnl_pct'].mean():.2f}%" if len(l) > 0 else "    Avg Loss:     N/A")
        p(f"    Profit Factor:{gp/max(gl,0.01):.2f}")

    # ---- BUY vs SELL ----
    p("\n" + "=" * 90)
    p("3. BUY vs SELL")
    p("=" * 90)
    for direction in ["BUY", "SELL"]:
        sub = finalized[finalized["signal"] == direction]
        w = sub[sub["is_winner"] == True]
        p(f"\n  {direction}: {len(sub)} trades finalizados")
        p(f"    Win Rate: {len(w)/max(len(sub),1)*100:.1f}%")
        p(f"    PnL Total: {sub['pnl_pct'].sum():.2f}%")
        p(f"    Avg PnL: {sub['pnl_pct'].mean():.2f}%")

    # ---- BY SYMBOL ----
    p("\n" + "=" * 90)
    p("4. PERFORMANCE POR SÍMBOLO (Top 20 por volume)")
    p("=" * 90)
    sym_stats = []
    for sym, grp in finalized.groupby("symbol"):
        w = grp[grp["is_winner"] == True]
        sym_stats.append({
            "symbol": sym,
            "n": len(grp),
            "wr": len(w) / max(len(grp), 1) * 100,
            "pnl": grp["pnl_pct"].sum(),
            "avg_pnl": grp["pnl_pct"].mean(),
        })
    sym_df = pd.DataFrame(sym_stats).sort_values("n", ascending=False)
    p(f"  {'Símbolo':15s} {'N':>5s} {'WR':>7s} {'PnL Total':>10s} {'Avg PnL':>9s}")
    p(f"  {'-'*15} {'-'*5} {'-'*7} {'-'*10} {'-'*9}")
    for _, row in sym_df.head(30).iterrows():
        p(f"  {row['symbol']:15s} {row['n']:5.0f} {row['wr']:6.1f}% {row['pnl']:9.2f}% {row['avg_pnl']:8.2f}%")

    # Top winners and losers symbols
    p(f"\n  --- TOP 10 MELHORES SÍMBOLOS (por PnL) ---")
    for _, row in sym_df.sort_values("pnl", ascending=False).head(10).iterrows():
        p(f"    {row['symbol']:15s} PnL={row['pnl']:+8.2f}%  WR={row['wr']:.0f}%  N={row['n']:.0f}")
    p(f"\n  --- TOP 10 PIORES SÍMBOLOS (por PnL) ---")
    for _, row in sym_df.sort_values("pnl", ascending=True).head(10).iterrows():
        p(f"    {row['symbol']:15s} PnL={row['pnl']:+8.2f}%  WR={row['wr']:.0f}%  N={row['n']:.0f}")

    # ---- CONFIDENCE ANALYSIS ----
    p("\n" + "=" * 90)
    p("5. ANÁLISE POR CONFIANÇA")
    p("=" * 90)
    p(f"  {'Conf':>5s} {'N':>5s} {'WR':>7s} {'PnL Total':>10s} {'Avg PnL':>9s}")
    p(f"  {'-'*5} {'-'*5} {'-'*7} {'-'*10} {'-'*9}")
    for conf in sorted(finalized["confidence"].unique()):
        sub = finalized[finalized["confidence"] == conf]
        w = sub[sub["is_winner"] == True]
        p(f"  {conf:5.0f} {len(sub):5d} {len(w)/max(len(sub),1)*100:6.1f}% {sub['pnl_pct'].sum():9.2f}% {sub['pnl_pct'].mean():8.2f}%")

    # ---- ML ANALYSIS ----
    p("\n" + "=" * 90)
    p("6. ANÁLISE DO MODELO ML")
    p("=" * 90)
    ml_sub = finalized[finalized["ml_probability"].notna() & (finalized["ml_probability"] > 0)]
    if len(ml_sub) > 0:
        for label, lo, hi in [("Prob < 30% (FORTE CONTRA)", 0, 0.30),
                               ("Prob 30-50% (CONTRA)", 0.30, 0.50),
                               ("Prob 50-75% (NEUTRO)", 0.50, 0.75),
                               ("Prob >= 75% (A FAVOR)", 0.75, 1.01)]:
            sub = ml_sub[(ml_sub["ml_probability"] >= lo) & (ml_sub["ml_probability"] < hi)]
            if len(sub) == 0:
                continue
            w = sub[sub["is_winner"] == True]
            p(f"  {label}: N={len(sub)}, WR={len(w)/max(len(sub),1)*100:.1f}%, "
              f"PnL={sub['pnl_pct'].sum():.2f}%, Avg={sub['pnl_pct'].mean():.2f}%")

        p(f"\n  ML Vote breakdown (finalizados com ML data):")
        for v in sorted(ml_sub["ml_vote"].unique()):
            sub = ml_sub[ml_sub["ml_vote"] == v]
            w = sub[sub["is_winner"] == True]
            p(f"    Vote={v:+.0f}: N={len(sub)}, WR={len(w)/max(len(sub),1)*100:.1f}%, PnL={sub['pnl_pct'].sum():.2f}%")
    else:
        p("  Sem dados de ML suficientes.")

    # ---- LSTM ANALYSIS ----
    p("\n" + "=" * 90)
    p("7. ANÁLISE DO MODELO Bi-LSTM")
    p("=" * 90)
    lstm_sub = finalized[finalized["lstm_probability"].notna() & (finalized["lstm_probability"] > 0)]
    if len(lstm_sub) > 0:
        for label, lo, hi in [("Prob < 40%", 0, 0.40),
                               ("Prob 40-50%", 0.40, 0.50),
                               ("Prob 50-60%", 0.50, 0.60),
                               ("Prob >= 60%", 0.60, 1.01)]:
            sub = lstm_sub[(lstm_sub["lstm_probability"] >= lo) & (lstm_sub["lstm_probability"] < hi)]
            if len(sub) == 0:
                continue
            w = sub[sub["is_winner"] == True]
            p(f"  {label}: N={len(sub)}, WR={len(w)/max(len(sub),1)*100:.1f}%, "
              f"PnL={sub['pnl_pct'].sum():.2f}%, Avg={sub['pnl_pct'].mean():.2f}%")
    else:
        p("  Sem dados de LSTM suficientes.")

    # ---- VOTER ANALYSIS ----
    p("\n" + "=" * 90)
    p("8. ANÁLISE POR VOTER (indicadores individuais)")
    p("=" * 90)
    voter_cols = [c for c in df.columns if c.startswith("voter_") and c not in ("voter_votes",)]
    for vc in voter_cols:
        voter_name = vc.replace("voter_", "")
        sub = finalized[finalized[vc] != 0]
        if len(sub) < 10:
            continue
        favor = sub[sub[vc] > 0]
        contra = sub[sub[vc] < 0]
        fw = favor[favor["is_winner"] == True]
        cw = contra[contra["is_winner"] == True]
        p(f"  {voter_name:20s} | A FAVOR: N={len(favor):4d}, WR={len(fw)/max(len(favor),1)*100:5.1f}% | "
          f"CONTRA: N={len(contra):4d}, WR={len(cw)/max(len(contra),1)*100:5.1f}%")

    # ---- CONFLUENCE ANALYSIS ----
    p("\n" + "=" * 90)
    p("9. ANÁLISE POR CONFLUÊNCIA (score + n votos)")
    p("=" * 90)
    conf_sub = finalized[finalized["confluence_score"] > 0]
    if len(conf_sub) > 0:
        p(f"\n  Score ranges:")
        for label, lo, hi in [("< 40%", 0, 0.40), ("40-55%", 0.40, 0.55),
                               ("55-65%", 0.55, 0.65), ("65-75%", 0.65, 0.75),
                               (">= 75%", 0.75, 1.01)]:
            sub = conf_sub[(conf_sub["confluence_score"] >= lo) & (conf_sub["confluence_score"] < hi)]
            if len(sub) == 0:
                continue
            w = sub[sub["is_winner"] == True]
            p(f"    {label:10s}: N={len(sub):4d}, WR={len(w)/max(len(sub),1)*100:5.1f}%, "
              f"PnL={sub['pnl_pct'].sum():.2f}%, Avg={sub['pnl_pct'].mean():.2f}%")

        p(f"\n  Por número de votos a favor:")
        for nv in sorted(conf_sub["votes_for"].unique()):
            sub = conf_sub[conf_sub["votes_for"] == nv]
            if len(sub) < 3:
                continue
            w = sub[sub["is_winner"] == True]
            p(f"    Votos={nv:.0f}: N={len(sub):4d}, WR={len(w)/max(len(sub),1)*100:5.1f}%, "
              f"PnL={sub['pnl_pct'].sum():.2f}%")

    # ---- MARKET REGIME ----
    p("\n" + "=" * 90)
    p("10. ANÁLISE POR REGIME DE MERCADO")
    p("=" * 90)
    regime_sub = finalized[finalized["market_regime"] != ""]
    if len(regime_sub) > 0:
        regime_stats = []
        for reg, grp in regime_sub.groupby("market_regime"):
            w = grp[grp["is_winner"] == True]
            regime_stats.append({"regime": reg, "n": len(grp),
                                "wr": len(w)/max(len(grp),1)*100,
                                "pnl": grp["pnl_pct"].sum()})
        rdf = pd.DataFrame(regime_stats).sort_values("n", ascending=False)
        p(f"  {'Regime':35s} {'N':>5s} {'WR':>7s} {'PnL':>10s}")
        for _, r in rdf.iterrows():
            p(f"  {r['regime']:35s} {r['n']:5.0f} {r['wr']:6.1f}% {r['pnl']:9.2f}%")

    # ---- TEMPORAL ANALYSIS ----
    p("\n" + "=" * 90)
    p("11. ANÁLISE TEMPORAL (hora do dia, dia da semana)")
    p("=" * 90)
    hour_sub = finalized[finalized["hour"].notna()]
    if len(hour_sub) > 0:
        p(f"\n  {'Hora':>5s} {'N':>5s} {'WR':>7s} {'PnL':>10s}")
        for h in range(24):
            sub = hour_sub[hour_sub["hour"] == h]
            if len(sub) < 3:
                continue
            w = sub[sub["is_winner"] == True]
            p(f"  {h:5d} {len(sub):5d} {len(w)/max(len(sub),1)*100:6.1f}% {sub['pnl_pct'].sum():9.2f}%")

    # ---- RISK:REWARD ----
    p("\n" + "=" * 90)
    p("12. ANÁLISE DE RISK:REWARD")
    p("=" * 90)
    rr_sub = finalized[(finalized["risk_reward"] > 0) & (finalized["risk_reward"] < 20)]
    if len(rr_sub) > 0:
        for label, lo, hi in [("R:R < 1.0", 0, 1.0), ("R:R 1.0-1.5", 1.0, 1.5),
                               ("R:R 1.5-2.0", 1.5, 2.0), ("R:R 2.0-3.0", 2.0, 3.0),
                               ("R:R >= 3.0", 3.0, 100)]:
            sub = rr_sub[(rr_sub["risk_reward"] >= lo) & (rr_sub["risk_reward"] < hi)]
            if len(sub) == 0:
                continue
            w = sub[sub["is_winner"] == True]
            p(f"  {label:15s}: N={len(sub):4d}, WR={len(w)/max(len(sub),1)*100:5.1f}%, "
              f"PnL={sub['pnl_pct'].sum():.2f}%, Avg={sub['pnl_pct'].mean():.2f}%")

        p(f"\n  Stop Loss Distance (risk_distance_pct):")
        for label, lo, hi in [("SL < 1%", 0, 1), ("SL 1-2%", 1, 2), ("SL 2-3%", 2, 3),
                               ("SL 3-5%", 3, 5), ("SL 5-10%", 5, 10), ("SL > 10%", 10, 100)]:
            sub = rr_sub[(rr_sub["risk_distance_pct"] >= lo) & (rr_sub["risk_distance_pct"] < hi)]
            if len(sub) == 0:
                continue
            w = sub[sub["is_winner"] == True]
            p(f"    {label:10s}: N={len(sub):4d}, WR={len(w)/max(len(sub),1)*100:5.1f}%, "
              f"PnL={sub['pnl_pct'].sum():.2f}%")

    # ---- RSI ZONES ----
    p("\n" + "=" * 90)
    p("13. ANÁLISE POR RSI (zonas)")
    p("=" * 90)
    rsi_sub = finalized[finalized["rsi"] > 0]
    if len(rsi_sub) > 0:
        for label, lo, hi in [("Oversold (< 30)", 0, 30), ("Approaching OS (30-40)", 30, 40),
                               ("Neutral (40-60)", 40, 60), ("Approaching OB (60-70)", 60, 70),
                               ("Overbought (> 70)", 70, 101)]:
            sub = rsi_sub[(rsi_sub["rsi"] >= lo) & (rsi_sub["rsi"] < hi)]
            if len(sub) == 0:
                continue
            w = sub[sub["is_winner"] == True]
            p(f"  {label:25s}: N={len(sub):4d}, WR={len(w)/max(len(sub),1)*100:5.1f}%, "
              f"PnL={sub['pnl_pct'].sum():.2f}%")

    # ---- ADX STRENGTH ----
    p("\n" + "=" * 90)
    p("14. ANÁLISE POR ADX (força da tendência)")
    p("=" * 90)
    adx_sub = finalized[finalized["adx"] > 0]
    if len(adx_sub) > 0:
        for label, lo, hi in [("Lateral (< 20)", 0, 20), ("Fraca (20-25)", 20, 25),
                               ("Moderada (25-35)", 25, 35), ("Forte (35-50)", 35, 50),
                               ("Muito Forte (> 50)", 50, 200)]:
            sub = adx_sub[(adx_sub["adx"] >= lo) & (adx_sub["adx"] < hi)]
            if len(sub) == 0:
                continue
            w = sub[sub["is_winner"] == True]
            p(f"  {label:25s}: N={len(sub):4d}, WR={len(w)/max(len(sub),1)*100:5.1f}%, "
              f"PnL={sub['pnl_pct'].sum():.2f}%")

    # ---- TREND ----
    p("\n" + "=" * 90)
    p("15. ANÁLISE POR TREND")
    p("=" * 90)
    trend_sub = finalized[finalized["trend"] != ""]
    if len(trend_sub) > 0:
        for t, grp in trend_sub.groupby("trend"):
            w = grp[grp["is_winner"] == True]
            p(f"  {t:25s}: N={len(grp):4d}, WR={len(w)/max(len(grp),1)*100:5.1f}%, "
              f"PnL={grp['pnl_pct'].sum():.2f}%")

    # ---- EVOLUTION OVER TIME ----
    p("\n" + "=" * 90)
    p("16. EVOLUÇÃO SEMANAL")
    p("=" * 90)
    week_sub = finalized[finalized["week"].notna()]
    if len(week_sub) > 0:
        week_sub_copy = week_sub.copy()
        week_sub_copy["year_week"] = week_sub_copy["timestamp"].apply(
            lambda x: f"{x.isocalendar()[0]}-W{x.isocalendar()[1]:02d}" if x else "")
        p(f"  {'Semana':>10s} {'N':>5s} {'WR':>7s} {'PnL':>10s} {'Cum PnL':>10s}")
        cum = 0
        for yw in sorted(week_sub_copy["year_week"].unique()):
            if not yw:
                continue
            sub = week_sub_copy[week_sub_copy["year_week"] == yw]
            w = sub[sub["is_winner"] == True]
            cum += sub["pnl_pct"].sum()
            p(f"  {yw:>10s} {len(sub):5d} {len(w)/max(len(sub),1)*100:6.1f}% "
              f"{sub['pnl_pct'].sum():9.2f}% {cum:9.2f}%")

    # ---- SOURCE ANALYSIS ----
    p("\n" + "=" * 90)
    p("17. POR FONTE DO SINAL")
    p("=" * 90)
    for src in finalized["source"].unique():
        sub = finalized[finalized["source"] == src]
        w = sub[sub["is_winner"] == True]
        p(f"  {str(src):15s}: N={len(sub)}, WR={len(w)/max(len(sub),1)*100:.1f}%, PnL={sub['pnl_pct'].sum():.2f}%")

    # ---- OPERATION TYPE ----
    p("\n" + "=" * 90)
    p("18. POR TIPO DE OPERAÇÃO")
    p("=" * 90)
    for ot in finalized["operation_type"].unique():
        sub = finalized[finalized["operation_type"] == ot]
        if len(sub) < 2:
            continue
        w = sub[sub["is_winner"] == True]
        p(f"  {str(ot):20s}: N={len(sub)}, WR={len(w)/max(len(sub),1)*100:.1f}%, PnL={sub['pnl_pct'].sum():.2f}%")

    p("\n" + "=" * 90)
    p("FIM DO RELATÓRIO")
    p("=" * 90)

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("CARREGANDO TODOS OS SINAIS...")
    print("=" * 60)

    signals = load_all_signals()
    print(f"[OK] {len(signals)} sinais carregados (excl. JCTUSDT)")

    print("\n[AVALIANDO OUTCOMES via signal_tracker...]")
    results = evaluate_signals(signals)
    print(f"[OK] {len(results)} sinais avaliados")

    print("\n[CONSTRUINDO DATAFRAME...]")
    df = build_dataframe(results)
    print(f"[OK] DataFrame: {len(df)} rows x {len(df.columns)} cols")

    print("\n[GERANDO RELATÓRIO...]")
    report = generate_report(df)

    # Save
    report_path = ROOT / "metrics_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n[SALVO] {report_path}")

    # Also save CSV for further analysis
    csv_path = ROOT / "metrics_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"[SALVO] {csv_path}")

    print("\n" + report)
