"""Análise limpa — remove duplicados/spam e calcula posições óptimas por período."""
import pandas as pd
import numpy as np
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

df = pd.read_csv(ROOT / "metrics_data.csv")
df = df[~df["symbol"].isin(["JCTUSDT"])]
df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
fin = df[df["outcome"].isin(["SL_HIT", "TP1_HIT", "TP2_HIT", "EXPIRED"])].copy()
fin = fin[fin["signal"].isin(["BUY", "SELL"])].copy()
fin = fin.dropna(subset=["ts"]).sort_values("ts")

print("=" * 80)
print("1. IDENTIFICAR PERIODO INFLADO (spam de sinais)")
print("=" * 80)

fin["date"] = fin["ts"].dt.date
daily = fin.groupby("date").agg(
    n_signals=("signal", "count"),
    n_buy=("signal", lambda x: (x == "BUY").sum()),
    n_sell=("signal", lambda x: (x == "SELL").sum()),
    wr=("is_winner", "mean"),
    pnl=("pnl_pct", "sum"),
).reset_index()

print("\nDias com MAIS de 20 sinais (suspeitos de spam):")
spam_days = daily[daily["n_signals"] > 20].sort_values("n_signals", ascending=False)
print(f"  {'Data':12s} {'N':>5s} {'BUY':>5s} {'SELL':>5s} {'WR':>7s} {'PnL':>10s}")
for _, r in spam_days.iterrows():
    print(f"  {str(r['date']):12s} {r['n_signals']:5.0f} {r['n_buy']:5.0f} "
          f"{r['n_sell']:5.0f} {r['wr']*100:6.1f}% {r['pnl']:9.2f}%")

total_spam = spam_days["n_signals"].sum()
total_all = len(fin)
print(f"\nTotal sinais em dias spam: {total_spam} ({total_spam/total_all*100:.1f}% de {total_all})")

# Filtrar duplicados: mesmo simbolo+direcao dentro de 4h
print("\n" + "=" * 80)
print("2. REMOVER DUPLICADOS (mesmo simbolo+direcao em < 4h)")
print("=" * 80)

fin_sorted = fin.sort_values(["symbol", "signal", "ts"])
keep_mask = []
last_key = {}
for idx, row in fin_sorted.iterrows():
    key = f"{row['symbol']}_{row['signal']}"
    ts = row["ts"]
    if key in last_key and (ts - last_key[key]) < timedelta(hours=4):
        keep_mask.append(False)
    else:
        keep_mask.append(True)
        last_key[key] = ts

fin_sorted["keep"] = keep_mask
clean = fin_sorted[fin_sorted["keep"]].copy().sort_values("ts")
removed = len(fin_sorted) - len(clean)
print(f"Sinais originais: {len(fin_sorted)}")
print(f"Duplicados removidos: {removed}")
print(f"Sinais unicos: {len(clean)}")

print(f"\n--- ANTES (com duplicados) ---")
for d in ["BUY", "SELL"]:
    sub = fin[fin["signal"] == d]
    w = sub[sub["is_winner"] == True]
    print(f"  {d}: N={len(sub)}, WR={len(w)/max(len(sub),1)*100:.1f}%, PnL={sub['pnl_pct'].sum():.1f}%")

print(f"\n--- DEPOIS (sem duplicados, dados limpos) ---")
for d in ["BUY", "SELL"]:
    sub = clean[clean["signal"] == d]
    w = sub[sub["is_winner"] == True]
    print(f"  {d}: N={len(sub)}, WR={len(w)/max(len(sub),1)*100:.1f}%, PnL={sub['pnl_pct'].sum():.1f}%")

# Confidence por direcao limpa
print(f"\n--- CONFIDENCE (dados limpos) ---")
for d in ["BUY", "SELL"]:
    for c in [6, 7, 8, 9]:
        sub = clean[(clean["signal"] == d) & (clean["confidence"] >= c)]
        if len(sub) >= 5:
            w = sub[sub["is_winner"] == True]
            print(f"  {d} conf>={c}: N={len(sub)}, WR={len(w)/max(len(sub),1)*100:.1f}%, PnL={sub['pnl_pct'].sum():.1f}%")

print("\n" + "=" * 80)
print("3. ANALISE SEMANAL (dados limpos)")
print("=" * 80)

clean["week"] = clean["ts"].dt.strftime("%Y-W%U")
weekly = clean.groupby("week").agg(
    n=("signal", "count"),
    n_buy=("signal", lambda x: (x == "BUY").sum()),
    n_sell=("signal", lambda x: (x == "SELL").sum()),
    wins=("is_winner", "sum"),
    wr=("is_winner", "mean"),
    pnl=("pnl_pct", "sum"),
).reset_index()

print(f"\n{'Semana':10s} {'N':>4s} {'BUY':>4s} {'SELL':>4s} {'Wins':>5s} {'WR':>7s} {'PnL':>10s}")
for _, r in weekly.sort_values("week").iterrows():
    print(f"{r['week']:10s} {r['n']:4.0f} {r['n_buy']:4.0f} {r['n_sell']:4.0f} "
          f"{r['wins']:5.0f} {r['wr']*100:6.1f}% {r['pnl']:9.2f}%")

print(f"\nMedia semanal: {weekly['n'].mean():.0f} sinais, WR={weekly['wr'].mean()*100:.1f}%, "
      f"PnL={weekly['pnl'].mean():.1f}%")

print("\n" + "=" * 80)
print("4. QUANTOS SINAIS ABERTOS POR PERIODO PARA PnL POSITIVO?")
print("=" * 80)

# Simular diferentes limites de posicoes abertas por dia
print("\n--- Simulacao: max N sinais por DIA ---")
print(f"  {'MaxDia':>6s} {'N_exec':>7s} {'Wins':>5s} {'WR':>7s} {'PnL':>10s} {'PnL/trade':>10s}")

for max_per_day in [1, 2, 3, 4, 5, 6, 8, 10, 999]:
    # Simular: por cada dia, pegar apenas os primeiros max_per_day sinais
    selected = []
    for date, grp in clean.groupby("date"):
        selected.append(grp.head(max_per_day))
    if not selected:
        continue
    sel = pd.concat(selected)
    w = sel[sel["is_winner"] == True]
    label = "ALL" if max_per_day > 100 else str(max_per_day)
    pnl_trade = sel["pnl_pct"].mean()
    print(f"  {label:>6s} {len(sel):7d} {len(w):5d} {len(w)/max(len(sel),1)*100:6.1f}% "
          f"{sel['pnl_pct'].sum():9.2f}% {pnl_trade:9.2f}%")

# Simular: max N por dia, mas PRIORIZANDO por confidence (melhores primeiro)
print("\n--- Simulacao: max N por dia PRIORIZANDO por confianca ---")
print(f"  {'MaxDia':>6s} {'N_exec':>7s} {'Wins':>5s} {'WR':>7s} {'PnL':>10s} {'PnL/trade':>10s}")

for max_per_day in [1, 2, 3, 4, 5, 6, 8, 10, 999]:
    selected = []
    for date, grp in clean.groupby("date"):
        top = grp.sort_values("confidence", ascending=False).head(max_per_day)
        selected.append(top)
    if not selected:
        continue
    sel = pd.concat(selected)
    w = sel[sel["is_winner"] == True]
    label = "ALL" if max_per_day > 100 else str(max_per_day)
    pnl_trade = sel["pnl_pct"].mean()
    print(f"  {label:>6s} {len(sel):7d} {len(w):5d} {len(w)/max(len(sel),1)*100:6.1f}% "
          f"{sel['pnl_pct'].sum():9.2f}% {pnl_trade:9.2f}%")

# Analise por HORA do dia
print("\n" + "=" * 80)
print("5. MELHORES HORAS DO DIA (dados limpos)")
print("=" * 80)

clean["hour"] = clean["ts"].dt.hour
print(f"\n{'Hora':>5s} {'N':>5s} {'WR':>7s} {'PnL':>10s}")
for h in range(24):
    sub = clean[clean["hour"] == h]
    if len(sub) >= 5:
        w = sub[sub["is_winner"] == True]
        print(f"  {h:3d}h {len(sub):5d} {len(w)/max(len(sub),1)*100:6.1f}% {sub['pnl_pct'].sum():9.2f}%")

# Analise BUY vs SELL por regime de mercado
print("\n" + "=" * 80)
print("6. BUY vs SELL POR REGIME (dados limpos)")
print("=" * 80)

if "market_regime_base" in clean.columns:
    for reg in clean["market_regime_base"].dropna().unique():
        sub = clean[clean["market_regime_base"] == reg]
        if len(sub) < 10:
            continue
        for d in ["BUY", "SELL"]:
            ds = sub[sub["signal"] == d]
            if len(ds) < 5:
                continue
            w = ds[ds["is_winner"] == True]
            print(f"  {reg:12s} {d}: N={len(ds)}, WR={len(w)/max(len(ds),1)*100:.1f}%, PnL={ds['pnl_pct'].sum():.1f}%")

# Calcular: quantos sinais simultaneos maximizam PnL
print("\n" + "=" * 80)
print("7. POSICOES SIMULTANEAS OPTIMAS (simulacao)")
print("=" * 80)

# Simular: janela rolante de 24h, max N posicoes abertas
print(f"\n{'MaxPos':>6s} {'N_exec':>7s} {'WR':>7s} {'PnL':>10s} {'MaxDD':>10s}")

for max_pos in [1, 2, 3, 4, 5, 6, 8]:
    open_positions = []
    executed = []

    for _, sig in clean.iterrows():
        ts = sig["ts"]
        # Fechar posicoes que ja expirariam (assume 24h max)
        open_positions = [p for p in open_positions if (ts - p["ts"]) < timedelta(hours=24)]

        if len(open_positions) < max_pos:
            open_positions.append(sig)
            executed.append(sig)

    if not executed:
        continue
    ex = pd.DataFrame(executed)
    w = ex[ex["is_winner"] == True]
    # Calcular max drawdown simples
    cumsum = ex["pnl_pct"].cumsum()
    peak = cumsum.cummax()
    dd = (cumsum - peak).min()

    print(f"  {max_pos:5d} {len(ex):7d} {len(w)/max(len(ex),1)*100:6.1f}% "
          f"{ex['pnl_pct'].sum():9.2f}% {dd:9.2f}%")

print("\n[FIM]")
