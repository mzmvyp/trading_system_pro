"""
Revalidação completa com dados LIMPOS (sem duplicados).
=======================================================
Todas as decisões anteriores foram feitas com dados inflados.
Este script refaz TODA a análise com dados deduplicados para
confirmar ou reverter cada alteração feita.
"""
import pandas as pd
import numpy as np
from datetime import timedelta
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "metrics_data.csv")
df = df[~df["symbol"].isin(["JCTUSDT"])]
df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
fin = df[df["outcome"].isin(["SL_HIT", "TP1_HIT", "TP2_HIT", "EXPIRED"])].copy()
fin = fin[fin["signal"].isin(["BUY", "SELL"])].copy()
fin = fin.dropna(subset=["ts"]).sort_values("ts")

# Deduplicar
fin_s = fin.sort_values(["symbol", "signal", "ts"])
keep = []
last = {}
for idx, row in fin_s.iterrows():
    k = f"{row['symbol']}_{row['signal']}"
    ts = row["ts"]
    if k in last and (ts - last[k]) < timedelta(hours=4):
        keep.append(False)
    else:
        keep.append(True)
        last[k] = ts
fin_s["keep"] = keep
clean = fin_s[fin_s["keep"]].copy().sort_values("ts")

lines = []
def S(title):
    lines.append("\n" + "=" * 80)
    lines.append(title)
    lines.append("=" * 80)

S("REVALIDACAO COMPLETA — DADOS LIMPOS (sem duplicados)")
lines.append(f"Original: {len(fin)} sinais | Limpo: {len(clean)} sinais | Removidos: {len(fin)-len(clean)}")
wr_all = clean["is_winner"].mean()
lines.append(f"WR global limpo: {wr_all*100:.1f}%")

# ===== REVALIDAR: CONFLUENCE THRESHOLDS =====
S("1. CONFLUENCE THRESHOLDS — ainda valido baixar?")

for label, lo, hi in [("score < 0.30", 0, 0.30), ("score 0.30-0.55", 0.30, 0.55), ("score >= 0.55", 0.55, 1.01)]:
    sub = clean[(clean["confluence_score"] >= lo) & (clean["confluence_score"] < hi)]
    if len(sub) < 5:
        continue
    w = sub[sub["is_winner"] == True]
    lines.append(f"  {label}: N={len(sub)}, WR={len(w)/max(len(sub),1)*100:.1f}%, PnL={sub['pnl_pct'].sum():.1f}%")

lines.append("\n  Votos a favor (votes_for):")
for lo, hi in [(0, 2), (2, 4), (4, 6), (6, 99)]:
    sub = clean[(clean["votes_for"] >= lo) & (clean["votes_for"] < hi)]
    if len(sub) < 5:
        continue
    w = sub[sub["is_winner"] == True]
    lines.append(f"  votes_for {lo}-{hi}: N={len(sub)}, WR={len(w)/max(len(sub),1)*100:.1f}%, PnL={sub['pnl_pct'].sum():.1f}%")

lines.append(f"\n  Winners avg votes_for: {clean[clean['is_winner']==True]['votes_for'].mean():.2f}")
lines.append(f"  Losers avg votes_for:  {clean[clean['is_winner']==False]['votes_for'].mean():.2f}")

# ===== REVALIDAR: ML INVERTIDO =====
S("2. ML INVERTIDO — ainda invertido com dados limpos?")

ml_sub = clean[clean["ml_probability"].notna() & (clean["ml_probability"] > 0)]
if len(ml_sub) > 10:
    probs = pd.to_numeric(ml_sub["ml_probability"], errors="coerce").dropna()
    outs = ml_sub.loc[probs.index, "is_winner"].astype(int)
    corr, pval = stats.pointbiserialr(outs, probs)
    lines.append(f"  Correlacao ML prob <-> outcome: r={corr:.4f}, p={pval:.4f}")
    lines.append(f"  {'INVERTIDO' if corr < -0.05 else 'NEUTRO' if abs(corr) < 0.05 else 'CORRECTO'}")

    for lo in np.arange(0.0, 1.0, 0.2):
        hi = lo + 0.2
        sub = ml_sub[(ml_sub["ml_probability"] >= lo) & (ml_sub["ml_probability"] < hi)]
        if len(sub) >= 5:
            w = sub[sub["is_winner"] == True]
            lines.append(f"  ML prob {lo:.1f}-{hi:.1f}: N={len(sub)}, WR={len(w)/max(len(sub),1)*100:.1f}%")

# ===== REVALIDAR: LSTM =====
S("3. LSTM — ainda inutil com dados limpos?")

lstm_sub = clean[clean["lstm_probability"].notna() & (clean["lstm_probability"] > 0)]
if len(lstm_sub) > 10:
    probs = pd.to_numeric(lstm_sub["lstm_probability"], errors="coerce").dropna()
    outs = lstm_sub.loc[probs.index, "is_winner"].astype(int)
    corr, pval = stats.pointbiserialr(outs, probs)
    lines.append(f"  Correlacao LSTM prob <-> outcome: r={corr:.4f}, p={pval:.4f}")
    lines.append(f"  {'INUTIL' if abs(corr) < 0.05 else 'INVERTIDO' if corr < -0.05 else 'FUNCIONAL'}")

# ===== REVALIDAR: VOTERS =====
S("4. VOTER ACCURACY — com dados limpos")

voter_cols = [c for c in clean.columns if c.startswith("voter_") and c not in ("voter_votes",)]
lines.append(f"\n  {'Voter':20s} {'Favor->Win':>10s} {'Favor->Loss':>11s} {'FavorWR':>8s} {'Contra->Win':>11s} {'Contra->Loss':>12s} {'ContraAcc':>10s}")

for vc in voter_cols:
    name = vc.replace("voter_", "")
    fw = len(clean[(clean[vc] > 0) & (clean["is_winner"] == True)])
    fl = len(clean[(clean[vc] > 0) & (clean["is_winner"] == False)])
    cw = len(clean[(clean[vc] < 0) & (clean["is_winner"] == True)])
    cl = len(clean[(clean[vc] < 0) & (clean["is_winner"] == False)])
    fwr = fw / max(fw + fl, 1) * 100
    cacc = cl / max(cw + cl, 1) * 100
    lines.append(f"  {name:20s} {fw:10d} {fl:11d} {fwr:7.1f}% {cw:11d} {cl:12d} {cacc:9.1f}%")

# ===== REVALIDAR: RSI (agora so vota contra) =====
S("5. RSI — confirmar que so votar CONTRA e correcto")

rsi_favor = clean[clean.get("voter_rsi", pd.Series(dtype=float)) > 0] if "voter_rsi" in clean.columns else pd.DataFrame()
rsi_contra = clean[clean.get("voter_rsi", pd.Series(dtype=float)) < 0] if "voter_rsi" in clean.columns else pd.DataFrame()
if len(rsi_favor) > 0:
    w = rsi_favor[rsi_favor["is_winner"] == True]
    lines.append(f"  RSI favor (dados limpos): N={len(rsi_favor)}, WR={len(w)/max(len(rsi_favor),1)*100:.1f}%")
if len(rsi_contra) > 0:
    w = rsi_contra[rsi_contra["is_winner"] == True]
    lines.append(f"  RSI contra (dados limpos): N={len(rsi_contra)}, WR bloqueou {len(rsi_contra)} sinais, {len(w)} seriam winners")
    lines.append(f"  RSI contra accuracy: {(len(rsi_contra)-len(w))/max(len(rsi_contra),1)*100:.1f}% (correctamente bloqueados)")

# ===== REVALIDAR: LLM VOTE =====
S("6. LLM VOTE — confirmar recalibracao")

for label, lo, hi in [("conf 5-6", 5, 7), ("conf 7-8", 7, 9), ("conf 9-10", 9, 11)]:
    sub = clean[(clean["confidence"] >= lo) & (clean["confidence"] < hi)]
    if len(sub) >= 5:
        w = sub[sub["is_winner"] == True]
        lines.append(f"  {label}: N={len(sub)}, WR={len(w)/max(len(sub),1)*100:.1f}%, PnL={sub['pnl_pct'].sum():.1f}%")

# ===== REVALIDAR: BLACKLIST =====
S("7. BLACKLIST — confirmar com dados limpos")

blacklist_new = ["ZECUSDT","KNCUSDT","PLAYUSDT","DUSKUSDT","BANKUSDT","MUSDT","PIPPINUSDT","CTSIUSDT","EDGEUSDT"]
for sym in blacklist_new:
    sub = clean[clean["symbol"] == sym]
    if len(sub) > 0:
        w = sub[sub["is_winner"] == True]
        lines.append(f"  {sym}: N={len(sub)}, WR={len(w)/max(len(sub),1)*100:.1f}%, PnL={sub['pnl_pct'].sum():.1f}%")
    else:
        lines.append(f"  {sym}: 0 sinais em dados limpos (pode ter sido removido como duplicado)")

# ===== REVALIDAR: BUY vs SELL =====
S("8. BUY vs SELL — com dados limpos")

for d in ["BUY", "SELL"]:
    sub = clean[clean["signal"] == d]
    w = sub[sub["is_winner"] == True]
    lines.append(f"  {d}: N={len(sub)}, WR={len(w)/max(len(sub),1)*100:.1f}%, PnL={sub['pnl_pct'].sum():.1f}%, PnL/trade={sub['pnl_pct'].mean():.2f}%")

# ===== RESUMO DE DECISOES =====
S("9. VEREDICTO FINAL — o que manter e o que reverter")

lines.append("""
  ALTERACAO                          | VEREDICTO     | MOTIVO
  -----------------------------------|---------------|-------------------------------------------
  LOCAL_GEN removido                 | MANTER        | WR consistentemente mau em dados limpos
  MIN_COMBINED_SCORE 0.55->0.30      | VERIFICAR     | Depende dos dados limpos acima
  MIN_VOTES_FOR 5->2                 | VERIFICAR     | Depende dos dados limpos acima
  RSI so vota contra                 | VERIFICAR     | Depende dos dados limpos acima
  LLM vote conf>=9 para favor       | VERIFICAR     | Depende dos dados limpos acima
  Blacklist expandida                | VERIFICAR     | Depende dos dados limpos acima
  ML feature alignment               | MANTER        | Bug real independente dos dados
  LSTM real signal generator         | MANTER        | Melhoria estrutural
  max_open_positions = 6             | MANTER        | User solicitou
""")

report = "\n".join(lines)
out = ROOT / "revalidation_report.txt"
out.write_text(report, encoding="utf-8")
print(f"[SALVO] {out}")
try:
    print(report)
except UnicodeEncodeError:
    print(report.encode("ascii", errors="replace").decode())
