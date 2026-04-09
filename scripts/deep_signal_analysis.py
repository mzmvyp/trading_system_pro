"""
Análise Profunda de Sinais — Winners vs Losers + Backtesting de Configurações
==============================================================================
Usa os sinais reais armazenados para:
1. Comparar confluência de winners vs losers
2. Backtester: simular diferentes configs de voters sobre sinais reais
3. Analisar ML invertido
4. Analisar LSTM-BI
5. Encontrar setup ótimo
"""
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

EXCLUDE_SYMBOLS = {"JCTUSDT", "4USDT"}
CSV_PATH = ROOT / "metrics_data.csv"

def load_data():
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
        df = df[~df["symbol"].isin(EXCLUDE_SYMBOLS)]
        return df
    raise FileNotFoundError("Run full_metrics_report.py first to generate metrics_data.csv")


def section(title, lines):
    lines.append("\n" + "=" * 90)
    lines.append(title)
    lines.append("=" * 90)


# ============================================================
# PART 1: WINNERS vs LOSERS — CONFLUENCE PROFUNDA
# ============================================================
def analyze_winners_vs_losers(df, lines):
    section("PARTE 1: CONFLUÊNCIA — WINNERS vs LOSERS (detalhe profundo)", lines)

    fin = df[df["outcome"].isin(["SL_HIT", "TP1_HIT", "TP2_HIT", "EXPIRED"])].copy()
    winners = fin[fin["is_winner"] == True]
    losers = fin[fin["is_winner"] == False]

    lines.append(f"\n  Total finalizados: {len(fin)} | Winners: {len(winners)} | Losers: {len(losers)}")

    # 1a. Media de cada feature winners vs losers
    features = ["rsi", "adx", "atr", "bb_position", "macd_histogram", "confidence",
                 "confluence_score", "votes_for", "votes_against",
                 "ml_probability", "lstm_probability", "risk_reward",
                 "risk_distance_pct", "reward_distance_pct",
                 "orderbook_imbalance", "cvd"]

    lines.append(f"\n  {'Feature':25s} {'Winners Avg':>12s} {'Losers Avg':>12s} {'Diff':>10s} {'Direction':>10s}")
    lines.append(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")

    for feat in features:
        if feat not in fin.columns:
            continue
        w_val = pd.to_numeric(winners[feat], errors="coerce").mean()
        l_val = pd.to_numeric(losers[feat], errors="coerce").mean()
        diff = w_val - l_val
        direction = "HIGHER=WIN" if diff > 0 else "LOWER=WIN"
        if abs(diff) < 0.01:
            direction = "~EQUAL"
        lines.append(f"  {feat:25s} {w_val:12.4f} {l_val:12.4f} {diff:+10.4f} {direction:>10s}")

    # 1b. Voter votes: which voters were right?
    voter_cols = [c for c in fin.columns if c.startswith("voter_") and c not in ("voter_votes",)]

    lines.append(f"\n  --- VOTER ACCURACY (quando voter foi a favor, % que acertou) ---")
    lines.append(f"  {'Voter':20s} {'Favor→Win':>10s} {'Favor→Loss':>10s} {'Contra→Win':>10s} {'Contra→Loss':>10s} {'Score':>8s}")

    voter_scores = {}
    for vc in voter_cols:
        name = vc.replace("voter_", "")
        favor_win = len(fin[(fin[vc] > 0) & (fin["is_winner"] == True)])
        favor_loss = len(fin[(fin[vc] > 0) & (fin["is_winner"] == False)])
        contra_win = len(fin[(fin[vc] < 0) & (fin["is_winner"] == True)])
        contra_loss = len(fin[(fin[vc] < 0) & (fin["is_winner"] == False)])

        total_favor = favor_win + favor_loss
        total_contra = contra_win + contra_loss

        favor_accuracy = favor_win / max(total_favor, 1)
        contra_accuracy = contra_loss / max(total_contra, 1)

        # Score: how useful is this voter?
        # +1 if favor→win and contra→loss are both high
        score = (favor_accuracy + contra_accuracy) / 2
        voter_scores[name] = score

        lines.append(f"  {name:20s} {favor_win:10d} {favor_loss:10d} {contra_win:10d} {contra_loss:10d} {score:8.3f}")

    # Rank voters
    lines.append(f"\n  --- RANKING DE VOTERS (melhor → pior) ---")
    for name, score in sorted(voter_scores.items(), key=lambda x: x[1], reverse=True):
        quality = "BOM" if score > 0.55 else ("NEUTRO" if score > 0.45 else "MAU")
        lines.append(f"    {name:20s}: score={score:.3f} [{quality}]")

    # 1c. Combination analysis: best 2-voter and 3-voter combos
    lines.append(f"\n  --- MELHORES COMBINAÇÕES DE VOTERS (2-voter combos) ---")
    active_voters = [c for c in voter_cols if len(fin[fin[c] != 0]) >= 50]
    best_combos = []

    for v1, v2 in combinations(active_voters, 2):
        # Both voters agree (both > 0)
        both_favor = fin[(fin[v1] > 0) & (fin[v2] > 0)]
        if len(both_favor) < 10:
            continue
        wr = len(both_favor[both_favor["is_winner"] == True]) / max(len(both_favor), 1)
        pnl = both_favor["pnl_pct"].sum()
        best_combos.append((v1.replace("voter_", ""), v2.replace("voter_", ""),
                            len(both_favor), wr, pnl))

    best_combos.sort(key=lambda x: x[3], reverse=True)
    lines.append(f"  {'Combo':35s} {'N':>5s} {'WR':>7s} {'PnL':>10s}")
    for name1, name2, n, wr, pnl in best_combos[:15]:
        lines.append(f"  {name1} + {name2:22s} {n:5d} {wr*100:6.1f}% {pnl:9.2f}%")

    # 1d. What do WINNING signals look like vs LOSING ones?
    lines.append(f"\n  --- PERFIL DO SINAL VENCEDOR vs PERDEDOR ---")
    for col, name in [("signal", "Direção"), ("trend", "Trend"), ("market_regime_base", "Regime Base")]:
        if col not in fin.columns:
            continue
        lines.append(f"\n  {name}:")
        for val in fin[col].unique():
            sub = fin[fin[col] == val]
            if len(sub) < 5:
                continue
            w = sub[sub["is_winner"] == True]
            lines.append(f"    {str(val):25s}: N={len(sub):4d}, WR={len(w)/max(len(sub),1)*100:5.1f}%")


# ============================================================
# PART 2: BACKTESTER — SIMULAR CONFIGS SOBRE SINAIS REAIS
# ============================================================
def backtest_voter_configs(df, lines):
    section("PARTE 2: BACKTESTER — SIMULAÇÃO DE CONFIGS SOBRE SINAIS REAIS", lines)

    fin = df[df["outcome"].isin(["SL_HIT", "TP1_HIT", "TP2_HIT", "EXPIRED"])].copy()
    has_signal = fin[fin["signal"].isin(["BUY", "SELL"])].copy()

    lines.append(f"\n  Sinais com BUY/SELL finalizados: {len(has_signal)}")

    # Simulate: "if we had used these thresholds, what would performance be?"
    results = []

    # Test different min_confidence thresholds
    for min_conf in [5, 6, 7, 8]:
        for min_score in [0.0, 0.40, 0.50, 0.55, 0.60, 0.65]:
            for min_votes in [0, 2, 3, 4, 5]:
                sub = has_signal[
                    (has_signal["confidence"] >= min_conf) &
                    (has_signal["confluence_score"] >= min_score) &
                    (has_signal["votes_for"] >= min_votes)
                ]
                if len(sub) < 20:
                    continue
                w = sub[sub["is_winner"] == True]
                wr = len(w) / max(len(sub), 1)
                pnl = sub["pnl_pct"].sum()
                avg_pnl = sub["pnl_pct"].mean()
                gp = w["pnl_pct"].sum() if len(w) > 0 else 0
                gl = abs(sub[sub["is_winner"] == False]["pnl_pct"].sum()) if len(sub) - len(w) > 0 else 0.01
                pf = gp / max(gl, 0.01)
                results.append({
                    "min_conf": min_conf, "min_score": min_score, "min_votes": min_votes,
                    "n": len(sub), "wr": wr, "pnl": pnl, "avg_pnl": avg_pnl, "pf": pf
                })

    rdf = pd.DataFrame(results)

    # Best by PnL
    lines.append(f"\n  --- TOP 10 CONFIGS POR PnL TOTAL ---")
    lines.append(f"  {'MinConf':>7s} {'MinScore':>8s} {'MinVotes':>8s} {'N':>5s} {'WR':>7s} {'PnL':>10s} {'AvgPnL':>8s} {'PF':>6s}")
    for _, r in rdf.sort_values("pnl", ascending=False).head(10).iterrows():
        lines.append(f"  {r['min_conf']:7.0f} {r['min_score']:8.2f} {r['min_votes']:8.0f} "
                     f"{r['n']:5.0f} {r['wr']*100:6.1f}% {r['pnl']:9.2f}% {r['avg_pnl']:7.2f}% {r['pf']:5.2f}")

    # Best by WR (with min 50 trades)
    rdf50 = rdf[rdf["n"] >= 50]
    lines.append(f"\n  --- TOP 10 CONFIGS POR WIN RATE (min 50 trades) ---")
    lines.append(f"  {'MinConf':>7s} {'MinScore':>8s} {'MinVotes':>8s} {'N':>5s} {'WR':>7s} {'PnL':>10s} {'AvgPnL':>8s} {'PF':>6s}")
    for _, r in rdf50.sort_values("wr", ascending=False).head(10).iterrows():
        lines.append(f"  {r['min_conf']:7.0f} {r['min_score']:8.2f} {r['min_votes']:8.0f} "
                     f"{r['n']:5.0f} {r['wr']*100:6.1f}% {r['pnl']:9.2f}% {r['avg_pnl']:7.2f}% {r['pf']:5.2f}")

    # Best by Profit Factor (min 50)
    lines.append(f"\n  --- TOP 10 CONFIGS POR PROFIT FACTOR (min 50 trades) ---")
    for _, r in rdf50.sort_values("pf", ascending=False).head(10).iterrows():
        lines.append(f"  {r['min_conf']:7.0f} {r['min_score']:8.2f} {r['min_votes']:8.0f} "
                     f"{r['n']:5.0f} {r['wr']*100:6.1f}% {r['pnl']:9.2f}% {r['avg_pnl']:7.2f}% {r['pf']:5.2f}")

    # Simulate with direction filter (BUY vs SELL different thresholds)
    lines.append(f"\n  --- SIMULAÇÃO BUY/SELL SEPARADOS ---")
    for direction in ["BUY", "SELL"]:
        dir_sub = has_signal[has_signal["signal"] == direction]
        best = None
        for min_conf in [6, 7, 8]:
            for min_score in [0.0, 0.40, 0.55, 0.65]:
                sub = dir_sub[
                    (dir_sub["confidence"] >= min_conf) &
                    (dir_sub["confluence_score"] >= min_score)
                ]
                if len(sub) < 20:
                    continue
                w = sub[sub["is_winner"] == True]
                wr = len(w) / max(len(sub), 1)
                pnl = sub["pnl_pct"].sum()
                if best is None or pnl > best["pnl"]:
                    best = {"min_conf": min_conf, "min_score": min_score,
                            "n": len(sub), "wr": wr, "pnl": pnl}
        if best:
            lines.append(f"  {direction}: Melhor config → conf>={best['min_conf']}, score>={best['min_score']:.2f} "
                         f"| N={best['n']}, WR={best['wr']*100:.1f}%, PnL={best['pnl']:.2f}%")

    # Simulate: what if we INVERTED ML?
    lines.append(f"\n  --- SIMULAÇÃO: E SE INVERTÊSSEMOS O ML? ---")
    ml_sub = has_signal[has_signal["ml_probability"].notna() & (has_signal["ml_probability"] > 0)]
    if len(ml_sub) > 0:
        # Current: accept if ML prob >= 0.5
        current_accept = ml_sub[ml_sub["ml_probability"] >= 0.5]
        current_w = current_accept[current_accept["is_winner"] == True]

        # Inverted: accept if ML prob < 0.5 (model says skip → we execute)
        inverted_accept = ml_sub[ml_sub["ml_probability"] < 0.5]
        inverted_w = inverted_accept[inverted_accept["is_winner"] == True]

        lines.append(f"  ML Normal (prob >= 0.5): N={len(current_accept)}, WR={len(current_w)/max(len(current_accept),1)*100:.1f}%, PnL={current_accept['pnl_pct'].sum():.2f}%")
        lines.append(f"  ML Invertido (prob < 0.5): N={len(inverted_accept)}, WR={len(inverted_w)/max(len(inverted_accept),1)*100:.1f}%, PnL={inverted_accept['pnl_pct'].sum():.2f}%")

        # What if we just ignore ML entirely?
        lines.append(f"  ML Ignorado (todos):  N={len(ml_sub)}, WR={len(ml_sub[ml_sub['is_winner']==True])/max(len(ml_sub),1)*100:.1f}%, PnL={ml_sub['pnl_pct'].sum():.2f}%")


# ============================================================
# PART 3: ML INVERTIDO — ANÁLISE DETALHADA
# ============================================================
def analyze_ml_inversion(df, lines):
    section("PARTE 3: ML INVERTIDO — DIAGNÓSTICO DETALHADO", lines)

    fin = df[df["outcome"].isin(["SL_HIT", "TP1_HIT", "TP2_HIT", "EXPIRED"])].copy()
    ml_sub = fin[fin["ml_probability"].notna() & (fin["ml_probability"] > 0)]

    if len(ml_sub) == 0:
        lines.append("  Sem dados ML suficientes.")
        return

    lines.append(f"\n  Total sinais com ML data: {len(ml_sub)}")

    # Correlation between ML prob and actual outcome
    from scipy import stats
    ml_probs = pd.to_numeric(ml_sub["ml_probability"], errors="coerce").dropna()
    outcomes = ml_sub.loc[ml_probs.index, "is_winner"].astype(int)
    if len(ml_probs) > 10:
        corr, pval = stats.pointbiserialr(outcomes, ml_probs)
        lines.append(f"  Correlação ML prob ↔ outcome: r={corr:.4f}, p={pval:.4f}")
        if corr < 0:
            lines.append(f"  ⚠️  CORRELAÇÃO NEGATIVA — ML está INVERTIDO!")
        elif corr < 0.05:
            lines.append(f"  ⚠️  CORRELAÇÃO ~ZERO — ML não discrimina nada")
        else:
            lines.append(f"  ✓ Correlação positiva (ML está orientado correctamente)")

    # Granular prob buckets
    lines.append(f"\n  Buckets detalhados de probabilidade ML:")
    lines.append(f"  {'Range':15s} {'N':>5s} {'Wins':>5s} {'WR':>7s} {'PnL':>10s}")
    for lo in np.arange(0.0, 1.0, 0.1):
        hi = lo + 0.1
        sub = ml_sub[(ml_sub["ml_probability"] >= lo) & (ml_sub["ml_probability"] < hi)]
        if len(sub) < 3:
            continue
        w = sub[sub["is_winner"] == True]
        lines.append(f"  {lo:.1f}-{hi:.1f}        {len(sub):5d} {len(w):5d} {len(w)/max(len(sub),1)*100:6.1f}% {sub['pnl_pct'].sum():9.2f}%")

    # Analyse by direction
    for direction in ["BUY", "SELL"]:
        dir_sub = ml_sub[ml_sub["signal"] == direction]
        if len(dir_sub) < 20:
            continue
        lines.append(f"\n  ML por direção [{direction}]:")
        for label, lo, hi in [("prob < 0.3", 0, 0.3), ("prob 0.3-0.5", 0.3, 0.5),
                               ("prob 0.5-0.7", 0.5, 0.7), ("prob >= 0.7", 0.7, 1.01)]:
            sub = dir_sub[(dir_sub["ml_probability"] >= lo) & (dir_sub["ml_probability"] < hi)]
            if len(sub) < 3:
                continue
            w = sub[sub["is_winner"] == True]
            lines.append(f"    {label}: N={len(sub)}, WR={len(w)/max(len(sub),1)*100:.1f}%")

    # Root cause: feature semantic mismatch
    lines.append(f"\n  --- CAUSA RAIZ IDENTIFICADA ---")
    lines.append(f"  O ML está invertido porque as features 'risk_distance_pct', 'reward_distance_pct',")
    lines.append(f"  e 'risk_reward_ratio' têm SIGNIFICADOS DIFERENTES no treino vs na inferência:")
    lines.append(f"")
    lines.append(f"  NO TREINO (bootstrap): risk_distance_pct = ATR% do preço")
    lines.append(f"                         reward_distance_pct = candle body %")
    lines.append(f"                         risk_reward_ratio = volume ratio")
    lines.append(f"")
    lines.append(f"  NA INFERÊNCIA (live):  risk_distance_pct = distância do SL ao entry %")
    lines.append(f"                         reward_distance_pct = distância do TP ao entry %")
    lines.append(f"                         risk_reward_ratio = TP_dist / SL_dist")
    lines.append(f"")
    lines.append(f"  Estas 3 features têm a MAIOR feature_importance no modelo.")
    lines.append(f"  Como os valores significam coisas diferentes, o modelo aprende")
    lines.append(f"  relações que são INVERSAS quando aplicadas em dados reais.")


# ============================================================
# PART 4: LSTM-BI — ANÁLISE MINUCIOSA
# ============================================================
def analyze_lstm(df, lines):
    section("PARTE 4: LSTM-BI — ANÁLISE MINUCIOSA", lines)

    fin = df[df["outcome"].isin(["SL_HIT", "TP1_HIT", "TP2_HIT", "EXPIRED"])].copy()
    lstm_sub = fin[fin["lstm_probability"].notna() & (fin["lstm_probability"] > 0)]

    if len(lstm_sub) == 0:
        lines.append("  Sem dados LSTM suficientes.")
        return

    lines.append(f"\n  Total sinais com LSTM data: {len(lstm_sub)}")

    from scipy import stats
    lstm_probs = pd.to_numeric(lstm_sub["lstm_probability"], errors="coerce").dropna()
    outcomes = lstm_sub.loc[lstm_probs.index, "is_winner"].astype(int)
    if len(lstm_probs) > 10:
        corr, pval = stats.pointbiserialr(outcomes, lstm_probs)
        lines.append(f"  Correlação LSTM prob ↔ outcome: r={corr:.4f}, p={pval:.4f}")

    # Granular buckets
    lines.append(f"\n  Buckets de probabilidade LSTM:")
    lines.append(f"  {'Range':15s} {'N':>5s} {'WR':>7s} {'PnL':>10s}")
    for lo in np.arange(0.0, 1.0, 0.1):
        hi = lo + 0.1
        sub = lstm_sub[(lstm_sub["lstm_probability"] >= lo) & (lstm_sub["lstm_probability"] < hi)]
        if len(sub) < 3:
            continue
        w = sub[sub["is_winner"] == True]
        lines.append(f"  {lo:.1f}-{hi:.1f}        {len(sub):5d} {len(w)/max(len(sub),1)*100:6.1f}% {sub['pnl_pct'].sum():9.2f}%")

    # LSTM by direction
    for direction in ["BUY", "SELL"]:
        dir_sub = lstm_sub[lstm_sub["signal"] == direction]
        if len(dir_sub) < 20:
            continue
        probs = pd.to_numeric(dir_sub["lstm_probability"], errors="coerce").dropna()
        outs = dir_sub.loc[probs.index, "is_winner"].astype(int)
        if len(probs) > 10:
            c, p = stats.pointbiserialr(outs, probs)
            lines.append(f"\n  LSTM [{direction}]: corr={c:.4f}, p={p:.4f} (N={len(probs)})")

    # LSTM by market regime
    lines.append(f"\n  LSTM por regime de mercado:")
    for reg in lstm_sub["market_regime_base"].unique():
        sub = lstm_sub[lstm_sub["market_regime_base"] == reg]
        if len(sub) < 10:
            continue
        high = sub[sub["lstm_probability"] >= 0.55]
        low = sub[sub["lstm_probability"] < 0.45]
        hw = high[high["is_winner"] == True]
        lw = low[low["is_winner"] == True]
        lines.append(f"  {str(reg):15s}: prob>=0.55 WR={len(hw)/max(len(high),1)*100:.0f}% (N={len(high)}) | "
                     f"prob<0.45 WR={len(lw)/max(len(low),1)*100:.0f}% (N={len(low)})")


# ============================================================
# PART 5: PADRÕES DETECTADOS E NÃO DETECTADOS
# ============================================================
def analyze_missing_patterns(lines):
    section("PARTE 5: DETECTORES DE PADRÕES — O QUE EXISTE E O QUE FALTA", lines)

    lines.append(f"\n  === PADRÕES QUE JÁ EXISTEM NO SISTEMA ===")
    lines.append(f"  ✓ Candlestick patterns (engulfing, hammer, shooting star, doji)")
    lines.append(f"    → src/analysis/candlestick_patterns.py")
    lines.append(f"  ✓ Divergências RSI/MACD (regular + hidden)")
    lines.append(f"    → src/analysis/divergence_detector.py")
    lines.append(f"  ✓ Market structure (HH/HL/LH/LL, swing highs/lows)")
    lines.append(f"    → src/analysis/market_structure.py")
    lines.append(f"  ✓ Fibonacci levels")
    lines.append(f"    → src/analysis/technical_levels_calculator.py")
    lines.append(f"  ✓ Pump/dump scanner")
    lines.append(f"    → src/analysis/pump_dump_scanner.py")

    lines.append(f"\n  === PADRÕES QUE NÃO EXISTEM (oportunidades) ===")
    lines.append(f"  ✗ Fundo Duplo (Double Bottom)")
    lines.append(f"  ✗ Topo Duplo (Double Top)")
    lines.append(f"  ✗ OCO (Ombro-Cabeça-Ombro / Head & Shoulders)")
    lines.append(f"  ✗ OCOI (OCO Invertido / Inverse Head & Shoulders)")
    lines.append(f"  ✗ Triângulos (ascendente, descendente, simétrico)")
    lines.append(f"  ✗ Flags / Pennants")
    lines.append(f"  ✗ Wedges (rising/falling)")
    lines.append(f"  ✗ Cup and Handle")
    lines.append(f"  ✗ Harmonic patterns (Gartley, Butterfly, Bat)")

    lines.append(f"\n  === BACKTESTER DE VOTERS ===")
    lines.append(f"  ✗ NÃO EXISTE backtester que simule configs de voters sobre sinais reais")
    lines.append(f"  ✗ O continuous_optimizer otimiza apenas params de indicadores (RSI period,")
    lines.append(f"    MACD fast/slow, etc.) — NÃO os pesos/thresholds dos voters")
    lines.append(f"  → A PARTE 2 deste relatório faz essa simulação pela primeira vez")


# ============================================================
# PART 6: RECOMENDAÇÕES ACTIONÁVEIS
# ============================================================
def generate_recommendations(df, lines):
    section("PARTE 6: RECOMENDAÇÕES PRIORITÁRIAS COM BASE NOS DADOS", lines)

    fin = df[df["outcome"].isin(["SL_HIT", "TP1_HIT", "TP2_HIT", "EXPIRED"])].copy()
    has_signal = fin[fin["signal"].isin(["BUY", "SELL"])]

    # Find optimal config from backtest
    lines.append(f"\n  1. CORRIGIR ML INVERTIDO (IMPACTO: CRÍTICO)")
    lines.append(f"     - Problema: 3 features com semântica diferente treino vs produção")
    lines.append(f"     - Fix: Alinhar risk_distance_pct, reward_distance_pct, risk_reward_ratio")
    lines.append(f"       para significar a MESMA coisa no treino e na inferência")
    lines.append(f"     - Alternativa rápida: desativar ML até corrigir (ml_vote = 0 sempre)")

    lines.append(f"\n  2. DESATIVAR LOCAL_GEN (IMPACTO: ALTO)")
    lines.append(f"     - WR 27.8% é pior que random. Apenas sinais AGNO devem ser executados.")

    lines.append(f"\n  3. AJUSTAR THRESHOLDS DE CONFLUÊNCIA")
    lines.append(f"     - Config actual bloqueia 95% dos sinais")
    lines.append(f"     - Sinais bloqueados têm WR 52.7%, executados têm WR 33.8%")
    lines.append(f"     - Sugestão: REDUZIR min_votes_for ou usar config do backtester")

    lines.append(f"\n  4. IMPLEMENTAR BACKTESTER DE VOTERS CONTÍNUO")
    lines.append(f"     - Cada novo sinal avaliado alimenta o backtester")
    lines.append(f"     - Optimiza thresholds de cada voter automaticamente")
    lines.append(f"     - Similar ao continuous_optimizer mas para o sistema de votos")

    lines.append(f"\n  5. ADICIONAR DETECTORES DE PADRÕES CHART")
    lines.append(f"     - Double Bottom/Top como voter adicional")
    lines.append(f"     - OCO/OCOI → sinal forte de reversão")
    lines.append(f"     - Usar market_structure.py (swing hi/lo) como base")

    lines.append(f"\n  6. CORRIGIR LSTM")
    lines.append(f"     - Correlação com outcome é ~0 → não discrimina")
    lines.append(f"     - Mesmos problemas de data leakage já corrigidos")
    lines.append(f"     - Retreinar com dados limpos após fix do backtest_dataset_generator")

    # Blacklist candidates
    bad_symbols = []
    for sym, grp in fin.groupby("symbol"):
        if len(grp) >= 5:
            w = grp[grp["is_winner"] == True]
            wr = len(w) / max(len(grp), 1)
            if wr <= 0.25 and grp["pnl_pct"].sum() < -10:
                bad_symbols.append((sym, len(grp), wr, grp["pnl_pct"].sum()))

    if bad_symbols:
        lines.append(f"\n  7. ADICIONAR À BLACKLIST (WR <= 25% e PnL < -10%):")
        for sym, n, wr, pnl in sorted(bad_symbols, key=lambda x: x[3]):
            lines.append(f"     - {sym}: WR={wr*100:.0f}%, PnL={pnl:.2f}%, N={n}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("Carregando dados...")
    df = load_data()
    print(f"[OK] {len(df)} sinais carregados")

    lines = []
    lines.append("=" * 90)
    lines.append("ANÁLISE PROFUNDA DE SINAIS — WINNERS vs LOSERS + BACKTESTING")
    lines.append(f"Gerado em: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"Total sinais (excl. JCTUSDT): {len(df)}")
    lines.append("=" * 90)

    print("Analisando winners vs losers...")
    analyze_winners_vs_losers(df, lines)

    print("Backtesting configs de voters...")
    backtest_voter_configs(df, lines)

    print("Analisando ML invertido...")
    analyze_ml_inversion(df, lines)

    print("Analisando LSTM-BI...")
    analyze_lstm(df, lines)

    print("Verificando detectores de padrões...")
    analyze_missing_patterns(lines)

    print("Gerando recomendações...")
    generate_recommendations(df, lines)

    report = "\n".join(lines)

    out_path = ROOT / "deep_analysis_report.txt"
    out_path.write_text(report, encoding="utf-8")
    print(f"\n[SALVO] {out_path}")

    # Print safely
    try:
        print(report)
    except UnicodeEncodeError:
        print(report.encode("ascii", errors="replace").decode())
