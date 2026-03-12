"""
Backtesting - Simulacao de Performance Historica
=================================================

Pagina de backtesting que avalia como o modelo ML teria
performado em sinais historicos.
"""

import sys
from pathlib import Path

import json
import os
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime

# Adicionar raiz do projeto ao path
root = Path(__file__).resolve().parent.parent.parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.trading.signal_tracker import evaluate_all_signals, get_performance_summary  # noqa: E402

st.set_page_config(page_title="Backtesting", page_icon="🔬", layout="wide")

st.title("🔬 Backtesting - Simulacao Historica")
st.markdown("Avaliacao de como o modelo ML teria filtrado sinais historicos")
st.markdown("---")


# ================================================================
# FUNCOES
# ================================================================

def load_model():
    """Carrega modelo ML e scaler"""
    models_path = "ml_models/signal_validators.pkl"
    scaler_path = "ml_models/scaler_simple.pkl"
    info_path = "ml_models/model_info_simple.json"

    if not os.path.exists(models_path):
        return None, None, None, None

    with open(models_path, 'rb') as f:
        models = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(info_path, 'r') as f:
        info = json.load(f)

    best_name = info.get('best_model', 'LogisticRegression')
    feature_cols = info.get('feature_columns', [])
    model = models.get(best_name)

    return model, scaler, feature_cols, best_name


def extract_features(signal, feature_cols):
    """Extrai features de um sinal para predicao"""
    indicators = signal.get('indicators', {})

    # Calcular campos derivados
    trend = signal.get('trend', indicators.get('trend', 'neutral'))
    trend_map = {'strong_bullish': 2, 'bullish': 1, 'neutral': 0, 'bearish': -1, 'strong_bearish': -2}
    trend_encoded = trend_map.get(str(trend).lower(), 0)

    sentiment = signal.get('sentiment', indicators.get('sentiment', 'neutral'))
    sentiment_map = {'bullish': 1, 'neutral': 0, 'bearish': -1}
    sentiment_encoded = sentiment_map.get(str(sentiment).lower(), 0)

    signal_type = signal.get('signal', 'NO_SIGNAL')
    signal_encoded = 1 if signal_type == 'BUY' else 0

    entry_price = signal.get('entry_price', 0)
    stop_loss = signal.get('stop_loss', 0)
    tp1 = signal.get('take_profit_1', 0)

    risk_distance_pct = abs(entry_price - stop_loss) / entry_price * 100 if entry_price > 0 and stop_loss > 0 else 2.0
    reward_distance_pct = abs(tp1 - entry_price) / entry_price * 100 if entry_price > 0 and tp1 > 0 else 2.0
    risk_reward_ratio = reward_distance_pct / risk_distance_pct if risk_distance_pct > 0 else 1.0

    feature_map = {
        'rsi': signal.get('rsi', indicators.get('rsi', 50)),
        'macd_histogram': signal.get('macd_histogram', indicators.get('macd_histogram', 0)),
        'adx': signal.get('adx', indicators.get('adx', 25)),
        'atr': signal.get('atr', indicators.get('atr', 0)),
        'bb_position': signal.get('bb_position', indicators.get('bb_position', 0.5)),
        'cvd': signal.get('cvd', indicators.get('cvd', 0)),
        'orderbook_imbalance': signal.get('orderbook_imbalance', indicators.get('orderbook_imbalance', 0.5)),
        'bullish_tf_count': signal.get('bullish_tf_count', indicators.get('bullish_tf_count', 0)),
        'bearish_tf_count': signal.get('bearish_tf_count', indicators.get('bearish_tf_count', 0)),
        'confidence': signal.get('confidence', 5),
        'trend_encoded': trend_encoded,
        'sentiment_encoded': sentiment_encoded,
        'signal_encoded': signal_encoded,
        'risk_distance_pct': risk_distance_pct,
        'reward_distance_pct': reward_distance_pct,
        'risk_reward_ratio': risk_reward_ratio,
    }

    return [feature_map.get(col, 0) for col in feature_cols]


# ================================================================
# CARREGAR DADOS
# ================================================================

model, scaler, feature_cols, model_name = load_model()

if model is None:
    st.error("⚠️ Nenhum modelo ML treinado. Acesse 'ml dashboard' para treinar primeiro.")
    st.stop()

st.success(f"Modelo carregado: **{model_name}** ({len(feature_cols)} features)")


@st.cache_data(ttl=120)
def load_evaluations():
    return evaluate_all_signals()


with st.spinner("Carregando sinais avaliados..."):
    evaluations = load_evaluations()

if not evaluations:
    st.warning("Nenhum sinal encontrado para backtesting.")
    st.stop()

# Filtrar apenas sinais finalizados
finalized = [e for e in evaluations if e.get('outcome') in ('SL_HIT', 'TP1_HIT', 'TP2_HIT', 'EXPIRED')]

if not finalized:
    st.warning("Nenhum sinal finalizado para backtesting.")
    st.stop()

st.info(f"**{len(finalized)}** sinais finalizados encontrados para backtesting")

# ================================================================
# SIMULACAO: ML FILTER vs SEM FILTER
# ================================================================

st.header("📊 Comparativo: Com ML vs Sem ML")

# Parametros
with st.expander("⚙️ Parametros do Backtest"):
    ml_threshold = st.slider("Threshold de probabilidade ML", 0.3, 0.8, 0.5, 0.05,
                              help="Sinais com probabilidade acima deste valor serao executados")

# Rodar backtest
results_all = []  # Sem filtro ML
results_ml = []   # Com filtro ML

for sig in finalized:
    pnl = sig.get('pnl_percent', 0)
    outcome = sig.get('outcome', '')
    is_win = pnl > 0

    # Resultado sem filtro
    results_all.append({
        'timestamp': sig.get('timestamp', ''),
        'symbol': sig.get('symbol', ''),
        'signal': sig.get('signal', ''),
        'source': sig.get('source', ''),
        'confidence': sig.get('confidence', 0),
        'outcome': outcome,
        'pnl_percent': pnl,
        'ml_decision': 'N/A',
        'ml_probability': 0,
    })

    # Resultado com filtro ML
    try:
        features = extract_features(sig, feature_cols)
        X = np.array([features])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = scaler.transform(X)

        pred = model.predict(X_scaled)[0]
        prob = float(model.predict_proba(X_scaled)[0][1]) if hasattr(model, 'predict_proba') else float(pred)

        ml_decision = 'EXECUTE' if prob >= ml_threshold else 'SKIP'

        results_ml.append({
            'timestamp': sig.get('timestamp', ''),
            'symbol': sig.get('symbol', ''),
            'signal': sig.get('signal', ''),
            'source': sig.get('source', ''),
            'confidence': sig.get('confidence', 0),
            'outcome': outcome,
            'pnl_percent': pnl if ml_decision == 'EXECUTE' else 0,
            'executed': ml_decision == 'EXECUTE',
            'ml_decision': ml_decision,
            'ml_probability': prob,
            'original_pnl': pnl,
        })
    except Exception:
        results_ml.append({
            'timestamp': sig.get('timestamp', ''),
            'symbol': sig.get('symbol', ''),
            'signal': sig.get('signal', ''),
            'outcome': outcome,
            'pnl_percent': pnl,
            'executed': True,
            'ml_decision': 'ERROR',
            'ml_probability': 0,
            'original_pnl': pnl,
        })

df_all = pd.DataFrame(results_all)
df_ml = pd.DataFrame(results_ml)

# ================================================================
# METRICAS COMPARATIVAS
# ================================================================

# Sem filtro
total_all = len(df_all)
wins_all = len(df_all[df_all['pnl_percent'] > 0])
wr_all = wins_all / total_all * 100 if total_all else 0
pnl_all = df_all['pnl_percent'].sum()

# Com filtro ML
executed = df_ml[df_ml['executed'] == True]
skipped = df_ml[df_ml['executed'] == False]
total_exec = len(executed)
wins_exec = len(executed[executed['original_pnl'] > 0]) if total_exec > 0 else 0
wr_exec = wins_exec / total_exec * 100 if total_exec else 0
pnl_exec = executed['original_pnl'].sum() if total_exec > 0 else 0

# Sinais que ML teria evitado (e eram realmente ruins)
skipped_losses = len(skipped[skipped['original_pnl'] <= 0]) if len(skipped) > 0 else 0
skipped_wins = len(skipped[skipped['original_pnl'] > 0]) if len(skipped) > 0 else 0

st.subheader("Resultados")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Sem Filtro ML")
    c1, c2, c3 = st.columns(3)
    c1.metric("Trades", total_all)
    c2.metric("Win Rate", f"{wr_all:.1f}%")
    c3.metric("PnL Total", f"{pnl_all:.2f}%")

with col2:
    st.markdown("### Com Filtro ML")
    c1, c2, c3 = st.columns(3)
    c1.metric("Trades", total_exec, delta=f"{total_exec - total_all}")
    c2.metric("Win Rate", f"{wr_exec:.1f}%",
              delta=f"{wr_exec - wr_all:+.1f}pp")
    c3.metric("PnL Total", f"{pnl_exec:.2f}%",
              delta=f"{pnl_exec - pnl_all:+.2f}%")

# Sinais evitados
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Sinais Filtrados", len(skipped))
with col2:
    st.metric("Losses Evitados", skipped_losses,
              help="Sinais que ML rejeitou e que realmente eram loss")
with col3:
    st.metric("Wins Perdidos", skipped_wins,
              help="Sinais que ML rejeitou mas eram win (falsos negativos)")
with col4:
    filter_precision = skipped_losses / len(skipped) * 100 if len(skipped) > 0 else 0
    st.metric("Precisao do Filtro", f"{filter_precision:.1f}%",
              help="% dos sinais filtrados que eram realmente ruins")

# ================================================================
# GRAFICOS COMPARATIVOS
# ================================================================

st.markdown("---")
st.subheader("📈 Equity Curve Comparativa")

col1, col2 = st.columns(2)

with col1:
    # Equity sem filtro
    df_all_sorted = df_all.sort_values("timestamp")
    df_all_sorted["cumulative_pnl"] = df_all_sorted["pnl_percent"].cumsum()

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=list(range(len(df_all_sorted))),
        y=df_all_sorted["cumulative_pnl"],
        mode="lines",
        name="Sem ML",
        line=dict(color="#ff4444", width=2)
    ))

    # Equity com filtro ML
    if total_exec > 0:
        executed_sorted = executed.sort_values("timestamp")
        executed_sorted["cumulative_pnl"] = executed_sorted["original_pnl"].cumsum()
        fig1.add_trace(go.Scatter(
            x=list(range(len(executed_sorted))),
            y=executed_sorted["cumulative_pnl"],
            mode="lines",
            name="Com ML",
            line=dict(color="#00cc00", width=2)
        ))

    fig1.update_layout(
        title="Equity Curve: Sem ML vs Com ML",
        xaxis_title="Trade #",
        yaxis_title="PnL Acumulado %",
        template="plotly_dark",
        height=400
    )
    fig1.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    # Distribuicao de PnL
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(
        x=df_all["pnl_percent"], name="Sem ML",
        marker_color="rgba(255, 68, 68, 0.5)", nbinsx=20
    ))
    if total_exec > 0:
        fig2.add_trace(go.Histogram(
            x=executed["original_pnl"], name="Com ML",
            marker_color="rgba(0, 204, 0, 0.5)", nbinsx=20
        ))
    fig2.update_layout(
        title="Distribuicao de PnL por Trade",
        xaxis_title="PnL %",
        yaxis_title="Quantidade",
        template="plotly_dark",
        barmode="overlay",
        height=400
    )
    fig2.add_vline(x=0, line_dash="dash", line_color="white")
    st.plotly_chart(fig2, use_container_width=True)


# ================================================================
# ANALISE POR PROBABILIDADE ML
# ================================================================

st.markdown("---")
st.subheader("🎯 Analise por Faixa de Probabilidade ML")

if len(df_ml) > 0:
    # Criar faixas de probabilidade
    bins = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    labels = ['0-30%', '30-40%', '40-50%', '50-60%', '60-70%', '70-80%', '80-100%']
    df_ml['prob_range'] = pd.cut(df_ml['ml_probability'], bins=bins, labels=labels, include_lowest=True)

    prob_stats = []
    for label in labels:
        subset = df_ml[df_ml['prob_range'] == label]
        if len(subset) > 0:
            wins = len(subset[subset['original_pnl'] > 0])
            prob_stats.append({
                'Faixa': label,
                'Sinais': len(subset),
                'Win Rate': f"{wins / len(subset) * 100:.1f}%",
                'PnL Medio': f"{subset['original_pnl'].mean():.2f}%",
                'PnL Total': f"{subset['original_pnl'].sum():.2f}%",
            })

    if prob_stats:
        st.dataframe(pd.DataFrame(prob_stats), use_container_width=True, hide_index=True)


# ================================================================
# TABELA DETALHADA
# ================================================================

st.markdown("---")
st.subheader("📋 Detalhes dos Sinais")

if len(df_ml) > 0:
    display_cols = ['timestamp', 'symbol', 'signal', 'source', 'confidence',
                    'outcome', 'original_pnl', 'ml_decision', 'ml_probability']
    display_cols = [c for c in display_cols if c in df_ml.columns]

    df_display = df_ml[display_cols].copy()
    if 'ml_probability' in df_display.columns:
        df_display['ml_probability'] = df_display['ml_probability'].apply(lambda x: f"{x:.1%}")
    if 'original_pnl' in df_display.columns:
        df_display = df_display.rename(columns={'original_pnl': 'PnL %'})

    def color_decision(val):
        if val == 'EXECUTE':
            return 'color: #00cc00'
        elif val == 'SKIP':
            return 'color: #ff4444'
        return ''

    styled = df_display.style.applymap(color_decision, subset=['ml_decision'])
    st.dataframe(styled, use_container_width=True, hide_index=True, height=500)


# ================================================================
# FOOTER
# ================================================================

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Backtesting | Simulacao historica - resultados passados nao garantem resultados futuros</div>",
    unsafe_allow_html=True
)
