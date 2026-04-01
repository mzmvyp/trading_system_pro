"""
Overview - Resumo Geral do Sistema de Trading
==============================================

Pagina principal do dashboard com visao geral de:
- Status do sistema (ML, sinais, posicoes)
- Performance recente
- Sinais ativos
- Health check dos componentes
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Adicionar raiz do projeto ao path
root = Path(__file__).resolve().parent.parent.parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.trading.signal_tracker import (  # noqa: E402
    compute_voter_accuracy,
    evaluate_all_signals,
    get_performance_summary,
)

st.set_page_config(page_title="Overview", page_icon="📋", layout="wide")

st.title("📋 Overview - Resumo do Sistema")
st.markdown("---")


# ================================================================
# FUNCOES DE CARGA DE DADOS
# ================================================================

def load_model_info():
    info_path = "ml_models/model_info_simple.json"
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            return json.load(f)
    return {}


def load_performance_history():
    perf_path = "ml_models/model_performance.json"
    if os.path.exists(perf_path):
        with open(perf_path, 'r') as f:
            return json.load(f)
    return []


def load_prediction_log():
    log_path = "ml_models/prediction_log.json"
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            return json.load(f)
    return []


def load_buffer():
    buffer_path = "ml_models/online_learning_buffer.json"
    if os.path.exists(buffer_path):
        with open(buffer_path, 'r') as f:
            return json.load(f)
    return []


def count_signal_files():
    count = 0
    if os.path.exists("signals"):
        for f in os.listdir("signals"):
            if f.endswith('.json') and '_last_analysis' not in f and 'cache' not in f:
                count += 1
    return count


# ================================================================
# HEALTH CHECK DOS COMPONENTES
# ================================================================

st.header("🔧 Status dos Componentes")

col1, col2, col3, col4 = st.columns(4)

model_info = load_model_info()
has_model = os.path.exists("ml_models/signal_validators.pkl")

with col1:
    if has_model:
        st.success(f"**ML Model:** {model_info.get('best_model', 'Ativo')}")
    else:
        st.error("**ML Model:** Nao treinado")

with col2:
    signal_count = count_signal_files()
    if signal_count > 0:
        st.success(f"**Sinais Salvos:** {signal_count}")
    else:
        st.warning("**Sinais Salvos:** 0")

with col3:
    buffer = load_buffer()
    predictions = load_prediction_log()
    if predictions:
        st.success(f"**Predicoes ML:** {len(predictions)}")
    else:
        st.warning("**Predicoes ML:** 0")

with col4:
    buffer_size = len(buffer)
    if buffer_size > 0:
        st.info(f"**Buffer OL:** {buffer_size}/50")
    else:
        st.info("**Buffer OL:** Vazio")

st.markdown("---")


# ================================================================
# PERFORMANCE DE SINAIS
# ================================================================

st.header("📊 Performance de Sinais")


@st.cache_data(ttl=120)
def load_evaluations():
    try:
        return evaluate_all_signals()
    except Exception:
        return []


evaluations = load_evaluations()

if evaluations:
    summary = get_performance_summary(evaluations)
    df = pd.DataFrame(evaluations)

    # KPIs principais
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric("Total Sinais", summary.get("total_signals", 0))
    with col2:
        st.metric("Finalizados", summary.get("closed", 0))
    with col3:
        st.metric("Ativos", summary.get("active", 0))
    with col4:
        wr = summary.get("win_rate", 0)
        st.metric("Win Rate", f"{wr:.1f}%",
                   delta=f"{wr - 50:.1f}pp vs 50%" if wr > 0 else None)
    with col5:
        total_pnl = summary.get("total_pnl", 0)
        st.metric("PnL Total", f"{total_pnl:.2f}%",
                   delta=f"{total_pnl:.2f}%")
    with col6:
        pf = summary.get("profit_factor", 0)
        st.metric("Profit Factor", f"{pf:.2f}")

    st.markdown("---")

    # Graficos lado a lado
    col_left, col_right = st.columns(2)

    with col_left:
        # PnL acumulado
        closed_df = df[df["outcome"].isin(["SL_HIT", "TP1_HIT", "TP2_HIT", "EXPIRED"])].copy()
        if not closed_df.empty:
            sorted_closed = closed_df.sort_values("timestamp")
            sorted_closed["cumulative_pnl"] = sorted_closed["pnl_percent"].cumsum()

            fig_cum = go.Figure()
            fig_cum.add_trace(go.Scatter(
                x=sorted_closed["timestamp"],
                y=sorted_closed["cumulative_pnl"],
                mode="lines+markers",
                name="PnL Acumulado",
                line=dict(color="#00ccff", width=2),
                fill="tozeroy",
                fillcolor="rgba(0, 204, 255, 0.1)"
            ))
            fig_cum.update_layout(
                title="PnL Acumulado",
                template="plotly_dark",
                height=350,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            fig_cum.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_cum, use_container_width=True)
        else:
            st.info("Nenhum sinal finalizado para exibir PnL.")

    with col_right:
        # Distribuicao de outcomes
        closed_df = df[df["outcome"].isin(["SL_HIT", "TP1_HIT", "TP2_HIT", "EXPIRED"])].copy()
        if not closed_df.empty:
            outcome_counts = closed_df["outcome"].value_counts()
            colors_map = {
                "TP2_HIT": "#00ff00",
                "TP1_HIT": "#00cc00",
                "SL_HIT": "#ff4444",
                "EXPIRED": "#888888"
            }
            fig_pie = px.pie(
                names=outcome_counts.index,
                values=outcome_counts.values,
                title="Distribuicao de Resultados",
                color=outcome_counts.index,
                color_discrete_map=colors_map
            )
            fig_pie.update_layout(template="plotly_dark", height=350,
                                  margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Nenhum resultado para exibir.")

    # Sinais ativos
    active_signals = [e for e in evaluations if e.get("outcome") == "ACTIVE"]
    if active_signals:
        st.markdown("---")
        st.subheader(f"🟢 Sinais Ativos ({len(active_signals)})")
        df_active = pd.DataFrame(active_signals)
        display_cols = ["timestamp", "symbol", "source", "signal", "confidence",
                        "entry_price", "stop_loss", "take_profit_1", "pnl_percent"]
        display_cols = [c for c in display_cols if c in df_active.columns]
        st.dataframe(df_active[display_cols], use_container_width=True, hide_index=True)

    # Ultimos sinais finalizados
    if not closed_df.empty:
        st.markdown("---")
        st.subheader("📋 Ultimos Sinais Finalizados")
        recent = closed_df.sort_values("timestamp", ascending=False).head(10)
        display_cols = ["timestamp", "symbol", "signal", "source", "confidence",
                        "outcome", "pnl_percent", "duration_hours"]
        display_cols = [c for c in display_cols if c in recent.columns]
        st.dataframe(recent[display_cols], use_container_width=True, hide_index=True)

else:
    st.info("Nenhum sinal encontrado. Execute o sistema para gerar sinais.")


# ================================================================
# STATUS ML
# ================================================================

st.markdown("---")
st.header("🤖 Status ML")

if model_info:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Modelo", model_info.get("best_model", "N/A"))
    with col2:
        st.metric("Accuracy", f"{model_info.get('best_accuracy', 0):.1%}")
    with col3:
        st.metric("F1 Score", f"{model_info.get('best_f1', 0):.3f}")
    with col4:
        st.metric("Retreinos", model_info.get("retrain_count", 0))

    # Performance history
    perf_history = load_performance_history()
    if perf_history and len(perf_history) >= 2:
        df_perf = pd.DataFrame(perf_history)
        df_perf['timestamp'] = pd.to_datetime(df_perf['timestamp'])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_perf['timestamp'], y=df_perf['accuracy'],
            mode='lines+markers', name='Accuracy',
            line=dict(color='#e94560', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=df_perf['timestamp'], y=df_perf['f1_score'],
            mode='lines+markers', name='F1',
            line=dict(color='#533483', width=2)
        ))
        fig.update_layout(
            title="Evolucao do Modelo",
            template="plotly_dark",
            height=300,
            yaxis=dict(range=[0, 1]),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Modelo ML nao treinado. Acesse a aba 'ml dashboard' para treinar.")

# ================================================================
# VOTER ACCURACY (resumo compacto)
# ================================================================
st.markdown("---")
st.header("🗳️ Voter Accuracy")

voter_data = compute_voter_accuracy()
if voter_data:
    all_names = ["rsi", "macd", "trend", "adx", "bb", "orderbook", "mtf", "cvd", "ml", "lstm", "regime"]
    rows = []
    for name in all_names:
        v = voter_data.get(name, {})
        acc = v.get("accuracy")
        total = v.get("total", 0)
        if total > 0 and acc is not None:
            rows.append({"Votante": name.upper(), "Accuracy": acc, "Votos": total})

    if rows:
        df_v = pd.DataFrame(rows).sort_values("Accuracy", ascending=False)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_v["Votante"], y=[a * 100 for a in df_v["Accuracy"]],
            marker_color=[
                "#00cc00" if a >= 0.6 else "#ffaa00" if a >= 0.5 else "#ff4444"
                for a in df_v["Accuracy"]
            ],
            text=[f"{a * 100:.0f}%" for a in df_v["Accuracy"]],
            textposition="outside",
        ))
        fig.add_hline(y=50, line_dash="dash", line_color="gray")
        fig.update_layout(
            template="plotly_dark", height=300,
            yaxis=dict(range=[0, 100], title="Accuracy %"),
            margin=dict(l=20, r=20, t=10, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        # DeepSeek win rate
        ds = voter_data.get("deepseek", {})
        ds_wr = ds.get("win_rate")
        if ds_wr is not None:
            st.caption(f"DeepSeek Win Rate: **{ds_wr * 100:.1f}%** ({ds.get('total', 0)} sinais)")
    else:
        st.info("Aguardando dados de voter tracking...")
else:
    st.info("Voter tracking ativado. Dados aparecerao apos os proximos sinais.")

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Trading System Pro - Overview | Dados atualizados a cada 2 min</div>",
    unsafe_allow_html=True
)
