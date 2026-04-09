"""
Signal Analytics - Pagina de Analise de Performance de Sinais
==============================================================

Avalia TODOS os sinais emitidos contra dados reais de mercado.
Mostra metricas completas de performance por fonte, direcao, simbolo.
"""

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

st.set_page_config(page_title="Signal Analytics", page_icon="📊", layout="wide")

st.title("📊 Signal Analytics - Performance de Sinais")
st.markdown("Avaliacao de todos os sinais emitidos contra dados reais do mercado")
st.markdown("---")

# Botao de refresh
if st.button("🔄 Atualizar Dados", type="primary"):
    st.cache_data.clear()


@st.cache_data(ttl=120)
def load_evaluations():
    return evaluate_all_signals()


with st.spinner("Buscando dados de mercado e avaliando sinais..."):
    evaluations = load_evaluations()

if not evaluations:
    st.warning("Nenhum sinal encontrado em signals/. Execute o sistema primeiro.")
    st.stop()

_EXCLUDE_SYMBOLS = {"JCTUSDT", "4USDT"}
_PNL_OUTLIER_CAP = 15.0

evaluations = [
    e for e in evaluations if e.get("symbol") not in _EXCLUDE_SYMBOLS
]

for e in evaluations:
    pnl = e.get("pnl_percent", 0)
    if abs(pnl) > _PNL_OUTLIER_CAP:
        e["pnl_percent"] = _PNL_OUTLIER_CAP if pnl > 0 else -_PNL_OUTLIER_CAP

df = pd.DataFrame(evaluations)

# ================================================================
# METRICAS GERAIS
# ================================================================
summary = get_performance_summary(evaluations)

st.header("Resumo Geral")
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.metric("Total Sinais", summary.get("total_signals", 0))
with col2:
    st.metric("Finalizados", summary.get("closed", 0))
with col3:
    st.metric("Ativos", summary.get("active", 0))
with col4:
    wr = summary.get("win_rate", 0)
    st.metric("Win Rate", f"{wr:.1f}%")
with col5:
    total_pnl = summary.get("total_pnl", 0)
    st.metric("PnL Total", f"{total_pnl:.2f}%", delta=f"{total_pnl:.2f}%")
with col6:
    pf = summary.get("profit_factor", 0)
    st.metric("Profit Factor", f"{pf:.2f}")

st.markdown("---")

# ================================================================
# KPIs AVANCADOS
# ================================================================
col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    st.metric("Avg Win", f"{summary.get('avg_win', 0):.2f}%")
with col2:
    st.metric("Avg Loss", f"{summary.get('avg_loss', 0):.2f}%")
with col3:
    st.metric("Expectancy", f"{summary.get('expectancy', 0):.3f}%")
with col4:
    st.metric("Melhor Trade", f"{summary.get('best_trade', 0):.2f}%")
with col5:
    st.metric("Pior Trade", f"{summary.get('worst_trade', 0):.2f}%")
with col6:
    st.metric("Duracao Media", f"{summary.get('avg_duration_hours', 0):.1f}h")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("SL Hit", summary.get("sl_hits", 0))
with col2:
    st.metric("TP1 Hit", summary.get("tp1_hits", 0))
with col3:
    st.metric("TP2 Hit", summary.get("tp2_hits", 0))
with col4:
    st.metric("Expirados", summary.get("expired", 0))

st.markdown("---")

# ================================================================
# COMPARACAO: EXECUTADOS vs NAO EXECUTADOS
# ================================================================
exec_metrics = summary.get("executed_metrics", {})
noexec_metrics = summary.get("not_executed_metrics", {})
exec_count = summary.get("executed_count", 0)
noexec_count = summary.get("not_executed_count", 0)

if exec_count > 0 or noexec_count > 0:
    st.header("Executados vs Apenas Gerados")
    st.caption(f"Executados: {exec_count} sinais | Apenas gerados: {noexec_count} sinais")

    comp_col1, comp_col2 = st.columns(2)

    with comp_col1:
        st.subheader("EXECUTADOS (Real/Paper)")
        if exec_metrics and exec_metrics.get("closed", 0) > 0:
            e1, e2, e3, e4 = st.columns(4)
            with e1:
                st.metric("Fechados", exec_metrics.get("closed", 0))
            with e2:
                st.metric("Win Rate", f"{exec_metrics.get('win_rate', 0):.1f}%")
            with e3:
                st.metric("PnL Total", f"{exec_metrics.get('total_pnl', 0):.2f}%")
            with e4:
                st.metric("Profit Factor", f"{exec_metrics.get('profit_factor', 0):.2f}")
        else:
            st.info("Nenhum sinal executado finalizado ainda")

    with comp_col2:
        st.subheader("NAO EXECUTADOS (Apenas gerados)")
        if noexec_metrics and noexec_metrics.get("closed", 0) > 0:
            n1, n2, n3, n4 = st.columns(4)
            with n1:
                st.metric("Fechados", noexec_metrics.get("closed", 0))
            with n2:
                st.metric("Win Rate", f"{noexec_metrics.get('win_rate', 0):.1f}%")
            with n3:
                st.metric("PnL Total", f"{noexec_metrics.get('total_pnl', 0):.2f}%")
            with n4:
                st.metric("Profit Factor", f"{noexec_metrics.get('profit_factor', 0):.2f}")
        else:
            st.info("Nenhum sinal nao-executado finalizado ainda")

    # Motivos pelos quais os sinais NAO foram executados (para ajustar filtros)
    if noexec_count > 0 and "non_execution_reason" in df.columns:
        exec_col = df["executed"] if "executed" in df.columns else pd.Series([False] * len(df))
        not_exec = df[~exec_col.astype(bool)]
        reasons = not_exec["non_execution_reason"].fillna("(nao registado)").replace("", "(nao registado)")
        reason_counts = reasons.value_counts()
        if not reason_counts.empty:
            st.subheader("Por que nao foram executados?")
            st.caption("Motivo registado no momento em que o sinal foi gerado (validate_risk_and_position). Sinais antigos podem mostrar (nao registado).")
            reason_df = pd.DataFrame({
                "Motivo": reason_counts.index.tolist(),
                "Quantidade": reason_counts.values.tolist(),
            })
            st.dataframe(reason_df, use_container_width=True, hide_index=True)

    st.markdown("---")

# ================================================================
# TABS DE ANALISE
# ================================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📋 Todos os Sinais",
    "📈 Performance por Fonte",
    "🎯 Performance por Simbolo",
    "📊 Graficos",
    "🔬 Long vs Short",
    "📉 MFE/MAE",
    "🗳️ Voter Accuracy",
    "🆚 DeepSeek vs Local"
])

# ================================================================
# TAB 1: TABELA DE SINAIS
# ================================================================
with tab1:
    st.subheader("Todos os Sinais Emitidos")

    # Filtros
    col_f1, col_f2, col_f3, col_f4, col_f5 = st.columns(5)
    with col_f1:
        filter_source = st.multiselect("Fonte", df["source"].unique().tolist(), default=df["source"].unique().tolist())
    with col_f2:
        filter_signal = st.multiselect("Direcao", df["signal"].unique().tolist(), default=df["signal"].unique().tolist())
    with col_f3:
        filter_outcome = st.multiselect("Resultado", df["outcome"].unique().tolist(), default=df["outcome"].unique().tolist())
    with col_f4:
        filter_symbol = st.multiselect("Simbolo", sorted(df["symbol"].unique().tolist()), default=sorted(df["symbol"].unique().tolist()))
    with col_f5:
        exec_options = ["Todos", "Executados", "Nao Executados"]
        filter_executed = st.selectbox("Execucao", exec_options, index=0)

    # Aplicar filtros
    mask = (
        df["source"].isin(filter_source) &
        df["signal"].isin(filter_signal) &
        df["outcome"].isin(filter_outcome) &
        df["symbol"].isin(filter_symbol)
    )
    if filter_executed == "Executados":
        mask = mask & (df["executed"].astype(bool))
    elif filter_executed == "Nao Executados":
        mask = mask & (~df["executed"].astype(bool) | df["executed"].isna())
    df_filtered = df[mask].copy()

    # Preparar coluna de status de execucao para exibicao
    df_filtered["exec_status"] = df_filtered["executed"].apply(
        lambda x: "SIM" if bool(x) else "NAO"
    )
    df_filtered["ml_prob_display"] = df_filtered["ml_probability"].apply(
        lambda x: f"{x:.0%}" if isinstance(x, (int, float)) and x == x else "-"
    )

    # Formatacao
    display_df = df_filtered[[
        "timestamp", "symbol", "source", "signal", "confidence",
        "entry_price", "stop_loss", "take_profit_1", "take_profit_2",
        "outcome", "exit_price", "pnl_percent", "duration_hours",
        "exec_status", "ml_prob_display"
    ]].copy()

    display_df.columns = [
        "Data/Hora", "Simbolo", "Fonte", "Direcao", "Confianca",
        "Entry", "SL", "TP1", "TP2",
        "Resultado", "Exit", "PnL %", "Duracao (h)",
        "Executado", "ML Prob"
    ]

    # Colorir PnL
    def color_pnl(val):
        if isinstance(val, (int, float)):
            if val > 0:
                return "color: #00cc00"
            elif val < 0:
                return "color: #ff4444"
        return ""

    # Colorir outcome
    def color_outcome(val):
        colors = {
            "TP2_HIT": "background-color: #004400; color: #00ff00",
            "TP1_HIT": "background-color: #003300; color: #00cc00",
            "SL_HIT": "background-color: #440000; color: #ff4444",
            "ACTIVE": "background-color: #333300; color: #ffff00",
            "EXPIRED": "background-color: #333333; color: #aaaaaa",
        }
        return colors.get(val, "")

    styled = display_df.style.map(color_pnl, subset=["PnL %"])
    styled = styled.map(color_outcome, subset=["Resultado"])
    st.dataframe(styled, use_container_width=True, hide_index=True, height=500)

    st.caption(f"Total: {len(df_filtered)} sinais")

# ================================================================
# TAB 2: PERFORMANCE POR FONTE
# ================================================================
with tab2:
    st.subheader("Performance por Fonte de Sinal")

    closed_df = df[df["outcome"].isin(["SL_HIT", "TP1_HIT", "TP2_HIT", "EXPIRED"])].copy()

    if not closed_df.empty:
        for source in closed_df["source"].unique():
            source_data = closed_df[closed_df["source"] == source]
            source_evals = [e for e in evaluations if e.get("source") == source and e["outcome"] in ("SL_HIT", "TP1_HIT", "TP2_HIT", "EXPIRED")]
            source_summary = get_performance_summary(source_evals + [e for e in evaluations if e.get("source") == source and e["outcome"] == "ACTIVE"])

            st.markdown(f"### {source}")
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            with col1:
                st.metric("Sinais", source_summary.get("closed", 0))
            with col2:
                st.metric("Win Rate", f"{source_summary.get('win_rate', 0):.1f}%")
            with col3:
                st.metric("PnL Total", f"{source_summary.get('total_pnl', 0):.2f}%")
            with col4:
                st.metric("Profit Factor", f"{source_summary.get('profit_factor', 0):.2f}")
            with col5:
                st.metric("Avg Win", f"{source_summary.get('avg_win', 0):.2f}%")
            with col6:
                st.metric("Avg Loss", f"{source_summary.get('avg_loss', 0):.2f}%")

            # Breakdown de outcomes
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("SL Hit", source_summary.get("sl_hits", 0))
            with col2:
                st.metric("TP1 Hit", source_summary.get("tp1_hits", 0))
            with col3:
                st.metric("TP2 Hit", source_summary.get("tp2_hits", 0))
            with col4:
                st.metric("Expirados", source_summary.get("expired", 0))

            st.markdown("---")
    else:
        st.info("Nenhum sinal finalizado ainda.")

# ================================================================
# TAB 3: PERFORMANCE POR SIMBOLO
# ================================================================
with tab3:
    st.subheader("Performance por Simbolo")

    if not closed_df.empty:
        symbol_stats = []
        for sym in sorted(closed_df["symbol"].unique()):
            sym_evals = [e for e in evaluations if e.get("symbol") == sym]
            sym_summary = get_performance_summary(sym_evals)
            symbol_stats.append({
                "Simbolo": sym,
                "Sinais": sym_summary.get("closed", 0),
                "Win Rate": f"{sym_summary.get('win_rate', 0):.1f}%",
                "PnL Total %": round(sym_summary.get("total_pnl", 0), 2),
                "PnL Medio %": round(sym_summary.get("avg_pnl", 0), 2),
                "Profit Factor": round(sym_summary.get("profit_factor", 0), 2),
                "Melhor": f"{sym_summary.get('best_trade', 0):.2f}%",
                "Pior": f"{sym_summary.get('worst_trade', 0):.2f}%",
                "SL": sym_summary.get("sl_hits", 0),
                "TP1": sym_summary.get("tp1_hits", 0),
                "TP2": sym_summary.get("tp2_hits", 0),
            })

        df_symbols = pd.DataFrame(symbol_stats)
        st.dataframe(df_symbols, use_container_width=True, hide_index=True)

        # Grafico de PnL por simbolo
        fig = px.bar(
            df_symbols, x="Simbolo", y="PnL Total %",
            color="PnL Total %",
            color_continuous_scale=["#ff4444", "#ffff00", "#00cc00"],
            title="PnL Total por Simbolo"
        )
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Nenhum sinal finalizado ainda.")

# ================================================================
# TAB 4: GRAFICOS
# ================================================================
with tab4:
    st.subheader("Visualizacoes")

    chart_exec_only = st.toggle(
        "Mostrar apenas sinais EXECUTADOS (resultado real)",
        value=True,
        help="Ativado = mostra apenas trades realmente executados na exchange. "
             "Desativado = inclui sinais apenas gerados (nao reflete a conta real)."
    )

    if chart_exec_only:
        chart_df = closed_df[closed_df["executed"].astype(bool)].copy()
        if chart_df.empty:
            st.warning("Nenhum sinal executado finalizado. Mostrando todos.")
            chart_df = closed_df.copy()
    else:
        chart_df = closed_df.copy()

    if not chart_df.empty:
        col1, col2 = st.columns(2)

        with col1:
            # PnL acumulado ao longo do tempo
            sorted_closed = chart_df.sort_values("timestamp")
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
                title="PnL Acumulado ao Longo do Tempo",
                xaxis_title="Data",
                yaxis_title="PnL %",
                template="plotly_dark",
                height=400
            )
            fig_cum.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_cum, use_container_width=True)

        with col2:
            # Distribuicao de PnL
            fig_dist = px.histogram(
                chart_df, x="pnl_percent", nbins=30,
                title="Distribuicao de PnL por Trade",
                color_discrete_sequence=["#00ccff"]
            )
            fig_dist.update_layout(
                template="plotly_dark",
                xaxis_title="PnL %",
                yaxis_title="Quantidade",
                height=400
            )
            fig_dist.add_vline(x=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_dist, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            # Outcomes pie chart
            outcome_counts = chart_df["outcome"].value_counts()
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
            fig_pie.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # Confianca vs PnL scatter
            fig_conf = px.scatter(
                chart_df, x="confidence", y="pnl_percent",
                color="outcome", size=chart_df["pnl_percent"].abs() + 0.1,
                title="Confianca vs PnL",
                color_discrete_map=colors_map,
                hover_data=["symbol", "source", "signal"]
            )
            fig_conf.update_layout(template="plotly_dark", height=400)
            fig_conf.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_conf, use_container_width=True)

        # Heatmap: Win Rate por Simbolo x Fonte
        st.subheader("Heatmap: Win Rate por Simbolo x Fonte")
        pivot_data = []
        for sym in chart_df["symbol"].unique():
            for src in chart_df["source"].unique():
                subset = chart_df[(chart_df["symbol"] == sym) & (chart_df["source"] == src)]
                if len(subset) > 0:
                    wins = len(subset[subset["pnl_percent"] > 0])
                    wr = wins / len(subset) * 100
                    pivot_data.append({"Simbolo": sym, "Fonte": src, "Win Rate": wr, "N": len(subset)})

        if pivot_data:
            df_pivot = pd.DataFrame(pivot_data)
            fig_heat = px.density_heatmap(
                df_pivot, x="Fonte", y="Simbolo", z="Win Rate",
                color_continuous_scale="RdYlGn",
                title="Win Rate por Simbolo x Fonte",
                text_auto=".0f"
            )
            fig_heat.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Nenhum sinal finalizado ainda.")

# ================================================================
# TAB 5: LONG vs SHORT
# ================================================================
with tab5:
    st.subheader("Analise: Long vs Short")

    if not closed_df.empty:
        for direction in ["BUY", "SELL"]:
            dir_label = "LONG" if direction == "BUY" else "SHORT"
            dir_evals = [e for e in evaluations if e.get("signal") == direction]
            dir_summary = get_performance_summary(dir_evals)

            st.markdown(f"### {dir_label}")
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            with col1:
                st.metric("Sinais", dir_summary.get("closed", 0))
            with col2:
                st.metric("Win Rate", f"{dir_summary.get('win_rate', 0):.1f}%")
            with col3:
                st.metric("PnL Total", f"{dir_summary.get('total_pnl', 0):.2f}%")
            with col4:
                st.metric("Profit Factor", f"{dir_summary.get('profit_factor', 0):.2f}")
            with col5:
                st.metric("Avg Win", f"{dir_summary.get('avg_win', 0):.2f}%")
            with col6:
                st.metric("Avg Loss", f"{dir_summary.get('avg_loss', 0):.2f}%")
            st.markdown("---")

        # Comparativo visual
        long_evals = [e for e in evaluations if e.get("signal") == "BUY" and e["outcome"] in ("SL_HIT", "TP1_HIT", "TP2_HIT", "EXPIRED")]
        short_evals = [e for e in evaluations if e.get("signal") == "SELL" and e["outcome"] in ("SL_HIT", "TP1_HIT", "TP2_HIT", "EXPIRED")]

        comp_data = []
        for label, evals in [("LONG", long_evals), ("SHORT", short_evals)]:
            for e in evals:
                comp_data.append({"Direcao": label, "PnL %": e["pnl_percent"], "Simbolo": e["symbol"]})

        if comp_data:
            df_comp = pd.DataFrame(comp_data)
            fig_box = px.box(
                df_comp, x="Direcao", y="PnL %", color="Direcao",
                color_discrete_map={"LONG": "#00cc00", "SHORT": "#ff4444"},
                title="Distribuicao PnL: Long vs Short",
                points="all"
            )
            fig_box.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.info("Nenhum sinal finalizado ainda.")

# ================================================================
# TAB 6: MFE/MAE ANALYSIS
# ================================================================
with tab6:
    st.subheader("Maximum Favorable/Adverse Excursion")
    st.markdown("""
    - **MFE (Max Favorable Excursion)**: Maximo que o preco foi a seu favor durante o trade
    - **MAE (Max Adverse Excursion)**: Maximo que o preco foi contra voce durante o trade
    - Util para otimizar stops e alvos
    """)

    if not closed_df.empty:
        col1, col2 = st.columns(2)

        with col1:
            st.metric("MFE Medio", f"{summary.get('avg_mfe', 0):.2f}%")
        with col2:
            st.metric("MAE Medio", f"{summary.get('avg_mae', 0):.2f}%")

        # Scatter MFE vs MAE colorido por outcome
        colors_map_full = {
            "TP2_HIT": "#00ff00",
            "TP1_HIT": "#00cc00",
            "SL_HIT": "#ff4444",
            "EXPIRED": "#888888"
        }

        fig_mfe = px.scatter(
            closed_df, x="max_adverse", y="max_favorable",
            color="outcome", hover_data=["symbol", "pnl_percent", "signal", "source"],
            color_discrete_map=colors_map_full,
            title="MFE vs MAE por Trade",
            labels={"max_adverse": "MAE % (contra voce)", "max_favorable": "MFE % (a seu favor)"}
        )
        fig_mfe.update_layout(template="plotly_dark", height=500)
        # Linha diagonal (MFE = MAE)
        max_val = max(closed_df["max_favorable"].max(), closed_df["max_adverse"].max(), 1)
        fig_mfe.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val],
            mode="lines", line=dict(dash="dash", color="gray"),
            name="MFE = MAE", showlegend=True
        ))
        st.plotly_chart(fig_mfe, use_container_width=True)

        # Insights
        st.subheader("Insights para Otimizacao")

        # Analisar se SL esta muito apertado (MAE dos winners)
        winners = closed_df[closed_df["pnl_percent"] > 0]
        losers = closed_df[closed_df["pnl_percent"] <= 0]

        if not winners.empty and not losers.empty:
            avg_winner_mae = winners["max_adverse"].mean()
            avg_loser_mae = losers["max_adverse"].mean()
            avg_winner_mfe = winners["max_favorable"].mean()

            st.markdown(f"""
            **Analise dos Stops:**
            - MAE medio dos **winners**: {avg_winner_mae:.2f}% (quanto o preco foi contra antes de ir a favor)
            - MAE medio dos **losers**: {avg_loser_mae:.2f}%
            - MFE medio dos **winners**: {avg_winner_mfe:.2f}% (potencial maximo capturado)

            **Recomendacoes:**
            """)

            if avg_winner_mae > 2.0:
                st.warning(f"Stop Loss pode estar APERTADO demais. Winners tiveram drawdown medio de {avg_winner_mae:.1f}% antes de lucrar.")
            else:
                st.success(f"Stop Loss parece adequado. Winners com drawdown medio de {avg_winner_mae:.1f}%.")

            capture_ratio = summary.get("avg_pnl", 0) / avg_winner_mfe * 100 if avg_winner_mfe > 0 else 0
            if capture_ratio < 30:
                st.warning(f"Capturando apenas {capture_ratio:.0f}% do MFE medio. Considere ajustar Take Profits.")
            elif capture_ratio > 60:
                st.success(f"Boa captura: {capture_ratio:.0f}% do MFE medio.")
    else:
        st.info("Nenhum sinal finalizado ainda.")

# ================================================================
# TAB 7: VOTER ACCURACY
# ================================================================
with tab7:
    st.subheader("Accuracy por Votante")
    st.markdown(
        "Mostra a % de acerto de cada componente do sistema de confluencia. "
        "Um voto **correto** = votou a favor e trade ganhou, OU votou contra e trade perdeu."
    )

    voter_data = compute_voter_accuracy()

    if not voter_data:
        st.info(
            "Sem dados de voter tracking ainda. "
            "Apos o deploy com voter_votes habilitado, os dados vao acumular aqui."
        )
    else:
        # Separar votantes tecnicos, ML/LSTM e DeepSeek
        tech_voters = ["rsi", "macd", "trend", "adx", "bb", "orderbook", "mtf", "cvd", "regime"]
        model_voters = ["ml", "lstm"]

        # KPIs: DeepSeek win rate
        ds = voter_data.get("deepseek", {})
        ds_total = ds.get("total", 0)
        ds_wr = ds.get("win_rate")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "DeepSeek Win Rate",
                f"{ds_wr * 100:.1f}%" if ds_wr is not None else "N/A",
                help="% de sinais gerados pelo DeepSeek que foram vencedores"
            )
        with col2:
            st.metric("Total Sinais Avaliados", ds_total)
        with col3:
            ds_wins = ds.get("wins", 0)
            st.metric("Wins / Losses", f"{ds_wins} / {ds_total - ds_wins}")

        st.markdown("---")

        # Tabela de accuracy por votante tecnico
        st.markdown("### Votantes Tecnicos")
        rows = []
        for name in tech_voters:
            v = voter_data.get(name, {})
            total = v.get("total", 0)
            acc = v.get("accuracy")
            rows.append({
                "Votante": name.upper(),
                "Total Votos": total,
                "Corretos": v.get("correct", 0),
                "Accuracy %": f"{acc * 100:.1f}%" if acc is not None else "N/A",
                "Votou FOR": v.get("voted_for", 0),
                "Votou AGAINST": v.get("voted_against", 0),
                "Accuracy": acc if acc is not None else 0,
            })

        if rows:
            df_voters = pd.DataFrame(rows)

            # Ordenar por accuracy
            df_voters = df_voters.sort_values("Accuracy", ascending=False)

            # Mostrar tabela
            display_cols = ["Votante", "Total Votos", "Corretos", "Accuracy %", "Votou FOR", "Votou AGAINST"]
            st.dataframe(
                df_voters[display_cols],
                use_container_width=True,
                hide_index=True,
            )

            # Grafico de barras
            df_chart = df_voters[df_voters["Total Votos"] > 0].copy()
            if not df_chart.empty:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=df_chart["Votante"],
                    y=[a * 100 for a in df_chart["Accuracy"]],
                    marker_color=[
                        "#00cc00" if a >= 0.6 else "#ffaa00" if a >= 0.5 else "#ff4444"
                        for a in df_chart["Accuracy"]
                    ],
                    text=[f"{a * 100:.1f}%" for a in df_chart["Accuracy"]],
                    textposition="outside",
                ))
                fig.add_hline(y=50, line_dash="dash", line_color="gray",
                              annotation_text="50% (moeda)")
                fig.update_layout(
                    title="Accuracy por Votante Tecnico",
                    yaxis_title="Accuracy %",
                    template="plotly_dark",
                    height=400,
                    yaxis=dict(range=[0, 100]),
                )
                st.plotly_chart(fig, use_container_width=True)

        # ML e LSTM
        st.markdown("### Modelos ML / LSTM")
        ml_rows = []
        for name in model_voters:
            v = voter_data.get(name, {})
            total = v.get("total", 0)
            acc = v.get("accuracy")
            ml_rows.append({
                "Modelo": name.upper(),
                "Total Votos": total,
                "Corretos": v.get("correct", 0),
                "Accuracy %": f"{acc * 100:.1f}%" if acc is not None else "N/A",
                "Votou FOR": v.get("voted_for", 0),
                "Votou AGAINST": v.get("voted_against", 0),
            })

        df_ml = pd.DataFrame(ml_rows)
        st.dataframe(df_ml, use_container_width=True, hide_index=True)

        # Insights automaticos
        st.markdown("### Insights")
        all_voters = tech_voters + model_voters
        good = []
        bad = []
        for name in all_voters:
            v = voter_data.get(name, {})
            acc = v.get("accuracy")
            total = v.get("total", 0)
            if acc is not None and total >= 10:
                if acc >= 0.60:
                    good.append((name.upper(), acc, total))
                elif acc < 0.45:
                    bad.append((name.upper(), acc, total))

        if good:
            good.sort(key=lambda x: x[1], reverse=True)
            lines = [f"- **{n}**: {a * 100:.1f}% accuracy ({t} votos)" for n, a, t in good]
            st.success("**Votantes com boa accuracy (>= 60%):**\n" + "\n".join(lines))

        if bad:
            bad.sort(key=lambda x: x[1])
            lines = [f"- **{n}**: {a * 100:.1f}% accuracy ({t} votos)" for n, a, t in bad]
            st.error(
                "**Votantes com accuracy ruim (< 45%) — considere desabilitar ou recalibrar:**\n"
                + "\n".join(lines)
            )

        if not good and not bad:
            st.info("Poucos dados para gerar insights (minimo 10 votos por votante).")

# ================================================================
# TAB 8: DEEPSEEK vs LOCAL GENERATOR
# ================================================================
with tab8:
    st.subheader("DeepSeek vs Local Generator — Shadow Mode")
    st.markdown(
        "Compara a performance do DeepSeek (LLM) com o gerador local (100% tecnico). "
        "O gerador local roda em **shadow mode** — gera sinais mas **nunca executa**."
    )

    # Filtrar sinais por fonte
    closed_outcomes = ["SL_HIT", "TP1_HIT", "TP2_HIT", "EXPIRED"]

    df_agno = df[(df["source"] == "AGNO") & (df["outcome"].isin(closed_outcomes))].copy()
    df_local = df[(df["source"] == "LOCAL_GEN") & (df["outcome"].isin(closed_outcomes))].copy()

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### AGNO (DeepSeek)")
        if not df_agno.empty:
            agno_wins = df_agno["outcome"].isin(["TP1_HIT", "TP2_HIT"]).sum()
            agno_total = len(df_agno)
            agno_wr = agno_wins / agno_total * 100 if agno_total > 0 else 0
            agno_pnl = df_agno["pnl_percent"].sum() if "pnl_percent" in df_agno.columns else 0

            st.metric("Win Rate", f"{agno_wr:.1f}%")
            st.metric("Total Sinais", agno_total)
            st.metric("Wins / Losses", f"{agno_wins} / {agno_total - agno_wins}")
            st.metric("PnL Total", f"{agno_pnl:.2f}%")
        else:
            st.info("Nenhum sinal AGNO finalizado.")

    with col_b:
        st.markdown("### LOCAL_GEN (Tecnico)")
        if not df_local.empty:
            local_wins = df_local["outcome"].isin(["TP1_HIT", "TP2_HIT"]).sum()
            local_total = len(df_local)
            local_wr = local_wins / local_total * 100 if local_total > 0 else 0
            local_pnl = df_local["pnl_percent"].sum() if "pnl_percent" in df_local.columns else 0

            st.metric("Win Rate", f"{local_wr:.1f}%")
            st.metric("Total Sinais", local_total)
            st.metric("Wins / Losses", f"{local_wins} / {local_total - local_wins}")
            st.metric("PnL Total", f"{local_pnl:.2f}%")
        else:
            st.info("Nenhum sinal LOCAL_GEN finalizado ainda. Aguarde sinais serem avaliados.")

    # Tabela de concordancia/divergencia
    if not df_local.empty and not df_agno.empty:
        st.markdown("---")
        st.subheader("Concordancia de Sinais")
        st.markdown(
            "Quando DeepSeek e Local Generator analisam o mesmo ativo no mesmo momento, "
            "eles concordam ou divergem na direcao?"
        )

        # Tentar matching por symbol + timestamp (aproximado)
        matches = []
        for _, local_row in df_local.iterrows():
            sym = local_row.get("symbol", "")
            ts = str(local_row.get("timestamp", ""))[:16]  # Match até minutos
            sig_local = local_row.get("signal", "")
            out_local = local_row.get("outcome", "")

            # Buscar AGNO correspondente
            mask = (df_agno["symbol"] == sym) & (df_agno["timestamp"].astype(str).str[:16] == ts)
            agno_match = df_agno[mask]
            if not agno_match.empty:
                agno_row = agno_match.iloc[0]
                sig_agno = agno_row.get("signal", "")
                out_agno = agno_row.get("outcome", "")
                concordam = sig_local == sig_agno
                matches.append({
                    "Symbol": sym,
                    "Timestamp": ts,
                    "AGNO": sig_agno,
                    "LOCAL": sig_local,
                    "Concordam": "Sim" if concordam else "Nao",
                    "Outcome AGNO": out_agno,
                    "Outcome LOCAL": out_local,
                })

        if matches:
            df_match = pd.DataFrame(matches)
            concordance_pct = (df_match["Concordam"] == "Sim").mean() * 100
            st.metric("Taxa de Concordancia", f"{concordance_pct:.0f}%")
            st.dataframe(df_match, use_container_width=True, hide_index=True)
        else:
            st.info("Ainda sem dados suficientes para calcular concordancia.")

    elif df_local.empty:
        st.markdown("---")
        st.info(
            "O gerador local esta ativo em **shadow mode**. "
            "Os primeiros sinais aparecerão aqui apos serem avaliados contra o mercado."
        )

# ================================================================
# FOOTER
# ================================================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Signal Analytics | Dados atualizados a cada 2 min</div>",
    unsafe_allow_html=True
)
