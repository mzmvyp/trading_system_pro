"""
Optimizer Status — Dashboard do Rolling Backtest Optimizer
==========================================================

Mostra:
- Status atual do optimizer (rodando, último ciclo, próximo)
- Parâmetros otimizados por símbolo (atuais vs defaults)
- Histórico de otimizações (log)
- Walk-forward: score IS vs OOS por janela
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

root = Path(__file__).resolve().parent.parent.parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.backtesting.backtest_engine import BacktestParams  # noqa: E402

st.set_page_config(page_title="Optimizer Status", page_icon="⚙️", layout="wide")
st.title("⚙️ Optimizer Status — Rolling Backtest")
st.markdown("Otimização contínua walk-forward com Monte Carlo validation")
st.markdown("---")

if st.button("🔄 Atualizar"):
    st.cache_data.clear()


@st.cache_data(ttl=60)
def load_data():
    from src.optimizer.rolling_backtest_optimizer import (
        get_current_dynamic_params,
        get_optimizer_log,
        get_optimizer_status,
    )
    return get_optimizer_status(), get_current_dynamic_params(), get_optimizer_log()


status, params, log = load_data()

# ================================================================
# STATUS GERAL
# ================================================================
st.header("Status do Optimizer")

col1, col2, col3, col4 = st.columns(4)
with col1:
    running = status.get("running", False)
    st.metric("Status", "Rodando" if running else "Parado")
with col2:
    st.metric("Ciclos Completos", status.get("cycle", 0))
with col3:
    last_run = status.get("last_run", "N/A")
    if last_run != "N/A":
        try:
            dt = datetime.fromisoformat(last_run)
            hours_ago = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
            last_run = f"{hours_ago:.1f}h atrás"
        except Exception:
            pass
    st.metric("Último Ciclo", last_run)
with col4:
    applied = status.get("params_applied", 0)
    total = status.get("symbols_optimized", 0)
    st.metric("Params Aplicados", f"{applied}/{total}")

# Resultado por símbolo
results = status.get("results", {})
if results:
    st.subheader("Resultado por Símbolo (Último Ciclo)")
    rows = []
    for sym, r in results.items():
        rows.append({
            "Símbolo": sym,
            "OOS Score": r.get("score", 0),
            "Aplicado": "Sim" if r.get("applied") else "Não",
            "Qualidade OK": "Sim" if r.get("passes_quality") else "Não",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ================================================================
# PARÂMETROS ATUAIS
# ================================================================
st.markdown("---")
st.header("Parâmetros Otimizados vs Defaults")

defaults = BacktestParams()
symbols_data = params.get("symbols", {})

if symbols_data:
    updated = params.get("updated_at", "N/A")
    st.info(f"Última atualização: {updated}")

    for sym, sym_data in symbols_data.items():
        with st.expander(f"{sym} — score={sym_data.get('score', 0):.4f}", expanded=False):
            opt_params = sym_data.get("params", {})
            metrics = sym_data.get("metrics", {})
            wf = sym_data.get("walk_forward", {})

            # Métricas OOS
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.metric("Win Rate OOS", f"{metrics.get('oos_win_rate', 0):.1f}%")
            with col_b:
                st.metric("Profit Factor", f"{metrics.get('oos_profit_factor', 0):.2f}")
            with col_c:
                st.metric("Return OOS", f"{metrics.get('oos_return', 0):.2f}%")
            with col_d:
                st.metric("Degradation", f"{wf.get('avg_degradation', 0):.1f}%")

            # Tabela de params otimizados vs defaults
            rows = []
            for field_name in BacktestParams.__dataclass_fields__:
                default_val = getattr(defaults, field_name)
                opt_val = opt_params.get(field_name, default_val)
                changed = abs(float(opt_val) - float(default_val)) > 0.01
                rows.append({
                    "Parâmetro": field_name,
                    "Default": default_val,
                    "Otimizado": opt_val,
                    "Alterado": "★" if changed else "",
                })
            df_params = pd.DataFrame(rows)
            st.dataframe(df_params, use_container_width=True, hide_index=True)
else:
    st.warning(
        "Nenhum parâmetro otimizado encontrado. "
        "O optimizer precisa completar pelo menos um ciclo."
    )

# ================================================================
# HISTÓRICO / LOG
# ================================================================
st.markdown("---")
st.header("Histórico de Otimizações")

if log:
    # Mostrar últimas 20 entradas
    recent = log[-20:]
    recent.reverse()

    rows = []
    for entry in recent:
        rows.append({
            "Timestamp": entry.get("timestamp", "")[:19],
            "Símbolo": entry.get("symbol", ""),
            "OOS Score": entry.get("best_oos_score", 0),
            "Degradation": f"{entry.get('avg_degradation', 0):.1f}%",
            "Qualidade OK": "Sim" if entry.get("passes_quality") else "Não",
            "Monte Carlo": "Pass" if entry.get("monte_carlo_pass") else "Fail",
            "Aplicado": "Sim" if entry.get("applied") else "Não",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Gráfico de evolução do score por símbolo
    st.subheader("Evolução do OOS Score")
    symbols_in_log = set(e.get("symbol", "") for e in log)

    fig = go.Figure()
    for sym in sorted(symbols_in_log):
        sym_entries = [e for e in log if e.get("symbol") == sym]
        if len(sym_entries) >= 2:
            fig.add_trace(go.Scatter(
                x=[e.get("timestamp", "")[:19] for e in sym_entries],
                y=[e.get("best_oos_score", 0) for e in sym_entries],
                name=sym,
                mode="lines+markers",
            ))

    if fig.data:
        fig.update_layout(
            xaxis_title="Data",
            yaxis_title="OOS Score",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Dados insuficientes para gráfico (precisa de pelo menos 2 ciclos por símbolo).")

    # Walk-forward detail para a entrada mais recente
    if recent and "windows" in recent[0]:
        st.subheader(f"Walk-Forward Detail — {recent[0].get('symbol', '')}")
        windows = recent[0]["windows"]
        wf_rows = []
        for w in windows:
            wf_rows.append({
                "Janela": w.get("window", 0),
                "IS Score": w.get("is_score", 0),
                "OOS Score": w.get("oos_score", 0),
                "Degradation": f"{w.get('degradation_pct', 0):.1f}%",
                "OOS Trades": w.get("oos_trades", 0),
                "OOS Win Rate": f"{w.get('oos_win_rate', 0):.1f}%",
                "OOS PF": w.get("oos_profit_factor", 0),
                "OOS Return": f"{w.get('oos_return', 0):.2f}%",
            })
        st.dataframe(pd.DataFrame(wf_rows), use_container_width=True, hide_index=True)
else:
    st.info("Nenhuma otimização registrada. Aguarde o optimizer completar o primeiro ciclo.")

# ================================================================
# SETUP VALIDATOR — Melhores e Piores Setups
# ================================================================
st.markdown("---")
st.header("Setup Validator — Contextos Históricos")

try:
    from src.optimizer.setup_validator import get_setup_validator
    sv = get_setup_validator()

    best_setups = sv.get_best_setups(min_samples=10)
    worst_setups = sv.get_worst_setups(min_samples=10)

    if best_setups or worst_setups:
        col_best, col_worst = st.columns(2)

        with col_best:
            st.subheader("Melhores Setups")
            if best_setups:
                best_rows = []
                for s in best_setups[:15]:
                    best_rows.append({
                        "Setup": s["setup"],
                        "Win Rate": f"{s['win_rate']:.1f}%",
                        "Avg PnL": f"{s['avg_pnl']:.2f}%",
                        "Amostras": s["total"],
                        "Avg Horas": f"{s.get('avg_hours', 0):.1f}",
                    })
                st.dataframe(pd.DataFrame(best_rows), use_container_width=True, hide_index=True)
            else:
                st.info("Nenhum setup com dados suficientes ainda.")

        with col_worst:
            st.subheader("Piores Setups (Evitar)")
            if worst_setups:
                worst_rows = []
                for s in worst_setups[:15]:
                    worst_rows.append({
                        "Setup": s["setup"],
                        "Win Rate": f"{s['win_rate']:.1f}%",
                        "Avg PnL": f"{s['avg_pnl']:.2f}%",
                        "Amostras": s["total"],
                    })
                st.dataframe(pd.DataFrame(worst_rows), use_container_width=True, hide_index=True)
            else:
                st.info("Nenhum setup ruim detectado ainda.")

        # Stats gerais
        total_setups = len(sv.statistics)
        total_signals = sum(s.get("total", 0) for s in sv.statistics.values())
        st.info(f"Total: {total_setups} setups rastreados, {total_signals} sinais validados")
    else:
        st.info("Setup Validator ainda sem dados. Aguarde o optimizer completar um ciclo.")
except Exception:
    st.info("Setup Validator não disponível.")

# ================================================================
# FOOTER
# ================================================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Optimizer Status | Atualiza a cada 60s</div>",
    unsafe_allow_html=True,
)
