"""
ML Model Dashboard - Painel de Acompanhamento
==============================================

Dashboard em Streamlit para acompanhar:
- Performance do modelo de validacao
- Historico de predicoes
- Online learning status
- Metricas em tempo real
- Auto-treino quando nao existe modelo
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# Configuracao da pagina
st.set_page_config(
    page_title="ML Signal Validator - Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    .stMetric {
        background: linear-gradient(145deg, #0f3460 0%, #1a1a2e 100%);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #e94560;
        box-shadow: 0 4px 15px rgba(233, 69, 96, 0.2);
    }
    .stMetric label {
        color: #a1a1aa !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #e94560 !important;
        font-size: 2rem !important;
    }
    h1, h2, h3 {
        color: #e94560 !important;
    }
    .info-card {
        background: linear-gradient(145deg, #0f3460 0%, #1a1a2e 100%);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #533483;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


def load_model_info():
    """Carrega informacoes do modelo"""
    info_path = "ml_models/model_info_simple.json"
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            return json.load(f)
    return {}


def load_performance_history():
    """Carrega historico de performance"""
    perf_path = "ml_models/model_performance.json"
    if os.path.exists(perf_path):
        with open(perf_path, 'r') as f:
            return json.load(f)
    return []


def load_prediction_log():
    """Carrega log de predicoes"""
    log_path = "ml_models/prediction_log.json"
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            return json.load(f)
    return []


def load_online_learning_buffer():
    """Carrega buffer do online learning"""
    buffer_path = "ml_models/online_learning_buffer.json"
    if os.path.exists(buffer_path):
        with open(buffer_path, 'r') as f:
            return json.load(f)
    return []


def model_exists():
    """Verifica se existe modelo treinado"""
    return os.path.exists("ml_models/signal_validators.pkl")


def main():
    st.title("🤖 ML Signal Validator - Dashboard")
    st.markdown("---")

    # Carregar dados
    model_info = load_model_info()
    perf_history = load_performance_history()
    predictions = load_prediction_log()
    buffer = load_online_learning_buffer()
    has_model = model_exists()

    # ================= ALERTA SE NAO TEM MODELO =================
    if not has_model:
        st.error("⚠️ **Nenhum modelo ML treinado!** O sistema precisa de um modelo para validar sinais.")
        st.markdown("""
        **Opcoes para criar o modelo inicial:**
        1. **Treinar do Zero (Bootstrap)** - Gera dados a partir de historico da Binance (recomendado para inicio)
        2. **Alimentar com Sinais** - Usa sinais ja avaliados do signal tracker (precisa de sinais existentes)
        """)

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🚀 Treinar Modelo do Zero (Bootstrap)", type="primary", use_container_width=True):
                with st.spinner("Treinando modelo via bootstrap (pode levar 1-2 min)..."):
                    try:
                        from src.ml.train_from_signals import run_training_pipeline
                        success = run_training_pipeline()
                        if success:
                            st.success("✅ Modelo treinado com sucesso!")
                            st.rerun()
                        else:
                            st.error("❌ Falha no treinamento. Verifique os logs.")
                    except Exception as e:
                        st.error(f"❌ Erro: {e}")

        with col_b:
            if st.button("📡 Alimentar com Sinais e Treinar", use_container_width=True):
                with st.spinner("Avaliando sinais e treinando..."):
                    from src.ml.online_learning import seed_from_evaluated_signals
                    result = seed_from_evaluated_signals(force_retrain=True)
                if result.get("success"):
                    rt = result.get("retrain_result", {})
                    if rt.get("success"):
                        st.success(f"✅ Modelo criado! Accuracy: {rt.get('new_accuracy', 0):.1%} | F1: {rt.get('new_f1', 0):.3f}")
                        st.rerun()
                    else:
                        st.warning(f"Sinais adicionados ao buffer ({result.get('buffer_total', 0)}), mas treino falhou: {rt.get('reason', rt.get('error', 'desconhecido'))}")
                else:
                    st.error(f"❌ {result.get('error', 'Erro desconhecido')}")

        if buffer:
            st.info(f"Buffer atual: {len(buffer)} amostras. Minimo para treinar: 10.")
            if len(buffer) >= 10:
                if st.button("🔄 Treinar com dados do Buffer"):
                    with st.spinner("Treinando..."):
                        from src.ml.online_learning import manual_retrain
                        result = manual_retrain()
                    if result.get("success"):
                        st.success(f"✅ Modelo criado! Accuracy: {result.get('new_accuracy', 0):.1%}")
                        st.rerun()
                    else:
                        st.error(f"Falhou: {result.get('reason', result.get('error', ''))}")

        st.markdown("---")

    # ================= SIDEBAR =================
    with st.sidebar:
        st.header("⚙️ Configuracoes")

        # Status do modelo
        st.subheader("📊 Status do Modelo")
        if has_model:
            best_model = model_info.get('best_model', 'N/A')
            st.success(f"**Modelo Ativo:** {best_model}")
        else:
            st.error("**Sem modelo treinado**")

        # Info
        if model_info:
            accuracy = model_info.get('best_accuracy', 0)
            f1 = model_info.get('best_f1', 0)
            st.metric("Accuracy", f"{accuracy:.1%}")
            st.metric("F1 Score", f"{f1:.3f}")

        # Online Learning
        st.subheader("🔄 Online Learning")
        buffer_size = len(buffer)
        threshold = 50
        st.metric("Buffer", f"{buffer_size}/{threshold}")
        st.progress(min(buffer_size / threshold, 1.0))

        if buffer_size >= threshold:
            st.warning(f"Buffer cheio! {buffer_size} amostras prontas para retreino.")

        st.caption(f"Retreino automatico ao atingir {threshold} amostras no buffer. "
                   f"Usa TODOS os dados do buffer + dataset original.")

        # Aviso: buffer cheio mas modelo ainda nao treinado (ex.: primeiro uso ou retreino falhou)
        if len(buffer) >= 50 and (not model_info or not model_info.get("best_model")):
            st.warning("Buffer cheio e nenhum modelo ativo. Clique em **Forcar Retreino** para treinar o modelo com os sinais do buffer.")

        if st.button("📡 Alimentar com Sinais", type="primary", use_container_width=True):
            with st.spinner("Avaliando sinais e populando buffer..."):
                from src.ml.online_learning import seed_from_evaluated_signals
                result = seed_from_evaluated_signals(force_retrain=True)
            if result.get("success"):
                st.success(f"✅ {result.get('signals_added', 0)} sinais adicionados! Buffer: {result.get('buffer_total', 0)}")
                if result.get("retrain_result", {}).get("success"):
                    rt = result["retrain_result"]
                    st.success(f"🎯 Modelo retreinado! Accuracy: {rt.get('new_accuracy', 0):.1%} | F1: {rt.get('new_f1', 0):.3f}")
                st.rerun()
            else:
                st.error(f"❌ {result.get('error', 'Erro desconhecido')}")

        if st.button("🔄 Forcar Retreino", use_container_width=True):
            with st.spinner("Retreinando modelo..."):
                from src.ml.online_learning import manual_retrain
                result = manual_retrain()
            if result.get("success"):
                st.success(f"✅ Accuracy: {result.get('new_accuracy', 0):.1%} | F1: {result.get('new_f1', 0):.3f}")
                st.rerun()
            else:
                st.error(f"Falhou: {result.get('reason', result.get('error', ''))}")

        # Info sobre retreino
        st.markdown("---")
        st.caption("**Como funciona o retreino:**")
        st.caption("• Combina dataset original + buffer")
        st.caption("• Treina ensemble (LogReg, RF, GB)")
        st.caption("• Salva apenas se F1 melhorar")
        st.caption("• Buffer limpo apos retreino")


    # ================= MAIN CONTENT =================
    if not has_model:
        return

    # Metricas principais
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Modelo Ativo",
            model_info.get('best_model', 'N/A'),
        )

    with col2:
        accuracy = model_info.get('best_accuracy', 0)
        st.metric(
            "Accuracy",
            f"{accuracy:.1%}",
            delta=f"+{accuracy-0.5:.1%} vs random" if accuracy > 0.5 else None
        )

    with col3:
        total_predictions = len(predictions)
        st.metric(
            "Total Predicoes",
            f"{total_predictions:,}"
        )

    with col4:
        retrains = model_info.get('retrain_count', 0)
        st.metric(
            "Retreinos",
            retrains
        )

    st.markdown("---")

    # ================= GRAFICOS =================

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("📈 Performance ao Longo do Tempo")

        if perf_history:
            df_perf = pd.DataFrame(perf_history)
            df_perf['timestamp'] = pd.to_datetime(df_perf['timestamp'])

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_perf['timestamp'],
                y=df_perf['accuracy'],
                mode='lines+markers',
                name='Accuracy',
                line=dict(color='#e94560', width=3),
                marker=dict(size=8)
            ))
            fig.add_trace(go.Scatter(
                x=df_perf['timestamp'],
                y=df_perf['f1_score'],
                mode='lines+markers',
                name='F1 Score',
                line=dict(color='#533483', width=3),
                marker=dict(size=8)
            ))

            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                yaxis=dict(range=[0, 1])
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Sem historico de performance ainda. O grafico aparecera apos o primeiro retreino.")

    with col_right:
        st.subheader("🎯 Distribuicao de Predicoes")

        if predictions:
            df_pred = pd.DataFrame(predictions)

            if 'predicted_success' in df_pred.columns:
                success_count = int(df_pred['predicted_success'].sum())
                fail_count = len(df_pred) - success_count

                fig = go.Figure(data=[go.Pie(
                    labels=['EXECUTE (Sucesso)', 'SKIP (Falha)'],
                    values=[success_count, fail_count],
                    hole=0.5,
                    marker_colors=['#2ecc71', '#e74c3c']
                )])

                fig.update_layout(
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=20, r=20, t=20, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)

                # Stats extras
                st.caption(f"Total: {len(df_pred)} | EXECUTE: {success_count} ({success_count/len(df_pred)*100:.0f}%) | SKIP: {fail_count} ({fail_count/len(df_pred)*100:.0f}%)")
        else:
            st.info("Sem predicoes registradas ainda. Predicoes aparecerao quando sinais forem validados pelo ML.")

    # ================= ONLINE LEARNING =================

    st.markdown("---")
    st.subheader("🔄 Online Learning - Buffer de Dados")

    if buffer:
        df_buffer = pd.DataFrame(buffer)

        # Stats do buffer
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            st.metric("Total no Buffer", len(buffer))
        with col_s2:
            tp_count = len([b for b in buffer if b.get('target') == 1])
            st.metric("TP (Sucesso)", tp_count)
        with col_s3:
            sl_count = len(buffer) - tp_count
            st.metric("SL (Falha)", sl_count)
        with col_s4:
            win_rate = tp_count / len(buffer) * 100 if buffer else 0
            st.metric("Win Rate Buffer", f"{win_rate:.1f}%")

        col1, col2 = st.columns(2)

        with col1:
            # Distribuicao por resultado
            if 'result' in df_buffer.columns:
                result_counts = df_buffer['result'].value_counts()
                fig = px.bar(
                    x=result_counts.index,
                    y=result_counts.values,
                    labels={'x': 'Resultado', 'y': 'Quantidade'},
                    color=result_counts.index,
                    color_discrete_map={
                        'TP1': '#2ecc71',
                        'TP2': '#27ae60',
                        'SL': '#e74c3c',
                        'TIMEOUT': '#f39c12'
                    }
                )
                fig.update_layout(
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Distribuicao por simbolo
            if 'symbol' in df_buffer.columns:
                symbol_counts = df_buffer['symbol'].value_counts()
                fig = px.pie(
                    values=symbol_counts.values,
                    names=symbol_counts.index,
                    hole=0.4
                )
                fig.update_layout(
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)

        # Tabela de dados recentes
        display_cols = [c for c in ['timestamp', 'symbol', 'signal_type', 'result', 'return_pct', 'rsi', 'confidence'] if c in df_buffer.columns]
        st.dataframe(
            df_buffer[display_cols].tail(15),
            use_container_width=True
        )
    else:
        st.info("Buffer de online learning vazio. Sinais serao adicionados automaticamente quando trades forem avaliados.")

    # ================= FEATURES IMPORTANCE =================

    st.markdown("---")
    st.subheader("🔍 Importancia das Features")

    features_importance = model_info.get('feature_importance', {})

    if features_importance:
        df_feat = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in features_importance.items()
        ]).sort_values('importance', ascending=True)

        fig = px.bar(
            df_feat,
            x='importance',
            y='feature',
            orientation='h',
            color='importance',
            color_continuous_scale='Reds'
        )
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            coloraxis_showscale=False
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Importancia das features nao disponivel. Disponivel para modelos RandomForest e GradientBoosting.")

    # ================= ULTIMAS PREDICOES =================

    st.markdown("---")
    st.subheader("📋 Ultimas Predicoes")

    if predictions:
        df_pred = pd.DataFrame(predictions)

        # Colorir recomendacao
        def color_rec(val):
            if val == 'EXECUTE':
                return 'color: #00cc00'
            elif val == 'SKIP':
                return 'color: #ff4444'
            return ''

        display_cols = [c for c in ['timestamp', 'symbol', 'deepseek_signal', 'predicted_success', 'probability', 'recommendation', 'model_used'] if c in df_pred.columns]
        styled = df_pred[display_cols].tail(20).style
        if 'recommendation' in display_cols:
            styled = styled.applymap(color_rec, subset=['recommendation'])
        st.dataframe(styled, use_container_width=True)
    else:
        st.info("Nenhuma predicao registrada ainda.")

    # ================= MODEL INFO =================

    st.markdown("---")
    with st.expander("🔧 Detalhes do Modelo"):
        if model_info:
            col1, col2 = st.columns(2)
            with col1:
                st.json({
                    "best_model": model_info.get("best_model"),
                    "training_date": model_info.get("training_date", model_info.get("last_retrain")),
                    "n_features": model_info.get("n_features"),
                    "train_samples": model_info.get("train_samples"),
                    "test_samples": model_info.get("test_samples"),
                    "retrain_count": model_info.get("retrain_count"),
                })
            with col2:
                results = model_info.get("results", {})
                if results:
                    st.json(results)


if __name__ == "__main__":
    main()
