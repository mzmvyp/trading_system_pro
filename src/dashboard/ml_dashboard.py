"""
ML Model Dashboard - Painel de Acompanhamento
==============================================

Dashboard em Streamlit para acompanhar:
- Performance do modelo de validacao
- Historico de predicoes
- Online learning status
- Metricas em tempo real

Autor: Trading Bot
Data: 2026-01-13
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
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


def main():
    st.title("🤖 ML Signal Validator - Dashboard")
    st.markdown("---")
    
    # Carregar dados
    model_info = load_model_info()
    perf_history = load_performance_history()
    predictions = load_prediction_log()
    buffer = load_online_learning_buffer()
    
    # ================= SIDEBAR =================
    with st.sidebar:
        st.header("⚙️ Configuracoes")
        
        # Status do modelo
        st.subheader("📊 Status do Modelo")
        best_model = model_info.get('best_model', 'N/A')
        st.info(f"**Modelo Ativo:** {best_model}")
        
        # Info
        if model_info:
            st.metric("Accuracy", f"{model_info.get('best_accuracy', 0):.1%}")
            st.metric("F1 Score", f"{model_info.get('best_f1', 0):.3f}")
            
        # Online Learning
        st.subheader("🔄 Online Learning")
        st.metric("Buffer", f"{len(buffer)}/50")
        st.progress(min(len(buffer) / 50, 1.0))

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
            from src.ml.online_learning import manual_retrain
            result = manual_retrain()
            if result.get("success"):
                st.success(f"Modelo treinado! Accuracy: {result.get('new_accuracy', 0):.1%} | F1: {result.get('new_f1', 0):.3f}")
                st.rerun()
            else:
                st.error(result.get("error", result.get("reason", str(result))))
            
    # ================= MAIN CONTENT =================
    
    # Metricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Modelo Ativo",
            model_info.get('best_model', 'N/A'),
            delta=None
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
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Sem historico de performance ainda. O grafico aparecera apos o primeiro retreino.")
            
    with col_right:
        st.subheader("🎯 Distribuicao de Predicoes")
        
        if predictions:
            df_pred = pd.DataFrame(predictions)
            
            # Contar predicoes
            if 'predicted_success' in df_pred.columns:
                success_count = df_pred['predicted_success'].sum()
                fail_count = len(df_pred) - success_count
                
                fig = go.Figure(data=[go.Pie(
                    labels=['Sucesso', 'Falha'],
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
        else:
            st.info("Sem predicoes registradas ainda.")
            
    # ================= ONLINE LEARNING =================
    
    st.markdown("---")
    st.subheader("🔄 Online Learning - Buffer de Dados")
    
    if buffer:
        df_buffer = pd.DataFrame(buffer)
        
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
                
        # Tabela de dados
        st.dataframe(
            df_buffer[['timestamp', 'symbol', 'signal_type', 'result', 'return_pct']].tail(10),
            use_container_width=True
        )
    else:
        st.info("Buffer de online learning vazio. Resultados de trades serao adicionados automaticamente.")
        
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
        st.info("Importancia das features nao disponivel para este modelo.")
        
    # ================= ULTIMAS PREDICOES =================
    
    st.markdown("---")
    st.subheader("📋 Ultimas Predicoes")
    
    if predictions:
        df_pred = pd.DataFrame(predictions)
        st.dataframe(df_pred.tail(20), use_container_width=True)
    else:
        st.info("Nenhuma predicao registrada ainda.")
        

if __name__ == "__main__":
    main()

