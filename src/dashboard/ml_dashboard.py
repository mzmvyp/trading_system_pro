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

import asyncio
import glob
import json
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

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


def load_bilstm_model_info():
    """Carrega informações do modelo Bi-LSTM (bilstm_model_info.json)."""
    info_path = "ml_models/bilstm_model_info.json"
    if os.path.exists(info_path):
        try:
            with open(info_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def backtest_dataset_exists():
    """Verifica se existe dataset do backtest (X_train_latest.npy)."""
    return os.path.exists("ml_dataset/backtest/X_train_latest.npy")


def model_exists():
    """Verifica se existe modelo treinado"""
    return os.path.exists("ml_models/signal_validators.pkl")


def main():
    st.title("🤖 ML Signal Validator - Dashboard")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 ML Validator (sklearn)",
        "🧠 Bi-LSTM Sequence",
        "🔬 Backtest Explorer",
        "⚙️ Optimizer",
    ])

    with tab1:
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
                       f"Usa TODOS os sinais disponiveis (sem limite). "
                       f"Auto-retreino tambem roda a cada 12h no monitor mode (sklearn + Bi-LSTM).")

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

            # Retreino COMPLETO: pipeline com RandomizedSearchCV + Calibração (1–2 min)
            if st.button("🔬 Retreino completo (tuning + calibração)", use_container_width=True):
                with st.spinner("Pipeline completo: tuning de hiperparâmetros e calibração (pode levar 1–2 min)..."):
                    try:
                        from src.ml.train_from_signals import run_training_pipeline
                        success = run_training_pipeline()
                        if success:
                            st.success("✅ Pipeline completo concluído! Modelo tunado e calibrado salvo.")
                            st.rerun()
                        else:
                            st.error("Falhou: dados insuficientes ou erro no pipeline. Use 'Alimentar com Sinais' antes se tiver poucos sinais.")
                    except Exception as e:
                        st.error(f"Erro: {e}")

            # Info sobre retreino
            st.markdown("---")
            st.caption("**Por que ainda GradientBoosting?** O nome é o algoritmo que ganhou (melhor F1) entre LogReg, RF e GB no retreino rápido.")
            st.caption("**Dois tipos de retreino:**")
            st.caption("• **Forcar Retreino** = rápido, 3 modelos com params fixos, escolhe o melhor.")
            st.caption("• **Retreino completo** = 1–2 min, RandomizedSearchCV + CalibratedClassifierCV (prob mais realista).")
            st.caption("• **Alimentar com Sinais** = preenche o buffer com sinais avaliados (SL/TP) e dispara retreino rápido se buffer ≥ 10.")
            st.caption("• Buffer limpo apos retreino. Auto-retreino a cada 12h no monitor.")

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
                styled = styled.map(color_rec, subset=['recommendation'])
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

    with tab2:
        _render_bilstm_tab()
    with tab3:
        _render_backtest_explorer_tab()
    with tab4:
        _render_optimizer_tab()


def _render_bilstm_tab():
    """Aba Bi-LSTM: status, gerar dataset, treinar."""
    st.subheader("🧠 Bi-LSTM Sequence Validator")
    st.caption("Modelo de sequência temporal para validar sinais. Treinado com sinais reais.")

    bilstm_info = load_bilstm_model_info()
    has_bilstm = bool(bilstm_info)

    # Sidebar: parâmetros para dataset e treino
    with st.sidebar:
        st.subheader("🧠 Bi-LSTM - Parâmetros")
        seq_length = st.slider("Sequence Length", 20, 120, 60, key="bilstm_seq")
        epochs_bilstm = st.slider("Epochs", 10, 200, 100, key="bilstm_epochs")
        batch_size_bilstm = st.selectbox("Batch size", [16, 32, 64], index=1, key="bilstm_batch")
        n_optuna_trials = st.slider("Optuna Trials", 10, 50, 20, key="bilstm_optuna_trials")
        n_wf_folds = st.slider("Walk-Forward Folds", 3, 8, 5, key="bilstm_wf_folds")

    if has_bilstm:
        res = bilstm_info.get("results", {})
        train_res = res.get("train", {})
        test_res = res.get("test", {})
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Tipo", bilstm_info.get("type", "Bi-LSTM"))
            st.metric("Data treino", bilstm_info.get("training_date", "N/A")[:10] if isinstance(bilstm_info.get("training_date"), str) else "N/A")
        with col2:
            st.metric("Accuracy (treino)", f"{train_res.get('accuracy', 0):.1%}")
            st.metric("F1 (treino)", f"{train_res.get('f1_score', 0):.3f}")
        with col3:
            st.metric("Accuracy (teste)", f"{test_res.get('accuracy', 0):.1%}")
            st.metric("F1 (teste)", f"{test_res.get('f1_score', 0):.3f}")
        st.metric("Sequence length", bilstm_info.get("sequence_length", "N/A"))
        st.metric("Features", bilstm_info.get("n_features", "N/A"))
        st.metric("Amostras treino/teste", f"{bilstm_info.get('train_samples', 0)} / {bilstm_info.get('test_samples', 0)}")
        st.metric("Fonte", bilstm_info.get("data_source", "N/A"))
        # Métricas avançadas
        if test_res.get("auc"):
            st.metric("AUC (teste)", f"{test_res['auc']:.3f}")
        if test_res.get("balanced_accuracy"):
            st.metric("Balanced Acc (teste)", f"{test_res['balanced_accuracy']:.1%}")
        if test_res.get("op_accuracy"):
            st.metric("Op Accuracy (teste)", f"{test_res['op_accuracy']:.1%} (n={test_res.get('n_operational', 0)})")
        if bilstm_info.get("optuna_best_score"):
            st.metric("Optuna Score", f"{bilstm_info['optuna_best_score']:.4f}")
            st.metric("Op Accuracy (Optuna)", f"{bilstm_info.get('optuna_op_accuracy', 0):.1%}")
        # Walk-Forward metrics
        wf = bilstm_info.get("walk_forward")
        if wf:
            st.divider()
            st.caption("📈 Walk-Forward Validation")
            wf_cols = st.columns(3)
            with wf_cols[0]:
                st.metric("WF Score", f"{wf['avg_score']:.4f} ± {wf['std_score']:.4f}")
            with wf_cols[1]:
                st.metric("WF Op Acc", f"{wf['avg_op_accuracy']:.1%} ± {wf['std_op_accuracy']:.1%}")
            with wf_cols[2]:
                st.metric("WF AUC", f"{wf['avg_auc']:.3f}")
            st.caption(f"Folds: {wf['n_rounds']} rounds avaliados")
    else:
        st.warning("Nenhum modelo Bi-LSTM treinado. Gere o dataset e treine abaixo.")

    dataset_ok = backtest_dataset_exists()
    if not dataset_ok:
        st.info("Gere o dataset de sinais reais primeiro.")

    # ===== GERAR DATASET DE SINAIS REAIS =====
    if st.button("📊 Gerar Dataset de Sinais Reais", type="primary", use_container_width=True):
        with st.spinner("Gerando dataset a partir dos sinais reais (pode levar 5-15 min)..."):
            try:
                from src.ml.real_signal_dataset_generator import RealSignalDatasetGenerator
                generator = RealSignalDatasetGenerator(
                    sequence_length=seq_length,
                    max_signals=5000,
                )
                stats = asyncio.run(generator.generate())
                total = stats.get("total_trades", 0)
                if total >= 50 and "error" not in stats:
                    wins = stats.get("winning_trades", 0)
                    losses = stats.get("losing_trades", 0)
                    st.success(
                        f"Dataset gerado: {total} sinais reais | "
                        f"{wins} wins / {losses} losses | "
                        f"Pares: {len(stats.get('symbols_processed', []))}"
                    )
                else:
                    st.warning(f"Poucos sinais com outcome ({total}). Fallback para backtest...")
                    from src.ml.backtest_dataset_generator import BacktestDatasetGenerator
                    bt_gen = BacktestDatasetGenerator(
                        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
                        interval="1h", sequence_length=seq_length,
                        days_back=180, n_param_variations=20,
                    )
                    stats = asyncio.run(bt_gen.generate())
                    st.success(f"Dataset backtest gerado: {stats.get('total_trades', 0)} trades")
                st.rerun()
            except Exception as e:
                st.error(f"Erro ao gerar dataset: {e}")

    # ===== TREINAR BI-LSTM =====
    if dataset_ok and st.button("🎯 Treinar Bi-LSTM", use_container_width=True):
        with st.spinner("Treinando Bi-LSTM..."):
            try:
                from src.ml.lstm_sequence_validator import LSTMSequenceValidator
                validator = LSTMSequenceValidator()
                results = validator.train(epochs=epochs_bilstm, batch_size=batch_size_bilstm)
                test_acc = results.get("test", {}).get("accuracy", 0)
                test_f1 = results.get("test", {}).get("f1_score", 0)
                st.success(f"Treino concluido! Accuracy: {test_acc:.1%} | F1: {test_f1:.3f}")
                st.rerun()
            except Exception as e:
                st.error(f"Erro ao treinar: {e}")

    # ===== OPTUNA — OTIMIZAR HIPERPARÂMETROS =====
    if dataset_ok and st.button("🔬 Otimizar com Optuna", use_container_width=True):
        with st.spinner(f"Otimizando hiperparametros ({n_optuna_trials} trials)... pode levar 30-60 min"):
            try:
                from src.ml.lstm_sequence_validator import LSTMSequenceValidator
                validator = LSTMSequenceValidator()
                results = validator.train_with_optuna(
                    n_trials=n_optuna_trials,
                    epochs_per_trial=30,
                )
                if results.get("success"):
                    bp = results.get("best_params", {})
                    st.success(
                        f"Otimizacao concluida! Score: {results.get('best_score', 0):.4f} | "
                        f"Op Accuracy: {results.get('op_accuracy', 0):.1%} "
                        f"(n={results.get('n_operational', 0)}) | "
                        f"Units: {bp.get('units_1', '?')}+{bp.get('units_2', '?')} | "
                        f"LR: {bp.get('lr', '?'):.5f}"
                    )
                else:
                    st.error(f"Erro: {results.get('reason', 'desconhecido')}")
                st.rerun()
            except Exception as e:
                st.error(f"Erro ao otimizar: {e}")

    # ===== WALK-FORWARD VALIDATION =====
    if dataset_ok and st.button("📈 Walk-Forward Validation", use_container_width=True,
                                 help="Treina/testa em janelas temporais rolantes — métrica mais realista"):
        with st.spinner(f"Walk-Forward ({n_wf_folds} folds × {n_optuna_trials} trials)... pode levar 1-2h"):
            try:
                from src.ml.lstm_sequence_validator import LSTMSequenceValidator
                validator = LSTMSequenceValidator()
                results = validator.walk_forward_validate(
                    n_folds=n_wf_folds,
                    n_optuna_trials=n_optuna_trials,
                    epochs_per_trial=20,
                )
                if results.get("success"):
                    st.success(
                        f"Walk-Forward concluído! "
                        f"Score: {results['avg_score']:.4f} ± {results['std_score']:.4f} | "
                        f"Op Acc: {results['avg_op_accuracy']:.1%} | "
                        f"AUC: {results['avg_auc']:.3f} | "
                        f"F1: {results['avg_f1']:.3f} | "
                        f"Rounds: {results['n_rounds']}"
                    )
                    # Mostrar resultados por fold
                    with st.expander("Detalhes por fold"):
                        for fr in results["fold_results"]:
                            st.write(
                                f"**Fold {fr['fold']}**: score={fr['best_score']:.4f} | "
                                f"op_acc={fr['op_accuracy']:.1%} | AUC={fr['auc']:.3f} | "
                                f"train={fr['train_size']} test={fr['test_size']}"
                            )
                else:
                    st.error(f"Erro: {results.get('reason', 'desconhecido')}")
                st.rerun()
            except Exception as e:
                st.error(f"Erro no Walk-Forward: {e}")


async def _fetch_data_with_timeout(engine, symbol: str, interval: str, start_dt, end_dt, timeout: int = 90):
    """Executa fetch_data com timeout para não travar o dashboard."""
    return await asyncio.wait_for(
        engine.fetch_data(symbol, interval, start_dt, end_dt),
        timeout=float(timeout),
    )


def _render_backtest_explorer_tab():
    """Aba Backtest Explorer: rodar backtest avulso e visualizar."""
    st.subheader("🔬 Backtest Explorer")
    st.caption("Avalie como a estratégia teria performado em um período histórico.")
    st.info(
        "**Manual:** você escolhe o período e clica em Rodar. "
        "O **Optimizer** (aba Optimizer / bot no Docker) roda **automaticamente a cada 6h** e salva o melhor setup em `data/optimization/`."
    )

    symbol_bt = st.selectbox("Símbolo", ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"], key="bt_symbol")
    interval_bt = st.selectbox("Intervalo", ["1h", "4h", "15m", "5m"], index=0, key="bt_interval")
    end_d = datetime.now(timezone.utc).date()
    start_d = end_d - timedelta(days=90)
    start_date = st.date_input("Início", start_d, key="bt_start")
    end_date = st.date_input("Fim", end_d, key="bt_end")

    if st.button("▶ Rodar Backtest", type="primary", key="bt_run"):
        with st.status("Buscando dados e rodando backtest...") as status:
            try:
                from src.backtesting.backtest_engine import BacktestEngine
                start_dt = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc)
                end_dt = datetime.combine(end_date, datetime.min.time()).replace(tzinfo=timezone.utc)
                engine = BacktestEngine()
                status.update(label="Buscando dados na Binance (máx. 90s)...", state="running")
                # Timeout para não travar: 90s (períodos longos = vários requests)
                try:
                    df = asyncio.run(_fetch_data_with_timeout(engine, symbol_bt, interval_bt, start_dt, end_dt, timeout=90))
                except asyncio.TimeoutError:
                    st.error("Timeout ao buscar dados. Tente um período menor (ex.: 30–60 dias) ou outro intervalo.")
                    status.update(label="Timeout.", state="error")
                    return
                if df.empty or len(df) < 50:
                    st.error("Dados insuficientes para o período. Tente outro intervalo ou datas.")
                    return
                status.update(label="Calculando indicadores e sinais...", state="running")
                df = engine.calculate_indicators(df)
                df = engine.generate_signals(df)
                trades = engine.simulate_trades(df)
                metrics = engine.calculate_metrics(trades)
                status.update(label="Concluído.", state="complete")
            except Exception as e:
                st.error(f"Erro no backtest (ex.: Binance API): {e}")
                return

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Win Rate", f"{metrics.win_rate:.1f}%")
        col2.metric("Return %", f"{metrics.total_return_pct:.2f}%")
        col3.metric("Sharpe", f"{metrics.sharpe_ratio:.2f}")
        col4.metric("Max DD %", f"{metrics.max_drawdown_pct:.2f}%")
        col5.metric("Profit Factor", f"{metrics.profit_factor:.2f}" if metrics.profit_factor != float('inf') else "∞")
        col6.metric("Trades", metrics.total_trades)

        if metrics.equity_curve:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=metrics.equity_curve, mode='lines', name='Equity', line=dict(color='#e94560')))
            fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', title="Curva de equity")
            st.plotly_chart(fig, use_container_width=True)

        if metrics.trades:
            rows = []
            for t in metrics.trades[:50]:
                rows.append({
                    "entry_time": str(t.entry_time)[:19] if t.entry_time else "",
                    "direction": t.direction,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "pnl_pct": round(t.pnl_pct, 2),
                    "exit_reason": t.exit_reason,
                    "is_winner": t.is_winner,
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            pnls = [t.pnl_pct for t in metrics.trades]
            fig_hist = px.histogram(x=pnls, nbins=30, labels={'x': 'PnL %'}, title="Distribuição PnL")
            fig_hist.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_hist, use_container_width=True)
            wins = sum(1 for t in metrics.trades if t.is_winner)
            fig_pie = go.Figure(data=[go.Pie(labels=["Win", "Loss"], values=[wins, len(metrics.trades) - wins], hole=0.5, marker_colors=['#2ecc71', '#e74c3c'])])
            fig_pie.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', title="Win / Loss")
            st.plotly_chart(fig_pie, use_container_width=True)


def _render_optimizer_tab():
    """Aba Optimizer: resultados do continuous optimizer e rodar otimização."""
    st.subheader("⚙️ Resultados do Optimizer")
    st.caption("Configurações salvas pelo continuous optimizer e execução manual.")

    # Mostrar resultado da última otimização manual (se existir)
    if st.session_state.get("last_optimization_done"):
        msg = st.session_state.get("last_optimization_done")
        st.success(f"✅ **Última otimização manual:** {msg}")
        if st.button("Limpar mensagem", key="clear_opt_msg"):
            st.session_state.pop("last_optimization_done", None)
            st.rerun()

    config_files = sorted(glob.glob("data/optimization/best_config_*.json"))
    if config_files:
        rows = []
        for path in config_files:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                name = os.path.basename(path).replace("best_config_", "").replace(".json", "")
                metrics = data.get("metrics") or {}
                rows.append({
                    "arquivo": name,
                    "score": data.get("score", 0),
                    "win_rate": metrics.get("win_rate", 0),
                    "return_pct": metrics.get("total_return_pct", 0),
                    "sharpe": metrics.get("sharpe_ratio", 0),
                    "drawdown": metrics.get("max_drawdown_pct", 0),
                })
            except (json.JSONDecodeError, IOError):
                continue
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        for path in config_files[:5]:
            with st.expander(os.path.basename(path)):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        st.json(json.load(f))
                except Exception:
                    pass
    else:
        st.info("Nenhum resultado de otimização encontrado em data/optimization/.")

    st.subheader("Rodar otimização agora")
    sym_opt = st.selectbox("Símbolo", ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "XRPUSDT", "BNBUSDT"], key="opt_symbol")
    interval_opt = st.selectbox("Intervalo", ["1h", "4h"], index=0, key="opt_interval")
    n_iter = st.slider("Iterações", 50, 500, 200, key="opt_niter")
    end_opt = datetime.now(timezone.utc)
    start_opt = end_opt - timedelta(days=180)

    if st.button("▶ Rodar Otimização", type="primary", key="opt_run"):
        with st.status("Otimizando... Aguarde o término (pode levar vários minutos).") as status:
            try:
                status.update(label="Iniciando otimização...", state="running")
                from src.backtesting.optimization_engine import OptimizationEngine
                engine = OptimizationEngine(symbol=sym_opt, interval=interval_opt)
                status.update(label=f"Rodando {n_iter} iterações para {sym_opt}...", state="running")
                results = asyncio.run(engine.run_optimization(start_opt, end_opt, n_iterations=n_iter))
                if results:
                    engine.save_results()
                    top = engine.get_top_results(10)
                    best = top[0]
                    m = best.metrics
                    # Salvar best_config para o símbolo (mesmo formato do continuous optimizer)
                    try:
                        from src.backtesting.continuous_optimizer import save_best_config
                        save_best_config(
                            symbol=sym_opt,
                            interval=interval_opt,
                            params=best.params,
                            score=best.score,
                            metrics={
                                "total_trades": m.total_trades,
                                "win_rate": round(m.win_rate, 2),
                                "total_return_pct": round(m.total_return_pct, 2),
                                "sharpe_ratio": round(m.sharpe_ratio, 2),
                                "max_drawdown_pct": round(m.max_drawdown_pct, 2),
                                "profit_factor": round(m.profit_factor, 2),
                            },
                        )
                    except Exception as save_err:
                        st.warning(f"Config salvo nos resultados; best_config não atualizado: {save_err}")
                    status.update(
                        label=f"Concluído. Melhor score: {best.score:.4f} | WR: {m.win_rate:.1f}% | Return: {m.total_return_pct:.2f}%",
                        state="complete"
                    )
                    st.session_state["last_optimization_done"] = (
                        f"{sym_opt} {interval_opt} — score {best.score:.4f}, "
                        f"WR {m.win_rate:.1f}%, Return {m.total_return_pct:.2f}%, Sharpe {m.sharpe_ratio:.2f}, DD {m.max_drawdown_pct:.2f}%"
                    )
                    st.success(f"✅ **Otimização concluída.** Melhor: score={best.score:.4f}, WR={m.win_rate:.1f}%, Return={m.total_return_pct:.2f}%")
                    for i, r in enumerate(top[:10], 1):
                        mr = r.metrics
                        st.write(f"{i}. Score: {r.score:.4f} | WR: {mr.win_rate:.1f}% | Return: {mr.total_return_pct:.2f}% | Sharpe: {mr.sharpe_ratio:.2f} | DD: {mr.max_drawdown_pct:.2f}%")
                    st.rerun()
                else:
                    status.update(label="Nenhum resultado.", state="error")
                    st.warning("Nenhum resultado (dados insuficientes?).")
            except Exception as e:
                status.update(label="Erro.", state="error")
                st.error(f"Erro na otimização: {e}")


if __name__ == "__main__":
    main()
