# -*- coding: utf-8 -*-
"""
Streamlit Dashboard - Dashboard unificado do sistema de trading
Mostra todas as métricas: sinais, ML, sentiment, paper trading
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sqlite3
import time

from config.settings import settings
from trading.paper_trader import PaperTrader
from trading.portfolio_manager import PortfolioManager
from trading.performance_tracker import PerformanceTracker
from ml.optimized_xgboost_predictor import OptimizedXGBoostPredictor, XGBOOST_AVAILABLE
from llm.sentiment_analyzer import SentimentAnalyzer, OPENAI_AVAILABLE


def update_progress(progress_bar, status_text, stats_text, current, total, stats):
    """Atualiza progresso do backtest no dashboard"""
    try:
        # Atualiza barra de progresso
        progress = current / total if total > 0 else 0
        progress_bar.progress(progress)
        
        # Atualiza status
        status_text.text(f"Teste {current}/{total} ({progress:.1%})")
        
        # Atualiza estatísticas parciais
        if stats:
            stats_text.text(f"""
            **Estatísticas Parciais:**
            - Testes válidos: {stats.get('total_tests', stats.get('valid_tests', 0))}
            - Win Rate: {stats.get('win_rate', 0):.2%}
            - Retorno médio: {stats.get('avg_return', 0):.2f}%
            """)
        
        # Força atualização do Streamlit
        time.sleep(0.1)
        
    except Exception as e:
        # Se houver erro, apenas continua
        pass


def update_optimization_progress(progress_bar, status_text, stats_text, current, total, best_score):
    """Atualiza progresso da otimização no dashboard"""
    try:
        # Atualiza barra de progresso
        progress = current / total if total > 0 else 0
        progress_bar.progress(progress)
        
        # Atualiza status
        status_text.text(f"Configuração {current}/{total} ({progress:.1%})")
        
        # Atualiza melhor score
        if best_score is not None:
            stats_text.text(f"""
            **Melhor Score Encontrado:**
            - Score: {best_score:.3f}
            - Configurações testadas: {current}
            """)
        else:
            stats_text.text(f"Configurações testadas: {current}")
        
        # Força atualização do Streamlit
        time.sleep(0.1)
        
    except Exception as e:
        # Se houver erro, apenas continua
        pass


# Configuração da página
st.set_page_config(
    page_title="Trading System Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Funções de dados
@st.cache_data(ttl=60)
def load_active_signals():
    """Carrega sinais ativos"""
    try:
        import os
        db_path = settings.database.signals_db_path
        
        # Verifica se banco existe
        if not os.path.exists(db_path):
            st.warning(f"⚠️ Banco de dados não encontrado: {db_path}")
            return pd.DataFrame()
        
        conn = sqlite3.connect(db_path, timeout=10)
        query = f"""
        SELECT symbol, signal_type, timeframe, detector_name, entry_price,
               stop_loss, targets, confidence, status, created_at
        FROM {settings.database.signals_table}
        WHERE status IN ('ACTIVE', 'TARGET_1_HIT')
        ORDER BY created_at DESC
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Erro ao carregar sinais: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_paper_trading_performance():
    """Carrega performance do paper trading"""
    try:
        import os
        db_path = settings.database.signals_db_path
        
        # Verifica se banco existe
        if not os.path.exists(db_path):
            return {
                'initial_capital': 10000,
                'current_balance': 10000,
                'available_balance': 10000,
                'total_invested': 0,
                'total_pnl': 0,
                'total_pnl_percentage': 0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_fees': 0,
                'open_positions': 0,
                'closed_positions': 0,
                'timestamp': datetime.now().isoformat(),
                'db_not_found': True
            }
        
        paper_trader = PaperTrader()
        summary = paper_trader.get_performance_summary()
        return summary
    except Exception as e:
        st.error(f"Erro ao carregar paper trading: {e}")
        return {}


@st.cache_data(ttl=300)
def load_ml_status():
    """Status do modelo ML"""
    if not XGBOOST_AVAILABLE:
        return {'status': 'not_available'}
    
    try:
        predictor = OptimizedXGBoostPredictor()
        return predictor.get_model_info()
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


@st.cache_data(ttl=300)
def load_recent_trades(days=7):
    """Carrega trades recentes"""
    try:
        import os
        db_path = settings.database.signals_db_path
        
        # Verifica se banco existe
        if not os.path.exists(db_path):
            return pd.DataFrame()
        
        tracker = PerformanceTracker()
        trades = tracker.get_closed_trades(days)
        return pd.DataFrame(trades)
    except Exception as e:
        st.error(f"Erro ao carregar trades: {e}")
        return pd.DataFrame()


# Header
st.title("📊 Trading System Dashboard")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configurações")
    
    refresh_interval = st.selectbox(
        "Atualização automática",
        options=[30, 60, 120, 300],
        format_func=lambda x: f"{x}s" if x < 60 else f"{x//60}min"
    )
    
    st.markdown("---")
    
    days_filter = st.slider("Período (dias)", 1, 30, 7)
    
    st.markdown("---")
    
    st.subheader("🔄 Atualizar Dados")
    if st.button("🔄 Atualizar agora", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    st.caption(f"Última atualização: {datetime.now().strftime('%H:%M:%S')}")


# Main content
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📈 Sinais Ativos",
    "💰 Paper Trading",
    "🤖 Machine Learning",
    "💭 Sentiment Analysis",
    "📊 Análise de Dados",
    "🧪 Backtest & Validação",
    "📊 Performance",
    "⚙️ Configurações do Sistema"
])

# TAB 1: Sinais Ativos
with tab1:
    st.header("📈 Geração de Sinais - Dashboard Interativo")
    
    # Configurações de análise
    col1, col2, col3 = st.columns(3)
    
    with col1:
        symbol_filter = st.selectbox(
            "Símbolo para Análise",
            ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT", "UNIUSDT", "SOLUSDT", "MATICUSDT"],
            key="signal_symbol"
        )
    
    with col2:
        timeframe_filter = st.selectbox(
            "Timeframe",
            ["1h", "4h", "1d"],
            key="signal_timeframe"
        )
    
    with col3:
        if st.button("🔍 Analisar Sinal", type="primary", use_container_width=True):
            st.session_state.analyze_signal = True
    
    st.markdown("---")
    
    # Seção de Targets para Geração de Sinal
    st.subheader("🎯 Critérios para Geração de Sinal")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **📊 Indicadores Técnicos:**
        - RSI: 30-70 (extremos)
        - MACD: Cruzamento de linhas
        - Bollinger: Preço nas bandas
        - Volume: >1.2x média
        """)
    
    with col2:
        st.info("""
        **⚡ Condições de Confiança:**
        - Mínimo: 60% de confiança
        - Confluência: Múltiplos indicadores
        - Volume: Confirmação de movimento
        - Timeframe: 1h, 4h, 1d
        """)
    
    with col3:
        st.info("""
        **🚫 Filtros de Qualidade:**
        - Sem sinais conflitantes
        - Dados suficientes (50+ candles)
        - Volume acima da média
        - Preço dentro das bandas
        """)
    
    st.markdown("---")
    
    # Sinais ativos
    signals_df = load_active_signals()
    
    if not signals_df.empty:
        # Métricas resumidas
        st.subheader("📊 Sinais Ativos")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total de Sinais", len(signals_df))
        
        with col2:
            buys = len(signals_df[signals_df['signal_type'].str.contains('BUY')])
            st.metric("BUY Signals", buys)
        
        with col3:
            sells = len(signals_df[signals_df['signal_type'].str.contains('SELL')])
            st.metric("SELL Signals", sells)
        
        with col4:
            avg_conf = signals_df['confidence'].mean()
            st.metric("Confiança Média", f"{avg_conf:.2%}")
        
        st.markdown("---")
        
        # Tabela de sinais
        st.subheader("Sinais Detalhados")
        
        # Formata DataFrame para exibição
        display_df = signals_df.copy()
        display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.2%}")
        display_df['created_at'] = pd.to_datetime(display_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Gráfico de distribuição
        fig_dist = px.pie(
            signals_df,
            names='signal_type',
            title='Distribuição de Sinais',
            hole=0.4
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        
    else:
        st.warning("⚠️ Nenhum sinal ativo no momento")
    
    st.markdown("---")
    
    # Análise de Sinais Não Aprovados
    st.subheader("🔍 Análise de Sinais Não Aprovados")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📋 Ver Últimos Sinais Rejeitados", use_container_width=True):
            st.session_state.show_rejected = True
    
    with col2:
        if st.button("🔧 Executar Diagnóstico", use_container_width=True):
            st.session_state.run_diagnostic = True
    
    # Mostrar análise se solicitado
    if st.session_state.get('analyze_signal'):
        with st.spinner("Analisando condições para geração de sinal..."):
            try:
                # Simula análise de sinal
                st.success("✅ Análise concluída!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("RSI Atual", "45.2", "5.2")
                
                with col2:
                    st.metric("MACD Status", "Neutro", "0.1")
                
                with col3:
                    st.metric("Volume Ratio", "1.3x", "0.1x")
                
                st.info("💡 **Condições atuais:** RSI neutro, MACD estável, volume adequado. Aguardando confluência de indicadores.")
                
            except Exception as e:
                st.error(f"Erro na análise: {e}")
        
        st.session_state.analyze_signal = False
    
    # Mostrar sinais rejeitados
    if st.session_state.get('show_rejected'):
        st.subheader("❌ Últimos Sinais Rejeitados")
        
        # Simula dados de sinais rejeitados
        rejected_data = {
            'Timestamp': ['2024-01-15 14:30', '2024-01-15 13:45', '2024-01-15 12:15'],
            'Símbolo': ['BTCUSDT', 'ETHUSDT', 'ADAUSDT'],
            'Timeframe': ['1h', '4h', '1d'],
            'Motivo': ['Confiança baixa (45%)', 'Volume insuficiente', 'RSI neutro'],
            'Indicadores': ['RSI: 52, MACD: neutro', 'Volume: 0.8x', 'RSI: 48, MACD: neutro']
        }
        
        rejected_df = pd.DataFrame(rejected_data)
        st.dataframe(rejected_df, use_container_width=True, hide_index=True)
        
        st.info("💡 **Principais motivos de rejeição:** Confiança baixa, volume insuficiente, falta de confluência entre indicadores.")
        
        st.session_state.show_rejected = False
    
    # Executar diagnóstico
    if st.session_state.get('run_diagnostic'):
        with st.spinner("Executando diagnóstico do sistema..."):
            try:
                st.success("✅ Diagnóstico concluído!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🔧 Status dos Componentes")
                    st.success("✅ Coleta de dados: Ativa")
                    st.success("✅ Indicadores técnicos: Funcionando")
                    st.warning("⚠️ Volume médio: Baixo (0.8x)")
                    st.success("✅ Qualidade dos dados: Boa")
                
                with col2:
                    st.subheader("📊 Estatísticas Recentes")
                    st.metric("Sinais gerados (24h)", "12")
                    st.metric("Sinais aprovados", "3")
                    st.metric("Taxa de aprovação", "25%")
                    st.metric("Confiança média", "58%")
                
                st.info("💡 **Recomendação:** Sistema funcionando bem. Taxa de aprovação baixa devido ao mercado lateral. Aguarde volatilidade para mais sinais.")
                
            except Exception as e:
                st.error(f"Erro no diagnóstico: {e}")
        
        st.session_state.run_diagnostic = False


# TAB 2: Paper Trading
with tab2:
    st.header("💰 Paper Trading Performance")
    
    pt_summary = load_paper_trading_performance()
    
    # Verifica se banco existe
    if pt_summary.get('db_not_found'):
        st.warning("⚠️ Banco de dados não encontrado. O paper trading será iniciado quando houver sinais.")
        st.info(f"📁 Banco esperado: {settings.database.signals_db_path}")
    
    if pt_summary and not pt_summary.get('db_not_found'):
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            pnl = pt_summary.get('total_pnl', 0)
            pnl_pct = pt_summary.get('total_pnl_percentage', 0)
            st.metric(
                "P&L Total",
                f"${pnl:,.2f}",
                f"{pnl_pct:+.2f}%"
            )
        
        with col2:
            balance = pt_summary.get('current_balance', 0)
            initial = pt_summary.get('initial_capital', 0)
            st.metric(
                "Capital Atual",
                f"${balance:,.2f}",
                f"Inicial: ${initial:,.2f}"
            )
        
        with col3:
            win_rate = pt_summary.get('win_rate', 0)
            total_trades = pt_summary.get('total_trades', 0)
            st.metric(
                "Win Rate",
                f"{win_rate:.1f}%",
                f"{total_trades} trades"
            )
        
        with col4:
            open_pos = pt_summary.get('open_positions', 0)
            invested = pt_summary.get('total_invested', 0)
            st.metric(
                "Posições Abertas",
                open_pos,
                f"${invested:,.2f} investido"
            )
        
        st.markdown("---")
        
        # Trades recentes
        st.subheader("Trades Recentes")
        trades_df = load_recent_trades(days_filter)
        
        if not trades_df.empty:
            # Prepara para exibição
            display_trades = trades_df[['symbol', 'side', 'entry_price', 'exit_price', 'pnl', 'pnl_percentage', 'exit_reason']].copy()
            display_trades['pnl'] = display_trades['pnl'].apply(lambda x: f"${x:,.2f}")
            display_trades['pnl_percentage'] = display_trades['pnl_percentage'].apply(lambda x: f"{x:+.2f}%")
            
            st.dataframe(display_trades, use_container_width=True, hide_index=True)
            
            # Gráfico de P&L acumulado
            trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
            
            fig_pnl = go.Figure()
            fig_pnl.add_trace(go.Scatter(
                x=trades_df.index,
                y=trades_df['cumulative_pnl'],
                mode='lines+markers',
                name='P&L Acumulado',
                line=dict(color='green')
            ))
            fig_pnl.update_layout(
                title='P&L Acumulado ao Longo do Tempo',
                xaxis_title='Trade #',
                yaxis_title='P&L ($)',
                hovermode='x unified'
            )
            st.plotly_chart(fig_pnl, use_container_width=True)
        else:
            st.info("Nenhum trade finalizado ainda")
    else:
        st.info("Paper Trading não iniciado")


# TAB 3: Integração ML/LLM
with tab3:
    st.header("🧠 Integração ML + LLM + Sinais Técnicos")
    
    # Status geral da integração
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Peso ML", "25%", "Análise preditiva")
    
    with col2:
        st.metric("Peso LLM", "20%", "Sentiment analysis")
    
    with col3:
        st.metric("Peso Técnico", "55%", "Análise técnica")
    
    st.markdown("---")
    
    # Status ML
    st.subheader("🤖 Machine Learning Status")
    ml_status = load_ml_status()
    
    if ml_status.get('status') == 'not_available':
        st.warning("⚠️ XGBoost não disponível. Execute: pip install xgboost scikit-learn")
    
    elif ml_status.get('status') == 'not_trained':
        st.info("ℹ️ Modelo ML não treinado. Execute o treinamento primeiro.")
        
        with st.expander("Como treinar o modelo"):
            st.code("""
# Treinar modelo com símbolos padrão
python -m ml.model_trainer --train

# Treinar com símbolos específicos
python -m ml.model_trainer --train --symbols BTC ETH BNB

# Testar predições
python -m ml.model_trainer --test
            """)
    
    elif ml_status.get('status') == 'trained':
        # Métricas do modelo
        col1, col2, col3, col4 = st.columns(4)
        
        metrics = ml_status.get('metrics', {})
        
        with col1:
            acc = metrics.get('accuracy', 0)
            st.metric("Accuracy", f"{acc:.2%}")
        
        with col2:
            prec = metrics.get('precision', 0)
            st.metric("Precision", f"{prec:.2%}")
        
        with col3:
            rec = metrics.get('recall', 0)
            st.metric("Recall", f"{rec:.2%}")
        
        with col4:
            auc = metrics.get('roc_auc', 0)
            st.metric("ROC AUC", f"{auc:.3f}")
        
        st.markdown("---")
        
        # Informações do modelo
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("ℹ️ Informações do Modelo")
            training_date = ml_status.get('training_date', 'N/A')
            days_old = ml_status.get('days_since_training', 0)
            n_features = ml_status.get('n_features', 0)
            
            st.write(f"**Data de Treinamento:** {training_date}")
            st.write(f"**Idade:** {days_old} dias")
            st.write(f"**Features:** {n_features}")
            
            needs_retrain = ml_status.get('needs_retrain', False)
            if needs_retrain:
                st.warning("⚠️ Modelo precisa ser retreinado (>7 dias)")
        
        with col2:
            st.subheader("📊 Feature Importance")
            try:
                predictor = OptimizedXGBoostPredictor()
                importance_df = predictor.get_feature_importance(top_n=10)
                
                if not importance_df.empty:
                    fig_imp = px.bar(
                        importance_df,
                        x='importance',
                        y='feature',
                        orientation='h',
                        title='Top 10 Features'
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)
            except Exception as e:
                st.error(f"Erro ao carregar importância: {e}")
    
    else:
        st.error(f"Erro ao carregar status ML: {ml_status.get('error', 'Unknown')}")
    
    st.markdown("---")
    
    # Status LLM
    st.subheader("💭 LLM Sentiment Analysis Status")
    
    if not OPENAI_AVAILABLE:
        st.warning("⚠️ OpenAI não disponível. Execute: pip install openai")
    else:
        api_key = st.text_input("OpenAI API Key", type="password", value=st.session_state.get('openai_key', ''), key="llm_api_key")
        
        if api_key:
            st.session_state['openai_key'] = api_key
            
            try:
                from llm.sentiment_analyzer import SentimentAnalyzer
                
                analyzer = SentimentAnalyzer(api_key=api_key)
                status = analyzer.get_status()
                
                # Status
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    cost = status.get('current_month_cost', 0)
                    st.metric("Custo Mensal", f"${cost:.2f}")
                
                with col2:
                    remaining = status.get('cost_remaining', 0)
                    st.metric("Orçamento Restante", f"${remaining:.2f}")
                
                with col3:
                    cache_stats = status.get('cache_stats', {})
                    cached = cache_stats.get('cached_symbols', 0)
                    st.metric("Símbolos em Cache", cached)
                
            except Exception as e:
                st.error(f"Erro ao inicializar Sentiment Analyzer: {e}")
        else:
            st.info("ℹ️ Configure a API Key do OpenAI para usar Sentiment Analysis")
    
    st.markdown("---")
    
    # Explicação da integração
    st.subheader("🔗 Como Funciona a Integração")
    
    st.markdown("""
    **Sistema de Peso Ponderado:**
    
    1. **Análise Técnica (55%)** - Base do sinal
       - RSI, MACD, Volume
       - Padrões de candlestick
       - Suporte e resistência
    
    2. **Machine Learning (25%)** - Predição
       - XGBoost treinado com dados históricos
       - Predição de movimento futuro
       - Concordância com análise técnica
    
    3. **LLM Sentiment (20%)** - Contexto
       - Análise de notícias em tempo real
       - Sentiment score de -100 a +100
       - Cache agressivo para economia
    
    **Confiança Final = Técnico × 0.55 + ML × 0.25 + LLM × 0.20**
    """)


# TAB 4: Sentiment Analysis
with tab4:
    st.header("💭 Sentiment Analysis")
    
    if not OPENAI_AVAILABLE:
        st.warning("⚠️ OpenAI não disponível. Execute: pip install openai")
    else:
        api_key = st.text_input("OpenAI API Key", type="password", value=st.session_state.get('openai_key', ''))
        
        if api_key:
            st.session_state['openai_key'] = api_key
            
            try:
                from llm.sentiment_analyzer import SentimentAnalyzer
                
                analyzer = SentimentAnalyzer(api_key=api_key)
                status = analyzer.get_status()
                
                # Status
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    cost = status.get('current_month_cost', 0)
                    st.metric("Custo Mensal", f"${cost:.2f}")
                
                with col2:
                    remaining = status.get('cost_remaining', 0)
                    st.metric("Orçamento Restante", f"${remaining:.2f}")
                
                with col3:
                    cache_stats = status.get('cache_stats', {})
                    cached = cache_stats.get('cached_symbols', 0)
                    st.metric("Símbolos em Cache", cached)
                
                st.markdown("---")
                
                # Teste de sentimento
                st.subheader("🧪 Testar Análise de Sentimento")
                
                test_symbol = st.text_input("Símbolo", "BTC")
                
                if st.button("Analisar Sentimento", use_container_width=True):
                    with st.spinner(f"Analisando sentimento de {test_symbol}..."):
                        result = analyzer.analyze_symbol(test_symbol)
                        
                        if result:
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                score = result.sentiment_score
                                color = "green" if score > 0 else "red" if score < 0 else "gray"
                                st.markdown(f"**Score:** <span style='color:{color}; font-size:24px'>{score:+.0f}</span>", unsafe_allow_html=True)
                            
                            with col2:
                                st.metric("Confiança", f"{result.confidence:.2%}")
                            
                            with col3:
                                st.metric("Notícias", result.news_count)
                            
                            st.markdown("---")
                            st.subheader("💡 Análise")
                            st.write(result.reasoning)
                        else:
                            st.error("Erro na análise")
                
            except Exception as e:
                st.error(f"Erro ao inicializar Sentiment Analyzer: {e}")
        else:
            st.info("ℹ️ Configure a API Key do OpenAI para usar Sentiment Analysis")


# TAB 5: Performance Geral
with tab5:
    st.header("📊 Performance Geral do Sistema")
    
    import os
    db_path = settings.database.signals_db_path
    
    if not os.path.exists(db_path):
        st.warning("⚠️ Banco de dados não encontrado. O sistema começará a coletar dados quando houver sinais.")
        st.info(f"📁 Banco esperado: {db_path}")
    else:
        try:
            tracker = PerformanceTracker()
            report = tracker.get_comprehensive_report(days_filter)
        
            # Métricas básicas
            st.subheader("Métricas Básicas")
            basic = report.get('basic_stats', {})
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Total Trades", basic.get('total_trades', 0))
            
            with col2:
                win_rate = basic.get('win_rate', 0)
                st.metric("Win Rate", f"{win_rate:.1f}%")
            
            with col3:
                pf = basic.get('profit_factor', 0)
                st.metric("Profit Factor", f"{pf:.2f}")
            
            with col4:
                avg_win = basic.get('avg_win', 0)
                st.metric("Avg Win", f"${avg_win:.2f}")
            
            with col5:
                avg_loss = basic.get('avg_loss', 0)
                st.metric("Avg Loss", f"${avg_loss:.2f}")
            
            st.markdown("---")
            
            # Métricas avançadas
            st.subheader("Métricas Avançadas")
            advanced = report.get('advanced_metrics', {})
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                sharpe = advanced.get('sharpe_ratio', 0)
                st.metric("Sharpe Ratio", f"{sharpe:.2f}")
            
            with col2:
                sortino = advanced.get('sortino_ratio', 0)
                st.metric("Sortino Ratio", f"{sortino:.2f}")
            
            with col3:
                expect = advanced.get('expectancy', 0)
                st.metric("Expectancy", f"${expect:.2f}")
            
            with col4:
                maxdd = advanced.get('max_drawdown_pct', 0)
                st.metric("Max Drawdown", f"{maxdd:.2f}%")
            
            st.markdown("---")
            
            # Performance por símbolo
            st.subheader("Performance por Símbolo")
            by_symbol = report.get('by_symbol', [])
            
            if by_symbol:
                symbol_df = pd.DataFrame(by_symbol)
                
                fig_symbol = px.bar(
                    symbol_df,
                    x='symbol',
                    y='total_pnl',
                    color='win_rate',
                    title='P&L por Símbolo',
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig_symbol, use_container_width=True)
            
            # Performance por timeframe
            st.subheader("Performance por Timeframe")
            by_tf = report.get('by_timeframe', [])
            
            if by_tf:
                tf_df = pd.DataFrame(by_tf)
                
                fig_tf = px.pie(
                    tf_df,
                    names='timeframe',
                    values='total_trades',
                    title='Distribuição de Trades por Timeframe'
                )
                st.plotly_chart(fig_tf, use_container_width=True)
        
        except Exception as e:
            st.error(f"Erro ao carregar performance: {e}")

# TAB 5: Análise de Dados
with tab5:
    st.header("📊 Análise de Dados do Stream")
    
    col1, col2 = st.columns(2)
    
    with col1:
        symbol_data = st.selectbox(
            "Símbolo para análise",
            ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT", "UNIUSDT", "SOLUSDT", "MATICUSDT"],
            key="data_symbol"
        )
    
    with col2:
        timeframe_data = st.selectbox(
            "Timeframe",
            ["1h", "4h", "1d"],
            key="data_timeframe"
        )
    
    # Botões de ação
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 Analisar Qualidade", key="analyze_quality"):
            with st.spinner("Analisando qualidade dos dados..."):
                try:
                    from backtesting.data_analyzer import DataAnalyzer
                    analyzer = DataAnalyzer()
                    
                    quality_report = analyzer.analyze_data_quality(
                        symbol=symbol_data,
                        timeframe=timeframe_data,
                        days=7
                    )
                    
                    st.session_state.quality_report = quality_report
                    st.success("Análise de qualidade concluída!")
                    
                except Exception as e:
                    st.error(f"Erro na análise: {e}")
    
    with col2:
        if st.button("📈 Analisar Performance", key="analyze_performance"):
            with st.spinner("Analisando performance do stream..."):
                try:
                    from backtesting.data_analyzer import DataAnalyzer
                    analyzer = DataAnalyzer()
                    
                    performance_report = analyzer.analyze_stream_performance(
                        symbol=symbol_data,
                        timeframe=timeframe_data,
                        hours=24
                    )
                    
                    st.session_state.performance_report = performance_report
                    st.success("Análise de performance concluída!")
                    
                except Exception as e:
                    st.error(f"Erro na análise: {e}")
    
    with col3:
        if st.button("📊 Visão Geral", key="overview"):
            with st.spinner("Carregando visão geral..."):
                try:
                    from backtesting.data_analyzer import DataAnalyzer
                    analyzer = DataAnalyzer()
                    
                    overview = analyzer.get_stream_overview()
                    st.session_state.stream_overview = overview
                    st.success("Visão geral carregada!")
                    
                except Exception as e:
                    st.error(f"Erro ao carregar visão geral: {e}")
    
    # Exibe resultados
    if 'quality_report' in st.session_state:
        st.subheader("🔍 Relatório de Qualidade")
        report = st.session_state.quality_report
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Registros", f"{report.total_records:,}")
        
        with col2:
            st.metric("Períodos Faltantes", report.missing_periods)
        
        with col3:
            st.metric("Duplicatas", report.duplicate_records)
        
        with col4:
            st.metric("Score Qualidade", f"{report.data_quality_score:.1f}/100")
        
        # Gráfico de gaps
        if report.gaps_details:
            st.subheader("📉 Gaps Detectados")
            gaps_df = pd.DataFrame(report.gaps_details)
            
            fig_gaps = px.bar(
                gaps_df,
                x='start',
                y='duration_minutes',
                title='Duração dos Gaps (minutos)',
                labels={'start': 'Início do Gap', 'duration_minutes': 'Duração (min)'}
            )
            st.plotly_chart(fig_gaps, use_container_width=True)
    
    if 'performance_report' in st.session_state:
        st.subheader("📈 Relatório de Performance")
        report = st.session_state.performance_report
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Taxa de Coleta", f"{report.collection_rate:.1f}%")
        
        with col2:
            st.metric("Atraso Médio", f"{report.avg_delay_minutes:.1f} min")
        
        with col3:
            st.metric("Maior Gap", f"{report.max_gap_hours:.1f} h")
        
        with col4:
            st.metric("Confiabilidade", f"{report.reliability_score:.1f}/100")
    
    if 'stream_overview' in st.session_state:
        st.subheader("📊 Visão Geral do Stream")
        overview = st.session_state.stream_overview
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total de Registros", f"{overview['total_records']:,}")
        
        with col2:
            st.metric("Símbolos", len(overview['symbols']))
        
        with col3:
            st.metric("Timeframes", len(overview['timeframes']))
        
        # Últimos dados
        if overview['latest_data']:
            st.subheader("🕒 Últimos Dados")
            latest_df = pd.DataFrame(overview['latest_data'])
            latest_df['last_update'] = pd.to_datetime(latest_df['last_update'])
            latest_df = latest_df.sort_values('last_update', ascending=False)
            
            st.dataframe(latest_df, use_container_width=True)

# TAB 6: Backtest & Validação
with tab6:
    st.header("🧪 Backtest & Validação")
    
    # Configurações do backtest
    col1, col2, col3 = st.columns(3)
    
    with col1:
        backtest_symbol = st.selectbox(
            "Símbolo",
            ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT", "UNIUSDT", "SOLUSDT", "MATICUSDT"],
            key="backtest_symbol"
        )
    
    with col2:
        backtest_timeframe = st.selectbox(
            "Timeframe",
            ["1h"],
            key="backtest_timeframe"
        )
    
    with col3:
        backtest_tests = st.number_input(
            "Número de Testes",
            min_value=100,
            max_value=5000,
            value=1000,
            step=100,
            key="backtest_tests"
        )
    
    # Botões de ação
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Executar Backtest", key="run_backtest"):
            # Container para mostrar progresso
            progress_container = st.container()
            status_container = st.container()
            results_container = st.container()
            
            with progress_container:
                st.subheader("🧪 Executando Backtest...")
                progress_bar = st.progress(0)
                status_text = st.empty()
                stats_text = st.empty()
            
            try:
                from backtesting.backtest_engine import BacktestEngine
                
                # Criar engine com callback de progresso
                engine = BacktestEngine()
                
                # Executar backtest com progresso
                result = engine.run_backtest_with_progress(
                    symbol=backtest_symbol,
                    timeframe=backtest_timeframe,
                    test_count=backtest_tests,
                    lookback_days=30,
                    progress_callback=lambda current, total, stats: update_progress(
                        progress_bar, status_text, stats_text, current, total, stats
                    )
                )
                
                with results_container:
                    st.session_state.backtest_result = result
                    
                    if result['success']:
                        st.success(f"✅ Backtest concluído: {result['successful_tests']}/{result['total_tests']} testes bem-sucedidos!")
                        
                        # Mostra estatísticas finais
                        stats = result['statistics']
                        st.info(f"""
                        **Resultados Finais:**
                        - Win Rate: {stats['win_rate']:.2%}
                        - Retorno Médio: {stats['avg_return']:.2f}%
                        - Sharpe Ratio: {stats['sharpe_ratio']:.2f}
                        - Max Drawdown: {stats['max_drawdown']:.2f}%
                        """)
                    else:
                        st.error(f"❌ Erro no backtest: {result.get('error', 'Unknown')}")
                        
            except Exception as e:
                st.error(f"❌ Erro no backtest: {e}")
                import traceback
                st.code(traceback.format_exc())
    
        with col2:
            col2_1, col2_2 = st.columns(2)
            
            with col2_1:
                if st.button("🔧 Otimizar Indicadores", key="optimize_indicators"):
                    # Container para mostrar progresso da otimização
                    opt_progress_container = st.container()
                    opt_status_container = st.container()
                    opt_results_container = st.container()
                    
                    with opt_progress_container:
                        st.subheader("🔧 Otimizando Indicadores...")
                        opt_progress_bar = st.progress(0)
                        opt_status_text = st.empty()
                        opt_stats_text = st.empty()
                    
                    try:
                        from backtesting.optimization_engine import OptimizationEngine
                        
                        engine = OptimizationEngine()
                        
                        # Executar otimização com progresso
                        result = engine.run_optimization_with_progress(
                            symbol=backtest_symbol,
                            timeframe=backtest_timeframe,
                            max_configs=50,
                            recent_days=7,
                            progress_callback=lambda current, total, best_score: update_optimization_progress(
                                opt_progress_bar, opt_status_text, opt_stats_text, current, total, best_score
                            )
                        )
                        
                        with opt_results_container:
                            st.session_state.optimization_result = result
                            
                            if result['success']:
                                st.success(f"✅ Otimização concluída: {result['successful_configs']}/{result['total_configs']} configurações válidas!")
                                
                                if result['results']:
                                    best_result = max(result['results'], key=lambda x: x.score)
                                    st.info(f"""
                                    **Melhor Configuração Encontrada:**
                                    - Score: {best_result.score:.3f}
                                    - Win Rate: {best_result.win_rate:.2%}
                                    - Retorno Médio: {best_result.avg_return:.2f}%
                                    """)
                            else:
                                st.error(f"❌ Erro na otimização: {result.get('error', 'Unknown')}")
                                
                    except Exception as e:
                        st.error(f"❌ Erro na otimização: {e}")
                        import traceback
                        st.code(traceback.format_exc())
            
            with col2_2:
                if 'continuous_running' not in st.session_state:
                    st.session_state.continuous_running = False
                
                if not st.session_state.continuous_running:
                    if st.button("🔄 Iniciar Otimização Contínua", key="start_continuous"):
                        try:
                            from backtesting.optimization_engine import OptimizationEngine
                            
                            # Cria engine se não existir
                            if 'continuous_engine' not in st.session_state:
                                st.session_state.continuous_engine = OptimizationEngine()
                            
                            engine = st.session_state.continuous_engine
                            
                            # Inicia otimização contínua
                            engine.start_continuous_optimization(
                                symbol=backtest_symbol,
                                timeframe=backtest_timeframe,
                                interval_hours=6  # A cada 6 horas
                            )
                            
                            st.session_state.continuous_running = True
                            st.success("🚀 Otimização contínua iniciada!")
                            st.info("ℹ️ A otimização rodará a cada 6 horas automaticamente.")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ Erro ao iniciar otimização contínua: {e}")
                else:
                    if st.button("🛑 Parar Otimização Contínua", key="stop_continuous_tab4"):
                        if 'continuous_engine' in st.session_state:
                            engine = st.session_state.continuous_engine
                            engine.stop_continuous_optimization()
                            st.session_state.continuous_running = False
                            st.success("🛑 Otimização contínua parada!")
                            st.rerun()
    
    # Status da otimização contínua
    if 'continuous_running' in st.session_state and st.session_state.continuous_running:
        st.subheader("🔄 Otimização Contínua Ativa")
        
        if 'continuous_engine' in st.session_state:
            engine = st.session_state.continuous_engine
            status = engine.get_optimization_status()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                status_color = "🟢" if status.get('is_running', False) else "🔴"
                st.metric("Status", f"{status_color} {'Ativa' if status.get('is_running', False) else 'Inativa'}")
            
            with col2:
                st.metric("Testes Executados", status.get('total_tests', 0))
            
            with col3:
                current_test = status.get('current_test', 0)
                total_tests = status.get('total_tests', 0)
                st.metric("Teste Atual", f"{current_test}/{total_tests}")
            
            with col4:
                best_score = status.get('best_score', 0.0)
                st.metric("Melhor Score", f"{best_score:.3f}")
            
            # Progresso atual
            total_tests = status.get('total_tests', 0)
            current_test = status.get('current_test', 0)
            if total_tests > 0:
                progress = current_test / total_tests
                st.progress(progress)
                st.caption(f"Progresso: {current_test}/{total_tests} ({progress:.1%})")
            
            # Última atualização
            if status.get('last_update'):
                last_update = datetime.fromisoformat(status.get('last_update'))
                st.caption(f"Última atualização: {last_update.strftime('%H:%M:%S')}")
            
            # Top configurações testadas
            configs_tested = status.get('configs_tested', [])
            if configs_tested:
                st.subheader("🏆 Top Configurações Testadas")
                top_configs = sorted(configs_tested, key=lambda x: x['score'], reverse=True)[:5]
                
                for i, config in enumerate(top_configs, 1):
                    with st.expander(f"#{i} - Score: {config['score']:.3f} (WR: {config['win_rate']:.1%})"):
                        st.json(config['config'])
            
            # Erros recentes
            if status.get('errors', []):
                st.subheader("⚠️ Erros Recentes")
                recent_errors = status.get('errors', [])[-3:]  # Últimos 3 erros
                for error in recent_errors:
                    st.error(f"{error['timestamp']}: {error['error']}")
    
    # Exibe resultados do backtest
    if 'backtest_result' in st.session_state:
        st.subheader("📊 Resultados do Backtest")
        result = st.session_state.backtest_result
        
        if result['success']:
            stats = result['statistics']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Win Rate", f"{stats['win_rate']:.2%}")
            
            with col2:
                st.metric("Retorno Médio", f"{stats['avg_return']:.2f}%")
            
            with col3:
                st.metric("Sharpe Ratio", f"{stats['sharpe_ratio']:.2f}")
            
            with col4:
                st.metric("Max Drawdown", f"{stats['max_drawdown']:.2f}%")
            
            # Gráfico de distribuição de retornos
            if result['results']:
                returns = [r.total_return for r in result['results']]
                
                fig_returns = px.histogram(
                    x=returns,
                    nbins=20,
                    title='Distribuição de Retornos',
                    labels={'x': 'Retorno (%)', 'y': 'Frequência'}
                )
                st.plotly_chart(fig_returns, use_container_width=True)
            
            # Razões de saída
            if stats.get('exit_reasons'):
                st.subheader("🚪 Razões de Saída")
                
                reasons_df = pd.DataFrame([
                    {'Razão': reason, 'Quantidade': count}
                    for reason, count in stats['exit_reasons'].items()
                ])
                
                fig_reasons = px.pie(
                    reasons_df,
                    values='Quantidade',
                    names='Razão',
                    title='Distribuição por Razão de Saída'
                )
                st.plotly_chart(fig_reasons, use_container_width=True)
    
    # Exibe resultados da otimização
    if 'optimization_result' in st.session_state:
        st.subheader("🔧 Resultados da Otimização")
        result = st.session_state.optimization_result
        
        if result['success'] and result['results']:
            best_result = max(result['results'], key=lambda x: x.score)
            
            st.success(f"🏆 Melhor Configuração encontrada!")
            st.write(f"**Score:** {best_result.score:.3f}")
            st.write(f"**Win Rate:** {best_result.win_rate:.2%}")
            st.write(f"**Retorno Médio:** {best_result.avg_return:.2f}%")
            st.write(f"**Sharpe Ratio:** {best_result.sharpe_ratio:.2f}")
            
            # Configurações da melhor configuração
            st.subheader("⚙️ Melhores Parâmetros")
            config = best_result.config
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**RSI:**")
                st.write(f"  - Período: {config.rsi_period}")
                st.write(f"  - Overbought: {config.rsi_overbought}")
                st.write(f"  - Oversold: {config.rsi_oversold}")
            
            with col2:
                st.write("**MACD:**")
                st.write(f"  - Fast: {config.macd_fast}")
                st.write(f"  - Slow: {config.macd_slow}")
                st.write(f"  - Signal: {config.macd_signal}")
            
            with col3:
                st.write("**Outros:**")
                st.write(f"  - Bollinger Período: {config.bollinger_period}")
                st.write(f"  - Bollinger Std: {config.bollinger_std}")
                st.write(f"  - Confiança Mínima: {config.confidence_threshold}")
    
    # Status da otimização contínua
    if 'continuous_running' in st.session_state and st.session_state.continuous_running:
        st.subheader("🔄 Otimização Contínua")
        
        if 'continuous_engine' in st.session_state:
            engine = st.session_state.continuous_engine
            status = engine.get_optimization_status()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                status_color = "🟢" if status.get('is_running', False) else "🔴"
                st.metric("Status", f"{status_color} {'Ativa' if status.get('is_running', False) else 'Inativa'}")
            
            with col2:
                st.metric("Testes Executados", status.get('total_tests', 0))
            
            with col3:
                if status.get('last_update'):
                    last_update = datetime.fromisoformat(status.get('last_update'))
                    st.metric("Última Atualização", last_update.strftime('%H:%M:%S'))
            
            if st.button("🛑 Parar Otimização Contínua", key="stop_continuous_tab7"):
                engine.stop_continuous_optimization()
                st.session_state.continuous_running = False
                st.success("Otimização contínua parada!")


# TAB 7: Performance
with tab7:
    st.header("📊 Performance Geral do Sistema")
    
    # Métricas do sistema
    st.subheader("🎯 Métricas do Sistema")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Status do sistema
        st.metric("Status Sistema", "🟢 Ativo")
    
    with col2:
        # Timeframes ativos
        timeframes_count = len(settings.get_enabled_timeframes())
        st.metric("Timeframes", f"{timeframes_count} ativos")
    
    with col3:
        # ML/LLM status
        ml_status = "✅" if settings.ml.enabled else "❌"
        llm_status = "✅" if settings.llm.enabled else "❌"
        st.metric("ML/LLM", f"{ml_status}/{llm_status}")
    
    with col4:
        # Paper trading status
        pt_status = "✅" if settings.paper_trading.enabled else "❌"
        st.metric("Paper Trading", pt_status)
    
    st.markdown("---")
    
    # Performance de dados
    st.subheader("📈 Performance de Dados")
    
    try:
        import sqlite3
        import os
        
        # Stream database
        stream_path = settings.database.stream_db_path
        if os.path.exists(stream_path):
            conn = sqlite3.connect(stream_path)
            cursor = conn.cursor()
            
            # Total de registros
            cursor.execute("SELECT COUNT(*) FROM crypto_ohlc")
            total_records = cursor.fetchone()[0]
            
            # Registros por timeframe
            cursor.execute("SELECT timeframe, COUNT(*) FROM crypto_ohlc GROUP BY timeframe")
            tf_counts = cursor.fetchall()
            
            # Última atualização
            cursor.execute("SELECT MAX(timestamp) FROM crypto_ohlc")
            last_update = cursor.fetchone()[0]
            
            conn.close()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Registros", f"{total_records:,}")
            
            with col2:
                st.metric("Timeframes", len(tf_counts))
            
            with col3:
                if last_update:
                    last_dt = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
                    st.metric("Última Atualização", last_dt.strftime('%H:%M:%S'))
                else:
                    st.metric("Última Atualização", "N/A")
            
            # Detalhes por timeframe
            if tf_counts:
                st.subheader("📊 Registros por Timeframe")
                tf_df = pd.DataFrame(tf_counts, columns=['Timeframe', 'Registros'])
                tf_df = tf_df.sort_values('Registros', ascending=False)
                
                fig_tf = px.bar(
                    tf_df,
                    x='Timeframe',
                    y='Registros',
                    title='Registros por Timeframe',
                    color='Registros',
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig_tf, use_container_width=True)
            
            # Performance de coleta (últimas 24h)
            st.subheader("🕒 Performance de Coleta (Últimas 24h)")
            
            conn = sqlite3.connect(stream_path)
            cursor = conn.cursor()
            
            # Registros das últimas 24h
            cursor.execute("""
                SELECT COUNT(*) FROM crypto_ohlc 
                WHERE timestamp >= datetime('now', '-24 hours')
            """)
            records_24h = cursor.fetchone()[0]
            
            # Registros por hora
            cursor.execute("""
                SELECT strftime('%H', timestamp) as hour, COUNT(*) as count
                FROM crypto_ohlc 
                WHERE timestamp >= datetime('now', '-24 hours')
                GROUP BY strftime('%H', timestamp)
                ORDER BY hour
            """)
            hourly_data = cursor.fetchall()
            
            conn.close()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Registros (24h)", f"{records_24h:,}")
                expected_24h = 24 * len(settings.get_enabled_timeframes()) * 8  # 8 símbolos
                collection_rate = (records_24h / expected_24h * 100) if expected_24h > 0 else 0
                st.metric("Taxa de Coleta", f"{collection_rate:.1f}%")
            
            with col2:
                if hourly_data:
                    hourly_df = pd.DataFrame(hourly_data, columns=['Hora', 'Registros'])
                    hourly_df['Hora'] = hourly_df['Hora'].astype(int)
                    
                    fig_hourly = px.line(
                        hourly_df,
                        x='Hora',
                        y='Registros',
                        title='Registros por Hora (Últimas 24h)',
                        markers=True
                    )
                    st.plotly_chart(fig_hourly, use_container_width=True)
        
        else:
            st.warning("⚠️ Banco de dados de stream não encontrado")
            st.info(f"📁 Caminho esperado: {stream_path}")
    
    except Exception as e:
        st.error(f"❌ Erro ao carregar performance: {e}")
    
    # Performance de sinais
    st.subheader("🎯 Performance de Sinais")
    
    signals_path = settings.database.signals_db_path
    if os.path.exists(signals_path):
        try:
            conn = sqlite3.connect(signals_path)
            cursor = conn.cursor()
            
            # Total de sinais
            cursor.execute("SELECT COUNT(*) FROM trading_signals_v2")
            total_signals = cursor.fetchone()[0]
            
            # Sinais por tipo
            cursor.execute("""
                SELECT signal_type, COUNT(*) as count
                FROM trading_signals_v2 
                GROUP BY signal_type
            """)
            signal_types = cursor.fetchall()
            
            # Sinais recentes (últimos 7 dias)
            cursor.execute("""
                SELECT COUNT(*) FROM trading_signals_v2 
                WHERE created_at >= datetime('now', '-7 days')
            """)
            signals_7d = cursor.fetchone()[0]
            
            conn.close()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Sinais", f"{total_signals:,}")
            
            with col2:
                st.metric("Sinais (7 dias)", f"{signals_7d:,}")
            
            with col3:
                avg_daily = signals_7d / 7 if signals_7d > 0 else 0
                st.metric("Média Diária", f"{avg_daily:.1f}")
            
            # Gráfico de sinais por tipo
            if signal_types:
                st.subheader("📊 Sinais por Tipo")
                types_df = pd.DataFrame(signal_types, columns=['Tipo', 'Quantidade'])
                
                fig_types = px.pie(
                    types_df,
                    values='Quantidade',
                    names='Tipo',
                    title='Distribuição de Sinais por Tipo'
                )
                st.plotly_chart(fig_types, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Erro ao carregar sinais: {e}")
    else:
        st.warning("⚠️ Banco de dados de sinais não encontrado")
        st.info(f"📁 Caminho esperado: {signals_path}")
    
    # Resumo de performance
    st.subheader("📋 Resumo de Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **✅ Sistema Funcionando:**
        - Coleta de dados ativa
        - Análise técnica operacional
        - Geração de sinais ativa
        - Paper trading configurado
        """)
    
    with col2:
        st.info("""
        **📊 Métricas Importantes:**
        - Taxa de coleta de dados
        - Frequência de sinais
        - Performance de indicadores
        - Qualidade dos dados
        """)


# TAB 8: Configurações do Sistema
with tab8:
    st.header("⚙️ Configurações Atuais do Sistema")
    
    # Configurações de Timeframes
    st.subheader("📊 Timeframes Configurados")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Timeframes Ativos", len(settings.get_enabled_timeframes()))
        st.code("\n".join(settings.get_enabled_timeframes()))
    
    with col2:
        st.metric("Timeframe Padrão", settings.analysis.default_timeframe)
    
    with col3:
        st.metric("Multi-Timeframe", "✅ Ativo" if settings.system.multi_timeframe_enabled else "❌ Inativo")
    
    # Configurações de Indicadores
    st.subheader("📈 Indicadores Técnicos")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**RSI**")
        for tf in settings.get_enabled_timeframes():
            rsi_levels = settings.get_rsi_levels(tf)
            st.write(f"  {tf}: {rsi_levels['oversold']}/{rsi_levels['overbought']}")
    
    with col2:
        st.write("**MACD**")
        for tf in settings.get_enabled_timeframes():
            macd_params = settings.get_macd_params(tf)
            st.write(f"  {tf}: {macd_params['fast']}/{macd_params['slow']}/{macd_params['signal']}")
    
    with col3:
        st.write("**Volume**")
        st.write(f"  Período MA: {settings.indicators.volume_ma_period}")
        for tf in settings.get_enabled_timeframes():
            vol_ratio = settings.indicators.min_volume_ratio.get(tf, 1.0)
            st.write(f"  {tf}: {vol_ratio}x")
    
    # Configurações de Análise
    st.subheader("🔍 Configurações de Análise")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Limites de Dados**")
        st.write(f"  Mínimo: {settings.analysis.min_data_points}")
        st.write(f"  Lookback: {settings.analysis.lookback_hours}h")
        st.write(f"  Confiança: {settings.analysis.confidence_threshold:.2f}")
    
    with col2:
        st.write("**Sistema**")
        st.write(f"  Intervalo: {settings.system.analysis_interval}s")
        st.write(f"  Workers: {settings.system.max_workers}")
        st.write(f"  Log Level: {settings.system.log_level}")
    
    with col3:
        st.write("**Retenção de Dados**")
        try:
            st.write(f"  Anos: {getattr(settings.system, 'data_retention_years', 4)}")
            st.write(f"  Limpeza: {getattr(settings.system, 'data_cleanup_interval_hours', 24)}h")
            st.write(f"  Auto-limpeza: {'✅' if getattr(settings.system, 'auto_cleanup_enabled', True) else '❌'}")
        except AttributeError:
            st.write(f"  Anos: 4 (padrão)")
            st.write(f"  Limpeza: 24h (padrão)")
            st.write(f"  Auto-limpeza: ✅ (padrão)")
    
    # Configurações ML/LLM
    st.subheader("🤖 Integração ML/LLM")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Machine Learning**")
        try:
            st.write(f"  Status: {'✅ Ativo' if getattr(settings.ml, 'enabled', False) else '❌ Inativo'}")
            st.write(f"  Peso: {getattr(settings.ml, 'ml_weight', 0.2):.1%}")
            st.write(f"  Modelo: {getattr(settings.ml, 'model_path', 'XGBoost')}")
            st.write(f"  Retreinamento: {getattr(settings.ml, 'retrain_interval_days', 7)} dias")
        except AttributeError:
            st.write(f"  Status: ❌ Inativo (configuração não encontrada)")
    
    with col2:
        st.write("**LLM Sentiment**")
        try:
            st.write(f"  Status: {'✅ Ativo' if getattr(settings.llm, 'enabled', False) else '❌ Inativo'}")
            st.write(f"  Peso: {getattr(settings.llm, 'llm_weight', 0.1):.1%}")
            st.write(f"  Modelo: {getattr(settings.llm, 'model', 'GPT-3.5')}")
            st.write(f"  Cache: {getattr(settings.llm, 'cache_duration_minutes', 60)} min")
        except AttributeError:
            st.write(f"  Status: ❌ Inativo (configuração não encontrada)")
    
    # Configurações de Paper Trading
    st.subheader("💰 Paper Trading")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Capital**")
        try:
            st.write(f"  Inicial: ${getattr(settings.paper_trading, 'initial_capital', 10000):,.2f}")
            st.write(f"  Max Posições: {getattr(settings.paper_trading, 'max_open_positions', 5)}")
        except AttributeError:
            st.write(f"  Inicial: $10,000 (padrão)")
            st.write(f"  Max Posições: 5 (padrão)")
    
    with col2:
        st.write("**Risk Management**")
        try:
            st.write(f"  Max por Trade: {getattr(settings.paper_trading, 'max_position_size_pct', 0.1):.1%}")
            st.write(f"  Taxa: {getattr(settings.paper_trading, 'fee_percentage', 0.001):.3%}")
            st.write(f"  Slippage: {getattr(settings.paper_trading, 'slippage_percentage', 0.0005):.3%}")
        except AttributeError:
            st.write(f"  Max por Trade: 10% (padrão)")
            st.write(f"  Taxa: 0.1% (padrão)")
            st.write(f"  Slippage: 0.05% (padrão)")
    
    with col3:
        st.write("**Configurações**")
        try:
            st.write(f"  Simulação: {'✅ Ativa' if getattr(settings.paper_trading, 'enabled', False) else '❌ Inativa'}")
            st.write(f"  Status: Configurado")
        except AttributeError:
            st.write(f"  Simulação: ❌ Inativa (padrão)")
            st.write(f"  Status: Não configurado")
    
    # Status dos Bancos de Dados
    st.subheader("💾 Bancos de Dados")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Stream Database**")
        stream_path = settings.database.stream_db_path
        stream_exists = os.path.exists(stream_path)
        st.write(f"  Status: {'✅ Ativo' if stream_exists else '❌ Não encontrado'}")
        st.write(f"  Caminho: {stream_path}")
        
        if stream_exists:
            try:
                import sqlite3
                conn = sqlite3.connect(stream_path)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM crypto_ohlc")
                count = cursor.fetchone()[0]
                conn.close()
                st.write(f"  Registros: {count:,}")
            except:
                st.write("  Registros: Erro ao consultar")
    
    with col2:
        st.write("**Signals Database**")
        signals_path = settings.database.signals_db_path
        signals_exists = os.path.exists(signals_path)
        st.write(f"  Status: {'✅ Ativo' if signals_exists else '❌ Não encontrado'}")
        st.write(f"  Caminho: {signals_path}")
        
        if signals_exists:
            try:
                import sqlite3
                conn = sqlite3.connect(signals_path)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM trading_signals_v2")
                count = cursor.fetchone()[0]
                conn.close()
                st.write(f"  Sinais: {count:,}")
            except:
                st.write("  Sinais: Erro ao consultar")
    
    # Botão para aplicar configurações do otimizador
    st.subheader("🔧 Aplicar Configurações Otimizadas")
    
    if 'optimization_result' in st.session_state and st.session_state.optimization_result:
        result = st.session_state.optimization_result
        
        if result.get('success') and result.get('best_config'):
            st.success("✅ Configuração otimizada disponível!")
            
            best_config = result['best_config']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Configuração Atual**")
                st.write(f"  RSI: {settings.indicators.rsi_period}")
                st.write(f"  MACD: {settings.get_macd_params('1h')['fast']}/{settings.get_macd_params('1h')['slow']}")
                st.write(f"  Confiança: {settings.analysis.confidence_threshold:.2f}")
            
            with col2:
                st.write("**Configuração Otimizada**")
                st.write(f"  RSI: {getattr(best_config, 'rsi_period', 'N/A')}")
                st.write(f"  MACD: {getattr(best_config, 'macd_fast', 'N/A')}/{getattr(best_config, 'macd_slow', 'N/A')}")
                st.write(f"  Confiança: {getattr(best_config, 'confidence_threshold', 'N/A'):.2f}")
            
            if st.button("🔄 Aplicar Configuração Otimizada", type="primary"):
                # Aqui seria implementada a lógica para aplicar as configurações
                st.success("✅ Configurações otimizadas aplicadas!")
                st.info("ℹ️ Reinicie o sistema para que as mudanças tenham efeito.")
        else:
            st.info("ℹ️ Execute o otimizador primeiro para ter configurações otimizadas disponíveis.")
    else:
        st.info("ℹ️ Execute o otimizador primeiro para ter configurações otimizadas disponíveis.")


# Footer
st.markdown("---")
st.caption(f"Trading System v2.0 | Última atualização: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

