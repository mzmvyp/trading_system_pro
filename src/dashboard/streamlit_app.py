"""
Dashboard Streamlit para Monitoramento de Paper Trading
"""

import asyncio
import glob
import hashlib
import hmac
import json
import os
import time
from datetime import datetime
from urllib.parse import urlencode

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv

from src.trading.paper_trading import real_paper_trading  # Usar instância global

# Carregar variáveis de ambiente
load_dotenv()

# Configurar página
st.set_page_config(
    page_title="Paper Trading Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("📊 Dashboard de Trading")
st.markdown("---")

# ============================================================
# FUNÇÕES PARA BINANCE FUTURES API
# ============================================================

def get_binance_config():
    """Retorna configuração da Binance (testnet ou produção)"""
    use_testnet = os.getenv("BINANCE_TESTNET", "false").lower() == "true"
    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_SECRET_KEY", "")

    if use_testnet:
        base_url = "https://testnet.binancefuture.com"
        mode = "TESTNET"
    else:
        base_url = "https://fapi.binance.com"
        mode = "PRODUÇÃO"

    return {
        "base_url": base_url,
        "api_key": api_key,
        "api_secret": api_secret,
        "mode": mode,
        "testnet": use_testnet
    }

def binance_signature(params: dict, secret: str) -> str:
    """Gera assinatura HMAC SHA256"""
    query_string = urlencode(params)
    signature = hmac.new(
        secret.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    return signature

@st.cache_data(ttl=10)
def get_binance_positions():
    """Obtém posições abertas na Binance Futures"""
    config = get_binance_config()

    if not config["api_key"] or not config["api_secret"]:
        return {"error": "API keys não configuradas", "positions": []}

    try:
        params = {"timestamp": int(time.time() * 1000)}
        params["signature"] = binance_signature(params, config["api_secret"])

        headers = {"X-MBX-APIKEY": config["api_key"]}
        url = f"{config['base_url']}/fapi/v2/positionRisk"

        response = requests.get(url, params=params, headers=headers, timeout=10)

        if response.status_code == 200:
            all_positions = response.json()
            # Filtrar apenas posições com quantidade != 0
            open_positions = [p for p in all_positions if float(p.get("positionAmt", 0)) != 0]
            return {"positions": open_positions, "mode": config["mode"]}
        else:
            return {"error": f"Erro {response.status_code}: {response.text}", "positions": []}
    except Exception as e:
        return {"error": str(e), "positions": []}

@st.cache_data(ttl=10)
def get_binance_open_orders():
    """Obtém ordens abertas na Binance Futures"""
    config = get_binance_config()

    if not config["api_key"] or not config["api_secret"]:
        return {"error": "API keys não configuradas", "orders": []}

    try:
        params = {"timestamp": int(time.time() * 1000)}
        params["signature"] = binance_signature(params, config["api_secret"])

        headers = {"X-MBX-APIKEY": config["api_key"]}
        url = f"{config['base_url']}/fapi/v1/openOrders"

        response = requests.get(url, params=params, headers=headers, timeout=10)

        if response.status_code == 200:
            return {"orders": response.json(), "mode": config["mode"]}
        else:
            return {"error": f"Erro {response.status_code}: {response.text}", "orders": []}
    except Exception as e:
        return {"error": str(e), "orders": []}

def close_binance_position(symbol: str) -> dict:
    """Fecha posição aberta na Binance Futures com ordem de mercado"""
    config = get_binance_config()

    if not config["api_key"] or not config["api_secret"]:
        return {"success": False, "error": "API keys não configuradas"}

    try:
        # 1. Obter posição atual
        positions_data = get_binance_positions()
        if "error" in positions_data:
            return {"success": False, "error": positions_data["error"]}

        # Encontrar posição do símbolo
        position = None
        for pos in positions_data.get("positions", []):
            if pos.get("symbol") == symbol:
                position = pos
                break

        if not position:
            return {"success": False, "error": f"Nenhuma posição aberta para {symbol}"}

        position_amt = float(position.get("positionAmt", 0))
        if position_amt == 0:
            return {"success": False, "error": f"Posição de {symbol} já está fechada"}

        # 2. Determinar lado oposto
        close_side = "SELL" if position_amt > 0 else "BUY"
        quantity = abs(position_amt)

        # 3. Cancelar ordens abertas do símbolo primeiro
        cancel_result = cancel_binance_orders(symbol)

        # 4. Enviar ordem de mercado para fechar
        params = {
            "symbol": symbol,
            "side": close_side,
            "type": "MARKET",
            "quantity": quantity,
            "reduceOnly": "true",
            "timestamp": int(time.time() * 1000)
        }
        params["signature"] = binance_signature(params, config["api_secret"])

        headers = {"X-MBX-APIKEY": config["api_key"]}
        url = f"{config['base_url']}/fapi/v1/order"

        response = requests.post(url, params=params, headers=headers, timeout=10)

        if response.status_code == 200:
            order = response.json()
            return {
                "success": True,
                "message": f"Posição {symbol} fechada com sucesso",
                "order_id": order.get("orderId"),
                "avg_price": order.get("avgPrice"),
                "canceled_orders": cancel_result.get("canceled", 0)
            }
        else:
            return {"success": False, "error": f"Erro {response.status_code}: {response.text}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def cancel_binance_orders(symbol: str) -> dict:
    """Cancela todas as ordens abertas de um símbolo"""
    config = get_binance_config()

    if not config["api_key"] or not config["api_secret"]:
        return {"success": False, "error": "API keys não configuradas"}

    try:
        params = {
            "symbol": symbol,
            "timestamp": int(time.time() * 1000)
        }
        params["signature"] = binance_signature(params, config["api_secret"])

        headers = {"X-MBX-APIKEY": config["api_key"]}
        url = f"{config['base_url']}/fapi/v1/allOpenOrders"

        response = requests.delete(url, params=params, headers=headers, timeout=10)

        if response.status_code == 200:
            return {"success": True, "canceled": response.json().get("code", 0) == 200}
        else:
            # Código 200 com erro significa que não tinha ordens
            return {"success": True, "canceled": 0}
    except Exception as e:
        return {"success": False, "error": str(e)}

@st.cache_data(ttl=10)
def get_binance_balance():
    """Obtém saldo da conta Binance Futures"""
    config = get_binance_config()

    if not config["api_key"] or not config["api_secret"]:
        return {"error": "API keys não configuradas"}

    try:
        params = {"timestamp": int(time.time() * 1000)}
        params["signature"] = binance_signature(params, config["api_secret"])

        headers = {"X-MBX-APIKEY": config["api_key"]}
        url = f"{config['base_url']}/fapi/v2/balance"

        response = requests.get(url, params=params, headers=headers, timeout=10)

        if response.status_code == 200:
            balances = response.json()
            # Encontrar USDT
            usdt_balance = next((b for b in balances if b.get("asset") == "USDT"), None)
            if usdt_balance:
                return {
                    "balance": float(usdt_balance.get("balance", 0)),
                    "available": float(usdt_balance.get("availableBalance", 0)),
                    "unrealized_pnl": float(usdt_balance.get("crossUnPnl", 0)),
                    "mode": config["mode"]
                }
            return {"error": "USDT não encontrado", "mode": config["mode"]}
        else:
            return {"error": f"Erro {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": str(e)}

# Função helper para obter P&L em % (suporta campo pnl antigo)
def get_pnl_percent(trade):
    """Obtém P&L em % do trade, com fallback para campo pnl antigo"""
    pnl_percent = trade.get("pnl_percent")
    if pnl_percent is None and trade.get("pnl") is not None:
        # Converter pnl absoluto para % aproximado
        entry = trade.get("entry_price", 1)
        size = trade.get("position_size", 1)
        if entry > 0 and size > 0:
            pnl_percent = (trade["pnl"] / (entry * size)) * 100
        else:
            pnl_percent = 0
    return pnl_percent if pnl_percent is not None else 0

@st.cache_data(ttl=5)
def get_current_price(symbol):
    """Obtém o preço atual do símbolo via Binance API"""
    try:
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return float(data['price'])
        else:
            return None
    except Exception as e:
        st.error(f"Erro ao obter preço de {symbol}: {e}")
        return None

# Função para carregar dados do portfólio (CORRIGIDO: cache reduzido para 2s)
@st.cache_data(ttl=2)
def load_portfolio_data():
    """Carrega dados do portfólio (paper: state.json; real: Binance API)"""
    try:
        if os.path.exists("portfolio/state.json"):
            with open("portfolio/state.json", "r", encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
    # Modo REAL (testnet/produção): usar dados da Binance para não mostrar dashboard vazio
    if os.getenv("TRADING_MODE", "").lower() == "real":
        return load_binance_as_portfolio()
    return None


@st.cache_data(ttl=10)
def load_binance_as_portfolio():
    """Quando TRADING_MODE=real, monta um 'portfólio' a partir das posições Binance para o dashboard."""
    data = get_binance_positions()
    if "error" in data or not data.get("positions"):
        return None
    positions = {}
    for p in data["positions"]:
        amt = float(p.get("positionAmt", 0))
        if amt == 0:
            continue
        symbol = p.get("symbol", "")
        key = f"BINANCE_{symbol}"
        positions[key] = {
            "symbol": symbol,
            "entry_price": float(p.get("entryPrice", 0)),
            "signal": "BUY" if amt > 0 else "SELL",
            "position_size": abs(amt),
            "unrealized_pnl": float(p.get("unRealizedProfit", 0)),
            "mark_price": float(p.get("markPrice", 0)),
            "status": "OPEN",
            "source": "BINANCE",
        }
    return {"positions": positions, "trade_history": [], "_binance_mode": True}

# Função para carregar histórico de trades
@st.cache_data(ttl=5)
def load_trade_history():
    """Carrega histórico de trades - Paper + Real (sinais e execution records)"""
    trades = []

    # 1. Paper trading: portfolio/state.json
    try:
        state = load_portfolio_data()
        if state and "trade_history" in state:
            trades.extend(state["trade_history"])
    except Exception:
        pass

    # 2. Real trading: execution records em real_orders/
    try:
        exec_files = glob.glob("real_orders/execution_*.json")
        for filepath in exec_files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    record = json.load(f)
                    # Converter execution record para formato de trade
                    trades.append({
                        "trade_id": record.get("main_order_id", os.path.basename(filepath)),
                        "symbol": record.get("symbol", ""),
                        "signal": record.get("signal", ""),
                        "source": record.get("source", "UNKNOWN"),
                        "entry_price": record.get("entry_price", 0),
                        "stop_loss": record.get("stop_loss", 0),
                        "take_profit_1": record.get("take_profit_1", 0),
                        "take_profit_2": record.get("take_profit_2", 0),
                        "position_size": record.get("position_size", 0),
                        "status": record.get("status", "OPEN"),
                        "timestamp": record.get("timestamp", ""),
                        "leverage": record.get("leverage", 20),
                        "_source_file": "real_orders",
                    })
            except (json.JSONDecodeError, IOError):
                continue
    except Exception:
        pass

    # 3. Sinais salvos em signals/ (para trades que não têm execution record)
    try:
        signal_files = glob.glob("signals/agno_*_*.json")
        signal_files = [f for f in signal_files if "_last_analysis" not in f]
        existing_symbols_ts = {(t.get("symbol", ""), t.get("timestamp", "")[:16]) for t in trades}

        for filepath in signal_files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    sig = json.load(f)
                    sig_type = sig.get("signal", "")
                    if sig_type not in ("BUY", "SELL"):
                        continue
                    # Evitar duplicatas
                    key = (sig.get("symbol", ""), sig.get("timestamp", "")[:16])
                    if key in existing_symbols_ts:
                        continue
                    trades.append({
                        "trade_id": os.path.basename(filepath).replace(".json", ""),
                        "symbol": sig.get("symbol", ""),
                        "signal": sig_type,
                        "source": sig.get("source", "UNKNOWN"),
                        "entry_price": sig.get("entry_price", 0),
                        "stop_loss": sig.get("stop_loss", 0),
                        "take_profit_1": sig.get("take_profit_1", 0),
                        "take_profit_2": sig.get("take_profit_2", 0),
                        "confidence": sig.get("confidence", 0),
                        "status": "SIGNAL",
                        "timestamp": sig.get("timestamp", ""),
                        "_source_file": "signals",
                    })
            except (json.JSONDecodeError, IOError):
                continue
    except Exception:
        pass

    # Ordenar por timestamp (mais recente primeiro)
    trades.sort(key=lambda t: t.get("timestamp", ""), reverse=True)
    return trades

@st.cache_data(ttl=5)
def load_last_signals():
    """Carrega os últimos sinais gerados para cada par/fonte"""
    signals_by_pair = {}

    # Top 10 pares
    pairs = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT",
             "XRPUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT"]

    for pair in pairs:
        signals_by_pair[pair] = {
            "DEEPSEEK": None,
            "AGNO": None
        }

    # Buscar arquivos de sinais
    signal_files = glob.glob("signals/agno_*_*.json")

    # Filtrar apenas arquivos de sinais (não os _last_analysis)
    signal_files = [f for f in signal_files if "_last_analysis" not in f]

    # Ordenar por data (mais recente primeiro)
    signal_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    for filepath in signal_files:
        try:
            with open(filepath, "r", encoding='utf-8') as f:
                signal = json.load(f)

            symbol = signal.get("symbol", "")
            source = signal.get("source", "UNKNOWN")

            # Apenas processar se for um dos 10 pares
            if symbol in pairs:
                # Se ainda não temos sinal para este par/fonte, usar este
                if source in ["DEEPSEEK", "AGNO"]:
                    if signals_by_pair[symbol][source] is None:
                        signals_by_pair[symbol][source] = signal

        except Exception:
            continue

    return signals_by_pair


@st.cache_data(ttl=5)
def load_last_analysis_timestamps():
    """Carrega timestamps da última análise de cada par"""
    analysis_times = {}

    pairs = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT",
             "XRPUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT"]

    for pair in pairs:
        analysis_times[pair] = None

        # Verificar arquivo de última análise
        last_analysis_file = f"signals/agno_{pair}_last_analysis.json"
        if os.path.exists(last_analysis_file):
            try:
                with open(last_analysis_file, "r", encoding='utf-8') as f:
                    data = json.load(f)
                    analysis_times[pair] = {
                        "timestamp": data.get("timestamp"),
                        "signal": data.get("signal"),
                        "confidence": data.get("confidence", 0)
                    }
            except Exception:
                pass

    return analysis_times

# Função para obter preços atuais dos principais pares
@st.cache_data(ttl=10)
def get_market_prices():
    """Obtém preços atuais dos principais pares de criptomoedas"""
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "MATICUSDT"]
    prices = {}

    for symbol in symbols:
        try:
            response = requests.get("https://fapi.binance.com/fapi/v1/ticker/price", params={'symbol': symbol}, timeout=5)
            if response.status_code == 200:
                data = response.json()
                # Verificar se a chave 'price' existe e se é válida
                if isinstance(data, dict) and 'price' in data:
                    try:
                        prices[symbol] = float(data['price'])
                    except (ValueError, TypeError):
                        # Se não conseguir converter, pular este símbolo
                        continue
                else:
                    # Resposta não tem a estrutura esperada
                    continue
            else:
                # Status code não é 200, pular este símbolo
                continue
        except requests.exceptions.RequestException:
            # Erro de conexão/timeout, pular este símbolo
            continue
        except Exception:
            # Outro erro, pular este símbolo silenciosamente
            continue

    return prices

# Carregar dados
portfolio_data = load_portfolio_data()
trade_history = load_trade_history()

# Sidebar - Controles
with st.sidebar:
    st.header("⚙️ Controles")

    # Botão para iniciar análise contínua
    st.subheader("🚀 Sistema de Trading")
    if st.button("▶️ Iniciar Análise Contínua", type="primary", use_container_width=True):
        st.info("📡 Iniciando análise contínua...")
        st.code("python main.py --symbol BTCUSDT --mode monitor --paper", language="bash")
        st.warning("⚠️ Execute este comando no terminal para iniciar a análise contínua")

    if st.button("⏹️ Parar Análise", use_container_width=True):
        st.info("⏹️ Comando para parar será executado")

    st.markdown("---")

    # Auto-refresh
    auto_refresh = st.checkbox("🔄 Auto-refresh (5s)", value=False)

    # Botão de refresh manual
    if st.button("🔄 Atualizar Agora"):
        st.rerun()

    st.markdown("---")

    # Informações do sistema
    st.header("ℹ️ Informações")
    st.info("Dashboard atualizado em tempo real com dados do paper trading.")
    st.markdown("""
    **Recursos:**
    - 📊 Resumo do portfólio
    - 📈 Gráficos de performance
    - 💰 Posições abertas
    - 📜 Histórico de trades
    - 📉 Análise de resultados
    """)

    st.markdown("---")
    st.markdown("**Paginas:**")
    st.page_link("src/dashboard/pages/signal_analytics.py", label="📊 Signal Analytics", icon="📊")

# Layout principal
if portfolio_data:
    if portfolio_data.get("_binance_mode"):
        st.info("🔶 **Modo Real (Testnet)** — Posições e saldo da Binance Futures. Use a aba **Binance Futures** para detalhes e ordens.")
    # KPIs principais - Foco em P&L em PORCENTAGEM
    col1, col2, col3, col4 = st.columns(4)

    # Calcular P&L acumulado em % (soma de todos os trades fechados)
    closed_trades = [t for t in trade_history if t.get("status") in ["CLOSED", "CLOSED_PARTIAL"]]
    realized_pnl_percent = sum([get_pnl_percent(t) for t in closed_trades])

    # Calcular P&L médio não realizado (posições abertas)
    open_positions = portfolio_data.get("positions", {})
    unrealized_pnl_percent = 0.0
    market_prices = get_market_prices()

    open_pnl_list = []
    for pos_key, position in open_positions.items():
        symbol = position.get("symbol")
        entry_price = position.get("entry_price", 0)
        signal_type = position.get("signal", "BUY")
        current_price = market_prices.get(symbol) or position.get("mark_price") or entry_price
        if entry_price > 0:
            if signal_type == "BUY":
                pnl_percent = ((current_price - entry_price) / entry_price) * 100
            else:  # SELL
                pnl_percent = ((entry_price - current_price) / entry_price) * 100
            open_pnl_list.append(pnl_percent)

    if open_pnl_list:
        unrealized_pnl_percent = sum(open_pnl_list) / len(open_pnl_list)

    # P&L total acumulado (soma de todos os trades fechados)
    total_pnl_percent = realized_pnl_percent

    with col1:
        st.metric(
            "💰 P&L Acumulado",
            f"{realized_pnl_percent:+.2f}%",
            delta="Trades fechados"
        )

    with col2:
        st.metric(
            "📈 P&L Médio Aberto",
            f"{unrealized_pnl_percent:+.2f}%",
            delta="Posições abertas"
        )

    with col3:
        st.metric(
            "💵 P&L Total",
            f"{total_pnl_percent:+.2f}%",
            delta=f"{'✅' if total_pnl_percent >= 0 else '❌'}"
        )

    with col4:
        open_count = len(open_positions)
        st.metric(
            "📊 Posições Abertas",
            open_count
        )

    st.markdown("---")

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["📈 Overview", "💰 Posições Abertas", "📜 Histórico", "📊 Signal Analytics", "📉 Análise", "💹 Preços de Mercado", "🔍 Monitor Sistema", "🔶 Binance Futures"])

    with tab1:
        st.header("📈 Visão Geral do Portfólio")

        # Calcular estatísticas (apenas %)
        closed_trades = [t for t in trade_history if t.get("status") in ["CLOSED", "CLOSED_PARTIAL"]]
        open_trades = [t for t in trade_history if t.get("status") == "OPEN"]
        winning_trades = len([t for t in closed_trades if get_pnl_percent(t) > 0])
        losing_trades = len([t for t in closed_trades if get_pnl_percent(t) < 0])
        win_rate = (winning_trades / len(closed_trades) * 100) if closed_trades else 0
        total_pnl_percent = sum([get_pnl_percent(t) for t in closed_trades])

        # Métricas de performance
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("🎯 Win Rate", f"{win_rate:.1f}%")

        with col2:
            st.metric("✅ Trades Ganhadores", winning_trades)

        with col3:
            st.metric("❌ Trades Perdedores", losing_trades)

        with col4:
            st.metric("💰 P&L Acumulado", f"{total_pnl_percent:+.2f}%")

        with col5:
            st.metric("📊 Trades Abertos", len(open_trades))

        # Mostrar detalhes dos trades fechados
        if closed_trades:
            st.subheader("📋 Últimos Trades Fechados")

            closed_list = []
            for trade in closed_trades[-10:]:  # Últimos 10 trades
                entry_price = trade.get('entry_price', 0)
                stop_loss = trade.get('stop_loss', 0)
                take_profit_1 = trade.get('take_profit_1', 0)
                take_profit_2 = trade.get('take_profit_2', 0)
                position_size = trade.get('position_size', 0)
                position_value = trade.get('position_value', 0)

                # Calcular diferenças percentuais
                sl_diff = ((stop_loss - entry_price) / entry_price * 100) if entry_price > 0 else 0
                tp1_diff = ((take_profit_1 - entry_price) / entry_price * 100) if entry_price > 0 else 0
                tp2_diff = ((take_profit_2 - entry_price) / entry_price * 100) if entry_price > 0 else 0

                pnl_percent = get_pnl_percent(trade)
                closed_list.append({
                    "Data": trade.get("timestamp", "N/A")[:16],
                    "Símbolo": trade.get("symbol", "N/A"),
                    "Fonte": trade.get("source", "UNKNOWN"),
                    "Tipo": trade.get("signal", "N/A"),
                    "Entrada": f"${entry_price:,.2f}",
                    "Saída": f"${trade.get('close_price', 0):,.2f}" if trade.get('close_price') else "N/A",
                    "P&L": f"{pnl_percent:+.2f}%",
                    "Motivo": trade.get('close_reason', 'N/A')
                })

            df_closed = pd.DataFrame(closed_list)
            st.dataframe(df_closed, use_container_width=True, hide_index=True)

        # Mostrar posições abertas no overview também
        if open_trades:
            st.subheader("🔄 Posições Abertas Atualmente")

            open_list = []
            for trade in open_trades:
                entry_price = trade.get('entry_price', 0)
                stop_loss = trade.get('stop_loss', 0)
                take_profit_1 = trade.get('take_profit_1', 0)
                take_profit_2 = trade.get('take_profit_2', 0)
                position_size = trade.get('position_size', 0)
                position_value = trade.get('position_value', 0)

                # Calcular diferenças percentuais
                sl_diff = ((stop_loss - entry_price) / entry_price * 100) if entry_price > 0 else 0
                tp1_diff = ((take_profit_1 - entry_price) / entry_price * 100) if entry_price > 0 else 0
                tp2_diff = ((take_profit_2 - entry_price) / entry_price * 100) if entry_price > 0 else 0

                open_list.append({
                    "Data": trade.get("timestamp", "N/A")[:16],
                    "Símbolo": trade.get("symbol", "N/A"),
                    "Fonte": trade.get("source", "UNKNOWN"),  # DEEPSEEK ou AGNO
                    "Tipo": trade.get("signal", "N/A"),
                    "Entrada": f"${entry_price:,.2f}",
                    "Tamanho": f"{position_size:.6f}",
                    "Valor": f"${position_value:,.2f}",
                    "Stop Loss": f"${stop_loss:,.2f} ({sl_diff:+.1f}%)",
                    "Take Profit 1": f"${take_profit_1:,.2f} ({tp1_diff:+.1f}%)",
                    "Take Profit 2": f"${take_profit_2:,.2f} ({tp2_diff:+.1f}%)",
                    "Confiança": f"{trade.get('confidence', 0)}/10"
                })

            df_open = pd.DataFrame(open_list)
            st.dataframe(df_open, use_container_width=True, hide_index=True)

        # Gráfico de performance
        if len(trade_history) > 0:
            st.subheader("📊 Performance ao Longo do Tempo")

            # Preparar dados para gráfico
            trades_df = pd.DataFrame(trade_history)
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
            trades_df = trades_df.sort_values('timestamp')

            # Verificar se coluna 'pnl_percent' existe e preencher valores nulos
            if 'pnl_percent' not in trades_df.columns:
                trades_df['pnl_percent'] = 0.0
            else:
                # Preencher valores nulos com 0 (trades abertos ainda não têm P&L)
                trades_df['pnl_percent'] = trades_df['pnl_percent'].fillna(0.0)

            # Calcular P&L acumulado em % apenas para trades fechados
            trades_df['cumulative_pnl_percent'] = trades_df['pnl_percent'].cumsum()

            # Criar gráfico
            fig = go.Figure()

            last_pnl = trades_df['cumulative_pnl_percent'].iloc[-1] if len(trades_df) > 0 else 0
            fig.add_trace(go.Scatter(
                x=trades_df['timestamp'],
                y=trades_df['cumulative_pnl_percent'],
                mode='lines+markers',
                name='P&L Acumulado',
                line=dict(color='green' if last_pnl >= 0 else 'red', width=2),
                marker=dict(size=8)
            ))

            fig.update_layout(
                title="Evolução do P&L Acumulado",
                xaxis_title="Data",
                yaxis_title="P&L Acumulado (%)",
                hovermode='x unified',
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("💰 Posições Abertas")

        positions = portfolio_data.get("positions", {})

        if positions:
            market_prices = get_market_prices()

            # Exibir posições em cards com botões de fechar
            for position_key, position in positions.items():
                # Extrair dados da posição
                symbol = position.get("symbol", position_key.split("_")[0])
                source = position.get("source", "UNKNOWN")
                entry_price = position.get('entry_price', 0)
                position_size = position.get('position_size', 0)
                signal_type = position.get("signal", "BUY")
                confidence = position.get('confidence', 0)
                operation_type = position.get("operation_type", "SWING_TRADE")

                # Emoji para tipo de operação
                type_emoji = {
                    "SCALP": "⚡",
                    "DAY_TRADE": "☀️",
                    "SWING_TRADE": "🌊",
                    "POSITION_TRADE": "🏔️"
                }
                type_display = f"{type_emoji.get(operation_type, '📊')} {operation_type.replace('_', ' ')}"

                # Obter preço atual e calcular P&L
                current_price = market_prices.get(symbol, entry_price)
                if signal_type == "BUY":
                    pnl_percent = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
                else:  # SELL
                    pnl_percent = ((entry_price - current_price) / entry_price * 100) if entry_price > 0 else 0

                # Cor do P&L
                pnl_color = "green" if pnl_percent > 0 else "red" if pnl_percent < 0 else "gray"

                # Card da posição
                with st.container():
                    col1, col2 = st.columns([4, 1])

                    with col1:
                        st.markdown(f"""
                        **{symbol}** ({source}) - {signal_type} | {type_display}
                        - **Entrada:** ${entry_price:,.2f}
                        - **Atual:** ${current_price:,.2f}
                        - **P&L:** :{pnl_color}[{pnl_percent:+.2f}%]
                        - **Confiança:** {confidence}/10
                        """)

                    with col2:
                        # Botão para fechar posição
                        if st.button("❌ Fechar", key=f"close_{position_key}"):
                            # Obter preço atualizado
                            fresh_price = get_current_price(symbol)
                            if fresh_price:
                                # Usar instância global do sistema de trading
                                try:
                                    result = asyncio.run(real_paper_trading.close_position_manual(position_key, fresh_price))

                                    if result.get("success"):
                                        st.success(result.get("message"))
                                        st.rerun()
                                    else:
                                        st.error(result.get("error"))
                                except Exception as e:
                                    st.error(f"Erro ao fechar posição: {e}")
                            else:
                                st.error(f"Não foi possível obter preço atual de {symbol}")

                    st.markdown("---")
        else:
            st.info("ℹ️ Nenhuma posição aberta no momento.")

    with tab3:
        st.header("📜 Histórico de Trades e Sinais")

        if trade_history:
            # Preparar dados para tabela
            history_list = []
            for trade in trade_history:
                pnl_percent = get_pnl_percent(trade)
                entry = trade.get("entry_price", 0)
                sl = trade.get("stop_loss", 0)
                tp1 = trade.get("take_profit_1", 0)
                tp2 = trade.get("take_profit_2", 0)
                sig_type = trade.get("signal", "N/A")
                source = trade.get("source", trade.get("_source_file", "N/A"))
                confidence = trade.get("confidence", "")
                status = trade.get("status", "N/A")

                # Icone de direcao
                dir_icon = "🟢 LONG" if sig_type == "BUY" else "🔴 SHORT" if sig_type == "SELL" else sig_type

                history_list.append({
                    "Data": trade.get("timestamp", "N/A")[:16] if trade.get("timestamp") else "N/A",
                    "Simbolo": trade.get("symbol", "N/A"),
                    "Fonte": source.upper(),
                    "Direcao": dir_icon,
                    "Conf.": confidence if confidence else "-",
                    "Entry": f"${entry:,.4f}" if entry else "-",
                    "SL": f"${sl:,.4f}" if sl else "-",
                    "TP1": f"${tp1:,.4f}" if tp1 else "-",
                    "TP2": f"${tp2:,.4f}" if tp2 else "-",
                    "Status": status,
                    "P&L": f"{pnl_percent:+.2f}%" if pnl_percent != 0 else "-",
                })

            df_history = pd.DataFrame(history_list)

            # Filtros
            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                sources_avail = df_history["Fonte"].unique().tolist()
                filter_src = st.multiselect("Fonte", sources_avail, default=sources_avail, key="hist_src")
            with col_f2:
                dirs_avail = df_history["Direcao"].unique().tolist()
                filter_dir = st.multiselect("Direcao", dirs_avail, default=dirs_avail, key="hist_dir")
            with col_f3:
                status_avail = df_history["Status"].unique().tolist()
                filter_status = st.multiselect("Status", status_avail, default=status_avail, key="hist_status")

            mask = (
                df_history["Fonte"].isin(filter_src) &
                df_history["Direcao"].isin(filter_dir) &
                df_history["Status"].isin(filter_status)
            )
            st.dataframe(df_history[mask], use_container_width=True, hide_index=True, height=500)
            st.caption(f"Total: {mask.sum()} sinais/trades")
        else:
            st.info("ℹ️ Nenhum trade ou sinal registrado ainda.")

    with tab4:
        st.header("📊 Signal Analytics - Performance Real")
        st.markdown("Avalia cada sinal emitido contra dados reais do mercado (Binance klines)")

        try:
            from src.trading.signal_tracker import evaluate_all_signals, get_performance_summary

            @st.cache_data(ttl=120)
            def _load_evaluations():
                return evaluate_all_signals()

            with st.spinner("Buscando dados de mercado..."):
                evals = _load_evaluations()

            if not evals:
                st.info("Nenhum sinal BUY/SELL encontrado em signals/")
            else:
                summary = get_performance_summary(evals)
                df_evals = pd.DataFrame(evals)

                # KPIs
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                c1.metric("Total Sinais", summary.get("total_signals", 0))
                c2.metric("Finalizados", summary.get("closed", 0))
                c3.metric("Ativos", summary.get("active", 0))
                c4.metric("Win Rate", f"{summary.get('win_rate', 0):.1f}%")
                total_pnl = summary.get("total_pnl", 0)
                c5.metric("PnL Total", f"{total_pnl:+.2f}%", delta=f"{total_pnl:+.2f}%")
                c6.metric("Profit Factor", f"{summary.get('profit_factor', 0):.2f}")

                c1, c2, c3, c4, c5, c6 = st.columns(6)
                c1.metric("Avg Win", f"{summary.get('avg_win', 0):.2f}%")
                c2.metric("Avg Loss", f"{summary.get('avg_loss', 0):.2f}%")
                c3.metric("Expectancy", f"{summary.get('expectancy', 0):.3f}%")
                c4.metric("SL Hit", summary.get("sl_hits", 0))
                c5.metric("TP1 Hit", summary.get("tp1_hits", 0))
                c6.metric("TP2 Hit", summary.get("tp2_hits", 0))

                st.markdown("---")

                # Tabs internas
                stab1, stab2, stab3, stab4 = st.tabs(["📋 Sinais", "📈 Por Fonte", "🎯 Por Simbolo", "📊 Graficos"])

                with stab1:
                    # Tabela completa de sinais
                    display_cols = ["timestamp", "symbol", "source", "signal", "confidence",
                                    "entry_price", "stop_loss", "take_profit_1", "take_profit_2",
                                    "outcome", "exit_price", "pnl_percent", "duration_hours"]
                    available_cols = [c for c in display_cols if c in df_evals.columns]
                    disp = df_evals[available_cols].copy()
                    disp.columns = [c.replace("_", " ").title() for c in available_cols]
                    st.dataframe(disp, use_container_width=True, hide_index=True, height=400)

                with stab2:
                    # Performance por fonte
                    for source in df_evals["source"].unique():
                        src_evals = [e for e in evals if e.get("source") == source]
                        src_sum = get_performance_summary(src_evals)
                        st.markdown(f"**{source}**: {src_sum.get('closed', 0)} trades | "
                                    f"Win Rate: {src_sum.get('win_rate', 0):.1f}% | "
                                    f"PnL: {src_sum.get('total_pnl', 0):+.2f}% | "
                                    f"PF: {src_sum.get('profit_factor', 0):.2f} | "
                                    f"SL: {src_sum.get('sl_hits', 0)} TP1: {src_sum.get('tp1_hits', 0)} TP2: {src_sum.get('tp2_hits', 0)}")

                    # Long vs Short
                    st.markdown("---")
                    for direction, label in [("BUY", "LONG"), ("SELL", "SHORT")]:
                        dir_evals = [e for e in evals if e.get("signal") == direction]
                        dir_sum = get_performance_summary(dir_evals)
                        st.markdown(f"**{label}**: {dir_sum.get('closed', 0)} trades | "
                                    f"Win Rate: {dir_sum.get('win_rate', 0):.1f}% | "
                                    f"PnL: {dir_sum.get('total_pnl', 0):+.2f}% | "
                                    f"PF: {dir_sum.get('profit_factor', 0):.2f}")

                with stab3:
                    # Performance por simbolo
                    sym_rows = []
                    for sym in sorted(df_evals["symbol"].unique()):
                        sym_evals = [e for e in evals if e.get("symbol") == sym]
                        sym_sum = get_performance_summary(sym_evals)
                        sym_rows.append({
                            "Simbolo": sym,
                            "Trades": sym_sum.get("closed", 0),
                            "Win Rate": f"{sym_sum.get('win_rate', 0):.0f}%",
                            "PnL %": round(sym_sum.get("total_pnl", 0), 2),
                            "PF": round(sym_sum.get("profit_factor", 0), 2),
                            "SL": sym_sum.get("sl_hits", 0),
                            "TP1": sym_sum.get("tp1_hits", 0),
                            "TP2": sym_sum.get("tp2_hits", 0),
                        })
                    if sym_rows:
                        df_sym = pd.DataFrame(sym_rows)
                        st.dataframe(df_sym, use_container_width=True, hide_index=True)

                        # Bar chart PnL por simbolo
                        fig_bar = px.bar(df_sym, x="Simbolo", y="PnL %",
                                         color="PnL %",
                                         color_continuous_scale=["#ff4444", "#ffff00", "#00cc00"],
                                         title="PnL por Simbolo")
                        fig_bar.update_layout(template="plotly_dark", height=350)
                        st.plotly_chart(fig_bar, use_container_width=True)

                with stab4:
                    closed_evals = df_evals[df_evals["outcome"].isin(["SL_HIT", "TP1_HIT", "TP2_HIT", "EXPIRED"])]
                    if not closed_evals.empty:
                        col1, col2 = st.columns(2)

                        with col1:
                            # PnL acumulado
                            sorted_c = closed_evals.sort_values("timestamp")
                            sorted_c["cum_pnl"] = sorted_c["pnl_percent"].cumsum()
                            fig_cum = go.Figure()
                            fig_cum.add_trace(go.Scatter(
                                x=sorted_c["timestamp"], y=sorted_c["cum_pnl"],
                                mode="lines+markers", fill="tozeroy",
                                line=dict(color="#00ccff", width=2),
                                fillcolor="rgba(0,204,255,0.1)"
                            ))
                            fig_cum.update_layout(title="PnL Acumulado", template="plotly_dark", height=350)
                            fig_cum.add_hline(y=0, line_dash="dash", line_color="gray")
                            st.plotly_chart(fig_cum, use_container_width=True)

                        with col2:
                            # Pie chart outcomes
                            oc = closed_evals["outcome"].value_counts()
                            colors_map = {"TP2_HIT": "#00ff00", "TP1_HIT": "#00cc00",
                                          "SL_HIT": "#ff4444", "EXPIRED": "#888888"}
                            fig_pie = px.pie(names=oc.index, values=oc.values,
                                             title="Distribuicao de Resultados",
                                             color=oc.index, color_discrete_map=colors_map)
                            fig_pie.update_layout(template="plotly_dark", height=350)
                            st.plotly_chart(fig_pie, use_container_width=True)

                        # Confianca vs PnL
                        if "confidence" in closed_evals.columns:
                            fig_scatter = px.scatter(
                                closed_evals, x="confidence", y="pnl_percent",
                                color="outcome", hover_data=["symbol", "source", "signal"],
                                color_discrete_map=colors_map,
                                title="Confianca vs PnL")
                            fig_scatter.update_layout(template="plotly_dark", height=350)
                            fig_scatter.add_hline(y=0, line_dash="dash", line_color="gray")
                            st.plotly_chart(fig_scatter, use_container_width=True)
                    else:
                        st.info("Nenhum sinal finalizado ainda para graficos.")

        except ImportError as e:
            st.error(f"Erro ao importar signal_tracker: {e}")
        except Exception as e:
            st.error(f"Erro ao carregar analytics: {e}")

    with tab5:
        st.header("📉 Análise Detalhada")

        if len(closed_trades) > 0:
            # Estatísticas dos trades fechados
            st.subheader("📊 Estatísticas dos Trades")

            # Filtrar apenas trades com P&L válido (em %)
            pnl_percent_values = [get_pnl_percent(t) for t in closed_trades if get_pnl_percent(t) != 0]

            col1, col2 = st.columns(2)

            with col1:
                # Distribuição de P&L
                if pnl_percent_values:
                    fig_pnl = px.histogram(
                        x=pnl_percent_values,
                        nbins=20,
                        title="Distribuição de P&L",
                        labels={"x": "P&L (%)", "y": "Frequência"}
                    )
                    st.plotly_chart(fig_pnl, use_container_width=True)
                else:
                    st.info("Nenhum dado de P&L disponível")

            with col2:
                # Box plot de P&L
                if pnl_percent_values:
                    fig_box = go.Figure()
                    fig_box.add_trace(go.Box(
                        y=pnl_percent_values,
                        name="P&L Distribution",
                        boxmean='sd'
                    ))
                    fig_box.update_layout(
                        title="Distribuição de P&L (Box Plot)",
                        yaxis_title="P&L (%)"
                    )
                    st.plotly_chart(fig_box, use_container_width=True)
                else:
                    st.info("Nenhum dado de P&L disponível")

            # Estatísticas descritivas (em %)
            st.subheader("📈 Estatísticas Descritivas")

            if pnl_percent_values:
                stats = {
                    "Média": f"{sum(pnl_percent_values) / len(pnl_percent_values):+.2f}%",
                    "Mediana": f"{sorted(pnl_percent_values)[len(pnl_percent_values)//2]:+.2f}%",
                    "Máximo": f"{max(pnl_percent_values):+.2f}%",
                    "Mínimo": f"{min(pnl_percent_values):+.2f}%",
                    "Total Acumulado": f"{sum(pnl_percent_values):+.2f}%"
                }
            else:
                stats = {"Mensagem": "Nenhum dado disponível"}

            st.json(stats)
        else:
            st.info("ℹ️ Não há trades fechados para análise.")

    with tab6:
        st.header("💹 Preços de Mercado em Tempo Real")

        # Obter preços atuais
        market_prices = get_market_prices()

        if market_prices:
            # Criar DataFrame com preços
            prices_data = []
            for symbol, price in market_prices.items():
                prices_data.append({
                    "Par": symbol,
                    "Preço Atual": f"${price:,.2f}" if price >= 1 else f"${price:.6f}",
                    "Preço Numérico": price
                })

            df_prices = pd.DataFrame(prices_data)
            df_prices = df_prices.sort_values("Preço Numérico", ascending=False)

            # Mostrar tabela
            st.dataframe(
                df_prices[["Par", "Preço Atual"]],
                use_container_width=True,
                hide_index=True
            )

            # Gráfico de barras
            fig_prices = px.bar(
                df_prices,
                x="Par",
                y="Preço Numérico",
                title="Preços Atuais dos Principais Pares",
                labels={"Preço Numérico": "Preço (USDT)", "Par": "Par de Negociação"}
            )
            fig_prices.update_layout(height=500)
            st.plotly_chart(fig_prices, use_container_width=True)
        else:
            st.warning("⚠️ Não foi possível carregar preços de mercado.")

    # =====================
    # NOVA ABA: MONITOR DO SISTEMA
    # =====================
    with tab7:
        st.header("🔍 Monitor do Sistema de Sinais")
        st.markdown("Acompanhe se o sistema está gerando sinais corretamente para todos os 10 pares monitorados.")

        # Top 10 pares
        monitored_pairs = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT",
                          "XRPUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT"]

        # Carregar dados
        positions = portfolio_data.get("positions", {}) if portfolio_data else {}
        last_signals = load_last_signals()
        last_analysis = load_last_analysis_timestamps()

        # Construir tabela de monitoramento
        monitor_data = []

        for pair in monitored_pairs:
            row = {
                "Par": pair,
                "DEEPSEEK": "—",
                "AGNO": "—",
                "Última Análise": "—",
                "Status": "⚪"
            }

            # Verificar posição DEEPSEEK aberta
            deepseek_key = f"{pair}_DEEPSEEK"
            deepseek_short_key = f"{pair}_DEEPSEEK_SHORT"

            has_deepseek_position = False
            if deepseek_key in positions and positions[deepseek_key].get("status") == "OPEN":
                signal_type = positions[deepseek_key].get("signal", "?")
                confidence = positions[deepseek_key].get("confidence", 0)
                row["DEEPSEEK"] = f"✅ {signal_type} ({confidence}/10)"
                has_deepseek_position = True
            elif deepseek_short_key in positions and positions[deepseek_short_key].get("status") == "OPEN":
                signal_type = positions[deepseek_short_key].get("signal", "?")
                confidence = positions[deepseek_short_key].get("confidence", 0)
                row["DEEPSEEK"] = f"✅ {signal_type} ({confidence}/10)"
                has_deepseek_position = True

            # Se não tem posição DEEPSEEK, mostrar último sinal
            if not has_deepseek_position:
                if last_signals.get(pair, {}).get("DEEPSEEK"):
                    last_ds = last_signals[pair]["DEEPSEEK"]
                    signal_type = last_ds.get("signal", "N/A")
                    confidence = last_ds.get("confidence", 0)

                    if signal_type == "NO_SIGNAL":
                        row["DEEPSEEK"] = f"⏸️ NO_SIGNAL ({confidence}/10)"
                    elif confidence < 7:
                        row["DEEPSEEK"] = f"❌ {signal_type} ({confidence}/10) - Baixa"
                    else:
                        row["DEEPSEEK"] = f"⚠️ {signal_type} ({confidence}/10) - Não exec."
                else:
                    row["DEEPSEEK"] = "❓ Sem sinal"

            # Verificar posição AGNO aberta
            agno_key = f"{pair}_AGNO"
            agno_short_key = f"{pair}_AGNO_SHORT"

            has_agno_position = False
            if agno_key in positions and positions[agno_key].get("status") == "OPEN":
                signal_type = positions[agno_key].get("signal", "?")
                confidence = positions[agno_key].get("confidence", 0)
                row["AGNO"] = f"✅ {signal_type} ({confidence}/10)"
                has_agno_position = True
            elif agno_short_key in positions and positions[agno_short_key].get("status") == "OPEN":
                signal_type = positions[agno_short_key].get("signal", "?")
                confidence = positions[agno_short_key].get("confidence", 0)
                row["AGNO"] = f"✅ {signal_type} ({confidence}/10)"
                has_agno_position = True

            # Se não tem posição AGNO, mostrar último sinal
            if not has_agno_position:
                if last_signals.get(pair, {}).get("AGNO"):
                    last_ag = last_signals[pair]["AGNO"]
                    signal_type = last_ag.get("signal", "N/A")
                    confidence = last_ag.get("confidence", 0)

                    if signal_type == "NO_SIGNAL":
                        row["AGNO"] = f"⏸️ NO_SIGNAL ({confidence}/10)"
                    elif confidence < 7:
                        row["AGNO"] = f"❌ {signal_type} ({confidence}/10) - Baixa"
                    else:
                        row["AGNO"] = f"⚠️ {signal_type} ({confidence}/10) - Não exec."
                else:
                    row["AGNO"] = "❓ Sem sinal"

            # Verificar posições antigas (UNKNOWN/LEGACY)
            unknown_key = f"{pair}_SHORT"

            if unknown_key in positions and positions[unknown_key].get("status") == "OPEN":
                source = positions[unknown_key].get("source", "LEGACY")
                signal_type = positions[unknown_key].get("signal", "?")
                confidence = positions[unknown_key].get("confidence", 0)
                # Adicionar nota sobre posição legada
                if row["DEEPSEEK"] == "—" or "Sem sinal" in row["DEEPSEEK"]:
                    row["DEEPSEEK"] = f"🔄 {signal_type} ({confidence}/10) - {source}"

            # Última análise
            if last_analysis.get(pair):
                timestamp = last_analysis[pair].get("timestamp", "")
                if timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        if dt.tzinfo is None:
                            now = datetime.now()
                        else:
                            now = datetime.now(dt.tzinfo)
                        diff = now - dt
                        minutes = int(diff.total_seconds() / 60)

                        if minutes < 60:
                            row["Última Análise"] = f"há {minutes}min"
                        elif minutes < 1440:
                            hours = minutes // 60
                            row["Última Análise"] = f"há {hours}h"
                        else:
                            days = minutes // 1440
                            row["Última Análise"] = f"há {days}d"
                    except Exception:
                        row["Última Análise"] = timestamp[:16]

            # Status geral
            if has_deepseek_position and has_agno_position:
                row["Status"] = "🟢 Completo"
            elif has_deepseek_position or has_agno_position:
                row["Status"] = "🟡 Parcial"
            elif "NO_SIGNAL" in row["DEEPSEEK"] or "NO_SIGNAL" in row["AGNO"]:
                row["Status"] = "⚪ Aguardando"
            elif "Baixa" in row["DEEPSEEK"] and "Baixa" in row["AGNO"]:
                row["Status"] = "🔴 Sem oportunidade"
            else:
                row["Status"] = "⚪ Aguardando"

            monitor_data.append(row)

        # Mostrar tabela
        df_monitor = pd.DataFrame(monitor_data)

        # Estilizar tabela
        st.dataframe(
            df_monitor,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Par": st.column_config.TextColumn("Par", width="small"),
                "DEEPSEEK": st.column_config.TextColumn("🤖 DEEPSEEK", width="medium"),
                "AGNO": st.column_config.TextColumn("🧠 AGNO", width="medium"),
                "Última Análise": st.column_config.TextColumn("⏰ Última Análise", width="small"),
                "Status": st.column_config.TextColumn("📊 Status", width="small"),
            }
        )

        # Legenda
        st.markdown("---")
        st.markdown('''
        **Legenda:**
        - ✅ **Posição aberta** - Trade em andamento
        - ❌ **Baixa confiança** - Sinal gerado mas não executado (confiança < 7)
        - ⏸️ **NO_SIGNAL** - Análise feita mas sem oportunidade
        - ⚠️ **Não executado** - Sinal válido mas não executado (posição existente, etc.)
        - ❓ **Sem sinal** - Nenhum sinal encontrado para este par
        - 🔄 **LEGACY** - Posição antiga do sistema anterior

        **Status:**
        - 🟢 **Completo** - Tem posição DEEPSEEK e AGNO
        - 🟡 **Parcial** - Tem posição de uma fonte apenas
        - ⚪ **Aguardando** - Sem posição, aguardando próxima análise
        - 🔴 **Sem oportunidade** - Ambas fontes com confiança baixa
        ''')

        # Seção de Fechamento Rápido
        st.markdown("---")
        st.subheader("⚡ Fechamento Rápido")
        st.markdown("Feche posições abertas com um clique (usa preço atual de mercado)")

        if positions:
            # Organizar posições em grid 3 colunas
            position_keys = list(positions.keys())
            cols_per_row = 3

            for i in range(0, len(position_keys), cols_per_row):
                cols = st.columns(cols_per_row)

                for j in range(cols_per_row):
                    idx = i + j
                    if idx < len(position_keys):
                        position_key = position_keys[idx]
                        position = positions[position_key]

                        symbol = position.get("symbol", position_key.split("_")[0])
                        source = position.get("source", "UNKNOWN")
                        signal_type = position.get("signal", "?")

                        with cols[j]:
                            button_label = f"❌ {symbol} ({source})"
                            if st.button(button_label, key=f"quick_close_{position_key}"):
                                # Obter preço atualizado
                                fresh_price = get_current_price(symbol)
                                if fresh_price:
                                    try:
                                        result = asyncio.run(real_paper_trading.close_position_manual(position_key, fresh_price))

                                        if result.get("success"):
                                            st.success(f"✅ {symbol} fechado!")
                                            st.rerun()
                                        else:
                                            st.error(result.get("error"))
                                    except Exception as e:
                                        st.error(f"Erro: {e}")
                                else:
                                    st.error(f"Erro ao obter preço de {symbol}")
        else:
            st.info("Nenhuma posição aberta para fechar.")

        # Estatísticas rápidas
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            positions_deepseek = sum(1 for row in monitor_data if "✅" in row["DEEPSEEK"])
            st.metric("🤖 Posições DEEPSEEK", positions_deepseek)

        with col2:
            positions_agno = sum(1 for row in monitor_data if "✅" in row["AGNO"])
            st.metric("🧠 Posições AGNO", positions_agno)

        with col3:
            low_confidence = sum(1 for row in monitor_data if "Baixa" in row["DEEPSEEK"] or "Baixa" in row["AGNO"])
            st.metric("❌ Baixa Confiança", low_confidence)

        with col4:
            no_signal = sum(1 for row in monitor_data if "Sem sinal" in row["DEEPSEEK"] or "Sem sinal" in row["AGNO"])
            st.metric("❓ Sem Sinal", no_signal)

    # ============================================================
    # TAB 8: BINANCE FUTURES
    # ============================================================
    with tab8:
        st.header("🔶 Binance Futures - Posições Reais")

        # Verificar configuração
        config = get_binance_config()

        # Indicador de modo
        if config["testnet"]:
            st.info(f"🧪 **Modo: TESTNET** - Conectado a {config['base_url']}")
        else:
            st.warning(f"⚠️ **Modo: PRODUÇÃO** - Conectado a {config['base_url']} - ORDENS REAIS!")

        # Botão de atualização
        col_refresh, col_info = st.columns([1, 3])
        with col_refresh:
            if st.button("🔄 Atualizar Binance", key="refresh_binance"):
                st.cache_data.clear()
                st.rerun()

        # Obter dados
        balance_data = get_binance_balance()
        positions_data = get_binance_positions()
        orders_data = get_binance_open_orders()

        # Mostrar saldo
        st.subheader("💰 Saldo da Conta")
        if "error" in balance_data:
            st.error(f"Erro ao obter saldo: {balance_data['error']}")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("💵 Saldo Total", f"${balance_data.get('balance', 0):,.2f} USDT")
            with col2:
                st.metric("✅ Disponível", f"${balance_data.get('available', 0):,.2f} USDT")
            with col3:
                unrealized = balance_data.get('unrealized_pnl', 0)
                st.metric("📊 P&L Não Realizado", f"${unrealized:,.2f}",
                         delta=f"{unrealized:+,.2f}" if unrealized != 0 else None)

        st.markdown("---")

        # Mostrar posições
        st.subheader("📊 Posições Abertas")
        if "error" in positions_data:
            st.error(f"Erro ao obter posições: {positions_data['error']}")
        elif not positions_data.get("positions"):
            st.success("✅ Nenhuma posição aberta na Binance Futures")
        else:
            positions = positions_data["positions"]

            # Métricas resumo primeiro
            total_pnl = sum(float(pos.get("unRealizedProfit", 0)) for pos in positions)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Total Posições", len(positions))
            with col2:
                st.metric("💰 P&L Total", f"${total_pnl:,.2f}")
            with col3:
                longs = sum(1 for pos in positions if float(pos.get("positionAmt", 0)) > 0)
                shorts = len(positions) - longs
                st.metric("📈/📉 Long/Short", f"{longs}/{shorts}")
            with col4:
                # Botão para fechar TODAS as posições
                if st.button("🚫 Fechar TODAS", key="close_all_positions", type="secondary"):
                    st.session_state["confirm_close_all"] = True

            # Confirmar fechamento de todas
            if st.session_state.get("confirm_close_all"):
                st.warning("⚠️ Tem certeza que deseja fechar TODAS as posições?")
                col_yes, col_no = st.columns(2)
                with col_yes:
                    if st.button("✅ Sim, fechar todas", key="confirm_yes_all"):
                        for pos in positions:
                            symbol = pos.get("symbol", "")
                            result = close_binance_position(symbol)
                            if result.get("success"):
                                st.success(f"✅ {symbol} fechada")
                            else:
                                st.error(f"❌ {symbol}: {result.get('error')}")
                        st.session_state["confirm_close_all"] = False
                        st.cache_data.clear()
                        time.sleep(1)
                        st.rerun()
                with col_no:
                    if st.button("❌ Não, cancelar", key="confirm_no_all"):
                        st.session_state["confirm_close_all"] = False
                        st.rerun()

            st.markdown("---")

            # Mostrar cada posição com botão de fechar
            for pos in positions:
                symbol = pos.get("symbol", "")
                position_amt = float(pos.get("positionAmt", 0))
                entry_price = float(pos.get("entryPrice", 0))
                mark_price = float(pos.get("markPrice", 0))
                unrealized_pnl = float(pos.get("unRealizedProfit", 0))
                leverage = pos.get("leverage", "1")

                # Determinar lado
                side = "LONG 📈" if position_amt > 0 else "SHORT 📉"

                # Calcular ROI
                if entry_price > 0:
                    if position_amt > 0:  # LONG
                        roi = ((mark_price - entry_price) / entry_price) * 100
                    else:  # SHORT
                        roi = ((entry_price - mark_price) / entry_price) * 100
                else:
                    roi = 0

                # Cor do P&L
                pnl_color = "green" if unrealized_pnl >= 0 else "red"

                # Container para cada posição
                with st.container():
                    col1, col2, col3, col4, col5, col6 = st.columns([2, 1.5, 1.5, 1.5, 1.5, 1])

                    with col1:
                        st.markdown(f"**{symbol}** {side}")
                        st.caption(f"Tamanho: {abs(position_amt):,.4f} | Alavancagem: {leverage}x")

                    with col2:
                        st.metric("Entry", f"${entry_price:,.4f}")

                    with col3:
                        st.metric("Mark", f"${mark_price:,.4f}")

                    with col4:
                        st.metric("P&L", f"${unrealized_pnl:,.2f}", delta=f"{roi:+.2f}%")

                    with col5:
                        # Mostrar ordens SL/TP
                        orders_for_symbol = [o for o in orders_data.get("orders", []) if o.get("symbol") == symbol]
                        sl_count = len([o for o in orders_for_symbol if "STOP" in o.get("type", "")])
                        tp_count = len([o for o in orders_for_symbol if "TAKE_PROFIT" in o.get("type", "")])
                        st.caption(f"🛑 SL: {sl_count} | 🎯 TP: {tp_count}")

                    with col6:
                        # Botão de fechar
                        btn_key = f"close_{symbol}"
                        if st.button("❌ Fechar", key=btn_key, type="primary"):
                            with st.spinner(f"Fechando {symbol}..."):
                                result = close_binance_position(symbol)
                                if result.get("success"):
                                    st.success(f"✅ {result.get('message')}")
                                    st.cache_data.clear()
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error(f"❌ Erro: {result.get('error')}")

                    st.markdown("---")

        st.markdown("---")

        # Mostrar ordens abertas
        st.subheader("📋 Ordens Abertas (SL/TP)")
        if "error" in orders_data:
            st.error(f"Erro ao obter ordens: {orders_data['error']}")
        elif not orders_data.get("orders"):
            st.info("ℹ️ Nenhuma ordem aberta")
        else:
            orders = orders_data["orders"]

            # Agrupar por símbolo
            orders_by_symbol = {}
            for order in orders:
                symbol = order.get("symbol", "")
                if symbol not in orders_by_symbol:
                    orders_by_symbol[symbol] = []
                orders_by_symbol[symbol].append(order)

            # Criar DataFrame
            orders_list = []
            for order in orders:
                order_type = order.get("type", "")
                side = order.get("side", "")
                quantity = float(order.get("origQty", 0))
                stop_price = float(order.get("stopPrice", 0))

                # Ícone baseado no tipo
                if "STOP" in order_type:
                    type_icon = "🛑 Stop Loss"
                elif "TAKE_PROFIT" in order_type:
                    type_icon = "🎯 Take Profit"
                else:
                    type_icon = order_type

                orders_list.append({
                    "Símbolo": order.get("symbol", ""),
                    "Tipo": type_icon,
                    "Lado": side,
                    "Quantidade": quantity,
                    "Trigger": f"${stop_price:,.4f}" if stop_price > 0 else "-",
                    "Status": order.get("status", "")
                })

            df_orders = pd.DataFrame(orders_list)
            st.dataframe(df_orders, use_container_width=True, hide_index=True)

            st.caption(f"Total: {len(orders)} ordens abertas")

else:
    st.warning("⚠️ Nenhum dado de portfólio encontrado. Execute alguns trades primeiro!")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        📊 Paper Trading Dashboard | Atualizado em tempo real
    </div>
    """,
    unsafe_allow_html=True
)
