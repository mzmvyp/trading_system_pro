"""
Signal Tracker - Rastreamento de Sinais contra Preço Real
==========================================================

Le sinais salvos em signals/ e verifica contra dados reais da Binance:
- Se TP1, TP2 ou SL foi atingido
- Calcula PnL real de cada sinal
- Agrega performance por source, direction, symbol
"""

import json
import os
import glob
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from src.core.logger import get_logger

logger = get_logger(__name__)

# Binance public API (sem auth)
BINANCE_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"


def get_klines(symbol: str, interval: str = "1m", start_time_ms: int = None, limit: int = 1500) -> List:
    """Busca klines (candles) da Binance Futures"""
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_time_ms:
        params["startTime"] = start_time_ms

    try:
        resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning(f"Erro ao buscar klines de {symbol}: {e}")
        return []


def load_all_signals(signals_dir: str = "signals") -> List[Dict]:
    """Carrega todos os sinais salvos"""
    signals = []
    pattern = os.path.join(signals_dir, "agno_*_*.json")

    for filepath in glob.glob(pattern):
        # Ignorar arquivos de last_analysis
        if "_last_analysis" in filepath:
            continue
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                sig = json.load(f)
                sig["_filepath"] = filepath
                signals.append(sig)
        except (json.JSONDecodeError, IOError):
            continue

    # Ordenar por timestamp
    signals.sort(key=lambda s: s.get("timestamp", ""), reverse=True)
    return signals


def evaluate_signal(signal: Dict) -> Dict:
    """
    Avalia um sinal contra dados reais do mercado.

    Verifica se o preço atingiu SL, TP1 ou TP2 depois da emissão do sinal.
    Calcula PnL baseado no que aconteceu.

    Returns:
        Dict com resultado da avaliação
    """
    symbol = signal.get("symbol", "")
    signal_type = signal.get("signal", "")
    entry_price = signal.get("entry_price", 0)
    stop_loss = signal.get("stop_loss", 0)
    tp1 = signal.get("take_profit_1", 0)
    tp2 = signal.get("take_profit_2", 0)
    source = signal.get("source", "UNKNOWN")
    confidence = signal.get("confidence", 0)
    timestamp_str = signal.get("timestamp", "")

    result = {
        "symbol": symbol,
        "signal": signal_type,
        "source": source,
        "confidence": confidence,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "take_profit_1": tp1,
        "take_profit_2": tp2,
        "timestamp": timestamp_str,
        "outcome": "PENDING",  # PENDING, SL_HIT, TP1_HIT, TP2_HIT, ACTIVE
        "pnl_percent": 0.0,
        "exit_price": 0.0,
        "exit_time": "",
        "duration_hours": 0.0,
        "max_favorable": 0.0,  # Máximo a favor (MFE)
        "max_adverse": 0.0,  # Máximo contra (MAE)
    }

    if signal_type not in ("BUY", "SELL") or not entry_price or not symbol:
        result["outcome"] = "INVALID"
        return result

    # Parse timestamp
    try:
        if "T" in timestamp_str:
            sig_time = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            if sig_time.tzinfo:
                sig_time = sig_time.replace(tzinfo=None)
        else:
            sig_time = datetime.strptime(timestamp_str[:19], "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        result["outcome"] = "INVALID"
        return result

    start_ms = int(sig_time.timestamp() * 1000)

    # Buscar klines de 5 min desde a emissão do sinal
    # 1500 candles de 5min = ~5.2 dias de dados
    klines = get_klines(symbol, interval="5m", start_time_ms=start_ms, limit=1500)
    if not klines:
        # Tentar com intervalo maior
        klines = get_klines(symbol, interval="15m", start_time_ms=start_ms, limit=1000)

    if not klines:
        result["outcome"] = "NO_DATA"
        return result

    is_long = signal_type == "BUY"
    max_favorable = 0.0
    max_adverse = 0.0

    for candle in klines:
        high = float(candle[2])
        low = float(candle[3])
        candle_time_ms = candle[0]
        candle_time = datetime.fromtimestamp(candle_time_ms / 1000)

        if is_long:
            # LONG: favorável = preço sobe, adverso = preço desce
            favorable = (high - entry_price) / entry_price * 100
            adverse = (entry_price - low) / entry_price * 100
        else:
            # SHORT: favorável = preço desce, adverso = preço sobe
            favorable = (entry_price - low) / entry_price * 100
            adverse = (high - entry_price) / entry_price * 100

        max_favorable = max(max_favorable, favorable)
        max_adverse = max(max_adverse, adverse)

        # Verificar SL (prioridade: SL verificado primeiro no mesmo candle)
        if stop_loss > 0:
            sl_hit = False
            if is_long and low <= stop_loss:
                sl_hit = True
            elif not is_long and high >= stop_loss:
                sl_hit = True

            if sl_hit:
                result["outcome"] = "SL_HIT"
                result["exit_price"] = stop_loss
                result["exit_time"] = candle_time.isoformat()
                result["duration_hours"] = (candle_time - sig_time).total_seconds() / 3600
                if is_long:
                    result["pnl_percent"] = (stop_loss - entry_price) / entry_price * 100
                else:
                    result["pnl_percent"] = (entry_price - stop_loss) / entry_price * 100
                result["max_favorable"] = max_favorable
                result["max_adverse"] = max_adverse
                return result

        # Verificar TP1
        if tp1 > 0:
            tp1_hit = False
            if is_long and high >= tp1:
                tp1_hit = True
            elif not is_long and low <= tp1:
                tp1_hit = True

            if tp1_hit:
                # TP1 bateu - agora verificar se TP2 também bate depois
                # Continuar verificando klines restantes para TP2
                result["outcome"] = "TP1_HIT"
                result["exit_price"] = tp1
                result["exit_time"] = candle_time.isoformat()
                result["duration_hours"] = (candle_time - sig_time).total_seconds() / 3600
                if is_long:
                    result["pnl_percent"] = (tp1 - entry_price) / entry_price * 100
                else:
                    result["pnl_percent"] = (entry_price - tp1) / entry_price * 100

                # Continuar para ver se TP2 bate (com SL no break-even)
                for candle2 in klines[klines.index(candle):]:
                    h2 = float(candle2[2])
                    l2 = float(candle2[3])
                    ct2 = datetime.fromtimestamp(candle2[0] / 1000)

                    # Após TP1, SL move para break-even (entry_price)
                    if is_long and l2 <= entry_price:
                        break  # Break-even atingido, fica com PnL do TP1
                    if not is_long and h2 >= entry_price:
                        break

                    if tp2 > 0:
                        if is_long and h2 >= tp2:
                            result["outcome"] = "TP2_HIT"
                            result["exit_price"] = tp2
                            result["exit_time"] = ct2.isoformat()
                            result["duration_hours"] = (ct2 - sig_time).total_seconds() / 3600
                            if is_long:
                                result["pnl_percent"] = (tp2 - entry_price) / entry_price * 100
                            else:
                                result["pnl_percent"] = (entry_price - tp2) / entry_price * 100
                            break
                        if not is_long and l2 <= tp2:
                            result["outcome"] = "TP2_HIT"
                            result["exit_price"] = tp2
                            result["exit_time"] = ct2.isoformat()
                            result["duration_hours"] = (ct2 - sig_time).total_seconds() / 3600
                            result["pnl_percent"] = (entry_price - tp2) / entry_price * 100
                            break

                # PnL final: 50% no TP1 + 50% no TP2 (ou break-even)
                if result["outcome"] == "TP2_HIT":
                    pnl_tp1 = ((tp1 - entry_price) / entry_price * 100) if is_long else ((entry_price - tp1) / entry_price * 100)
                    pnl_tp2 = ((tp2 - entry_price) / entry_price * 100) if is_long else ((entry_price - tp2) / entry_price * 100)
                    result["pnl_percent"] = (pnl_tp1 * 0.5) + (pnl_tp2 * 0.5)

                result["max_favorable"] = max_favorable
                result["max_adverse"] = max_adverse
                return result

    # Se não atingiu nada, sinal ainda ativo ou expirou
    now = datetime.now()
    hours_since = (now - sig_time).total_seconds() / 3600

    if hours_since > 120:  # > 5 dias
        result["outcome"] = "EXPIRED"
    else:
        result["outcome"] = "ACTIVE"

    # PnL não realizado (usar último preço)
    if klines:
        last_close = float(klines[-1][4])
        if is_long:
            result["pnl_percent"] = (last_close - entry_price) / entry_price * 100
        else:
            result["pnl_percent"] = (entry_price - last_close) / entry_price * 100
        result["exit_price"] = last_close

    result["duration_hours"] = hours_since
    result["max_favorable"] = max_favorable
    result["max_adverse"] = max_adverse
    return result


def evaluate_all_signals(signals_dir: str = "signals", cache_file: str = "signals/evaluations_cache.json") -> List[Dict]:
    """
    Avalia todos os sinais e cacheia resultados.
    Sinais já finalizados (SL_HIT, TP1_HIT, TP2_HIT, EXPIRED) não são re-avaliados.
    """
    signals = load_all_signals(signals_dir)

    # Carregar cache
    cache = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
                for item in cached:
                    key = f"{item.get('symbol')}_{item.get('timestamp')}_{item.get('source')}"
                    cache[key] = item
        except (json.JSONDecodeError, IOError):
            pass

    results = []
    updated = False

    for sig in signals:
        sig_type = sig.get("signal", "")
        if sig_type not in ("BUY", "SELL"):
            continue

        key = f"{sig.get('symbol')}_{sig.get('timestamp')}_{sig.get('source')}"

        # Verificar cache - sinais finalizados não precisam re-avaliar
        if key in cache:
            cached_result = cache[key]
            outcome = cached_result.get("outcome", "")
            if outcome in ("SL_HIT", "TP1_HIT", "TP2_HIT", "EXPIRED", "INVALID"):
                results.append(cached_result)
                continue

        # Avaliar contra dados reais
        evaluation = evaluate_signal(sig)
        results.append(evaluation)
        cache[key] = evaluation
        updated = True

    # Salvar cache atualizado
    if updated:
        try:
            os.makedirs(os.path.dirname(cache_file) if os.path.dirname(cache_file) else ".", exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(list(cache.values()), f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            logger.warning(f"Erro ao salvar cache de avaliações: {e}")

    return results


def get_performance_summary(evaluations: List[Dict]) -> Dict:
    """Calcula métricas agregadas de performance"""
    if not evaluations:
        return {}

    # Filtrar apenas sinais finalizados
    closed = [e for e in evaluations if e["outcome"] in ("SL_HIT", "TP1_HIT", "TP2_HIT", "EXPIRED")]
    active = [e for e in evaluations if e["outcome"] == "ACTIVE"]
    total = len(closed)

    if total == 0:
        return {
            "total_signals": len(evaluations),
            "closed": 0,
            "active": len(active),
            "win_rate": 0,
        }

    wins = [e for e in closed if e["pnl_percent"] > 0]
    losses = [e for e in closed if e["pnl_percent"] <= 0]

    pnl_list = [e["pnl_percent"] for e in closed]
    total_pnl = sum(pnl_list)
    avg_pnl = total_pnl / total if total else 0
    avg_win = sum(e["pnl_percent"] for e in wins) / len(wins) if wins else 0
    avg_loss = sum(e["pnl_percent"] for e in losses) / len(losses) if losses else 0

    sl_hits = len([e for e in closed if e["outcome"] == "SL_HIT"])
    tp1_hits = len([e for e in closed if e["outcome"] == "TP1_HIT"])
    tp2_hits = len([e for e in closed if e["outcome"] == "TP2_HIT"])
    expired = len([e for e in closed if e["outcome"] == "EXPIRED"])

    # Profit factor
    gross_profit = sum(e["pnl_percent"] for e in wins) if wins else 0
    gross_loss = abs(sum(e["pnl_percent"] for e in losses)) if losses else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    # Expectancy (esperança matemática)
    win_rate = len(wins) / total * 100 if total else 0
    expectancy = (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss)

    # Duração média
    durations = [e["duration_hours"] for e in closed if e["duration_hours"] > 0]
    avg_duration = sum(durations) / len(durations) if durations else 0

    # MFE/MAE médios
    mfe_list = [e["max_favorable"] for e in closed]
    mae_list = [e["max_adverse"] for e in closed]

    return {
        "total_signals": len(evaluations),
        "closed": total,
        "active": len(active),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "avg_pnl": avg_pnl,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "best_trade": max(pnl_list) if pnl_list else 0,
        "worst_trade": min(pnl_list) if pnl_list else 0,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "sl_hits": sl_hits,
        "tp1_hits": tp1_hits,
        "tp2_hits": tp2_hits,
        "expired": expired,
        "avg_duration_hours": avg_duration,
        "avg_mfe": sum(mfe_list) / len(mfe_list) if mfe_list else 0,
        "avg_mae": sum(mae_list) / len(mae_list) if mae_list else 0,
    }
