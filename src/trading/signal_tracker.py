"""
Signal Tracker - Rastreamento de Sinais contra Preço Real
==========================================================

Le sinais salvos em signals/ e verifica contra dados reais da Binance:
- Se TP1, TP2 ou SL foi atingido
- Calcula PnL real de cada sinal
- Agrega performance por source, direction, symbol
"""

import glob
import json
import os
from datetime import datetime, timezone
from typing import Dict, List

import requests

from src.core.logger import get_logger

logger = get_logger(__name__)

# Binance public API (sem auth)
BINANCE_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"

# Log onde cada sinal registra o que cada sistema disse (LLM, ML, LSTM) para contabilizar acertividade
MODEL_VOTES_LOG = "signals/model_votes_log.jsonl"


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

    # Rastreabilidade: sinal foi efetivamente executado?
    executed = signal.get("executed", False)
    execution_mode = signal.get("execution_mode", None)
    ml_probability = signal.get("ml_probability", None)
    ml_prediction = signal.get("ml_prediction", None)

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
        "executed": executed,
        "execution_mode": execution_mode,
        "ml_probability": ml_probability,
        "ml_prediction": ml_prediction,
        "non_execution_reason": signal.get("non_execution_reason", ""),
    }

    if signal_type not in ("BUY", "SELL") or not entry_price or not symbol:
        result["outcome"] = "INVALID"
        return result

    # Parse timestamp (manter UTC-aware para subtração com now/candle_time)
    try:
        if "T" in timestamp_str:
            sig_time = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            if sig_time.tzinfo is None:
                sig_time = sig_time.replace(tzinfo=timezone.utc)
        else:
            sig_time = datetime.strptime(timestamp_str[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
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
        candle_time = datetime.fromtimestamp(candle_time_ms / 1000, tz=timezone.utc)

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
                _add_model_attribution(result, signal)
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
                    ct2 = datetime.fromtimestamp(candle2[0] / 1000, tz=timezone.utc)

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
                _add_model_attribution(result, signal)
                return result

    # Se não atingiu nada, sinal ainda ativo ou expirou
    now = datetime.now(timezone.utc)
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
    _add_model_attribution(result, signal)
    return result


def _add_model_attribution(result: Dict, signal: Dict) -> None:
    """
    Preenche no result se cada modelo acertou ou errou (para relatório de validadores).
    Só aplica quando o sinal está finalizado (SL/TP/EXPIRED).
    """
    outcome = result.get("outcome", "")
    if outcome not in ("SL_HIT", "TP1_HIT", "TP2_HIT", "EXPIRED"):
        result["is_winner"] = False
        result["ml_correct"] = None
        result["lstm_correct"] = None
        return

    pnl = result.get("pnl_percent", 0)
    is_winner = outcome in ("TP1_HIT", "TP2_HIT") or (outcome == "EXPIRED" and pnl > 0)
    result["is_winner"] = is_winner

    # ML: previu 1 (sucesso) ou 0 (falha) — acertou se previsão == resultado real
    ml_pred = signal.get("ml_prediction")
    if ml_pred is not None:
        result["ml_correct"] = (int(ml_pred) == 1) == is_winner
    else:
        result["ml_correct"] = None

    # LSTM: votou a favor (prob > 0.6) ou contra (prob < 0.4) — acertou se (a favor e win) ou (contra e loss)
    lstm_prob = signal.get("lstm_probability")
    if lstm_prob is not None:
        if lstm_prob > 0.6:
            result["lstm_correct"] = is_winner
        elif lstm_prob < 0.4:
            result["lstm_correct"] = not is_winner
        else:
            result["lstm_correct"] = None  # neutro
    else:
        result["lstm_correct"] = None


def _append_model_votes_log(signal: Dict, evaluation: Dict) -> None:
    """
    Regista no log o que cada sistema disse neste sinal (LLM=direção, ML, LSTM)
    para depois contabilizar qual tem maior acertividade. Inclui sinais não executados.
    """
    outcome = evaluation.get("outcome", "")
    if outcome not in ("SL_HIT", "TP1_HIT", "TP2_HIT", "EXPIRED"):
        return
    is_winner = evaluation.get("is_winner", False)
    lstm_prob = signal.get("lstm_probability")
    if lstm_prob is not None:
        if lstm_prob > 0.6:
            lstm_vote = 1
        elif lstm_prob < 0.4:
            lstm_vote = -1
        else:
            lstm_vote = 0
    else:
        lstm_vote = None
    record = {
        "symbol": signal.get("symbol", ""),
        "timestamp": signal.get("timestamp", ""),
        "source": signal.get("source", ""),
        "signal": signal.get("signal", ""),  # BUY = long, SELL = short (o que o LLM disse)
        "outcome": outcome,
        "is_winner": is_winner,
        "executed": signal.get("executed", False),
        "ml_prediction": signal.get("ml_prediction"),
        "lstm_probability": lstm_prob,
        "lstm_vote": lstm_vote,
        "ml_correct": evaluation.get("ml_correct"),
        "lstm_correct": evaluation.get("lstm_correct"),
        "confluence_score": signal.get("confluence_score"),
        "confluence_votes_for": signal.get("confluence_votes_for"),
    }
    try:
        os.makedirs(os.path.dirname(MODEL_VOTES_LOG) or ".", exist_ok=True)
        with open(MODEL_VOTES_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception as e:
        logger.warning(f"Erro ao registrar voto no log: {e}")


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
        if evaluation.get("outcome") in ("SL_HIT", "TP1_HIT", "TP2_HIT", "EXPIRED"):
            _append_model_votes_log(sig, evaluation)

    # Salvar cache atualizado
    if updated:
        try:
            os.makedirs(os.path.dirname(cache_file) if os.path.dirname(cache_file) else ".", exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(list(cache.values()), f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            logger.warning(f"Erro ao salvar cache de avaliações: {e}")

    return results


def _calc_metrics(evals: List[Dict]) -> Dict:
    """Calcula métricas para uma lista de avaliações"""
    closed = [e for e in evals if e["outcome"] in ("SL_HIT", "TP1_HIT", "TP2_HIT", "EXPIRED")]
    active = [e for e in evals if e["outcome"] == "ACTIVE"]
    total = len(closed)

    if total == 0:
        return {
            "total_signals": len(evals),
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

    gross_profit = sum(e["pnl_percent"] for e in wins) if wins else 0
    gross_loss = abs(sum(e["pnl_percent"] for e in losses)) if losses else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    win_rate = len(wins) / total * 100 if total else 0
    expectancy = (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss)

    durations = [e["duration_hours"] for e in closed if e["duration_hours"] > 0]
    avg_duration = sum(durations) / len(durations) if durations else 0

    mfe_list = [e["max_favorable"] for e in closed]
    mae_list = [e["max_adverse"] for e in closed]

    return {
        "total_signals": len(evals),
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


def get_system_accuracy_report(log_path: str = None) -> Dict:
    """
    Lê o log de votos (model_votes_log.jsonl) e retorna acertividade por sistema:
    - Por fonte (AGNO, DEEPSEEK): quantos disseram long/short e % acertos
    - Por direção (BUY, SELL): quantos e % acertos
    - ML e LSTM: acurácia
    Deduplica por (symbol, timestamp, source) ficando com a última ocorrência.
    """
    path = log_path or MODEL_VOTES_LOG
    if not os.path.exists(path):
        return {"by_source": {}, "by_direction": {}, "ml": {}, "lstm": {}, "total_records": 0}

    records = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logger.warning(f"Erro ao ler log de votos: {e}")
        return {"by_source": {}, "by_direction": {}, "ml": {}, "lstm": {}, "total_records": 0}

    if not records:
        return {"by_source": {}, "by_direction": {}, "ml": {}, "lstm": {}, "total_records": 0}

    # Deduplica por (symbol, timestamp, source) — fica a última
    seen = {}
    for r in records:
        key = (r.get("symbol"), r.get("timestamp"), r.get("source"))
        seen[key] = r
    unique = list(seen.values())

    # Por fonte (AGNO, DEEPSEEK)
    by_source = {}
    for r in unique:
        src = r.get("source", "UNKNOWN")
        if src not in by_source:
            by_source[src] = {"total": 0, "wins": 0, "buys": 0, "sells": 0, "buy_wins": 0, "sell_wins": 0}
        by_source[src]["total"] += 1
        if r.get("is_winner"):
            by_source[src]["wins"] += 1
        sig = r.get("signal", "")
        if sig == "BUY":
            by_source[src]["buys"] += 1
            if r.get("is_winner"):
                by_source[src]["buy_wins"] += 1
        elif sig == "SELL":
            by_source[src]["sells"] += 1
            if r.get("is_winner"):
                by_source[src]["sell_wins"] += 1
    for src, d in by_source.items():
        d["accuracy_pct"] = (d["wins"] / d["total"] * 100) if d["total"] else 0
        d["buy_accuracy_pct"] = (d["buy_wins"] / d["buys"] * 100) if d["buys"] else None
        d["sell_accuracy_pct"] = (d["sell_wins"] / d["sells"] * 100) if d["sells"] else None

    # Por direção (BUY=long, SELL=short)
    by_direction = {"BUY": {"total": 0, "wins": 0}, "SELL": {"total": 0, "wins": 0}}
    for r in unique:
        sig = r.get("signal", "")
        if sig in ("BUY", "SELL"):
            by_direction[sig]["total"] += 1
            if r.get("is_winner"):
                by_direction[sig]["wins"] += 1
    for sig in ("BUY", "SELL"):
        t = by_direction[sig]["total"]
        w = by_direction[sig]["wins"]
        by_direction[sig]["accuracy_pct"] = (w / t * 100) if t else 0

    # ML
    ml_with = [r for r in unique if r.get("ml_correct") is not None]
    ml_correct = sum(1 for r in ml_with if r["ml_correct"])
    ml_total = len(ml_with)
    ml_acc = (ml_correct / ml_total * 100) if ml_total else None

    # LSTM
    lstm_with = [r for r in unique if r.get("lstm_correct") is not None]
    lstm_correct = sum(1 for r in lstm_with if r["lstm_correct"])
    lstm_total = len(lstm_with)
    lstm_acc = (lstm_correct / lstm_total * 100) if lstm_total else None

    return {
        "by_source": by_source,
        "by_direction": by_direction,
        "ml": {"accuracy_pct": ml_acc, "correct": ml_correct, "total": ml_total},
        "lstm": {"accuracy_pct": lstm_acc, "correct": lstm_correct, "total": lstm_total},
        "total_records": len(unique),
    }


def get_model_validator_metrics(evaluations: List[Dict]) -> Dict:
    """
    Calcula acurácia por modelo (ML, LSTM) sobre sinais finalizados.
    Permite ver quem acerta/erra e se vale a pena manter ou remover um modelo.

    Returns:
        Dict com ml_accuracy, lstm_accuracy, both_agree_right_pct, counts, etc.
    """
    closed = [e for e in evaluations if e.get("outcome") in ("SL_HIT", "TP1_HIT", "TP2_HIT", "EXPIRED")]
    if not closed:
        return {
            "total_closed": 0,
            "ml_accuracy": None,
            "ml_correct": 0,
            "ml_total": 0,
            "lstm_accuracy": None,
            "lstm_correct": 0,
            "lstm_total": 0,
            "both_agree_right_pct": None,
            "both_agree_total": 0,
        }

    ml_with_pred = [e for e in closed if e.get("ml_correct") is not None]
    lstm_with_vote = [e for e in closed if e.get("lstm_correct") is not None]
    ml_correct = sum(1 for e in ml_with_pred if e["ml_correct"])
    lstm_correct = sum(1 for e in lstm_with_vote if e["lstm_correct"])

    # Quando ML e LSTM ambos deram opinião (for/contra) e ambos acertaram
    both_have = [e for e in closed if e.get("ml_correct") is not None and e.get("lstm_correct") is not None]
    both_agree_right = sum(1 for e in both_have if e["ml_correct"] and e["lstm_correct"])
    both_agree_total = len(both_have)

    return {
        "total_closed": len(closed),
        "ml_accuracy": (ml_correct / len(ml_with_pred) * 100) if ml_with_pred else None,
        "ml_correct": ml_correct,
        "ml_total": len(ml_with_pred),
        "lstm_accuracy": (lstm_correct / len(lstm_with_vote) * 100) if lstm_with_vote else None,
        "lstm_correct": lstm_correct,
        "lstm_total": len(lstm_with_vote),
        "both_agree_right_pct": (both_agree_right / both_agree_total * 100) if both_agree_total else None,
        "both_agree_total": both_agree_total,
    }


def get_performance_summary(evaluations: List[Dict]) -> Dict:
    """
    Calcula métricas agregadas de performance.
    Separa sinais EXECUTADOS (real/paper) de APENAS GERADOS para comparação.
    """
    if not evaluations:
        return {}

    # Métricas globais (todos os sinais)
    all_metrics = _calc_metrics(evaluations)

    # Separar por executados vs apenas gerados
    executed = [e for e in evaluations if e.get("executed")]
    not_executed = [e for e in evaluations if not e.get("executed")]

    all_metrics["executed_count"] = len(executed)
    all_metrics["not_executed_count"] = len(not_executed)

    # Métricas separadas para sinais EXECUTADOS
    if executed:
        all_metrics["executed_metrics"] = _calc_metrics(executed)
    else:
        all_metrics["executed_metrics"] = {}

    # Métricas separadas para sinais NÃO executados (para comparar)
    if not_executed:
        all_metrics["not_executed_metrics"] = _calc_metrics(not_executed)
    else:
        all_metrics["not_executed_metrics"] = {}

    return all_metrics
