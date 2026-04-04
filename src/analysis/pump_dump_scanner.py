"""
Pump/Dump Reversal Scanner — Captura reversões em micro-cap altcoins
====================================================================

Estratégia especializada para TOP GAINERS e TOP LOSERS (pares dinâmicos).
Usa timeframe curto (5min) para detectar exaustão de pumps/dumps e entrar
na reversão com scalp rápido.

Lógica:
- GAINER com pump >15%: espera exaustão → SHORT no topo
- LOSER com dump >15%: espera exaustão → LONG no fundo

Sinais de exaustão:
1. Volume decrescente (3 candles consecutivos de volume caindo)
2. Wicks grandes contra a direção do movimento (rejeição)
3. RSI extremo no 5min (>85 para pump, <15 para dump)
4. Candle de reversão (engulfing, hammer, shooting star)
5. Divergência preço vs volume (preço sobe mas volume cai)

Risk management:
- SL tight acima do último high (SHORT) ou abaixo do último low (LONG)
- TP1 = 30-50% do retrace esperado
- Timeout: 30min máximo
- Position size: metade do normal (alto risco)
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.core.logger import get_logger

logger = get_logger(__name__)

# Configurações do scanner
MIN_PUMP_PCT = 12.0       # Variação mínima 24h para considerar pump/dump
SCAN_CANDLES = 60         # Últimos 60 candles de 5min (5 horas)
RSI_PERIOD = 7            # RSI curto para reação rápida
RSI_EXTREME_HIGH = 82     # RSI de exaustão para pump
RSI_EXTREME_LOW = 18      # RSI de exaustão para dump
MIN_EXHAUSTION_SIGNALS = 3  # Mínimo de sinais de exaustão para entrar
TIMEOUT_MINUTES = 30      # Timeout máximo do trade
POSITION_SIZE_FACTOR = 0.5  # Metade do position size normal


async def scan_for_reversal(
    symbol: str,
    mover_type: str,
    price_change_24h: float,
) -> Dict[str, Any]:
    """
    Analisa um top mover para detectar reversão de pump/dump.

    Args:
        symbol: Par (ex: EDGEUSDT)
        mover_type: "gainer" ou "loser"
        price_change_24h: Variação % nas últimas 24h

    Returns:
        Dict com signal, entry_price, stop_loss, take_profit_1, take_profit_2,
        ou signal="NO_SIGNAL" se não há setup
    """
    result = {
        "signal": "NO_SIGNAL",
        "scanner": "pump_dump_reversal",
        "symbol": symbol,
        "mover_type": mover_type,
        "price_change_24h": price_change_24h,
        "operation_type": "SCALP",
        "max_duration_hours": TIMEOUT_MINUTES / 60,
        "position_size_factor": POSITION_SIZE_FACTOR,
    }

    # Só analisar se variação 24h é significativa
    if abs(price_change_24h) < MIN_PUMP_PCT:
        result["skip_reason"] = f"Variação 24h insuficiente ({price_change_24h:.1f}% < ±{MIN_PUMP_PCT}%)"
        return result

    try:
        # Fetch candles 5min
        from src.exchange.client import BinanceClient
        async with BinanceClient() as client:
            df = await client.get_klines(symbol, "5m", limit=SCAN_CANDLES, exclude_forming=True)

        if df is None or df.empty or len(df) < 20:
            result["skip_reason"] = "Dados insuficientes (< 20 candles 5min)"
            return result

        # Calcular indicadores no 5min
        analysis = _analyze_5min_data(df, mover_type)

        if not analysis:
            result["skip_reason"] = "Erro na análise técnica 5min"
            return result

        # Verificar sinais de exaustão
        exhaustion = _detect_exhaustion(df, analysis, mover_type)

        result["exhaustion_signals"] = exhaustion["signals"]
        result["exhaustion_count"] = exhaustion["count"]
        result["exhaustion_details"] = exhaustion["details"]
        result["rsi_5min"] = analysis["rsi"]
        result["volume_trend"] = analysis["volume_trend"]
        result["atr_5min"] = analysis["atr"]

        if exhaustion["count"] < MIN_EXHAUSTION_SIGNALS:
            result["skip_reason"] = (
                f"Exaustão insuficiente: {exhaustion['count']}/{MIN_EXHAUSTION_SIGNALS} sinais. "
                f"Detalhes: {', '.join(exhaustion['details'])}"
            )
            logger.info(
                f"[PUMP_DUMP] {symbol}: Sem reversão — "
                f"{exhaustion['count']}/{MIN_EXHAUSTION_SIGNALS} sinais de exaustão. "
                f"({mover_type}, {price_change_24h:+.1f}%)"
            )
            return result

        # Calcular entry, SL, TP
        current_price = float(df["close"].iloc[-1])
        atr = analysis["atr"]
        levels = _calculate_reversal_levels(df, current_price, atr, mover_type)

        if not levels:
            result["skip_reason"] = "Não foi possível calcular níveis de reversão"
            return result

        # Montar sinal
        if mover_type == "gainer":
            signal_dir = "SELL"  # Short no topo do pump
        else:
            signal_dir = "BUY"   # Long no fundo do dump

        result["signal"] = signal_dir
        result["entry_price"] = current_price
        result["stop_loss"] = levels["stop_loss"]
        result["take_profit_1"] = levels["tp1"]
        result["take_profit_2"] = levels["tp2"]
        result["sl_method"] = levels["sl_method"]
        result["tp1_method"] = levels["tp1_method"]
        result["tp2_method"] = levels["tp2_method"]
        result["confidence"] = min(10, 5 + exhaustion["count"])  # 5-10 baseado em sinais
        result["risk_reward"] = levels["risk_reward"]
        result["reasoning"] = (
            f"Reversão de {'pump' if mover_type == 'gainer' else 'dump'}: "
            f"{exhaustion['count']} sinais de exaustão detectados no 5min. "
            f"RSI(7)={analysis['rsi']:.1f}, Volume={'decrescente' if analysis['volume_trend'] == 'declining' else analysis['volume_trend']}. "
            f"{', '.join(exhaustion['details'][:3])}"
        )

        logger.info(
            f"[PUMP_DUMP] {symbol}: {signal_dir} detectado! "
            f"Exaustão={exhaustion['count']}/{MIN_EXHAUSTION_SIGNALS}, "
            f"RSI={analysis['rsi']:.1f}, "
            f"Entry=${current_price:.4f}, SL=${levels['stop_loss']:.4f}, "
            f"TP1=${levels['tp1']:.4f}, R:R={levels['risk_reward']:.1f}"
        )

        return result

    except Exception as e:
        logger.warning(f"[PUMP_DUMP] Erro ao escanear {symbol}: {e}")
        result["skip_reason"] = f"Erro: {e}"
        return result


def _analyze_5min_data(df: pd.DataFrame, mover_type: str) -> Optional[Dict]:
    """Calcula indicadores técnicos em candles de 5min."""
    try:
        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        volume = df["volume"].values.astype(float)

        n = len(close)
        if n < 20:
            return None

        # RSI curto (7 períodos)
        rsi = _calculate_rsi(close, RSI_PERIOD)

        # ATR (7 períodos para 5min)
        atr = _calculate_atr(high, low, close, period=7)

        # Volume trend: comparar média dos últimos 5 candles vs anteriores 10
        recent_vol = np.mean(volume[-5:]) if n >= 5 else np.mean(volume)
        prev_vol = np.mean(volume[-15:-5]) if n >= 15 else np.mean(volume[:max(1, n-5)])

        if prev_vol > 0:
            vol_ratio = recent_vol / prev_vol
        else:
            vol_ratio = 1.0

        if vol_ratio < 0.7:
            volume_trend = "declining"  # Volume caindo = exaustão
        elif vol_ratio > 1.3:
            volume_trend = "increasing"
        else:
            volume_trend = "stable"

        # Candle patterns nos últimos 3 candles
        patterns = _detect_candle_patterns(df.iloc[-5:], mover_type)

        # Wicks (sombras) — indicam rejeição
        recent_candles = df.iloc[-3:]
        avg_wick_ratio = _calculate_wick_rejection(recent_candles, mover_type)

        # Preço vs Volume divergência
        # Pump: preço subindo mas volume caindo = divergência bearish
        # Dump: preço caindo mas volume caindo = divergência bullish
        price_direction = close[-1] - close[-5] if n >= 5 else 0
        price_vol_divergence = False
        if mover_type == "gainer" and price_direction > 0 and volume_trend == "declining":
            price_vol_divergence = True
        elif mover_type == "loser" and price_direction < 0 and volume_trend == "declining":
            price_vol_divergence = True

        # Momentum: variação % nos últimos 5 candles (25min)
        momentum_pct = ((close[-1] - close[-6]) / close[-6] * 100) if n >= 6 else 0

        return {
            "rsi": rsi,
            "atr": atr,
            "volume_trend": volume_trend,
            "vol_ratio": vol_ratio,
            "patterns": patterns,
            "avg_wick_ratio": avg_wick_ratio,
            "price_vol_divergence": price_vol_divergence,
            "momentum_25min": momentum_pct,
            "current_price": float(close[-1]),
            "recent_high": float(np.max(high[-10:])),
            "recent_low": float(np.min(low[-10:])),
        }

    except Exception as e:
        logger.warning(f"[PUMP_DUMP] Erro ao analisar dados 5min: {e}")
        return None


def _detect_exhaustion(
    df: pd.DataFrame, analysis: Dict, mover_type: str
) -> Dict[str, Any]:
    """
    Detecta sinais de exaustão do pump/dump.
    Retorna contagem de sinais e detalhes.
    """
    signals = []
    details = []

    rsi = analysis["rsi"]
    vol_trend = analysis["volume_trend"]
    patterns = analysis["patterns"]
    wick_ratio = analysis["avg_wick_ratio"]
    divergence = analysis["price_vol_divergence"]
    momentum = analysis["momentum_25min"]

    # 1. RSI extremo
    if mover_type == "gainer" and rsi > RSI_EXTREME_HIGH:
        signals.append("rsi_extreme")
        details.append(f"RSI(7)={rsi:.0f} > {RSI_EXTREME_HIGH} (sobrecompra extrema)")
    elif mover_type == "loser" and rsi < RSI_EXTREME_LOW:
        signals.append("rsi_extreme")
        details.append(f"RSI(7)={rsi:.0f} < {RSI_EXTREME_LOW} (sobrevenda extrema)")

    # 2. Volume decrescente (exaustão de momentum)
    if vol_trend == "declining":
        signals.append("volume_declining")
        details.append(f"Volume decrescente (ratio={analysis['vol_ratio']:.2f})")

    # 3. Wicks de rejeição (sombras contra a direção)
    if wick_ratio > 0.5:  # Wick > 50% do corpo do candle
        signals.append("wick_rejection")
        details.append(f"Wicks de rejeição (ratio={wick_ratio:.2f})")

    # 4. Candle patterns de reversão
    if patterns:
        signals.append("candle_pattern")
        details.append(f"Patterns: {', '.join(patterns)}")

    # 5. Divergência preço vs volume
    if divergence:
        signals.append("price_vol_divergence")
        if mover_type == "gainer":
            details.append("Divergência: preço subindo + volume caindo")
        else:
            details.append("Divergência: preço caindo + volume caindo")

    # 6. Momentum desacelerando (últimos 25min)
    if mover_type == "gainer" and momentum < 0.5:
        # Pump perdendo força (subindo menos de 0.5% nos últimos 25min)
        signals.append("momentum_fading")
        details.append(f"Momentum desacelerando ({momentum:+.2f}% em 25min)")
    elif mover_type == "loser" and momentum > -0.5:
        # Dump perdendo força (caindo menos de 0.5% nos últimos 25min)
        signals.append("momentum_fading")
        details.append(f"Momentum desacelerando ({momentum:+.2f}% em 25min)")

    # 7. Preço estagnando perto do extremo (consolidação no topo/fundo)
    close = df["close"].values.astype(float)
    if len(close) >= 6:
        last_6_range = (np.max(close[-6:]) - np.min(close[-6:])) / close[-6] * 100
        if last_6_range < 1.0:  # Range < 1% nos últimos 30min
            signals.append("price_stagnation")
            details.append(f"Consolidação: range={last_6_range:.2f}% em 30min")

    return {
        "signals": signals,
        "count": len(signals),
        "details": details,
    }


def _calculate_reversal_levels(
    df: pd.DataFrame, current_price: float, atr: float, mover_type: str
) -> Optional[Dict]:
    """
    Calcula SL/TP para trade de reversão.

    SHORT (gainer): SL acima do recent high + margem ATR, TP em retrace
    LONG (loser): SL abaixo do recent low - margem ATR, TP em retrace
    """
    if atr <= 0 or current_price <= 0:
        return None

    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)

    # Últimos 12 candles (1 hora) para encontrar extremos
    recent_high = float(np.max(high[-12:]))
    recent_low = float(np.min(low[-12:]))

    # Margem de segurança: 0.5 ATR acima/abaixo do extremo
    margin = atr * 0.5

    if mover_type == "gainer":
        # SHORT no topo: SL acima do high recente
        sl = recent_high + margin
        # TP1: retrace de ~40% do pump recente (do high ao preço médio)
        range_size = recent_high - recent_low
        tp1 = current_price - (range_size * 0.3)
        tp2 = current_price - (range_size * 0.5)

        # Garantir TP > 0
        if tp1 <= 0:
            tp1 = current_price * 0.97
        if tp2 <= 0:
            tp2 = current_price * 0.95

        sl_method = f"acima_high_5min(${recent_high:.4f})+ATR_margin"
        tp1_method = "retrace_30%_pump"
        tp2_method = "retrace_50%_pump"

    else:
        # LONG no fundo: SL abaixo do low recente
        sl = recent_low - margin
        # TP1: retrace de ~40% do dump recente
        range_size = recent_high - recent_low
        tp1 = current_price + (range_size * 0.3)
        tp2 = current_price + (range_size * 0.5)

        # Garantir SL > 0
        if sl <= 0:
            sl = current_price * 0.95

        sl_method = f"abaixo_low_5min(${recent_low:.4f})-ATR_margin"
        tp1_method = "retrace_30%_dump"
        tp2_method = "retrace_50%_dump"

    # Calcular R:R
    sl_dist = abs(current_price - sl)
    tp1_dist = abs(tp1 - current_price)
    rr = tp1_dist / sl_dist if sl_dist > 0 else 0

    # Rejeitar se R:R < 1.2 (mais flexível que o normal por ser scalp rápido)
    if rr < 1.2:
        logger.info(
            f"[PUMP_DUMP] R:R insuficiente: {rr:.1f} < 1.2 "
            f"(SL_dist={sl_dist:.4f}, TP1_dist={tp1_dist:.4f})"
        )
        return None

    return {
        "stop_loss": round(sl, 8),
        "tp1": round(tp1, 8),
        "tp2": round(tp2, 8),
        "sl_method": sl_method,
        "tp1_method": tp1_method,
        "tp2_method": tp2_method,
        "risk_reward": round(rr, 2),
    }


# ===== Indicadores simples (sem dependência de talib) =====

def _calculate_rsi(close: np.ndarray, period: int = 7) -> float:
    """RSI simples para 5min."""
    if len(close) < period + 1:
        return 50.0

    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    # EMA-based RSI
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))


def _calculate_atr(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 7
) -> float:
    """ATR simples para 5min."""
    if len(close) < period + 1:
        return float(np.mean(high - low)) if len(high) > 0 else 0.0

    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
    )
    # SMA do True Range
    atr = np.mean(tr[-period:])
    return float(atr)


def _detect_candle_patterns(df: pd.DataFrame, mover_type: str) -> list:
    """
    Detecta patterns de reversão nos últimos candles.
    Retorna lista de nomes de patterns encontrados.
    """
    patterns = []
    if len(df) < 2:
        return patterns

    for i in range(-2, 0):
        try:
            row = df.iloc[i]
            o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])

            body = abs(c - o)
            full_range = h - l
            if full_range <= 0:
                continue

            body_ratio = body / full_range

            if mover_type == "gainer":
                # Pump: procurar shooting star, bearish engulfing
                upper_wick = h - max(o, c)
                if upper_wick > body * 2 and body_ratio < 0.3:
                    patterns.append("shooting_star")
                if c < o and body_ratio > 0.6:
                    # Bearish candle forte
                    if i == -1:  # Último candle
                        prev = df.iloc[i - 1]
                        if float(prev["close"]) > float(prev["open"]):
                            patterns.append("bearish_engulfing")

            else:
                # Dump: procurar hammer, bullish engulfing
                lower_wick = min(o, c) - l
                if lower_wick > body * 2 and body_ratio < 0.3:
                    patterns.append("hammer")
                if c > o and body_ratio > 0.6:
                    if i == -1:
                        prev = df.iloc[i - 1]
                        if float(prev["close"]) < float(prev["open"]):
                            patterns.append("bullish_engulfing")
        except (IndexError, KeyError):
            continue

    return patterns


def _calculate_wick_rejection(df: pd.DataFrame, mover_type: str) -> float:
    """
    Calcula ratio médio de wicks de rejeição nos últimos candles.
    Para pump (gainer): upper wicks indicam rejeição no topo.
    Para dump (loser): lower wicks indicam rejeição no fundo.
    """
    ratios = []
    for _, row in df.iterrows():
        try:
            o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
            body = abs(c - o)
            if body <= 0:
                body = 0.0001 * max(abs(o), 1)

            if mover_type == "gainer":
                # Upper wick = rejeição no topo
                wick = h - max(o, c)
            else:
                # Lower wick = rejeição no fundo
                wick = min(o, c) - l

            ratios.append(wick / body if body > 0 else 0)
        except (KeyError, ZeroDivisionError):
            continue

    return float(np.mean(ratios)) if ratios else 0.0
