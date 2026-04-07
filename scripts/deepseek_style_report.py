"""
Relatório inferido dos logs DeepSeek: classificação aproximada Scalp / Day / Swing.

Critérios (mutuamente exclusivos por prioridade):
  SWING: timeframes 4h e 1d alinhados com o sinal (BUY→bullish, SELL→bearish).
  SCALP: 5m e 15m alinhados com o sinal (curto prazo dominante).
  DAY: restante com sinal BUY/SELL (âncora intradiário 1h típico do bot).
  NO_CLASS: NO_SIGNAL, JSON inválido ou sem dados de TF.

Uso: python scripts/deepseek_style_report.py [pasta_deepseek_logs]
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


TF_LIST = ("5m", "15m", "1h", "4h", "1d")


def extract_tf_trends(text: str) -> dict[str, str]:
    """Extrai tendências do bloco analyze_multiple_timeframes (aspas simples típicas do log)."""
    if not text:
        return {}
    trends: dict[str, str] = {}
    for tf in TF_LIST:
        # Ex.: '5m': {'trend': 'bearish'
        pat = r"['\"]" + re.escape(tf) + r"['\"]:\s*\{\s*['\"]trend['\"]:\s*['\"](\w+)['\"]"
        m = re.search(pat, text)
        if m:
            trends[tf] = m.group(1).lower()
    return trends


def extract_signal_from_response(text: str) -> dict | None:
    if not text:
        return None
    for m in re.finditer(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE):
        try:
            d = json.loads(m.group(1))
            if "signal" in d:
                return d
        except json.JSONDecodeError:
            continue
    # RunOutput: blocos ```json``` muitas vezes com JSON aninhado — falha no non-greedy.
    # Fallback: último "signal": "BUY|SELL|NO_SIGNAL" na resposta do assistente.
    matches = list(re.finditer(r'"signal"\s*:\s*"(BUY|SELL|NO_SIGNAL)"', text))
    if not matches:
        return None
    sig = matches[-1].group(1)
    return {"signal": sig}


def aligns(tf_trend: str, signal: str) -> bool:
    t = (tf_trend or "").lower()
    if signal == "BUY":
        return t == "bullish"
    if signal == "SELL":
        return t == "bearish"
    return False


def classify_style(signal: str, trends: dict[str, str]) -> str:
    if signal not in ("BUY", "SELL"):
        return "NO_CLASS"
    if not trends:
        return "NO_CLASS"
    s, l = trends.get("4h"), trends.get("1d")
    if s and l and aligns(s, signal) and aligns(l, signal):
        return "SWING"
    a, b = trends.get("5m"), trends.get("15m")
    if a and b and aligns(a, signal) and aligns(b, signal):
        return "SCALP"
    return "DAY"


def load_response_blob(path: Path) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return ""
    r = data.get("response_received")
    if isinstance(r, str):
        return r
    return str(r) if r is not None else ""


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("deepseek_logs")
    if not root.is_dir():
        print(f"Pasta não encontrada: {root}")
        sys.exit(1)

    files = list(root.rglob("*.json"))
    by_style = Counter()
    by_style_signal = defaultdict(Counter)  # style -> BUY/SELL/NO_SIGNAL
    no_tf = 0
    errors = 0

    for fp in files:
        try:
            text = load_response_blob(fp)
            sig_obj = extract_signal_from_response(text)
            trends = extract_tf_trends(text)
            if not trends:
                no_tf += 1
            signal = (sig_obj or {}).get("signal", "UNKNOWN")

            if signal not in ("BUY", "SELL"):
                by_style["NO_CLASS"] += 1
                by_style_signal["NO_CLASS"][str(signal)] += 1
                continue

            style = classify_style(signal, trends)
            if style == "NO_CLASS":
                by_style["NO_CLASS"] += 1
                by_style_signal["NO_CLASS"][signal] += 1
            else:
                by_style[style] += 1
                by_style_signal[style][signal] += 1
        except Exception:
            errors += 1

    total = len(files)
    tradable = by_style["SCALP"] + by_style["DAY"] + by_style["SWING"]

    print("=== Relatório DeepSeek (inferido por timeframes nos JSONs) ===\n")
    print(f"Pasta: {root.resolve()}")
    print(f"Total de ficheiros JSON: {total}")
    print(f"Sem bloco de timeframes detetável: {no_tf} (muitos logs antigos ou formato diferente)")
    print(f"Erros de leitura: {errors}\n")

    print("--- Distribuição estimada (apenas BUY/SELL com TF) ---")
    for k in ("SCALP", "DAY", "SWING", "NO_CLASS"):
        print(f"  {k}: {by_style[k]}")

    if tradable:
        print("\n--- % dentro dos classificados (SCALP+DAY+SWING) ---")
        for k in ("SCALP", "DAY", "SWING"):
            print(f"  {k}: {100 * by_style[k] / tradable:.1f}%")

    print("\n--- BUY vs SELL por estilo ---")
    for st in ("SCALP", "DAY", "SWING"):
        c = by_style_signal[st]
        if not sum(c.values()):
            continue
        print(f"  {st}: BUY={c['BUY']} SELL={c['SELL']}")

    print(
        "\nNota: O bot corre análises ~1h; DAY/SWING aqui reflectem *alinhamento de TF na mensagem*,"
        " não duração real da posição."
    )


if __name__ == "__main__":
    main()
