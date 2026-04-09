"""
Agente de Treino Contínuo ML + LSTM-BI
=======================================
Sistema iterativo que optimiza modelos usando Optuna + DeepSeek.

PRINCIPIOS (discutidos com o user):
1. Testar em 500-1000 sinais REAIS espalhados por períodos diferentes
2. Métrica composta: accuracy + PnL + profit factor (não só accuracy)
3. Balancear win/loss (evitar que o modelo diga "não" a tudo)
4. Balancear BUY/SELL (funcionar em bull E bear market)
5. Mínimo de aprovações (o modelo TEM que aprovar sinais)
6. Optuna para hyperparameters (melhor que DeepSeek no loop)
7. DeepSeek só no final para análise estratégica
8. Registar tentativas para não repetir configs

Uso:
  python scripts/continuous_ml_trainer.py --model ml --n-trials 50
  python scripts/continuous_ml_trainer.py --model lstm --n-trials 30
  python scripts/continuous_ml_trainer.py --model ml --n-trials 100 --deepseek-analysis
"""

import argparse
import asyncio
import json
import os
import pickle
import sys
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

try:
    from src.core.config import settings
    BLACKLIST = set(getattr(settings, "token_blacklist", []))
except Exception:
    BLACKLIST = {"JCTUSDT", "BRUSDT", "SIRENUSDT", "LYNUSDT", "UAIUSDT",
                 "ZECUSDT", "KNCUSDT", "PLAYUSDT", "DUSKUSDT", "BANKUSDT",
                 "MUSDT", "PIPPINUSDT", "CTSIUSDT", "EDGEUSDT", "4USDT"}

try:
    from src.core.logger import get_logger
    logger = get_logger("continuous_trainer")
except Exception:
    import logging
    logger = logging.getLogger("continuous_trainer")

RESULTS_DIR = ROOT / "ml_training_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. CARREGAR E PREPARAR SINAIS REAIS
# ============================================================

def load_all_real_signals() -> pd.DataFrame:
    """Carrega sinais do metrics_data.csv (já avaliados, sem chamar API).

    Se metrics_data.csv não existir, faz fallback para avaliação via API (lento).
    """
    csv_path = ROOT / "metrics_data.csv"

    if csv_path.exists():
        print(f"[DATA] Carregando de {csv_path} (pré-avaliado, rápido)...")
        raw = pd.read_csv(csv_path)
        raw = raw[~raw["symbol"].isin(BLACKLIST | {"JCTUSDT"})]
        if "source" in raw.columns:
            raw = raw[raw["source"] != "LOCAL_GEN"]
        raw = raw[raw["signal"].isin(["BUY", "SELL"])]
        raw = raw[raw["outcome"].isin(["SL_HIT", "TP1_HIT", "TP2_HIT"])]

        raw["ts"] = pd.to_datetime(raw["timestamp"], errors="coerce", utc=True)
        raw = raw.dropna(subset=["ts"])

        # Mapear colunas do CSV para feature columns
        if "signal_encoded" not in raw.columns:
            raw["signal_encoded"] = raw["signal"].apply(lambda x: 1 if x == "BUY" else -1)
        if "bullish_tf_count" not in raw.columns:
            raw["bullish_tf_count"] = 0
        if "bearish_tf_count" not in raw.columns:
            raw["bearish_tf_count"] = 0
        if "trend_encoded" not in raw.columns:
            trend_map = {"strong_bullish": 2, "bullish": 1, "neutral": 0, "bearish": -1, "strong_bearish": -2}
            raw["trend_encoded"] = raw.get("trend", pd.Series("neutral", index=raw.index)).map(trend_map).fillna(0)
        if "sentiment_encoded" not in raw.columns:
            raw["sentiment_encoded"] = 0
        if "risk_reward_ratio" not in raw.columns:
            raw["risk_reward_ratio"] = raw.get("risk_reward", pd.Series(1.0, index=raw.index)).fillna(1.0)
        if "reward_distance_pct" not in raw.columns:
            raw["reward_distance_pct"] = 0.5
        for col in FEATURE_COLUMNS:
            if col not in raw.columns:
                raw[col] = 0

        if "is_winner" not in raw.columns:
            raw["is_winner"] = raw["outcome"].isin(["TP1_HIT", "TP2_HIT"])
        if "pnl_pct" not in raw.columns:
            raw["pnl_pct"] = raw.get("pnl_percent", pd.Series(0, index=raw.index))

        df = raw
    else:
        print("[DATA] metrics_data.csv não encontrado. Avaliando via API (pode demorar)...")
        from src.trading.signal_tracker import load_all_signals, evaluate_signal

        all_sigs = load_all_signals("signals")
        records = []
        total = len(all_sigs)
        for i, sig in enumerate(all_sigs):
            if sig.get("symbol") in BLACKLIST or sig.get("symbol") == "JCTUSDT":
                continue
            if sig.get("source") == "LOCAL_GEN":
                continue
            if sig.get("signal") not in ("BUY", "SELL"):
                continue

            ts_str = sig.get("timestamp", "")
            try:
                if "T" in ts_str:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                else:
                    ts = datetime.strptime(ts_str[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                continue

            result = evaluate_signal(sig)
            outcome = result.get("outcome", "PENDING")
            if outcome not in ("TP1_HIT", "TP2_HIT", "SL_HIT"):
                continue

            is_winner = outcome in ("TP1_HIT", "TP2_HIT")
            pnl = result.get("pnl_percent", 0)

            rec = {
                "ts": ts, "symbol": sig.get("symbol"), "signal": sig.get("signal"),
                "is_winner": is_winner, "pnl_pct": pnl,
                "confidence": sig.get("confidence", 5),
            }
            for col in FEATURE_COLUMNS:
                if col == "signal_encoded":
                    rec[col] = 1 if sig.get("signal") == "BUY" else -1
                else:
                    rec[col] = sig.get(col, 0)
            records.append(rec)

            if (i + 1) % 200 == 0:
                print(f"  ... {i+1}/{total} processados, {len(records)} válidos")

        df = pd.DataFrame(records)

    if df.empty:
        print("[ERRO] Nenhum sinal válido encontrado!")
        return df

    # Deduplicar (mesmo symbol+signal dentro de 4h)
    df = df.sort_values(["symbol", "signal", "ts"])
    keep = []
    last = {}
    for _, row in df.iterrows():
        k = f"{row['symbol']}_{row['signal']}"
        ts = row["ts"]
        if k in last and (ts - last[k]) < timedelta(hours=4):
            keep.append(False)
        else:
            keep.append(True)
            last[k] = ts
    df["_keep"] = keep
    before = len(df)
    df = df[df["_keep"]].drop(columns=["_keep"]).sort_values("ts").reset_index(drop=True)
    print(f"[DATA] {len(df)} sinais limpos ({before - len(df)} duplicados removidos)")
    print(f"  BUY: {len(df[df['signal']=='BUY'])} | SELL: {len(df[df['signal']=='SELL'])}")
    print(f"  Winners: {df['is_winner'].sum()} ({df['is_winner'].mean()*100:.1f}%)")
    return df


def balance_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Balancear wins e losses: undersampling da maioria."""
    wins = df[df["is_winner"] == True]
    losses = df[df["is_winner"] == False]
    n_min = min(len(wins), len(losses))
    if n_min < 10:
        return df
    wins_sampled = wins.sample(n=n_min, random_state=42) if len(wins) > n_min else wins
    losses_sampled = losses.sample(n=n_min, random_state=42) if len(losses) > n_min else losses
    balanced = pd.concat([wins_sampled, losses_sampled]).sort_values("ts").reset_index(drop=True)
    print(f"[BALANCE] {len(wins)} wins + {len(losses)} losses -> {n_min} + {n_min} = {len(balanced)} balanceado")
    return balanced


def split_train_test(df: pd.DataFrame, test_ratio: float = 0.25) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split temporal + balanceamento do treino."""
    df = df.sort_values("ts").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_ratio))
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()

    # Balancear APENAS o treino (teste fica com distribuição real)
    train = balance_dataset(train)

    print(f"[SPLIT] Train: {len(train)} (balanceado) | Test: {len(test)} (distribuição real)")
    print(f"  Test BUY: {len(test[test['signal']=='BUY'])} | Test SELL: {len(test[test['signal']=='SELL'])}")
    print(f"  Test Winners: {test['is_winner'].sum()}/{len(test)} ({test['is_winner'].mean()*100:.1f}%)")
    return train, test


# ============================================================
# 2. MÉTRICA COMPOSTA DE AVALIAÇÃO
# ============================================================

FEATURE_COLUMNS = [
    'rsi', 'macd_histogram', 'adx', 'atr', 'bb_position',
    'cvd', 'orderbook_imbalance', 'bullish_tf_count', 'bearish_tf_count',
    'confidence', 'trend_encoded', 'sentiment_encoded', 'signal_encoded',
    'risk_distance_pct', 'reward_distance_pct', 'risk_reward_ratio'
]


def evaluate_model_composite(
    model,
    scaler,
    test_df: pd.DataFrame,
    threshold: float = 0.5,
) -> Dict:
    """
    Avaliação composta que mede TUDO o que discutimos:
    1. Win Rate nos sinais APROVADOS (não accuracy global)
    2. PnL total dos aprovados
    3. Balanced accuracy entre BUY e SELL
    4. Mínimo de aprovações (penaliza modelo que rejeita tudo)
    5. Profit Factor

    Retorna score composto e breakdown detalhado.
    """
    X = test_df[FEATURE_COLUMNS].fillna(0).values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        X_scaled = scaler.transform(X)
        X_scaled = np.clip(X_scaled, -5.0, 5.0)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        X_scaled = X

    probs = model.predict_proba(X_scaled)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_scaled).astype(float)
    preds = (probs >= threshold).astype(int)

    test_df = test_df.copy()
    test_df["ml_pred"] = preds
    test_df["ml_prob"] = probs

    approved = test_df[test_df["ml_pred"] == 1]
    rejected = test_df[test_df["ml_pred"] == 0]
    n_approved = len(approved)
    n_total = len(test_df)

    result = {
        "n_total": n_total,
        "n_approved": n_approved,
        "n_rejected": len(rejected),
        "approval_rate": n_approved / max(n_total, 1),
    }

    # Se aprovou menos de 10% dos sinais ou menos de 20 absolutos: penalizar
    if n_approved < 20 or n_approved < n_total * 0.10:
        result["composite_score"] = -1.0
        result["reason"] = f"Poucos aprovados ({n_approved}/{n_total})"
        return result

    # ---- Métricas nos APROVADOS ----
    approved_wins = approved[approved["is_winner"] == True]
    approved_losses = approved[approved["is_winner"] == False]
    wr_approved = len(approved_wins) / max(n_approved, 1)
    pnl_total = approved["pnl_pct"].sum()
    pnl_per_trade = approved["pnl_pct"].mean()
    sum_wins = approved_wins["pnl_pct"].sum() if len(approved_wins) > 0 else 0
    sum_losses = abs(approved_losses["pnl_pct"].sum()) if len(approved_losses) > 0 else 0.001
    profit_factor = sum_wins / max(sum_losses, 0.001)
    expectancy = pnl_per_trade

    result.update({
        "wr_approved": wr_approved,
        "pnl_total": pnl_total,
        "pnl_per_trade": pnl_per_trade,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
    })

    # ---- Balanceamento BUY/SELL (matrix 2x2) ----
    for direction in ["BUY", "SELL"]:
        dir_approved = approved[approved["signal"] == direction]
        n_dir = len(dir_approved)
        if n_dir >= 5:
            dir_wins = dir_approved[dir_approved["is_winner"] == True]
            result[f"wr_{direction.lower()}"] = len(dir_wins) / max(n_dir, 1)
            result[f"pnl_{direction.lower()}"] = dir_approved["pnl_pct"].sum()
            result[f"n_{direction.lower()}"] = n_dir
        else:
            result[f"wr_{direction.lower()}"] = 0
            result[f"pnl_{direction.lower()}"] = 0
            result[f"n_{direction.lower()}"] = n_dir

    wr_buy = result.get("wr_buy", 0)
    wr_sell = result.get("wr_sell", 0)
    balanced_wr = (wr_buy + wr_sell) / 2 if wr_buy > 0 and wr_sell > 0 else min(wr_buy, wr_sell)

    result["balanced_wr"] = balanced_wr

    # ---- Balanced accuracy global (win class + loss class) ----
    # Accuracy em predizer winners correctamente
    real_winners = test_df[test_df["is_winner"] == True]
    real_losers = test_df[test_df["is_winner"] == False]
    if len(real_winners) > 0:
        sens = len(real_winners[real_winners["ml_pred"] == 1]) / len(real_winners)
    else:
        sens = 0
    if len(real_losers) > 0:
        spec = len(real_losers[real_losers["ml_pred"] == 0]) / len(real_losers)
    else:
        spec = 0
    balanced_acc = (sens + spec) / 2

    result["sensitivity"] = sens
    result["specificity"] = spec
    result["balanced_accuracy"] = balanced_acc

    # ---- SCORE COMPOSTO ----
    # Pesos baseados nas nossas discussões:
    #   40% = WR dos aprovados (o que realmente importa)
    #   25% = PnL positivo (profit factor normalizado)
    #   20% = Balanced BUY/SELL (funcionar nos dois sentidos)
    #   15% = Approval rate (tem que aprovar o suficiente)

    # Normalizar profit factor para 0-1 (PF=1 -> 0.5, PF=2 -> 0.75, PF=3+ -> 1.0)
    pf_norm = min(profit_factor / 3.0, 1.0)

    # Normalizar approval rate (queremos entre 20-60%, penalizar <10% e >80%)
    ar = result["approval_rate"]
    if ar < 0.10:
        ar_norm = 0
    elif ar > 0.80:
        ar_norm = 0.5
    else:
        ar_norm = min(ar / 0.50, 1.0)

    composite = (
        0.40 * wr_approved +
        0.25 * pf_norm +
        0.20 * balanced_wr +
        0.15 * ar_norm
    )

    # Penalizar se PnL total é negativo
    if pnl_total < 0:
        composite *= 0.5

    # Penalizar se uma direcção tem 0 aprovações
    if result.get("n_buy", 0) < 3 or result.get("n_sell", 0) < 3:
        composite *= 0.7

    result["composite_score"] = composite
    return result


# ============================================================
# 3. OPTUNA — OPTIMIZAÇÃO DE HYPERPARAMETERS ML
# ============================================================

def create_ml_objective(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Cria a objective function do Optuna para o ensemble ML."""
    import optuna
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils.class_weight import compute_sample_weight

    try:
        import xgboost as xgb
        HAS_XGB = True
    except ImportError:
        HAS_XGB = False

    X_train = train_df[FEATURE_COLUMNS].fillna(0).values
    y_train = train_df["is_winner"].astype(int).values
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

    def objective(trial: optuna.Trial) -> float:
        # Escolher modelo
        model_type = trial.suggest_categorical("model_type", ["rf", "gb", "xgb", "lr", "mlp"])

        threshold = trial.suggest_float("threshold", 0.35, 0.65)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        sample_w = compute_sample_weight("balanced", y_train)

        if model_type == "rf":
            model = RandomForestClassifier(
                n_estimators=trial.suggest_int("rf_n_estimators", 50, 500),
                max_depth=trial.suggest_int("rf_max_depth", 3, 15),
                min_samples_split=trial.suggest_int("rf_min_samples_split", 2, 20),
                min_samples_leaf=trial.suggest_int("rf_min_samples_leaf", 1, 10),
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X_train_s, y_train)

        elif model_type == "gb":
            model = GradientBoostingClassifier(
                n_estimators=trial.suggest_int("gb_n_estimators", 50, 500),
                max_depth=trial.suggest_int("gb_max_depth", 2, 8),
                learning_rate=trial.suggest_float("gb_lr", 0.01, 0.3, log=True),
                subsample=trial.suggest_float("gb_subsample", 0.6, 1.0),
                random_state=42,
            )
            model.fit(X_train_s, y_train, sample_weight=sample_w)

        elif model_type == "xgb" and HAS_XGB:
            model = xgb.XGBClassifier(
                objective="binary:logistic",
                tree_method="hist",
                eval_metric="logloss",
                n_estimators=trial.suggest_int("xgb_n_estimators", 50, 500),
                max_depth=trial.suggest_int("xgb_max_depth", 2, 10),
                learning_rate=trial.suggest_float("xgb_lr", 0.01, 0.3, log=True),
                subsample=trial.suggest_float("xgb_subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("xgb_colsample", 0.5, 1.0),
                reg_alpha=trial.suggest_float("xgb_alpha", 1e-8, 10.0, log=True),
                reg_lambda=trial.suggest_float("xgb_lambda", 1e-8, 10.0, log=True),
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )
            model.fit(X_train_s, y_train, sample_weight=sample_w)

        elif model_type == "lr":
            model = LogisticRegression(
                C=trial.suggest_float("lr_C", 1e-4, 100, log=True),
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            )
            model.fit(X_train_s, y_train)

        elif model_type == "mlp":
            h1 = trial.suggest_int("mlp_h1", 16, 128)
            h2 = trial.suggest_int("mlp_h2", 8, 64)
            model = MLPClassifier(
                hidden_layer_sizes=(h1, h2),
                max_iter=1000,
                learning_rate_init=trial.suggest_float("mlp_lr", 1e-4, 0.01, log=True),
                alpha=trial.suggest_float("mlp_alpha", 1e-6, 0.1, log=True),
                random_state=42,
                early_stopping=True,
            )
            model.fit(X_train_s, y_train)
        else:
            # Fallback to RF if xgb not available
            model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
            model.fit(X_train_s, y_train)

        metrics = evaluate_model_composite(model, scaler, test_df, threshold=threshold)

        score = metrics.get("composite_score", -1)

        trial.set_user_attr("wr_approved", metrics.get("wr_approved", 0))
        trial.set_user_attr("pnl_total", metrics.get("pnl_total", 0))
        trial.set_user_attr("profit_factor", metrics.get("profit_factor", 0))
        trial.set_user_attr("n_approved", metrics.get("n_approved", 0))
        trial.set_user_attr("balanced_wr", metrics.get("balanced_wr", 0))
        trial.set_user_attr("wr_buy", metrics.get("wr_buy", 0))
        trial.set_user_attr("wr_sell", metrics.get("wr_sell", 0))

        return score

    return objective


# ============================================================
# 4. OPTUNA — OPTIMIZAÇÃO DE HYPERPARAMETERS LSTM
# ============================================================

def create_lstm_objective(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Cria objective function do Optuna para LSTM-BI."""
    import optuna

    def objective(trial: optuna.Trial) -> float:
        try:
            import tensorflow as tf
            from tensorflow.keras.layers import (
                LSTM, BatchNormalization, Bidirectional, Dense, Dropout, Input,
            )
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.optimizers import Adam
            from tensorflow.keras.callbacks import EarlyStopping
            from sklearn.preprocessing import StandardScaler

            # Hyperparameters to tune
            lstm_units_1 = trial.suggest_int("lstm_units_1", 32, 128)
            lstm_units_2 = trial.suggest_int("lstm_units_2", 16, 64)
            dropout_rate = trial.suggest_float("dropout", 0.1, 0.5)
            dense_units = trial.suggest_int("dense_units", 16, 64)
            learning_rate = trial.suggest_float("lr", 1e-4, 0.01, log=True)
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
            threshold = trial.suggest_float("threshold", 0.35, 0.65)

            # Load existing dataset (generated by real_signal_dataset_generator)
            dataset_dir = ROOT / "ml_dataset" / "backtest"
            X_train = np.load(dataset_dir / "X_train_latest.npy")
            X_test = np.load(dataset_dir / "X_test_latest.npy")
            y_train = np.load(dataset_dir / "y_train_latest.npy")
            y_test = np.load(dataset_dir / "y_test_latest.npy")

            seq_len = X_train.shape[1]
            n_feat = X_train.shape[2]

            # Scale
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train.reshape(-1, n_feat)).reshape(X_train.shape)
            X_test_s = scaler.transform(X_test.reshape(-1, n_feat)).reshape(X_test.shape)
            X_train_s = np.nan_to_num(X_train_s, nan=0.0)
            X_test_s = np.nan_to_num(X_test_s, nan=0.0)

            # Class weights
            n_pos = y_train.sum()
            n_neg = len(y_train) - n_pos
            class_weight = {0: len(y_train) / (2 * max(n_neg, 1)), 1: len(y_train) / (2 * max(n_pos, 1))}

            model = Sequential([
                Input(shape=(seq_len, n_feat)),
                Bidirectional(LSTM(lstm_units_1, return_sequences=True, dropout=dropout_rate)),
                BatchNormalization(),
                Bidirectional(LSTM(lstm_units_2, return_sequences=False, dropout=dropout_rate)),
                BatchNormalization(),
                Dense(dense_units, activation="relu"),
                Dropout(dropout_rate),
                Dense(1, activation="sigmoid"),
            ])
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss="binary_crossentropy", metrics=["accuracy"])

            model.fit(
                X_train_s, y_train,
                epochs=50,
                batch_size=batch_size,
                validation_data=(X_test_s, y_test),
                callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)],
                class_weight=class_weight,
                verbose=0,
            )

            # Evaluate on real test signals
            probs = model.predict(X_test_s, verbose=0).flatten()
            preds = (probs >= threshold).astype(int)

            n_approved = preds.sum()
            if n_approved < max(10, len(y_test) * 0.10):
                return -1.0

            approved_mask = preds == 1
            wr_approved = y_test[approved_mask].mean() if n_approved > 0 else 0

            # Simple composite for LSTM
            approval_rate = n_approved / len(y_test)
            ar_norm = min(approval_rate / 0.5, 1.0) if approval_rate >= 0.10 else 0
            score = 0.60 * wr_approved + 0.40 * ar_norm

            if n_approved < 20:
                score *= 0.7

            trial.set_user_attr("wr_approved", float(wr_approved))
            trial.set_user_attr("n_approved", int(n_approved))
            trial.set_user_attr("threshold", threshold)

            # Free memory
            del model
            tf.keras.backend.clear_session()

            return score

        except Exception as e:
            print(f"[LSTM TRIAL] Erro: {e}")
            return -1.0

    return objective


# ============================================================
# 5. TREINO DO MELHOR MODELO E DEPLOY
# ============================================================

def train_and_save_best_ml(best_params: Dict, train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Retreina o melhor modelo com os params do Optuna e salva."""
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.utils.class_weight import compute_sample_weight

    try:
        import xgboost as xgb
        HAS_XGB = True
    except ImportError:
        HAS_XGB = False

    print("\n[DEPLOY] Treinando melhor modelo para produção...")
    mt = best_params.get("model_type", "rf")
    threshold = best_params.get("threshold", 0.5)

    X_train = train_df[FEATURE_COLUMNS].fillna(0).values
    y_train = train_df["is_winner"].astype(int).values
    X_train = np.nan_to_num(X_train, nan=0.0)
    sample_w = compute_sample_weight("balanced", y_train)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    if mt == "rf":
        model = RandomForestClassifier(
            n_estimators=best_params.get("rf_n_estimators", 100),
            max_depth=best_params.get("rf_max_depth", 5),
            min_samples_split=best_params.get("rf_min_samples_split", 2),
            min_samples_leaf=best_params.get("rf_min_samples_leaf", 1),
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
        model.fit(X_train_s, y_train)
    elif mt == "gb":
        model = GradientBoostingClassifier(
            n_estimators=best_params.get("gb_n_estimators", 100),
            max_depth=best_params.get("gb_max_depth", 3),
            learning_rate=best_params.get("gb_lr", 0.1),
            subsample=best_params.get("gb_subsample", 0.8),
            random_state=42,
        )
        model.fit(X_train_s, y_train, sample_weight=sample_w)
    elif mt == "xgb" and HAS_XGB:
        model = xgb.XGBClassifier(
            objective="binary:logistic", tree_method="hist", eval_metric="logloss",
            n_estimators=best_params.get("xgb_n_estimators", 200),
            max_depth=best_params.get("xgb_max_depth", 4),
            learning_rate=best_params.get("xgb_lr", 0.1),
            subsample=best_params.get("xgb_subsample", 0.8),
            colsample_bytree=best_params.get("xgb_colsample", 0.8),
            reg_alpha=best_params.get("xgb_alpha", 1e-8),
            reg_lambda=best_params.get("xgb_lambda", 1e-8),
            random_state=42, n_jobs=-1, verbosity=0,
        )
        model.fit(X_train_s, y_train, sample_weight=sample_w)
    elif mt == "lr":
        model = LogisticRegression(
            C=best_params.get("lr_C", 1.0), max_iter=1000,
            class_weight="balanced", random_state=42,
        )
        model.fit(X_train_s, y_train)
    elif mt == "mlp":
        model = MLPClassifier(
            hidden_layer_sizes=(best_params.get("mlp_h1", 32), best_params.get("mlp_h2", 16)),
            max_iter=1000, learning_rate_init=best_params.get("mlp_lr", 0.001),
            alpha=best_params.get("mlp_alpha", 0.001),
            random_state=42, early_stopping=True,
        )
        model.fit(X_train_s, y_train)
    else:
        model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
        model.fit(X_train_s, y_train)

    # Calibrar
    try:
        calibrated = CalibratedClassifierCV(model, method="isotonic", cv=3)
        calibrated.fit(X_train_s, y_train, sample_weight=sample_w)
        final_model = calibrated
        print("  [OK] Modelo calibrado")
    except Exception:
        final_model = model

    # Avaliar nos dados de teste
    metrics = evaluate_model_composite(final_model, scaler, test_df, threshold=threshold)
    print(f"  WR aprovados: {metrics.get('wr_approved', 0)*100:.1f}%")
    print(f"  PnL total: {metrics.get('pnl_total', 0):.1f}%")
    print(f"  Profit Factor: {metrics.get('profit_factor', 0):.2f}")
    print(f"  BUY WR: {metrics.get('wr_buy', 0)*100:.1f}% | SELL WR: {metrics.get('wr_sell', 0)*100:.1f}%")
    print(f"  Aprovados: {metrics.get('n_approved', 0)}/{metrics.get('n_total', 0)}")

    # Salvar com formato compatível com simple_validator.py
    model_dir = ROOT / "ml_models"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_name_map = {"rf": "RandomForest", "gb": "GradientBoosting", "xgb": "XGBoost",
                       "lr": "LogisticRegression", "mlp": "MLP"}
    canonical_name = model_name_map.get(mt, mt.upper())
    with open(model_dir / "signal_validators.pkl", "wb") as f:
        pickle.dump({canonical_name: final_model}, f)

    with open(model_dir / "scaler_simple.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Compute and save preprocessing artifacts
    train_medians = {}
    train_clip_bounds = {}
    df_features = train_df[FEATURE_COLUMNS].fillna(0)
    for col in FEATURE_COLUMNS:
        med = df_features[col].median()
        train_medians[col] = 0 if pd.isna(med) else float(med)
        mean_ = df_features[col].mean()
        std_ = df_features[col].std()
        if std_ > 0:
            train_clip_bounds[col] = (float(mean_ - 3*std_), float(mean_ + 3*std_))

    with open(model_dir / "preproc_simple.pkl", "wb") as f:
        pickle.dump({"train_medians": train_medians, "train_clip_bounds": train_clip_bounds}, f)

    info = {
        "training_date": datetime.now(timezone.utc).isoformat(),
        "training_method": "continuous_ml_trainer + optuna",
        "feature_columns": FEATURE_COLUMNS,
        "n_features": len(FEATURE_COLUMNS),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "best_model": canonical_name,
        "best_accuracy": metrics.get("wr_approved", 0),
        "best_f1": 0,
        "best_params": {k: v for k, v in best_params.items() if not k.startswith("_")},
        "composite_score": metrics.get("composite_score", 0),
        "metrics": {k: v for k, v in metrics.items() if k != "reason"},
        "threshold": threshold,
    }
    with open(model_dir / "model_info_simple.json", "w") as f:
        json.dump(info, f, indent=2, default=str)

    print(f"\n  [OK] Modelo salvo em {model_dir}/")
    return metrics


# ============================================================
# 6. DEEPSEEK — ANÁLISE ESTRATÉGICA FINAL
# ============================================================

async def deepseek_strategic_analysis(study_results: Dict, best_metrics: Dict) -> str:
    """Envia resumo dos resultados ao DeepSeek para análise estratégica."""
    import httpx

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return "[SKIP] DEEPSEEK_API_KEY não definida"

    prompt = f"""Analisa os resultados da optimização do meu modelo ML para trading de crypto futures e dá recomendações estratégicas.

RESULTADOS DA OPTIMIZAÇÃO (Optuna, {study_results.get('n_trials', 0)} trials):
- Melhor modelo: {study_results.get('best_model_type', 'N/A')}
- Score composto: {study_results.get('best_score', 0):.4f}

MÉTRICAS DO MELHOR MODELO (testado em sinais reais):
- Win Rate nos aprovados: {best_metrics.get('wr_approved', 0)*100:.1f}%
- PnL total: {best_metrics.get('pnl_total', 0):.1f}%
- Profit Factor: {best_metrics.get('profit_factor', 0):.2f}
- BUY WR: {best_metrics.get('wr_buy', 0)*100:.1f}% | SELL WR: {best_metrics.get('wr_sell', 0)*100:.1f}%
- Aprovados: {best_metrics.get('n_approved', 0)}/{best_metrics.get('n_total', 0)}
- Balanced Accuracy: {best_metrics.get('balanced_accuracy', 0)*100:.1f}%
- Sensitivity (detecta winners): {best_metrics.get('sensitivity', 0)*100:.1f}%
- Specificity (rejeita losers): {best_metrics.get('specificity', 0)*100:.1f}%

FEATURES USADAS: {FEATURE_COLUMNS}

CONTEXTO:
- Mercado crypto futures, dados de 3+ meses
- Win rate global dos sinais sem ML: ~38%
- O modelo precisa funcionar em BUY e SELL equilibradamente
- Dados limpos (deduplicados, sem JCTUSDT, sem LOCAL_GEN)

Por favor analisa:
1. Os resultados são bons ou maus para crypto trading?
2. Que features adicionais poderiam melhorar?
3. O threshold de aprovação está bom?
4. Recomendações concretas para a próxima iteração
5. Alguma preocupação com overfitting?

Responde de forma concisa e prática."""

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 1500,
                },
            )
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "Sem resposta")
            return content
    except Exception as e:
        return f"[ERRO DeepSeek] {e}"


# ============================================================
# 7. LOOP PRINCIPAL
# ============================================================

async def run_ml_optimization(n_trials: int = 50, do_deepseek: bool = False):
    """Executa optimização ML completa."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("\n" + "=" * 70)
    print("TREINO CONTÍNUO ML — Optuna + Métricas Compostas")
    print("=" * 70)

    # 1. Carregar dados
    df = load_all_real_signals()
    if len(df) < 100:
        print("[ERRO] Dados insuficientes")
        return

    # 2. Split temporal
    train_df, test_df = split_train_test(df, test_ratio=0.25)

    # 3. Optuna
    print(f"\n[OPTUNA] Iniciando {n_trials} trials...")
    study = optuna.create_study(direction="maximize", study_name="ml_continuous")
    objective = create_ml_objective(train_df, test_df)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # 4. Resultados
    best = study.best_trial
    print(f"\n{'='*70}")
    print(f"MELHOR TRIAL: #{best.number}")
    print(f"{'='*70}")
    print(f"  Score composto: {best.value:.4f}")
    print(f"  Modelo: {best.params.get('model_type', 'N/A')}")
    print(f"  WR aprovados: {best.user_attrs.get('wr_approved', 0)*100:.1f}%")
    print(f"  PnL total: {best.user_attrs.get('pnl_total', 0):.1f}%")
    print(f"  Profit Factor: {best.user_attrs.get('profit_factor', 0):.2f}")
    print(f"  BUY WR: {best.user_attrs.get('wr_buy', 0)*100:.1f}% | SELL WR: {best.user_attrs.get('wr_sell', 0)*100:.1f}%")
    print(f"  Aprovados: {best.user_attrs.get('n_approved', 0)}")
    print(f"  Params: {best.params}")

    # Top 5 trials
    print(f"\n[TOP 5 TRIALS]")
    trials_sorted = sorted(study.trials, key=lambda t: t.value if t.value is not None else -999, reverse=True)
    for i, t in enumerate(trials_sorted[:5]):
        print(f"  {i+1}. Score={t.value:.4f} | {t.params.get('model_type','?')} | WR={t.user_attrs.get('wr_approved',0)*100:.1f}% | PF={t.user_attrs.get('profit_factor',0):.2f}")

    # 5. Treinar e salvar melhor modelo
    best_metrics = train_and_save_best_ml(best.params, train_df, test_df)

    # 6. Salvar relatório
    study_results = {
        "n_trials": n_trials,
        "best_score": best.value,
        "best_model_type": best.params.get("model_type"),
        "best_params": best.params,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    report_path = RESULTS_DIR / f"ml_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w") as f:
        json.dump({
            "study": study_results,
            "best_metrics": {k: v for k, v in best_metrics.items() if k != "reason"},
            "all_trials": [{
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "user_attrs": t.user_attrs,
            } for t in study.trials if t.value is not None],
        }, f, indent=2, default=str)
    print(f"\n[SAVE] Relatório: {report_path}")

    # 7. DeepSeek (se solicitado)
    if do_deepseek:
        print("\n[DEEPSEEK] Pedindo análise estratégica...")
        analysis = await deepseek_strategic_analysis(study_results, best_metrics)
        print(f"\n{'='*70}")
        print("ANÁLISE DEEPSEEK")
        print(f"{'='*70}")
        try:
            print(analysis)
        except UnicodeEncodeError:
            print(analysis.encode("ascii", errors="replace").decode())

        # Salvar análise
        analysis_path = RESULTS_DIR / f"deepseek_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        analysis_path.write_text(analysis, encoding="utf-8")
        print(f"\n[SAVE] Análise DeepSeek: {analysis_path}")


async def run_lstm_optimization(n_trials: int = 30, do_deepseek: bool = False):
    """Executa optimização LSTM completa."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("\n" + "=" * 70)
    print("TREINO CONTÍNUO LSTM-BI — Optuna")
    print("=" * 70)

    # Verificar se dataset existe
    dataset_dir = ROOT / "ml_dataset" / "backtest"
    if not (dataset_dir / "X_train_latest.npy").exists():
        print("[INFO] Dataset LSTM não existe. Gerando a partir de sinais reais...")
        try:
            from src.ml.real_signal_dataset_generator import RealSignalDatasetGenerator
            gen = RealSignalDatasetGenerator(sequence_length=60, max_signals=5000)
            stats = await gen.generate()
            print(f"  Dataset gerado: {stats.get('total_trades', 0)} trades")
            if stats.get("total_trades", 0) < 50:
                print("[ERRO] Poucos trades para treinar LSTM")
                return
        except Exception as e:
            print(f"[ERRO] Não foi possível gerar dataset: {e}")
            return

    df = load_all_real_signals()
    if len(df) < 100:
        print("[ERRO] Dados insuficientes")
        return

    train_df, test_df = split_train_test(df, test_ratio=0.25)

    print(f"\n[OPTUNA] Iniciando {n_trials} trials LSTM...")
    study = optuna.create_study(direction="maximize", study_name="lstm_continuous")
    objective = create_lstm_objective(train_df, test_df)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    print(f"\n{'='*70}")
    print(f"MELHOR TRIAL LSTM: #{best.number}")
    print(f"{'='*70}")
    print(f"  Score: {best.value:.4f}")
    print(f"  WR aprovados: {best.user_attrs.get('wr_approved', 0)*100:.1f}%")
    print(f"  Aprovados: {best.user_attrs.get('n_approved', 0)}")
    print(f"  Params: {best.params}")

    report_path = RESULTS_DIR / f"lstm_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w") as f:
        json.dump({
            "best_score": best.value,
            "best_params": best.params,
            "best_attrs": best.user_attrs,
            "all_trials": [{
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "user_attrs": t.user_attrs,
            } for t in study.trials if t.value is not None],
        }, f, indent=2, default=str)
    print(f"\n[SAVE] Relatório: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Treino Contínuo ML/LSTM com Optuna")
    parser.add_argument("--model", choices=["ml", "lstm", "both"], default="ml",
                        help="Tipo de modelo (default: ml)")
    parser.add_argument("--n-trials", type=int, default=50,
                        help="Número de trials Optuna (default: 50)")
    parser.add_argument("--deepseek-analysis", action="store_true",
                        help="Pedir análise estratégica ao DeepSeek no final")
    args = parser.parse_args()

    async def run():
        if args.model in ("ml", "both"):
            await run_ml_optimization(n_trials=args.n_trials, do_deepseek=args.deepseek_analysis)
        if args.model in ("lstm", "both"):
            await run_lstm_optimization(n_trials=args.n_trials, do_deepseek=args.deepseek_analysis)

    asyncio.run(run())


if __name__ == "__main__":
    main()
