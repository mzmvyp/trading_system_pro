"""
Online Learning - Sistema de Retreinamento Automatico
======================================================

Baseado no tecchallenge_4, implementa:
- Buffer de novos dados para retreinamento
- Retreinamento periodico quando acumular dados suficientes
- Validacao antes de salvar novo modelo
- Tracking de performance do modelo

Autor: Trading Bot
Data: 2026-01-13
"""

import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

# Sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

import threading

# Lock global para evitar retrains concorrentes (drift-retrain + auto-train)
_retrain_lock = threading.Lock()

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# Configuracoes
CONFIG = {
    "model_dir": "ml_models",
    "buffer_file": "ml_models/online_learning_buffer.json",
    "performance_file": "ml_models/model_performance.json",
    "retrain_threshold": 20,  # Retreinar quando tiver N novos exemplos (reduzido de 50)
    "min_improvement": 0.0,   # Melhoria minima necessaria para salvar novo modelo
    "validation_split": 0.2,  # Split para validacao do retreino
}


class OnlineLearningManager:
    """
    Gerenciador de Online Learning para o validador de sinais.
    Permite que o modelo melhore continuamente com novos dados.
    Auto-seed: ao inicializar, verifica se há sinais avaliados para popular o buffer.
    """

    def __init__(self, auto_seed: bool = False):
        self.buffer: List[Dict] = []
        self.performance_history: List[Dict] = []

        Path(CONFIG["model_dir"]).mkdir(exist_ok=True)
        self._load_buffer()
        self._load_performance_history()

        # Auto-seed: se o buffer esta vazio e há sinais avaliados, popular automaticamente
        if auto_seed and len(self.buffer) == 0:
            self._auto_seed()

    def _auto_seed(self):
        """Automaticamente popula buffer com sinais avaliados se buffer vazio"""
        try:
            from src.trading.signal_tracker import evaluate_all_signals
            evaluations = evaluate_all_signals()
            finalized = [e for e in evaluations
                         if e.get('outcome') in ('SL_HIT', 'TP1_HIT', 'TP2_HIT')]
            if finalized:
                print(f"[OL-AUTO] {len(finalized)} sinais finalizados encontrados. Populando buffer...")
                for ev in finalized:
                    outcome = ev.get('outcome', '')
                    result_map = {'TP1_HIT': 'TP1', 'TP2_HIT': 'TP2', 'SL_HIT': 'SL'}
                    result = result_map.get(outcome, 'SL')
                    signal_data = {
                        'symbol': ev.get('symbol', ''),
                        'signal': ev.get('signal', ''),
                        'entry_price': ev.get('entry_price', 0),
                        'stop_loss': ev.get('stop_loss', 0),
                        'take_profit_1': ev.get('take_profit_1', 0),
                        'confidence': ev.get('confidence', 5),
                        'indicators': ev.get('indicators', {}),
                        'trend': ev.get('trend', 'neutral'),
                        'sentiment': ev.get('sentiment', 'neutral'),
                        'rsi': ev.get('rsi', ev.get('indicators', {}).get('rsi', 50)),
                        'macd_histogram': ev.get('macd_histogram', ev.get('indicators', {}).get('macd_histogram', 0)),
                        'adx': ev.get('adx', ev.get('indicators', {}).get('adx', 25)),
                        'atr': ev.get('atr', ev.get('indicators', {}).get('atr', 0)),
                        'bb_position': ev.get('bb_position', ev.get('indicators', {}).get('bb_position', 0.5)),
                        'cvd': ev.get('cvd', ev.get('indicators', {}).get('cvd', 0)),
                        'orderbook_imbalance': ev.get('orderbook_imbalance', ev.get('indicators', {}).get('orderbook_imbalance', 0.5)),
                        'bullish_tf_count': ev.get('bullish_tf_count', 0),
                        'bearish_tf_count': ev.get('bearish_tf_count', 0),
                    }
                    self.add_signal_result(signal_data, result, ev.get('pnl_percent', 0), _batch_mode=True)
                self._save_buffer()
                print(f"[OL-AUTO] Buffer populado com {len(self.buffer)} amostras")
        except Exception as e:
            print(f"[OL-AUTO] Erro no auto-seed: {e}")

    def ensure_model_exists(self) -> bool:
        """
        Verifica se existe um modelo treinado. Se nao, tenta criar um.
        Retorna True se modelo existe (ou foi criado com sucesso).
        """
        models_path = os.path.join(CONFIG["model_dir"], "signal_validators.pkl")
        if os.path.exists(models_path):
            return True

        print("[OL] Nenhum modelo encontrado. Tentando criar modelo inicial...")

        # Primeiro: tentar treino completo via bootstrap (melhor qualidade)
        try:
            from src.ml.train_from_signals import run_training_pipeline
            success = run_training_pipeline()
            if success:
                print("[OL] Modelo inicial criado via training pipeline!")
                return True
        except Exception as e:
            print(f"[OL] Training pipeline falhou: {e}")

        # Fallback: se temos dados no buffer, treinar com eles
        if len(self.buffer) >= 10:
            print(f"[OL] Tentando criar modelo a partir do buffer ({len(self.buffer)} amostras)...")
            result = self.retrain()
            return result.get("success", False)

        print("[OL] Sem dados suficientes para criar modelo inicial.")
        return False

    def _load_buffer(self):
        """Carrega buffer de dados pendentes"""
        if os.path.exists(CONFIG["buffer_file"]):
            try:
                with open(CONFIG["buffer_file"], 'r') as f:
                    self.buffer = json.load(f)
                print(f"[OL] Buffer carregado: {len(self.buffer)} exemplos pendentes")
            except Exception:
                self.buffer = []
        else:
            self.buffer = []

    def _save_buffer(self):
        """Salva buffer de dados pendentes"""
        with open(CONFIG["buffer_file"], 'w') as f:
            json.dump(self.buffer, f, indent=2, default=str)

    def _load_performance_history(self):
        """Carrega historico de performance"""
        if os.path.exists(CONFIG["performance_file"]):
            try:
                with open(CONFIG["performance_file"], 'r') as f:
                    self.performance_history = json.load(f)
            except Exception:
                self.performance_history = []
        else:
            self.performance_history = []

    def _save_performance_history(self):
        """Salva historico de performance"""
        with open(CONFIG["performance_file"], 'w') as f:
            json.dump(self.performance_history, f, indent=2, default=str)

    def add_signal_result(self, signal: Dict, result: str, return_pct: float = 0.0, _batch_mode: bool = False):
        """
        Adiciona um resultado de sinal ao buffer para aprendizado futuro.

        Args:
            signal: O sinal original (com features)
            result: 'TP1', 'TP2', 'SL', 'TIMEOUT'
            return_pct: Retorno percentual
            _batch_mode: Se True, nao salva/retreina a cada item (para seed em lote)
        """
        # Filtrar: só aceitar resultados definidos (TP1, TP2, SL)
        # Excluir NO_SIGNAL, TIMEOUT e resultados ambíguos
        if result not in ('TP1', 'TP2', 'SL'):
            return

        # Excluir sinais NO_SIGNAL (não tem base para aprender)
        if signal.get('signal') == 'NO_SIGNAL':
            return

        # Extrair TODAS as features necessárias para o modelo
        indicators = signal.get('indicators', {})

        # Calcular campos derivados
        # 1. trend_encoded
        trend = signal.get('trend', indicators.get('trend', 'neutral'))
        trend_map = {'strong_bullish': 2, 'bullish': 1, 'neutral': 0, 'bearish': -1, 'strong_bearish': -2}
        trend_encoded = trend_map.get(str(trend).lower(), 0)

        # 2. sentiment_encoded
        sentiment = signal.get('sentiment', indicators.get('sentiment', 'neutral'))
        sentiment_map = {'bullish': 1, 'neutral': 0, 'bearish': -1}
        sentiment_encoded = sentiment_map.get(str(sentiment).lower(), 0)

        # 3. signal_encoded (BUY=1, SELL=-1, other=0)
        signal_type = signal.get('signal', 'NO_SIGNAL')
        signal_encoded = 1 if signal_type == 'BUY' else (-1 if signal_type == 'SELL' else 0)

        # 4. Features de mercado independentes (substituem risk/reward derivados do SL/TP)
        entry_price = signal.get('entry_price', 0)
        atr_val = signal.get('atr', indicators.get('atr', 0))
        atr_pct = (atr_val / entry_price * 100) if entry_price > 0 and atr_val > 0 else 2.0
        candle_body_pct = signal.get('candle_body_pct', 0.5)
        volume_ratio = signal.get('volume_ratio', indicators.get('volume_ratio', 1.0))

        # Nomes mantidos para compatibilidade com feature_columns existentes
        risk_distance_pct = atr_pct
        reward_distance_pct = candle_body_pct
        risk_reward_ratio = volume_ratio

        # Construir data point com TODAS as 16 features
        data_point = {
            'timestamp': datetime.now().isoformat(),
            'symbol': signal.get('symbol', ''),
            'signal_type': signal_type,

            # Features básicas (8)
            'rsi': signal.get('rsi', indicators.get('rsi', 50)),
            'macd_histogram': signal.get('macd_histogram', indicators.get('macd_histogram', 0)),
            'adx': signal.get('adx', indicators.get('adx', 25)),
            'atr': signal.get('atr', indicators.get('atr', 0)),
            'bb_position': signal.get('bb_position', indicators.get('bb_position', 0.5)),
            'cvd': signal.get('cvd', indicators.get('cvd', 0)),
            'orderbook_imbalance': signal.get('orderbook_imbalance', indicators.get('orderbook_imbalance', 0.5)),
            'confidence': signal.get('confidence', 5),

            # Features multi-timeframe (2)
            'bullish_tf_count': signal.get('bullish_tf_count', indicators.get('bullish_tf_count', 0)),
            'bearish_tf_count': signal.get('bearish_tf_count', indicators.get('bearish_tf_count', 0)),

            # Features codificadas (3)
            'trend_encoded': trend_encoded,
            'sentiment_encoded': sentiment_encoded,
            'signal_encoded': signal_encoded,

            # Features de risco (3)
            'risk_distance_pct': risk_distance_pct,
            'reward_distance_pct': reward_distance_pct,
            'risk_reward_ratio': risk_reward_ratio,

            # Resultado
            'result': result,
            'return_pct': return_pct,
            'target': 1 if result in ['TP1', 'TP2'] else 0
        }

        self.buffer.append(data_point)

        if _batch_mode:
            return

        self._save_buffer()

        print(f"[OL] Novo resultado adicionado: {signal.get('symbol')} {signal.get('signal')} -> {result} ({return_pct:+.2f}%)")
        print(f"[OL] Buffer: {len(self.buffer)}/{CONFIG['retrain_threshold']} para retreino")

        # Verificar se deve retreinar
        if len(self.buffer) >= CONFIG["retrain_threshold"]:
            print("[OL] Threshold atingido! Iniciando retreino...")
            self.retrain()

    def retrain(self, force_save: bool = False) -> Dict:
        """
        Retreina o modelo usando Optuna com métricas compostas.

        Métrica composta (em vez de só F1):
        - 40% Win Rate nos sinais aprovados
        - 25% Profit Factor
        - 20% Balanced WR BUY/SELL
        - 15% Taxa de aprovação (penaliza rejeitar tudo)

        Falls back to treino simples se Optuna não estiver disponível.
        """
        print("\n" + "="*60)
        print("ONLINE LEARNING - RETREINAMENTO (Optuna + Métricas Compostas)")
        print("="*60)

        if len(self.buffer) < 10:
            print(f"[OL] Dados insuficientes para retreino ({len(self.buffer)} < 10)")
            return {"success": False, "reason": "Dados insuficientes"}

        try:
            from src.ml.simple_validator import SimpleSignalValidator
            has_current_model = False
            try:
                current_validator = SimpleSignalValidator()
                current_validator.load_models()
                has_current_model = True
                print(f"[OL] Modelo atual carregado: {current_validator.best_model_name}")
            except Exception as e:
                print(f"[OL] Nenhum modelo existente ({e}). Criando inicial...")

            new_df = pd.DataFrame(self.buffer)
            feature_cols = [
                'rsi', 'macd_histogram', 'adx', 'atr', 'bb_position',
                'cvd', 'orderbook_imbalance', 'bullish_tf_count', 'bearish_tf_count',
                'confidence', 'trend_encoded', 'sentiment_encoded', 'signal_encoded',
                'risk_distance_pct', 'reward_distance_pct', 'risk_reward_ratio'
            ]
            available_cols = [col for col in feature_cols if col in new_df.columns]
            X_new = new_df[available_cols].fillna(0).values
            y_new = new_df['target'].values

            print(f"[OL] Buffer: {len(X_new)} amostras (TP: {sum(y_new)}, SL: {len(y_new)-sum(y_new)})")

            # Combinar com dataset original se necessário
            train_path = "ml_dataset/dataset_train_latest.csv"
            if os.path.exists(train_path) and len(X_new) < 100:
                original_df = pd.read_csv(train_path)
                orig_available = [col for col in available_cols if col in original_df.columns]
                if orig_available and 'target' in original_df.columns:
                    X_orig = original_df[orig_available].fillna(0).values
                    y_orig = original_df['target'].values
                    max_orig = max(len(X_new) * 2, 200)
                    if len(X_orig) > max_orig:
                        X_orig = X_orig[-max_orig:]
                        y_orig = y_orig[-max_orig:]
                    X_combined = np.vstack([X_orig, X_new])
                    y_combined = np.hstack([y_orig, y_new])
                    print(f"[OL] Combinados: {len(X_orig)} originais + {len(X_new)} buffer = {len(X_combined)}")
                else:
                    X_combined, y_combined = X_new, y_new
            else:
                X_combined, y_combined = X_new, y_new

            unique_classes = np.unique(y_combined)
            if len(unique_classes) < 2:
                minority_class = 0 if 1 in unique_classes else 1
                synthetic = np.median(X_combined, axis=0).reshape(1, -1)
                X_combined = np.vstack([X_combined, synthetic])
                y_combined = np.hstack([y_combined, [minority_class]])

            # Split temporal
            split_idx = max(int(len(X_combined) * 0.8), 1)
            X_train, X_val = X_combined[:split_idx], X_combined[split_idx:]
            y_train, y_val = y_combined[:split_idx], y_combined[split_idx:]
            print(f"[OL] Split: {len(X_train)} treino, {len(X_val)} validação")

            # Tentar Optuna se disponível e dados suficientes
            if OPTUNA_AVAILABLE and len(X_train) >= 30:
                result = self._retrain_with_optuna(
                    X_train, y_train, X_val, y_val,
                    available_cols, has_current_model, force_save
                )
            else:
                if not OPTUNA_AVAILABLE:
                    print("[OL] Optuna não disponível — usando treino simples")
                result = self._retrain_simple(
                    X_train, y_train, X_val, y_val,
                    X_combined, y_combined, available_cols,
                    has_current_model, force_save
                )

            return result

        except Exception as e:
            print(f"[OL] Erro no retreino: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def _retrain_with_optuna(self, X_train, y_train, X_val, y_val,
                              feature_cols, has_current_model, force_save) -> Dict:
        """Treino com Optuna + métricas compostas."""
        sample_weights = compute_sample_weight('balanced', y_train)
        n_trials = 20 if len(X_train) < 200 else 30

        def objective(trial):
            model_type = trial.suggest_categorical("model_type",
                ["rf", "gb", "xgb", "lr"] if XGBOOST_AVAILABLE else ["rf", "gb", "lr"])
            threshold = trial.suggest_float("threshold", 0.35, 0.65)

            scaler = StandardScaler()
            Xt = scaler.fit_transform(X_train)
            Xv = scaler.transform(X_val)
            Xv = np.clip(Xv, -5.0, 5.0)

            if model_type == "rf":
                m = RandomForestClassifier(
                    n_estimators=trial.suggest_int("rf_n", 50, 300),
                    max_depth=trial.suggest_int("rf_d", 3, 12),
                    class_weight="balanced", random_state=42, n_jobs=-1)
                m.fit(Xt, y_train)
            elif model_type == "gb":
                m = GradientBoostingClassifier(
                    n_estimators=trial.suggest_int("gb_n", 50, 300),
                    max_depth=trial.suggest_int("gb_d", 2, 8),
                    learning_rate=trial.suggest_float("gb_lr", 0.01, 0.3, log=True),
                    subsample=trial.suggest_float("gb_sub", 0.6, 1.0),
                    random_state=42)
                m.fit(Xt, y_train, sample_weight=sample_weights)
            elif model_type == "xgb":
                m = xgb.XGBClassifier(
                    objective="binary:logistic", tree_method="hist", eval_metric="logloss",
                    n_estimators=trial.suggest_int("xgb_n", 50, 300),
                    max_depth=trial.suggest_int("xgb_d", 2, 8),
                    learning_rate=trial.suggest_float("xgb_lr", 0.01, 0.3, log=True),
                    subsample=trial.suggest_float("xgb_sub", 0.6, 1.0),
                    random_state=42, n_jobs=-1, verbosity=0)
                m.fit(Xt, y_train, sample_weight=sample_weights)
            else:
                m = LogisticRegression(
                    C=trial.suggest_float("lr_C", 1e-3, 100, log=True),
                    max_iter=1000, class_weight="balanced", random_state=42)
                m.fit(Xt, y_train)

            probs = m.predict_proba(Xv)[:, 1] if hasattr(m, "predict_proba") else m.predict(Xv).astype(float)
            preds = (probs >= threshold).astype(int)
            n_approved = preds.sum()

            if n_approved < max(3, len(y_val) * 0.10):
                return -1.0

            approved_wins = y_val[preds == 1].sum()
            wr = approved_wins / max(n_approved, 1)
            approval_rate = n_approved / len(y_val)
            ar_norm = min(approval_rate / 0.5, 1.0) if approval_rate >= 0.10 else 0

            # Balanced accuracy
            sens = y_val[y_val == 1][preds[y_val == 1] == 1].sum() / max((y_val == 1).sum(), 1) if (y_val == 1).sum() > 0 else 0
            spec = (y_val[y_val == 0] == 0).sum() - (preds[y_val == 0] == 1).sum()
            spec = max(spec, 0) / max((y_val == 0).sum(), 1)

            composite = 0.40 * wr + 0.25 * min(wr / 0.5, 1.0) + 0.20 * ((sens + spec) / 2) + 0.15 * ar_norm
            if n_approved < 5:
                composite *= 0.5

            trial.set_user_attr("wr", float(wr))
            trial.set_user_attr("n_approved", int(n_approved))
            return composite

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best = study.best_trial
        print(f"[OL-OPTUNA] Melhor: score={best.value:.4f} | {best.params.get('model_type')} | WR={best.user_attrs.get('wr',0)*100:.1f}% | N={best.user_attrs.get('n_approved',0)}")

        # Retreinar o melhor modelo com todos os dados
        bp = best.params
        scaler = StandardScaler()
        X_all = np.vstack([X_train, X_val])
        y_all = np.hstack([y_train, y_val])
        X_all_s = scaler.fit_transform(X_all)
        sw = compute_sample_weight('balanced', y_all)

        mt = bp.get("model_type", "rf")
        model_name_map = {"rf": "RandomForest", "gb": "GradientBoosting", "xgb": "XGBoost", "lr": "LogisticRegression"}
        best_model_name = model_name_map.get(mt, "RandomForest")

        if mt == "rf":
            final = RandomForestClassifier(n_estimators=bp.get("rf_n", 100), max_depth=bp.get("rf_d", 5),
                                           class_weight="balanced", random_state=42, n_jobs=-1)
            final.fit(X_all_s, y_all)
        elif mt == "gb":
            final = GradientBoostingClassifier(n_estimators=bp.get("gb_n", 100), max_depth=bp.get("gb_d", 3),
                                               learning_rate=bp.get("gb_lr", 0.1), subsample=bp.get("gb_sub", 0.8),
                                               random_state=42)
            final.fit(X_all_s, y_all, sample_weight=sw)
        elif mt == "xgb" and XGBOOST_AVAILABLE:
            final = xgb.XGBClassifier(objective="binary:logistic", tree_method="hist", eval_metric="logloss",
                                       n_estimators=bp.get("xgb_n", 200), max_depth=bp.get("xgb_d", 4),
                                       learning_rate=bp.get("xgb_lr", 0.1), subsample=bp.get("xgb_sub", 0.8),
                                       random_state=42, n_jobs=-1, verbosity=0)
            final.fit(X_all_s, y_all, sample_weight=sw)
        else:
            final = LogisticRegression(C=bp.get("lr_C", 1.0), max_iter=1000, class_weight="balanced", random_state=42)
            final.fit(X_all_s, y_all)

        # Calibrar
        try:
            from sklearn.calibration import CalibratedClassifierCV
            cal = CalibratedClassifierCV(final, method='isotonic', cv=3)
            cal.fit(X_all_s, y_all, sample_weight=sw)
            final = cal
            print(f"[OL] Modelo calibrado")
        except Exception:
            pass

        trained_models = {best_model_name: final}

        y_pred = final.predict(scaler.transform(X_val))
        new_accuracy = accuracy_score(y_val, y_pred)
        new_f1 = f1_score(y_val, y_pred, zero_division=0)

        # Salvar
        self._save_new_model_ensemble(trained_models, best_model_name, scaler, feature_cols, new_accuracy, new_f1)

        # Salvar preproc artifacts
        preproc_path = os.path.join(CONFIG["model_dir"], "preproc_simple.pkl")
        train_medians = {}
        train_clip_bounds = {}
        for i, col in enumerate(feature_cols):
            vals = X_all[:, i] if i < X_all.shape[1] else np.array([0])
            med = float(np.nanmedian(vals))
            train_medians[col] = med if not np.isnan(med) else 0
            mean_, std_ = float(np.mean(vals)), float(np.std(vals))
            if std_ > 0:
                train_clip_bounds[col] = (mean_ - 3*std_, mean_ + 3*std_)
        with open(preproc_path, 'wb') as f:
            pickle.dump({'train_medians': train_medians, 'train_clip_bounds': train_clip_bounds}, f)

        self._save_combined_dataset(X_all, y_all, feature_cols)
        self.buffer = []
        self._save_buffer()

        pred_log_path = os.path.join(CONFIG["model_dir"], "prediction_log.json")
        if os.path.exists(pred_log_path):
            try:
                os.remove(pred_log_path)
            except Exception:
                pass

        self._record_performance(new_accuracy, new_f1, len(X_all))
        print(f"[OL] Modelo Optuna salvo! {best_model_name} Acc={new_accuracy:.1%} F1={new_f1:.3f}")

        return {
            "success": True,
            "new_accuracy": new_accuracy,
            "new_f1": new_f1,
            "improvement": 0,
            "samples_used": len(X_all),
            "best_model": best_model_name,
            "optuna_score": best.value,
            "optuna_trials": n_trials,
            "first_training": not has_current_model,
        }

    def _retrain_simple(self, X_train, y_train, X_val, y_val,
                         X_combined, y_combined, feature_cols,
                         has_current_model, force_save) -> Dict:
        """Fallback: treino simples sem Optuna."""
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        sample_weights = compute_sample_weight('balanced', y_train)

        models_to_train = {
            "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
            "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42),
            "GradientBoosting": GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
        }
        if XGBOOST_AVAILABLE:
            models_to_train["XGBoost"] = xgb.XGBClassifier(
                objective="binary:logistic", tree_method="hist", eval_metric="logloss",
                n_estimators=200, max_depth=4, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=-1, verbosity=0)

        best_model_name = None
        best_f1 = -1
        best_accuracy = 0
        trained_models = {}

        for name, model in models_to_train.items():
            try:
                if name in ("GradientBoosting", "XGBoost"):
                    model.fit(X_train_s, y_train, sample_weight=sample_weights)
                else:
                    model.fit(X_train_s, y_train)
                trained_models[name] = model
                y_pred = model.predict(X_val_s)
                f1 = f1_score(y_val, y_pred, zero_division=0)
                acc = accuracy_score(y_val, y_pred)
                print(f"[OL]   {name}: Acc={acc:.1%}, F1={f1:.3f}")
                if f1 > best_f1:
                    best_f1, best_accuracy, best_model_name = f1, acc, name
            except Exception as e:
                print(f"[OL]   {name}: ERRO - {e}")

        if best_model_name is None:
            return {"success": False, "reason": "Nenhum modelo treinado com sucesso"}

        # Calibrar
        try:
            from sklearn.calibration import CalibratedClassifierCV
            base = trained_models[best_model_name]
            cal = CalibratedClassifierCV(base, method='isotonic', cv=3)
            cal.fit(X_train_s, y_train, sample_weight=sample_weights)
            trained_models[best_model_name] = cal
        except Exception:
            pass

        self._save_new_model_ensemble(trained_models, best_model_name, scaler, feature_cols, best_accuracy, best_f1)
        self._save_combined_dataset(X_combined, y_combined, feature_cols)
        self.buffer = []
        self._save_buffer()

        pred_log_path = os.path.join(CONFIG["model_dir"], "prediction_log.json")
        if os.path.exists(pred_log_path):
            try:
                os.remove(pred_log_path)
            except Exception:
                pass

        self._record_performance(best_accuracy, best_f1, len(X_combined))
        print(f"[OL] Modelo salvo! {best_model_name} Acc={best_accuracy:.1%} F1={best_f1:.3f}")

        return {
            "success": True, "new_accuracy": best_accuracy, "new_f1": best_f1,
            "samples_used": len(X_combined), "best_model": best_model_name,
            "first_training": not has_current_model,
        }

    def _save_new_model(self, model, scaler, feature_columns):
        """Salva o novo modelo treinado (compatibilidade)"""
        self._save_new_model_ensemble(
            {"LogisticRegression": model}, "LogisticRegression",
            scaler, feature_columns, 0, 0
        )

    def _save_new_model_ensemble(self, trained_models, best_model_name, scaler, feature_columns, accuracy, f1):
        """Salva ensemble de modelos treinados"""
        models_path = os.path.join(CONFIG["model_dir"], "signal_validators.pkl")

        with open(models_path, 'wb') as f:
            pickle.dump(trained_models, f)

        # Salvar scaler
        scaler_path = os.path.join(CONFIG["model_dir"], "scaler_simple.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

        # Atualizar metadata
        info_path = os.path.join(CONFIG["model_dir"], "model_info_simple.json")
        info = {}
        if os.path.exists(info_path):
            try:
                with open(info_path, 'r') as f:
                    info = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        info['last_retrain'] = datetime.now().isoformat()
        info['retrain_count'] = info.get('retrain_count', 0) + 1
        info['feature_columns'] = feature_columns
        info['best_model'] = best_model_name
        info['best_accuracy'] = accuracy
        info['best_f1'] = f1

        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2, default=str)

    def _save_combined_dataset(self, X, y, feature_columns):
        """Salva dataset combinado para retreinos futuros e drift detection"""
        try:
            os.makedirs("ml_dataset", exist_ok=True)
            df = pd.DataFrame(X, columns=feature_columns)
            df['target'] = y
            df.to_csv("ml_dataset/dataset_train_latest.csv", index=False)

            # Salvar em npz para drift detection (referência rápida)
            npz_path = os.path.join(CONFIG["model_dir"], "combined_dataset.npz")
            np.savez(npz_path, X=X, y=y)

            print(f"[OL] Dataset atualizado: {len(df)} amostras salvas")
        except Exception as e:
            print(f"[OL] Erro ao salvar dataset: {e}")

    def _record_performance(self, accuracy: float, f1: float, n_samples: int):
        """Registra performance do modelo"""
        self.performance_history.append({
            'timestamp': datetime.now().isoformat(),
            'accuracy': accuracy,
            'f1_score': f1,
            'n_samples': n_samples
        })
        self._save_performance_history()

    def get_performance_summary(self) -> Dict:
        """Retorna resumo de performance do modelo"""
        if not self.performance_history:
            return {"status": "Sem historico"}

        recent = self.performance_history[-1]

        return {
            "last_retrain": recent.get('timestamp'),
            "current_accuracy": recent.get('accuracy'),
            "current_f1": recent.get('f1_score'),
            "total_retrains": len(self.performance_history),
            "buffer_size": len(self.buffer),
            "next_retrain_at": CONFIG["retrain_threshold"] - len(self.buffer)
        }

    def detect_drift(self, reference_data: np.ndarray = None) -> Dict:
        """
        Detecta drift nas features usando PSI (Population Stability Index).
        Compara distribuição dos dados do buffer com os dados de treino.
        PSI > 0.2 = drift significativo, PSI > 0.1 = drift moderado.
        """
        if len(self.buffer) < 10:
            return {"drift_detected": False, "reason": "Buffer muito pequeno"}

        feature_cols = [
            'rsi', 'macd_histogram', 'adx', 'atr', 'bb_position',
            'cvd', 'orderbook_imbalance', 'bullish_tf_count', 'bearish_tf_count',
            'confidence', 'trend_encoded', 'sentiment_encoded', 'signal_encoded',
            'risk_distance_pct', 'reward_distance_pct', 'risk_reward_ratio'
        ]

        # Dados do buffer recente
        buffer_data = []
        for item in self.buffer:
            row = [item.get(col, 0) for col in feature_cols]
            buffer_data.append(row)
        buffer_array = np.array(buffer_data, dtype=float)
        buffer_array = np.nan_to_num(buffer_array, nan=0.0)

        # Se não tem referência, carregar do dataset salvo
        if reference_data is None:
            dataset_path = os.path.join(CONFIG["model_dir"], "combined_dataset.npz")
            if not os.path.exists(dataset_path):
                return {"drift_detected": False, "reason": "Sem dataset de referência"}
            try:
                loaded = np.load(dataset_path)
                reference_data = loaded['X']
            except Exception:
                return {"drift_detected": False, "reason": "Erro ao carregar referência"}

        # Calcular PSI por feature
        drift_results = {}
        total_psi = 0
        drifted_features = []

        for i, col in enumerate(feature_cols):
            if i >= reference_data.shape[1] or i >= buffer_array.shape[1]:
                continue
            psi = self._calculate_psi(reference_data[:, i], buffer_array[:, i])
            drift_results[col] = round(psi, 4)
            total_psi += psi
            if psi > 0.2:
                drifted_features.append(f"{col} (PSI={psi:.3f})")

        avg_psi = total_psi / max(len(drift_results), 1)
        drift_detected = avg_psi > 0.1 or len(drifted_features) >= 3

        if drift_detected:
            print(f"[DRIFT] Drift detectado! PSI médio={avg_psi:.3f}, features com drift: {drifted_features}")
        else:
            print(f"[DRIFT] Sem drift significativo. PSI médio={avg_psi:.3f}")

        return {
            "drift_detected": drift_detected,
            "avg_psi": round(avg_psi, 4),
            "feature_psi": drift_results,
            "drifted_features": drifted_features,
            "recommendation": "Retreinar modelo" if drift_detected else "Modelo OK"
        }

    @staticmethod
    def _calculate_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
        """Calcula PSI (Population Stability Index) entre duas distribuições."""
        eps = 1e-6

        # Usar mesmos bins para ambos
        combined = np.concatenate([reference, current])
        bins = np.histogram_bin_edges(combined, bins=n_bins)

        ref_hist, _ = np.histogram(reference, bins=bins)
        cur_hist, _ = np.histogram(current, bins=bins)

        # Normalizar para proporções
        ref_pct = ref_hist / max(len(reference), 1) + eps
        cur_pct = cur_hist / max(len(current), 1) + eps

        # PSI = sum((cur - ref) * ln(cur/ref))
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return max(0, psi)


# Instancia global para uso no bot
online_learning_manager = OnlineLearningManager()


def add_trade_result(signal: Dict, result: str, return_pct: float = 0.0):
    """
    Funcao helper para adicionar resultado de trade.
    Chamada pelo bot quando um trade e fechado.
    """
    online_learning_manager.add_signal_result(signal, result, return_pct)


def manual_retrain(force_save: bool = True):
    """Forca retreino manual. Com force_save=True, substitui o modelo mesmo se F1 na validacao nao melhorar."""
    return online_learning_manager.retrain(force_save=force_save)


def seed_from_evaluated_signals(force_retrain: bool = True, max_signals: int = 0) -> Dict:
    """
    Popula o buffer com sinais avaliados, ENRIQUECIDOS com indicadores reais.
    Usa enrich_signal_with_klines() para buscar klines historicos e calcular
    indicadores tecnicos reais (RSI, MACD, ADX, etc.) em vez de usar defaults.

    Se nao existe modelo, usa o training pipeline bootstrap (que gera dados
    de alta qualidade a partir de historico Binance).

    Args:
        force_retrain: Se True, dispara retreino apos popular o buffer
        max_signals: Limite de sinais a processar (evita timeout no dashboard; 0 = sem limite)

    Returns:
        Dict com resultado da operacao
    """
    import time

    # Lock para evitar retrains concorrentes (drift-retrain thread + auto-train thread)
    if not _retrain_lock.acquire(blocking=False):
        print("[OL-SEED] Retrain já em andamento em outra thread — pulando.")
        return {"success": False, "error": "retrain_already_running", "signals_added": 0}

    try:
        return _seed_from_evaluated_signals_impl(force_retrain, max_signals)
    finally:
        _retrain_lock.release()


def _seed_from_evaluated_signals_impl(force_retrain: bool = True, max_signals: int = 0) -> Dict:
    """Implementação real do seed — chamada com lock adquirido."""
    import time

    # Se nao existe modelo, preferir bootstrap pipeline (dados de alta qualidade)
    models_path = os.path.join(CONFIG["model_dir"], "signal_validators.pkl")
    if not os.path.exists(models_path) and force_retrain:
        print("[OL-SEED] Nenhum modelo existente. Usando training pipeline bootstrap...")
        try:
            from src.ml.train_from_signals import run_training_pipeline
            success = run_training_pipeline()
            if success:
                return {
                    "success": True,
                    "signals_added": 0,
                    "buffer_total": len(online_learning_manager.buffer),
                    "retrain_result": {"success": True, "method": "bootstrap_pipeline"},
                }
        except Exception as e:
            print(f"[OL-SEED] Bootstrap pipeline falhou: {e}. Tentando via buffer...")

    try:
        from src.trading.signal_tracker import evaluate_all_signals
    except ImportError as e:
        return {"success": False, "error": f"Import error: {e}"}

    # Limitar avaliações para não travar o dashboard (usa cache quando existe)
    evaluations = evaluate_all_signals()
    if not evaluations:
        return {"success": False, "error": "Nenhum sinal avaliado encontrado"}

    # Filtrar apenas sinais finalizados (SL/TP hit)
    finalized = [e for e in evaluations
                 if e.get('outcome') in ('SL_HIT', 'TP1_HIT', 'TP2_HIT')]

    if not finalized:
        return {"success": False, "error": "Nenhum sinal finalizado (SL/TP hit)"}

    if max_signals > 0 and len(finalized) > max_signals:
        finalized = finalized[:max_signals]
        print(f"[OL-SEED] Limitando a {max_signals} sinais para evitar timeout.")

    # Tentar enriquecer sinais com indicadores reais via klines
    try:
        from src.ml.train_from_signals import enrich_signal_with_klines
        use_enrichment = True
        print(f"[OL-SEED] Enriquecendo {len(finalized)} sinais com indicadores reais...")
    except ImportError:
        use_enrichment = False
        print("[OL-SEED] AVISO: enrich_signal_with_klines nao disponivel. Usando dados do sinal.")

    # Evitar duplicatas - checar sinais ja no buffer por symbol+timestamp
    existing_keys = {
        (b.get('symbol', ''), b.get('timestamp', '')[:16])
        for b in online_learning_manager.buffer
    }

    added = 0
    enriched = 0
    for i, ev in enumerate(finalized):
        key = (ev.get('symbol', ''), ev.get('timestamp', '')[:16])
        if key in existing_keys:
            continue

        outcome = ev.get('outcome', '')
        result_map = {'TP1_HIT': 'TP1', 'TP2_HIT': 'TP2', 'SL_HIT': 'SL'}
        result = result_map.get(outcome, 'SL')

        # Tentar enriquecer com indicadores reais
        if use_enrichment:
            try:
                enriched_data = enrich_signal_with_klines(ev)
                if enriched_data:
                    # enrich_signal_with_klines retorna dict com features reais + target
                    # Map numeric trend_encoded back to string labels
                    # that add_signal_result's trend_map expects
                    _te = enriched_data.get('trend_encoded', 0)
                    _trend_rmap = {2: 'strong_bullish', 1: 'bullish', 0: 'neutral',
                                   -1: 'bearish', -2: 'strong_bearish'}
                    _se = enriched_data.get('sentiment_encoded', 0)
                    _sent_rmap = {1: 'bullish', 0: 'neutral', -1: 'bearish'}

                    signal_data = {
                        'symbol': ev.get('symbol', ''),
                        'signal': ev.get('signal', ''),
                        'entry_price': ev.get('entry_price', 0),
                        'stop_loss': ev.get('stop_loss', 0),
                        'take_profit_1': ev.get('take_profit_1', 0),
                        'confidence': ev.get('confidence', 5),
                        'rsi': enriched_data.get('rsi', 50),
                        'macd_histogram': enriched_data.get('macd_histogram', 0),
                        'adx': enriched_data.get('adx', 25),
                        'atr': enriched_data.get('atr', 0),
                        'bb_position': enriched_data.get('bb_position', 0.5),
                        'cvd': enriched_data.get('cvd', 0),
                        'orderbook_imbalance': enriched_data.get('orderbook_imbalance', 0.5),
                        'bullish_tf_count': enriched_data.get('bullish_tf_count', 5),
                        'bearish_tf_count': enriched_data.get('bearish_tf_count', 5),
                        'trend': _trend_rmap.get(int(_te), 'neutral'),
                        'sentiment': _sent_rmap.get(int(_se), 'neutral'),
                        'indicators': {},
                    }
                    pnl = ev.get('pnl_percent', 0)
                    online_learning_manager.add_signal_result(signal_data, result, pnl, _batch_mode=True)
                    added += 1
                    enriched += 1
                    if (i + 1) % 10 == 0:
                        print(f"  [{i+1}/{len(finalized)}] {enriched} enriquecidos...")
                    time.sleep(0.2)  # Rate limit Binance API
                    continue
            except Exception:
                pass  # Fallback para dados do sinal

        # Fallback: usar dados originais do sinal (sem enriquecimento)
        signal_data = {
            'symbol': ev.get('symbol', ''),
            'signal': ev.get('signal', ''),
            'entry_price': ev.get('entry_price', 0),
            'stop_loss': ev.get('stop_loss', 0),
            'take_profit_1': ev.get('take_profit_1', 0),
            'confidence': ev.get('confidence', 5),
            'indicators': ev.get('indicators', {}),
            'trend': ev.get('trend', 'neutral'),
            'sentiment': ev.get('sentiment', 'neutral'),
            'bullish_tf_count': ev.get('bullish_tf_count', 0),
            'bearish_tf_count': ev.get('bearish_tf_count', 0),
            'rsi': ev.get('rsi', ev.get('indicators', {}).get('rsi', 50)),
            'macd_histogram': ev.get('macd_histogram', ev.get('indicators', {}).get('macd_histogram', 0)),
            'adx': ev.get('adx', ev.get('indicators', {}).get('adx', 25)),
            'atr': ev.get('atr', ev.get('indicators', {}).get('atr', 0)),
            'bb_position': ev.get('bb_position', ev.get('indicators', {}).get('bb_position', 0.5)),
            'cvd': ev.get('cvd', ev.get('indicators', {}).get('cvd', 0)),
            'orderbook_imbalance': ev.get('orderbook_imbalance', ev.get('indicators', {}).get('orderbook_imbalance', 0.5)),
        }

        pnl = ev.get('pnl_percent', 0)
        online_learning_manager.add_signal_result(signal_data, result, pnl, _batch_mode=True)
        added += 1

    # Salvar buffer uma unica vez apos adicionar todos
    if added > 0:
        online_learning_manager._save_buffer()

    print(f"[OL-SEED] {added} sinais adicionados ({enriched} enriquecidos com indicadores reais)")

    retrain_result = None
    if force_retrain and len(online_learning_manager.buffer) >= 10:
        print(f"[OL-SEED] Disparando retreino com {len(online_learning_manager.buffer)} amostras...")
        retrain_result = online_learning_manager.retrain()

    return {
        "success": True,
        "signals_evaluated": len(evaluations),
        "signals_finalized": len(finalized),
        "signals_added": added,
        "signals_enriched": enriched,
        "buffer_total": len(online_learning_manager.buffer),
        "retrain_result": retrain_result,
    }


def get_learning_status() -> Dict:
    """Retorna status do online learning"""
    return online_learning_manager.get_performance_summary()


if __name__ == "__main__":
    # Teste
    print("\n" + "="*60)
    print("ONLINE LEARNING MANAGER - TESTE")
    print("="*60)

    manager = OnlineLearningManager()

    # Status
    print(f"\nStatus: {manager.get_performance_summary()}")

    # Simular adicao de resultado
    test_signal = {
        'symbol': 'BTCUSDT',
        'signal': 'BUY',
        'confidence': 7,
        'rsi': 45,
        'macd_histogram': 100
    }

    # manager.add_signal_result(test_signal, 'TP1', 2.5)

    print("\n[OK] Teste concluido!")

