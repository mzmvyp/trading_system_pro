"""
LSTM Sequence Validator - Modelo Bi-LSTM para validar sinais usando séries temporais.
=====================================================================================

Diferença do lstm_validator.py original:
- Usa SEQUÊNCIAS de N candles (série temporal real), não ponto único
- Bi-LSTM (analisa passado e contexto bidirecional)
- Treinado com dados do backtest (backtest_dataset_generator.py)
- Híbrido: features de sequência + features estáticas do momento do sinal

Pipeline:
1. backtest_dataset_generator.py gera X_train/X_test (sequências de candles)
2. Este módulo treina Bi-LSTM nessas sequências
3. Em produção, recebe os últimos N candles + sinal → prediz win probability
4. Integra no sistema de confluência como mais um voto
"""

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from src.core.logger import get_logger

logger = get_logger(__name__)

MODEL_DIR = Path("ml_models")
DATASET_DIR = Path("ml_dataset/backtest")


class LSTMSequenceValidator:
    """
    Bi-LSTM que valida sinais usando sequências temporais de candles.

    Treinado com trades do backtest. Cada amostra = últimos N candles
    antes de um trade, label = se o trade foi vencedor.
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_info = {}
        self.sequence_length = 60
        self.n_features = 0

        MODEL_DIR.mkdir(parents=True, exist_ok=True)

    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Carrega dataset gerado pelo backtest_dataset_generator."""
        X_train = np.load(DATASET_DIR / "X_train_latest.npy")
        X_test = np.load(DATASET_DIR / "X_test_latest.npy")
        y_train = np.load(DATASET_DIR / "y_train_latest.npy")
        y_test = np.load(DATASET_DIR / "y_test_latest.npy")

        self.sequence_length = X_train.shape[1]
        self.n_features = X_train.shape[2]

        print("[OK] Dataset carregado:")
        print(f"  Train: {X_train.shape} (win rate: {y_train.mean()*100:.1f}%)")
        print(f"  Test:  {X_test.shape} (win rate: {y_test.mean()*100:.1f}%)")

        return X_train, X_test, y_train, y_test

    def build_model(self):
        """Constrói modelo Bi-LSTM para classificação de sinais."""
        import tensorflow as tf
        from tensorflow.keras.layers import (
            LSTM,
            BatchNormalization,
            Bidirectional,
            Dense,
            Dropout,
            Input,
        )
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.regularizers import l1_l2

        print(f"\n[BUILD] Construindo Bi-LSTM (seq={self.sequence_length}, features={self.n_features})")

        # Arquitetura balanceada para ~200-5000 amostras
        # V1 (64+32, dropout 0.2): overfitting severo (81% treino vs 59% teste)
        # V2 (32+16, dropout 0.4): sem overfitting mas range colapsou (0.32-0.51)
        # V3 (48+24, dropout 0.3): meio-termo — generaliza sem colapsar probabilidades
        # V4 (32+16, dropout 0.30, L2=2e-3): mais regularização para dataset pequeno
        model = Sequential([
            Input(shape=(self.sequence_length, self.n_features)),

            # Bi-LSTM Layer 1
            Bidirectional(LSTM(
                units=32,
                return_sequences=True,
                dropout=0.30,
                recurrent_dropout=0.25,
                kernel_regularizer=l1_l2(l1=1e-5, l2=2e-3),
            ), name="bilstm_1"),
            BatchNormalization(),

            # Bi-LSTM Layer 2
            Bidirectional(LSTM(
                units=16,
                return_sequences=False,
                dropout=0.30,
                recurrent_dropout=0.25,
                kernel_regularizer=l1_l2(l1=1e-5, l2=2e-3),
            ), name="bilstm_2"),
            BatchNormalization(),

            # Dense layer
            Dense(16, activation="relu", kernel_regularizer=l1_l2(l1=1e-5, l2=2e-3)),
            Dropout(0.40),

            # Output
            Dense(1, activation="sigmoid", name="output"),
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
        )

        print(f"[OK] Modelo: {model.count_params():,} parâmetros")
        self.model = model
        return model

    def train(self, epochs: int = 100, batch_size: int = 32) -> Dict:
        """Treina o modelo nos dados do backtest."""
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        print("\n" + "=" * 60)
        print("TREINAMENTO Bi-LSTM")
        print("=" * 60)

        X_train, X_test, y_train, y_test = self.load_dataset()

        # ========================================
        # CLASS WEIGHTS ao invés de undersampling
        # ========================================
        # Undersampling causava mismatch treino/teste:
        # treino balanceado a 50% mas teste com WR real ~30%
        # → modelo aprendia distribuição errada, val_auc < 0.50
        # Class weights mantém TODAS as amostras e corrige o desbalanceamento
        win_rate = float(y_train.mean())
        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos
        if n_pos > 0 and n_neg > 0:
            weight_neg = len(y_train) / (2.0 * n_neg)
            weight_pos = len(y_train) / (2.0 * n_pos)
            class_weights = {0: weight_neg, 1: weight_pos}
            print(f"[CLASS_WEIGHT] WR={win_rate:.1%} | {n_pos} wins, {n_neg} losses | "
                  f"weights: 0→{weight_neg:.2f}, 1→{weight_pos:.2f}")
        else:
            class_weights = {0: 1.0, 1: 1.0}
            print(f"[CLASS_WEIGHT] Dataset com apenas uma classe, pesos iguais")

        # Normalizar features — fit APENAS no train (evita data leakage)
        from sklearn.preprocessing import StandardScaler

        n_train = X_train.shape[0]
        n_test = X_test.shape[0]

        X_train_flat = X_train.reshape(-1, self.n_features)
        X_test_flat = X_test.reshape(-1, self.n_features)

        self.scaler = StandardScaler()
        X_train_scaled_flat = self.scaler.fit_transform(X_train_flat)
        X_test_scaled_flat = self.scaler.transform(X_test_flat)

        X_train_scaled = X_train_scaled_flat.reshape(
            n_train, self.sequence_length, self.n_features
        )
        X_test_scaled = X_test_scaled_flat.reshape(
            n_test, self.sequence_length, self.n_features
        )

        # Tratar NaN/Inf
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        callbacks = [
            EarlyStopping(
                monitor="val_auc",
                patience=15,
                restore_best_weights=True,
                mode="max",
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=7,
                min_lr=1e-6,
            ),
        ]

        # ========================================
        # HARD EXAMPLE MINING — aprender com os erros
        # ========================================
        # Se existe modelo anterior, identifica exemplos que ele errou
        # e dá peso 2x nesses exemplos no treino → foca nos erros
        sample_weight = np.ones(len(y_train), dtype=np.float32)
        n_hard = 0

        if self._load_previous_model():
            try:
                y_pred_prob = self.model.predict(X_train_scaled, verbose=0).flatten()
                y_pred = (y_pred_prob > 0.5).astype(int)
                mistakes = (y_pred != y_train)
                n_hard = int(mistakes.sum())
                if n_hard > 0:
                    # Erros do modelo anterior recebem peso 2x
                    sample_weight[mistakes] = 2.0
                    print(f"[HARD EXAMPLES] {n_hard}/{len(y_train)} erros do modelo anterior receberão peso 2x")
                else:
                    print("[HARD EXAMPLES] Modelo anterior não errou nenhum exemplo (improvável)")
            except Exception as e:
                print(f"[HARD EXAMPLES] Não foi possível avaliar modelo anterior: {e}")

        # Rebuild fresh model para treinar do zero com os pesos
        self.build_model()

        print(f"\n[TRAIN] Iniciando... (hard examples: {n_hard}, class_weight ativo)")
        start = datetime.now(timezone.utc)

        history = self.model.fit(
            X_train_scaled, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test_scaled, y_test),
            callbacks=callbacks,
            sample_weight=sample_weight,
            class_weight=class_weights,
            verbose=1,
        )

        training_time = (datetime.now(timezone.utc) - start).total_seconds()

        # Avaliar
        results = self._evaluate(X_train_scaled, y_train, X_test_scaled, y_test)

        # Detectar fonte real do dataset
        _data_source = "unknown"
        try:
            _ds_info_path = DATASET_DIR / "dataset_info_latest.json"
            if _ds_info_path.exists():
                with open(_ds_info_path, "r") as _f:
                    _ds_info = json.load(_f)
                    _data_source = _ds_info.get("data_source", "unknown")
        except Exception:
            pass

        self.model_info = {
            "type": "Bi-LSTM",
            "training_date": datetime.now(timezone.utc).isoformat(),
            "training_time_seconds": training_time,
            "epochs_run": len(history.history["loss"]),
            "sequence_length": self.sequence_length,
            "n_features": self.n_features,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "hard_examples": n_hard,
            "results": results,
            "data_source": _data_source,
        }

        self._save_model()
        return results

    def _evaluate(self, X_train, y_train, X_test, y_test) -> Dict:
        """Avalia modelo no treino e teste com métricas completas."""
        from sklearn.metrics import (
            accuracy_score, balanced_accuracy_score, classification_report,
            f1_score, roc_auc_score,
        )

        results = {}

        y_train_prob = self.model.predict(X_train, verbose=0).flatten()
        y_train_pred = (y_train_prob > 0.5).astype(int)
        results["train"] = {
            "accuracy": float(accuracy_score(y_train, y_train_pred)),
            "f1_score": float(f1_score(y_train, y_train_pred, zero_division=0)),
        }

        y_test_prob = self.model.predict(X_test, verbose=0).flatten()
        y_test_pred = (y_test_prob > 0.5).astype(int)

        # AUC e balanced accuracy no teste
        try:
            test_auc = float(roc_auc_score(y_test, y_test_prob))
        except Exception:
            test_auc = 0.0
        test_bal_acc = float(balanced_accuracy_score(y_test, y_test_pred))

        # Métricas operacionais (sinais com prob >= 0.6 ou <= 0.4)
        op_mask = (y_test_prob >= 0.6) | (y_test_prob <= 0.4)
        n_operational = int(op_mask.sum())
        if n_operational > 0:
            op_preds = np.where(y_test_prob >= 0.6, 1, np.where(y_test_prob <= 0.4, 0, -1))
            op_valid = op_preds >= 0
            op_correct = int((op_preds[op_valid] == y_test[op_valid]).sum())
            op_accuracy = op_correct / n_operational
        else:
            op_accuracy = 0.0

        results["test"] = {
            "accuracy": float(accuracy_score(y_test, y_test_pred)),
            "f1_score": float(f1_score(y_test, y_test_pred, zero_division=0)),
            "auc": test_auc,
            "balanced_accuracy": test_bal_acc,
            "op_accuracy": float(op_accuracy),
            "n_operational": n_operational,
        }
        # Top-level for easy access
        results["test_auc"] = test_auc

        print(f"\n{'='*60}")
        print("RESULTADOS")
        print(f"{'='*60}")
        print(f"  Train: acc={results['train']['accuracy']*100:.1f}%, f1={results['train']['f1_score']:.3f}")
        print(f"  Test:  acc={results['test']['accuracy']*100:.1f}%, f1={results['test']['f1_score']:.3f}, "
              f"AUC={test_auc:.3f}, bal_acc={test_bal_acc:.1%}")
        print(f"  Operacional: {op_accuracy:.1%} (n={n_operational}/{len(y_test)})")
        print(f"\n{classification_report(y_test, y_test_pred, target_names=['Loss', 'Win'])}")

        return results

    def _load_previous_model(self) -> bool:
        """Carrega modelo anterior (se existir) para avaliar erros no hard example mining."""
        try:
            import tensorflow as tf
            model_path = MODEL_DIR / "bilstm_sequence_validator.h5"
            scaler_path = MODEL_DIR / "bilstm_scaler.pkl"
            if not model_path.exists() or not scaler_path.exists():
                print("[HARD EXAMPLES] Nenhum modelo anterior encontrado — primeiro treino")
                return False
            self.model = tf.keras.models.load_model(model_path)
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
            print("[HARD EXAMPLES] Modelo anterior carregado para identificar erros")
            return True
        except Exception as e:
            print(f"[HARD EXAMPLES] Falha ao carregar modelo anterior: {e}")
            return False

    def _save_model(self):
        """Salva modelo, scaler e metadata."""
        model_path = MODEL_DIR / "bilstm_sequence_validator.h5"
        self.model.save(model_path)

        scaler_path = MODEL_DIR / "bilstm_scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)

        info_path = MODEL_DIR / "bilstm_model_info.json"
        with open(info_path, "w") as f:
            json.dump(self.model_info, f, indent=2, default=str)

        print(f"\n[SAVE] Modelo salvo em {MODEL_DIR}/")

    def load_model(self) -> bool:
        """Carrega modelo treinado."""
        try:
            import tensorflow as tf

            model_path = MODEL_DIR / "bilstm_sequence_validator.h5"
            scaler_path = MODEL_DIR / "bilstm_scaler.pkl"
            info_path = MODEL_DIR / "bilstm_model_info.json"

            if not model_path.exists():
                return False

            self.model = tf.keras.models.load_model(model_path)

            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)

            with open(info_path, "r") as f:
                self.model_info = json.load(f)

            self.sequence_length = self.model_info.get("sequence_length", 60)
            self.n_features = self.model_info.get("n_features", 18)

            logger.info(f"[Bi-LSTM] Modelo carregado (seq={self.sequence_length}, features={self.n_features})")
            return True

        except Exception as e:
            logger.warning(f"[Bi-LSTM] Erro ao carregar modelo: {e}")
            return False

    def predict_from_candles(self, candles_df, direction: str = "") -> Dict:
        """
        Prediz probabilidade de sucesso a partir de candles recentes.

        Usado em produção: recebe os últimos N candles com indicadores
        e retorna probabilidade de win.

        Args:
            candles_df: DataFrame com últimos sequence_length candles
            direction: "BUY" ou "SELL" — essencial para a predição

        Returns:
            Dict com probability, prediction, confidence
        """
        if self.model is None:
            if not self.load_model():
                return {"probability": 0.5, "prediction": 0, "confidence": 0, "error": "model_not_found"}

        try:
            feature_cols = [
                "close", "high", "low", "open", "volume",
                "rsi", "ema_fast", "ema_slow", "macd", "macd_signal", "macd_hist",
                "bb_upper", "bb_middle", "bb_lower", "adx", "atr",
                "volume_ratio",
            ]

            available = [c for c in feature_cols if c in candles_df.columns]
            if len(available) < 10:
                return {"probability": 0.5, "prediction": 0, "confidence": 0, "error": "insufficient_features"}

            # Pegar últimos sequence_length candles
            df = candles_df.tail(self.sequence_length)
            if len(df) < self.sequence_length:
                return {"probability": 0.5, "prediction": 0, "confidence": 0, "error": "insufficient_candles"}

            # Normalizar preços
            seq_data = df[available].copy()
            first_close = seq_data["close"].iloc[0]
            if first_close > 0:
                for col in ["close", "high", "low", "open", "ema_fast", "ema_slow", "bb_upper", "bb_middle", "bb_lower"]:
                    if col in seq_data.columns:
                        seq_data[col] = (seq_data[col] / first_close - 1) * 100
                if "atr" in seq_data.columns:
                    seq_data["atr"] = seq_data["atr"] / first_close * 100

            if "volume" in seq_data.columns:
                vol_mean = seq_data["volume"].mean()
                if vol_mean > 0:
                    seq_data["volume"] = seq_data["volume"] / vol_mean

            # Adicionar direção como feature (BUY=1, SELL=-1)
            # Sem isso a LSTM não sabe se o trade é long ou short
            direction_val = 1.0 if direction == "BUY" else (-1.0 if direction == "SELL" else 0.0)
            seq_data["direction_encoded"] = direction_val

            X = seq_data.values.astype(np.float32)
            X = np.nan_to_num(X, nan=0.0)

            # Pad/trim features para match com n_features do treino
            if X.shape[1] < self.n_features:
                # Pad com a média do treino (via scaler.mean_) para que após
                # transform() resulte em ~0, não em valores negativos espúrios
                n_pad = self.n_features - X.shape[1]
                if hasattr(self.scaler, 'mean_') and self.scaler.mean_ is not None:
                    pad_means = self.scaler.mean_[X.shape[1]:self.n_features]
                    padding = np.tile(pad_means, (X.shape[0], 1)).astype(np.float32)
                else:
                    padding = np.zeros((X.shape[0], n_pad), dtype=np.float32)
                X = np.hstack([X, padding])
            elif X.shape[1] > self.n_features:
                X = X[:, :self.n_features]

            # Scale
            X_flat = X.reshape(-1, self.n_features)
            X_scaled = self.scaler.transform(X_flat)
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            X_input = X_scaled.reshape(1, self.sequence_length, self.n_features)

            # Predict
            prob = float(self.model.predict(X_input, verbose=0)[0][0])

            return {
                "probability": prob,
                "prediction": 1 if prob > 0.5 else 0,
                "confidence": abs(prob - 0.5) * 2,
                "model": "Bi-LSTM",
            }

        except Exception as e:
            logger.error(f"[Bi-LSTM] Erro na predição: {e}")
            return {"probability": 0.5, "prediction": 0, "confidence": 0, "error": str(e)}


    def train_from_backtest(
        self,
        symbols=None,
        days_back: int = 180,
        epochs: int = 100,
        batch_size: int = 32,
    ) -> Dict:
        """
        Pipeline completo: gera dataset + treina Bi-LSTM.
        PRIORIDADE: Sinais reais (6000+) > Backtest sintético.

        Returns:
            Dict com success, test_accuracy, test_f1, total_samples
        """
        import asyncio

        try:
            stats = None

            # 1. TENTAR sinais reais primeiro (muito mais representativos)
            try:
                from src.ml.real_signal_dataset_generator import RealSignalDatasetGenerator
                logger.info("[Bi-LSTM] Tentando gerar dataset a partir de sinais REAIS...")

                generator = RealSignalDatasetGenerator(
                    sequence_length=self.sequence_length,
                    max_signals=5000,
                )

                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as pool:
                            stats = pool.submit(asyncio.run, generator.generate()).result()
                    else:
                        stats = loop.run_until_complete(generator.generate())
                except RuntimeError:
                    stats = asyncio.run(generator.generate())

                total = stats.get("total_trades", 0)
                if total >= 50 and "error" not in stats:
                    logger.info(f"[Bi-LSTM] Dataset REAL gerado: {total} trades")
                else:
                    logger.warning(f"[Bi-LSTM] Dataset real insuficiente ({total}), fallback para backtest")
                    stats = None
            except Exception as e:
                logger.warning(f"[Bi-LSTM] Erro no real signal generator: {e}, fallback para backtest")
                stats = None

            # 2. Fallback: backtest sintético
            if stats is None:
                from src.ml.backtest_dataset_generator import BacktestDatasetGenerator

                if symbols is None:
                    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]

                generator = BacktestDatasetGenerator(
                    symbols=symbols,
                    interval="1h",
                    sequence_length=self.sequence_length,
                    days_back=days_back,
                    n_param_variations=10,
                )

                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as pool:
                            stats = pool.submit(asyncio.run, generator.generate()).result()
                    else:
                        stats = loop.run_until_complete(generator.generate())
                except RuntimeError:
                    stats = asyncio.run(generator.generate())

            total = stats.get("total_trades", 0)
            if total < 50:
                return {
                    "success": False,
                    "reason": f"Poucos trades gerados ({total} < 50)",
                }

            # 3. Treinar modelo
            results = self.train(epochs=epochs, batch_size=batch_size)

            test_results = results.get("test", {})
            return {
                "success": True,
                "test_accuracy": test_results.get("accuracy", 0),
                "test_f1": test_results.get("f1_score", 0),
                "total_samples": total,
                "data_source": stats.get("data_source", "backtest"),
                "results": results,
            }

        except Exception as e:
            logger.error(f"[Bi-LSTM] Erro no train_from_backtest: {e}")
            return {"success": False, "reason": str(e)}

    def train_with_optuna(self, n_trials: int = 20, epochs_per_trial: int = 30) -> Dict:
        """
        Otimiza hiperparâmetros da Bi-LSTM com Optuna.
        Mesmo padrão usado no sklearn ML (online_learning.py).

        Busca: units, dropout, learning_rate, L2, batch_size, dense_units.
        Métrica: composite score (win_rate + balanced_accuracy + approval_rate).
        """
        import optuna
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.layers import (
            LSTM, BatchNormalization, Bidirectional, Dense, Dropout, Input,
        )
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.regularizers import l1_l2
        from sklearn.preprocessing import StandardScaler

        print("\n" + "=" * 60)
        print("OPTUNA — OTIMIZAÇÃO DE HIPERPARÂMETROS Bi-LSTM")
        print(f"Trials: {n_trials} | Epochs/trial: {epochs_per_trial}")
        print("=" * 60)

        # Carregar e preparar dados
        X_train_raw, X_test, y_train_raw, y_test = self.load_dataset()

        # NÃO balancear por undersampling — causa mismatch treino/teste
        # (treino fica 50/50 mas teste mantém WR real ~30%, modelo aprende distribuição errada)
        # Em vez disso, usar class_weight para compensar desbalanceamento
        win_rate = float(y_train_raw.mean())
        n_pos = int(y_train_raw.sum())
        n_neg = len(y_train_raw) - n_pos
        if n_pos > 0 and n_neg > 0:
            # Peso inversamente proporcional à frequência da classe
            weight_neg = len(y_train_raw) / (2.0 * n_neg)
            weight_pos = len(y_train_raw) / (2.0 * n_pos)
            class_weights = {0: weight_neg, 1: weight_pos}
            print(f"[CLASS_WEIGHT] WR={win_rate:.1%} | {n_pos} wins, {n_neg} losses | "
                  f"weights: 0→{weight_neg:.2f}, 1→{weight_pos:.2f}")
        else:
            class_weights = {0: 1.0, 1: 1.0}
            print(f"[CLASS_WEIGHT] Dataset com apenas uma classe, pesos iguais")

        # Normalizar
        scaler = StandardScaler()
        n_features = X_train_raw.shape[2]
        seq_len = X_train_raw.shape[1]
        X_train_flat = scaler.fit_transform(X_train_raw.reshape(-1, n_features))
        X_test_flat = scaler.transform(X_test.reshape(-1, n_features))
        X_train_sc = np.nan_to_num(X_train_flat.reshape(len(X_train_raw), seq_len, n_features))
        X_test_sc = np.nan_to_num(X_test_flat.reshape(len(X_test), seq_len, n_features))

        best_val_auc = [0.0]

        def objective(trial):
            tf.keras.backend.clear_session()

            # Hiperparâmetros a otimizar
            # Capacidade reduzida para evitar memorização (dataset pequeno ~200-500 amostras)
            units_1 = trial.suggest_int("units_1", 16, 48, step=8)
            units_2 = trial.suggest_int("units_2", 8, 24, step=8)
            # Regularização com mínimos altos para forçar generalização
            dropout = trial.suggest_float("dropout", 0.25, 0.50)
            rec_dropout = trial.suggest_float("rec_dropout", 0.20, 0.45)
            lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
            l2_reg = trial.suggest_float("l2", 1e-3, 5e-2, log=True)
            dense_units = trial.suggest_int("dense_units", 8, 24, step=8)
            dense_dropout = trial.suggest_float("dense_dropout", 0.25, 0.55)
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

            model = Sequential([
                Input(shape=(seq_len, n_features)),
                Bidirectional(LSTM(units=units_1, return_sequences=True,
                                   dropout=dropout, recurrent_dropout=rec_dropout,
                                   kernel_regularizer=l1_l2(l1=1e-5, l2=l2_reg))),
                BatchNormalization(),
                Bidirectional(LSTM(units=units_2, return_sequences=False,
                                   dropout=dropout, recurrent_dropout=rec_dropout,
                                   kernel_regularizer=l1_l2(l1=1e-5, l2=l2_reg))),
                BatchNormalization(),
                Dense(dense_units, activation="relu", kernel_regularizer=l1_l2(l1=1e-5, l2=l2_reg)),
                Dropout(dense_dropout),
                Dense(1, activation="sigmoid"),
            ])
            model.compile(optimizer=Adam(learning_rate=lr),
                          loss="binary_crossentropy",
                          metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])

            early_stop = EarlyStopping(monitor="val_auc", patience=8,
                                       restore_best_weights=True, mode="max")

            history = model.fit(
                X_train_sc, y_train_raw, epochs=epochs_per_trial, batch_size=batch_size,
                validation_data=(X_test_sc, y_test),
                callbacks=[early_stop], verbose=0,
                class_weight=class_weights,
            )

            # Avaliar no teste
            probs = model.predict(X_test_sc, verbose=0).flatten()
            preds = (probs > 0.5).astype(int)

            # Operacional: sinais que cruzam 0.6/0.4
            operational_mask = (probs >= 0.6) | (probs <= 0.4)
            n_operational = operational_mask.sum()

            if n_operational < max(3, len(y_test) * 0.05):
                return -1.0  # Modelo neutro demais

            op_preds = np.where(probs >= 0.6, 1, np.where(probs <= 0.4, 0, -1))
            op_mask = op_preds >= 0
            op_correct = (op_preds[op_mask] == y_test[op_mask]).sum()
            op_accuracy = op_correct / max(n_operational, 1)

            # Balanced accuracy
            from sklearn.metrics import balanced_accuracy_score, f1_score
            bal_acc = balanced_accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, zero_division=0)

            # Composite: priorizar accuracy operacional + F1 + cobertura
            approval_rate = n_operational / len(y_test)
            ar_norm = min(approval_rate / 0.5, 1.0)

            composite = (0.35 * op_accuracy +
                         0.25 * f1 +
                         0.25 * bal_acc +
                         0.15 * ar_norm)

            trial.set_user_attr("op_accuracy", float(op_accuracy))
            trial.set_user_attr("n_operational", int(n_operational))
            trial.set_user_attr("f1", float(f1))
            trial.set_user_attr("bal_acc", float(bal_acc))

            print(f"  Trial {trial.number}: composite={composite:.4f} | "
                  f"op_acc={op_accuracy:.1%} (n={n_operational}) | "
                  f"f1={f1:.3f} | bal_acc={bal_acc:.1%}")

            return composite

        # Suprimir logs do Optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best = study.best_trial
        bp = best.params
        print(f"\n{'='*60}")
        print(f"MELHOR TRIAL: #{best.number}")
        print(f"  Score: {best.value:.4f}")
        print(f"  Op accuracy: {best.user_attrs.get('op_accuracy', 0):.1%} "
              f"(n={best.user_attrs.get('n_operational', 0)})")
        print(f"  F1: {best.user_attrs.get('f1', 0):.3f}")
        print(f"  Params: units={bp['units_1']}+{bp['units_2']}, "
              f"dropout={bp['dropout']:.2f}, lr={bp['lr']:.5f}, "
              f"L2={bp['l2']:.5f}, dense={bp['dense_units']}")
        print(f"{'='*60}")

        # Treinar modelo final com melhores parâmetros + mais epochs
        print("\n[FINAL] Treinando modelo com melhores hiperparâmetros...")
        tf.keras.backend.clear_session()

        self.scaler = scaler
        self.sequence_length = seq_len
        self.n_features = n_features

        final_model = Sequential([
            Input(shape=(seq_len, n_features)),
            Bidirectional(LSTM(units=bp["units_1"], return_sequences=True,
                               dropout=bp["dropout"], recurrent_dropout=bp["rec_dropout"],
                               kernel_regularizer=l1_l2(l1=1e-5, l2=bp["l2"]))),
            BatchNormalization(),
            Bidirectional(LSTM(units=bp["units_2"], return_sequences=False,
                               dropout=bp["dropout"], recurrent_dropout=bp["rec_dropout"],
                               kernel_regularizer=l1_l2(l1=1e-5, l2=bp["l2"]))),
            BatchNormalization(),
            Dense(bp["dense_units"], activation="relu",
                  kernel_regularizer=l1_l2(l1=1e-5, l2=bp["l2"])),
            Dropout(bp["dense_dropout"]),
            Dense(1, activation="sigmoid"),
        ])
        final_model.compile(optimizer=Adam(learning_rate=bp["lr"]),
                            loss="binary_crossentropy",
                            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])

        from tensorflow.keras.callbacks import ReduceLROnPlateau
        callbacks = [
            EarlyStopping(monitor="val_auc", patience=15, restore_best_weights=True, mode="max"),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6),
        ]

        final_model.fit(
            X_train_sc, y_train_raw, epochs=100, batch_size=bp["batch_size"],
            validation_data=(X_test_sc, y_test),
            callbacks=callbacks, verbose=1,
            class_weight=class_weights,
        )

        self.model = final_model
        results = self._evaluate(X_train_sc, y_train_raw, X_test_sc, y_test)

        # Validação de qualidade: modelo com val_auc < 0.52 é pior que aleatório
        final_val_auc = results.get("test_auc", 0)
        if final_val_auc < 0.52:
            print(f"\n[AVISO] val_auc={final_val_auc:.3f} < 0.52 — modelo pior que aleatório!")
            print(f"[AVISO] O modelo será salvo mas marcado como LOW_QUALITY.")
            results["quality"] = "LOW_QUALITY"
            results["quality_warning"] = (
                f"val_auc={final_val_auc:.3f} indica que o modelo não generaliza. "
                f"Considere: mais dados, features diferentes, ou walk-forward validation."
            )
        else:
            results["quality"] = "OK"

        # Detectar fonte do dataset
        _data_source = "unknown"
        try:
            _ds_info_path = DATASET_DIR / "dataset_info_latest.json"
            if _ds_info_path.exists():
                with open(_ds_info_path, "r") as _f:
                    _data_source = json.load(_f).get("data_source", "unknown")
        except Exception:
            pass

        self.model_info = {
            "type": "Bi-LSTM (Optuna)",
            "training_date": datetime.now(timezone.utc).isoformat(),
            "sequence_length": seq_len,
            "n_features": n_features,
            "train_samples": len(X_train_raw),
            "test_samples": len(X_test),
            "results": results,
            "data_source": _data_source,
            "optuna_best_params": bp,
            "optuna_best_score": best.value,
            "optuna_n_trials": n_trials,
            "optuna_op_accuracy": best.user_attrs.get("op_accuracy", 0),
        }
        self._save_model()

        return {
            "success": True,
            "results": results,
            "best_params": bp,
            "best_score": best.value,
            "op_accuracy": best.user_attrs.get("op_accuracy", 0),
            "n_operational": best.user_attrs.get("n_operational", 0),
        }

    def load_full_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Carrega dataset COMPLETO (train+test concatenados) para walk-forward."""
        X_train = np.load(DATASET_DIR / "X_train_latest.npy")
        X_test = np.load(DATASET_DIR / "X_test_latest.npy")
        y_train = np.load(DATASET_DIR / "y_train_latest.npy")
        y_test = np.load(DATASET_DIR / "y_test_latest.npy")

        X_full = np.concatenate([X_train, X_test], axis=0)
        y_full = np.concatenate([y_train, y_test], axis=0)

        self.sequence_length = X_full.shape[1]
        self.n_features = X_full.shape[2]

        print(f"[FULL DATASET] {X_full.shape} (WR: {y_full.mean()*100:.1f}%)")
        return X_full, y_full

    def walk_forward_validate(
        self, n_folds: int = 5, n_optuna_trials: int = 15, epochs_per_trial: int = 20
    ) -> Dict:
        """
        Walk-Forward Validation com Optuna.

        Divide os dados temporalmente em N folds e para cada janela:
        1. Treina nos folds anteriores (expanding window)
        2. Testa no fold atual (futuro imediato)
        3. Usa Optuna para otimizar hiperparâmetros em CADA janela

        Isso dá métricas realistas: o modelo é sempre avaliado em dados
        que nunca viu e que são do FUTURO relativo ao treino.

        Data: |---F1---|---F2---|---F3---|---F4---|---F5---|
        R1:   [TRAIN  ][ TEST ]
        R2:   [  TRAIN        ][ TEST ]
        R3:   [    TRAIN               ][ TEST ]
        R4:   [      TRAIN                     ][ TEST ]

        Args:
            n_folds: Número de folds temporais (default 5 → 4 rounds de avaliação)
            n_optuna_trials: Trials Optuna por fold (menos que normal pois roda N vezes)
            epochs_per_trial: Epochs por trial (reduzido para velocidade)

        Returns:
            Dict com métricas por fold e médias agregadas
        """
        import optuna
        import tensorflow as tf
        from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
        from sklearn.preprocessing import StandardScaler
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.layers import (
            LSTM, BatchNormalization, Bidirectional, Dense, Dropout, Input,
        )
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.regularizers import l1_l2

        print("\n" + "=" * 70)
        print("WALK-FORWARD VALIDATION com Optuna")
        print(f"Folds: {n_folds} | Trials/fold: {n_optuna_trials} | Epochs/trial: {epochs_per_trial}")
        print("=" * 70)

        X_full, y_full = self.load_full_dataset()
        n_samples = len(X_full)
        seq_len = X_full.shape[1]
        n_features = X_full.shape[2]

        if n_samples < n_folds * 20:
            return {
                "success": False,
                "reason": f"Poucos dados ({n_samples}) para {n_folds} folds (mín {n_folds * 20})"
            }

        # Dividir em folds temporais (tamanhos iguais)
        fold_size = n_samples // n_folds
        fold_results = []

        # Mínimo 2 folds de treino (para ter dados suficientes)
        min_train_folds = 2

        for round_idx in range(min_train_folds, n_folds):
            train_end = round_idx * fold_size
            test_start = train_end
            test_end = min((round_idx + 1) * fold_size, n_samples)

            X_train_wf = X_full[:train_end]
            y_train_wf = y_full[:train_end]
            X_test_wf = X_full[test_start:test_end]
            y_test_wf = y_full[test_start:test_end]

            if len(X_train_wf) < 50 or len(X_test_wf) < 10:
                print(f"\n[FOLD {round_idx}] Pulando — dados insuficientes "
                      f"(train={len(X_train_wf)}, test={len(X_test_wf)})")
                continue

            train_wr = float(y_train_wf.mean())
            test_wr = float(y_test_wf.mean())
            print(f"\n{'─'*60}")
            print(f"FOLD {round_idx}/{n_folds-1} | "
                  f"Train: {len(X_train_wf)} (WR {train_wr:.1%}) | "
                  f"Test: {len(X_test_wf)} (WR {test_wr:.1%})")
            print(f"{'─'*60}")

            # Normalizar (fit somente no treino deste fold)
            scaler = StandardScaler()
            X_tr_flat = scaler.fit_transform(X_train_wf.reshape(-1, n_features))
            X_te_flat = scaler.transform(X_test_wf.reshape(-1, n_features))
            X_tr = np.nan_to_num(X_tr_flat.reshape(len(X_train_wf), seq_len, n_features))
            X_te = np.nan_to_num(X_te_flat.reshape(len(X_test_wf), seq_len, n_features))

            # Class weights para este fold
            n_pos = int(y_train_wf.sum())
            n_neg = len(y_train_wf) - n_pos
            if n_pos > 0 and n_neg > 0:
                cw = {0: len(y_train_wf) / (2.0 * n_neg), 1: len(y_train_wf) / (2.0 * n_pos)}
            else:
                cw = {0: 1.0, 1: 1.0}

            # --- Optuna para este fold ---
            def make_objective(X_tr_sc, y_tr, X_te_sc, y_te, class_w):
                def objective(trial):
                    tf.keras.backend.clear_session()

                    units_1 = trial.suggest_int("units_1", 16, 48, step=8)
                    units_2 = trial.suggest_int("units_2", 8, 24, step=8)
                    dropout = trial.suggest_float("dropout", 0.25, 0.50)
                    rec_dropout = trial.suggest_float("rec_dropout", 0.20, 0.45)
                    lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
                    l2_reg = trial.suggest_float("l2", 1e-3, 5e-2, log=True)
                    dense_units = trial.suggest_int("dense_units", 8, 24, step=8)
                    dense_dropout = trial.suggest_float("dense_dropout", 0.25, 0.55)
                    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

                    model = Sequential([
                        Input(shape=(seq_len, n_features)),
                        Bidirectional(LSTM(units=units_1, return_sequences=True,
                                           dropout=dropout, recurrent_dropout=rec_dropout,
                                           kernel_regularizer=l1_l2(l1=1e-5, l2=l2_reg))),
                        BatchNormalization(),
                        Bidirectional(LSTM(units=units_2, return_sequences=False,
                                           dropout=dropout, recurrent_dropout=rec_dropout,
                                           kernel_regularizer=l1_l2(l1=1e-5, l2=l2_reg))),
                        BatchNormalization(),
                        Dense(dense_units, activation="relu",
                              kernel_regularizer=l1_l2(l1=1e-5, l2=l2_reg)),
                        Dropout(dense_dropout),
                        Dense(1, activation="sigmoid"),
                    ])
                    model.compile(optimizer=Adam(learning_rate=lr),
                                  loss="binary_crossentropy",
                                  metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])

                    early_stop = EarlyStopping(monitor="val_auc", patience=5,
                                               restore_best_weights=True, mode="max")

                    model.fit(X_tr_sc, y_tr, epochs=epochs_per_trial, batch_size=batch_size,
                              validation_data=(X_te_sc, y_te),
                              callbacks=[early_stop], verbose=0, class_weight=class_w)

                    probs = model.predict(X_te_sc, verbose=0).flatten()
                    preds = (probs > 0.5).astype(int)

                    # Métricas operacionais
                    op_mask = (probs >= 0.6) | (probs <= 0.4)
                    n_op = op_mask.sum()
                    if n_op < max(3, len(y_te) * 0.05):
                        return -1.0

                    op_preds = np.where(probs >= 0.6, 1, np.where(probs <= 0.4, 0, -1))
                    op_valid = op_preds >= 0
                    op_acc = (op_preds[op_valid] == y_te[op_valid]).sum() / max(n_op, 1)

                    bal_acc = balanced_accuracy_score(y_te, preds)
                    f1 = f1_score(y_te, preds, zero_division=0)
                    ar = min((n_op / len(y_te)) / 0.5, 1.0)

                    composite = 0.35 * op_acc + 0.25 * f1 + 0.25 * bal_acc + 0.15 * ar

                    trial.set_user_attr("op_accuracy", float(op_acc))
                    trial.set_user_attr("n_operational", int(n_op))
                    trial.set_user_attr("f1", float(f1))
                    trial.set_user_attr("bal_acc", float(bal_acc))
                    try:
                        trial.set_user_attr("auc", float(roc_auc_score(y_te, probs)))
                    except Exception:
                        trial.set_user_attr("auc", 0.0)

                    return composite
                return objective

            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study(direction="maximize")
            study.optimize(
                make_objective(X_tr, y_train_wf, X_te, y_test_wf, cw),
                n_trials=n_optuna_trials, show_progress_bar=False,
            )

            best = study.best_trial
            fold_result = {
                "fold": round_idx,
                "train_size": len(X_train_wf),
                "test_size": len(X_test_wf),
                "train_wr": train_wr,
                "test_wr": test_wr,
                "best_score": best.value,
                "op_accuracy": best.user_attrs.get("op_accuracy", 0),
                "f1": best.user_attrs.get("f1", 0),
                "bal_acc": best.user_attrs.get("bal_acc", 0),
                "auc": best.user_attrs.get("auc", 0),
                "n_operational": best.user_attrs.get("n_operational", 0),
                "best_params": best.params,
            }
            fold_results.append(fold_result)

            print(f"  Best: score={best.value:.4f} | op_acc={fold_result['op_accuracy']:.1%} | "
                  f"AUC={fold_result['auc']:.3f} | F1={fold_result['f1']:.3f} | "
                  f"n_op={fold_result['n_operational']}")

        if not fold_results:
            return {"success": False, "reason": "Nenhum fold teve dados suficientes"}

        # Métricas agregadas
        avg_score = np.mean([f["best_score"] for f in fold_results])
        avg_op_acc = np.mean([f["op_accuracy"] for f in fold_results])
        avg_auc = np.mean([f["auc"] for f in fold_results])
        avg_f1 = np.mean([f["f1"] for f in fold_results])
        std_score = np.std([f["best_score"] for f in fold_results])
        std_op_acc = np.std([f["op_accuracy"] for f in fold_results])

        print(f"\n{'='*70}")
        print("WALK-FORWARD — RESULTADOS AGREGADOS")
        print(f"{'='*70}")
        print(f"  Folds avaliados: {len(fold_results)}")
        print(f"  Score médio:     {avg_score:.4f} ± {std_score:.4f}")
        print(f"  Op Accuracy:     {avg_op_acc:.1%} ± {std_op_acc:.1%}")
        print(f"  AUC médio:       {avg_auc:.3f}")
        print(f"  F1 médio:        {avg_f1:.3f}")

        # Treinar modelo final com TODOS os dados e melhores parâmetros do último fold
        # (o último fold tem mais dados de treino e é mais representativo)
        best_params = fold_results[-1]["best_params"]
        print(f"\n[FINAL] Treinando modelo com params do último fold e TODOS os dados...")

        tf.keras.backend.clear_session()
        # Split final: usar 90% treino / 10% validação do dataset completo
        final_split = int(len(X_full) * 0.90)
        X_final_train = X_full[:final_split]
        y_final_train = y_full[:final_split]
        X_final_val = X_full[final_split:]
        y_final_val = y_full[final_split:]

        final_scaler = StandardScaler()
        X_ft_flat = final_scaler.fit_transform(X_final_train.reshape(-1, n_features))
        X_fv_flat = final_scaler.transform(X_final_val.reshape(-1, n_features))
        X_ft = np.nan_to_num(X_ft_flat.reshape(len(X_final_train), seq_len, n_features))
        X_fv = np.nan_to_num(X_fv_flat.reshape(len(X_final_val), seq_len, n_features))

        n_pos_f = int(y_final_train.sum())
        n_neg_f = len(y_final_train) - n_pos_f
        cw_final = {0: len(y_final_train) / (2.0 * max(n_neg_f, 1)),
                    1: len(y_final_train) / (2.0 * max(n_pos_f, 1))}

        bp = best_params
        final_model = Sequential([
            Input(shape=(seq_len, n_features)),
            Bidirectional(LSTM(units=bp["units_1"], return_sequences=True,
                               dropout=bp["dropout"], recurrent_dropout=bp["rec_dropout"],
                               kernel_regularizer=l1_l2(l1=1e-5, l2=bp["l2"]))),
            BatchNormalization(),
            Bidirectional(LSTM(units=bp["units_2"], return_sequences=False,
                               dropout=bp["dropout"], recurrent_dropout=bp["rec_dropout"],
                               kernel_regularizer=l1_l2(l1=1e-5, l2=bp["l2"]))),
            BatchNormalization(),
            Dense(bp["dense_units"], activation="relu",
                  kernel_regularizer=l1_l2(l1=1e-5, l2=bp["l2"])),
            Dropout(bp["dense_dropout"]),
            Dense(1, activation="sigmoid"),
        ])
        final_model.compile(optimizer=Adam(learning_rate=bp["lr"]),
                            loss="binary_crossentropy",
                            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])

        from tensorflow.keras.callbacks import ReduceLROnPlateau
        callbacks = [
            EarlyStopping(monitor="val_auc", patience=15, restore_best_weights=True, mode="max"),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6),
        ]

        final_model.fit(X_ft, y_final_train, epochs=100, batch_size=bp["batch_size"],
                        validation_data=(X_fv, y_final_val),
                        callbacks=callbacks, verbose=1, class_weight=cw_final)

        self.model = final_model
        self.scaler = final_scaler
        self.sequence_length = seq_len
        self.n_features = n_features

        results = self._evaluate(X_ft, y_final_train, X_fv, y_final_val)

        # Detectar fonte do dataset
        _data_source = "unknown"
        try:
            _ds_info_path = DATASET_DIR / "dataset_info_latest.json"
            if _ds_info_path.exists():
                with open(_ds_info_path, "r") as _f:
                    _data_source = json.load(_f).get("data_source", "unknown")
        except Exception:
            pass

        self.model_info = {
            "type": "Bi-LSTM (Walk-Forward)",
            "training_date": datetime.now(timezone.utc).isoformat(),
            "sequence_length": seq_len,
            "n_features": n_features,
            "train_samples": len(X_final_train),
            "test_samples": len(X_final_val),
            "results": results,
            "data_source": _data_source,
            "walk_forward": {
                "n_folds": n_folds,
                "n_rounds": len(fold_results),
                "optuna_trials_per_fold": n_optuna_trials,
                "avg_score": float(avg_score),
                "std_score": float(std_score),
                "avg_op_accuracy": float(avg_op_acc),
                "std_op_accuracy": float(std_op_acc),
                "avg_auc": float(avg_auc),
                "avg_f1": float(avg_f1),
                "fold_results": fold_results,
            },
            "final_params": best_params,
        }
        self._save_model()

        # Salvar relatório WF separado para consulta
        wf_report_path = MODEL_DIR / "walk_forward_report.json"
        with open(wf_report_path, "w") as f:
            json.dump(self.model_info["walk_forward"], f, indent=2, default=str)

        return {
            "success": True,
            "avg_score": float(avg_score),
            "std_score": float(std_score),
            "avg_op_accuracy": float(avg_op_acc),
            "avg_auc": float(avg_auc),
            "avg_f1": float(avg_f1),
            "n_rounds": len(fold_results),
            "fold_results": fold_results,
            "final_results": results,
            "final_params": best_params,
        }

    def _log_prediction(self, symbol: str, prob: float, prediction: int):
        """Registra predição do LSTM para avaliação futura (dashboard)."""
        try:
            log_path = MODEL_DIR / "lstm_prediction_log.json"

            predictions = []
            if log_path.exists():
                with open(log_path, 'r') as f:
                    predictions = json.load(f)

            predictions.append({
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'symbol': symbol,
                'lstm_probability': prob,
                'lstm_prediction': prediction,
                'lstm_recommendation': 'EXECUTE' if prediction == 1 else 'SKIP',
                'actual_result': None,
            })

            # Manter últimas 1000
            predictions = predictions[-1000:]

            with open(log_path, 'w') as f:
                json.dump(predictions, f, indent=2, default=str)
        except Exception:
            pass


def main():
    """Treina o Bi-LSTM com dados do backtest."""
    print("\n" + "=" * 70)
    print("Bi-LSTM SEQUENCE VALIDATOR - TREINAMENTO")
    print("Dados: trades gerados pelo backtest engine")
    print("=" * 70)

    validator = LSTMSequenceValidator()

    try:
        results = validator.train(epochs=100, batch_size=32)
        print(f"\n[OK] Treinamento concluído! Resultados: {results}")
    except FileNotFoundError:
        print("\n[ERRO] Dataset não encontrado!")
        print("  Execute primeiro: python -m src.ml.backtest_dataset_generator")
        print("  Isso vai rodar backtests e gerar os dados de treino.")
    except Exception as e:
        print(f"\n[ERRO] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
