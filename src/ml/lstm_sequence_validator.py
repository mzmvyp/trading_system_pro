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

        # Arquitetura balanceada para ~4000-5000 amostras
        # V1 (64+32, dropout 0.2): overfitting severo (81% treino vs 59% teste)
        # V2 (32+16, dropout 0.4): sem overfitting mas range colapsou (0.32-0.51)
        # V3 (48+24, dropout 0.3): meio-termo — generaliza sem colapsar probabilidades
        model = Sequential([
            Input(shape=(self.sequence_length, self.n_features)),

            # Bi-LSTM Layer 1 (48 units — entre 64 e 32)
            Bidirectional(LSTM(
                units=48,
                return_sequences=True,
                dropout=0.25,
                recurrent_dropout=0.25,
                kernel_regularizer=l1_l2(l1=1e-5, l2=5e-4),
            ), name="bilstm_1"),
            BatchNormalization(),

            # Bi-LSTM Layer 2 (24 units — entre 32 e 16)
            Bidirectional(LSTM(
                units=24,
                return_sequences=False,
                dropout=0.25,
                recurrent_dropout=0.25,
                kernel_regularizer=l1_l2(l1=1e-5, l2=5e-4),
            ), name="bilstm_2"),
            BatchNormalization(),

            # Dense layer
            Dense(24, activation="relu", kernel_regularizer=l1_l2(l1=1e-5, l2=5e-4)),
            Dropout(0.35),

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
        # BALANCEAR TREINO: igualar wins e losses
        # ========================================
        # Sem balanceamento, 71% losses → modelo prevê "loss" pra tudo
        # Balancear ANTES de normalizar para não desperdiçar cálculo
        win_rate = float(y_train.mean())
        if win_rate < 0.40 or win_rate > 0.60:
            idx_win = np.where(y_train == 1)[0]
            idx_loss = np.where(y_train == 0)[0]
            n_min = min(len(idx_win), len(idx_loss))
            if n_min >= 20:
                rng = np.random.RandomState(42)
                if len(idx_win) > n_min:
                    idx_win = rng.choice(idx_win, size=n_min, replace=False)
                if len(idx_loss) > n_min:
                    idx_loss = rng.choice(idx_loss, size=n_min, replace=False)
                balanced_idx = np.sort(np.concatenate([idx_win, idx_loss]))
                X_train = X_train[balanced_idx]
                y_train = y_train[balanced_idx]
                print(f"[BALANCE] Treino balanceado: {len(idx_win)} wins + {len(idx_loss)} losses = {len(X_train)} "
                      f"(era {win_rate*100:.0f}% → agora 50%)")
            else:
                print(f"[BALANCE] Poucos exemplos da classe minoritária ({n_min}) — mantendo desbalanceado")
        else:
            print(f"[BALANCE] Treino já balanceado ({win_rate*100:.0f}% win rate)")

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

        # class_weight REMOVIDO: o dataset já é balanceado por subsampling
        # Usar ambos (balance + class_weight) empurrava logits → 0 → sigmoid(0) = 0.5
        # Resultado: probs 0.43-0.55 sempre, modelo nunca votava operacionalmente

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

        print(f"\n[TRAIN] Iniciando... (hard examples: {n_hard}, sem class_weight)")
        start = datetime.now(timezone.utc)

        history = self.model.fit(
            X_train_scaled, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test_scaled, y_test),
            callbacks=callbacks,
            sample_weight=sample_weight,
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
        """Avalia modelo no treino e teste."""
        from sklearn.metrics import accuracy_score, classification_report, f1_score

        results = {}

        y_train_prob = self.model.predict(X_train, verbose=0).flatten()
        y_train_pred = (y_train_prob > 0.5).astype(int)
        results["train"] = {
            "accuracy": float(accuracy_score(y_train, y_train_pred)),
            "f1_score": float(f1_score(y_train, y_train_pred, zero_division=0)),
        }

        y_test_prob = self.model.predict(X_test, verbose=0).flatten()
        y_test_pred = (y_test_prob > 0.5).astype(int)
        results["test"] = {
            "accuracy": float(accuracy_score(y_test, y_test_pred)),
            "f1_score": float(f1_score(y_test, y_test_pred, zero_division=0)),
        }

        print(f"\n{'='*60}")
        print("RESULTADOS")
        print(f"{'='*60}")
        print(f"  Train: acc={results['train']['accuracy']*100:.1f}%, f1={results['train']['f1_score']:.3f}")
        print(f"  Test:  acc={results['test']['accuracy']*100:.1f}%, f1={results['test']['f1_score']:.3f}")
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
