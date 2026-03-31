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

        model = Sequential([
            Input(shape=(self.sequence_length, self.n_features)),

            # Bi-LSTM Layer 1 - captura padrões em ambas as direções
            Bidirectional(LSTM(
                units=64,
                return_sequences=True,
                dropout=0.2,
                recurrent_dropout=0.2,
                kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
            ), name="bilstm_1"),
            BatchNormalization(),

            # Bi-LSTM Layer 2
            Bidirectional(LSTM(
                units=32,
                return_sequences=False,
                dropout=0.2,
                recurrent_dropout=0.2,
                kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
            ), name="bilstm_2"),
            BatchNormalization(),

            # Dense layers
            Dense(32, activation="relu", kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
            Dropout(0.3),
            Dense(16, activation="relu"),
            Dropout(0.2),

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
        print("TREINAMENTO Bi-LSTM (dados do backtest)")
        print("=" * 60)

        X_train, X_test, y_train, y_test = self.load_dataset()

        # Normalizar features por candle (StandardScaler por feature)
        from sklearn.preprocessing import StandardScaler

        # Reshape para 2D, normalizar, reshape de volta
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]

        X_all = np.vstack([
            X_train.reshape(-1, self.n_features),
            X_test.reshape(-1, self.n_features),
        ])

        self.scaler = StandardScaler()
        X_all_scaled = self.scaler.fit_transform(X_all)

        X_train_scaled = X_all_scaled[: n_train * self.sequence_length].reshape(
            n_train, self.sequence_length, self.n_features
        )
        X_test_scaled = X_all_scaled[n_train * self.sequence_length :].reshape(
            n_test, self.sequence_length, self.n_features
        )

        # Tratar NaN/Inf
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        # Construir modelo
        self.build_model()

        # Class weights para lidar com desbalanceamento
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        if n_pos > 0 and n_neg > 0:
            class_weight = {0: len(y_train) / (2 * n_neg), 1: len(y_train) / (2 * n_pos)}
        else:
            class_weight = None

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

        print(f"\n[TRAIN] Iniciando... (class_weight={class_weight})")
        start = datetime.now(timezone.utc)

        history = self.model.fit(
            X_train_scaled, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test_scaled, y_test),
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1,
        )

        training_time = (datetime.now(timezone.utc) - start).total_seconds()

        # Avaliar
        results = self._evaluate(X_train_scaled, y_train, X_test_scaled, y_test)

        self.model_info = {
            "type": "Bi-LSTM",
            "training_date": datetime.now(timezone.utc).isoformat(),
            "training_time_seconds": training_time,
            "epochs_run": len(history.history["loss"]),
            "sequence_length": self.sequence_length,
            "n_features": self.n_features,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "results": results,
            "data_source": "backtest_dataset_generator",
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
            self.n_features = self.model_info.get("n_features", 17)

            logger.info(f"[Bi-LSTM] Modelo carregado (seq={self.sequence_length}, features={self.n_features})")
            return True

        except Exception as e:
            logger.warning(f"[Bi-LSTM] Erro ao carregar modelo: {e}")
            return False

    def predict_from_candles(self, candles_df) -> Dict:
        """
        Prediz probabilidade de sucesso a partir de candles recentes.

        Usado em produção: recebe os últimos N candles com indicadores
        e retorna probabilidade de win.

        Args:
            candles_df: DataFrame com últimos sequence_length candles
                       (deve ter as mesmas features do treino)

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

            X = seq_data.values.astype(np.float32)
            X = np.nan_to_num(X, nan=0.0)

            # Pad features se necessário
            if X.shape[1] < self.n_features:
                padding = np.zeros((X.shape[0], self.n_features - X.shape[1]), dtype=np.float32)
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
        Pipeline completo: gera dataset via backtest + treina Bi-LSTM.
        Chamado pelo auto-training em main.py a cada ciclo.

        Returns:
            Dict com success, test_accuracy, test_f1, total_samples
        """
        import asyncio

        try:
            # 1. Gerar dataset de treino via backtest
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

            # Rodar backtest (async → sync bridge)
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        stats = pool.submit(
                            asyncio.run, generator.generate()
                        ).result()
                else:
                    stats = loop.run_until_complete(generator.generate())
            except RuntimeError:
                stats = asyncio.run(generator.generate())

            total = stats.get("total_trades", 0)
            if total < 50:
                return {
                    "success": False,
                    "reason": f"Poucos trades gerados pelo backtest ({total} < 50)",
                }

            logger.info(f"[Bi-LSTM] Dataset gerado: {total} trades")

            # 2. Treinar modelo
            results = self.train(epochs=epochs, batch_size=batch_size)

            test_results = results.get("test", {})
            return {
                "success": True,
                "test_accuracy": test_results.get("accuracy", 0),
                "test_f1": test_results.get("f1_score", 0),
                "total_samples": total,
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
