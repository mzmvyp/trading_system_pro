"""
LSTM Signal Validator - Modelo para validar sinais do DeepSeek
==============================================================

Reutiliza a arquitetura LSTM do tecchallenge_4, adaptado para:
- Classificacao binaria (TP vs SL/NO_TRADE)
- Features do DeepSeek (RSI, MACD, ADX, etc)

Autor: Trading Bot
Data: 2026-01-13
"""

import json
import os
import pickle
import warnings
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from tensorflow import keras  # noqa: E402
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # noqa: E402
from tensorflow.keras.layers import LSTM, BatchNormalization, Dense, Dropout, Input  # noqa: E402
from tensorflow.keras.models import Sequential  # noqa: E402
from tensorflow.keras.optimizers import Adam  # noqa: E402
from tensorflow.keras.regularizers import l1_l2  # noqa: E402

warnings.filterwarnings('ignore')

# Configuracoes
CONFIG = {
    "model_dir": "ml_models",
    "dataset_dir": "ml_dataset",
    "sequence_length": 1,  # Cada sinal e independente (nao sequencial por enquanto)
    "test_size": 0.2,
    "random_state": 42,
}


class LSTMSignalValidator:
    """
    Modelo LSTM para validar sinais de trading do DeepSeek.
    Baseado na arquitetura do tecchallenge_4.
    """

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.model_info = {}

        # Criar diretorio de modelos
        os.makedirs(CONFIG["model_dir"], exist_ok=True)

    def load_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Carrega os datasets de treino e teste"""
        train_path = os.path.join(CONFIG["dataset_dir"], "dataset_train_latest.csv")
        test_path = os.path.join(CONFIG["dataset_dir"], "dataset_test_latest.csv")

        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Dataset de treino nao encontrado: {train_path}")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path) if os.path.exists(test_path) else None

        print(f"[OK] Dataset carregado: {len(train_df)} treino, {len(test_df) if test_df is not None else 0} teste")

        return train_df, test_df

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepara features para o modelo"""
        # Features numericas
        self.feature_columns = [
            'rsi', 'macd_histogram', 'adx', 'atr', 'bb_position',
            'cvd', 'orderbook_imbalance', 'bullish_tf_count', 'bearish_tf_count',
            'confidence', 'trend_encoded', 'sentiment_encoded', 'signal_encoded',
            'risk_distance_pct', 'reward_distance_pct', 'risk_reward_ratio'
        ]

        # Filtrar apenas colunas que existem
        available_cols = [col for col in self.feature_columns if col in df.columns]
        self.feature_columns = available_cols

        X = df[self.feature_columns].values
        y = df['target'].values

        return X, y

    def build_model(self, n_features: int) -> Sequential:
        """
        Constroi modelo LSTM adaptado para classificacao de sinais.
        Arquitetura baseada no tecchallenge_4.
        """
        print("\n[BUILD] Construindo modelo LSTM...")

        model = Sequential([
            # Input layer
            Input(shape=(1, n_features)),

            # LSTM Layer 1
            LSTM(
                units=64,
                return_sequences=True,
                dropout=0.25,
                recurrent_dropout=0.25,
                kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001),
                recurrent_regularizer=l1_l2(l1=0.0001, l2=0.0001),
                name='lstm_1'
            ),
            BatchNormalization(name='bn_lstm_1'),

            # LSTM Layer 2
            LSTM(
                units=32,
                return_sequences=True,
                dropout=0.25,
                recurrent_dropout=0.25,
                kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001),
                recurrent_regularizer=l1_l2(l1=0.0001, l2=0.0001),
                name='lstm_2'
            ),
            BatchNormalization(name='bn_lstm_2'),

            # LSTM Layer 3
            LSTM(
                units=16,
                return_sequences=False,
                dropout=0.25,
                recurrent_dropout=0.25,
                kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001),
                recurrent_regularizer=l1_l2(l1=0.0001, l2=0.0001),
                name='lstm_3'
            ),
            BatchNormalization(name='bn_lstm_3'),

            # Dense Layer 1
            Dense(16, activation='relu', kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001), name='dense_1'),
            BatchNormalization(name='bn_dense_1'),
            Dropout(0.2, name='dropout_1'),

            # Dense Layer 2
            Dense(8, activation='relu', kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001), name='dense_2'),
            Dropout(0.2, name='dropout_2'),

            # Output - Classificacao binaria (sigmoid)
            Dense(1, activation='sigmoid', name='output')
        ])

        # Compilar com Binary Crossentropy (classificacao)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )

        print(f"[OK] Modelo construido: {model.count_params():,} parametros")

        self.model = model
        return model

    def train(self, epochs: int = 100, batch_size: int = 8) -> Dict:
        """Treina o modelo"""
        print("\n" + "="*60)
        print("TREINAMENTO DO MODELO LSTM")
        print("="*60)

        # Carregar dados
        train_df, test_df = self.load_dataset()

        # Preparar features
        X_train, y_train = self.prepare_features(train_df)
        X_test, y_test = self.prepare_features(test_df) if test_df is not None else (None, None)

        print(f"[DATA] Features: {len(self.feature_columns)}")
        print(f"[DATA] Treino: {len(X_train)} amostras")
        if X_test is not None:
            print(f"[DATA] Teste: {len(X_test)} amostras")

        # Normalizar features
        X_train_scaled = self.scaler.fit_transform(X_train)
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)

        # Reshape para LSTM (samples, timesteps, features)
        X_train_lstm = X_train_scaled.reshape(-1, 1, X_train_scaled.shape[1])
        if X_test is not None:
            X_test_lstm = X_test_scaled.reshape(-1, 1, X_test_scaled.shape[1])

        # Construir modelo
        self.build_model(n_features=X_train_scaled.shape[1])

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_test is not None else 'loss',
                patience=15,
                restore_best_weights=True,
                min_delta=0.0001
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_test is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001
            )
        ]

        # Treinar
        print("\n[TRAIN] Iniciando treinamento...")
        start_time = datetime.now()

        validation_data = (X_test_lstm, y_test) if X_test is not None else None

        history = self.model.fit(
            X_train_lstm, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )

        training_time = (datetime.now() - start_time).total_seconds()

        # Avaliar
        results = self._evaluate(X_train_lstm, y_train, X_test_lstm, y_test)

        # Salvar info
        self.model_info = {
            "training_date": datetime.now().isoformat(),
            "training_time_seconds": training_time,
            "epochs_run": len(history.history['loss']),
            "final_loss": float(history.history['loss'][-1]),
            "final_accuracy": float(history.history['accuracy'][-1]),
            "feature_columns": self.feature_columns,
            "n_features": len(self.feature_columns),
            "train_samples": len(X_train),
            "test_samples": len(X_test) if X_test is not None else 0,
            "results": results
        }

        # Salvar modelo
        self._save_model()

        return results

    def _evaluate(self, X_train, y_train, X_test, y_test) -> Dict:
        """Avalia o modelo"""
        print("\n" + "="*60)
        print("AVALIACAO DO MODELO")
        print("="*60)

        results = {}

        # Predicoes treino
        y_train_pred_prob = self.model.predict(X_train, verbose=0)
        y_train_pred = (y_train_pred_prob > 0.5).astype(int).flatten()

        results['train'] = {
            'accuracy': float(accuracy_score(y_train, y_train_pred)),
            'f1_score': float(f1_score(y_train, y_train_pred, zero_division=0))
        }

        print(f"\n[TREINO] Accuracy: {results['train']['accuracy']*100:.1f}%")
        print(f"[TREINO] F1-Score: {results['train']['f1_score']:.3f}")

        # Predicoes teste
        if X_test is not None and len(X_test) > 0:
            y_test_pred_prob = self.model.predict(X_test, verbose=0)
            y_test_pred = (y_test_pred_prob > 0.5).astype(int).flatten()

            results['test'] = {
                'accuracy': float(accuracy_score(y_test, y_test_pred)),
                'f1_score': float(f1_score(y_test, y_test_pred, zero_division=0))
            }

            print(f"\n[TESTE] Accuracy: {results['test']['accuracy']*100:.1f}%")
            print(f"[TESTE] F1-Score: {results['test']['f1_score']:.3f}")

            # Classification Report
            print("\n[TESTE] Classification Report:")
            print(classification_report(y_test, y_test_pred, target_names=['Loss/NoTrade', 'TP']))

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_test_pred)
            print("\n[TESTE] Confusion Matrix:")
            print("         Pred: Loss  TP")
            print(f"  Real Loss:   {cm[0][0]:4d}  {cm[0][1]:4d}")
            print(f"  Real TP:     {cm[1][0]:4d}  {cm[1][1]:4d}")

        return results

    def _save_model(self):
        """Salva modelo e artefatos"""
        print("\n[SAVE] Salvando modelo...")

        # Modelo Keras
        model_path = os.path.join(CONFIG["model_dir"], "lstm_signal_validator.h5")
        self.model.save(model_path)
        print(f"  [OK] Modelo: {model_path}")

        # Scaler
        scaler_path = os.path.join(CONFIG["model_dir"], "scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"  [OK] Scaler: {scaler_path}")

        # Metadata
        info_path = os.path.join(CONFIG["model_dir"], "model_info.json")
        with open(info_path, 'w') as f:
            json.dump(self.model_info, f, indent=2, default=str)
        print(f"  [OK] Info: {info_path}")

    def load_model(self):
        """Carrega modelo salvo"""
        model_path = os.path.join(CONFIG["model_dir"], "lstm_signal_validator.h5")
        scaler_path = os.path.join(CONFIG["model_dir"], "scaler.pkl")
        info_path = os.path.join(CONFIG["model_dir"], "model_info.json")

        self.model = keras.models.load_model(model_path)

        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        with open(info_path, 'r') as f:
            self.model_info = json.load(f)
            self.feature_columns = self.model_info.get('feature_columns', [])

        print(f"[OK] Modelo carregado de {model_path}")

    def predict_signal(self, features: Dict[str, float]) -> Dict:
        """
        Faz predicao para um novo sinal.

        Args:
            features: Dicionario com features do sinal

        Returns:
            Dicionario com probabilidade e recomendacao
        """
        if self.model is None:
            self.load_model()

        # Preparar features
        X = np.array([[features.get(col, 0) for col in self.feature_columns]])
        X_scaled = self.scaler.transform(X)
        X_lstm = X_scaled.reshape(-1, 1, X_scaled.shape[1])

        # Predicao
        prob = float(self.model.predict(X_lstm, verbose=0)[0][0])

        return {
            'probability_success': prob,
            'recommendation': 'EXECUTE' if prob > 0.5 else 'SKIP',
            'confidence': abs(prob - 0.5) * 2  # 0 a 1, quanto mais longe de 0.5, mais confiante
        }


def main():
    """Funcao principal - treinar modelo"""
    print("\n" + "="*70)
    print("LSTM SIGNAL VALIDATOR - TREINAMENTO")
    print("="*70)

    validator = LSTMSignalValidator()

    try:
        validator.train(epochs=100, batch_size=8)

        print("\n" + "="*70)
        print("[OK] TREINAMENTO CONCLUIDO!")
        print("="*70)

    except Exception as e:
        print(f"\n[ERRO] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

