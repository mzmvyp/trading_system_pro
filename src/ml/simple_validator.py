"""
Simple Signal Validator - Modelo para validar sinais do DeepSeek
================================================================

Versao simplificada com tratamento robusto de dados.
Usa modelo mais simples (Dense) que funciona melhor com poucos dados.

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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

warnings.filterwarnings('ignore')

# Configuracoes
CONFIG = {
    "model_dir": "ml_models",
    "dataset_dir": "ml_dataset",
}


class SimpleSignalValidator:
    """
    Modelo simples para validar sinais de trading.
    Usa ensemble de modelos sklearn (mais robusto com poucos dados).
    """

    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.model_info = {}
        self.best_model_name = None

        os.makedirs(CONFIG["model_dir"], exist_ok=True)

    def load_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Carrega os datasets"""
        train_path = os.path.join(CONFIG["dataset_dir"], "dataset_train_latest.csv")
        test_path = os.path.join(CONFIG["dataset_dir"], "dataset_test_latest.csv")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path) if os.path.exists(test_path) else None

        print(f"[OK] Dataset: {len(train_df)} treino, {len(test_df) if test_df is not None else 0} teste")

        return train_df, test_df

    def prepare_features(self, df: pd.DataFrame, fit_scaler: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Prepara features com tratamento robusto"""
        # Features
        self.feature_columns = [
            'rsi', 'macd_histogram', 'adx', 'atr', 'bb_position',
            'cvd', 'orderbook_imbalance', 'bullish_tf_count', 'bearish_tf_count',
            'confidence', 'trend_encoded', 'sentiment_encoded', 'signal_encoded',
            'risk_distance_pct', 'reward_distance_pct', 'risk_reward_ratio'
        ]

        available_cols = [col for col in self.feature_columns if col in df.columns]
        self.feature_columns = available_cols

        X = df[self.feature_columns].copy()
        y = df['target'].values

        # Tratamento robusto de dados
        # 1. Substituir infinitos por NaN
        X = X.replace([np.inf, -np.inf], np.nan)

        # 2. Preencher NaN com mediana
        for col in X.columns:
            median_val = X[col].median()
            if pd.isna(median_val):
                median_val = 0
            X[col] = X[col].fillna(median_val)

        # 3. Clip valores extremos (3 desvios padrao)
        for col in X.columns:
            mean = X[col].mean()
            std = X[col].std()
            if std > 0:
                X[col] = X[col].clip(mean - 3*std, mean + 3*std)

        X = X.values

        # Normalizar
        if fit_scaler:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)

        return X, y

    def train(self) -> Dict:
        """Treina multiplos modelos e seleciona o melhor"""
        print("\n" + "="*60)
        print("TREINAMENTO - SIMPLE SIGNAL VALIDATOR")
        print("="*60)

        # Carregar dados
        train_df, test_df = self.load_dataset()

        # Preparar features
        X_train, y_train = self.prepare_features(train_df, fit_scaler=True)
        X_test, y_test = self.prepare_features(test_df, fit_scaler=False) if test_df is not None else (None, None)

        print(f"[DATA] Features: {len(self.feature_columns)}")
        print(f"[DATA] Treino: {len(X_train)} amostras (TP: {sum(y_train)}, Loss: {len(y_train)-sum(y_train)})")
        if X_test is not None:
            print(f"[DATA] Teste: {len(X_test)} amostras (TP: {sum(y_test)}, Loss: {len(y_test)-sum(y_test)})")

        # Definir modelos
        self.models = {
            'LogisticRegression': LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                class_weight='balanced',
                random_state=42
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                random_state=42
            ),
            'MLP': MLPClassifier(
                hidden_layer_sizes=(32, 16),
                max_iter=1000,
                random_state=42,
                early_stopping=True
            )
        }

        results = {}
        best_score = -1

        print("\n[TRAIN] Treinando modelos...")
        print("-" * 50)

        for name, model in self.models.items():
            try:
                # Treinar
                model.fit(X_train, y_train)

                # Cross-validation no treino
                cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='f1')

                # Avaliar no teste
                if X_test is not None and len(X_test) > 0:
                    y_pred = model.predict(X_test)
                    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred

                    acc = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, zero_division=0)

                    try:
                        auc = roc_auc_score(y_test, y_prob)
                    except Exception:
                        auc = 0.5

                    results[name] = {
                        'cv_f1_mean': float(np.mean(cv_scores)),
                        'cv_f1_std': float(np.std(cv_scores)),
                        'test_accuracy': float(acc),
                        'test_f1': float(f1),
                        'test_auc': float(auc)
                    }

                    score = f1  # Usar F1 como metrica principal

                    print(f"  {name:20s} | CV F1: {np.mean(cv_scores):.3f} | Test Acc: {acc:.1%} | F1: {f1:.3f} | AUC: {auc:.3f}")

                    if score > best_score:
                        best_score = score
                        self.best_model_name = name

            except Exception as e:
                print(f"  {name:20s} | ERRO: {e}")
                results[name] = {'error': str(e)}

        print("-" * 50)

        if self.best_model_name:
            print(f"\n[BEST] Melhor modelo: {self.best_model_name}")

            # Avaliar melhor modelo
            self._evaluate_best_model(X_test, y_test)

        # Salvar info
        self.model_info = {
            "training_date": datetime.now().isoformat(),
            "feature_columns": self.feature_columns,
            "n_features": len(self.feature_columns),
            "train_samples": len(X_train),
            "test_samples": len(X_test) if X_test is not None else 0,
            "best_model": self.best_model_name,
            "results": results
        }

        # Salvar modelos
        self._save_models()

        return results

    def _evaluate_best_model(self, X_test, y_test):
        """Avalia o melhor modelo em detalhes"""
        if self.best_model_name is None or X_test is None:
            return

        model = self.models[self.best_model_name]
        y_pred = model.predict(X_test)

        print(f"\n[EVAL] Avaliacao detalhada - {self.best_model_name}")
        print("=" * 50)

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Loss/NoTrade', 'TP']))

        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print("              Pred:")
        print("              Loss  TP")
        print(f"  Real Loss:  {cm[0][0]:4d}  {cm[0][1]:4d}")
        print(f"  Real TP:    {cm[1][0]:4d}  {cm[1][1]:4d}")

        # Feature importance (se disponivel)
        if hasattr(model, 'feature_importances_'):
            print("\nFeature Importance (Top 5):")
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:5]
            for i, idx in enumerate(indices):
                print(f"  {i+1}. {self.feature_columns[idx]}: {importances[idx]:.3f}")

    def _save_models(self):
        """Salva modelos e artefatos"""
        print("\n[SAVE] Salvando modelos...")

        # Todos os modelos
        models_path = os.path.join(CONFIG["model_dir"], "signal_validators.pkl")
        with open(models_path, 'wb') as f:
            pickle.dump(self.models, f)
        print(f"  [OK] Modelos: {models_path}")

        # Scaler
        scaler_path = os.path.join(CONFIG["model_dir"], "scaler_simple.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"  [OK] Scaler: {scaler_path}")

        # Metadata
        info_path = os.path.join(CONFIG["model_dir"], "model_info_simple.json")
        with open(info_path, 'w') as f:
            json.dump(self.model_info, f, indent=2, default=str)
        print(f"  [OK] Info: {info_path}")

    def load_models(self):
        """Carrega modelos salvos"""
        models_path = os.path.join(CONFIG["model_dir"], "signal_validators.pkl")
        scaler_path = os.path.join(CONFIG["model_dir"], "scaler_simple.pkl")
        info_path = os.path.join(CONFIG["model_dir"], "model_info_simple.json")

        with open(models_path, 'rb') as f:
            self.models = pickle.load(f)

        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        with open(info_path, 'r') as f:
            self.model_info = json.load(f)
            self.feature_columns = self.model_info.get('feature_columns', [])
            self.best_model_name = self.model_info.get('best_model')

        print(f"[OK] Modelos carregados. Melhor: {self.best_model_name}")

    def predict_signal(self, features: Dict[str, float], model_name: str = None) -> Dict:
        """
        Faz predicao para um novo sinal.

        Args:
            features: Dicionario com features do sinal
            model_name: Nome do modelo a usar (ou None para usar o melhor)

        Returns:
            Dicionario com probabilidade e recomendacao
        """
        if not self.models:
            self.load_models()

        model_name = model_name or self.best_model_name
        model = self.models.get(model_name)

        if model is None:
            return {'error': f'Modelo {model_name} nao encontrado'}

        # Preparar features
        X = np.array([[features.get(col, 0) for col in self.feature_columns]])

        # Tratar NaN/Inf nos inputs
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Aplicar scaler (pode retornar NaN se scaler mal treinado)
        try:
            X_scaled = self.scaler.transform(X)
            # Tratar NaN/Inf no output do scaler (scaler quebrado ou dados extremos)
            if np.any(np.isnan(X_scaled)) or np.any(np.isinf(X_scaled)):
                logger.warning("[ML] Scaler retornou NaN/Inf - usando features sem escala")
                X_scaled = X  # Fallback: usar features brutas
        except Exception as e:
            logger.warning(f"[ML] Erro no scaler.transform: {e} - usando features sem escala")
            X_scaled = X

        # Predicao
        pred = model.predict(X_scaled)[0]

        if hasattr(model, 'predict_proba'):
            prob = float(model.predict_proba(X_scaled)[0][1])
        else:
            prob = float(pred)

        # Proteger contra NaN na saída final
        if np.isnan(prob):
            logger.warning("[ML] predict_proba retornou NaN - usando 0.5 (neutro)")
            prob = 0.5
        if np.isnan(pred):
            pred = 0

        return {
            'probability_success': prob,
            'prediction': int(pred),
            'recommendation': 'EXECUTE' if pred == 1 else 'SKIP',
            'confidence': abs(prob - 0.5) * 2,
            'model_used': model_name
        }

    def validate_deepseek_signal(self, deepseek_signal: Dict) -> Dict:
        """
        Valida um sinal do DeepSeek usando o modelo treinado.

        Args:
            deepseek_signal: Sinal gerado pelo DeepSeek

        Returns:
            Sinal original + validacao do modelo
        """
        # Extrair features do sinal
        features = {
            'rsi': deepseek_signal.get('rsi', 50),
            'macd_histogram': deepseek_signal.get('macd_histogram', 0),
            'adx': deepseek_signal.get('adx', 25),
            'atr': deepseek_signal.get('atr', 0),
            'bb_position': deepseek_signal.get('bb_position', 0.5),
            'cvd': deepseek_signal.get('cvd', 0),
            'orderbook_imbalance': deepseek_signal.get('orderbook_imbalance', 0.5),
            'bullish_tf_count': deepseek_signal.get('bullish_tf_count', 0),
            'bearish_tf_count': deepseek_signal.get('bearish_tf_count', 0),
            'confidence': deepseek_signal.get('confidence', 5),
            'trend_encoded': deepseek_signal.get('trend_encoded', 0),
            'sentiment_encoded': deepseek_signal.get('sentiment_encoded', 0),
            'signal_encoded': 1 if deepseek_signal.get('signal') == 'BUY' else 0,
            'risk_distance_pct': deepseek_signal.get('risk_distance_pct', 2),
            'reward_distance_pct': deepseek_signal.get('reward_distance_pct', 2),
            'risk_reward_ratio': deepseek_signal.get('risk_reward_ratio', 1),
        }

        validation = self.predict_signal(features)

        result = {
            'original_signal': deepseek_signal,
            'model_validation': validation,
            'final_recommendation': 'EXECUTE' if validation['recommendation'] == 'EXECUTE' else 'SKIP',
            'confluence': deepseek_signal.get('signal', 'NO_SIGNAL') != 'NO_SIGNAL' and validation['prediction'] == 1
        }

        # Registrar predicao para dashboard
        self._log_prediction(deepseek_signal, validation)

        return result

    def _log_prediction(self, signal: Dict, validation: Dict):
        """
        Registra predicao completa para acompanhamento no dashboard.
        Salva dados ricos para poder comparar ML vs resultado real depois.
        """
        try:
            log_path = os.path.join(CONFIG["model_dir"], "prediction_log.json")

            # Carregar log existente
            predictions = []
            if os.path.exists(log_path):
                with open(log_path, 'r') as f:
                    predictions = json.load(f)

            # Registrar predicao com dados completos para analise futura
            entry = {
                'timestamp': datetime.now().isoformat(),
                'symbol': signal.get('symbol', 'UNKNOWN'),
                # Sinal do DeepSeek
                'deepseek_signal': signal.get('signal', 'NO_SIGNAL'),
                'deepseek_confidence': signal.get('confidence', 0),
                'entry_price': signal.get('entry_price', 0),
                'stop_loss': signal.get('stop_loss', 0),
                'take_profit_1': signal.get('take_profit_1', signal.get('take_profit', 0)),
                'take_profit_2': signal.get('take_profit_2', 0),
                'take_profit_3': signal.get('take_profit_3', 0),
                # Predicao do ML
                'ml_prediction': validation.get('prediction', 0),
                'ml_probability': validation.get('probability_success', validation.get('probability', 0.5)),
                'ml_recommendation': validation.get('recommendation', 'SKIP'),
                'ml_model_used': self.best_model_name,
                # Indicadores chave (para debug)
                'rsi': signal.get('rsi', signal.get('indicators', {}).get('rsi', None)),
                'adx': signal.get('adx', signal.get('indicators', {}).get('adx', None)),
                'trend': signal.get('trend', None),
                'sentiment': signal.get('sentiment', None),
                # Resultado real (preenchido depois pelo online_learning)
                'actual_result': None,  # 'TP' ou 'SL' - preenchido quando trade fecha
                'ml_was_correct': None,  # True/False - preenchido quando trade fecha
            }

            predictions.append(entry)

            # Manter ultimas 1000 predicoes (mais historico para analise)
            predictions = predictions[-1000:]

            # Salvar
            with open(log_path, 'w') as f:
                json.dump(predictions, f, indent=2, default=str)

        except Exception as e:
            print(f"[ML LOG] Erro ao salvar predicao: {e}")


def main():
    """Funcao principal"""
    print("\n" + "="*70)
    print("SIMPLE SIGNAL VALIDATOR - TREINAMENTO")
    print("="*70)

    validator = SimpleSignalValidator()

    try:
        validator.train()

        print("\n" + "="*70)
        print("[OK] TREINAMENTO CONCLUIDO!")
        print("="*70)

        # Teste rapido
        print("\n[TEST] Testando predicao com dados dummy...")
        test_features = {
            'rsi': 65,
            'macd_histogram': 50,
            'adx': 30,
            'confidence': 8,
            'signal_encoded': 1,
            'risk_reward_ratio': 2.0
        }
        result = validator.predict_signal(test_features)
        print(f"  Probabilidade: {result['probability_success']:.1%}")
        print(f"  Recomendacao: {result['recommendation']}")

    except Exception as e:
        print(f"\n[ERRO] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

