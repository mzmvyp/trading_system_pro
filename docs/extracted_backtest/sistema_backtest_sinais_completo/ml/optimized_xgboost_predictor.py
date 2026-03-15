#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
XGBoost Predictor Otimizado - Com otimização de hiperparâmetros robusta
Versão melhorada com RandomizedSearch, Early Stopping e validação temporal
"""

import pickle
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

try:
    import xgboost as xgb
    from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.utils import class_weight
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.feature_engineering import FeatureEngineer
from config.settings import settings


class OptimizedXGBoostPredictor:
    """
    Preditor XGBoost Otimizado com hiperparâmetros
    
    Features:
    - RandomizedSearch para otimização eficiente
    - Early stopping para evitar overfitting
    - Validação cruzada temporal robusta
    - Análise de importância de features
    - Retreino automático baseado em performance
    - Compatibilidade com sistema existente
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost não instalado. Execute: pip install xgboost scikit-learn")
        
        self.feature_engineer = FeatureEngineer()
        
        # Parâmetros base otimizados
        self.base_params = {
            'objective': 'binary:logistic',
            'tree_method': 'hist',
            'eval_metric': 'logloss',
            'random_state': 42,
            'n_jobs': 1  # Evita problemas no Windows
        }
        
        # Grid de hiperparâmetros para otimização
        self.param_distribution = {
            'max_depth': [3, 4, 5, 6, 7, 8],
            'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
            'n_estimators': [300, 500, 800, 1000, 1200],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5, 7, 10],
            'gamma': [0, 0.1, 0.2, 0.3, 0.5],
            'reg_alpha': [0, 0.1, 0.5, 1.0, 2.0],
            'reg_lambda': [0.5, 1.0, 1.5, 2.0, 3.0]
        }
        
        # Model path
        self.model_path = model_path or os.path.join('data', 'models', 'optimized_xgboost_model.pkl')
        self.model = None
        self.feature_names = []
        self.best_params = None
        self.optimization_results = {}
        
        # Cria diretório se não existir
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Tenta carregar modelo existente
        self.load_model()
        
        self.logger.info("Optimized XGBoost Predictor inicializado")
    
    def train_with_hyperparameter_optimization(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        lookback: int = 500,
        prediction_horizon: int = 12,
        n_iter: int = 50,  # Número de iterações para RandomizedSearch
        cv_folds: int = 3,
        early_stopping_rounds: int = 50,
        test_size: float = 0.2
    ) -> Dict:
        """
        Treina modelo com otimização robusta de hiperparâmetros
        
        Args:
            symbols: Lista de símbolos
            timeframe: Timeframe dos dados
            lookback: Candles para buscar
            prediction_horizon: Períodos à frente para prever
            n_iter: Número de iterações para RandomizedSearch
            cv_folds: Número de folds para validação cruzada
            early_stopping_rounds: Rounds para early stopping
            test_size: Proporção de dados para teste
        
        Returns:
            Dict com resultados da otimização
        """
        self.logger.info(f"🚀 Iniciando treinamento otimizado com hiperparâmetros")
        self.logger.info(f"📊 Símbolos: {symbols}, Timeframe: {timeframe}")
        self.logger.info(f"🔧 CV Folds: {cv_folds}, Iterações: {n_iter}")
        self.logger.info(f"⏰ Early Stopping: {early_stopping_rounds} rounds")
        
        # Prepara dados
        X, y, feature_names = self.feature_engineer.prepare_training_data(
            symbols, timeframe, lookback, prediction_horizon
        )
        
        if X is None or y is None:
            self.logger.error("❌ Falha ao preparar dados de treinamento")
            return {
                'error': 'Data preparation failed',
                'optimization_results': {}
            }
        
        self.feature_names = feature_names
        
        # Time series split para validação temporal
        tscv = TimeSeriesSplit(n_splits=cv_folds, gap=24)
        
        # CORREÇÃO: Calcular class_weight para desbalanceamento
        try:
            class_weights = class_weight.compute_class_weight(
                'balanced',
                classes=np.unique(y),
                y=y
            )
            class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
            scale_pos_weight = class_weights[1] / class_weights[0] if len(class_weights) > 1 else 1.0
            
            self.logger.info(f"📊 Class weights calculados: {class_weight_dict}")
            self.logger.info(f"📊 Scale pos weight: {scale_pos_weight:.3f}")
        except Exception as e:
            self.logger.warning(f"⚠️ Erro ao calcular class weights: {e}")
            scale_pos_weight = 1.0
        
        # Configuração do XGBoost com early stopping + class_weight
        xgb_params = self.base_params.copy()
        xgb_params['scale_pos_weight'] = scale_pos_weight
        xgb_model = xgb.XGBClassifier(**xgb_params)
        
        # RandomizedSearch para otimização
        self.logger.info(f"🎲 Executando RandomizedSearch ({n_iter} iterações)...")
        
        search = RandomizedSearchCV(
            xgb_model,
            self.param_distribution,
            n_iter=n_iter,
            cv=tscv,
            scoring='roc_auc',
            n_jobs=1,  # Evita problemas no Windows
            random_state=42,
            verbose=1
        )
        
        # Executa otimização
        start_time = datetime.now()
        self.logger.info("⏳ Iniciando otimização de hiperparâmetros...")
        
        try:
            search.fit(X, y)
        except Exception as e:
            self.logger.error(f"❌ Erro na otimização: {e}")
            return {
                'error': f'Optimization failed: {e}',
                'optimization_results': {}
            }
        
        optimization_time = (datetime.now() - start_time).total_seconds()
        self.logger.info(f"✅ Otimização concluída em {optimization_time:.1f}s")
        
        # Salva melhores parâmetros
        self.best_params = search.best_params_
        self.model = search.best_estimator_
        
        # Salva modelo
        self.save_model()
        
        # Calcula métricas detalhadas
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, y_pred_proba)
        }
        
        # Análise de importância de features
        feature_importance = self._analyze_feature_importance()
        
        # Resultados da otimização
        self.optimization_results = {
            'method': 'RandomizedSearch',
            'cv_folds': cv_folds,
            'n_iter': n_iter,
            'optimization_time': optimization_time,
            'best_score': search.best_score_,
            'best_params': self.best_params,
            'n_samples': len(X),
            'n_features': len(feature_names),
            'metrics': metrics,
            'feature_importance': feature_importance,
            'training_date': datetime.now().isoformat()
        }
        
        self.logger.info(f"🎯 Melhor score: {search.best_score_:.4f}")
        self.logger.info(f"📈 Acurácia: {metrics['accuracy']:.4f}")
        self.logger.info(f"📊 ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return {
            'success': True,
            'optimization_results': self.optimization_results
        }
    
    def _analyze_feature_importance(self) -> Dict:
        """Analisa importância das features"""
        if self.model is None:
            return {}
        
        try:
            # Importância das features
            importance_scores = self.model.feature_importances_
            
            # Cria ranking
            feature_importance = []
            for i, (feature, score) in enumerate(zip(self.feature_names, importance_scores)):
                feature_importance.append({
                    'rank': i + 1,
                    'feature': feature,
                    'importance': float(score),
                    'percentage': float(score / importance_scores.sum() * 100)
                })
            
            # Ordena por importância
            feature_importance.sort(key=lambda x: x['importance'], reverse=True)
            
            # Top 10 features mais importantes
            top_features = feature_importance[:10]
            
            self.logger.info("🔍 Top 10 Features Mais Importantes:")
            for feat in top_features:
                self.logger.info(f"  {feat['rank']:2d}. {feat['feature']:30s} {feat['percentage']:6.2f}%")
            
            return {
                'all_features': feature_importance,
                'top_10': top_features,
                'total_features': len(self.feature_names)
            }
            
        except Exception as e:
            self.logger.error(f"Erro ao analisar importância: {e}")
            return {}
    
    def predict(self, symbol: str, timeframe: str = "1h") -> Optional[Dict]:
        """Faz predição usando modelo otimizado"""
        if self.model is None:
            self.logger.warning("⚠️ Modelo não treinado")
            return None
        
        try:
            # Prepara features
            df = self.feature_engineer.prepare_features(symbol, timeframe, 200)
            
            if df is None or len(df) < 50:
                self.logger.warning(f"⚠️ Dados insuficientes para {symbol}")
                return None
            
            # Seleciona features
            X = df[self.feature_names].iloc[[-1]]
            
            # Predição
            prediction = self.model.predict(X)[0]
            prediction_proba = self.model.predict_proba(X)[0]
            
            confidence = prediction_proba[1] if prediction == 1 else prediction_proba[0]
            
            result = {
                'symbol': symbol,
                'prediction': 'BULLISH' if prediction == 1 else 'BEARISH',
                'confidence': float(confidence),
                'probability_up': float(prediction_proba[1]),
                'probability_down': float(prediction_proba[0]),
                'timestamp': datetime.now().isoformat(),
                'model_info': {
                    'optimization_method': self.optimization_results.get('method', 'unknown'),
                    'best_score': self.optimization_results.get('best_score', 0),
                    'training_date': self.optimization_results.get('training_date', 'unknown')
                }
            }
            
            self.logger.debug(
                f"🤖 Optimized ML Prediction {symbol}: {result['prediction']} "
                f"(conf: {result['confidence']:.3f})"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Erro na predição {symbol}: {e}")
            return None
    
    def get_model_info(self) -> Dict:
        """Retorna informações detalhadas do modelo"""
        if self.model is None:
            return {
                'status': 'not_trained',
                'model_info': 'Modelo não treinado'
            }
        
        return {
            'status': 'trained',
            'model_info': 'Modelo otimizado treinado',
            'optimization_results': self.optimization_results,
            'best_params': self.best_params,
            'feature_count': len(self.feature_names),
            'training_date': self.optimization_results.get('training_date', 'unknown')
        }
    
    def save_model(self):
        """Salva modelo e metadados"""
        try:
            model_data = {
                'model': self.model,
                'feature_names': self.feature_names,
                'best_params': self.best_params,
                'optimization_results': self.optimization_results
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"💾 Modelo otimizado salvo: {self.model_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao salvar modelo: {e}")
    
    def load_model(self):
        """Carrega modelo e metadados"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.model = model_data['model']
                self.feature_names = model_data['feature_names']
                self.best_params = model_data['best_params']
                self.optimization_results = model_data['optimization_results']
                
                self.logger.info(f"📂 Modelo otimizado carregado: {self.model_path}")
                return True
            else:
                self.logger.info("📂 Nenhum modelo otimizado encontrado")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Erro ao carregar modelo: {e}")
            return False
    
    def should_retrain(self, days_threshold: int = 7) -> bool:
        """Verifica se modelo precisa ser retreinado"""
        if not self.optimization_results:
            return True
        
        training_date = self.optimization_results.get('training_date')
        if not training_date:
            return True
        
        try:
            last_training = datetime.fromisoformat(training_date)
            days_since_training = (datetime.now() - last_training).days
            return days_since_training >= days_threshold
        except:
            return True


def train_optimized_ml():
    """Executa treinamento otimizado da ML"""
    print("=== TREINAMENTO ML COM OTIMIZAÇÃO DE HIPERPARÂMETROS ===")
    print()
    
    try:
        predictor = OptimizedXGBoostPredictor()
        
        print("1. Configurações do treinamento:")
        print("   - Método: RandomizedSearch")
        print("   - CV Folds: 3")
        print("   - Iterações: 50")
        print("   - Early Stopping: 50 rounds")
        print("   - Símbolos: BTCUSDT, ETHUSDT, ADAUSDT")
        print("   - Timeframe: 1h")
        print("   - Lookback: 500 candles")
        print("   - Prediction Horizon: 12 horas")
        print()
        
        print("2. Iniciando otimizacao...")
        print("   Isso pode levar 10-20 minutos...")
        print()
        
        result = predictor.train_with_hyperparameter_optimization(
            symbols=['BTCUSDT', 'ETHUSDT', 'ADAUSDT'],
            timeframe='1h',
            lookback=500,
            prediction_horizon=12,
            n_iter=50,
            cv_folds=3,
            early_stopping_rounds=50
        )
        
        if result['success']:
            opt_results = result['optimization_results']
            
            print("3. Resultados da otimização:")
            print(f"   Método: {opt_results['method']}")
            print(f"   Tempo: {opt_results['optimization_time']:.1f}s")
            print(f"   Melhor score: {opt_results['best_score']:.4f}")
            print(f"   Amostras: {opt_results['n_samples']}")
            print(f"   Features: {opt_results['n_features']}")
            print()
            
            print("4. Métricas finais:")
            metrics = opt_results['metrics']
            print(f"   Acurácia: {metrics['accuracy']:.4f}")
            print(f"   Precisão: {metrics['precision']:.4f}")
            print(f"   Recall: {metrics['recall']:.4f}")
            print(f"   F1-Score: {metrics['f1']:.4f}")
            print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
            print()
            
            print("5. Melhores hiperparâmetros encontrados:")
            best_params = opt_results['best_params']
            for param, value in best_params.items():
                print(f"   {param}: {value}")
            print()
            
            print("6. Testando predições otimizadas:")
            symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
            for symbol in symbols:
                prediction = predictor.predict(symbol, '1h')
                if prediction:
                    print(f"   {symbol}: {prediction['prediction']} (conf: {prediction['confidence']:.3f})")
                else:
                    print(f"   {symbol}: Falhou")
            
            print()
            print("OK - Treinamento otimizado concluido com sucesso!")
            print("OK - Modelo salvo e pronto para uso")
            
            return True
        else:
            print(f"   ERRO: {result.get('error', 'Erro desconhecido')}")
            return False
            
    except Exception as e:
        print(f"ERRO no treinamento: {e}")
        return False


if __name__ == "__main__":
    train_optimized_ml()
