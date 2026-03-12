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

import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Configuracoes
CONFIG = {
    "model_dir": "ml_models",
    "buffer_file": "ml_models/online_learning_buffer.json",
    "performance_file": "ml_models/model_performance.json",
    "retrain_threshold": 50,  # Retreinar quando tiver N novos exemplos
    "min_improvement": 0.0,   # Melhoria minima necessaria para salvar novo modelo
    "validation_split": 0.2,  # Split para validacao do retreino
}


class OnlineLearningManager:
    """
    Gerenciador de Online Learning para o validador de sinais.
    Permite que o modelo melhore continuamente com novos dados.
    """
    
    def __init__(self):
        self.buffer: List[Dict] = []
        self.performance_history: List[Dict] = []
        
        Path(CONFIG["model_dir"]).mkdir(exist_ok=True)
        self._load_buffer()
        self._load_performance_history()
        
    def _load_buffer(self):
        """Carrega buffer de dados pendentes"""
        if os.path.exists(CONFIG["buffer_file"]):
            try:
                with open(CONFIG["buffer_file"], 'r') as f:
                    self.buffer = json.load(f)
                print(f"[OL] Buffer carregado: {len(self.buffer)} exemplos pendentes")
            except:
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
            except:
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
        # CORRIGIDO: Extrair TODAS as features necessárias para o modelo
        # Sincronizado com simple_signal_validator.py
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

        # 3. signal_encoded
        signal_type = signal.get('signal', 'NO_SIGNAL')
        signal_encoded = 1 if signal_type == 'BUY' else 0

        # 4. risk_distance_pct e reward_distance_pct
        entry_price = signal.get('entry_price', 0)
        stop_loss = signal.get('stop_loss', 0)
        take_profit_1 = signal.get('take_profit_1', 0)

        if entry_price > 0 and stop_loss > 0:
            risk_distance_pct = abs(entry_price - stop_loss) / entry_price * 100
        else:
            risk_distance_pct = 2.0  # Default 2%

        if entry_price > 0 and take_profit_1 > 0:
            reward_distance_pct = abs(take_profit_1 - entry_price) / entry_price * 100
        else:
            reward_distance_pct = 2.0  # Default 2%

        # 5. risk_reward_ratio
        risk_reward_ratio = reward_distance_pct / risk_distance_pct if risk_distance_pct > 0 else 1.0

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
            print(f"[OL] Threshold atingido! Iniciando retreino...")
            self.retrain()
            
    def retrain(self) -> Dict:
        """
        Retreina o modelo com novos dados.
        
        Returns:
            Dict com resultados do retreino
        """
        print("\n" + "="*60)
        print("ONLINE LEARNING - RETREINAMENTO")
        print("="*60)
        
        if len(self.buffer) < 10:
            print(f"[OL] Dados insuficientes para retreino ({len(self.buffer)} < 10)")
            return {"success": False, "reason": "Dados insuficientes"}
            
        try:
            # Carregar modelo atual (se existir); senao primeiro treino
            from src.ml.simple_validator import SimpleSignalValidator
            current_validator = SimpleSignalValidator()
            models_path = os.path.join(CONFIG["model_dir"], "signal_validators.pkl")
            if os.path.exists(models_path):
                current_validator.load_models()
            else:
                current_validator.models = {}
                current_validator.best_model_name = None
                current_validator.scaler = None
                print("[OL] Nenhum modelo existente - primeiro treino")
            
            # Preparar novos dados
            new_df = pd.DataFrame(self.buffer)
            
            # CORRIGIDO: Sincronizar features com simple_signal_validator.py
            # Usando as mesmas 16 features para consistência entre treino e retreino
            feature_cols = [
                'rsi', 'macd_histogram', 'adx', 'atr', 'bb_position',
                'cvd', 'orderbook_imbalance', 'bullish_tf_count', 'bearish_tf_count',
                'confidence', 'trend_encoded', 'sentiment_encoded', 'signal_encoded',
                'risk_distance_pct', 'reward_distance_pct', 'risk_reward_ratio'
            ]
            
            available_cols = [col for col in feature_cols if col in new_df.columns]
            
            X_new = new_df[available_cols].fillna(0).values
            y_new = new_df['target'].values
            
            print(f"[OL] Novos dados: {len(X_new)} amostras")
            print(f"[OL] Distribuicao: {sum(y_new)} TP, {len(y_new)-sum(y_new)} SL")
            
            # Carregar dados originais de treino
            train_path = "ml_dataset/dataset_train_latest.csv"
            if os.path.exists(train_path):
                original_df = pd.read_csv(train_path)
                X_orig = original_df[available_cols].fillna(0).values
                y_orig = original_df['target'].values
                
                # Combinar dados
                X_combined = np.vstack([X_orig, X_new])
                y_combined = np.hstack([y_orig, y_new])
                
                print(f"[OL] Dados combinados: {len(X_combined)} amostras")
            else:
                X_combined = X_new
                y_combined = y_new
                
            # Split para validacao
            n = len(X_combined)
            split_idx = int(n * (1 - CONFIG["validation_split"]))
            
            X_train, X_val = X_combined[:split_idx], X_combined[split_idx:]
            y_train, y_val = y_combined[:split_idx], y_combined[split_idx:]
            
            # Treinar novo modelo
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Usar LogisticRegression (melhor modelo atual)
            new_model = LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42
            )
            new_model.fit(X_train_scaled, y_train)
            
            # Avaliar
            y_pred = new_model.predict(X_val_scaled)
            new_accuracy = accuracy_score(y_val, y_pred)
            new_f1 = f1_score(y_val, y_pred, zero_division=0)
            
            print(f"[OL] Novo modelo - Accuracy: {new_accuracy:.1%}, F1: {new_f1:.3f}")
            
            # Comparar com modelo atual
            current_model = current_validator.models.get(current_validator.best_model_name)
            if current_model:
                # Reescalar com o scaler atual
                X_val_current = current_validator.scaler.transform(X_val)
                y_pred_current = current_model.predict(X_val_current)
                current_accuracy = accuracy_score(y_val, y_pred_current)
                current_f1 = f1_score(y_val, y_pred_current, zero_division=0)
                
                print(f"[OL] Modelo atual - Accuracy: {current_accuracy:.1%}, F1: {current_f1:.3f}")
                
                improvement = new_f1 - current_f1
                print(f"[OL] Melhoria F1: {improvement:+.3f}")
                
                if improvement >= CONFIG["min_improvement"]:
                    # Salvar novo modelo
                    self._save_new_model(new_model, scaler, available_cols)
                    
                    # Limpar buffer
                    self.buffer = []
                    self._save_buffer()
                    
                    # Registrar performance
                    self._record_performance(new_accuracy, new_f1, len(X_combined))
                    
                    print(f"[OL] Novo modelo salvo com sucesso!")
                    
                    return {
                        "success": True,
                        "new_accuracy": new_accuracy,
                        "new_f1": new_f1,
                        "improvement": improvement,
                        "samples_used": len(X_combined)
                    }
                else:
                    print(f"[OL] Modelo nao melhorou suficiente. Mantendo atual.")
                    return {
                        "success": False,
                        "reason": "Sem melhoria suficiente",
                        "improvement": improvement
                    }
            else:
                # Nao tem modelo atual, salvar novo
                self._save_new_model(new_model, scaler, available_cols)
                self.buffer = []
                self._save_buffer()
                self._record_performance(new_accuracy, new_f1, len(X_combined))
                
                return {
                    "success": True,
                    "new_accuracy": new_accuracy,
                    "new_f1": new_f1,
                    "samples_used": len(X_combined)
                }
                
        except Exception as e:
            print(f"[OL] Erro no retreino: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}
            
    def _save_new_model(self, model, scaler, feature_columns):
        """Salva o novo modelo treinado"""
        # Salvar modelo
        models_path = os.path.join(CONFIG["model_dir"], "signal_validators.pkl")
        
        # Carregar modelos existentes e atualizar
        models = {}
        if os.path.exists(models_path):
            with open(models_path, 'rb') as f:
                models = pickle.load(f)
                
        models['LogisticRegression'] = model
        
        with open(models_path, 'wb') as f:
            pickle.dump(models, f)
            
        # Salvar scaler
        scaler_path = os.path.join(CONFIG["model_dir"], "scaler_simple.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
            
        # Atualizar metadata
        info_path = os.path.join(CONFIG["model_dir"], "model_info_simple.json")
        info = {}
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                info = json.load(f)
                
        info['last_retrain'] = datetime.now().isoformat()
        info['retrain_count'] = info.get('retrain_count', 0) + 1
        info['feature_columns'] = feature_columns
        info['best_model'] = 'LogisticRegression'

        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2, default=str)
            
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


# Instancia global para uso no bot
online_learning_manager = OnlineLearningManager()


def add_trade_result(signal: Dict, result: str, return_pct: float = 0.0):
    """
    Funcao helper para adicionar resultado de trade.
    Chamada pelo bot quando um trade e fechado.
    """
    online_learning_manager.add_signal_result(signal, result, return_pct)
    

def manual_retrain():
    """Forca retreino manual"""
    return online_learning_manager.retrain()


def seed_from_evaluated_signals(force_retrain: bool = True) -> Dict:
    """
    Popula o buffer do online learning com sinais ja avaliados pelo signal_tracker.
    Isso permite treinar o ML usando sinais existentes sem precisar de trades paper.

    Args:
        force_retrain: Se True, dispara retreino apos popular o buffer

    Returns:
        Dict com resultado da operacao
    """
    try:
        from src.trading.signal_tracker import evaluate_all_signals
    except ImportError as e:
        return {"success": False, "error": f"Import error: {e}"}

    evaluations = evaluate_all_signals()
    if not evaluations:
        return {"success": False, "error": "Nenhum sinal avaliado encontrado"}

    # Filtrar apenas sinais finalizados (SL/TP hit)
    finalized = [e for e in evaluations
                 if e.get('outcome') in ('SL_HIT', 'TP1_HIT', 'TP2_HIT')]

    if not finalized:
        return {"success": False, "error": "Nenhum sinal finalizado (SL/TP hit)"}

    # Evitar duplicatas - checar sinais ja no buffer por symbol+timestamp
    existing_keys = {
        (b.get('symbol', ''), b.get('timestamp', '')[:16])
        for b in online_learning_manager.buffer
    }

    added = 0
    for ev in finalized:
        key = (ev.get('symbol', ''), ev.get('timestamp', '')[:16])
        if key in existing_keys:
            continue

        outcome = ev.get('outcome', '')
        result_map = {'TP1_HIT': 'TP1', 'TP2_HIT': 'TP2', 'SL_HIT': 'SL'}
        result = result_map.get(outcome, 'SL')

        # Usar add_signal_result para manter formato consistente
        signal_data = {
            'symbol': ev.get('symbol', ''),
            'signal': ev.get('signal', ''),
            'entry_price': ev.get('entry_price', 0),
            'stop_loss': ev.get('stop_loss', 0),
            'take_profit_1': ev.get('take_profit_1', 0),
            'take_profit_2': ev.get('take_profit_2', 0),
            'confidence': ev.get('confidence', 5),
            'timestamp': ev.get('timestamp', ''),
            'source': ev.get('source', ''),
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

    print(f"[OL-SEED] {added} sinais adicionados ao buffer ({len(online_learning_manager.buffer)} total)")

    retrain_result = None
    if force_retrain and len(online_learning_manager.buffer) >= 10:
        print(f"[OL-SEED] Disparando retreino com {len(online_learning_manager.buffer)} amostras...")
        retrain_result = online_learning_manager.retrain()

    return {
        "success": True,
        "signals_evaluated": len(evaluations),
        "signals_finalized": len(finalized),
        "signals_added": added,
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

