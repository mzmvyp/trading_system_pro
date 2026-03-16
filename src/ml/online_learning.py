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

# Sklearn
from sklearn.preprocessing import StandardScaler

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
            print("[OL] Threshold atingido! Iniciando retreino...")
            self.retrain()

    def retrain(self) -> Dict:
        """
        Retreina o modelo com novos dados.
        Se nao existe modelo anterior, cria do zero (primeiro treinamento).
        Usa TODOS os dados do buffer + dataset original (se existir).

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
            # Tentar carregar modelo atual (pode nao existir ainda)
            from src.ml.simple_validator import SimpleSignalValidator
            current_validator = None
            has_current_model = False

            try:
                current_validator = SimpleSignalValidator()
                current_validator.load_models()
                has_current_model = True
                print(f"[OL] Modelo atual carregado: {current_validator.best_model_name}")
            except (FileNotFoundError, EOFError, Exception) as e:
                print(f"[OL] Nenhum modelo existente encontrado ({e}). Criando modelo inicial...")
                has_current_model = False

            # Preparar novos dados do buffer
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

            print(f"[OL] Novos dados do buffer: {len(X_new)} amostras")
            print(f"[OL] Distribuicao: {sum(y_new)} TP, {len(y_new)-sum(y_new)} SL")

            # Carregar dados originais de treino (se existir)
            train_path = "ml_dataset/dataset_train_latest.csv"
            if os.path.exists(train_path):
                original_df = pd.read_csv(train_path)
                orig_available = [col for col in available_cols if col in original_df.columns]
                if orig_available and 'target' in original_df.columns:
                    X_orig = original_df[orig_available].fillna(0).values
                    y_orig = original_df['target'].values
                    X_combined = np.vstack([X_orig, X_new])
                    y_combined = np.hstack([y_orig, y_new])
                    print(f"[OL] Dados combinados: {len(X_orig)} originais + {len(X_new)} buffer = {len(X_combined)} total")
                else:
                    X_combined = X_new
                    y_combined = y_new
            else:
                X_combined = X_new
                y_combined = y_new
                print(f"[OL] Sem dataset original. Treinando apenas com buffer ({len(X_combined)} amostras)")

            # Verificar variancia de classes
            unique_classes = np.unique(y_combined)
            if len(unique_classes) < 2:
                print(f"[OL] AVISO: Apenas uma classe nos dados ({unique_classes}). Adicionando dados sinteticos...")
                # Adicionar pelo menos 1 exemplo da classe faltante para evitar erro
                minority_class = 0 if 1 in unique_classes else 1
                synthetic = np.median(X_combined, axis=0).reshape(1, -1)
                X_combined = np.vstack([X_combined, synthetic])
                y_combined = np.hstack([y_combined, [minority_class]])

            # Split para validacao
            n = len(X_combined)
            split_idx = int(n * (1 - CONFIG["validation_split"]))
            split_idx = max(split_idx, 1)  # Garantir pelo menos 1 amostra no treino

            X_train, X_val = X_combined[:split_idx], X_combined[split_idx:]
            y_train, y_val = y_combined[:split_idx], y_combined[split_idx:]

            # Treinar ensemble de modelos
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            models_to_train = {
                "LogisticRegression": LogisticRegression(
                    max_iter=1000, class_weight='balanced', random_state=42
                ),
                "RandomForest": RandomForestClassifier(
                    n_estimators=100, max_depth=5, class_weight='balanced', random_state=42
                ),
                "GradientBoosting": GradientBoostingClassifier(
                    n_estimators=100, max_depth=3, random_state=42
                ),
            }

            best_model = None
            best_model_name = None
            best_f1 = -1
            best_accuracy = 0
            trained_models = {}

            for name, model in models_to_train.items():
                try:
                    model.fit(X_train_scaled, y_train)
                    trained_models[name] = model

                    y_pred = model.predict(X_val_scaled)
                    acc = accuracy_score(y_val, y_pred)
                    f1 = f1_score(y_val, y_pred, zero_division=0)
                    print(f"[OL]   {name}: Accuracy={acc:.1%}, F1={f1:.3f}")

                    if f1 > best_f1:
                        best_f1 = f1
                        best_accuracy = acc
                        best_model = model
                        best_model_name = name
                except Exception as e:
                    print(f"[OL]   {name}: ERRO - {e}")

            if best_model is None:
                return {"success": False, "reason": "Nenhum modelo treinado com sucesso"}

            new_accuracy = best_accuracy
            new_f1 = best_f1

            print(f"\n[OL] Melhor modelo: {best_model_name} (Accuracy={new_accuracy:.1%}, F1={new_f1:.3f})")

            # Comparar com modelo atual (se existir)
            should_save = False
            improvement = 0.0

            if has_current_model:
                current_model = current_validator.models.get(current_validator.best_model_name)
                if current_model:
                    try:
                        X_val_current = current_validator.scaler.transform(X_val)
                        y_pred_current = current_model.predict(X_val_current)
                        current_f1 = f1_score(y_val, y_pred_current, zero_division=0)

                        improvement = new_f1 - current_f1
                        print(f"[OL] Modelo atual F1: {current_f1:.3f} | Novo F1: {new_f1:.3f} | Melhoria: {improvement:+.3f}")

                        should_save = improvement >= CONFIG["min_improvement"]
                    except Exception as e:
                        print(f"[OL] Erro ao comparar com modelo atual: {e}. Salvando novo modelo.")
                        should_save = True
                else:
                    should_save = True
            else:
                # Primeiro modelo: sempre salvar
                should_save = True
                print("[OL] Primeiro treinamento - salvando modelo inicial")

            if should_save:
                # Salvar TODOS os modelos treinados (nao so o melhor)
                self._save_new_model_ensemble(trained_models, best_model_name, scaler, available_cols, new_accuracy, new_f1)

                # Salvar dataset combinado para retreinos futuros
                self._save_combined_dataset(X_combined, y_combined, available_cols)

                # Limpar buffer
                self.buffer = []
                self._save_buffer()

                # Registrar performance
                self._record_performance(new_accuracy, new_f1, len(X_combined))

                print("[OL] Novo modelo salvo com sucesso!")

                return {
                    "success": True,
                    "new_accuracy": new_accuracy,
                    "new_f1": new_f1,
                    "improvement": improvement,
                    "samples_used": len(X_combined),
                    "best_model": best_model_name,
                    "first_training": not has_current_model
                }
            else:
                print("[OL] Modelo nao melhorou suficiente. Mantendo atual.")
                return {
                    "success": False,
                    "reason": "Sem melhoria suficiente",
                    "improvement": improvement
                }

        except Exception as e:
            print(f"[OL] Erro no retreino: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

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
        """Salva dataset combinado para retreinos futuros"""
        try:
            os.makedirs("ml_dataset", exist_ok=True)
            df = pd.DataFrame(X, columns=feature_columns)
            df['target'] = y
            df.to_csv("ml_dataset/dataset_train_latest.csv", index=False)
            print(f"[OL] Dataset atualizado: {len(df)} amostras salvas em ml_dataset/dataset_train_latest.csv")
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


def seed_from_evaluated_signals(force_retrain: bool = True, max_signals: int = 150) -> Dict:
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
    evaluations = evaluate_all_signals(max_to_evaluate=200)
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
                    signal_data = {
                        'symbol': ev.get('symbol', ''),
                        'signal': ev.get('signal', ''),
                        'entry_price': ev.get('entry_price', 0),
                        'stop_loss': ev.get('stop_loss', 0),
                        'take_profit_1': ev.get('take_profit_1', 0),
                        'confidence': ev.get('confidence', 5),
                        # Indicadores REAIS do enriquecimento
                        'rsi': enriched_data.get('rsi', 50),
                        'macd_histogram': enriched_data.get('macd_histogram', 0),
                        'adx': enriched_data.get('adx', 25),
                        'atr': enriched_data.get('atr', 0),
                        'bb_position': enriched_data.get('bb_position', 0.5),
                        'cvd': enriched_data.get('cvd', 0),
                        'orderbook_imbalance': enriched_data.get('orderbook_imbalance', 0.5),
                        'bullish_tf_count': enriched_data.get('bullish_tf_count', 5),
                        'bearish_tf_count': enriched_data.get('bearish_tf_count', 5),
                        'trend': str(enriched_data.get('trend_encoded', 0)),
                        'sentiment': str(enriched_data.get('sentiment_encoded', 0)),
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

