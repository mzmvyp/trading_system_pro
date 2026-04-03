"""
Trading Agent usando AGNO para orquestração
"""
import asyncio
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agno.agent import Agent
from agno.models.deepseek import DeepSeek
from dotenv import load_dotenv

# CORREÇÃO: Importar logger
from src.core.logger import get_logger

# Carregar variáveis de ambiente
load_dotenv()

from src.analysis.agno_tools import (  # noqa: E402
    analyze_multiple_timeframes,
    analyze_order_flow,
    analyze_technical_indicators,
    execute_paper_trade,
    get_deepseek_analysis,
    get_market_data,
    validate_risk_and_position,
)
from src.trading.trend_filter import get_trend  # noqa: E402

# CORREÇÃO: Criar instância do logger
logger = get_logger(__name__)

class AgnoTradingAgent:
    """
    Agent de trading que usa AGNO para orquestrar análises
    """

    # Cooldown em memória - impossível de bypassar (compartilhado entre instâncias)
    _last_analysis_time: Dict[str, datetime] = {}
    _last_signal_cache: Dict[str, Dict[str, Any]] = {}  # Último sinal gerado por símbolo
    # Cache de acurácia dos modelos (compartilhado, atualiza a cada 30 min)
    _model_accuracy_cache: Optional[Dict] = None
    _model_accuracy_updated: Optional[datetime] = None

    def __init__(self, paper_trading: bool = True):
        """
        Inicializa o agent de trading.

        Args:
            paper_trading: Se True, apenas simula trades
        """
        self.paper_trading = paper_trading

        # Carregar validador ML (se disponível)
        self.ml_validator = None
        self._load_ml_validator()

        # Carregar Bi-LSTM sequence validator (se treinado)
        self.lstm_sequence_validator = None
        self._load_lstm_sequence_validator()

        # Obter API key - OBRIGATÓRIA
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError(
                "ERRO CRÍTICO: DEEPSEEK_API_KEY não encontrada. "
                "Configure a variável de ambiente DEEPSEEK_API_KEY com sua chave da API. "
                "Obtenha sua chave em: https://platform.deepseek.com/"
            )

        # OTIMIZADO: Agent sem ferramentas - dados são coletados localmente
        # e enviados em prompt único, reduzindo de ~5 API calls para 1 por símbolo
        self.agent = Agent(
            model=DeepSeek(id="deepseek-chat", api_key=api_key, temperature=0.3, max_tokens=500),
            instructions=self._get_instructions()
        )

        # Criar pastas necessárias
        Path("signals").mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
        Path("paper_trades").mkdir(exist_ok=True)

        # Criar estrutura de diretórios para respostas do DeepSeek (ano/mês/dia)
        today = datetime.now(timezone.utc)
        deepseek_logs_dir = Path(f"deepseek_logs/{today.year}/{today.month:02d}/{today.day:02d}")
        deepseek_logs_dir.mkdir(parents=True, exist_ok=True)

    def _load_ml_validator(self):
        """Carrega o validador ML se disponível"""
        try:
            from src.ml.simple_validator import SimpleSignalValidator
            validator = SimpleSignalValidator()
            validator.load_models()
            self.ml_validator = validator
            logger.info("[ML] Validador de sinais ML carregado - confluencia habilitada!")
        except Exception as e:
            logger.warning(f"[ML] Validador ML nao disponivel: {e}")
            self.ml_validator = None

    def _load_lstm_sequence_validator(self):
        """Carrega o Bi-LSTM sequence validator se treinado."""
        try:
            from src.ml.lstm_sequence_validator import LSTMSequenceValidator
            validator = LSTMSequenceValidator()
            if validator.load_model():
                self.lstm_sequence_validator = validator
                logger.info("[Bi-LSTM] Validador de sequências temporais carregado!")
            else:
                logger.info("[Bi-LSTM] Modelo não encontrado (execute backtest_dataset_generator + lstm_sequence_validator para treinar)")
        except Exception as e:
            logger.warning(f"[Bi-LSTM] Não disponível: {e}")

    def _get_model_accuracies(self) -> Dict[str, Optional[float]]:
        """
        Retorna acurácia atual do ML e LSTM (cache de 30 min).
        Modelos com acurácia < 60% não devem votar/bloquear sinais.

        Usa acurácia do MODELO TREINADO (model_info_simple.json / bilstm_model_info.json)
        como fonte primária. A acurácia operacional do log de votos é mantida apenas
        para comparação/log, mas NÃO é usada para decidir se o modelo pode votar.

        Returns:
            Dict com 'ml_accuracy', 'ml_trained_accuracy', 'lstm_accuracy',
            'lstm_trained_accuracy' (percentual ou None)
        """
        now = datetime.now(timezone.utc)
        cache_ttl = timedelta(minutes=30)

        if (
            AgnoTradingAgent._model_accuracy_cache is not None
            and AgnoTradingAgent._model_accuracy_updated is not None
            and (now - AgnoTradingAgent._model_accuracy_updated) < cache_ttl
        ):
            return AgnoTradingAgent._model_accuracy_cache

        try:
            import json as _json

            # === FONTE PRIMÁRIA: acurácia do modelo treinado (mesma que o dashboard mostra) ===
            ml_trained_acc = None
            lstm_trained_acc = None

            # ML: ler de model_info_simple.json
            ml_info_path = os.path.join("ml_models", "model_info_simple.json")
            if os.path.exists(ml_info_path):
                try:
                    with open(ml_info_path, "r") as f:
                        ml_info = _json.load(f)
                    ml_trained_acc_raw = ml_info.get("best_accuracy")
                    if ml_trained_acc_raw is not None:
                        # best_accuracy é armazenado como fração (0.628) — converter para percentual
                        ml_trained_acc = ml_trained_acc_raw * 100 if ml_trained_acc_raw <= 1.0 else ml_trained_acc_raw
                except Exception as e:
                    logger.warning(f"[ACCURACY] Erro ao ler ML model_info: {e}")

            # LSTM: ler de bilstm_model_info.json
            lstm_info_path = os.path.join("ml_models", "bilstm_model_info.json")
            if os.path.exists(lstm_info_path):
                try:
                    with open(lstm_info_path, "r") as f:
                        lstm_info = _json.load(f)
                    lstm_test_acc_raw = lstm_info.get("results", {}).get("test", {}).get("accuracy")
                    if lstm_test_acc_raw is not None:
                        lstm_trained_acc = lstm_test_acc_raw * 100 if lstm_test_acc_raw <= 1.0 else lstm_test_acc_raw
                except Exception as e:
                    logger.warning(f"[ACCURACY] Erro ao ler LSTM model_info: {e}")

            # Também ler do validador carregado em memória (fallback)
            if lstm_trained_acc is None and self.lstm_sequence_validator is not None:
                try:
                    mi = self.lstm_sequence_validator.model_info
                    lstm_test_acc_raw = mi.get("results", {}).get("test", {}).get("accuracy")
                    if lstm_test_acc_raw is not None:
                        lstm_trained_acc = lstm_test_acc_raw * 100 if lstm_test_acc_raw <= 1.0 else lstm_test_acc_raw
                except Exception:
                    pass

            # === FONTE SECUNDÁRIA (apenas para log): acurácia operacional do log de votos ===
            ml_op_acc = None
            ml_class_acc = None
            lstm_log_acc = None
            try:
                from src.trading.signal_tracker import get_system_accuracy_report
                report = get_system_accuracy_report()
                ml_op = report.get("ml_operational", {})
                ml_op_acc = ml_op.get("accuracy_pct")
                ml_class_acc = report.get("ml", {}).get("accuracy_pct")
                lstm_log_acc = report.get("lstm", {}).get("accuracy_pct")
            except Exception:
                pass

            # === Decisão: usar acurácia do modelo treinado ===
            ml_acc = ml_trained_acc
            lstm_acc = lstm_trained_acc

            result = {
                "ml_accuracy": ml_acc,
                "ml_trained_accuracy": ml_trained_acc,
                "ml_operational_accuracy": ml_op_acc,
                "ml_class_accuracy": ml_class_acc,
                "lstm_accuracy": lstm_acc,
                "lstm_trained_accuracy": lstm_trained_acc,
                "lstm_operational_accuracy": lstm_log_acc,
            }
            AgnoTradingAgent._model_accuracy_cache = result
            AgnoTradingAgent._model_accuracy_updated = now

            # Log detalhado
            ml_str = f"ML trained={ml_trained_acc:.1f}%" if ml_trained_acc is not None else "ML trained=N/A"
            ml_op_str = f"ML op_log={ml_op_acc:.1f}%" if ml_op_acc is not None else "ML op_log=N/A"
            lstm_str = f"LSTM trained={lstm_trained_acc:.1f}%" if lstm_trained_acc is not None else "LSTM trained=N/A"
            lstm_log_str = f"LSTM op_log={lstm_log_acc:.1f}%" if lstm_log_acc is not None else "LSTM op_log=N/A"
            logger.info(
                f"[ACCURACY] {ml_str} | {ml_op_str} | {lstm_str} | {lstm_log_str}"
            )
            return result
        except Exception as e:
            logger.warning(f"[ACCURACY] Erro ao obter acurácias: {e}")
            return {"ml_accuracy": None, "ml_trained_accuracy": None, "ml_operational_accuracy": None,
                    "ml_class_accuracy": None, "lstm_accuracy": None, "lstm_trained_accuracy": None,
                    "lstm_operational_accuracy": None}

    def _check_ml_prediction_bias(self) -> bool:
        """
        Verifica se o modelo ML está viciado (predizendo >70% como mesma classe).
        Threshold reduzido de 85%→70% para detectar bias mais cedo.

        Returns:
            True se modelo está viciado, False se distribuição é razoável
        """
        try:
            import json as _json
            predictions = []

            # Fonte primária: prediction_log.json
            pred_log_path = os.path.join("ml_models", "prediction_log.json")
            if os.path.exists(pred_log_path):
                with open(pred_log_path, 'r') as f:
                    predictions = _json.load(f)

            # Fallback: model_votes_log.jsonl (sempre tem ml_prediction se ML rodou)
            if not predictions:
                votes_log_path = os.path.join("signals", "model_votes_log.jsonl")
                if os.path.exists(votes_log_path):
                    with open(votes_log_path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    record = _json.loads(line)
                                    if record.get("ml_prediction") is not None:
                                        predictions.append(record)
                                except _json.JSONDecodeError:
                                    continue

            if not predictions:
                return False  # Sem dados, não podemos afirmar que é viciado

            # Considerar apenas as últimas 50 predições (janela recente)
            recent = predictions[-50:]
            if len(recent) < 10:
                return False  # Poucos dados para concluir

            skip_count = sum(1 for p in recent if p.get('ml_prediction', 0) == 0)
            skip_ratio = skip_count / len(recent)

            # Se >70% é mesma classe (SKIP ou EXECUTE), modelo está viciado
            # Threshold reduzido de 85% → 70%: detectar bias mais cedo
            if skip_ratio > 0.70 or skip_ratio < 0.30:
                logger.info(
                    f"[ML BIAS] Modelo viciado detectado: "
                    f"SKIP={skip_count}/{len(recent)} ({skip_ratio:.1%}), "
                    f"EXECUTE={len(recent)-skip_count}/{len(recent)} ({1-skip_ratio:.1%})"
                )
                return True

            return False
        except Exception:
            return False  # Na dúvida, não bloquear

    def _validate_with_ml_model(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Consulta o modelo ML para obter voto (NÃO tem poder de veto).
        ML é apenas mais um voto no sistema de confluência (voto 9 de 9+).

        Returns:
            Dict com probability, prediction, ml_vote (+1, 0, -1)
        """
        # Se validador nao disponivel, voto neutro
        if self.ml_validator is None:
            return {
                "skip_signal": False,
                "ml_vote": 0,
                "probability": 0.5,
                "prediction": 0,
                "reason": "Validador ML nao disponivel"
            }

        try:
            # Extrair features do sinal
            features = {
                'rsi': signal.get('rsi', signal.get('indicators', {}).get('rsi', 50)),
                'macd_histogram': signal.get('macd_histogram', signal.get('indicators', {}).get('macd_histogram', 0)),
                'adx': signal.get('adx', signal.get('indicators', {}).get('adx', 25)),
                'atr': signal.get('atr', signal.get('indicators', {}).get('atr', 0)),
                'bb_position': signal.get('bb_position', signal.get('indicators', {}).get('bb_position', 0.5)),
                'cvd': signal.get('cvd', signal.get('order_flow', {}).get('cvd', 0)),
                'orderbook_imbalance': signal.get('orderbook_imbalance', signal.get('order_flow', {}).get('orderbook_imbalance', 0.5)),
                'bullish_tf_count': signal.get('bullish_tf_count', 0),
                'bearish_tf_count': signal.get('bearish_tf_count', 0),
                'confidence': signal.get('confidence', 5),
                'trend_encoded': self._encode_trend(signal.get('trend', 'neutral')),
                'sentiment_encoded': self._encode_sentiment(signal.get('sentiment', 'neutral')),
                'signal_encoded': 1 if signal.get('signal') == 'BUY' else 0,
                'risk_distance_pct': self._calc_risk_distance(signal),
                'reward_distance_pct': self._calc_reward_distance(signal),
                'risk_reward_ratio': self._calc_risk_reward(signal),
            }

            # Fazer predicao
            result = self.ml_validator.predict_signal(features)

            probability = result.get('probability_success', 0.5)
            prediction = result.get('prediction', 0)

            # Registrar predicao no dashboard
            self.ml_validator._log_prediction(signal, result)

            # ML como VOTO (não veto):
            # Dados reais mostram: SKIP (prob<50%) tem 81% accuracy,
            # EXECUTE (prob>50%) tem apenas 32% accuracy.
            # Portanto: confiar MUITO no SKIP, exigir prob ALTA para EXECUTE.
            #
            # prob >= 0.75 → voto a favor (+1) — só sinais muito fortes
            # prob < 0.30 → voto FORTE contra (-2) — SKIP é 81% confiável
            # prob 0.30-0.50 → voto contra (-1)
            # prob 0.50-0.75 → neutro (0) — zona não confiável
            ml_vote = 0
            ml_is_biased = self._check_ml_prediction_bias()

            if ml_is_biased:
                ml_vote = 0  # Modelo viciado = voto neutro
                logger.info(
                    f"[ML VOTO] Modelo viciado (>70% mesma classe) — voto NEUTRO "
                    f"(prob={probability:.1%}, pred={prediction})"
                )
            elif probability >= 0.75:
                ml_vote = 1  # Voto a favor — só com alta confiança
                logger.info(
                    f"[ML VOTO] A FAVOR — prob={probability:.1%} (>= 75%)"
                )
            elif probability < 0.30:
                ml_vote = -2  # Voto FORTE contra — SKIP accuracy 81%
                logger.info(
                    f"[ML VOTO] FORTE CONTRA (-2) — prob={probability:.1%} (< 30%)"
                )
            elif probability < 0.50:
                ml_vote = -1  # Voto contra
                logger.info(
                    f"[ML VOTO] CONTRA — prob={probability:.1%} (< 50%)"
                )
            else:
                ml_vote = 0  # Neutro — zona 50-75% não confiável
                logger.info(
                    f"[ML VOTO] NEUTRO — prob={probability:.1%} (zona 50-75% não confiável)"
                )

            return {
                "skip_signal": False,  # ML NUNCA bloqueia sozinho
                "ml_vote": ml_vote,
                "probability": probability,
                "prediction": prediction,
                "reason": f"ML voto={'A FAVOR' if ml_vote > 0 else 'CONTRA' if ml_vote < 0 else 'NEUTRO'} (prob={probability:.1%})"
            }

        except Exception as e:
            logger.warning(f"[ML] Erro na validacao ML: {e}")
            return {
                "skip_signal": False,
                "ml_vote": 0,
                "probability": 0.5,
                "prediction": 0,
                "reason": f"Erro: {e}"
            }

    _voter_accuracy_cache = None
    _voter_accuracy_cache_time = None

    def _get_cached_voter_accuracy(self) -> Dict:
        """Get voter accuracy with 1h cache to avoid recomputing every analysis."""
        now = datetime.now(timezone.utc)
        if (self._voter_accuracy_cache is not None and
                self._voter_accuracy_cache_time is not None and
                (now - self._voter_accuracy_cache_time).total_seconds() < 3600):
            return self._voter_accuracy_cache

        try:
            from src.trading.signal_tracker import compute_voter_accuracy
            self._voter_accuracy_cache = compute_voter_accuracy()
            self._voter_accuracy_cache_time = now
            return self._voter_accuracy_cache
        except Exception:
            return {}

    def _encode_trend(self, trend: str) -> int:
        """Codifica tendencia para valor numerico"""
        trend_map = {'strong_bullish': 2, 'bullish': 1, 'neutral': 0, 'bearish': -1, 'strong_bearish': -2}
        return trend_map.get(trend.lower() if trend else 'neutral', 0)

    def _encode_sentiment(self, sentiment: str) -> int:
        """Codifica sentimento para valor numerico"""
        sentiment_map = {'bullish': 1, 'neutral': 0, 'bearish': -1}
        return sentiment_map.get(sentiment.lower() if sentiment else 'neutral', 0)

    def _calc_risk_distance(self, signal: Dict) -> float:
        """Calcula distancia do stop em %"""
        entry = signal.get('entry_price', 0)
        stop = signal.get('stop_loss', 0)
        if entry > 0 and stop > 0:
            return abs(entry - stop) / entry * 100
        return 2.0  # Default 2%

    def _calc_reward_distance(self, signal: Dict) -> float:
        """Calcula distancia do TP1 em %"""
        entry = signal.get('entry_price', 0)
        tp1 = signal.get('take_profit_1', signal.get('take_profit', 0))
        if entry > 0 and tp1 > 0:
            return abs(tp1 - entry) / entry * 100
        return 2.0  # Default 2%

    def _calc_risk_reward(self, signal: Dict) -> float:
        """Calcula risk/reward ratio"""
        risk = self._calc_risk_distance(signal)
        reward = self._calc_reward_distance(signal)
        if risk > 0:
            return reward / risk
        return 1.0

    def _calculate_technical_confluence(self, analysis_data: Dict, signal_direction: str) -> Dict[str, Any]:
        """
        Calcula score de confluência técnica local a partir dos dados já coletados.
        Cada condição alinhada com a direção do sinal conta como +1 voto.

        Args:
            analysis_data: Dados coletados por prepare_analysis_for_llm()
            signal_direction: "BUY" ou "SELL"

        Returns:
            Dict com votes_for, votes_against, total_votes, score (0-1), details
        """
        votes_for = 0
        votes_against = 0
        details = []
        # Tracking individual de cada votante: +1 (a favor), -1 (contra), 0 (neutro)
        voter_votes = {}

        is_buy = signal_direction == "BUY"

        # Carregar thresholds otimizados (se disponíveis)
        try:
            from src.backtesting.continuous_optimizer import load_best_config
            best = load_best_config(analysis_data.get("symbol", "BTCUSDT"), "1h")
        except Exception:
            best = None

        # Thresholds padrão (ou otimizados se disponíveis)
        rsi_oversold = getattr(best, "rsi_oversold", 30) if best else 30
        rsi_overbought = getattr(best, "rsi_overbought", 70) if best else 70
        adx_threshold = getattr(best, "adx_min_strength", getattr(best, "adx_threshold", 25)) if best else 25
        bb_std = getattr(best, "bb_std", 2.0) if best else 2.0
        volume_surge = getattr(best, "volume_surge_multiplier", 1.5) if best else 1.5
        thresholds_source = "optimizer" if best else "default"
        indicators = analysis_data.get("key_indicators", {})
        trend_data = analysis_data.get("trend_analysis", {})
        volume_flow = analysis_data.get("volume_flow", {})
        mtf = analysis_data.get("multi_timeframe", {})

        # 1. RSI zone alignment
        rsi = indicators.get("rsi", {}).get("value", 50)
        if is_buy and rsi < rsi_oversold:
            votes_for += 1
            voter_votes["rsi"] = 1
            details.append(f"RSI oversold ({rsi:.1f} < {rsi_oversold})")
        elif not is_buy and rsi > rsi_overbought:
            votes_for += 1
            voter_votes["rsi"] = 1
            details.append(f"RSI overbought ({rsi:.1f} > {rsi_overbought})")
        elif is_buy and rsi > rsi_overbought:
            votes_against += 1
            voter_votes["rsi"] = -1
            details.append(f"RSI contra BUY ({rsi:.1f} > {rsi_overbought})")
        elif not is_buy and rsi < rsi_oversold:
            votes_against += 1
            voter_votes["rsi"] = -1
            details.append(f"RSI contra SELL ({rsi:.1f} < {rsi_oversold})")
        else:
            voter_votes["rsi"] = 0

        # 2. MACD histogram direction (zona morta para valores insignificantes)
        macd_hist = indicators.get("macd", {}).get("histogram", 0)
        # Zona morta: MACD muito próximo de zero = neutro (não vota)
        # Threshold relativo ao preço do ativo (0.01% do preço)
        _price = indicators.get("close", indicators.get("price", 1))
        _macd_deadzone = abs(_price) * 0.0001 if _price else 0.0001
        if abs(macd_hist) < _macd_deadzone:
            voter_votes["macd"] = 0
            details.append(f"MACD neutro ({macd_hist:.6f}, zona morta)")
        elif (is_buy and macd_hist > 0) or (not is_buy and macd_hist < 0):
            votes_for += 1
            voter_votes["macd"] = 1
            details.append(f"MACD aligned ({macd_hist:.4f})")
        elif (is_buy and macd_hist < 0) or (not is_buy and macd_hist > 0):
            votes_against += 1
            voter_votes["macd"] = -1
            details.append(f"MACD contra ({macd_hist:.4f})")

        # 3. EMA alignment (trend direction)
        primary_trend = trend_data.get("primary_trend", "neutral")
        if (is_buy and primary_trend in ("bullish", "strong_bullish")) or \
           (not is_buy and primary_trend in ("bearish", "strong_bearish")):
            votes_for += 1
            voter_votes["trend"] = 1
            details.append(f"Trend aligned ({primary_trend})")
        elif (is_buy and primary_trend in ("bearish", "strong_bearish")) or \
             (not is_buy and primary_trend in ("bullish", "strong_bullish")):
            votes_against += 1
            voter_votes["trend"] = -1
            details.append(f"Trend contra ({primary_trend})")
        else:
            voter_votes["trend"] = 0

        # 4. ADX trend strength
        # ADX < 20 = mercado lateral forte → vota CONTRA (não neutro!)
        # ADX 20-25 = mercado indeciso → neutro
        # ADX >= 25 = tendência → vota a favor
        adx = trend_data.get("trend_strength_adx", trend_data.get("adx", 0))
        if adx >= adx_threshold:
            votes_for += 1
            voter_votes["adx"] = 1
            details.append(f"ADX strong trend ({adx:.1f} >= {adx_threshold})")
        elif adx < 20:
            votes_against += 1
            voter_votes["adx"] = -1
            details.append(f"ADX LATERAL ({adx:.1f} < 20 — mercado sem tendência)")
        else:
            voter_votes["adx"] = 0
            details.append(f"ADX weak trend ({adx:.1f} < {adx_threshold})")

        # 5. Bollinger Band position (thresholds ajustados pelo bb_std otimizado)
        bb_pos = indicators.get("bollinger", {}).get("position", 0.5)
        # bb_std mais alto = bandas mais largas = threshold mais generoso
        # bb_std=2.0 → 0.20/0.80, bb_std=1.5 → 0.25/0.75, bb_std=3.0 → 0.15/0.85
        bb_lower_thr = max(0.10, 0.40 - bb_std * 0.10)
        bb_upper_thr = min(0.90, 0.60 + bb_std * 0.10)
        if is_buy and bb_pos < bb_lower_thr:
            votes_for += 1
            voter_votes["bb"] = 1
            details.append(f"BB near lower band ({bb_pos:.2f} < {bb_lower_thr:.2f})")
        elif not is_buy and bb_pos > bb_upper_thr:
            votes_for += 1
            voter_votes["bb"] = 1
            details.append(f"BB near upper band ({bb_pos:.2f} > {bb_upper_thr:.2f})")
        elif is_buy and bb_pos > bb_upper_thr:
            votes_against += 1
            voter_votes["bb"] = -1
            details.append(f"BB contra BUY ({bb_pos:.2f} > {bb_upper_thr:.2f})")
        elif not is_buy and bb_pos < bb_lower_thr:
            votes_against += 1
            voter_votes["bb"] = -1
            details.append(f"BB contra SELL ({bb_pos:.2f} < {bb_lower_thr:.2f})")
        else:
            voter_votes["bb"] = 0

        # 6. Orderbook imbalance + volume surge alignment
        ob_imbalance = volume_flow.get("orderbook_imbalance", 0)
        ob_bias = volume_flow.get("orderbook_bias", "neutral")
        vol_ratio = volume_flow.get("volume_ratio", volume_flow.get("volume_surge_ratio", 1.0))
        has_volume_surge = vol_ratio >= volume_surge
        # Valores possíveis: strong_buy_pressure, buy_pressure, neutral, sell_pressure, strong_sell_pressure
        if (is_buy and "buy" in ob_bias) or (not is_buy and "sell" in ob_bias):
            votes_for += 1
            voter_votes["orderbook"] = 1
            surge_str = f", SURGE x{vol_ratio:.1f}" if has_volume_surge else ""
            details.append(f"Orderbook aligned ({ob_bias}, imb={ob_imbalance:.2f}{surge_str})")
        elif (is_buy and "sell" in ob_bias) or (not is_buy and "buy" in ob_bias):
            votes_against += 1
            voter_votes["orderbook"] = -1
            details.append(f"Orderbook contra ({ob_bias}, imb={ob_imbalance:.2f})")
        else:
            # Volume surge sem bias claro = ainda conta como confirmação parcial
            if has_volume_surge:
                voter_votes["orderbook"] = 0
                details.append(f"Orderbook neutro, mas volume surge x{vol_ratio:.1f} >= {volume_surge:.1f}")
            else:
                voter_votes["orderbook"] = 0

        # 7. Multi-timeframe alignment
        bullish_count = mtf.get("bullish_count", 0)
        bearish_count = mtf.get("bearish_count", 0)
        if is_buy and bullish_count >= 3:
            votes_for += 1
            voter_votes["mtf"] = 1
            details.append(f"MTF aligned ({bullish_count}/5 bullish)")
        elif not is_buy and bearish_count >= 3:
            votes_for += 1
            voter_votes["mtf"] = 1
            details.append(f"MTF aligned ({bearish_count}/5 bearish)")
        elif is_buy and bearish_count >= 3:
            votes_against += 1
            voter_votes["mtf"] = -1
            details.append(f"MTF contra BUY ({bearish_count}/5 bearish)")
        elif not is_buy and bullish_count >= 3:
            votes_against += 1
            voter_votes["mtf"] = -1
            details.append(f"MTF contra SELL ({bullish_count}/5 bullish)")
        else:
            voter_votes["mtf"] = 0

        # 8. CVD (Cumulative Volume Delta) alignment
        cvd_direction = volume_flow.get("cvd_direction", "neutral")
        if (is_buy and cvd_direction == "positive") or (not is_buy and cvd_direction == "negative"):
            votes_for += 1
            voter_votes["cvd"] = 1
            details.append(f"CVD aligned ({cvd_direction})")
        elif (is_buy and cvd_direction == "negative") or (not is_buy and cvd_direction == "positive"):
            votes_against += 1
            voter_votes["cvd"] = -1
            details.append(f"CVD contra ({cvd_direction})")
        else:
            voter_votes["cvd"] = 0

        # 9. Setup Validator — valida contexto contra win rate histórico
        try:
            from src.optimizer.setup_validator import validate_signal_before_trade
            symbol = analysis_data.get("symbol", "")
            rsi_val = rsi_data.get("rsi", 50)
            adx_val = trend_data.get("adx", 25)
            atr_val = trend_data.get("atr", 0)
            entry_price = analysis_data.get("price", {}).get("current", 1)
            ema_fast = trend_data.get("ema_fast", entry_price)
            ema_slow = trend_data.get("ema_slow", entry_price)
            if entry_price > ema_fast > ema_slow:
                sv_trend = "bullish"
            elif entry_price < ema_fast < ema_slow:
                sv_trend = "bearish"
            else:
                sv_trend = "neutral"

            sv_result = validate_signal_before_trade({
                "signal": signal_direction,
                "rsi": rsi_val,
                "adx": adx_val,
                "atr": atr_val,
                "entry_price": entry_price,
                "trend": sv_trend,
            })
            sv_vote = sv_result.get("vote", 0)
            sv_rec = sv_result.get("recommendation", "NEW_SETUP")
            sv_wr = sv_result.get("historical_win_rate", 0)
            sv_samples = sv_result.get("total_samples", 0)

            if sv_vote > 0:
                votes_for += 1
                voter_votes["setup_validator"] = 1
                details.append(f"Setup aprovado ({sv_rec}, WR={sv_wr:.0f}%, n={sv_samples})")
            elif sv_vote < 0:
                votes_against += 1
                voter_votes["setup_validator"] = -1
                details.append(f"Setup rejeitado ({sv_rec}, WR={sv_wr:.0f}%, n={sv_samples})")
            else:
                voter_votes["setup_validator"] = 0
                details.append(f"Setup neutro ({sv_rec}, n={sv_samples})")
        except Exception:
            voter_votes["setup_validator"] = 0

        # Adaptive voter weights: invert votes from voters with proven <45% accuracy
        # This turns consistently wrong voters into useful contrarian signals
        try:
            voter_accuracy = self._get_cached_voter_accuracy()
            CONTRARIAN_VOTERS = ["rsi", "macd", "trend", "adx", "bb", "orderbook", "cvd", "regime"]
            MIN_SAMPLES_FOR_INVERSION = 20
            INVERSION_THRESHOLD = 0.45  # Invert if accuracy < 45%

            for voter_name in CONTRARIAN_VOTERS:
                acc_data = voter_accuracy.get(voter_name, {})
                total = acc_data.get("total", 0)
                accuracy = acc_data.get("accuracy")

                if total >= MIN_SAMPLES_FOR_INVERSION and accuracy is not None and accuracy < INVERSION_THRESHOLD:
                    original_vote = voter_votes.get(voter_name, 0)
                    if original_vote != 0:
                        # Invert the vote
                        inverted = -original_vote
                        # Adjust vote counts
                        if original_vote > 0:
                            votes_for -= 1
                            votes_against += 1
                        else:
                            votes_against -= 1
                            votes_for += 1
                        voter_votes[voter_name] = inverted
                        details.append(
                            f"{voter_name.upper()} INVERTIDO (acc={accuracy:.0%}, n={total})"
                        )
        except Exception:
            pass

        total_votes = votes_for + votes_against
        score = votes_for / max(total_votes, 1)

        return {
            "votes_for": votes_for,
            "votes_against": votes_against,
            "total_votes": total_votes,
            "score": round(score, 3),
            "details": details,
            "voter_votes": voter_votes,
            "thresholds_source": "optimizer" if best else "default",
        }

    # REFATORADO: Constante de preços padrão para evitar duplicação
    DEFAULT_PRICES = {
        "BTCUSDT": 95000, "ETHUSDT": 2500, "SOLUSDT": 150,
        "BNBUSDT": 650, "ADAUSDT": 0.70, "XRPUSDT": 2.50,
        "DOGEUSDT": 0.25, "AVAXUSDT": 35, "DOTUSDT": 7,
        "LINKUSDT": 18, "PAXGUSDT": 5100
    }

    def _extract_price_from_text(self, text: str, min_price: float = 0.01, max_price: float = 1000000) -> Optional[float]:
        """
        REFATORADO: Função helper para extrair preço de texto.
        Elimina código duplicado de extração de preço.

        Args:
            text: Texto para buscar preço
            min_price: Preço mínimo válido
            max_price: Preço máximo válido

        Returns:
            Preço extraído ou None se não encontrado
        """
        if not text:
            return None

        price_patterns = [
            r"\$([0-9,]+\.?[0-9]+)",  # $90,563.50
            r"([0-9]{1,3}(?:[,.][0-9]{1,2})?)\s*(?:USD|USDT)",  # 90,563.50 USD
            r"preço[^0-9]*([0-9,]+\.?[0-9]+)",  # preço 90,563.50
            r"preco[^0-9]*([0-9,]+\.?[0-9]+)",  # preco 90,563.50
            r"entrada[^0-9]*\$?([0-9,]+\.?[0-9]*)",
            r"entry[^0-9]*\$?([0-9,]+\.?[0-9]*)",
            r"entry_price[^0-9]*[:=]\s*\$?([0-9,]+\.?[0-9]*)",
            r"current[^0-9]*price[^0-9]*\$?([0-9,]+\.?[0-9]*)"
        ]

        for pattern in price_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    price_str = match.group(1).replace(",", "")
                    price = float(price_str)
                    if min_price <= price <= max_price:
                        return price
                except ValueError:
                    continue

        return None

    def _get_default_price(self, symbol: str) -> Optional[float]:
        """Retorna preço padrão para um símbolo (None se desconhecido)"""
        return self.DEFAULT_PRICES.get(symbol)

    def _calculate_stop_loss(self, entry_price: float, signal_type: str, distance_pct: float = 0.02) -> float:
        """
        REFATORADO: Calcula stop loss baseado no tipo de sinal.

        Args:
            entry_price: Preço de entrada
            signal_type: BUY ou SELL
            distance_pct: Distância percentual do stop (default 2%)

        Returns:
            Preço do stop loss
        """
        if signal_type == "BUY":
            return entry_price * (1 - distance_pct)
        else:  # SELL
            return entry_price * (1 + distance_pct)

    def _get_instructions(self) -> str:
        """Retorna as instruções para o agent com regras claras de trading"""
        return """Voce e um trader quantitativo. Dados de mercado ja coletados no prompt.
Analise os dados e decida com base nestas REGRAS OBJETIVAS:

## REGRAS PARA BUY:
1. Tendencia 4h/1d deve ser bullish ou neutral (NUNCA compre contra tendencia bearish)
2. RSI entre 30-55 (zona de compra - NAO compre com RSI > 65 = sobrecomprado)
3. MACD histogram positivo OU crossover bullish recente
4. Preco proximo ao suporte (distance_to_support < 2%)
5. Confluencia: minimo 3/5 timeframes bullish (TF >= 3)
6. EMA alignment bullish (preco > EMA20 > EMA50)
7. Volume confirmando (OBV bullish ou orderbook buy pressure)

## REGRAS PARA SELL (SHORT):
1. Tendencia 4h/1d deve ser bearish ou neutral (NUNCA venda contra tendencia bullish)
2. RSI entre 45-70 (zona de venda - NAO venda com RSI < 35 = sobrevendido)
3. MACD histogram negativo OU crossover bearish recente
4. Preco proximo a resistencia (distance_to_resistance < 2%)
5. Confluencia: minimo 3/5 timeframes bearish (TF <= -3)
6. EMA alignment bearish (preco < EMA20 < EMA50)
7. Volume confirmando (OBV bearish ou orderbook sell pressure)

## IMPORTANTE - VOCE DEVE SEMPRE RESPONDER BUY OU SELL:
- Voce NAO decide se o sinal sera executado. O sistema decide com base na confianca.
- Analise os dados e diga qual direcao (BUY ou SELL) e MAIS PROVAVEL.
- Use a confianca (1-10) para indicar a forca do sinal.
- Confianca baixa (1-5) = sinal fraco, o sistema provavelmente NAO executara.
- Confianca alta (6-10) = sinal forte, o sistema provavelmente executara.
- NUNCA responda NO_SIGNAL. Sempre escolha BUY ou SELL com a confianca adequada.

## STOPS E TARGETS (OBRIGATORIO - NUNCA DEIXE EM 0):
- Stop loss DEVE estar atras de suporte/resistencia real (nao arbitrario)
- Para BUY: SL abaixo do suporte. Para SELL: SL acima da resistencia
- TP1 no proximo nivel de resistencia/suporte. TP2 no nivel seguinte
- Minimo 2:1 reward/risk (TP1 deve ser 2x a distancia do SL)
- TODOS os campos (entry_price, stop_loss, take_profit_1, take_profit_2) DEVEM ter valores reais > 0

## CONFIDENCE (escala 1-10):
- 1-3: Mercado indeciso/lateral, sem confluencia clara
- 4-5: Poucos indicadores alinhados, conflitos significativos
- 6-7: 4-5 regras confirmando, conflitos menores
- 8-10: 6+ regras confirmando, sem conflitos, tendencia forte

Responda APENAS com JSON:
```json
{"signal":"BUY/SELL","operation_type":"SCALP/DAY_TRADE/SWING_TRADE","entry_price":0,"stop_loss":0,"take_profit_1":0,"take_profit_2":0,"confidence":7,"reasoning":"Regras X,Y,Z confirmadas. Conflitos: nenhum/ABC"}
```"""

    async def analyze(self, symbol: str = "BTCUSDT") -> Dict[str, Any]:
        """
        Executa análise completa usando o AGNO Agent.
        CORRIGIDO: Verifica posição existente antes de analisar.

        Args:
            symbol: Símbolo para analisar

        Returns:
            Sinal de trading estruturado
        """
        logger.info(f"[AGNO] === Iniciando analise de {symbol} ===")

        # Verificar posição e limpar ordens órfãs antes de analisar
        from src.core.config import settings
        if settings.trading_mode == "real":
            try:
                from src.exchange.executor import BinanceFuturesExecutor
                executor = BinanceFuturesExecutor()

                # 1. Verificar posição existente
                existing_position = await executor.get_position(symbol)
                has_position = existing_position and "position_amt" in existing_position

                # 2. Se NÃO tem posição, verificar se tem ordens órfãs para cancelar
                if not has_position:
                    open_orders = await executor.get_open_orders(symbol)
                    if open_orders and len(open_orders) > 0:
                        logger.info(f"[LIMPEZA] {symbol}: Cancelando {len(open_orders)} ordens orfas")
                        await executor.cancel_all_orders(symbol)

                # 3. Se TEM posição, pular análise
                if has_position:
                    side = existing_position.get("side", "UNKNOWN")
                    amt = abs(existing_position.get("position_amt", 0))
                    logger.warning(f"[BINANCE] Ja existe posicao {side} aberta para {symbol} ({amt} unidades). Pulando analise.")
                    return {
                        "symbol": symbol,
                        "signal": "NO_SIGNAL",
                        "confidence": 0,
                        "reason": f"Ja existe posicao {side} aberta na Binance para {symbol}",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "source": "AGNO"
                    }
            except Exception as e:
                # Se erro ao verificar, continuar com análise (log warning)
                logger.warning(f"Erro ao verificar posicao na Binance: {e}")

        # A verificação de posição existente no paper trading será feita em validate_risk_and_position()

        # ========================================
        # COOLDOWN PÓS-FECHAMENTO: Impede análise de símbolo recém-fechado
        # Bug fix: DRIFTUSDT bateu TP2 e reabriu no mesmo ciclo
        # ========================================
        try:
            from src.trading.risk_manager import _check_sl_cooldown, _sl_cooldown_registry, _sl_cooldown_hours
            if _check_sl_cooldown(symbol):
                remaining = _sl_cooldown_hours - (datetime.now(timezone.utc) - _sl_cooldown_registry[symbol]).total_seconds() / 3600
                logger.warning(f"[COOLDOWN PÓS-FECHAMENTO] {symbol}: análise bloqueada — cooldown ativo ({remaining:.1f}h restantes)")
                return {
                    "symbol": symbol,
                    "signal": "NO_SIGNAL",
                    "confidence": 0,
                    "reason": f"Cooldown pós-fechamento ativo: {remaining:.1f}h restantes (evita reentrada imediata)",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
        except Exception as e:
            logger.warning(f"Erro ao verificar cooldown pós-fechamento: {e}")

        # ========================================
        # COOLDOWN ROBUSTO: Memória + Arquivo
        # Impede análise repetida do mesmo símbolo
        # ========================================
        from src.core.config import settings
        min_interval = settings.min_analysis_interval_hours

        # 1. Verificação em MEMÓRIA (impossível de bypassar)
        now = datetime.now(timezone.utc)
        if symbol in AgnoTradingAgent._last_analysis_time:
            last_time = AgnoTradingAgent._last_analysis_time[symbol]
            hours_since = (now - last_time).total_seconds() / 3600
            if hours_since < min_interval:
                remaining = int((min_interval - hours_since) * 60)
                logger.info(f"[COOLDOWN] {symbol}: ultima analise ha {int(hours_since*60)}min. Proximo em {remaining}min (intervalo {min_interval}h)")
                return {
                    "symbol": symbol,
                    "signal": "NO_SIGNAL",
                    "confidence": 0,
                    "reason": f"Cooldown ativo: ultima analise ha {int(hours_since*60)} minutos (minimo {min_interval}h)",
                    "timestamp": now.isoformat()
                }

        # 2. Verificação em ARQUIVO (para persistir entre restarts)
        try:
            last_analysis_file = f"signals/agno_{symbol}_last_analysis.json"
            if os.path.exists(last_analysis_file):
                with open(last_analysis_file, "r", encoding='utf-8') as f:
                    last_analysis = json.load(f)
                    last_timestamp_str = last_analysis.get("timestamp")
                    if last_timestamp_str:
                        try:
                            last_timestamp = datetime.fromisoformat(last_timestamp_str.replace('Z', '+00:00'))
                        except ValueError:
                            last_timestamp = datetime.fromisoformat(last_timestamp_str)

                        # Garantir que last_timestamp é timezone-aware (UTC)
                        if last_timestamp.tzinfo is None:
                            last_timestamp = last_timestamp.replace(tzinfo=timezone.utc)

                        hours_since = (now - last_timestamp).total_seconds() / 3600
                        if hours_since < min_interval:
                            # Atualizar cache em memória também
                            AgnoTradingAgent._last_analysis_time[symbol] = last_timestamp
                            remaining = int((min_interval - hours_since) * 60)
                            logger.info(f"[COOLDOWN] {symbol}: ultima analise ha {int(hours_since*60)}min (arquivo). Proximo em {remaining}min")
                            return {
                                "symbol": symbol,
                                "signal": "NO_SIGNAL",
                                "confidence": 0,
                                "reason": f"Cooldown ativo: ultima analise ha {int(hours_since*60)} minutos (minimo {min_interval}h)",
                                "timestamp": now.isoformat()
                            }
        except Exception as e:
            logger.warning(f"Erro ao verificar ultima analise em arquivo: {e}")

        try:
            # Verificar configuração de sinais
            from src.core.config import settings

            # OTIMIZAÇÃO: Só gerar sinal DeepSeek se estiver habilitado
            deepseek_signal = None

            if settings.accept_deepseek_signals:
                # 1. SINAL DEEPSEEK DIRETO (só se habilitado)
                logger.info(f"[DEEPSEEK] Gerando sinal DeepSeek para {symbol}...")
                deepseek_result = await get_deepseek_analysis(symbol)

                if isinstance(deepseek_result, dict) and "signal" in deepseek_result:
                    logger.info(f"[SINAL DEEPSEEK] Sinal direto: {deepseek_result.get('signal', 'N/A')}")
                    deepseek_signal = {
                        "symbol": symbol,
                        "source": "DEEPSEEK",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "signal": deepseek_result.get("signal", "NO_SIGNAL"),
                        "entry_price": deepseek_result.get("entry_price"),
                        "stop_loss": deepseek_result.get("stop_loss"),
                        "take_profit_1": deepseek_result.get("take_profit_1"),
                        "take_profit_2": deepseek_result.get("take_profit_2"),
                        "confidence": deepseek_result.get("confidence", 5),
                        "reasoning": deepseek_result.get("reasoning", ""),
                        "raw_response": deepseek_result.get("raw_response", "")
                    }

                    self._save_deepseek_response(
                        symbol,
                        deepseek_result.get("deepseek_prompt", ""),
                        deepseek_result.get("raw_response", ""),
                        deepseek_result.get("analysis_data", {})
                    )
                    # DeepSeek direto é apenas mais uma fonte de sinal.
                    # A decisão final de abrir/fechar posição é sempre feita
                    # pelo nosso sistema de confluência + ML/LSTM/risk.
                    self._save_signal(deepseek_signal)

                    if deepseek_signal.get("signal") in ["BUY", "SELL"]:
                        # HARD BLOCK: Verificar tendência 4h ANTES de tudo
                        try:
                            trend_data = await get_trend(symbol)
                        except Exception as e:
                            logger.warning(f"[TREND] Erro ao obter tendência: {e}")
                            trend_data = None

                        # Bloquear sinais contra tendência antes de validar
                        if trend_data:
                            ds_dir = deepseek_signal.get("signal")
                            if ds_dir == "BUY" and not trend_data.get("allow_long", True):
                                logger.warning(f"╔══ [BLOQUEADO] DEEPSEEK BUY {symbol} — Tendência de baixa no 4h")
                                deepseek_signal["signal"] = "NO_SIGNAL"
                            elif ds_dir == "SELL" and not trend_data.get("allow_short", True):
                                logger.warning(f"╔══ [BLOQUEADO] DEEPSEEK SELL {symbol} — Tendência de alta no 4h")
                                deepseek_signal["signal"] = "NO_SIGNAL"

                    # Validar SL/TP1/TP2 obrigatórios antes de executar
                    if deepseek_signal.get("signal") in ["BUY", "SELL"]:
                        ds_sl = deepseek_signal.get("stop_loss", 0) or 0
                        ds_tp1 = deepseek_signal.get("take_profit_1", 0) or 0
                        ds_tp2 = deepseek_signal.get("take_profit_2", 0) or 0
                        if ds_sl <= 0 or ds_tp1 <= 0 or ds_tp2 <= 0:
                            logger.error(f"[DEEPSEEK BLOCK] SL=${ds_sl}, TP1=${ds_tp1}, TP2=${ds_tp2} — valores obrigatórios faltando")
                            deepseek_signal["signal"] = "NO_SIGNAL"
                            deepseek_signal["block_reason"] = f"SL/TP inválidos: SL=${ds_sl}, TP1=${ds_tp1}, TP2=${ds_tp2}"

                    if deepseek_signal.get("signal") in ["BUY", "SELL"]:
                        # Obter capital real da Binance (async)
                        _ds_capital = None
                        try:
                            from src.exchange.executor import BinanceFuturesExecutor
                            _ds_executor = BinanceFuturesExecutor()
                            _ds_balance = await _ds_executor.get_balance()
                            if "error" not in _ds_balance:
                                _ds_capital = _ds_balance.get("available_balance", 0)
                        except Exception as e:
                            logger.error(f"[CAPITAL] Falha ao obter saldo da Binance: {e}")

                        validation = validate_risk_and_position(deepseek_signal, symbol, _trend_data=trend_data, _capital=_ds_capital)
                        if validation.get("can_execute"):
                            logger.info(f"[DEEPSEEK] Executando sinal {deepseek_signal.get('signal')} para {symbol}")
                            position_size = validation.get("recommended_position_size", validation.get("position_size"))
                            execution_result = execute_paper_trade(deepseek_signal, position_size)
                            if execution_result.get("success"):
                                self._mark_signal_executed(deepseek_signal, "paper", True, execution_result.get("message", ""))
                                logger.info(f"[DEEPSEEK] Trade executado: {execution_result.get('message', '')}")
                            else:
                                self._mark_signal_executed(deepseek_signal, "paper", False, execution_result.get("error", ""))
                                logger.warning(f"[DEEPSEEK] Falha: {execution_result.get('error', '')}")
                        else:
                            logger.info(f"[DEEPSEEK] Não executado: {validation.get('reason', '')}")
            else:
                logger.info("[DEEPSEEK] Sinais DEEPSEEK desabilitados. Pulando analise DeepSeek direta.")

            # 2. SINAL AGNO - OTIMIZADO: Coleta dados localmente + 1 única chamada DeepSeek
            # ANTES: AGNO agent chamava ferramentas via DeepSeek (5+ API calls por símbolo)
            # AGORA: Coleta tudo localmente e envia 1 prompt com dados prontos
            from src.analysis.market_classifier import classify_market_condition
            from src.prompts.deepseek_prompt import _create_analysis_prompt, prepare_analysis_for_llm

            logger.info(f"[AGNO] Coletando dados localmente para {symbol}...")
            analysis_data = await prepare_analysis_for_llm(symbol)

            # ========================================
            # DETECÇÃO DE REGIME DE MERCADO
            # Identifica: BULL, BEAR, SIDEWAYS + volatilidade
            # Usado para: ajustar TP/SL, penalizar sinais laterais
            # ========================================
            market_regime = None
            try:
                from src.analysis.market_regime_detector import MarketRegimeDetectorFutures
                regime_detector = MarketRegimeDetectorFutures()
                market_regime = await regime_detector.detect_regime(symbol)
                base_regime = market_regime.get("base_regime", "SIDEWAYS")
                regime_name = market_regime.get("regime", "UNKNOWN")
                adx_regime = market_regime.get("details", {}).get("trend", {}).get("adx", 0)
                vol_regime = market_regime.get("details", {}).get("volatility", {}).get("volatility", "NORMAL")
                logger.info(
                    f"[REGIME] {symbol}: {regime_name} (base={base_regime}, "
                    f"ADX={adx_regime:.1f}, vol={vol_regime})"
                )
            except Exception as e:
                logger.warning(f"[REGIME] Erro na detecção de regime: {e}")
                market_regime = {"base_regime": "SIDEWAYS", "regime": "UNKNOWN", "confidence": 0.5, "details": {}}

            if "error" not in analysis_data:
                market_classification = classify_market_condition(analysis_data)
                prompt = _create_analysis_prompt(analysis_data, market_classification)

                # UMA única chamada DeepSeek (ao invés de 5+ via tool calls)
                logger.info(f"[AGNO] Chamando DeepSeek (1 chamada única) para {symbol}")
                response = await self.agent.arun(prompt)
            else:
                error_msg = analysis_data.get("error", "")
                # Símbolos em settling/delisted/closed não devem ser analisados
                if any(kw in str(error_msg) for kw in ["delivering", "settling", "closed", "pre-trading", "-4108"]):
                    logger.warning(f"[AGNO] {symbol} não está ativo (settling/closed) — pulando análise")
                    return {"signal": "NO_SIGNAL", "confidence": 0, "source": "AGNO",
                            "reason": f"Symbol not trading: {error_msg}", "symbol": symbol}
                logger.warning(f"[AGNO] Erro ao coletar dados para {symbol}: {error_msg} — pulando análise (sem dados confiáveis)")
                return {"signal": "NO_SIGNAL", "confidence": 0, "source": "AGNO",
                        "reason": f"Dados indisponíveis: {error_msg}", "symbol": symbol}

            # Salvar resposta bruta para auditoria
            self._save_deepseek_response(symbol, prompt if "error" not in analysis_data else "", response, analysis_data if "error" not in analysis_data else {})

            # Processar resposta do AGNO (voto da LLM)
            agno_signal = await self._process_agent_response(response, symbol)
            agno_signal["source"] = "AGNO"

            # Adicionar indicadores técnicos ao sinal para validação ML
            # OTIMIZADO: Usar dados já coletados em analysis_data ao invés de chamar APIs novamente
            try:
                if "error" not in analysis_data:
                    indicators = analysis_data.get("key_indicators", {})
                    trend_analysis = analysis_data.get("trend_analysis", {})
                    volume_flow = analysis_data.get("volume_flow", {})

                    agno_signal["rsi"] = indicators.get("rsi", {}).get("value", 50)
                    agno_signal["macd_histogram"] = indicators.get("macd", {}).get("histogram", 0)
                    agno_signal["adx"] = trend_analysis.get("trend_strength_adx", 25)
                    agno_signal["atr"] = analysis_data.get("volatility", {}).get("atr_value", 0)
                    agno_signal["bb_position"] = indicators.get("bollinger", {}).get("position", 0.5)
                    agno_signal["trend"] = trend_analysis.get("primary_trend", "neutral")
                    agno_signal["cvd_direction"] = volume_flow.get("cvd_direction", "neutral")
                    # CVD numérico para ML model (derivado da direção se raw não disponível)
                    agno_signal["cvd"] = 1.0 if volume_flow.get("cvd_direction") == "positive" else -1.0 if volume_flow.get("cvd_direction") == "negative" else 0.0
                    agno_signal["orderbook_imbalance"] = volume_flow.get("orderbook_imbalance", 0.5)
                    agno_signal["bullish_tf_count"] = analysis_data.get("multi_timeframe", {}).get("bullish_count", 0)
                    agno_signal["bearish_tf_count"] = analysis_data.get("multi_timeframe", {}).get("bearish_count", 0)
                    agno_signal["indicators"] = indicators

                    logger.debug(f"[ML] Indicadores do analysis_data: RSI={agno_signal.get('rsi')}, ADX={agno_signal.get('adx')}")
                else:
                    # Fallback: coletar indicadores separadamente
                    # analyze_technical_indicators auto-carrega params otimizados por símbolo
                    tech_data = await analyze_technical_indicators(symbol)
                    if tech_data and "indicators" in tech_data:
                        indicators = tech_data["indicators"]
                        agno_signal["rsi"] = indicators.get("rsi", 50)
                        agno_signal["macd_histogram"] = indicators.get("macd_histogram", 0)
                        agno_signal["adx"] = indicators.get("adx", 25)
                        agno_signal["atr"] = indicators.get("atr", 0)
                        agno_signal["bb_position"] = indicators.get("bb_position", 0.5)
                        agno_signal["trend"] = indicators.get("trend", "neutral")
                        agno_signal["indicators"] = indicators

                    order_flow = await analyze_order_flow(symbol)
                    if order_flow and "error" not in order_flow:
                        agno_signal["cvd"] = order_flow.get("cvd", 0)
                        agno_signal["orderbook_imbalance"] = order_flow.get("orderbook_imbalance", 0.5)

                    mtf_data = await analyze_multiple_timeframes(symbol)
                    if mtf_data and "error" not in mtf_data:
                        agno_signal["bullish_tf_count"] = mtf_data.get("bullish_tf_count", 0)
                        agno_signal["bearish_tf_count"] = mtf_data.get("bearish_tf_count", 0)
            except Exception as e:
                logger.warning(f"[ML] Erro ao coletar indicadores para ML: {e}")

            # Log único: o que o DeepSeek devolveu (sempre visível no fluxo de trading)
            _sig = agno_signal.get("signal", "N/A")
            _conf = agno_signal.get("confidence", 0)
            _reasoning = agno_signal.get("reasoning", "")
            logger.info(f"[AGNO] DeepSeek devolveu: Sinal={_sig}, Confiança={_conf}/10")
            if _reasoning:
                logger.info(f"[AGNO] Reasoning: {_reasoning[:200]}")

            # DeepSeek SEMPRE deve devolver BUY ou SELL com confiança 1-10.
            # Quem decide abrir ou não é o nosso sistema (confluência + ML/LSTM/risk).
            # Se devolveu NO_SIGNAL (modelo antigo/fallback), tratar como confiança 0.
            if _sig == "NO_SIGNAL":
                logger.info("[AGNO] DeepSeek devolveu NO_SIGNAL - sistema não executará (confiança insuficiente)")
                agno_signal["confidence"] = 0

            # ========================================
            # CONFLUÊNCIA: LLM é um voto, não a decisão final
            # Calcular score técnico local + voto LLM → decisão por confluência
            # ========================================
            llm_signal_dir = agno_signal.get("signal", "NO_SIGNAL")
            is_buy = llm_signal_dir == "BUY"
            is_sideways = market_regime and market_regime.get("base_regime") == "SIDEWAYS"

            if llm_signal_dir in ["BUY", "SELL"] and "error" not in analysis_data:
                # Drift check: pausar ML se drift severo detectado
                _ml_paused_by_drift = False
                try:
                    from src.analysis.drift_detector import get_drift_detector
                    if get_drift_detector().should_pause_ml():
                        _ml_paused_by_drift = True
                        logger.warning("[DRIFT] ML pausado — drift severo detectado, aguardando retreino")
                except Exception:
                    pass

                # VALIDAÇÃO ML: obter voto ML ANTES da confluência
                if _ml_paused_by_drift:
                    ml_validation = {"probability": 0.5, "prediction": 0, "ml_vote": 0}
                else:
                    ml_validation = self._validate_with_ml_model(agno_signal)
                ml_prob = ml_validation.get('probability', 0)
                ml_pred = ml_validation.get('prediction', 0)
                ml_vote = ml_validation.get("ml_vote", 0)

                # Salvar probabilidade ML no arquivo do sinal
                filepath = agno_signal.get("_signal_file")
                if filepath and os.path.exists(filepath):
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            saved = json.load(f)
                        saved["ml_probability"] = ml_prob
                        saved["ml_prediction"] = ml_pred
                        with open(filepath, "w", encoding="utf-8") as f:
                            json.dump(saved, f, indent=2, ensure_ascii=False, default=str)
                    except Exception:
                        pass

                agno_signal["ml_vote"] = ml_vote
                ml_opinion = 'A FAVOR' if ml_vote > 0 else 'CONTRA' if ml_vote < 0 else 'NEUTRO'
                logger.info(
                    f"[ML VOTO] {agno_signal.get('signal')} {symbol} — "
                    f"ML vota {ml_opinion} (prob={ml_prob:.1%})"
                )

                confluence = self._calculate_technical_confluence(
                    {**analysis_data, "symbol": symbol}, llm_signal_dir
                )
                votes_for = confluence["votes_for"]
                votes_against = confluence["votes_against"]

                # LLM conta como 1 voto a favor (peso ~20% do total, como no sinais)
                # Confidence da LLM (1-10) modula o peso: alta confiança = voto forte
                llm_confidence = agno_signal.get("confidence", 5)
                llm_vote_weight = 1 if llm_confidence >= 5 else 0.5

                # Bi-LSTM sequence vote (se modelo treinado)
                # Mesma lógica calibrada do ML: confiar no SKIP, exigir alta prob para FOR
                lstm_vote = 0
                lstm_prob = 0.5
                MIN_ACCURACY_TO_VOTE = 55.0
                if self.lstm_sequence_validator is not None:
                    try:
                        from src.backtesting.backtest_engine import BacktestEngine
                        _engine = BacktestEngine()
                        _df = await _engine.fetch_data(
                            symbol, "1h",
                            datetime.now(timezone.utc) - timedelta(
                                hours=self.lstm_sequence_validator.sequence_length + 10
                            ),
                            datetime.now(timezone.utc),
                        )
                        if not _df.empty:
                            _df = _engine.calculate_indicators(_df)
                            lstm_result = self.lstm_sequence_validator.predict_from_candles(_df)
                            lstm_prob = lstm_result.get("probability", 0.5)

                            # Log prediction para avaliação futura
                            self.lstm_sequence_validator._log_prediction(
                                symbol, lstm_prob,
                                1 if lstm_prob > 0.5 else 0,
                            )

                            accuracies = self._get_model_accuracies()
                            lstm_acc = accuracies.get("lstm_accuracy")
                            lstm_is_reliable = (
                                lstm_acc is not None
                                and lstm_acc >= MIN_ACCURACY_TO_VOTE
                            )

                            # Mesmos thresholds calibrados do ML:
                            # prob >= 0.75 → FOR, prob < 0.30 → FORTE CONTRA,
                            # prob 0.30-0.50 → CONTRA, prob 0.50-0.75 → NEUTRO
                            if lstm_is_reliable:
                                if lstm_prob >= 0.75:
                                    lstm_vote = 1
                                    confluence["details"].append(
                                        f"Bi-LSTM A FAVOR (prob={lstm_prob:.1%})"
                                    )
                                elif lstm_prob < 0.30:
                                    lstm_vote = -2
                                    confluence["details"].append(
                                        f"Bi-LSTM FORTE CONTRA (prob={lstm_prob:.1%})"
                                    )
                                elif lstm_prob < 0.50:
                                    lstm_vote = -1
                                    confluence["details"].append(
                                        f"Bi-LSTM CONTRA (prob={lstm_prob:.1%})"
                                    )
                                else:
                                    confluence["details"].append(
                                        f"Bi-LSTM neutro (prob={lstm_prob:.1%})"
                                    )
                            else:
                                acc_str = (
                                    f"{lstm_acc:.1f}%" if lstm_acc is not None
                                    else "N/A"
                                )
                                confluence["details"].append(
                                    f"Bi-LSTM prob={lstm_prob:.1%} "
                                    f"(ignorado: acc={acc_str})"
                                )

                            agno_signal["lstm_probability"] = lstm_prob
                            acc_str = (
                                f"{lstm_acc:.1f}%" if lstm_acc is not None
                                else "N/A"
                            )
                            vote_str = (
                                "FOR(+1)" if lstm_vote > 0
                                else f"AGAINST({lstm_vote})" if lstm_vote < 0
                                else "NEUTRAL"
                            )
                            logger.info(
                                f"[Bi-LSTM] {symbol}: prob={lstm_prob:.1%}, "
                                f"acc={acc_str}, vote={vote_str}"
                            )
                    except Exception as e:
                        logger.warning(f"[Bi-LSTM] Erro na predição: {e}")

                # VOTO ML: integrado na confluência (pode ser -2, -1, 0, +1)
                # Votos: 1.RSI 2.MACD 3.EMA/Trend 4.ADX 5.BB 6.Orderbook
                #        7.MTF 8.CVD 9.SetupValidator 10.ML 11.LSTM + LLM(peso variável)
                ml_vote = agno_signal.get("ml_vote", 0)
                if ml_vote > 0:
                    votes_for += ml_vote
                    confluence["details"].append(f"ML a favor +{ml_vote} (prob={ml_prob:.1%})")
                elif ml_vote < 0:
                    votes_against += abs(ml_vote)
                    confluence["details"].append(f"ML contra {ml_vote} (prob={ml_prob:.1%})")
                else:
                    confluence["details"].append(f"ML neutro (prob={ml_prob:.1%})")

                # LSTM vote: pode ser -2 (forte contra), -1 (contra), 0, +1 (a favor)
                if lstm_vote > 0:
                    votes_for += lstm_vote
                elif lstm_vote < 0:
                    votes_against += abs(lstm_vote)

                # ========================================
                # REGIME DE MERCADO: voto extra na confluência
                # SIDEWAYS = voto contra (penaliza sinais em lateral)
                # BULL/BEAR alinhado com direção = voto a favor
                # ========================================
                regime_vote = 0
                is_sideways = False
                if market_regime:
                    base_regime = market_regime.get("base_regime", "SIDEWAYS")
                    vol_regime = market_regime.get("details", {}).get("volatility", {}).get("volatility", "NORMAL")

                    if base_regime == "SIDEWAYS":
                        is_sideways = True

                        # Avaliar qualidade do range para mean reversion
                        from src.analysis.technical_levels_calculator import assess_range_quality
                        range_quality = assess_range_quality(analysis_data)
                        agno_signal["_range_quality"] = range_quality

                        if range_quality["tradeable"]:
                            # Range BOM: mean reversion viável — penalidade reduzida
                            regime_vote = -1 if vol_regime == "SQUEEZE" else 0
                            if regime_vote != 0:
                                votes_against += abs(regime_vote)
                            confluence["details"].append(
                                f"REGIME LATERAL MEAN-REV ({base_regime}_{vol_regime}) — "
                                f"range quality={range_quality['quality_score']:.2f} ({range_quality['reason']})"
                            )
                        else:
                            # Range RUIM: sem mean reversion — penalidade normal/extra
                            regime_vote = -2 if vol_regime == "SQUEEZE" else -1
                            votes_against += abs(regime_vote)
                            confluence["details"].append(
                                f"REGIME LATERAL ({base_regime}_{vol_regime}) — voto contra x{abs(regime_vote)} "
                                f"(range: {range_quality['reason']})"
                            )
                    elif (is_buy and base_regime == "BULL") or (not is_buy and base_regime == "BEAR"):
                        regime_vote = 1
                        votes_for += 1
                        confluence["details"].append(
                            f"REGIME alinhado ({base_regime}) — voto a favor"
                        )
                    elif (is_buy and base_regime == "BEAR") or (not is_buy and base_regime == "BULL"):
                        regime_vote = -1
                        votes_against += 1
                        confluence["details"].append(
                            f"REGIME contra ({base_regime} vs {llm_signal_dir}) — voto contra"
                        )

                    agno_signal["market_regime"] = market_regime.get("regime", "UNKNOWN")
                    agno_signal["market_regime_base"] = base_regime

                total_for = votes_for + llm_vote_weight
                total_against = votes_against
                total_all = total_for + total_against
                combined_score = total_for / max(total_all, 1)

                # Sistema de votos (9 técnicos + ML + LSTM + LLM + Regime):
                # Mínimo 5 votos a favor OU score >= 55%
                # Se score >= 80% (forte concordância), aceitar com 4 votos
                # Em mercado LATERAL: exigir mais votos (mín 6) para filtrar ruído
                MIN_COMBINED_SCORE = 0.55
                if is_sideways:
                    MIN_VOTES_FOR = 6  # Lateral: exigir mais confirmação
                    MIN_COMBINED_SCORE = 0.60  # Lateral: score mínimo mais alto
                else:
                    MIN_VOTES_FOR = 4 if combined_score >= 0.80 else 5

                agno_signal["confluence_score"] = round(combined_score, 3)
                agno_signal["confluence_details"] = confluence["details"]
                agno_signal["confluence_votes_for"] = total_for
                agno_signal["confluence_votes_against"] = total_against
                agno_signal["confluence_thresholds"] = confluence["thresholds_source"]

                # Per-voter tracking: voto individual de cada componente
                # Será usado para calcular accuracy de cada votante
                voter_breakdown = confluence.get("voter_votes", {})
                voter_breakdown["ml"] = ml_vote
                voter_breakdown["lstm"] = lstm_vote
                voter_breakdown["llm"] = 1 if llm_vote_weight >= 1 else 0
                voter_breakdown["regime"] = regime_vote
                voter_breakdown["ml_prob"] = round(ml_prob, 4)
                voter_breakdown["lstm_prob"] = round(lstm_prob, 4)
                agno_signal["voter_votes"] = voter_breakdown

                logger.info(
                    f"[CONFLUENCE] {llm_signal_dir} {symbol}: "
                    f"score={combined_score:.1%} ({total_for:.0f} for / {total_against} against) "
                    f"| LLM_conf={agno_signal.get('confidence', '?')}/10 (peso={llm_vote_weight}) "
                    f"| ML={'A FAVOR' if ml_vote > 0 else 'CONTRA' if ml_vote < 0 else 'NEUTRO'} (prob={ml_prob:.1%}) "
                    f"| LSTM={'prob=' + f'{lstm_prob:.1%}' if lstm_vote or lstm_prob != 0.5 else 'N/A'} "
                    f"| REGIME={market_regime.get('regime', '?') if market_regime else 'N/A'} "
                    f"| thresholds: {confluence['thresholds_source']}"
                )

                # A confluência técnica + LSTM pode bloquear o sinal
                # independentemente da confiança que o DeepSeek atribuiu.
                if total_for < MIN_VOTES_FOR or combined_score < MIN_COMBINED_SCORE:
                    details_str = " | ".join(confluence["details"])
                    motivo_confl = (
                        f"confluência insuficiente: "
                        f"score={combined_score:.1%} (mín {MIN_COMBINED_SCORE:.0%}), "
                        f"votos={total_for:.0f} (mín {MIN_VOTES_FOR}). "
                        f"Votos: [{details_str}]"
                    )
                    logger.warning(f"╔══ [BLOQUEADO] {llm_signal_dir} {symbol} — Confluência: {motivo_confl}")
                    agno_signal["signal"] = "NO_SIGNAL"
                    agno_signal["block_reason"] = f"Confluência: {motivo_confl}"

            # ========================================
            # GARANTIR SL/TP1/TP2 OBRIGATÓRIOS
            # Nenhuma ordem pode ser aberta sem os 3 níveis
            # ========================================
            if agno_signal.get("signal") in ["BUY", "SELL"]:
                # 1. Garantir entry_price
                if not agno_signal.get("entry_price") or agno_signal.get("entry_price", 0) <= 0:
                    try:
                        market_data = await get_market_data(symbol)
                        if market_data and "current_price" in market_data:
                            agno_signal["entry_price"] = market_data["current_price"]
                    except Exception as e:
                        logger.error(f"[AGNO] Erro ao obter preço atual: {e}")

                entry = agno_signal.get("entry_price", 0)
                if entry <= 0:
                    logger.error(f"[AGNO] Sem entry_price para {symbol}. Não é possível executar.")
                    agno_signal["signal"] = "NO_SIGNAL"
                    agno_signal["block_reason"] = "Sem entry_price válido"
                else:
                    direction = agno_signal["signal"]
                    sl = agno_signal.get("stop_loss", 0) or 0
                    tp1 = agno_signal.get("take_profit_1", 0) or 0
                    tp2 = agno_signal.get("take_profit_2", 0) or 0
                    op_type = agno_signal.get("operation_type", "DAY_TRADE")

                    # ========================================
                    # REGIME-AWARE: Em mercado lateral → mean reversion ou SCALP
                    # Se range bom → mean reversion com BB targets + timeout 15min
                    # Se range ruim → SCALP normal (filtro extra já penalizou confluência)
                    # ========================================
                    sideways_mean_rev = False
                    if is_sideways and op_type in ("DAY_TRADE", "SWING_TRADE", "POSITION_TRADE"):
                        op_type = "SCALP"
                        agno_signal["operation_type"] = "SCALP"
                        agno_signal["operation_type_override"] = "regime_sideways"

                        range_quality = agno_signal.get("_range_quality", {})
                        if range_quality.get("tradeable"):
                            sideways_mean_rev = True
                            logger.info(
                                f"[REGIME→MEAN_REV] {symbol}: Mercado lateral + range bom "
                                f"(score={range_quality.get('quality_score', 0):.2f}). "
                                f"Forçando SCALP mean reversion — TP nas Bollinger Bands, timeout 15min"
                            )
                        else:
                            logger.info(
                                f"[REGIME→SCALP] {symbol}: Mercado lateral detectado. "
                                f"Forçando SCALP (era {agno_signal.get('operation_type', op_type)}) — alvos mais curtos"
                            )

                    # 2. Calcular SL/TP baseado em NIVEIS TECNICOS REAIS
                    # Em SIDEWAYS com range bom: usar BB mean reversion targets
                    # Caso contrário: usar suporte/resistencia, Fibonacci, BBands, EMAs, POC
                    # SEMPRE recalcular tecnicamente — mesmo se DeepSeek forneceu valores
                    needs_calc = True  # Sempre usar niveis tecnicos

                    # SIDEWAYS MEAN REVERSION: usar Bollinger Bands como targets
                    if sideways_mean_rev and "error" not in analysis_data:
                        try:
                            from src.analysis.technical_levels_calculator import (
                                calculate_sideways_mean_reversion_levels,
                                _collect_all_levels,
                            )
                            range_quality = agno_signal.get("_range_quality", {})
                            bb_data = range_quality.get("bb_data", {})
                            all_levels = _collect_all_levels(entry, analysis_data)
                            mr_levels = calculate_sideways_mean_reversion_levels(
                                entry_price=entry,
                                direction=direction,
                                bb_data=bb_data,
                                atr=all_levels.get("atr", 0),
                            )

                            old_sl, old_tp1, old_tp2 = sl, tp1, tp2
                            sl = mr_levels["stop_loss"]
                            tp1 = mr_levels["take_profit_1"]
                            tp2 = mr_levels["take_profit_2"]
                            agno_signal["stop_loss"] = sl
                            agno_signal["take_profit_1"] = tp1
                            agno_signal["take_profit_2"] = tp2
                            agno_signal["sl_method"] = mr_levels["sl_method"]
                            agno_signal["tp1_method"] = mr_levels["tp1_method"]
                            agno_signal["tp2_method"] = mr_levels["tp2_method"]
                            agno_signal["stop_loss_auto_calculated"] = True
                            agno_signal["tp1_auto_calculated"] = True
                            agno_signal["tp2_auto_calculated"] = True
                            agno_signal["sl_tp_source"] = "sideways_mean_reversion"
                            agno_signal["risk_reward"] = mr_levels.get("risk_reward", 0)
                            agno_signal["sideways_mode"] = "mean_reversion"
                            # Timeout reduzido: 15 minutos em mean reversion lateral
                            agno_signal["max_duration_hours"] = 0.25

                            logger.info(
                                f"[MEAN_REV SL/TP] {direction} @ ${entry:,.4f} | "
                                f"SL=${sl:,.4f} ({mr_levels['sl_method']}) | "
                                f"TP1=${tp1:,.4f} ({mr_levels['tp1_method']}) | "
                                f"TP2=${tp2:,.4f} ({mr_levels['tp2_method']}) | "
                                f"R:R={mr_levels.get('risk_reward', 0):.1f} | "
                                f"BB: [{bb_data.get('lower', 0):,.4f} — "
                                f"{bb_data.get('middle', 0):,.4f} — {bb_data.get('upper', 0):,.4f}]"
                            )
                            needs_calc = False
                        except Exception as e:
                            logger.warning(f"[MEAN_REV] Erro no calculador mean reversion: {e}. Usando tech levels padrão.")
                            needs_calc = True

                    if needs_calc and "error" not in analysis_data:
                        try:
                            from src.analysis.technical_levels_calculator import calculate_technical_sl_tp
                            tech_levels = calculate_technical_sl_tp(
                                entry_price=entry,
                                direction=direction,
                                analysis_data=analysis_data,
                                operation_type=op_type,
                            )
                            if "error" not in tech_levels:
                                # Sempre usar SL/TP técnicos — são baseados em niveis reais
                                old_sl, old_tp1, old_tp2 = sl, tp1, tp2

                                sl = tech_levels["stop_loss"]
                                agno_signal["stop_loss"] = sl
                                agno_signal["sl_method"] = tech_levels["sl_method"]
                                agno_signal["stop_loss_auto_calculated"] = True

                                tp1 = tech_levels["take_profit_1"]
                                agno_signal["take_profit_1"] = tp1
                                agno_signal["tp1_method"] = tech_levels["tp1_method"]
                                agno_signal["tp1_auto_calculated"] = True

                                tp2 = tech_levels["take_profit_2"]
                                agno_signal["take_profit_2"] = tp2
                                agno_signal["tp2_method"] = tech_levels["tp2_method"]
                                agno_signal["tp2_auto_calculated"] = True

                                if old_sl > 0 or old_tp1 > 0:
                                    logger.info(
                                        f"[TECH OVERRIDE] LLM: SL=${old_sl:,.2f} TP1=${old_tp1:,.2f} TP2=${old_tp2:,.2f} → "
                                        f"TECH: SL=${sl:,.2f} ({tech_levels['sl_method']}) "
                                        f"TP1=${tp1:,.2f} ({tech_levels['tp1_method']}) "
                                        f"TP2=${tp2:,.2f} ({tech_levels['tp2_method']})"
                                    )
                                else:
                                    logger.info(
                                        f"[TECH SL/TP] SL=${sl:,.2f} ({tech_levels['sl_method']}) | "
                                        f"TP1=${tp1:,.2f} ({tech_levels['tp1_method']}) | "
                                        f"TP2=${tp2:,.2f} ({tech_levels['tp2_method']}) | "
                                        f"R:R={tech_levels.get('risk_reward', 0):.1f}"
                                    )

                                agno_signal["sl_tp_source"] = "technical_levels"
                                agno_signal["risk_reward"] = tech_levels.get("risk_reward", 0)
                                agno_signal["technical_levels_detail"] = tech_levels.get("levels_used", {})
                            else:
                                logger.warning(f"[TECH SL/TP] Erro no calculador técnico: {tech_levels.get('error')}")
                        except Exception as e:
                            logger.warning(f"[TECH SL/TP] Falha no calculador técnico: {e}")

                    # Fallback ATR caso calculador técnico não tenha funcionado
                    sl = agno_signal.get("stop_loss", 0) or 0
                    tp1 = agno_signal.get("take_profit_1", 0) or 0
                    tp2 = agno_signal.get("take_profit_2", 0) or 0
                    if sl <= 0 or tp1 <= 0 or tp2 <= 0:
                        atr_value = agno_signal.get("atr", 0)
                        if atr_value and atr_value > 0:
                            sl_dist = atr_value * 1.5
                            tp1_dist = atr_value * 3.0
                            tp2_dist = atr_value * 5.0
                        else:
                            sl_dist = entry * 0.015
                            tp1_dist = entry * 0.03
                            tp2_dist = entry * 0.05
                        logger.warning("[FALLBACK SL/TP] Calculador técnico não cobriu todos os níveis, usando ATR/percentual")
                        if sl <= 0:
                            sl = (entry - sl_dist) if direction == "BUY" else (entry + sl_dist)
                            agno_signal["stop_loss"] = sl
                            agno_signal["stop_loss_auto_calculated"] = True
                        if tp1 <= 0:
                            tp1 = (entry + tp1_dist) if direction == "BUY" else (entry - tp1_dist)
                            agno_signal["take_profit_1"] = tp1
                            agno_signal["tp1_auto_calculated"] = True
                        if tp2 <= 0:
                            tp2 = (entry + tp2_dist) if direction == "BUY" else (entry - tp2_dist)
                            agno_signal["take_profit_2"] = tp2
                            agno_signal["tp2_auto_calculated"] = True
                        agno_signal["sl_tp_source"] = agno_signal.get("sl_tp_source", "atr_fallback")

                    # 3. HARD BLOCK: Se mesmo após cálculo, algum valor é inválido, NÃO executar
                    sl = agno_signal.get("stop_loss", 0) or 0
                    tp1 = agno_signal.get("take_profit_1", 0) or 0
                    tp2 = agno_signal.get("take_profit_2", 0) or 0

                    if sl <= 0 or tp1 <= 0 or tp2 <= 0:
                        logger.error(
                            f"[HARD BLOCK] {symbol}: SL=${sl}, TP1=${tp1}, TP2=${tp2} — "
                            f"Todos devem ser > 0. NÃO será executado."
                        )
                        agno_signal["signal"] = "NO_SIGNAL"
                        agno_signal["block_reason"] = f"SL/TP inválidos: SL=${sl}, TP1=${tp1}, TP2=${tp2}"
                    else:
                        logger.info(f"[AGNO] Entry=${entry:.2f}, SL=${sl:.2f}, TP1=${tp1:.2f}, TP2=${tp2:.2f}")

            # Salvar sinal AGNO
            self._save_signal(agno_signal)

            # FILTRO DE SINAIS: Verificar se sinais AGNO estão habilitados
            if not settings.accept_agno_signals and agno_signal.get("signal") in ["BUY", "SELL"]:
                logger.info("[AGNO] Sinais AGNO desabilitados (accept_agno_signals=False). Sinal ignorado.")

            # Quando o sinal é NO_SIGNAL (do modelo ou bloqueado por confluência), não há validadores ML/risco
            if agno_signal.get("signal") == "NO_SIGNAL":
                _reason = agno_signal.get("block_reason") or "Modelo optou por não dar sinal."
                # Logar indicadores técnicos pra saber o contexto do mercado
                _rsi = agno_signal.get("rsi", "?")
                _adx = agno_signal.get("adx", "?")
                _macd = agno_signal.get("macd_histogram", "?")
                _trend = agno_signal.get("trend", "?")
                _bb = agno_signal.get("bb_position", "?")
                logger.info(
                    f"[AGNO] Sem execução: {_reason} | "
                    f"RSI={_rsi}, ADX={_adx}, MACD_hist={_macd}, "
                    f"Trend={_trend}, BB_pos={_bb}"
                )

            if agno_signal.get("signal") in ["BUY", "SELL"]:
                logger.info(f"[AGNO] Validando {agno_signal.get('signal')} {symbol} (confluência + risco -> execução)...")

                # Obter tendência dinâmica para filtro
                try:
                    trend_data = await get_trend(symbol)
                except Exception as e:
                    logger.warning(f"[TREND] Erro ao obter tendência: {e}")
                    trend_data = None

                # Obter capital real da Binance (async) ANTES de chamar validate
                _capital_value = None
                try:
                    from src.exchange.executor import BinanceFuturesExecutor
                    _cap_executor = BinanceFuturesExecutor()
                    _cap_balance = await _cap_executor.get_balance()
                    if "error" not in _cap_balance:
                        _capital_value = _cap_balance.get("available_balance", 0)
                except Exception as e:
                    logger.error(f"[CAPITAL] Falha ao obter saldo da Binance: {e}")

                validation = validate_risk_and_position(agno_signal, symbol, _trend_data=trend_data, _capital=_capital_value)
                if validation.get("can_execute"):
                    _pos_size = validation.get("recommended_position_size", validation.get("position_size", "?"))
                    _leverage = validation.get("implied_leverage", "?")
                    logger.info(
                        f"[ENVIANDO] {agno_signal.get('signal')} {symbol} — "
                        f"Pos=${_pos_size}, Leverage={_leverage}x — verificando exchange..."
                    )
                    position_size = validation.get("recommended_position_size", validation.get("position_size"))

                    # VERIFICAR MODO DE TRADING: paper ou real
                    if settings.trading_mode == "real":
                        # MODO REAL: Executar na Binance Futures
                        from src.exchange.executor import BinanceFuturesExecutor
                        executor = BinanceFuturesExecutor()
                        execution_result = await executor.execute_signal(agno_signal, position_size=None)
                        if execution_result.get("success"):
                            agno_signal["_executed"] = True
                            self._mark_signal_executed(agno_signal, "real", True, execution_result.get("message", ""))
                            _real_leverage = execution_result.get("leverage", _leverage)
                            _real_pos = execution_result.get("position_size", _pos_size)
                            logger.warning(
                                f"╔══ [APROVADO] {agno_signal.get('signal')} {symbol} — "
                                f"Trade REAL executado! Pos=${_real_pos}, Leverage={_real_leverage}x"
                            )
                        else:
                            self._mark_signal_executed(agno_signal, "real", False, execution_result.get("error", ""))
                            logger.warning(
                                f"╔══ [BLOQUEADO] {agno_signal.get('signal')} {symbol} — "
                                f"Exchange rejeitou: {execution_result.get('error', '')}"
                            )
                    else:
                        # MODO PAPER: Simulação
                        execution_result = execute_paper_trade(agno_signal, position_size)
                        if execution_result.get("success"):
                            agno_signal["_executed"] = True
                            self._mark_signal_executed(agno_signal, "paper", True, execution_result.get("message", ""))
                            logger.info(f"[AGNO PAPER] Trade PAPER executado: {execution_result.get('message', '')}")
                        else:
                            self._mark_signal_executed(agno_signal, "paper", False, execution_result.get("error", ""))
                            logger.warning(f"[AGNO PAPER] Falha ao executar trade PAPER: {execution_result.get('error', '')}")
                else:
                    reason = validation.get("reason", "Desconhecido")
                    logger.warning(
                        f"╔══ [BLOQUEADO] {agno_signal.get('signal')} {symbol} — "
                        f"Risco/posição: {reason}"
                    )
                    filepath = agno_signal.get("_signal_file")
                    if filepath and os.path.exists(filepath):
                        try:
                            with open(filepath, "r", encoding="utf-8") as f:
                                saved = json.load(f)
                            saved["non_execution_reason"] = reason
                            with open(filepath, "w", encoding="utf-8") as f:
                                json.dump(saved, f, indent=2, ensure_ascii=False, default=str)
                        except Exception:
                            pass

            # ========================================
            # SHADOW MODE: Gerador Local de Sinais
            # Roda em paralelo com DeepSeek para coletar dados de accuracy.
            # NUNCA executa — apenas salva para comparação.
            # ========================================
            try:
                from src.analysis.local_signal_generator import LocalSignalGenerator
                local_gen = LocalSignalGenerator()
                local_result = local_gen.generate_signal(analysis_data, market_regime)
                local_signal_dir = local_result.get("signal", "NO_SIGNAL")

                if local_signal_dir in ("BUY", "SELL"):
                    # Construir sinal local com mesma estrutura do AGNO
                    local_signal = {
                        "symbol": symbol,
                        "source": "LOCAL_GEN",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "signal": local_signal_dir,
                        "confidence": local_result.get("confidence", 5),
                        "reasoning": local_result.get("reasoning", ""),
                        "entry_price": agno_signal.get("entry_price", 0),
                        "shadow_mode": True,
                        "executed": False,
                        "local_gen_score": local_result.get("weighted_score", 0),
                        "local_gen_components": local_result.get("component_scores", {}),
                    }

                    # Copiar indicadores do agno_signal (já coletados)
                    for key in ("rsi", "adx", "atr", "bb_position", "macd_histogram",
                                "trend", "cvd", "cvd_direction", "orderbook_imbalance",
                                "bullish_tf_count", "bearish_tf_count", "indicators"):
                        if key in agno_signal:
                            local_signal[key] = agno_signal[key]

                    # Rodar confluência técnica (mesma que o AGNO passa)
                    local_confluence = self._calculate_technical_confluence(
                        {**analysis_data, "symbol": symbol}, local_signal_dir
                    )
                    local_is_buy = local_signal_dir == "BUY"

                    # ML vote
                    local_ml = self._validate_with_ml_model(local_signal)
                    local_ml_vote = local_ml.get("ml_vote", 0)
                    local_ml_prob = local_ml.get("probability", 0)

                    # Regime vote
                    local_regime_vote = 0
                    if market_regime:
                        br = market_regime.get("base_regime", "SIDEWAYS")
                        if br == "SIDEWAYS":
                            local_regime_vote = -1
                        elif (local_is_buy and br == "BULL") or (not local_is_buy and br == "BEAR"):
                            local_regime_vote = 1
                        elif (local_is_buy and br == "BEAR") or (not local_is_buy and br == "BULL"):
                            local_regime_vote = -1

                    # Calcular totais
                    l_for = local_confluence["votes_for"] + (1 if local_ml_vote > 0 else 0) + (1 if local_regime_vote > 0 else 0)
                    l_against = local_confluence["votes_against"] + (abs(local_ml_vote) if local_ml_vote < 0 else 0) + (abs(local_regime_vote) if local_regime_vote < 0 else 0)
                    l_total = l_for + l_against
                    l_score = l_for / max(l_total, 1)

                    local_signal["confluence_score"] = round(l_score, 3)
                    local_signal["confluence_details"] = local_confluence["details"]
                    local_signal["confluence_votes_for"] = l_for
                    local_signal["confluence_votes_against"] = l_against

                    # Voter breakdown para tracking de accuracy
                    local_voters = local_confluence.get("voter_votes", {})
                    local_voters["ml"] = local_ml_vote
                    local_voters["lstm"] = 0  # LSTM já calculado acima — reusar se disponível
                    local_voters["llm"] = 0   # Sem LLM no gerador local
                    local_voters["regime"] = local_regime_vote
                    local_voters["ml_prob"] = round(local_ml_prob, 4)
                    local_voters["lstm_prob"] = 0.5
                    local_signal["voter_votes"] = local_voters
                    local_signal["ml_probability"] = local_ml_prob

                    # Calcular SL/TP técnicos (mesma lógica do AGNO)
                    entry = local_signal.get("entry_price", 0)
                    if entry > 0:
                        try:
                            from src.analysis.technical_levels_calculator import calculate_technical_sl_tp
                            op_type = "SCALP" if (market_regime and market_regime.get("base_regime") == "SIDEWAYS") else "DAY_TRADE"
                            tech_levels = calculate_technical_sl_tp(
                                entry_price=entry,
                                direction=local_signal_dir,
                                analysis_data=analysis_data,
                                operation_type=op_type,
                            )
                            if "error" not in tech_levels:
                                local_signal["stop_loss"] = tech_levels["stop_loss"]
                                local_signal["take_profit_1"] = tech_levels["take_profit_1"]
                                local_signal["take_profit_2"] = tech_levels["take_profit_2"]
                                local_signal["risk_reward"] = tech_levels.get("risk_reward", 0)
                                local_signal["operation_type"] = op_type
                        except Exception:
                            pass

                    # Salvar (NUNCA executa)
                    self._save_signal(local_signal)

                    # Comparar com AGNO
                    agno_dir = agno_signal.get("signal", "NO_SIGNAL")
                    match_str = "CONCORDAM" if local_signal_dir == agno_dir else "DIVERGEM"
                    logger.info(
                        f"[LOCAL_GEN] {symbol}: {local_signal_dir} conf={local_result.get('confidence')}/10 "
                        f"score={l_score:.1%} ({l_for} for/{l_against} against) "
                        f"| DeepSeek={agno_dir} → {match_str}"
                    )
                else:
                    logger.info(f"[LOCAL_GEN] {symbol}: NO_SIGNAL (score={local_result.get('weighted_score', 0):+.3f})")
            except Exception as e:
                logger.warning(f"[LOCAL_GEN] Erro no gerador local: {e}")

            # === NOTIFICAÇÃO POR EMAIL: Sinal gerado ===
            if agno_signal.get("signal") in ["BUY", "SELL"]:
                await self._send_signal_notification(agno_signal)

            # Retornar o sinal AGNO como principal (para compatibilidade)
            signal = agno_signal

            # Salvar timestamp da última análise (MEMÓRIA + ARQUIVO)
            analysis_time = datetime.now(timezone.utc)
            AgnoTradingAgent._last_analysis_time[symbol] = analysis_time
            try:
                last_analysis_file = f"signals/agno_{symbol}_last_analysis.json"
                with open(last_analysis_file, "w", encoding='utf-8') as f:
                    json.dump({
                        "symbol": symbol,
                        "timestamp": analysis_time.isoformat(),
                        "signal": signal.get("signal", "NO_SIGNAL"),
                        "confidence": signal.get("confidence", 0)
                    }, f, indent=2)
            except Exception as e:
                logger.warning(f"Erro ao salvar ultima analise: {e}")

            # Imprimir resumo
            self._print_summary(signal)

            return signal

        except Exception as e:
            logger.error(f"[AGNO] Erro na analise de {symbol}: {e}")
            return self._create_error_signal(symbol, str(e))

    def _extract_balanced_json(self, text: str) -> Optional[str]:
        """
        Extrai JSON balanceado corretamente, mesmo com objetos aninhados.
        CORREÇÃO: Resolve problema de regex que para no primeiro }
        """
        # Procurar por ```json ... ```
        json_block_match = re.search(r'```json\s*(\{.*?)\s*```', text, re.DOTALL)
        if json_block_match:
            start_pos = json_block_match.start(1)
            json_start = text.find('{', start_pos)
            if json_start == -1:
                return None
        else:
            # Procurar por { sem o bloco de código
            json_start = text.find('{')
            if json_start == -1:
                return None

        # Balancear chaves para encontrar o final correto do JSON
        count = 0
        in_string = False
        escape_next = False

        for i in range(json_start, len(text)):
            char = text[i]

            if escape_next:
                escape_next = False
                continue

            if char == '\\':
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if not in_string:
                if char == '{':
                    count += 1
                elif char == '}':
                    count -= 1
                    if count == 0:
                        # Encontrou o final balanceado
                        return text[json_start:i+1]

        return None

    async def _process_agent_response(self, response: Any, symbol: str) -> Dict[str, Any]:
        """Processa resposta do agent em formato estruturado (CORRIGIDO: agora é async)"""

        # CORRIGIDO: Extrair conteúdo real do RunOutput do AGNO
        # O AGNO retorna um objeto RunOutput, o conteúdo está em response.content
        response_text = None
        if hasattr(response, 'content'):
            # RunOutput do AGNO - conteúdo direto
            response_text = str(response.content) if response.content else None
            logger.debug(f"[AGNO] Conteúdo extraído de response.content: {response_text[:200] if response_text else 'None'}...")
        elif hasattr(response, 'output'):
            response_text = str(response.output)
        elif hasattr(response, 'messages') and len(response.messages) > 0:
            # Se for uma lista de mensagens, pegar a última
            last_message = response.messages[-1]
            if hasattr(last_message, 'content'):
                response_text = str(last_message.content)
            else:
                response_text = str(last_message)
        elif isinstance(response, dict):
            # Se já for dict, pode ser sinal direto
            if "signal" in response:
                logger.info(f"[SINAL DIRETO] Usando sinal do dict: {response.get('signal', 'N/A')}")
                return {
                    "symbol": symbol,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "signal": response.get("signal", "NO_SIGNAL"),
                    "entry_price": response.get("entry_price"),
                    "stop_loss": response.get("stop_loss"),
                    "take_profit_1": response.get("take_profit_1"),
                    "take_profit_2": response.get("take_profit_2"),
                    "confidence": response.get("confidence", 5),
                    "reasoning": response.get("reasoning", ""),
                    "agent_response": str(response)
                }
            else:
                response_text = str(response)
        else:
            # Fallback: tentar str() mas logar aviso
            response_text = str(response)
            logger.warning(f"[AGNO] Resposta não é RunOutput conhecido, usando str(): {type(response)}")

        # Extrair informações da resposta
        signal = {
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_response": response_text[:500] if response_text else "N/A",  # Limitar tamanho
        }

        if not response_text:
            logger.error("[ERRO] Não foi possível extrair conteúdo da resposta do AGNO")
            return {
                "symbol": symbol,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "signal": "NO_SIGNAL",
                "confidence": 0,
                "reason": "Erro ao extrair resposta do AGNO"
            }

        # MELHORIA: Tentar extrair JSON estruturado primeiro (mais confiável)
        # CORREÇÃO: Usar função para balancear chaves e capturar JSON aninhado corretamente
        json_text = self._extract_balanced_json(response_text)
        if json_text:
            try:
                structured = json.loads(json_text)
                logger.info(f"[JSON ESTRUTURADO] Sinal extraído via JSON: {structured.get('signal', 'N/A')}")
                # Validar campos obrigatórios
                if structured.get("signal") in ["BUY", "SELL", "NO_SIGNAL"]:
                    raw_conf = structured.get("confidence", 5)
                    if isinstance(raw_conf, (int, float)):
                        raw_conf = max(1, min(10, raw_conf))
                    else:
                        raw_conf = 5
                    signal.update({
                        "signal": structured.get("signal", "NO_SIGNAL"),
                        "entry_price": structured.get("entry_price"),
                        "stop_loss": structured.get("stop_loss"),
                        "take_profit_1": structured.get("take_profit_1"),
                        "take_profit_2": structured.get("take_profit_2"),
                        "confidence": raw_conf,
                        "reasoning": structured.get("reasoning", ""),
                        "operation_type": structured.get("operation_type", ""),
                    })
                    # Normalizar campos numéricos: tratar 0 ou negativos como ausentes
                    if signal["signal"] in ["BUY", "SELL"]:
                        if not signal.get("entry_price") or signal.get("entry_price", 0) <= 0:
                            logger.warning("[JSON] Sinal BUY/SELL sem entry_price válido, usando fallbacks")
                            signal["entry_price"] = None
                        if not signal.get("stop_loss") or signal.get("stop_loss", 0) <= 0:
                            signal["stop_loss"] = None
                        if not signal.get("take_profit_1") or signal.get("take_profit_1", 0) <= 0:
                            signal["take_profit_1"] = None
                        if not signal.get("take_profit_2") or signal.get("take_profit_2", 0) <= 0:
                            signal["take_profit_2"] = None
                    # Para NO_SIGNAL, retornamos direto (não deve ter preços)
                    if signal["signal"] == "NO_SIGNAL":
                        return signal
                    # Para BUY/SELL com JSON válido, pular para validação de preços
                    # (não resetar sinal nem rodar regex fallback desnecessário)
                    json_extracted = True
            except json.JSONDecodeError as e:
                logger.warning(f"[JSON] Erro ao decodificar JSON: {e}, usando fallback regex")
                json_extracted = False
        else:
            json_extracted = False

        # Regex fallback: só roda quando JSON NÃO extraiu o sinal
        if not json_extracted:
            signal["signal"] = "NO_SIGNAL"

            # CRÍTICO: Procurar primeiro por "SINAL FINAL" que é o mais importante
            final_signal_patterns = [
                r"SINAL\s+FINAL[:\s]+\*?\*?(BUY|SELL)\*?\*?",
                r"SINAL\s+FINAL[:\s]+(BUY|SELL)",
                r"###\s*\*\*SINAL\s+FINAL[:\s]+\*\*(BUY|SELL)",
                r"##\s+SINAL\s+FINAL[:\s]+(BUY|SELL)",
                r"RESUMO[^:]*Sinal\s+(BUY|SELL)",
                r"Conclusão[^:]*:\s*(BUY|SELL)",
                r"Recomendação[^:]*:\s*(BUY|SELL)"
            ]

            for pattern in final_signal_patterns:
                matches = list(re.finditer(pattern, response_text, re.IGNORECASE | re.MULTILINE))
                if matches:
                    last_match = matches[-1]
                    signal_type = last_match.group(1).upper()
                    if signal_type in ["BUY", "SELL"]:
                        signal["signal"] = signal_type
                        logger.info(f"[SINAL EXTRAIDO] Encontrado '{signal_type}' via padrão: {pattern[:50]}")
                        break

            # Se não encontrou padrão específico, procurar por qualquer BUY/SELL
            if signal["signal"] == "NO_SIGNAL":
                buy_matches = list(re.finditer(r'\bBUY\b', response_text, re.IGNORECASE))
                sell_matches = list(re.finditer(r'\bSELL\b', response_text, re.IGNORECASE))
                last_buy_pos = buy_matches[-1].start() if buy_matches else -1
                last_sell_pos = sell_matches[-1].start() if sell_matches else -1

                if last_buy_pos > last_sell_pos and last_buy_pos >= 0:
                    signal["signal"] = "BUY"
                    logger.warning(f"[SINAL FALLBACK] Usando BUY (última ocorrência na posição {last_buy_pos})")
                elif last_sell_pos > last_buy_pos and last_sell_pos >= 0:
                    signal["signal"] = "SELL"
                    logger.warning(f"[SINAL FALLBACK] Usando SELL (última ocorrência na posição {last_sell_pos})")
                elif last_buy_pos >= 0:
                    signal["signal"] = "BUY"
                    logger.warning(f"[SINAL FALLBACK] Usando BUY (última ocorrência na posição {last_buy_pos})")
                elif last_sell_pos >= 0:
                    signal["signal"] = "SELL"
                    logger.warning(f"[SINAL FALLBACK] Usando SELL (última ocorrência na posição {last_sell_pos})")

        # Para NO_SIGNAL, não deve ter entrada, stop ou targets
        if signal["signal"] == "NO_SIGNAL":
            # NO_SIGNAL = não executar
            signal["entry_price"] = None
            signal["stop_loss"] = None
            signal["take_profit_1"] = None
            signal["take_profit_2"] = None
        else:
            # VALIDAÇÃO FINAL: Garantir que entry_price e stop_loss existem antes de retornar
            if not signal.get("entry_price") or not signal.get("stop_loss"):
                logger.warning(f"[SEM PREÇO] {symbol}: sem entry_price ou stop_loss extraíveis da resposta")
                # Tentar extrair preço do texto novamente com padrões mais flexíveis
                if not signal.get("entry_price") and response_text:
                    # Procurar por qualquer número que pareça um preço (mais flexível)
                    price_patterns = [
                        r"\$([0-9,]+\.?[0-9]+)",  # $90,563.50
                        r"([0-9]{1,3}(?:[,.][0-9]{1,2})?)\s*(?:USD|USDT)",  # 90,563.50 USD
                        r"preço[^0-9]*([0-9,]+\.?[0-9]+)",  # preço 90,563.50
                    ]
                    for pattern in price_patterns:
                        match = re.search(pattern, response_text, re.IGNORECASE)
                        if match:
                            try:
                                price_str = match.group(1).replace(",", "")
                                price = float(price_str)
                                # Validar se é um preço razoável
                                if 0.01 <= price <= 1000000:
                                    signal["entry_price"] = price
                                    logger.warning(f"[FALLBACK] Preço extraído do texto: ${price}")
                                    break
                            except ValueError:
                                continue

                # Sem entry_price → descartar sinal (NÃO usar preço hardcoded $100)
                if not signal.get("entry_price"):
                    logger.warning(f"[DESCARTADO] {symbol}: sem entry_price real — descartando sinal")
                    signal["signal"] = "NO_SIGNAL"
                    signal["confidence"] = 0
                    signal["block_reason"] = "Sem entry_price real disponível"
                    return signal

                # Calcular stop loss se não tiver
                if not signal.get("stop_loss") and signal.get("entry_price"):
                    if signal["signal"] == "BUY":
                        signal["stop_loss"] = signal["entry_price"] * 0.98
                    else:  # SELL
                        signal["stop_loss"] = signal["entry_price"] * 1.02
                    logger.warning(f"[FALLBACK] Stop loss calculado: ${signal['stop_loss']}")
            # Para BUY/SELL, OBRIGATÓRIO ter entrada, stop e targets
            entry_patterns = [
                r"entrada[^0-9]*\$?([0-9,]+\.?[0-9]*)",
                r"entry[^0-9]*\$?([0-9,]+\.?[0-9]*)",
                r"entry_price[^0-9]*[:=]\s*\$?([0-9,]+\.?[0-9]*)",
                r"preço[^0-9]*\$?([0-9,]+\.?[0-9]*)",
                r"preco[^0-9]*\$?([0-9,]+\.?[0-9]*)",
                r"current[^0-9]*price[^0-9]*\$?([0-9,]+\.?[0-9]*)"
            ]

            # Se já tem entry_price do JSON, não precisa extrair
            if not signal.get("entry_price"):
                signal["entry_price"] = None
                for pattern in entry_patterns:
                    entry_match = re.search(pattern, response_text, re.IGNORECASE)
                    if entry_match:
                        try:
                            price = float(entry_match.group(1).replace(",", ""))
                            # CORRIGIDO: Validar se o preço é realista (suporta todas as moedas)
                            # BTC: 90k+, ETH: 3k+, SOL: 100+, ADA: 0.4+, DOGE: 0.1+, etc.
                            if 0.01 <= price <= 1000000:
                                signal["entry_price"] = price
                                logger.info(f"[PRECO EXTRAIDO] Entry price encontrado via regex: ${price}")
                                break
                        except ValueError:
                            continue

                # FALLBACK: Se não encontrou entry_price, usar valores padrão
                # CORREÇÃO: Não usar asyncio.run() aqui pois já estamos em um event loop
                # O preço será obtido no método analyze() antes de chamar _process_agent_response
                if not signal.get("entry_price"):
                    # Sem entry_price → descartar (NÃO usar preço hardcoded $100)
                    logger.warning(f"[DESCARTADO] {symbol}: sem entry_price real — descartando sinal")
                    signal["signal"] = "NO_SIGNAL"
                    signal["confidence"] = 0
                    signal["block_reason"] = "Sem entry_price real disponível"
                    return signal

            # CORRIGIDO: Stop Loss - melhor extração com validação
            if signal["signal"] == "BUY":
                # Para BUY, stop loss deve ser ABAIXO da entrada
                stop_patterns = [
                    r"stop[^0-9]*loss[^0-9]*\$?([0-9,]+\.?[0-9]*)",
                    r"stop[^0-9]*\$?([0-9,]+\.?[0-9]*)",
                    r"sl[^0-9]*\$?([0-9,]+\.?[0-9]*)"
                ]

                signal["stop_loss"] = None
                for pattern in stop_patterns:
                    stop_match = re.search(pattern, response_text, re.IGNORECASE)
                    if stop_match:
                        try:
                            stop_price = float(stop_match.group(1).replace(",", ""))
                            # CORRIGIDO: Validar stop loss (suporta todas as moedas)
                            # Para BUY, stop loss deve ser menor que entrada
                            if signal["entry_price"] and 0.01 <= stop_price < signal["entry_price"]:
                                signal["stop_loss"] = stop_price
                                logger.info(f"[STOP LOSS EXTRAIDO] Stop loss encontrado via regex: ${stop_price}")
                                break
                        except ValueError:
                            continue

                # Se não encontrou, calcular baseado em 2% abaixo da entrada
                if not signal["stop_loss"] and signal["entry_price"]:
                    signal["stop_loss"] = signal["entry_price"] * 0.98

            elif signal["signal"] == "SELL":
                # Para SELL, stop loss deve ser ACIMA da entrada
                stop_patterns = [
                    r"stop[^0-9]*loss[^0-9]*\$?([0-9,]+\.?[0-9]*)",
                    r"stop[^0-9]*\$?([0-9,]+\.?[0-9]*)",
                    r"sl[^0-9]*\$?([0-9,]+\.?[0-9]*)"
                ]

                signal["stop_loss"] = None
                for pattern in stop_patterns:
                    stop_match = re.search(pattern, response_text, re.IGNORECASE)
                    if stop_match:
                        try:
                            stop_price = float(stop_match.group(1).replace(",", ""))
                            # CORRIGIDO: Validar stop loss (suporta todas as moedas)
                            # Para SELL, stop loss deve ser maior que entrada
                            if signal["entry_price"] and stop_price > signal["entry_price"] and stop_price <= 1000000:
                                signal["stop_loss"] = stop_price
                                logger.info(f"[STOP LOSS EXTRAIDO] Stop loss encontrado via regex: ${stop_price}")
                                break
                        except ValueError:
                            continue

                # Se não encontrou, calcular baseado em 2% acima da entrada
                if not signal["stop_loss"] and signal["entry_price"]:
                    signal["stop_loss"] = signal["entry_price"] * 1.02

            # CORRIGIDO: Take Profit 1 - melhor extração
            if signal["signal"] == "BUY":
                # Para BUY, TP deve ser ACIMA da entrada
                tp1_patterns = [
                    r"take[^0-9]*profit[^0-9]*1[^0-9]*\$?([0-9,]+\.?[0-9]*)",
                    r"tp1[^0-9]*\$?([0-9,]+\.?[0-9]*)",
                    r"alvo[^0-9]*1[^0-9]*\$?([0-9,]+\.?[0-9]*)",
                    r"target[^0-9]*1[^0-9]*\$?([0-9,]+\.?[0-9]*)"
                ]

                signal["take_profit_1"] = None
                for pattern in tp1_patterns:
                    tp1_match = re.search(pattern, response_text, re.IGNORECASE)
                    if tp1_match:
                        try:
                            price = float(tp1_match.group(1).replace(",", ""))
                            if signal["entry_price"] and price > signal["entry_price"] and price <= 1000000:
                                signal["take_profit_1"] = price
                                break
                        except ValueError:
                            continue

                # Se não encontrou, calcular 2% acima
                if not signal["take_profit_1"] and signal["entry_price"]:
                    signal["take_profit_1"] = signal["entry_price"] * 1.02

            elif signal["signal"] == "SELL":
                # Para SELL, TP deve ser ABAIXO da entrada
                tp1_patterns = [
                    r"take[^0-9]*profit[^0-9]*1[^0-9]*\$?([0-9,]+\.?[0-9]*)",
                    r"tp1[^0-9]*\$?([0-9,]+\.?[0-9]*)",
                    r"alvo[^0-9]*1[^0-9]*\$?([0-9,]+\.?[0-9]*)",
                    r"target[^0-9]*1[^0-9]*\$?([0-9,]+\.?[0-9]*)"
                ]

                signal["take_profit_1"] = None
                for pattern in tp1_patterns:
                    tp1_match = re.search(pattern, response_text, re.IGNORECASE)
                    if tp1_match:
                        try:
                            price = float(tp1_match.group(1).replace(",", ""))
                            # CORRIGIDO: price >= 0.01 ao invés de price >= 1000
                            # Isso permite TP para moedas baratas como DOGE, ADA, etc.
                            if signal["entry_price"] and price < signal["entry_price"] and price >= 0.01:
                                signal["take_profit_1"] = price
                                break
                        except ValueError:
                            continue

                # Se não encontrou, calcular 2% abaixo
                if not signal["take_profit_1"] and signal["entry_price"]:
                    signal["take_profit_1"] = signal["entry_price"] * 0.98

            # CORRIGIDO: Take Profit 2 - melhor extração
            if signal["signal"] == "BUY":
                tp2_patterns = [
                    r"take[^0-9]*profit[^0-9]*2[^0-9]*\$?([0-9,]+\.?[0-9]*)",
                    r"tp2[^0-9]*\$?([0-9,]+\.?[0-9]*)",
                    r"alvo[^0-9]*2[^0-9]*\$?([0-9,]+\.?[0-9]*)",
                    r"target[^0-9]*2[^0-9]*\$?([0-9,]+\.?[0-9]*)"
                ]

                signal["take_profit_2"] = None
                for pattern in tp2_patterns:
                    tp2_match = re.search(pattern, response_text, re.IGNORECASE)
                    if tp2_match:
                        try:
                            price = float(tp2_match.group(1).replace(",", ""))
                            if signal["entry_price"] and price > signal["take_profit_1"] and price <= 1000000:
                                signal["take_profit_2"] = price
                                break
                        except ValueError:
                            continue

                # Se não encontrou, calcular 5% acima
                if not signal["take_profit_2"] and signal["entry_price"]:
                    signal["take_profit_2"] = signal["entry_price"] * 1.05

            elif signal["signal"] == "SELL":
                tp2_patterns = [
                    r"take[^0-9]*profit[^0-9]*2[^0-9]*\$?([0-9,]+\.?[0-9]*)",
                    r"tp2[^0-9]*\$?([0-9,]+\.?[0-9]*)",
                    r"alvo[^0-9]*2[^0-9]*\$?([0-9,]+\.?[0-9]*)",
                    r"target[^0-9]*2[^0-9]*\$?([0-9,]+\.?[0-9]*)"
                ]

                signal["take_profit_2"] = None
                for pattern in tp2_patterns:
                    tp2_match = re.search(pattern, response_text, re.IGNORECASE)
                    if tp2_match:
                        try:
                            price = float(tp2_match.group(1).replace(",", ""))
                            # CORRIGIDO: price >= 0.01 ao invés de price >= 1000
                            # Isso permite TP para moedas baratas como DOGE, ADA, etc.
                            if signal["entry_price"] and signal["take_profit_1"] and price < signal["take_profit_1"] and price >= 0.01:
                                signal["take_profit_2"] = price
                                break
                        except ValueError:
                            continue

                # Se não encontrou, calcular 5% abaixo
                if not signal["take_profit_2"] and signal["entry_price"]:
                    signal["take_profit_2"] = signal["entry_price"] * 0.95

        # VALIDAÇÃO FINAL: Garantir que entry_price e stop_loss existem antes de retornar
        # NOTA: Não podemos usar asyncio.run() aqui pois já estamos em um event loop
        # Vamos usar o preço atual que já foi coletado anteriormente ou calcular baseado em valores padrão
        if signal["signal"] in ["BUY", "SELL"]:
            if not signal.get("entry_price") or not signal.get("stop_loss"):
                logger.error(f"[ERRO CRITICO] Sinal {signal['signal']} sem entry_price ou stop_loss definidos!")
                logger.error(f"Entry: {signal.get('entry_price')}, Stop: {signal.get('stop_loss')}")
                logger.error(f"Response preview: {response_text[:500] if response_text else 'N/A'}...")

                # Tentar extrair preço do texto novamente com padrões mais flexíveis
                if not signal.get("entry_price") and response_text:
                    # Procurar por qualquer número que pareça um preço (mais flexível)
                    price_patterns = [
                        r"\$([0-9,]+\.?[0-9]+)",  # $90,563.50
                        r"([0-9]{1,3}(?:[,.][0-9]{1,2})?)\s*(?:USD|USDT)",  # 90,563.50 USD
                        r"preço[^0-9]*([0-9,]+\.?[0-9]+)",  # preço 90,563.50
                    ]
                    for pattern in price_patterns:
                        match = re.search(pattern, response_text, re.IGNORECASE)
                        if match:
                            try:
                                price_str = match.group(1).replace(",", "")
                                # Se o padrão capturou com ponto, manter o ponto; se não, assumir decimal
                                if "." in price_str:
                                    price = float(price_str)
                                else:
                                    # Se não tem ponto, pode ser um número inteiro ou precisamos adicionar ponto decimal
                                    price = float(price_str)
                                # Validar se é um preço razoável
                                if 0.01 <= price <= 1000000:
                                    signal["entry_price"] = price
                                    logger.warning(f"[FALLBACK] Preço extraído do texto: ${price}")
                                    break
                            except ValueError:
                                continue

                # Sem entry_price → descartar sinal (NÃO usar preço hardcoded $100)
                if not signal.get("entry_price"):
                    logger.warning(f"[DESCARTADO] {symbol}: sem entry_price real — descartando sinal")
                    signal["signal"] = "NO_SIGNAL"
                    signal["confidence"] = 0
                    signal["block_reason"] = "Sem entry_price real disponível"
                    return signal

                # Calcular stop loss se não tiver
                if not signal.get("stop_loss") and signal.get("entry_price"):
                    if signal["signal"] == "BUY":
                        signal["stop_loss"] = signal["entry_price"] * 0.98
                    else:  # SELL
                        signal["stop_loss"] = signal["entry_price"] * 1.02
                    logger.warning(f"[FALLBACK] Stop loss calculado: ${signal['stop_loss']}")

        # Extrair confiança - corrigir regex para capturar corretamente
        conf_patterns = [
            r"confiança[^0-9]*([0-9]+)/10",
            r"confiança[^0-9]*([0-9]+)",
            r"confidence[^0-9]*([0-9]+)/10",
            r"confidence[^0-9]*([0-9]+)"
        ]

        signal["confidence"] = 5  # Default
        for pattern in conf_patterns:
            conf_match = re.search(pattern, response_text, re.IGNORECASE)
            if conf_match:
                signal["confidence"] = int(conf_match.group(1))
                break
        signal["confidence"] = max(1, min(10, signal.get("confidence", 5)))

        return signal

    async def _send_signal_notification(self, signal: Dict[str, Any]):
        """Envia notificação por email/todos os canais quando um sinal é gerado."""
        try:
            from src.core.config import settings
            if not settings.email_notifications_enabled:
                return

            from src.services.notification_service import NotificationService

            notify = NotificationService()
            if not notify.channels:
                return

            symbol = signal.get("symbol", "UNKNOWN")
            direction = signal.get("signal", "UNKNOWN")
            confidence = signal.get("confidence", 0)
            entry = signal.get("entry_price", 0)
            sl = signal.get("stop_loss", 0)
            tp1 = signal.get("take_profit_1", signal.get("tp1", 0))
            tp2 = signal.get("take_profit_2", signal.get("tp2", 0))
            source = signal.get("source", "AGNO")
            reasoning = signal.get("reasoning", "N/A")

            title = f"Novo Sinal: {direction} {symbol}"
            message = (
                f"Simbolo: {symbol}\n"
                f"Direcao: {direction}\n"
                f"Fonte: {source}\n"
                f"Confianca: {confidence}/10\n"
                f"Entrada: ${entry:.2f}\n"
                f"Stop Loss: ${sl:.2f}\n"
                f"Alvo 1 (TP1): ${tp1:.2f}\n"
                f"Alvo 2 (TP2): ${tp2:.2f}\n"
                f"Motivo: {reasoning[:200]}\n"
                f"Horario: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
            )

            from src.services.notification_service import NotificationPriority
            priority = NotificationPriority.HIGH if confidence >= 8 else NotificationPriority.MEDIUM
            await notify.send(message, title, priority)
            logger.info(f"[NOTIFY] Notificacao enviada: {direction} {symbol}")

        except Exception as e:
            logger.debug(f"[NOTIFY] Erro ao enviar notificacao: {e}")

    def _save_signal(self, signal: Dict[str, Any]):
        """Salva sinal em arquivo JSON com deduplicação."""
        symbol = signal.get('symbol', 'UNKNOWN')
        signal_type = signal.get('signal', 'UNKNOWN')
        source = signal.get('source', 'UNKNOWN')

        # Deduplicação: não salvar se sinal é idêntico ao último para este símbolo+source
        cache_key = f"{symbol}_{source}"
        last = AgnoTradingAgent._last_signal_cache.get(cache_key)
        if last:
            # Comparar campos relevantes (ignorar timestamp)
            same_signal = (
                last.get('signal') == signal_type
                and last.get('entry_price') == signal.get('entry_price')
                and last.get('stop_loss') == signal.get('stop_loss')
                and last.get('tp1', last.get('take_profit_1')) == signal.get('tp1', signal.get('take_profit_1'))
            )
            if same_signal:
                logger.info(f"[DEDUP] Sinal duplicado ignorado: {signal_type} {symbol} ({source})")
                return

        # Salvar no cache
        AgnoTradingAgent._last_signal_cache[cache_key] = signal

        # Campos de rastreabilidade (sinal começa como NÃO executado)
        signal["executed"] = False
        signal["execution_mode"] = None
        signal["execution_result"] = None
        signal["ml_probability"] = signal.get("ml_probability", None)

        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        filename = f"signals/agno_{symbol}_{timestamp}.json"
        signal["_signal_file"] = filename

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(signal, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"[SALVO] Sinal salvo: {filename}")

    def _mark_signal_executed(self, signal: Dict[str, Any], mode: str, success: bool, details: str = ""):
        """
        Atualiza o arquivo do sinal marcando como executado.

        Args:
            signal: O sinal original (deve ter _signal_file)
            mode: 'real' ou 'paper'
            success: Se a execução foi bem-sucedida
            details: Mensagem adicional (erro ou confirmação)
        """
        filepath = signal.get("_signal_file")
        if not filepath or not os.path.exists(filepath):
            # Tentar encontrar pelo padrão
            symbol = signal.get("symbol", "")
            if not filepath:
                logger.warning(f"[TRACK] Sinal sem _signal_file: {symbol}")
                return

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                saved = json.load(f)

            saved["executed"] = success
            saved["execution_mode"] = mode
            saved["execution_result"] = "SUCCESS" if success else f"FAILED: {details}"
            saved["execution_time"] = datetime.now(timezone.utc).isoformat()

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(saved, f, indent=2, ensure_ascii=False, default=str)

            status = "EXECUTADO" if success else "FALHOU"
            logger.info(f"[TRACK] {signal.get('symbol')} {signal.get('signal')} -> {status} ({mode}) | {filepath}")

        except Exception as e:
            logger.warning(f"[TRACK] Erro ao atualizar sinal: {e}")

    def _save_deepseek_response(self, symbol: str, prompt: str, response: Any, analysis_data: Dict[str, Any] = None):
        """
        Salva prompt e resposta do DeepSeek em diretório organizado por data (ano/mês/dia)
        para auditoria e verificação de sinais gerados.

        Args:
            symbol: Símbolo analisado
            prompt: Prompt de texto enviado ao DeepSeek
            response: Resposta recebida do DeepSeek
            analysis_data: JSON de análise enviado (dados sumarizados)
        """
        try:
            now = datetime.now(timezone.utc)
            # Criar diretório: deepseek_logs/YYYY/MM/DD
            log_dir = Path(f"deepseek_logs/{now.year}/{now.month:02d}/{now.day:02d}")
            log_dir.mkdir(parents=True, exist_ok=True)

            # Nome do arquivo: symbol_timestamp.json
            timestamp = now.strftime('%Y%m%d_%H%M%S')
            filename = log_dir / f"{symbol}_{timestamp}.json"

            # Preparar dados para salvar
            response_data = {
                "symbol": symbol,
                "timestamp": now.isoformat(),
                "prompt_sent": prompt,  # Prompt de texto enviado
                "analysis_data_sent": analysis_data if analysis_data else {},  # JSON de análise enviado
                "response_received": str(response),  # Resposta bruta do DeepSeek
                "response_type": type(response).__name__
            }

            # Salvar arquivo
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(response_data, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"[DEEPSEEK LOG] Prompt e resposta salvos: {filename}")

        except Exception as e:
            logger.error(f"Erro ao salvar resposta do DeepSeek: {e}")
            # Não bloquear o fluxo se houver erro ao salvar

    def _print_summary(self, signal: Dict[str, Any]):
        """Loga resumo do sinal"""
        entry = f"${signal['entry_price']:,.2f}" if signal.get('entry_price') else "N/A"
        sl = f"${signal['stop_loss']:,.2f}" if signal.get('stop_loss') else "N/A"
        sig_type = signal.get('signal', 'N/A')
        was_executed = signal.get('_executed', False)
        block_reason = signal.get('block_reason', '')
        if was_executed:
            status = "EXECUTADO"
        elif sig_type == "NO_SIGNAL":
            status = f"SEM SINAL ({block_reason[:50]})" if block_reason else "SEM SINAL"
        else:
            status = f"NAO EXECUTADO ({block_reason[:50]})" if block_reason else "NAO EXECUTADO"
        logger.info(
            f"╚══ [RESULTADO] {sig_type} {signal.get('symbol', '')} | "
            f"{status} | Conf={signal.get('confidence', 0)}/10 | "
            f"Entry={entry} | SL={sl}"
        )

    def _create_error_signal(self, symbol: str, error: str) -> Dict[str, Any]:
        """Cria sinal de erro"""
        return {
            "symbol": symbol,
            "signal": "HOLD",
            "confidence": 0,
            "error": error,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    async def monitor_continuous(self, symbols: List[str], interval: int = 300):
        """
        Monitora múltiplos símbolos continuamente.

        Args:
            symbols: Lista de símbolos
            interval: Intervalo em segundos
        """
        logger.info(f"[MONITOR] Monitoramento continuo de {symbols} | Intervalo: {interval}s")

        while True:
            for symbol in symbols:
                try:
                    await self.analyze(symbol)
                except Exception as e:
                    logger.error(f"[MONITOR] Erro em {symbol}: {e}")

                await asyncio.sleep(10)  # Pausa entre símbolos

            logger.info(f"[MONITOR] Aguardando {interval}s...")
            await asyncio.sleep(interval)
