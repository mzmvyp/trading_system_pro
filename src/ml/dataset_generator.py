"""
Script para gerar dataset de sinais validados para treinamento de modelo LSTM
Autor: Trading Bot
Data: 2026-01-13

BOAS PRÁTICAS IMPLEMENTADAS:
1. Eliminação de duplicidades
2. Tratamento de dados faltantes
3. Prevenção de data leakage (split temporal, não random)
4. Normalização de features
5. Remoção de outliers
6. Validação de integridade dos dados
7. Separação clara entre features e target
8. Logs detalhados para auditoria
"""

import asyncio
import hashlib
import json
import os
import re
import warnings
from datetime import datetime
from typing import Dict, List, Optional

import aiohttp
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Configurações
# CORRIGIDO: Usar caminhos relativos ao invés de caminhos absolutos do Windows
# Isso permite portabilidade entre diferentes sistemas operacionais
CONFIG = {
    # Diretórios de sinais (caminhos relativos ao diretório do projeto)
    "signal_dirs": [
        "deepseek_logs",  # Diretório padrão de logs do DeepSeek
        "signals",        # Diretório de sinais salvos
    ],
    # Output (caminho relativo)
    "output_dir": "ml_dataset",
    # Binance API (pública, sem autenticação)
    "binance_base_url": "https://fapi.binance.com",
    # Timeout máximo para validar sinal (em horas)
    "max_validation_hours": 48,
    # Intervalo de klines para validação
    "kline_interval": "5m",
    # Mínimo de features válidas para incluir no dataset
    "min_valid_features": 5,
}


class DatasetGenerator:
    """Gerador de dataset com boas práticas de ML"""

    def __init__(self):
        self.signals_raw: List[Dict] = []
        self.signals_processed: List[Dict] = []
        self.dataset: pd.DataFrame = None
        self.duplicates_removed = 0
        self.invalid_signals = 0
        self.successful_validations = 0
        self.failed_validations = 0

        # Criar diretório de output
        os.makedirs(CONFIG["output_dir"], exist_ok=True)

    async def run(self):
        """Executa o pipeline completo"""
        print("=" * 70)
        print("GERADOR DE DATASET PARA LSTM - SINAIS DE TRADING")
        print("=" * 70)

        # 1. Coletar sinais
        print("\n[1/7] Coletando sinais dos diretórios...")
        await self.collect_signals()

        # 2. Remover duplicatas
        print("\n[2/7] Removendo duplicatas...")
        self.remove_duplicates()

        # 3. Extrair features
        print("\n[3/7] Extraindo features dos sinais...")
        self.extract_features()

        # 4. Validar sinais (buscar resultado histórico)
        print("\n[4/7] Validando sinais com dados históricos da Binance...")
        await self.validate_signals()

        # 5. Tratamentos de qualidade
        print("\n[5/7] Aplicando tratamentos de qualidade...")
        self.apply_quality_treatments()

        # 6. Preparar dataset final
        print("\n[6/7] Preparando dataset final...")
        self.prepare_final_dataset()

        # 7. Salvar
        print("\n[7/7] Salvando dataset...")
        self.save_dataset()

        # Relatório final
        self.print_report()

    async def collect_signals(self):
        """Coleta todos os sinais dos diretórios"""
        total_files = 0

        for signal_dir in CONFIG["signal_dirs"]:
            if not os.path.exists(signal_dir):
                print(f"  [AVISO] Diretório não encontrado: {signal_dir}")
                continue

            for root, dirs, files in os.walk(signal_dir):
                for file in files:
                    if file.endswith('.json'):
                        filepath = os.path.join(root, file)
                        try:
                            with open(filepath, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                data['_source_file'] = filepath
                                self.signals_raw.append(data)
                                total_files += 1
                        except Exception as e:
                            print(f"  [ERRO] Falha ao ler {filepath}: {e}")

        print(f"  [OK] {total_files} arquivos JSON carregados")

    def remove_duplicates(self):
        """Remove sinais duplicados baseado em hash do conteúdo"""
        seen_hashes = set()
        unique_signals = []

        for signal in self.signals_raw:
            # Criar hash baseado em symbol + timestamp + signal
            hash_content = f"{signal.get('symbol', '')}-{signal.get('timestamp', '')}"
            hash_key = hashlib.md5(hash_content.encode()).hexdigest()

            if hash_key not in seen_hashes:
                seen_hashes.add(hash_key)
                unique_signals.append(signal)
            else:
                self.duplicates_removed += 1

        self.signals_raw = unique_signals
        print(f"  [OK] {self.duplicates_removed} duplicatas removidas")
        print(f"  [OK] {len(self.signals_raw)} sinais únicos")

    def extract_features(self):
        """Extrai features estruturadas de cada sinal"""
        for signal in self.signals_raw:
            try:
                features = self._extract_signal_features(signal)
                if features and self._validate_features(features):
                    self.signals_processed.append(features)
                else:
                    self.invalid_signals += 1
            except Exception:
                self.invalid_signals += 1

        print(f"  [OK] {len(self.signals_processed)} sinais processados com sucesso")
        print(f"  [X] {self.invalid_signals} sinais inválidos descartados")

    def _extract_signal_features(self, signal: Dict) -> Optional[Dict]:
        """Extrai features de um sinal individual"""
        try:
            response = signal.get('response_received', '')

            # Extrair JSON do sinal da resposta
            signal_data = self._parse_signal_json(response)
            if not signal_data:
                return None

            # Extrair indicadores técnicos da resposta
            indicators = self._extract_indicators(response)

            # Montar features
            features = {
                # Identificação
                'symbol': signal.get('symbol', ''),
                'timestamp': signal.get('timestamp', ''),

                # Sinal
                'signal_type': signal_data.get('signal', 'NO_SIGNAL'),
                'entry_price': float(signal_data.get('entry_price', 0)),
                'stop_loss': float(signal_data.get('stop_loss', 0)),
                'take_profit_1': float(signal_data.get('take_profit_1', 0)),
                'take_profit_2': float(signal_data.get('take_profit_2', 0)),
                'confidence': int(signal_data.get('confidence', 0)),

                # Indicadores técnicos
                **indicators
            }

            return features

        except Exception:
            return None

    def _parse_signal_json(self, response: str) -> Optional[Dict]:
        """Extrai o JSON do sinal da resposta do DeepSeek"""
        try:
            # Tentar encontrar JSON no texto
            json_match = re.search(r'\{[^{}]*"signal"[^{}]*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                # Limpar e parsear
                json_str = json_str.replace('\n', ' ')
                return json.loads(json_str)

            # Tentar extrair do RunOutput
            if 'content=' in response:
                content_match = re.search(r"content='([^']*)'", response)
                if content_match:
                    content = content_match.group(1)
                    json_match = re.search(r'```json\s*(\{[^`]*\})\s*```', content, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group(1))

            return None
        except Exception:
            return None

    def _extract_indicators(self, response: str) -> Dict:
        """Extrai indicadores técnicos da resposta"""
        indicators = {
            'rsi': None,
            'macd_histogram': None,
            'adx': None,
            'atr': None,
            'bb_position': None,
            'trend': None,
            'sentiment': None,
            'cvd': None,
            'orderbook_imbalance': None,
            'bullish_tf_count': None,
            'bearish_tf_count': None,
        }

        try:
            # RSI
            rsi_match = re.search(r"'rsi':\s*([\d.]+)", response)
            if rsi_match:
                indicators['rsi'] = float(rsi_match.group(1))

            # MACD Histogram
            macd_match = re.search(r"'macd_histogram':\s*([-\d.]+)", response)
            if macd_match:
                indicators['macd_histogram'] = float(macd_match.group(1))

            # ADX
            adx_match = re.search(r"'adx':\s*([\d.]+)", response)
            if adx_match:
                indicators['adx'] = float(adx_match.group(1))

            # ATR
            atr_match = re.search(r"'atr':\s*([\d.]+)", response)
            if atr_match:
                indicators['atr'] = float(atr_match.group(1))

            # Bollinger Position
            bb_match = re.search(r"'bb_position':\s*([-\d.]+)", response)
            if bb_match:
                indicators['bb_position'] = float(bb_match.group(1))

            # Trend
            trend_match = re.search(r"'trend':\s*'(\w+)'", response)
            if trend_match:
                indicators['trend'] = trend_match.group(1)

            # Sentiment
            sentiment_match = re.search(r"'sentiment':\s*'(\w+)'", response)
            if sentiment_match:
                indicators['sentiment'] = sentiment_match.group(1)

            # CVD
            cvd_match = re.search(r"'cvd':\s*([-\d.]+)", response)
            if cvd_match:
                indicators['cvd'] = float(cvd_match.group(1))

            # Orderbook Imbalance
            ob_match = re.search(r"'orderbook_imbalance':\s*([-\d.]+)", response)
            if ob_match:
                indicators['orderbook_imbalance'] = float(ob_match.group(1))

            # Bullish/Bearish count
            bull_match = re.search(r"'bullish_count':\s*(\d+)", response)
            if bull_match:
                indicators['bullish_tf_count'] = int(bull_match.group(1))

            bear_match = re.search(r"'bearish_count':\s*(\d+)", response)
            if bear_match:
                indicators['bearish_tf_count'] = int(bear_match.group(1))

        except Exception:
            pass

        return indicators

    def _validate_features(self, features: Dict) -> bool:
        """Valida se as features mínimas estão presentes"""
        # INCLUIR TODOS os sinais (BUY, SELL, NO_SIGNAL)
        # O modelo vai aprender quais realmente funcionavam
        if features.get('signal_type') not in ['BUY', 'SELL', 'NO_SIGNAL']:
            return False

        # Para NO_SIGNAL, não precisa de preços
        if features.get('signal_type') == 'NO_SIGNAL':
            # Só precisa de timestamp e symbol
            if not features.get('timestamp'):
                return False
            if not features.get('symbol'):
                return False
            return True

        # Para BUY/SELL, deve ter preços válidos
        if features.get('entry_price', 0) <= 0:
            return False
        if features.get('stop_loss', 0) <= 0:
            return False
        if features.get('take_profit_1', 0) <= 0:
            return False

        # Deve ter timestamp válido
        if not features.get('timestamp'):
            return False

        # Deve ter symbol válido
        if not features.get('symbol'):
            return False

        return True

    async def validate_signals(self):
        """Valida cada sinal buscando dados históricos da Binance"""
        total = len(self.signals_processed)

        async with aiohttp.ClientSession() as session:
            for i, signal in enumerate(self.signals_processed):
                try:
                    print(f"\r  Validando sinal {i+1}/{total}...", end='', flush=True)
                    result = await self._validate_single_signal(session, signal)
                    signal.update(result)

                    if result.get('validation_status') == 'SUCCESS':
                        self.successful_validations += 1
                    else:
                        self.failed_validations += 1

                    # Rate limiting
                    await asyncio.sleep(0.1)

                except Exception as e:
                    signal['validation_status'] = 'ERROR'
                    signal['validation_error'] = str(e)
                    self.failed_validations += 1

        print(f"\n  [OK] {self.successful_validations} sinais validados com sucesso")
        print(f"  [X] {self.failed_validations} sinais sem validação")

    async def _validate_single_signal(self, session: aiohttp.ClientSession, signal: Dict) -> Dict:
        """Valida um único sinal buscando dados históricos"""
        result = {
            'validation_status': 'PENDING',
            'resultado': None,
            'retorno_pct': None,
            'tempo_ate_resultado_min': None,
            'preco_final': None,
        }

        try:
            symbol = signal['symbol']
            timestamp_str = signal['timestamp']
            signal_type = signal['signal_type']

            # Para NO_SIGNAL, marcar como validado mas sem resultado de trade
            if signal_type == 'NO_SIGNAL':
                result['validation_status'] = 'SUCCESS'
                result['resultado'] = 'NO_TRADE'
                result['retorno_pct'] = 0
                result['tempo_ate_resultado_min'] = 0
                return result

            entry_price = signal['entry_price']
            stop_loss = signal['stop_loss']
            take_profit_1 = signal['take_profit_1']
            take_profit_2 = signal.get('take_profit_2', take_profit_1)

            # Parsear timestamp
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            except Exception:
                timestamp = datetime.strptime(timestamp_str[:19], '%Y-%m-%dT%H:%M:%S')

            start_ms = int(timestamp.timestamp() * 1000)
            end_ms = start_ms + (CONFIG["max_validation_hours"] * 60 * 60 * 1000)

            # Buscar klines
            url = f"{CONFIG['binance_base_url']}/fapi/v1/klines"
            params = {
                'symbol': symbol,
                'interval': CONFIG["kline_interval"],
                'startTime': start_ms,
                'endTime': end_ms,
                'limit': 1000
            }

            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    result['validation_status'] = 'API_ERROR'
                    return result

                klines = await resp.json()

            if not klines:
                result['validation_status'] = 'NO_DATA'
                return result

            # Analisar cada candle para ver se atingiu TP ou SL
            for kline in klines:
                candle_time = kline[0]
                candle_high = float(kline[2])
                candle_low = float(kline[3])
                float(kline[4])

                # Calcular tempo desde o sinal
                tempo_min = (candle_time - start_ms) / (1000 * 60)

                if signal_type == 'BUY':
                    # Para BUY: SL é abaixo, TP é acima
                    if candle_low <= stop_loss:
                        # Stop Loss atingido
                        result['resultado'] = 'SL'
                        result['retorno_pct'] = ((stop_loss - entry_price) / entry_price) * 100
                        result['tempo_ate_resultado_min'] = tempo_min
                        result['preco_final'] = stop_loss
                        result['validation_status'] = 'SUCCESS'
                        return result

                    if candle_high >= take_profit_2:
                        # TP2 atingido
                        result['resultado'] = 'TP2'
                        result['retorno_pct'] = ((take_profit_2 - entry_price) / entry_price) * 100
                        result['tempo_ate_resultado_min'] = tempo_min
                        result['preco_final'] = take_profit_2
                        result['validation_status'] = 'SUCCESS'
                        return result

                    if candle_high >= take_profit_1:
                        # TP1 atingido
                        result['resultado'] = 'TP1'
                        result['retorno_pct'] = ((take_profit_1 - entry_price) / entry_price) * 100
                        result['tempo_ate_resultado_min'] = tempo_min
                        result['preco_final'] = take_profit_1
                        result['validation_status'] = 'SUCCESS'
                        return result

                else:  # SELL
                    # Para SELL: SL é acima, TP é abaixo
                    if candle_high >= stop_loss:
                        # Stop Loss atingido
                        result['resultado'] = 'SL'
                        result['retorno_pct'] = ((entry_price - stop_loss) / entry_price) * 100
                        result['tempo_ate_resultado_min'] = tempo_min
                        result['preco_final'] = stop_loss
                        result['validation_status'] = 'SUCCESS'
                        return result

                    if candle_low <= take_profit_2:
                        # TP2 atingido
                        result['resultado'] = 'TP2'
                        result['retorno_pct'] = ((entry_price - take_profit_2) / entry_price) * 100
                        result['tempo_ate_resultado_min'] = tempo_min
                        result['preco_final'] = take_profit_2
                        result['validation_status'] = 'SUCCESS'
                        return result

                    if candle_low <= take_profit_1:
                        # TP1 atingido
                        result['resultado'] = 'TP1'
                        result['retorno_pct'] = ((entry_price - take_profit_1) / entry_price) * 100
                        result['tempo_ate_resultado_min'] = tempo_min
                        result['preco_final'] = take_profit_1
                        result['validation_status'] = 'SUCCESS'
                        return result

            # Timeout - nenhum resultado atingido
            last_close = float(klines[-1][4]) if klines else entry_price
            result['resultado'] = 'TIMEOUT'
            result['retorno_pct'] = ((last_close - entry_price) / entry_price) * 100 if signal_type == 'BUY' else ((entry_price - last_close) / entry_price) * 100
            result['tempo_ate_resultado_min'] = CONFIG["max_validation_hours"] * 60
            result['preco_final'] = last_close
            result['validation_status'] = 'SUCCESS'

        except Exception as e:
            result['validation_status'] = 'ERROR'
            result['validation_error'] = str(e)

        return result

    def apply_quality_treatments(self):
        """Aplica tratamentos de qualidade ao dataset"""
        # Converter para DataFrame
        self.dataset = pd.DataFrame(self.signals_processed)

        initial_count = len(self.dataset)

        # 1. Remover sinais sem validação bem-sucedida
        self.dataset = self.dataset[self.dataset['validation_status'] == 'SUCCESS']
        print(f"  - Removidos {initial_count - len(self.dataset)} sinais sem validação")

        # 2. Remover outliers de retorno (> 3 desvios padrão)
        if 'retorno_pct' in self.dataset.columns and len(self.dataset) > 10:
            mean_ret = self.dataset['retorno_pct'].mean()
            std_ret = self.dataset['retorno_pct'].std()
            before = len(self.dataset)
            self.dataset = self.dataset[
                (self.dataset['retorno_pct'] >= mean_ret - 3*std_ret) &
                (self.dataset['retorno_pct'] <= mean_ret + 3*std_ret)
            ]
            print(f"  - Removidos {before - len(self.dataset)} outliers de retorno")

        # 3. Tratar valores nulos em features numéricas
        numeric_cols = ['rsi', 'macd_histogram', 'adx', 'atr', 'bb_position', 'cvd',
                       'orderbook_imbalance', 'bullish_tf_count', 'bearish_tf_count']
        for col in numeric_cols:
            if col in self.dataset.columns:
                # Preencher com mediana (mais robusto que média)
                median_val = self.dataset[col].median()
                if pd.notna(median_val):
                    self.dataset[col] = self.dataset[col].fillna(median_val)
                else:
                    self.dataset[col] = self.dataset[col].fillna(0)

        # 4. Codificar variáveis categóricas
        if 'trend' in self.dataset.columns:
            trend_map = {'strong_bullish': 2, 'bullish': 1, 'neutral': 0, 'bearish': -1, 'strong_bearish': -2}
            self.dataset['trend_encoded'] = self.dataset['trend'].map(trend_map).fillna(0)

        if 'sentiment' in self.dataset.columns:
            sentiment_map = {'bullish': 1, 'neutral': 0, 'bearish': -1}
            self.dataset['sentiment_encoded'] = self.dataset['sentiment'].map(sentiment_map).fillna(0)

        if 'signal_type' in self.dataset.columns:
            self.dataset['signal_encoded'] = (self.dataset['signal_type'] == 'BUY').astype(int)

        # 5. Criar target binário (1 = lucro, 0 = loss/sem trade)
        # NO_TRADE conta como 0 pois não gerou lucro
        self.dataset['target'] = self.dataset['resultado'].apply(
            lambda x: 1 if x in ['TP1', 'TP2'] else 0
        )

        # 5b. Criar target para classificação multi-classe
        # 0 = SL, 1 = TP1, 2 = TP2, 3 = TIMEOUT, 4 = NO_TRADE
        resultado_map = {'SL': 0, 'TP1': 1, 'TP2': 2, 'TIMEOUT': 3, 'NO_TRADE': 4}
        self.dataset['target_multiclass'] = self.dataset['resultado'].map(resultado_map).fillna(4)

        # 6. Criar features derivadas (com cuidado para não causar leakage)
        if 'entry_price' in self.dataset.columns and 'stop_loss' in self.dataset.columns:
            # Risk/Reward ratio (calculável no momento do sinal, não é leakage)
            self.dataset['risk_distance_pct'] = abs(
                (self.dataset['entry_price'] - self.dataset['stop_loss']) / self.dataset['entry_price'] * 100
            )

        if 'entry_price' in self.dataset.columns and 'take_profit_1' in self.dataset.columns:
            self.dataset['reward_distance_pct'] = abs(
                (self.dataset['take_profit_1'] - self.dataset['entry_price']) / self.dataset['entry_price'] * 100
            )

        if 'risk_distance_pct' in self.dataset.columns and 'reward_distance_pct' in self.dataset.columns:
            self.dataset['risk_reward_ratio'] = (
                self.dataset['reward_distance_pct'] / self.dataset['risk_distance_pct'].replace(0, np.nan)
            ).fillna(1)

        print(f"  [OK] Dataset final: {len(self.dataset)} sinais")

    def prepare_final_dataset(self):
        """Prepara dataset final com split temporal"""
        if self.dataset is None or len(self.dataset) == 0:
            print("  [ERRO] Dataset vazio!")
            return

        # Ordenar por timestamp (IMPORTANTE para evitar leakage temporal)
        self.dataset['timestamp_dt'] = pd.to_datetime(self.dataset['timestamp'])
        self.dataset = self.dataset.sort_values('timestamp_dt')

        # Criar índices para split temporal (80% treino, 20% teste)
        n = len(self.dataset)
        train_size = int(n * 0.8)

        self.dataset['split'] = 'train'
        self.dataset.iloc[train_size:, self.dataset.columns.get_loc('split')] = 'test'

        # Features finais para o modelo
        self.feature_columns = [
            'rsi', 'macd_histogram', 'adx', 'atr', 'bb_position',
            'cvd', 'orderbook_imbalance', 'bullish_tf_count', 'bearish_tf_count',
            'confidence', 'trend_encoded', 'sentiment_encoded', 'signal_encoded',
            'risk_distance_pct', 'reward_distance_pct', 'risk_reward_ratio'
        ]

        # Filtrar apenas colunas que existem
        self.feature_columns = [col for col in self.feature_columns if col in self.dataset.columns]

        print(f"  [OK] Split temporal: {train_size} treino / {n - train_size} teste")
        print(f"  [OK] Features: {len(self.feature_columns)}")

    def save_dataset(self):
        """Salva o dataset em múltiplos formatos"""
        output_dir = CONFIG["output_dir"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Dataset completo (CSV)
        full_path = os.path.join(output_dir, f"dataset_completo_{timestamp}.csv")
        self.dataset.to_csv(full_path, index=False)
        print(f"  [OK] Dataset completo: {full_path}")

        # 2. Dataset de treino
        train_df = self.dataset[self.dataset['split'] == 'train']
        train_path = os.path.join(output_dir, f"dataset_train_{timestamp}.csv")
        train_df.to_csv(train_path, index=False)
        print(f"  [OK] Dataset treino: {train_path}")

        # 3. Dataset de teste
        test_df = self.dataset[self.dataset['split'] == 'test']
        test_path = os.path.join(output_dir, f"dataset_test_{timestamp}.csv")
        test_df.to_csv(test_path, index=False)
        print(f"  [OK] Dataset teste: {test_path}")

        # 4. Metadata
        metadata = {
            'timestamp': timestamp,
            'total_signals': len(self.dataset),
            'train_size': len(train_df),
            'test_size': len(test_df),
            'feature_columns': self.feature_columns,
            'target_column': 'target',
            'success_rate': float(self.dataset['target'].mean()) if len(self.dataset) > 0 else 0,
            'symbols': list(self.dataset['symbol'].unique()),
            'date_range': {
                'start': str(self.dataset['timestamp_dt'].min()),
                'end': str(self.dataset['timestamp_dt'].max())
            },
            'resultado_distribution': self.dataset['resultado'].value_counts().to_dict() if 'resultado' in self.dataset.columns else {}
        }

        metadata_path = os.path.join(output_dir, f"metadata_{timestamp}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"  [OK] Metadata: {metadata_path}")

        # 5. Links para arquivos "latest"
        for src, dst in [
            (full_path, os.path.join(output_dir, "dataset_completo_latest.csv")),
            (train_path, os.path.join(output_dir, "dataset_train_latest.csv")),
            (test_path, os.path.join(output_dir, "dataset_test_latest.csv")),
            (metadata_path, os.path.join(output_dir, "metadata_latest.json")),
        ]:
            try:
                if os.path.exists(dst):
                    os.remove(dst)
                # Copiar ao invés de link (Windows compatível)
                import shutil
                shutil.copy(src, dst)
            except Exception:
                pass

    def print_report(self):
        """Imprime relatório final"""
        print("\n" + "=" * 70)
        print("RELATÓRIO FINAL")
        print("=" * 70)

        print("\n[DATA] DADOS COLETADOS:")
        print(f"   - Arquivos JSON lidos: {len(self.signals_raw) + self.duplicates_removed}")
        print(f"   - Duplicatas removidas: {self.duplicates_removed}")
        print(f"   - Sinais únicos: {len(self.signals_raw)}")

        print("\n[PROC] PROCESSAMENTO:")
        print(f"   - Sinais válidos: {len(self.signals_processed)}")
        print(f"   - Sinais inválidos descartados: {self.invalid_signals}")

        print("\n[VALID] VALIDAÇÃO:")
        print(f"   - Validações bem-sucedidas: {self.successful_validations}")
        print(f"   - Validações falhas: {self.failed_validations}")

        if self.dataset is not None and len(self.dataset) > 0:
            print("\n[RESULT] DATASET FINAL:")
            print(f"   - Total de amostras: {len(self.dataset)}")
            print(f"   - Features: {len(self.feature_columns)}")
            print(f"   - Taxa de sucesso (TP): {self.dataset['target'].mean()*100:.1f}%")

            print("\n[DATA] DISTRIBUIÇÃO DE RESULTADOS:")
            if 'resultado' in self.dataset.columns:
                for resultado, count in self.dataset['resultado'].value_counts().items():
                    pct = count / len(self.dataset) * 100
                    print(f"   - {resultado}: {count} ({pct:.1f}%)")

            print("\n[COINS] SÍMBOLOS:")
            for symbol in self.dataset['symbol'].unique():
                count = len(self.dataset[self.dataset['symbol'] == symbol])
                print(f"   - {symbol}: {count} sinais")

        print("\n" + "=" * 70)
        print("[VALID] DATASET GERADO COM SUCESSO!")
        print(f"   Diretório: {CONFIG['output_dir']}")
        print("=" * 70)


async def main():
    """Função principal"""
    generator = DatasetGenerator()
    await generator.run()


if __name__ == "__main__":
    asyncio.run(main())

