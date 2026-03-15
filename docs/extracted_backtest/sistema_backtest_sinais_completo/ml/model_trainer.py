# -*- coding: utf-8 -*-
"""
Model Trainer - Script para treinar e retreinar modelo XGBoost
Pode ser executado manualmente ou agendado (cron/task scheduler)
"""
import logging
import argparse
from datetime import datetime
from typing import List

from ml.optimized_xgboost_predictor import OptimizedXGBoostPredictor
from config.settings import settings


def setup_logging(log_file: str = "model_training.log"):
    """Configura logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )


def train_model(
    symbols: List[str] = None,
    timeframe: str = "1h",
    lookback: int = 1000,
    prediction_horizon: int = 36,
    n_splits: int = 5
):
    """
    Treina modelo XGBoost
    
    Args:
        symbols: Lista de símbolos (None = usa config)
        timeframe: Timeframe dos dados
        lookback: Candles por símbolo
        prediction_horizon: Períodos à frente (36 = 3h)
        n_splits: Splits para validação
    """
    logger = logging.getLogger(__name__)
    
    if symbols is None:
        symbols = settings.get_analysis_symbols()
    
    logger.info("=" * 70)
    logger.info("🚀 TREINAMENTO DO MODELO XGBOOST")
    logger.info("=" * 70)
    logger.info(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Símbolos: {symbols}")
    logger.info(f"Timeframe: {timeframe}")
    logger.info(f"Lookback: {lookback} candles")
    logger.info(f"Horizonte: {prediction_horizon} períodos ({prediction_horizon * 60 / 60:.1f}h para 1h)")
    logger.info(f"Validação: {n_splits} splits temporais")
    logger.info("=" * 70)
    
    # Inicializa predictor
    predictor = OptimizedXGBoostPredictor()
    
    # Treina
    start_time = datetime.now()
    
    try:
        metrics = predictor.train(
            symbols=symbols,
            timeframe=timeframe,
            lookback=lookback,
            prediction_horizon=prediction_horizon,
            n_splits=n_splits
        )
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        logger.info("=" * 70)
        logger.info("✅ TREINAMENTO CONCLUÍDO")
        logger.info("=" * 70)
        logger.info(f"Tempo: {elapsed:.1f}s")
        
        # Verifica se treinamento teve sucesso
        if 'error' in metrics:
            logger.error(f"❌ Erro no treinamento: {metrics.get('error', 'Unknown')}")
            return False
        
        logger.info(f"Amostras: {metrics.get('n_samples', 0)}")
        logger.info(f"Features: {metrics.get('n_features', 0)}")
        logger.info("")
        logger.info("📊 MÉTRICAS FINAIS:")
        avg_metrics = metrics.get('avg_metrics', {})
        logger.info(f"   Accuracy:  {avg_metrics.get('accuracy', 0):.3f}")
        logger.info(f"   Precision: {avg_metrics.get('precision', 0):.3f}")
        logger.info(f"   Recall:    {avg_metrics.get('recall', 0):.3f}")
        logger.info(f"   F1 Score:  {avg_metrics.get('f1', 0):.3f}")
        logger.info(f"   ROC AUC:   {avg_metrics.get('roc_auc', 0):.3f}")
        logger.info("=" * 70)
        
        # Mostra importância de features
        try:
            feature_importance = predictor.get_feature_importance(top_n=10)
            if not feature_importance.empty:
                logger.info("\n🔝 TOP 10 FEATURES MAIS IMPORTANTES:")
                for idx, row in feature_importance.iterrows():
                    logger.info(f"   {row['feature']:30s}: {row['importance']:.4f}")
        except Exception as e:
            logger.warning(f"⚠️ Não foi possível mostrar feature importance: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro no treinamento: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_predictions(symbols: List[str] = None):
    """Testa predições do modelo treinado"""
    logger = logging.getLogger(__name__)
    
    if symbols is None:
        symbols = settings.get_analysis_symbols()[:5]  # Apenas 5 para teste
    
    logger.info("\n" + "=" * 70)
    logger.info("🧪 TESTE DE PREDIÇÕES")
    logger.info("=" * 70)
    
    predictor = OptimizedXGBoostPredictor()
    
    if predictor.model is None:
        logger.error("❌ Modelo não encontrado. Execute o treinamento primeiro.")
        return False
    
    results = predictor.predict_batch(symbols)
    
    logger.info(f"\n📊 PREDIÇÕES PARA {len(results)} SÍMBOLOS:\n")
    
    for symbol, pred in results.items():
        direction_emoji = "📈" if pred['prediction'] == 'BULLISH' else "📉"
        logger.info(
            f"{direction_emoji} {symbol:8s}: {pred['prediction']:8s} "
            f"(conf: {pred['confidence']:.3f}, prob_up: {pred['probability_up']:.3f})"
        )
    
    return True


def main():
    """Função principal"""
    parser = argparse.ArgumentParser(
        description="Treinador de Modelo XGBoost para Trading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:

  # Treinar modelo com símbolos padrão
  python -m ml.model_trainer --train

  # Treinar com símbolos específicos
  python -m ml.model_trainer --train --symbols BTC ETH BNB

  # Treinar com parâmetros customizados
  python -m ml.model_trainer --train --lookback 2000 --horizon 72

  # Apenas testar predições
  python -m ml.model_trainer --test

  # Treinar e testar
  python -m ml.model_trainer --train --test
        """
    )
    
    parser.add_argument('--train', action='store_true',
                       help='Treina o modelo')
    
    parser.add_argument('--test', action='store_true',
                       help='Testa predições')
    
    parser.add_argument('--symbols', nargs='+',
                       help='Símbolos para treinar (padrão: usa config)')
    
    parser.add_argument('--timeframe', type=str, default='1h',
                       help='Timeframe (padrão: 1h)')
    
    parser.add_argument('--lookback', type=int, default=1000,
                       help='Candles por símbolo (padrão: 1000)')
    
    parser.add_argument('--horizon', type=int, default=36,
                       help='Períodos à frente (padrão: 36 = 3h)')
    
    parser.add_argument('--n-splits', type=int, default=5,
                       help='Splits de validação (padrão: 5)')
    
    parser.add_argument('--log-file', type=str, default='model_training.log',
                       help='Arquivo de log')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_file)
    
    # Executa ações
    if not args.train and not args.test:
        parser.print_help()
        return
    
    if args.train:
        success = train_model(
            symbols=args.symbols,
            timeframe=args.timeframe,
            lookback=args.lookback,
            prediction_horizon=args.horizon,
            n_splits=args.n_splits
        )
        
        if not success:
            return
    
    if args.test:
        test_predictions(symbols=args.symbols)


if __name__ == "__main__":
    main()

