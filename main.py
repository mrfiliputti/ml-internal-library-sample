"""
Script Principal - Demonstração da Biblioteca Car Price Prediction

Este script demonstra o uso completo da biblioteca interna de ML,
seguindo as melhores práticas apresentadas nas aulas.
"""

import logging
from pathlib import Path

# Importa os módulos da biblioteca
from src import (
    DataIngestion,
    DataValidator,
    ModelTrainer,
    ModelEvaluator,
    setup_logger
)


def main():
    """
    Função principal que executa o pipeline completo de ML.

    Pipeline:
    1. Geração/Carregamento de dados
    2. Validação de qualidade
    3. Divisão treino/teste
    4. Treinamento do modelo
    5. Avaliação de performance
    """
    # Configura logging
    logger = setup_logger(__name__, level=logging.INFO)
    logger.info("=" * 70)
    logger.info("INICIANDO PIPELINE DE MACHINE LEARNING - CAR PRICE PREDICTION")
    logger.info("=" * 70)

    try:
        # ============================================================
        # ETAPA 1: INGESTÃO DE DADOS
        # ============================================================
        logger.info("\n[ETAPA 1/5] INGESTÃO DE DADOS")
        logger.info("-" * 70)

        data_path = "data/cars.csv"
        ingestion = DataIngestion(data_path)

        # Gera dados sintéticos para demonstração
        # Em produção, usaríamos: ingestion.load_data()
        data = ingestion.generate_synthetic_data(n_samples=1000, random_state=42)

        logger.info(f"Shape dos dados: {data.shape}")
        logger.info(f"Colunas: {list(data.columns)}")
        logger.info(f"\nPrimeiras linhas:\n{data.head()}")

        # ============================================================
        # ETAPA 2: VALIDAÇÃO DOS DADOS
        # ============================================================
        logger.info("\n[ETAPA 2/5] VALIDAÇÃO DE QUALIDADE DOS DADOS")
        logger.info("-" * 70)

        validator = DataValidator(data)

        # Define tipos esperados
        expected_types = {
            'year': 'int',
            'mileage': 'int',
            'engine_size': 'float',
            'horsepower': 'int',
            'num_doors': 'int',
            'price': 'float'
        }

        # Executa todas as validações
        validator.validate_all(expected_types=expected_types)

        # Exibe resumo
        print("\n" + validator.get_summary())

        # ============================================================
        # ETAPA 3: DIVISÃO DOS DADOS
        # ============================================================
        logger.info("\n[ETAPA 3/5] DIVISÃO DOS DADOS EM TREINO E TESTE")
        logger.info("-" * 70)

        X_train, X_test, y_train, y_test = ingestion.split_data(
            test_size=0.2,
            random_state=42
        )

        logger.info(f"Conjunto de treino: {len(X_train)} amostras")
        logger.info(f"Conjunto de teste:  {len(X_test)} amostras")
        logger.info(f"Features: {list(X_train.columns)}")

        # ============================================================
        # ETAPA 4: TREINAMENTO DO MODELO
        # ============================================================
        logger.info("\n[ETAPA 4/5] TREINAMENTO DO MODELO")
        logger.info("-" * 70)

        # Inicializa e treina o modelo
        trainer = ModelTrainer(use_scaling=True)
        trainer.fit(X_train, y_train)

        # Exibe importância das features (coeficientes)
        importance = trainer.get_feature_importance()
        if importance is not None:
            logger.info("\nImportância das Features (Coeficientes):")
            for feature, coef in zip(X_train.columns, importance):
                logger.info(f"  {feature:15s}: {coef:10.2f}")

        # Salva o modelo treinado
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)

        trainer.save(
            model_path="models/car_price_model.pkl",
            scaler_path="models/scaler.pkl"
        )

        # ============================================================
        # ETAPA 5: AVALIAÇÃO DO MODELO
        # ============================================================
        logger.info("\n[ETAPA 5/5] AVALIAÇÃO DO MODELO")
        logger.info("-" * 70)

        # Faz predições
        predictions = trainer.predict(X_test)

        # Cria avaliador
        evaluator = ModelEvaluator(y_test.values, predictions)

        # Calcula métricas
        metrics = evaluator.calculate_metrics()

        # Exibe relatório completo
        print("\n" + evaluator.get_report())

        # Exibe comparação de algumas predições
        logger.info("\nComparação de Predições (primeiras 10 amostras):")
        comparison = evaluator.compare_predictions(n_samples=10)
        print("\n" + comparison.to_string())

        # Calcula intervalos de confiança
        evaluator.calculate_prediction_intervals(confidence=0.95)
        logger.info("\nIntervalos de Confiança (95%) calculados")

        # ============================================================
        # CONCLUSÃO
        # ============================================================
        logger.info("\n" + "=" * 70)
        logger.info("PIPELINE CONCLUÍDO COM SUCESSO!")
        logger.info("=" * 70)
        logger.info("\nResultados Principais:")
        logger.info(f"  • R² Score: {metrics['r2']:.4f}")
        logger.info(f"  • RMSE:     ${metrics['rmse']:,.2f}")
        logger.info(f"  • MAE:      ${metrics['mae']:,.2f}")
        logger.info(f"  • MAPE:     {metrics['mape']:.2f}%")
        logger.info("\nModelo salvo em: models/car_price_model.pkl")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"\nERRO durante a execução: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
