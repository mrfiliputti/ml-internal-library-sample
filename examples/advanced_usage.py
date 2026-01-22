"""
Exemplo Avançado - Time de Data Science

Este exemplo demonstra recursos avançados da biblioteca:
- Validação de dados
- Análise de importância de features
- Salvamento/carregamento de modelos
- Intervalos de confiança

Uso:
    python examples/advanced_usage.py
"""

import sys
from pathlib import Path
import pandas as pd

from src import (
    DataIngestion,
    DataValidator,
    ModelTrainer,
    ModelEvaluator,
    save_pickle,
    load_pickle
)


def validar_qualidade_dados(data: pd.DataFrame) -> dict:
    """Valida qualidade dos dados antes do treinamento."""
    print("\n🔍 Validando qualidade dos dados...")
    
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
    
    # Executa validações
    results = validator.validate_all(expected_types=expected_types)
    
    # Exibe resumo
    print(validator.get_summary())
    
    return results


def analisar_importancia_features(trainer: ModelTrainer, feature_names: list) -> None:
    """Analisa e exibe importância das features."""
    print("\n📊 Análise de Importância das Features:")
    print("-" * 70)
    
    importance = trainer.get_feature_importance()
    
    if importance is not None:
        # Ordena por importância absoluta
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Coeficiente': importance,
            'Importância_Abs': abs(importance)
        }).sort_values('Importância_Abs', ascending=False)
        
        print(feature_importance.to_string(index=False))
        
        # Identifica features mais importantes
        top_feature = feature_importance.iloc[0]
        print(f"\n🏆 Feature mais importante: {top_feature['Feature']}")
        print(f"   Coeficiente: {top_feature['Coeficiente']:.2f}")


def gerar_relatorio_completo(evaluator: ModelEvaluator) -> None:
    """Gera relatório completo de avaliação."""
    print("\n📈 Relatório Completo de Avaliação:")
    print(evaluator.get_report())
    
    # Intervalos de confiança
    intervals = evaluator.calculate_prediction_intervals(confidence=0.95)
    print("\n🎯 Intervalos de Confiança (95%):")
    print(f"   • Calculados para {len(intervals['lower_bound'])} predições")
    print(f"   • Amplitude média: ${(intervals['upper_bound'] - intervals['lower_bound']).mean():,.2f}")


def salvar_modelo_producao(trainer: ModelTrainer, output_dir: str = "models") -> None:
    """Salva modelo para uso em produção."""
    print(f"\n💾 Salvando modelo em produção...")
    
    Path(output_dir).mkdir(exist_ok=True)
    
    model_path = f"{output_dir}/production_model.pkl"
    scaler_path = f"{output_dir}/production_scaler.pkl"
    
    trainer.save(model_path, scaler_path)
    
    print(f"   ✓ Modelo salvo: {model_path}")
    print(f"   ✓ Scaler salvo: {scaler_path}")


def demonstrar_carregamento_modelo(data_teste: pd.DataFrame) -> None:
    """Demonstra como carregar e usar modelo salvo."""
    print("\n📂 Demonstrando carregamento de modelo salvo...")
    
    try:
        # Carrega modelo e scaler
        model = load_pickle("models/production_model.pkl")
        scaler = load_pickle("models/production_scaler.pkl")
        
        print("   ✓ Modelo carregado com sucesso!")
        
        # Faz predição
        X_scaled = scaler.transform(data_teste)
        prediction = model.predict(X_scaled)
        
        print(f"   ✓ Predição realizada: ${prediction[0]:,.2f}")
        
    except FileNotFoundError:
        print("   ⚠️ Modelo não encontrado. Execute o salvamento primeiro.")


def main():
    """Exemplo avançado com todos os recursos."""
    
    print("=" * 70)
    print("EXEMPLO AVANÇADO - TIME DE DATA SCIENCE")
    print("=" * 70)
    
    # ========================================
    # 1. INGESTÃO E VALIDAÇÃO
    # ========================================
    print("\n[ETAPA 1] INGESTÃO E VALIDAÇÃO DE DADOS")
    print("-" * 70)
    
    ingestion = DataIngestion("data/cars.csv")
    data = ingestion.generate_synthetic_data(n_samples=1000, random_state=42)
    
    # Valida qualidade
    validation_results = validar_qualidade_dados(data)
    
    # ========================================
    # 2. PREPARAÇÃO DOS DADOS
    # ========================================
    print("\n[ETAPA 2] PREPARAÇÃO DOS DADOS")
    print("-" * 70)
    
    X_train, X_test, y_train, y_test = ingestion.split_data(
        test_size=0.2,
        random_state=42
    )
    
    print(f"✓ Dados divididos: {len(X_train)} treino, {len(X_test)} teste")
    
    # ========================================
    # 3. TREINAMENTO COM ANÁLISE
    # ========================================
    print("\n[ETAPA 3] TREINAMENTO E ANÁLISE")
    print("-" * 70)
    
    trainer = ModelTrainer(use_scaling=True)
    trainer.fit(X_train, y_train)
    
    print("✓ Modelo treinado")
    
    # Analisa importância das features
    analisar_importancia_features(trainer, X_train.columns.tolist())
    
    # ========================================
    # 4. AVALIAÇÃO DETALHADA
    # ========================================
    print("\n[ETAPA 4] AVALIAÇÃO DETALHADA")
    print("-" * 70)
    
    predictions = trainer.predict(X_test)
    evaluator = ModelEvaluator(y_test.values, predictions)
    
    # Calcula métricas
    metrics = evaluator.calculate_metrics()
    
    # Gera relatório completo
    gerar_relatorio_completo(evaluator)
    
    # Comparação detalhada
    print("\n📋 Comparação Detalhada (primeiras 10 predições):")
    comparison = evaluator.compare_predictions(n_samples=10)
    print(comparison.to_string())
    
    # ========================================
    # 5. PERSISTÊNCIA DO MODELO
    # ========================================
    print("\n[ETAPA 5] PERSISTÊNCIA DO MODELO")
    print("-" * 70)
    
    salvar_modelo_producao(trainer)
    
    # Demonstra carregamento
    demonstrar_carregamento_modelo(X_test.iloc[:1])
    
    # ========================================
    # 6. PREDIÇÃO EM BATCH
    # ========================================
    print("\n[ETAPA 6] PREDIÇÃO EM BATCH")
    print("-" * 70)
    
    # Simula múltiplos carros para avaliar
    carros_novos = pd.DataFrame({
        'year': [2023, 2020, 2018, 2022, 2019],
        'mileage': [5000, 30000, 60000, 10000, 45000],
        'engine_size': [2.0, 1.6, 3.0, 2.5, 1.8],
        'horsepower': [180, 120, 250, 200, 140],
        'num_doors': [4, 4, 2, 4, 4]
    })
    
    precos_preditos = trainer.predict(carros_novos)
    
    print("\n📊 Predições em Batch:")
    resultado_batch = carros_novos.copy()
    resultado_batch['Preço_Predito'] = precos_preditos
    print(resultado_batch.to_string())
    
    print("\n💰 Estatísticas dos Preços Preditos:")
    print(f"   • Média: ${precos_preditos.mean():,.2f}")
    print(f"   • Mínimo: ${precos_preditos.min():,.2f}")
    print(f"   • Máximo: ${precos_preditos.max():,.2f}")
    
    # ========================================
    # CONCLUSÃO
    # ========================================
    print("\n" + "=" * 70)
    print("✅ EXEMPLO AVANÇADO CONCLUÍDO!")
    print("=" * 70)
    print("\n📚 Recursos Demonstrados:")
    print("   ✓ Validação completa de dados")
    print("   ✓ Análise de importância de features")
    print("   ✓ Métricas e intervalos de confiança")
    print("   ✓ Salvamento/carregamento de modelos")
    print("   ✓ Predições em batch")
    print("=" * 70)


if __name__ == "__main__":
    main()
