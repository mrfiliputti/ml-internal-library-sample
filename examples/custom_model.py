"""
Exemplo de Customização - Time de ML Engineering

Demonstra como usar a biblioteca com modelos customizados
e configurações avançadas.

Uso:
    python examples/custom_model.py
"""

import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor

from src import DataIngestion, ModelTrainer, ModelEvaluator


def comparar_modelos(X_train, X_test, y_train, y_test):
    """Compara diferentes modelos usando a mesma API."""
    
    print("\n🔬 COMPARAÇÃO DE MODELOS")
    print("=" * 70)
    
    # Define modelos para comparar
    modelos = {
        'Linear Regression': None,  # Modelo padrão
        'Ridge (alpha=1.0)': Ridge(alpha=1.0),
        'Lasso (alpha=0.1)': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    }
    
    resultados = []
    
    for nome, modelo in modelos.items():
        print(f"\n📊 Treinando: {nome}")
        print("-" * 70)
        
        # Treina usando a API da biblioteca
        trainer = ModelTrainer(model=modelo, use_scaling=True)
        trainer.fit(X_train, y_train)
        
        # Avalia
        predictions = trainer.predict(X_test)
        evaluator = ModelEvaluator(y_test.values, predictions)
        metrics = evaluator.calculate_metrics()
        
        # Armazena resultados
        resultados.append({
            'Modelo': nome,
            'R²': metrics['r2'],
            'RMSE': metrics['rmse'],
            'MAE': metrics['mae'],
            'MAPE': metrics['mape']
        })
        
        print(f"   R² Score: {metrics['r2']:.4f}")
        print(f"   RMSE: ${metrics['rmse']:,.2f}")
        print(f"   MAE: ${metrics['mae']:,.2f}")
    
    # Exibe comparação final
    print("\n" + "=" * 70)
    print("📈 COMPARAÇÃO FINAL")
    print("=" * 70)
    
    df_resultados = pd.DataFrame(resultados)
    df_resultados = df_resultados.sort_values('R²', ascending=False)
    
    print("\n" + df_resultados.to_string(index=False))
    
    # Identifica melhor modelo
    melhor = df_resultados.iloc[0]
    print(f"\n🏆 Melhor Modelo: {melhor['Modelo']}")
    print(f"   R²: {melhor['R²']:.4f}")
    
    return df_resultados


def exemplo_sem_normalizacao(X_train, X_test, y_train, y_test):
    """Demonstra treinamento sem normalização."""
    
    print("\n\n⚙️ EXEMPLO: TREINAMENTO SEM NORMALIZAÇÃO")
    print("=" * 70)
    
    # Treina sem scaling
    trainer = ModelTrainer(use_scaling=False)
    trainer.fit(X_train, y_train)
    
    predictions = trainer.predict(X_test)
    evaluator = ModelEvaluator(y_test.values, predictions)
    metrics = evaluator.calculate_metrics()
    
    print("\n📊 Resultados sem normalização:")
    print(f"   R² Score: {metrics['r2']:.4f}")
    print(f"   RMSE: ${metrics['rmse']:,.2f}")
    
    return metrics


def exemplo_pipeline_completo():
    """Pipeline completo de experimentação."""
    
    print("\n🚀 PIPELINE COMPLETO DE EXPERIMENTAÇÃO")
    print("=" * 70)
    
    # 1. Dados
    print("\n1️⃣ Gerando dados...")
    ingestion = DataIngestion("data/cars.csv")
    ingestion.generate_synthetic_data(n_samples=800, random_state=42)
    X_train, X_test, y_train, y_test = ingestion.split_data(test_size=0.2)
    print(f"   ✓ {len(X_train)} amostras de treino")
    
    # 2. Compara modelos
    print("\n2️⃣ Comparando modelos...")
    resultados = comparar_modelos(X_train, X_test, y_train, y_test)
    
    # 3. Testa sem normalização
    print("\n3️⃣ Testando sem normalização...")
    exemplo_sem_normalizacao(X_train, X_test, y_train, y_test)
    
    # 4. Recomendação final
    print("\n" + "=" * 70)
    print("💡 RECOMENDAÇÕES")
    print("=" * 70)
    
    melhor_modelo = resultados.iloc[0]['Modelo']
    melhor_r2 = resultados.iloc[0]['R²']
    
    print(f"\n✅ Modelo recomendado: {melhor_modelo}")
    print(f"   • R² Score: {melhor_r2:.4f}")
    print("   • Use normalização: Sim (melhora performance)")
    print("   • Adequado para: Produção")


def exemplo_modelo_customizado_avancado():
    """Demonstra uso de modelo totalmente customizado."""
    
    print("\n\n🎯 MODELO CUSTOMIZADO AVANÇADO")
    print("=" * 70)
    
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Ridge
    
    # Cria pipeline customizado
    custom_pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('ridge', Ridge(alpha=1.0))
    ])
    
    print("\n🔧 Pipeline customizado:")
    print("   • Polynomial Features (degree=2)")
    print("   • Ridge Regression (alpha=1.0)")
    
    # Dados
    ingestion = DataIngestion("data/cars.csv")
    ingestion.generate_synthetic_data(n_samples=500, random_state=42)
    X_train, X_test, y_train, y_test = ingestion.split_data(test_size=0.2)
    
    # Treina usando a biblioteca (sem scaling pois o pipeline já tem)
    trainer = ModelTrainer(model=custom_pipeline, use_scaling=False)
    trainer.fit(X_train, y_train)
    
    predictions = trainer.predict(X_test)
    evaluator = ModelEvaluator(y_test.values, predictions)
    metrics = evaluator.calculate_metrics()
    
    print("\n📊 Resultados:")
    print(f"   R² Score: {metrics['r2']:.4f}")
    print(f"   RMSE: ${metrics['rmse']:,.2f}")
    print(f"   MAE: ${metrics['mae']:,.2f}")
    
    print("\n✅ A biblioteca suporta qualquer modelo scikit-learn!")


def main():
    """Executa todos os exemplos de customização."""
    
    print("=" * 70)
    print("EXEMPLO DE CUSTOMIZAÇÃO - TIME DE ML ENGINEERING")
    print("=" * 70)
    
    # Pipeline completo
    exemplo_pipeline_completo()
    
    # Modelo customizado avançado
    exemplo_modelo_customizado_avancado()
    
    print("\n" + "=" * 70)
    print("✅ EXEMPLOS DE CUSTOMIZAÇÃO CONCLUÍDOS!")
    print("=" * 70)
    print("\n🎓 Aprendizados:")
    print("   • A biblioteca aceita qualquer modelo scikit-learn")
    print("   • API consistente (fit/predict) facilita experimentação")
    print("   • Normalização pode ser ativada/desativada conforme necessário")
    print("   • Suporta pipelines complexos do scikit-learn")
    print("=" * 70)


if __name__ == "__main__":
    main()
