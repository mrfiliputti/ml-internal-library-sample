"""
Exemplo Básico de Uso - Time de Vendas

Este exemplo demonstra como o time de vendas pode usar a biblioteca
para fazer predições rápidas de preços de carros.

Uso:
    python examples/basic_usage.py
"""

import sys
from pathlib import Path

# Adiciona o diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

# Importa a biblioteca interna instalada
# Após instalação via pip: pip install car-price-prediction
from src import DataIngestion, ModelTrainer, ModelEvaluator


def main():
    """Exemplo básico: treinar e fazer predição."""
    
    print("=" * 70)
    print("EXEMPLO BÁSICO - TIME DE VENDAS")
    print("=" * 70)
    
    # PASSO 1: Carregar dados
    print("\n1️⃣ Carregando dados...")
    ingestion = DataIngestion("data/cars.csv")
    
    # Gera dados de exemplo (em produção, use: ingestion.load_data())
    data = ingestion.generate_synthetic_data(n_samples=500)
    print(f"   ✓ {len(data)} carros carregados")
    
    # PASSO 2: Preparar dados
    print("\n2️⃣ Preparando dados para treinamento...")
    X_train, X_test, y_train, y_test = ingestion.split_data(test_size=0.2)
    print(f"   ✓ Treino: {len(X_train)} amostras")
    print(f"   ✓ Teste: {len(X_test)} amostras")
    
    # PASSO 3: Treinar modelo
    print("\n3️⃣ Treinando modelo de predição...")
    trainer = ModelTrainer(use_scaling=True)
    trainer.fit(X_train, y_train)
    print("   ✓ Modelo treinado com sucesso!")
    
    # PASSO 4: Fazer predições
    print("\n4️⃣ Fazendo predições...")
    predictions = trainer.predict(X_test)
    
    # PASSO 5: Avaliar resultados
    print("\n5️⃣ Avaliando resultados...")
    evaluator = ModelEvaluator(y_test.values, predictions)
    metrics = evaluator.calculate_metrics()
    
    print(f"\n   📊 R² Score: {metrics['r2']:.4f}")
    print(f"   📊 RMSE: ${metrics['rmse']:,.2f}")
    print(f"   📊 MAE: ${metrics['mae']:,.2f}")
    
    # PASSO 6: Predição de um carro específico
    print("\n6️⃣ Exemplo: Predição para um carro específico")
    print("-" * 70)
    
    # Dados de um carro novo
    novo_carro = pd.DataFrame({
        'year': [2022],
        'mileage': [15000],
        'engine_size': [2.0],
        'horsepower': [150],
        'num_doors': [4]
    })
    
    preco_predito = trainer.predict(novo_carro)[0]
    
    print("\n   Características do Carro:")
    print("   • Ano: 2022")
    print("   • Quilometragem: 15,000 km")
    print("   • Motor: 2.0L")
    print("   • Potência: 150 HP")
    print("   • Portas: 4")
    print(f"\n   💰 Preço Predito: ${preco_predito:,.2f}")
    
    print("\n" + "=" * 70)
    print("✅ EXEMPLO CONCLUÍDO COM SUCESSO!")
    print("=" * 70)


if __name__ == "__main__":
    main()
