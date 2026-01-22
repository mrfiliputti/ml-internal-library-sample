#!/usr/bin/env python
"""
Quick Start - Demonstração Rápida (5 minutos)

Este script demonstra o uso mais simples possível da biblioteca.
Perfeito para times que querem começar rapidamente.

Uso:
    python quick_start.py
"""

from src import DataIngestion, ModelTrainer, ModelEvaluator
import pandas as pd


def main():
    print("\n" + "🚀 QUICK START - CAR PRICE PREDICTION".center(70, "="))
    print("\nBem-vindo! Vamos treinar um modelo em 5 passos simples.\n")
    
    # ============================================
    # PASSO 1: Dados
    # ============================================
    print("📊 [1/5] Carregando dados...")
    ingestion = DataIngestion("data/cars.csv")
    data = ingestion.generate_synthetic_data(n_samples=500)
    print(f"      ✓ {len(data)} carros carregados\n")
    
    # ============================================
    # PASSO 2: Preparação
    # ============================================
    print("🔧 [2/5] Preparando dados...")
    X_train, X_test, y_train, y_test = ingestion.split_data(test_size=0.2)
    print(f"      ✓ Treino: {len(X_train)} | Teste: {len(X_test)}\n")
    
    # ============================================
    # PASSO 3: Treinamento
    # ============================================
    print("🎓 [3/5] Treinando modelo...")
    trainer = ModelTrainer()
    trainer.fit(X_train, y_train)
    print("      ✓ Modelo treinado!\n")
    
    # ============================================
    # PASSO 4: Avaliação
    # ============================================
    print("📈 [4/5] Avaliando performance...")
    predictions = trainer.predict(X_test)
    evaluator = ModelEvaluator(y_test.values, predictions)
    metrics = evaluator.calculate_metrics()
    
    print(f"      ✓ R² Score: {metrics['r2']:.3f}")
    print(f"      ✓ RMSE: ${metrics['rmse']:,.0f}")
    print(f"      ✓ MAE: ${metrics['mae']:,.0f}\n")
    
    # ============================================
    # PASSO 5: Predição Prática
    # ============================================
    print("💰 [5/5] Fazendo uma predição...")
    
    carro_exemplo = pd.DataFrame({
        'year': [2023],
        'mileage': [10000],
        'engine_size': [2.0],
        'horsepower': [180],
        'num_doors': [4]
    })
    
    preco = trainer.predict(carro_exemplo)[0]
    
    print("\n" + "─" * 70)
    print("      🚗 Carro de Exemplo:")
    print("         • Ano: 2023")
    print("         • KM: 10,000")
    print("         • Motor: 2.0L")
    print("         • HP: 180")
    print(f"\n      💵 Preço Estimado: ${preco:,.2f}")
    print("─" * 70)
    
    print("\n" + "✅ PRONTO! Você já pode usar a biblioteca!".center(70, "="))
    print("\n💡 Próximos passos:")
    print("   1. Veja exemplos avançados em: examples/")
    print("   2. Consulte a documentação completa no README.md")
    print("   3. Execute: python main.py (exemplo completo)")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
