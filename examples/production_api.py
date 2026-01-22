"""
Exemplo de API de Produção - Time de DevOps/Platform

Demonstra como criar uma API REST para servir predições
usando a biblioteca internamente.

Requer: pip install flask

Uso:
    python examples/production_api.py
    
    # Testar:
    curl -X POST http://localhost:5000/predict \
         -H "Content-Type: application/json" \
         -d '{"year": 2022, "mileage": 15000, "engine_size": 2.0, "horsepower": 150, "num_doors": 4}'
"""

import sys
from pathlib import Path

# Adiciona o diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from flask import Flask, request, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("⚠️  Flask não instalado. Execute: pip install flask")

import pandas as pd

from src import DataIngestion, ModelTrainer, load_pickle


# Inicializa a aplicação Flask
app = Flask(__name__)

# Variável global para o modelo (carregado na inicialização)
MODEL = None
SCALER = None


def inicializar_modelo():
    """Inicializa ou treina o modelo na startup."""
    global MODEL, SCALER
    
    print("\n🚀 Inicializando modelo...")
    
    model_path = Path("models/production_model.pkl")
    scaler_path = Path("models/production_scaler.pkl")
    
    # Tenta carregar modelo existente
    if model_path.exists() and scaler_path.exists():
        print("📂 Carregando modelo existente...")
        MODEL = load_pickle(str(model_path))
        SCALER = load_pickle(str(scaler_path))
        print("✓ Modelo carregado com sucesso!")
    else:
        print("🔧 Modelo não encontrado. Treinando novo modelo...")
        
        # Treina novo modelo
        ingestion = DataIngestion("data/cars.csv")
        ingestion.generate_synthetic_data(n_samples=1000, random_state=42)
        X_train, X_test, y_train, y_test = ingestion.split_data(test_size=0.2)
        
        trainer = ModelTrainer(use_scaling=True)
        trainer.fit(X_train, y_train)
        
        # Salva para futuras inicializações
        Path("models").mkdir(exist_ok=True)
        trainer.save(str(model_path), str(scaler_path))
        
        MODEL = trainer.model
        SCALER = trainer.scaler
        
        print("✓ Modelo treinado e salvo!")


@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de health check."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL is not None,
        'version': '1.0.0'
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint para predição de preço de carro.
    
    Request JSON:
    {
        "year": 2022,
        "mileage": 15000,
        "engine_size": 2.0,
        "horsepower": 150,
        "num_doors": 4
    }
    
    Response JSON:
    {
        "predicted_price": 25000.50,
        "confidence_interval": {
            "lower": 23000.00,
            "upper": 27000.00
        }
    }
    """
    try:
        # Valida request
        data = request.get_json()
        
        required_fields = ['year', 'mileage', 'engine_size', 'horsepower', 'num_doors']
        missing_fields = [f for f in required_fields if f not in data]
        
        if missing_fields:
            return jsonify({
                'error': f'Campos obrigatórios ausentes: {missing_fields}'
            }), 400
        
        # Prepara dados para predição
        input_df = pd.DataFrame([{
            'year': data['year'],
            'mileage': data['mileage'],
            'engine_size': data['engine_size'],
            'horsepower': data['horsepower'],
            'num_doors': data['num_doors']
        }])
        
        # Faz predição
        if SCALER is not None:
            X_scaled = SCALER.transform(input_df)
            prediction = MODEL.predict(X_scaled)[0]
        else:
            prediction = MODEL.predict(input_df)[0]
        
        # Calcula intervalo de confiança (±10% como exemplo simples)
        margin = prediction * 0.10
        
        # Retorna resultado
        return jsonify({
            'predicted_price': round(float(prediction), 2),
            'confidence_interval': {
                'lower': round(float(prediction - margin), 2),
                'upper': round(float(prediction + margin), 2)
            },
            'input': data
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Endpoint para predição em batch.
    
    Request JSON:
    {
        "cars": [
            {"year": 2022, "mileage": 15000, "engine_size": 2.0, "horsepower": 150, "num_doors": 4},
            {"year": 2020, "mileage": 30000, "engine_size": 1.6, "horsepower": 120, "num_doors": 4}
        ]
    }
    """
    try:
        data = request.get_json()
        
        if 'cars' not in data or not isinstance(data['cars'], list):
            return jsonify({'error': 'Campo "cars" deve ser uma lista'}), 400
        
        # Prepara dados
        input_df = pd.DataFrame(data['cars'])
        
        # Valida colunas
        required_cols = ['year', 'mileage', 'engine_size', 'horsepower', 'num_doors']
        missing_cols = [c for c in required_cols if c not in input_df.columns]
        
        if missing_cols:
            return jsonify({'error': f'Colunas ausentes: {missing_cols}'}), 400
        
        # Predições
        if SCALER is not None:
            X_scaled = SCALER.transform(input_df)
            predictions = MODEL.predict(X_scaled)
        else:
            predictions = MODEL.predict(input_df)
        
        # Formata resposta
        results = []
        for i, pred in enumerate(predictions):
            results.append({
                'car_index': i,
                'input': data['cars'][i],
                'predicted_price': round(float(pred), 2)
            })
        
        return jsonify({
            'predictions': results,
            'total': len(results)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def main():
    """Inicia o servidor da API."""
    
    if not FLASK_AVAILABLE:
        print("\n❌ Flask não está instalado.")
        print("   Instale com: pip install flask")
        return
    
    print("=" * 70)
    print("API DE PRODUÇÃO - CAR PRICE PREDICTION")
    print("=" * 70)
    
    # Inicializa modelo
    inicializar_modelo()
    
    print("\n" + "=" * 70)
    print("🌐 SERVIDOR INICIADO")
    print("=" * 70)
    print("\n📍 Endpoints disponíveis:")
    print("   • GET  /health           - Health check")
    print("   • POST /predict          - Predição única")
    print("   • POST /predict/batch    - Predição em batch")
    print("\n🔧 Exemplos de uso:")
    print("""
    # Health check
    curl http://localhost:5000/health
    
    # Predição única
    curl -X POST http://localhost:5000/predict \\
         -H "Content-Type: application/json" \\
         -d '{"year": 2022, "mileage": 15000, "engine_size": 2.0, "horsepower": 150, "num_doors": 4}'
    
    # Predição em batch
    curl -X POST http://localhost:5000/predict/batch \\
         -H "Content-Type: application/json" \\
         -d '{"cars": [{"year": 2022, "mileage": 15000, "engine_size": 2.0, "horsepower": 150, "num_doors": 4}]}'
    """)
    print("=" * 70)
    print("\n🚀 Servidor rodando em http://localhost:5000")
    print("   Pressione CTRL+C para parar\n")
    
    # Inicia servidor
    app.run(host='0.0.0.0', port=5000, debug=False)


if __name__ == "__main__":
    main()
