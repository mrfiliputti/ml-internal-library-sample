# Exemplos de Uso da Biblioteca

Esta pasta contém exemplos práticos de como diferentes times podem consumir a biblioteca `car-price-prediction` internamente.

## 📂 Exemplos Disponíveis

### 1. `basic_usage.py` - Time de Vendas
**Caso de uso**: Predição rápida de preços para o time de vendas.

```bash
python examples/basic_usage.py
```

**O que demonstra**:
- ✅ Carregamento simples de dados
- ✅ Treinamento básico de modelo
- ✅ Predição para um carro específico
- ✅ Avaliação de performance

---

### 2. `advanced_usage.py` - Time de Data Science
**Caso de uso**: Análise completa com validações e métricas detalhadas.

```bash
python examples/advanced_usage.py
```

**O que demonstra**:
- ✅ Validação completa de qualidade de dados
- ✅ Análise de importância de features
- ✅ Métricas avançadas (intervalos de confiança)
- ✅ Salvamento e carregamento de modelos
- ✅ Predições em batch

---

### 3. `custom_model.py` - Time de ML Engineering
**Caso de uso**: Experimentação com diferentes modelos e configurações.

```bash
python examples/custom_model.py
```

**O que demonstra**:
- ✅ Uso de modelos customizados (Ridge, Lasso, Random Forest)
- ✅ Comparação de múltiplos modelos
- ✅ Configurações avançadas (com/sem normalização)
- ✅ Pipelines complexos do scikit-learn

---

### 4. `production_api.py` - Time de DevOps/Platform
**Caso de uso**: API REST para servir predições em produção.

**Instalação adicional**:
```bash
pip install flask
```

**Executar**:
```bash
python examples/production_api.py
```

**Testar**:
```bash
# Health check
curl http://localhost:5000/health

# Predição única
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"year": 2022, "mileage": 15000, "engine_size": 2.0, "horsepower": 150, "num_doors": 4}'
```

**O que demonstra**:
- ✅ API REST com Flask
- ✅ Endpoints para predições únicas e em batch
- ✅ Health check para monitoramento
- ✅ Carregamento de modelo na inicialização
- ✅ Tratamento de erros e validação

---

## 🎯 Casos de Uso por Time

| Time | Exemplo | Foco |
|------|---------|------|
| **Vendas** | `basic_usage.py` | Predições rápidas |
| **Data Science** | `advanced_usage.py` | Análise e experimentação |
| **ML Engineering** | `custom_model.py` | Customização e otimização |
| **DevOps/Platform** | `production_api.py` | Deploy e produção |

---

## 💡 Dicas de Uso

### Para começar rápido:
```python
from src import DataIngestion, ModelTrainer, ModelEvaluator

# 1. Carregar dados
ingestion = DataIngestion("data/cars.csv")
data = ingestion.generate_synthetic_data(n_samples=1000)

# 2. Treinar
X_train, X_test, y_train, y_test = ingestion.split_data()
trainer = ModelTrainer()
trainer.fit(X_train, y_train)

# 3. Predizer
predictions = trainer.predict(X_test)
```

### Para customizar modelo:
```python
from sklearn.ensemble import RandomForestRegressor
from src import ModelTrainer

# Use seu próprio modelo
custom_model = RandomForestRegressor(n_estimators=100)
trainer = ModelTrainer(model=custom_model)
trainer.fit(X_train, y_train)
```

### Para salvar modelo:
```python
trainer.save("models/meu_modelo.pkl", "models/meu_scaler.pkl")
```

---

## 📚 Documentação Completa

Para documentação completa da API, consulte:
- [README.md](../README.md) - Visão geral do projeto
- Docstrings nos módulos em [src/](../src/)
- Notebook de demonstração: [notebooks/demo_usage.ipynb](../notebooks/demo_usage.ipynb)

---

## 🤝 Suporte

Dúvidas ou problemas? Entre em contato com o time de ML ou abra uma issue no repositório interno.
