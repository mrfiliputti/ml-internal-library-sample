# Guia de Integração para Times Internos

Este guia mostra como diferentes times podem integrar e consumir a biblioteca `car-price-prediction` em seus projetos.

---

## 📋 Índice

1. [Instalação e Setup](#instalação-e-setup)
2. [Integração por Cenário](#integração-por-cenário)
3. [Boas Práticas](#boas-práticas)
4. [Troubleshooting](#troubleshooting)

---

## 🔧 Instalação e Setup

### Método 1: Via PyPI Interno (Recomendado para Produção)

```bash
# Configure o index interno
pip config set global.index-url https://pypi.empresa.com/simple

# Instale a biblioteca
pip install car-price-prediction==1.0.0
```

**Adicione ao `requirements.txt`**:
```txt
car-price-prediction==1.0.0
```

### Método 2: Via Git (Desenvolvimento)

```bash
# Clone e instale em modo edição
git clone https://github.com/empresa/car-price-prediction.git
cd car-price-prediction
pip install -e .
```

### Método 3: Via Wheel File (Air-gapped)

```bash
# Baixe o .whl do artefato interno
pip install car_price_prediction-1.0.0-py3-none-any.whl
```

---

## 🎯 Integração por Cenário

### Cenário 1: Dashboard de Vendas (Streamlit/Dash)

**Time**: Vendas / BI  
**Objetivo**: Interface para predições em tempo real

```python
# app.py
import streamlit as st
from car_price_prediction import ModelTrainer, load_pickle
import pandas as pd

# Carrega modelo pré-treinado
@st.cache_resource
def carregar_modelo():
    model = load_pickle("models/production_model.pkl")
    scaler = load_pickle("models/production_scaler.pkl")
    return model, scaler

model, scaler = carregar_modelo()

# Interface
st.title("🚗 Preditor de Preços de Carros")

year = st.slider("Ano", 2010, 2024, 2022)
mileage = st.number_input("Quilometragem", 0, 200000, 15000)
engine_size = st.number_input("Tamanho do Motor (L)", 1.0, 5.0, 2.0)
horsepower = st.number_input("Potência (HP)", 70, 400, 150)
num_doors = st.selectbox("Portas", [2, 4, 5])

if st.button("Calcular Preço"):
    dados = pd.DataFrame({
        'year': [year],
        'mileage': [mileage],
        'engine_size': [engine_size],
        'horsepower': [horsepower],
        'num_doors': [num_doors]
    })
    
    X_scaled = scaler.transform(dados)
    preco = model.predict(X_scaled)[0]
    
    st.success(f"💰 Preço Estimado: ${preco:,.2f}")
```

**Executar**:
```bash
streamlit run app.py
```

---

### Cenário 2: API REST com FastAPI (Produção)

**Time**: Backend / Platform  
**Objetivo**: Endpoint para microserviços

```python
# api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from car_price_prediction import load_pickle
import pandas as pd

app = FastAPI(title="Car Price Prediction API")

# Modelos Pydantic
class CarInput(BaseModel):
    year: int
    mileage: int
    engine_size: float
    horsepower: int
    num_doors: int

class PredictionOutput(BaseModel):
    predicted_price: float
    confidence_lower: float
    confidence_upper: float

# Carrega modelo na inicialização
@app.on_event("startup")
async def load_model():
    global model, scaler
    model = load_pickle("models/production_model.pkl")
    scaler = load_pickle("models/production_scaler.pkl")

@app.post("/predict", response_model=PredictionOutput)
async def predict(car: CarInput):
    try:
        # Prepara dados
        df = pd.DataFrame([car.dict()])
        X_scaled = scaler.transform(df)
        
        # Predição
        price = float(model.predict(X_scaled)[0])
        
        # Intervalo de confiança (±10%)
        margin = price * 0.10
        
        return PredictionOutput(
            predicted_price=price,
            confidence_lower=price - margin,
            confidence_upper=price + margin
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "1.0.0"}
```

**requirements.txt**:
```txt
fastapi==0.104.1
uvicorn==0.24.0
car-price-prediction==1.0.0
```

**Executar**:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**Docker**:
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

### Cenário 3: Pipeline de Dados (Airflow/Prefect)

**Time**: Data Engineering  
**Objetivo**: ETL automatizado com treinamento periódico

```python
# dags/car_price_training_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from car_price_prediction import DataIngestion, ModelTrainer, ModelEvaluator

def extract_data(**context):
    """Extrai dados do data lake."""
    ingestion = DataIngestion("s3://datalake/cars/latest.csv")
    data = ingestion.load_data()
    context['ti'].xcom_push(key='data', value=data.to_json())

def train_model(**context):
    """Treina modelo com dados extraídos."""
    import pandas as pd
    
    data_json = context['ti'].xcom_pull(key='data', task_ids='extract')
    data = pd.read_json(data_json)
    
    # Prepara dados
    ingestion = DataIngestion("")
    ingestion.data = data
    X_train, X_test, y_train, y_test = ingestion.split_data()
    
    # Treina
    trainer = ModelTrainer()
    trainer.fit(X_train, y_train)
    
    # Avalia
    predictions = trainer.predict(X_test)
    evaluator = ModelEvaluator(y_test.values, predictions)
    metrics = evaluator.calculate_metrics()
    
    # Salva se performance boa
    if metrics['r2'] > 0.85:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trainer.save(f"models/model_{timestamp}.pkl", 
                    f"models/scaler_{timestamp}.pkl")
        return "success"
    else:
        raise ValueError(f"R² muito baixo: {metrics['r2']}")

default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'car_price_training',
    default_args=default_args,
    schedule_interval='@weekly',
    catchup=False
) as dag:
    
    extract = PythonOperator(
        task_id='extract',
        python_callable=extract_data
    )
    
    train = PythonOperator(
        task_id='train',
        python_callable=train_model
    )
    
    extract >> train
```

---

### Cenário 4: Jupyter Notebook / Análise Ad-hoc

**Time**: Data Science  
**Objetivo**: Experimentação e análise

```python
# notebook.ipynb

# Instale se necessário
# !pip install car-price-prediction

from car_price_prediction import *
import pandas as pd
import matplotlib.pyplot as plt

# 1. Carregue seus dados
df = pd.read_csv("meus_dados.csv")

# 2. Valide qualidade
validator = DataValidator(df)
validator.validate_all()
print(validator.get_summary())

# 3. Experimente diferentes modelos
from sklearn.ensemble import RandomForestRegressor

models = {
    'Linear': None,
    'Random Forest': RandomForestRegressor(n_estimators=100)
}

results = {}
for name, model in models.items():
    trainer = ModelTrainer(model=model)
    # ... treine e avalie
    
# 4. Visualize resultados
# ... plots
```

---

### Cenário 5: Batch Processing (Spark/Dask)

**Time**: Big Data  
**Objetivo**: Predições em larga escala

```python
# spark_batch.py
from pyspark.sql import SparkSession
from car_price_prediction import load_pickle
import pandas as pd

spark = SparkSession.builder.appName("CarPriceBatch").getOrCreate()

# Carrega modelo
model = load_pickle("models/production_model.pkl")
scaler = load_pickle("models/production_scaler.pkl")

# Função UDF para predição
def predict_price(year, mileage, engine_size, horsepower, num_doors):
    df = pd.DataFrame({
        'year': [year],
        'mileage': [mileage],
        'engine_size': [engine_size],
        'horsepower': [horsepower],
        'num_doors': [num_doors]
    })
    X_scaled = scaler.transform(df)
    return float(model.predict(X_scaled)[0])

# Registra UDF
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType

predict_udf = udf(predict_price, DoubleType())

# Processa dados em batch
df = spark.read.parquet("s3://data/cars/*.parquet")
df_with_predictions = df.withColumn(
    "predicted_price",
    predict_udf("year", "mileage", "engine_size", "horsepower", "num_doors")
)

# Salva resultados
df_with_predictions.write.parquet("s3://output/predictions/")
```

---

## ✅ Boas Práticas

### 1. Versionamento de Modelos

```python
from datetime import datetime

# Use timestamps nos nomes
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = f"models/model_v1_{timestamp}.pkl"

# Ou versionamento semântico
model_path = "models/model_v1.2.3.pkl"
```

### 2. Logging Estruturado

```python
from car_price_prediction import setup_logger

logger = setup_logger(__name__)

logger.info("Iniciando treinamento", extra={
    'n_samples': len(data),
    'model_type': 'LinearRegression'
})
```

### 3. Tratamento de Erros

```python
try:
    trainer = ModelTrainer()
    trainer.fit(X_train, y_train)
except ValueError as e:
    logger.error(f"Erro no treinamento: {e}")
    # Fallback ou alerta
```

### 4. Monitoramento em Produção

```python
from car_price_prediction import ModelEvaluator

# Calcule métricas periodicamente
evaluator = ModelEvaluator(y_real, y_pred)
metrics = evaluator.calculate_metrics()

# Envie para sistema de monitoramento
send_to_datadog(metrics)
send_to_prometheus(metrics)
```

---

## 🔧 Troubleshooting

### Problema: "ModuleNotFoundError: No module named 'car_price_prediction'"

**Solução**:
```bash
# Verifique instalação
pip list | grep car-price

# Reinstale
pip install --force-reinstall car-price-prediction
```

### Problema: Modelo não carrega

**Solução**:
```python
from pathlib import Path

# Verifique se arquivo existe
model_path = Path("models/model.pkl")
if not model_path.exists():
    print(f"Arquivo não encontrado: {model_path}")
    # Treine novo modelo ou baixe backup
```

### Problema: Performance baixa

**Solução**:
```python
# 1. Use normalização
trainer = ModelTrainer(use_scaling=True)

# 2. Tente outros modelos
from sklearn.ensemble import RandomForestRegressor
trainer = ModelTrainer(model=RandomForestRegressor())

# 3. Valide dados
validator = DataValidator(data)
validator.validate_all()
```

### Problema: Versão incompatível

**Solução**:
```bash
# Pin versões específicas
pip install car-price-prediction==1.0.0 scikit-learn==1.3.0
```

---

## 📞 Suporte

- **Documentação**: Consulte o [README.md](README.md)
- **Exemplos**: Veja a pasta [examples/](examples/)
- **Issues**: Abra issue no repositório interno
- **Slack**: Canal `#ml-library-support`
- **Email**: ml-team@empresa.com

---

## 📝 Changelog

Consulte [CHANGELOG.md](CHANGELOG.md) para ver as mudanças em cada versão.

**Versão Atual**: 1.0.0  
**Última Atualização**: 22/01/2026
