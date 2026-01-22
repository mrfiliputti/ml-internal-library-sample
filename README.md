# Car Price Prediction - Projeto de Machine Learning

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)
![Coverage](https://img.shields.io/badge/coverage-80%25+-success.svg)

Uma biblioteca interna de Machine Learning para predição de preços de carros, demonstrando as melhores práticas de desenvolvimento de bibliotecas ML corporativas.

---

## Quick Start (5 minutos)

```bash
# 1. Instale a biblioteca
pip install car-price-prediction

# 2. Execute o quick start
python quick_start.py
```

**Ou use diretamente no código**:

```python
from car_price_prediction import DataIngestion, ModelTrainer
import pandas as pd

# Carregue dados
ingestion = DataIngestion("data.csv")
data = ingestion.load_data()

# Prepare e treine
X_train, X_test, y_train, y_test = ingestion.split_data()
trainer = ModelTrainer()
trainer.fit(X_train, y_train)

# Predição
novo_carro = pd.DataFrame({'year': [2023], 'mileage': [10000], ...})
preco = trainer.predict(novo_carro)[0]
print(f"Preço: ${preco:,.2f}")
```

---

## Sobre o Projeto

Este projeto foi desenvolvido seguindo as melhores práticas ensinadas nas aulas de desenvolvimento de bibliotecas internas de ML, incluindo:

- ✅ **Modularidade** (Aula 2): Código organizado em funções, classes e pacotes reutilizáveis
- ✅ **Documentação** (Aula 3): Docstrings detalhadas seguindo padrão Google/NumPy
- ✅ **Versionamento** (Aula 4): Semantic Versioning e empacotamento com setuptools
- ✅ **PEP8** (Aula 5): Código seguindo convenções de estilo Python
- ✅ **Testes Automatizados** (Aula 6): Suite completa de testes unitários
- ✅ **Design de API** (Aula 7): Interface consistente inspirada em scikit-learn

## Estrutura do Projeto

```
sample_project_ml/
│
├── data/                      # Dados de entrada
│   └── .gitkeep
│
├── examples/                  # Exemplos de uso por time
│   ├── basic_usage.py         # Time de Vendas
│   ├── advanced_usage.py      # Time de Data Science
│   ├── custom_model.py        # Time de ML Engineering
│   ├── production_api.py      # Time de DevOps
│   └── README.md              # Guia dos exemplos
│
├── notebooks/                 # Jupyter notebooks demonstrativos
│   └── demo_usage.ipynb       # Demo completo com visualizações
│
├── src/                       # Código fonte da biblioteca
│   ├── __init__.py            # API pública
│   ├── data_ingestion.py      # Carregamento e preparação de dados
│   ├── data_validation.py     # Validação de qualidade dos dados
│   ├── model_trainer.py       # Treinamento de modelos
│   ├── model_evaluation.py    # Avaliação e métricas
│   └── utils.py               # Funções auxiliares
│
├── tests/                     # Testes unitários
│   ├── test_data_ingestion.py
│   ├── test_data_validation.py
│   ├── test_model_trainer.py
│   └── test_model_evaluation.py
│
├── .github/workflows/         # CI/CD
│   └── ci.yml                 # GitHub Actions pipeline
│
├── main.py                    # Script principal de exemplo
├── requirements.txt           # Dependências do projeto
├── pyproject.toml            # Configuração de build (moderna)
├── setup.py                  # Configuração de build (compatibilidade)
├── CONTRIBUTING.md           # Guia de contribuição
├── CHANGELOG.md              # Histórico de versões
└── README.md                 # Este arquivo
```

## Instalação e Uso para Times Internos

### Opção 1: Instalação via Repositório Interno (Recomendado)

```bash
# Clone o repositório interno
git clone https://github.com/empresa/car-price-prediction.git
cd car-price-prediction

# Instale a biblioteca
pip install .

# Ou em modo desenvolvimento (para contribuir)
pip install -e .[dev]
```

### Opção 2: Instalação via Index Interno (PyPI Privado)

```bash
# Configure o index interno da empresa
pip install car-price-prediction --index-url https://pypi.empresa.com/simple
```

### Opção 3: Instalação via Wheel File

```bash
# Baixe o .whl do repositório de artefatos
pip install car_price_prediction-1.0.0-py3-none-any.whl
```

---

##  Guia de Uso por Time

### Time de Vendas - Uso Básico

**Caso de uso**: Predição rápida de preços.

```python
from car_price_prediction import DataIngestion, ModelTrainer

# 1. Prepare seus dados
ingestion = DataIngestion("seus_dados.csv")
data = ingestion.load_data()
X_train, X_test, y_train, y_test = ingestion.split_data()

# 2. Treine o modelo
trainer = ModelTrainer()
trainer.fit(X_train, y_train)

# 3. Faça predições
import pandas as pd
novo_carro = pd.DataFrame({
    'year': [2022],
    'mileage': [15000],
    'engine_size': [2.0],
    'horsepower': [150],
    'num_doors': [4]
})
preco = trainer.predict(novo_carro)[0]
print(f"Preço estimado: ${preco:,.2f}")
```

**Executar exemplo completo**:
```bash
python examples/basic_usage.py
```

---

### Time de Data Science - Uso Avançado

**Caso de uso**: Análise completa com validações e métricas.

```python
from car_price_prediction import (
    DataIngestion, 
    DataValidator, 
    ModelTrainer, 
    ModelEvaluator
)

# 1. Carregue e valide dados
ingestion = DataIngestion("data.csv")
data = ingestion.load_data()

validator = DataValidator(data)
validator.validate_all()
print(validator.get_summary())

# 2. Treine e avalie
X_train, X_test, y_train, y_test = ingestion.split_data()
trainer = ModelTrainer(use_scaling=True)
trainer.fit(X_train, y_train)

# 3. Análise detalhada
predictions = trainer.predict(X_test)
evaluator = ModelEvaluator(y_test, predictions)
print(evaluator.get_report())

# 4. Salve o modelo
trainer.save("models/modelo_v1.pkl", "models/scaler_v1.pkl")
```

**Executar exemplo completo**:
```bash
python examples/advanced_usage.py
```

---

### Time de ML Engineering - Customização

**Caso de uso**: Experimentar com diferentes modelos.

```python
from car_price_prediction import DataIngestion, ModelTrainer, ModelEvaluator
from sklearn.ensemble import RandomForestRegressor

# Use seu próprio modelo
custom_model = RandomForestRegressor(n_estimators=100, random_state=42)

trainer = ModelTrainer(model=custom_model, use_scaling=True)
trainer.fit(X_train, y_train)

# A API continua a mesma!
predictions = trainer.predict(X_test)
```

**Executar exemplo de comparação de modelos**:
```bash
python examples/custom_model.py
```

---

### Time de DevOps - API de Produção

**Caso de uso**: Servir predições via API REST.

**Instalação adicional**:
```bash
pip install flask
```

**Executar API**:
```bash
python examples/production_api.py
```

**Testar endpoints**:
```bash
# Health check
curl http://localhost:5000/health

# Predição única
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"year": 2022, "mileage": 15000, "engine_size": 2.0, "horsepower": 150, "num_doors": 4}'

# Predição em batch
curl -X POST http://localhost:5000/predict/batch \
     -H "Content-Type: application/json" \
     -d '{"cars": [{"year": 2022, "mileage": 15000, "engine_size": 2.0, "horsepower": 150, "num_doors": 4}]}'
```

---

## Exemplos Completos

Todos os exemplos estão no diretório [`examples/`](examples/):

| Arquivo | Time | Descrição |
|---------|------|-----------|
| [`basic_usage.py`](examples/basic_usage.py) | Vendas | Uso básico e predições simples |
| [`advanced_usage.py`](examples/advanced_usage.py) | Data Science | Validação, análise e persistência |
| [`custom_model.py`](examples/custom_model.py) | ML Engineering | Customização e comparação de modelos |
| [`production_api.py`](examples/production_api.py) | DevOps/Platform | API REST para produção |

---

### Executando o Script Principal

```bash
python main.py
```

### Executando Notebook de Demonstração

```bash
jupyter notebook notebooks/demo_usage.ipynb
```

## Testes

### Executar Todos os Testes

```bash
python -m pytest tests/ -v
```

### Executar Testes com Cobertura

```bash
python -m pytest tests/ --cov=src --cov-report=html
```

### Executar Teste Específico

```bash
python -m unittest tests.test_data_ingestion
```

## Métricas de Avaliação

A biblioteca calcula automaticamente as seguintes métricas:

- **RMSE** (Root Mean Squared Error): Erro quadrático médio
- **MAE** (Mean Absolute Error): Erro absoluto médio
- **R² Score**: Coeficiente de determinação
- **MAPE** (Mean Absolute Percentage Error): Erro percentual médio

## 🔧 Configuração de CI/CD

O projeto inclui configuração de CI/CD com GitHub Actions que:

- ✅ Executa linting (flake8)
- ✅ Roda todos os testes automaticamente
- ✅ Gera relatório de cobertura
- ✅ Valida em múltiplas versões do Python (3.8, 3.9, 3.10)

## Documentação

Toda a biblioteca segue o padrão de documentação com docstrings detalhadas:

```python
def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'ModelTrainer':
    """
    Treina o modelo com os dados fornecidos.

    Parameters
    ----------
    X_train : pd.DataFrame
        Features de treino.
    y_train : pd.Series
        Target de treino.

    Returns
    -------
    self : ModelTrainer
        Retorna a própria instância (padrão scikit-learn).
    """
```

## Princípios Aplicados

### 1. Modularidade (DRY)
- Código organizado em módulos independentes
- Funções e classes reutilizáveis
- Separação clara de responsabilidades

### 2. API Consistente
- Interface inspirada em scikit-learn
- Métodos `fit()`, `predict()` padronizados
- Nomenclatura intuitiva e coerente

### 3. Qualidade de Código
- PEP8 compliance
- Type hints para melhor IDE support
- Logging estruturado

### 4. Testabilidade
- Cobertura de testes > 80%
- Testes unitários e de integração
- CI/CD automatizado

## Versionamento

Este projeto segue [Semantic Versioning](https://semver.org/):

- **MAJOR**: Mudanças incompatíveis na API
- **MINOR**: Novas funcionalidades (compatível)
- **PATCH**: Correções de bugs

**Versão Atual**: 1.0.0

## Contribuindo

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

**Consulte [CONTRIBUTING.md](CONTRIBUTING.md) para guia detalhado de contribuição.**

---

## Distribuição Interna

### Build do Pacote

```bash
# Instale ferramentas de build
pip install build twine

# Gere os pacotes
python -m build

# Arquivos gerados em dist/:
# - car_price_prediction-1.0.0-py3-none-any.whl
# - car_price_prediction-1.0.0.tar.gz
```

### Publicação no PyPI Interno

```bash
# Configure credenciais do PyPI interno
# ~/.pypirc

# Publique no index interno
twine upload --repository-url https://pypi.empresa.com/simple dist/*
```

### Instalação pelos Times

```bash
# Via index interno
pip install car-price-prediction --index-url https://pypi.empresa.com/simple

# Via arquivo .whl
pip install car_price_prediction-1.0.0-py3-none-any.whl
```

---

## 🔄 CI/CD e Qualidade

### Pipeline Automatizado

O projeto inclui CI/CD configurado (`.github/workflows/ci.yml`) que:

- ✅ Executa em Python 3.8, 3.9, 3.10, 3.11
- ✅ Valida estilo com **flake8** (PEP8)
- ✅ Verifica formatação com **black**
- ✅ Roda **todos os testes** automaticamente
- ✅ Gera relatório de **cobertura**
- ✅ Constrói e valida **pacotes**
- ✅ Verifica **vulnerabilidades de segurança**

### Executar Localmente

```bash
# Testes
pytest tests/ -v --cov=src

# Linting
flake8 src/ tests/

# Formatação
black --check src/ tests/

# Cobertura HTML
pytest tests/ --cov=src --cov-report=html
# Abra htmlcov/index.html
```

---

## 📖 Documentação da API

### Classes Principais

#### `DataIngestion`
```python
from car_price_prediction import DataIngestion

ingestion = DataIngestion("data.csv")
data = ingestion.load_data()
X_train, X_test, y_train, y_test = ingestion.split_data(test_size=0.2)
```

#### `DataValidator`
```python
from car_price_prediction import DataValidator

validator = DataValidator(data)
validator.validate_all()
print(validator.get_summary())
```

#### `ModelTrainer`
```python
from car_price_prediction import ModelTrainer

trainer = ModelTrainer(use_scaling=True)
trainer.fit(X_train, y_train)
predictions = trainer.predict(X_test)
trainer.save("model.pkl", "scaler.pkl")
```

#### `ModelEvaluator`
```python
from car_price_prediction import ModelEvaluator

evaluator = ModelEvaluator(y_true, y_pred)
metrics = evaluator.calculate_metrics()
print(evaluator.get_report())
```

**Para documentação completa, consulte os docstrings nos módulos.**

## Licença

Este projeto está sob a licença MIT.

## Autores

- **Fernando Filiputti** - Desenvolvimento inicial

## Documentação Adicional

- **[INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)** - Guia completo de integração para times
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Como contribuir com o projeto
- **[CHANGELOG.md](CHANGELOG.md)** - Histórico de versões e mudanças
- **[examples/](examples/)** - Exemplos práticos de uso por cenário

## Agradecimentos

- Inspirado nas melhores práticas de empresas como Airbnb, Uber e Nubank
- Baseado nos princípios ensinados no curso de bibliotecas internas de ML
- Comunidade Python e scikit-learn pelo excelente design de API
