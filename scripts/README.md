# Scripts de Automação

Scripts úteis para gerenciamento e deploy da biblioteca.

## 📜 Scripts Disponíveis

### 1. Build e Testes

```bash
# Windows
.\scripts\test.ps1

# Linux/Mac
./scripts/test.sh
```

### 2. Build do Pacote

```bash
# Windows
.\scripts\build.ps1

# Linux/Mac
./scripts/build.sh
```

### 3. Deploy para PyPI Interno

```bash
# Windows
.\scripts\deploy.ps1

# Linux/Mac
./scripts/deploy.sh
```

## 🔧 Comandos Rápidos

### Desenvolvimento

```bash
# Instala dependências de dev
pip install -e .[dev]

# Roda testes
pytest tests/ -v

# Roda com cobertura
pytest tests/ --cov=src --cov-report=html

# Linting
flake8 src/ tests/
black src/ tests/

# Type checking
mypy src/
```

### Build

```bash
# Limpa builds anteriores
rm -rf build/ dist/ *.egg-info

# Build novo
python -m build

# Verifica pacote
twine check dist/*
```

### Deploy

```bash
# Test PyPI
twine upload --repository-url https://test.pypi.org/legacy/ dist/*

# PyPI Interno
twine upload --repository-url https://pypi.empresa.com/simple dist/*
```

## 📦 Estrutura dos Scripts

Todos os scripts automatizam tarefas comuns:

- **test**: Executa testes e verifica qualidade
- **build**: Constrói pacotes wheel e source
- **deploy**: Publica no repositório interno
- **clean**: Limpa arquivos temporários
