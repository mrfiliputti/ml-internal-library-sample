# Guia de Contribuição

Obrigado por contribuir com o projeto Car Price Prediction! 🎉

## Como Contribuir

### 1. Fork e Clone

```bash
git clone https://github.com/seu-usuario/car-price-prediction.git
cd car-price-prediction
```

### 2. Crie um Ambiente Virtual

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Instale Dependências

```bash
pip install -r requirements.txt
pip install -e .[dev]
```

### 4. Crie uma Branch

```bash
git checkout -b feature/nome-da-feature
```

### 5. Faça suas Alterações

Siga as convenções:
- **PEP8**: Código deve seguir PEP8
- **Docstrings**: Use Google ou NumPy style
- **Testes**: Adicione testes para novas features

### 6. Execute os Testes

```bash
# Todos os testes
pytest tests/ -v

# Com cobertura
pytest tests/ --cov=src --cov-report=html
```

### 7. Verifique o Estilo

```bash
# Linting
flake8 src/ tests/

# Formatação
black src/ tests/

# Type checking (opcional)
mypy src/
```

### 8. Commit e Push

```bash
git add .
git commit -m "feat: descrição da mudança"
git push origin feature/nome-da-feature
```

### 9. Abra um Pull Request

Descreva:
- O que foi mudado
- Por que foi mudado
- Como testar

## Convenções de Código

### Nomenclatura

- **Variáveis/Funções**: `snake_case`
- **Classes**: `PascalCase`
- **Constantes**: `UPPER_CASE`
- **Módulos**: `snake_case.py`

### Docstrings

```python
def function_name(param1: str, param2: int) -> bool:
    """
    Breve descrição da função.

    Descrição mais detalhada se necessário.

    Parameters
    ----------
    param1 : str
        Descrição do parâmetro 1.
    param2 : int
        Descrição do parâmetro 2.

    Returns
    -------
    bool
        Descrição do retorno.

    Examples
    --------
    >>> function_name("test", 42)
    True
    """
    pass
```

### Commits

Use conventional commits:
- `feat:` Nova feature
- `fix:` Correção de bug
- `docs:` Documentação
- `test:` Testes
- `refactor:` Refatoração
- `style:` Formatação

## Testes

- Escreva testes para toda nova funcionalidade
- Mantenha cobertura > 80%
- Use fixtures do pytest quando apropriado

## Perguntas?

Abra uma issue ou entre em contato com o time!
