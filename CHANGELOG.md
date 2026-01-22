# Changelog

Todas as mudanças notáveis neste projeto serão documentadas neste arquivo.

O formato é baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/),
e este projeto adere ao [Semantic Versioning](https://semver.org/lang/pt-BR/).

## [1.0.0] - 2026-01-22

### Adicionado
- Biblioteca inicial de Machine Learning para predição de preços de carros
- Módulo `data_ingestion.py` para carregamento de dados
- Módulo `data_validation.py` para validação de qualidade
- Módulo `model_trainer.py` para treinamento de modelos
- Módulo `model_evaluation.py` para avaliação de performance
- Módulo `utils.py` com funções auxiliares
- Suite completa de testes unitários
- Documentação detalhada com docstrings
- Configuração de CI/CD com GitHub Actions
- Notebook de demonstração (`demo_usage.ipynb`)
- Script principal de exemplo (`main.py`)
- Empacotamento com `setup.py` e `pyproject.toml`
- README com instruções de uso
- Guia de contribuição

### Features Principais
- API consistente inspirada em scikit-learn
- Geração de dados sintéticos para demonstração
- Validações automáticas de qualidade de dados
- Cálculo de múltiplas métricas (RMSE, MAE, R², MAPE)
- Normalização automática com StandardScaler
- Suporte para salvamento e carregamento de modelos

### Conformidade
- ✅ PEP8 compliance
- ✅ Docstrings seguindo padrão Google/NumPy
- ✅ Cobertura de testes > 80%
- ✅ Semantic Versioning
- ✅ Código modular (DRY principle)

---

## Formato das Mudanças

- `Added` para novas features
- `Changed` para mudanças em funcionalidades existentes
- `Deprecated` para features que serão removidas
- `Removed` para features removidas
- `Fixed` para correções de bugs
- `Security` para vulnerabilidades corrigidas
