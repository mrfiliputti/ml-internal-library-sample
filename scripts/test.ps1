# Script de Testes e Qualidade
# Windows PowerShell

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  TESTES E VALIDAÇÃO DE QUALIDADE" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Cores para output
$SUCCESS = "Green"
$ERROR = "Red"
$INFO = "Yellow"

# 1. Linting com flake8
Write-Host "[1/4] Executando flake8 (PEP8)..." -ForegroundColor $INFO
flake8 src/ tests/ --max-line-length=88 --statistics
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Erros de linting encontrados!`n" -ForegroundColor $ERROR
    exit 1
}
Write-Host "✓ Linting OK`n" -ForegroundColor $SUCCESS

# 2. Verificação de formatação
Write-Host "[2/4] Verificando formatação (black)..." -ForegroundColor $INFO
black --check src/ tests/
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Código precisa ser formatado! Execute: black src/ tests/`n" -ForegroundColor $ERROR
    exit 1
}
Write-Host "✓ Formatação OK`n" -ForegroundColor $SUCCESS

# 3. Testes unitários
Write-Host "[3/4] Executando testes unitários..." -ForegroundColor $INFO
pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html
if ($LASTEXITCODE -ne 0) {
    Write-Host "`n✗ Testes falharam!`n" -ForegroundColor $ERROR
    exit 1
}
Write-Host "`n✓ Todos os testes passaram`n" -ForegroundColor $SUCCESS

# 4. Type checking (opcional)
Write-Host "[4/4] Type checking (mypy)..." -ForegroundColor $INFO
mypy src/ --ignore-missing-imports 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Type checking OK`n" -ForegroundColor $SUCCESS
} else {
    Write-Host "⚠ Type checking com avisos (não-crítico)`n" -ForegroundColor $INFO
}

# Resumo
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  ✓ TODOS OS CHECKS PASSARAM!" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "📊 Relatório de cobertura: htmlcov/index.html`n" -ForegroundColor $INFO
