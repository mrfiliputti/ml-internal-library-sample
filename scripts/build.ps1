# Script de Build do Pacote
# Windows PowerShell

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  BUILD DO PACOTE" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

$SUCCESS = "Green"
$ERROR = "Red"
$INFO = "Yellow"

# 1. Limpa builds anteriores
Write-Host "[1/4] Limpando builds anteriores..." -ForegroundColor $INFO
if (Test-Path "dist") { Remove-Item -Recurse -Force dist }
if (Test-Path "build") { Remove-Item -Recurse -Force build }
if (Test-Path "*.egg-info") { Remove-Item -Recurse -Force *.egg-info }
Write-Host "✓ Limpeza concluída`n" -ForegroundColor $SUCCESS

# 2. Instala ferramentas de build
Write-Host "[2/4] Verificando ferramentas de build..." -ForegroundColor $INFO
pip install --upgrade build twine wheel setuptools 2>&1 | Out-Null
Write-Host "✓ Ferramentas OK`n" -ForegroundColor $SUCCESS

# 3. Build do pacote
Write-Host "[3/4] Construindo pacote..." -ForegroundColor $INFO
python -m build
if ($LASTEXITCODE -ne 0) {
    Write-Host "`n✗ Build falhou!`n" -ForegroundColor $ERROR
    exit 1
}
Write-Host "✓ Build concluído`n" -ForegroundColor $SUCCESS

# 4. Valida pacotes
Write-Host "[4/4] Validando pacotes..." -ForegroundColor $INFO
twine check dist/*
if ($LASTEXITCODE -ne 0) {
    Write-Host "`n✗ Validação falhou!`n" -ForegroundColor $ERROR
    exit 1
}
Write-Host "✓ Pacotes válidos`n" -ForegroundColor $SUCCESS

# Lista arquivos gerados
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  ✓ BUILD CONCLUÍDO!" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "📦 Pacotes gerados em dist/:`n" -ForegroundColor $INFO
Get-ChildItem dist/ | ForEach-Object {
    $size = [math]::Round($_.Length / 1KB, 2)
    Write-Host "   • $($_.Name) ($size KB)" -ForegroundColor White
}

Write-Host "`n💡 Próximos passos:" -ForegroundColor $INFO
Write-Host "   1. Teste localmente: pip install dist/*.whl" -ForegroundColor White
Write-Host "   2. Publique: .\scripts\deploy.ps1`n" -ForegroundColor White
