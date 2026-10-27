param([string]$PythonPath = 'python', [switch]$Gpu)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) { & $PythonPath -m venv .venv }
$TeamPython = Join-Path (Get-Location) '.venv\Scripts\python.exe'
& $TeamPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
& $TeamPython -m pip uninstall -y onnxruntime onnxruntime-gpu
if ($Gpu) {
    & $TeamPython -m pip install --upgrade --force-reinstall torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed' }
    & $TeamPython -m pip install onnxruntime-gpu==1.24.4
}
else { & $TeamPython -m pip install onnxruntime==1.24.4 }
if ($LASTEXITCODE -ne 0) { throw 'ONNX Runtime installation failed' }
Write-Host 'Set OCR_DEVICE and EMBEDDING_DEVICE in .env to cpu/cuda, then python -m app.'
