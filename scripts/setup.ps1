# Kangaroo setup for Windows: creates .venv, installs PyTorch + Kangaroo, installs the desktop app.
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 [-Torch cu128|cu126|cu124|cpu]
# Pick the CUDA build that matches your driver (RTX 50-series needs cu128 or newer). Use `cpu` for no GPU.
param([ValidateSet('cu128', 'cu126', 'cu124', 'cpu')][string]$Torch = 'cu128')
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path .venv)) { python -m venv .venv }
$py = Join-Path $root '.venv\Scripts\python.exe'
& $py -m pip install --upgrade pip
& $py -m pip install torch --index-url "https://download.pytorch.org/whl/$Torch"
& $py -m pip install -e ".[quant,hub]"

Push-Location app
npm install
# Some npm versions block Electron's binary download; harmless if it is already installed.
if (-not (Test-Path node_modules\electron\path.txt)) { node node_modules\electron\install.js }
Pop-Location

Write-Host "`nDone. Start the app with:  cd app; npm start"
