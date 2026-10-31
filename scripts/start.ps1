$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw 'Create the environment first: py -3.10 -m venv .venv'
}
if (-not $env:HF_HOME) {
    $env:HF_HOME = Join-Path $projectRoot '.venv\huggingface'
}
Push-Location $projectRoot
try {
    & $python -m streamlit run app.py --server.address 127.0.0.1 @args
    if ($LASTEXITCODE -ne 0) { throw "Streamlit exited with code $LASTEXITCODE" }
} finally {
    Pop-Location
}
