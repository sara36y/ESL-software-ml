# ESL web demo — http://localhost:8000
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (Test-Path "venv\Scripts\Activate.ps1") {
    & "venv\Scripts\Activate.ps1"
} elseif (Test-Path ".venv\Scripts\Activate.ps1") {
    & ".venv\Scripts\Activate.ps1"
}

$py = Get-Command py -ErrorAction SilentlyContinue
if ($py) {
    & py -3.11 -m uvicorn web.server:app --host 0.0.0.0 --port 8000 --reload
} else {
    & python -m uvicorn web.server:app --host 0.0.0.0 --port 8000 --reload
}
