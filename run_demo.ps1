# ESL desktop demo — full (threaded) or sprint: .\run_demo.ps1 --sprint
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (Test-Path "venv\Scripts\Activate.ps1") {
    & "venv\Scripts\Activate.ps1"
} elseif (Test-Path ".venv\Scripts\Activate.ps1") {
    & ".venv\Scripts\Activate.ps1"
}

$py = Get-Command py -ErrorAction SilentlyContinue
if ($py) {
    & py -3.11 demo.py @args
} else {
    & python demo.py @args
}
