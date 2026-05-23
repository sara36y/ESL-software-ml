# Create .venv with Python 3.11 or 3.10 and install requirements.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

. "$PSScriptRoot\scripts\esl_python.ps1"

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    Write-Host "The 'py' launcher was not found. Install Python 3.11 from https://www.python.org/downloads/"
    Write-Host "and tick 'Add python.exe to PATH', then open a new terminal and run this script again."
    exit 1
}

$tag = $null
foreach ($t in @('-3.11', '-3.10')) {
    & py $t -c "import sys; sys.exit(0 if sys.version_info[:2] in ((3,10),(3,11)) else 1)" 2>$null
    if ($LASTEXITCODE -eq 0) {
        $tag = $t
        break
    }
}

if ($null -eq $tag) {
    Write-EslPythonMissingHint
    exit 1
}

Write-Host "Creating .venv with $tag ..."
& py $tag -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Write-Host ""
Write-Host "Done. Examples:"
Write-Host "  .\run_demo.ps1 --sprint --source 0"
Write-Host "  .\run_web.ps1"
