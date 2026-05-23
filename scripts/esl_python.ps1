# Shared helpers: resolve Python 3.10 / 3.11 for this project (TensorFlow pins).
# Dot-source from repo root: ` . "$PSScriptRoot\scripts\esl_python.ps1" `

function Get-EslPython {
    param(
        [string]$RepoRoot = (Get-Location).Path
    )

    foreach ($vd in @('.venv', 'venv')) {
        $exe = Join-Path $RepoRoot "$vd\Scripts\python.exe"
        if (-not (Test-Path $exe)) { continue }
        & $exe -c "import sys; sys.exit(0 if sys.version_info[:2] in ((3,10),(3,11)) else 1)" 2>$null
        if ($LASTEXITCODE -eq 0) {
            return [pscustomobject]@{ Type = 'Exe'; Path = $exe }
        }
    }

    if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
        return $null
    }

    foreach ($tag in @('-3.11', '-3.10')) {
        & py $tag -c "import sys; sys.exit(0 if sys.version_info[:2] in ((3,10),(3,11)) else 1)" 2>$null
        if ($LASTEXITCODE -eq 0) {
            return [pscustomobject]@{ Type = 'PyLauncher'; Tag = $tag }
        }
    }

    return $null
}

function Write-EslPythonMissingHint {
    Write-Host @"
No usable Python 3.10 or 3.11 found.

This project uses TensorFlow (see requirements.txt). On Windows, official wheels
typically support Python 3.10-3.11 (not 3.13).

Fix:
  1. Install Python 3.11 from https://www.python.org/downloads/ (enable 'Add to PATH').
  2. In this folder run:   .\setup_venv.ps1
  3. Then run:             .\run_demo.ps1   or   .\run_web.ps1
"@ -ForegroundColor Yellow
}

function Invoke-EslPython {
    param(
        [Parameter(Mandatory)][string[]]$Arguments,
        [string]$RepoRoot = (Get-Location).Path
    )

    $p = Get-EslPython -RepoRoot $RepoRoot
    if ($null -eq $p) {
        Write-EslPythonMissingHint
        exit 1
    }

    if ($p.Type -eq 'Exe') {
        & $p.Path @Arguments
    } else {
        & py $p.Tag @Arguments
    }
}
