# ESL web demo — http://localhost:8000
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

. "$PSScriptRoot\scripts\esl_python.ps1"

$p = Get-EslPython -RepoRoot $PSScriptRoot
if ($null -eq $p) {
    Write-EslPythonMissingHint
    exit 1
}

$uvicornArgs = @(
    '-m', 'uvicorn', 'web.server:app',
    '--host', '0.0.0.0', '--port', '8000', '--reload'
)

if ($p.Type -eq 'Exe') {
    & $p.Path @uvicornArgs
} else {
    & py $p.Tag @uvicornArgs
}
