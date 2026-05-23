# ESL desktop demo — full or sprint: .\run_demo.ps1 --sprint
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

. "$PSScriptRoot\scripts\esl_python.ps1"

$p = Get-EslPython -RepoRoot $PSScriptRoot
if ($null -eq $p) {
    Write-EslPythonMissingHint
    exit 1
}

if ($p.Type -eq 'Exe') {
    & $p.Path demo.py @args
} else {
    & py $p.Tag demo.py @args
}
