param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CommandArguments
)
$ErrorActionPreference = 'Stop'
$versionDirectory = $PSScriptRoot
$pythonPath = Join-Path $versionDirectory '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Create the V2 environment first; see README.md.' }
$savedPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = Join-Path $versionDirectory 'src'
    & $pythonPath -m asteroid_validator_v2 @CommandArguments
    $commandExitCode = $LASTEXITCODE
} finally { $env:PYTHONPATH = $savedPythonPath }
exit $commandExitCode
