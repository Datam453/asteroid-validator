param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CommandArguments
)
$ErrorActionPreference = 'Stop'
$savedPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = Join-Path $PSScriptRoot 'src'
    & (Join-Path $PSScriptRoot '.venv\Scripts\python.exe') @CommandArguments
    $commandExitCode = $LASTEXITCODE
} finally { $env:PYTHONPATH = $savedPythonPath }
exit $commandExitCode
