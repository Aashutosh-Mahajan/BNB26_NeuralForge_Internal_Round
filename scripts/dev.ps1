param([int]$Port = 8010)
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$pythonExecutable = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonExecutable)) {
    throw "Create .venv and install requirements.txt first. See README.md."
}
$arguments = @("-m", "uvicorn", "blackbox.api.app:app", "--reload", "--host", "127.0.0.1", "--port", "$Port")
if (Test-Path -LiteralPath ".env") {
    $arguments += @("--env-file", ".env")
}
& $pythonExecutable @arguments
exit $LASTEXITCODE
