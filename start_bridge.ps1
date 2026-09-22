<#
    Starts the VoiceLK TTS Bridge on Windows.

    It reuses the existing training/inference virtual environment (that is where
    torch and the NLP libraries already live), installs the few service-level
    packages the first time, and then runs the API with uvicorn.

    Usage examples:
        .\start_bridge.ps1
        .\start_bridge.ps1 -Port 8001 -Reload
        .\start_bridge.ps1 -VenvPath "D:\path\to\another\venv"
#>

[CmdletBinding()]
param(
    [string] $VenvPath = "",
    [string] $BindHost = "0.0.0.0",
    [int]    $Port = 8000,
    [switch] $Reload,
    [switch] $SkipInstall
)

$ErrorActionPreference = "Stop"

$BridgeDir = $PSScriptRoot
$ProjectRoot = Split-Path -Parent $BridgeDir

if ([string]::IsNullOrWhiteSpace($VenvPath)) {
    $VenvPath = Join-Path $ProjectRoot "voicelk_ml\venv_training"
}

$Python = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Python executable not found at '$Python'. Pass -VenvPath with the environment that has torch installed."
}

Write-Host "VoiceLK TTS Bridge" -ForegroundColor Cyan
Write-Host "  interpreter : $Python"
Write-Host "  project     : $ProjectRoot"
Write-Host "  listening   : http://${BindHost}:$Port"

if (-not $SkipInstall) {
    Write-Host "Checking service dependencies..." -ForegroundColor Cyan
    & $Python -m pip install -q -r (Join-Path $BridgeDir "requirements.txt")
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
}

$uvicornArgs = @("-m", "uvicorn", "app.main:app", "--host", $BindHost, "--port", $Port)
if ($Reload) { $uvicornArgs += "--reload" }

Push-Location $BridgeDir
try {
    & $Python @uvicornArgs
}
finally {
    Pop-Location
}
