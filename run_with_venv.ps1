# PowerShell script to run Python with venv
# Bypasses execution policy by using python.exe directly

$venvPython = ".\venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    & $venvPython $args
} else {
    Write-Host "Error: venv Python not found at $venvPython"
    exit 1
}

