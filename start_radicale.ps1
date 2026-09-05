# Starts Radicale (local CalDAV server, legacy appointment demo) on http://127.0.0.1:5232
# Run this in its OWN PowerShell window. Radicale 3.7+ needs Windows Developer Mode (symlinks).
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$data = Join-Path $PSScriptRoot "radicale-data"
New-Item -ItemType Directory -Force -Path $data | Out-Null
# radicale.config uses a relative storage path; run from the data dir so it lands there
Push-Location $data
& $py -m radicale --config (Join-Path $PSScriptRoot "radicale.config")
Pop-Location
