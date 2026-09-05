# Setup for the local voice agent — target: Windows 11 + single NVIDIA GPU (tuned for RTX 3090 24 GB)
#
# Run from the project folder in a normal PowerShell (no admin needed, except for the
# optional firewall rule at the end, which is skipped with a hint if not elevated):
#     cd C:\ai-voice-local
#     Set-ExecutionPolicy -Scope Process Bypass
#     .\setup.ps1
#
# What it does (idempotent — safe to re-run):
#   1. Python 3.12 venv at .\.venv (inside the project folder, git-ignored)
#   2. pip packages: faster-whisper + CUDA 12 wheels, piper-tts, pyVoIP<2, scipy, ollama client, ...
#   3. Ollama (winget) + performance env vars + pull of the LLM (default qwen3.8:27b, ~18 GB)
#   4. Piper voices into .\voices\ (en_US-lessac-high, de_DE-thorsten-high)
#   5. config.ini skeleton at .\config.ini (git-ignored)
#   6. Environment check: python tests\check_env.py
#
# Override the LLM:   .\setup.ps1 -Llm qwen3.5:9b
param(
    [string]$Llm = "qwen3.8:27b",
    [string[]]$Voices = @("en_US-lessac-high", "de_DE-thorsten-high"),
    [switch]$SkipModels
)

# Deliberately NOT "Stop": pip/ollama write progress to stderr, which PowerShell 5.1 would
# otherwise turn into terminating errors. Critical steps check $LASTEXITCODE instead.
$ErrorActionPreference = "Continue"
$project   = $PSScriptRoot
$venv      = Join-Path $project ".venv"
$localDir  = $project
$py        = Join-Path $venv "Scripts\python.exe"

function Step($msg) { Write-Host ""; Write-Host "==> $msg" -ForegroundColor Cyan }
function Warn($msg) { Write-Host "    ! $msg" -ForegroundColor Yellow }
function Fail($msg) { Write-Host ""; Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }

# ---------------------------------------------------------------- 0. GPU sanity
Step "GPU"
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
} else {
    Warn "nvidia-smi not found - install the NVIDIA driver first (https://www.nvidia.com/drivers)."
}

# ---------------------------------------------------------------- 1. Python 3.12 venv
Step "Python 3.12"
$have312 = $false
if (Get-Command py -ErrorAction SilentlyContinue) {
    $have312 = [bool](& py --list | Select-String "3\.12")
}
if (-not $have312) {
    Write-Host "    Python 3.12 not found - installing via winget ..."
    winget install --id Python.Python.3.12 --exact --silent --accept-package-agreements --accept-source-agreements
    $env:PATH = [Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" + [Environment]::GetEnvironmentVariable("PATH", "User")
}
if (-not (Test-Path $py)) {
    Write-Host "    Creating venv at $venv"
    py -3.12 -m venv $venv
    if (-not (Test-Path $py)) { Fail "venv creation failed - is Python 3.12 installed? (py --list)" }
}
& $py --version

# ---------------------------------------------------------------- 2. pip packages
Step "pip packages"
& $py -m pip install --upgrade pip wheel --quiet
# CUDA 12 runtime libs as pip wheels (cuBLAS + cuDNN 9) - no separate CUDA toolkit install needed.
& $py -m pip install --upgrade `
    "faster-whisper>=1.1" nvidia-cublas-cu12 nvidia-cudnn-cu12 `
    piper-tts "pyVoIP<2" scipy numpy sounddevice ollama `
    fastapi uvicorn websockets `
    caldav radicale
if ($LASTEXITCODE -ne 0) { Fail "pip install failed (see above)." }
& $py -m pip list --format=columns | Select-String "faster-whisper|ctranslate2|nvidia-cu|piper|pyVoIP|ollama|scipy"

# ---------------------------------------------------------------- 3. Ollama + LLM
Step "Ollama"
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Host "    Ollama not found - installing via winget ..."
    winget install --id Ollama.Ollama --exact --silent --accept-package-agreements --accept-source-agreements
    $env:PATH = [Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" + [Environment]::GetEnvironmentVariable("PATH", "User")
}
ollama --version

# Performance settings for a single 24 GB card (user-level env vars, picked up by the Ollama app):
#   flash attention + q8_0 KV cache roughly halve the context memory of a 27B model
#   keep the model resident (the agent also passes keep_alive=-1)
$ollamaEnv = @{
    "OLLAMA_FLASH_ATTENTION" = "1"
    "OLLAMA_KV_CACHE_TYPE"   = "q8_0"
    "OLLAMA_KEEP_ALIVE"      = "-1"
    "OLLAMA_MAX_LOADED_MODELS" = "1"
}
$changed = $false
foreach ($k in $ollamaEnv.Keys) {
    if ([Environment]::GetEnvironmentVariable($k, "User") -ne $ollamaEnv[$k]) {
        [Environment]::SetEnvironmentVariable($k, $ollamaEnv[$k], "User")
        $changed = $true
    }
    Set-Item -Path "Env:$k" -Value $ollamaEnv[$k]
}
if ($changed) {
    Write-Host "    Ollama env vars updated - restarting the Ollama app so it picks them up ..."
    Get-Process -Name "ollama*" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep 2
}
$ollamaApp = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama app.exe"
if (-not (Get-Process -Name "ollama" -ErrorAction SilentlyContinue)) {
    if (Test-Path $ollamaApp) { Start-Process $ollamaApp } else { Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden }
    Start-Sleep 4
}
if (-not $SkipModels) {
    Write-Host "    Pulling $Llm (this is ~18 GB for qwen3.8:27b - go get a coffee) ..."
    ollama pull $Llm
    if ($LASTEXITCODE -ne 0) { Fail "ollama pull $Llm failed - is the Ollama app running? (ollama list)" }
}

# ---------------------------------------------------------------- 4. Piper voices
Step "Piper voices -> $project\voices"
$voiceDir = Join-Path $project "voices"
New-Item -ItemType Directory -Force -Path $voiceDir | Out-Null
if (-not $SkipModels) {
    & $py -m piper.download_voices --download-dir $voiceDir @Voices
}
Get-ChildItem $voiceDir -Filter *.onnx | ForEach-Object { Write-Host "    $($_.Name)  $([math]::Round($_.Length/1MB)) MB" }

# ---------------------------------------------------------------- 5. config.ini
Step "config.ini"
$cfg = Join-Path $localDir "config.ini"
if (-not (Test-Path $cfg)) {
    Copy-Item (Join-Path $project "config.example.ini") $cfg
    Warn "Created $cfg - fill in the FRITZ!Box IP-phone password before the call tests."
} else {
    Write-Host "    exists: $cfg"
}
if ($Llm -ne "qwen3.8:27b") {
    Warn "You chose -Llm $Llm : set  llm = $Llm  under [models] in $cfg"
}

# ---------------------------------------------------------------- 6. Firewall (optional, needs admin)
Step "Windows firewall rule for the venv python (SIP 5060/UDP + RTP)"
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if ($isAdmin) {
    if (-not (Get-NetFirewallRule -DisplayName "VoiceAgent Python" -ErrorAction SilentlyContinue)) {
        New-NetFirewallRule -DisplayName "VoiceAgent Python" -Direction Inbound -Program $py -Profile Private -Action Allow | Out-Null   # SIP/RTP + Glass Box :8080
        Write-Host "    rule created"
    } else { Write-Host "    rule exists" }
} else {
    Warn "Not elevated - skipped. Run once as admin if the FRITZ!Box cannot reach the agent:"
    Warn "  New-NetFirewallRule -DisplayName 'VoiceAgent Python' -Direction Inbound -Program '$py' -Profile Private -Action Allow"
}

# ---------------------------------------------------------------- 7. Check
Step "Environment check"
& $py (Join-Path $project "tests\check_env.py")

Write-Host ""
Write-Host "Done. Next steps:" -ForegroundColor Green
Write-Host "  $venv\Scripts\Activate.ps1"
Write-Host "  cd $project"
Write-Host "  python tests\test_whisper.py       # STT on the GPU"
Write-Host "  python tests\test_call.py          # FRITZ!Box registration + greeting (needs config.ini)"
Write-Host "  python agent\call_agent.py         # full agent"
Write-Host ""
Write-Host "Recommended once: NVIDIA Control Panel -> Manage 3D settings -> 'CUDA - Sysmem Fallback Policy'"
Write-Host "  = 'Prefer No Sysmem Fallback' (otherwise VRAM overflow silently spills to RAM and gets 10x slower)."
