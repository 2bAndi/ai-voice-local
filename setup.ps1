# Setup for the local voice agent
# The venv lives deliberately OUTSIDE OneDrive (sync conflicts, tens of thousands of small files):
$venv = "C:\Users\broic\Code\voiceagent-venv"

Write-Host "Creating venv at $venv ..."
py -3.12 -m venv $venv
if (-not $?) {
    Write-Host "ERROR: Python 3.12 not found. Please install it from python.org (py -0 lists installed versions)." -ForegroundColor Red
    exit 1
}

& "$venv\Scripts\python.exe" -m pip install --upgrade pip
& "$venv\Scripts\pip.exe" install faster-whisper nvidia-cublas-cu12 nvidia-cudnn-cu12
& "$venv\Scripts\pip.exe" install piper-tts "pyVoIP<2" scipy sounddevice

Write-Host ""
Write-Host "Done. Activate with:" -ForegroundColor Green
Write-Host "  C:\Users\broic\Code\voiceagent-venv\Scripts\Activate.ps1"
Write-Host "Then continue in the project folder:"
Write-Host "  cd C:\ai-voice-local"
