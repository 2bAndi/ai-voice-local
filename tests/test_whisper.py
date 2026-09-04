r"""Step 2: Whisper GPU test.
Automatically picks the first audio file in audio\ (wav, m4a, mp3, ogg, flac) —
faster-whisper decodes these formats directly (PyAV is bundled).

Usage (from the project folder, venv active):
    python tests\test_whisper.py          # auto-detect the spoken language
    python tests\test_whisper.py en       # force a language code (en, de, da, ...)
"""
import os
import sys
import time
from pathlib import Path


def add_cuda_dlls():
    """Make the pip-installed NVIDIA DLLs (cuBLAS/cuDNN) discoverable on Windows.
    Must run BEFORE importing faster_whisper."""
    nvidia = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    for sub in ("cublas", "cudnn"):
        p = nvidia / sub / "bin"
        if p.exists():
            os.add_dll_directory(str(p))
            os.environ["PATH"] = str(p) + os.pathsep + os.environ.get("PATH", "")


add_cuda_dlls()

AUDIO_DIR = Path(__file__).resolve().parent.parent / "audio"
FORMATS = (".wav", ".m4a", ".mp3", ".ogg", ".flac")
LANGUAGE = sys.argv[1] if len(sys.argv) > 1 else None   # None = auto-detect

candidates = sorted(p for p in AUDIO_DIR.iterdir()
                    if p.suffix.lower() in FORMATS and not p.name.startswith("greeting"))
if not candidates:
    sys.exit(f"No test recording found in {AUDIO_DIR} ({', '.join(FORMATS)}).")

audio_file = candidates[0]
print(f"Test file: {audio_file.name}")

from faster_whisper import WhisperModel

print("Loading model large-v3-turbo ...")
t0 = time.time()
model = WhisperModel("large-v3-turbo", device="cuda", compute_type="int8_float16")
print(f"Model loaded in {time.time()-t0:.1f}s")

t0 = time.time()
segments, info = model.transcribe(str(audio_file), language=LANGUAGE, vad_filter=True)
for s in segments:
    print(f"[{s.start:6.1f}s] {s.text}")
print(f"\nTranscription in {time.time()-t0:.2f}s  |  audio length: {info.duration:.1f}s "
      f"|  language: {info.language}")
print("SUCCESS if the text is correct and transcription was clearly faster than the audio length.")
