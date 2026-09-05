r"""Step 2: Whisper GPU test.
Automatically picks the first audio file in audio\ (wav, m4a, mp3, ogg, flac) —
faster-whisper decodes these formats directly (PyAV is bundled).

Usage (from the project folder, venv active):
    python tests\test_whisper.py          # auto-detect the spoken language
    python tests\test_whisper.py en       # force a language code (en, de, da, ...)
"""
import sys
import time
from pathlib import Path


PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from agent import settings  # noqa: E402  (registers the CUDA DLL dirs first)

AUDIO_DIR = PROJECT / "audio"
FORMATS = (".wav", ".m4a", ".mp3", ".ogg", ".flac")
LANGUAGE = sys.argv[1] if len(sys.argv) > 1 else None   # None = auto-detect

candidates = sorted(p for p in AUDIO_DIR.iterdir()
                    if p.suffix.lower() in FORMATS and not p.name.startswith("greeting"))
if not candidates:
    sys.exit(f"No test recording found in {AUDIO_DIR} ({', '.join(FORMATS)}).")

audio_file = candidates[0]
print(f"Test file: {audio_file.name}")

print(f"Loading model {settings.STT_MODEL} ({settings.STT_COMPUTE}) ...")
t0 = time.time()
model = settings.load_whisper()
print(f"Model loaded in {time.time()-t0:.1f}s")

t0 = time.time()
segments, info = model.transcribe(str(audio_file), language=LANGUAGE, vad_filter=True,
                                  beam_size=settings.STT_BEAM)
for s in segments:
    print(f"[{s.start:6.1f}s] {s.text}")
print(f"\nTranscription in {time.time()-t0:.2f}s  |  audio length: {info.duration:.1f}s "
      f"|  language: {info.language}")
print("SUCCESS if the text is correct and transcription was clearly faster than the audio length.")
