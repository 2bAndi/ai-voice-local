r"""Streaming STT offline test: feed a recorded call (caller channel) in 20 ms slices.

Prints every partial with its audio time, the stable partial at the pause, the endpoint and
the final - i.e. exactly what the Glass Box sequence will show. Works with either backend.

Usage (venv active, project folder):
    python tests\test_stt_stream.py                          # latest calls\*.wav, backend from config.ini
    python tests\test_stt_stream.py calls\<file>.wav sherpa  # force a backend
    python tests\test_stt_stream.py ... --realtime           # pace the feed like a phone call
    python tests\test_stt_stream.py ... --cpu tiny           # CPU whisper model (dev machines without GPU)
"""
import sys
import time
import wave
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from agent import settings, stt_stream  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
flags = [a for a in sys.argv[1:] if a.startswith("--")]
wav = Path(args[0]) if args else sorted((PROJECT / "calls").glob("*.wav"))[-1]
backend = args[1] if len(args) > 1 else settings.STT_BACKEND
realtime = "--realtime" in flags

with wave.open(str(wav)) as w:
    a = np.frombuffer(w.readframes(w.getnframes()), np.int16)
    if w.getnchannels() == 2:
        a = a.reshape(-1, 2)[:, 0]          # left = caller
    assert w.getframerate() == 8000
print(f"{wav.name}: {len(a)/8000:.1f} s caller audio, backend {backend}")

if backend == "sherpa":
    stt = stt_stream.SherpaStream(PROJECT / settings.SHERPA_MODEL, PROJECT / settings.VAD_MODEL)
else:
    if "--cpu" in flags:
        from faster_whisper import WhisperModel
        name = sys.argv[sys.argv.index("--cpu") + 1] if len(sys.argv) > sys.argv.index("--cpu") + 1 else "tiny"
        model = WhisperModel(name, device="cpu", compute_type="int8")
    else:
        model = settings.load_whisper()
    stt = stt_stream.WhisperStream(model, PROJECT / settings.VAD_MODEL, settings.LANGUAGE,
                                   beam=settings.STT_BEAM, interval=settings.STT_INTERVAL)
stt.reset()

t_wall = time.time()
n_utt = 0
for i in range(0, len(a) - 160, 160):
    chunk = a[i:i + 160].tobytes()
    t_audio = i / 8000
    if realtime:
        time.sleep(max(0.0, t_audio - (time.time() - t_wall)))
    for r in stt.feed(chunk):
        if r.kind == "speech_start":
            print(f"{t_audio:7.2f}s  ▶ speech")
        elif r.kind == "partial":
            tag = "STABLE " if r.stable else "partial"
            ms = f"{r.decode_ms:4.0f} ms" if r.decode_ms else "       "
            print(f"{t_audio:7.2f}s    {tag} {ms}  \"{r.text}\"")
        elif r.kind == "endpoint":
            print(f"{t_audio:7.2f}s  ■ endpoint after {r.silence_ms} ms pause")
        elif r.kind == "final":
            n_utt += 1
            print(f"{t_audio:7.2f}s  ✔ FINAL #{n_utt} ({r.decode_ms:.0f} ms after endpoint, lang {r.language} {r.probability or ''}): \"{r.text}\"\n")
print(f"{n_utt} utterances, {time.time()-t_wall:.1f} s wall")
