r"""Record real keyboard typing (e.g. via a Jabra headset mic) as the agent's "working" sound.

The agent plays a short typing clip while tools run (see make_typing_clip in call_agent.py,
which synthesizes one). A real recording sounds much more natural. This script:

  1. lists the input devices and picks the one matching --device (substring, default "Jabra")
  2. records N seconds (default 12) - type normally on the keyboard, occasional short pauses
  3. trims leading/trailing silence, normalizes to a quiet background level,
     resamples to 8 kHz mono s16 and writes  audio\typing_8k.wav
  call_agent.py picks that file up automatically and cuts random 1.2-1.6 s slices from it.

Usage (venv active, project folder):
    python tests\record_typing.py                 # 12 s from the first "Jabra" input device
    python tests\record_typing.py --seconds 20
    python tests\record_typing.py --device "USB"  # other mic
    python tests\record_typing.py --list          # just show devices
    python tests\record_typing.py --play          # play back the result on the default output
"""
import argparse
import sys
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

PROJECT = Path(__file__).resolve().parent.parent
RAW = PROJECT / "audio" / "typing_raw.wav"
OUT = PROJECT / "audio" / "typing_8k.wav"
TARGET_PEAK = 0.18       # quiet background character (same level as the synthetic clip)


def pick_device(name: str) -> int:
    devs = sd.query_devices()
    for i, d in enumerate(devs):
        if d["max_input_channels"] > 0 and name.lower() in d["name"].lower():
            return i
    sys.exit(f"No input device containing '{name}'. Use --list and --device.")


def list_devices():
    for i, d in enumerate(sd.query_devices()):
        io = ("in " if d["max_input_channels"] else "   ") + ("out" if d["max_output_channels"] else "   ")
        print(f"  [{i:2d}] {io}  {d['name']}  ({int(d['default_samplerate'])} Hz)")


def write_wav(path: Path, pcm16: np.ndarray, rate: int):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm16.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="Jabra")
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--play", action="store_true")
    a = ap.parse_args()

    if a.list:
        list_devices()
        return
    if a.play:
        with wave.open(str(OUT), "rb") as w:
            data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
            sd.play(data, w.getframerate())
            sd.wait()
        return

    dev = pick_device(a.device)
    info = sd.query_devices(dev)
    rate = int(info["default_samplerate"])
    print(f"Recording {a.seconds:.0f}s from [{dev}] {info['name']} @ {rate} Hz")
    for n in (3, 2, 1):
        print(f"  {n} ...")
        time.sleep(1)
    print("  TYPE NOW (normal pace, short pauses are fine)")
    audio = sd.rec(int(a.seconds * rate), samplerate=rate, channels=1, dtype="float32", device=dev)
    sd.wait()
    print("  done.")
    x = audio[:, 0]

    # --- trim silence at both ends (threshold relative to peak)
    peak = float(np.abs(x).max()) or 1.0
    thr = peak * 0.05
    idx = np.where(np.abs(x) > thr)[0]
    if len(idx) == 0:
        sys.exit("Only silence recorded - is the Jabra mic muted or another device selected?")
    x = x[max(0, idx[0] - int(0.1 * rate)): idx[-1] + int(0.2 * rate)]

    # --- DC removal + gentle high-pass (mic rumble), then normalize
    x = x - x.mean()
    b = np.array([1.0, -1.0]); a_ = np.array([1.0, -0.995])
    from scipy.signal import lfilter
    x = lfilter(b, a_, x).astype(np.float32)
    x = x / (np.abs(x).max() or 1.0) * TARGET_PEAK

    raw16 = (x * 32767).astype(np.int16)
    write_wav(RAW, raw16, rate)

    y = resample_poly(x, up=8000, down=rate)
    y = y / (np.abs(y).max() or 1.0) * TARGET_PEAK
    out16 = (y * 32767).astype(np.int16)
    write_wav(OUT, out16, 8000)
    print(f"Saved {RAW.name} ({len(raw16)/rate:.1f}s @ {rate} Hz) and {OUT.name} "
          f"({len(out16)/8000:.1f}s @ 8 kHz). Listen: python tests\\record_typing.py --play")
    if len(out16) / 8000 < 4:
        print("  ! Under 4 s of usable audio - record again with more typing for better variety.")


if __name__ == "__main__":
    main()
