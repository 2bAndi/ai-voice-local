r"""Convert a WAV file to 8 kHz mono, 16 bit (pre-stage of the telephone format).
Reused later as a building block in the live pipeline.

Usage:
    python tests\resample_to_8k.py audio\greeting.wav audio\greeting_8k.wav
"""
import sys
import wave
import numpy as np
from scipy.signal import resample_poly


def wav_to_8k_mono(src: str, dst: str) -> None:
    with wave.open(src, "rb") as w:
        rate = w.getframerate()
        channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        raw = w.readframes(w.getnframes())

    if sampwidth != 2:
        sys.exit(f"Expected a 16-bit WAV, got {sampwidth*8} bit.")

    audio = np.frombuffer(raw, dtype=np.int16)
    if channels == 2:
        audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)

    if rate != 8000:
        audio = resample_poly(audio, up=8000, down=rate).astype(np.int16)

    with wave.open(dst, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(8000)
        out.writeframes(audio.tobytes())
    print(f"OK: {dst} (8 kHz mono, {len(audio)/8000:.1f}s)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    wav_to_8k_mono(sys.argv[1], sys.argv[2])
