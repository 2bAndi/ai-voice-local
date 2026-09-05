r"""Per-call audio recording for the Glass Box (Phase 2) - listen to a call and scrub through it.

Stereo 8 kHz 16-bit WAV next to the event file:  calls\<stamp>_<corr8>.wav
    left  = caller  (inbound RTP, what SP-A hands to SP-B)
    right = agent   (outbound: greeting, TTS sentences, typing filler)

Both channels are placed on the SAME clock as the events (events.call_t0()), so a position
in the audio player equals the `t` of an event: the page can jump the map/sequence to the
moment the audio is at, and clicking an event row seeks the audio.

    rec = CallRecorder(events.call_t0())
    rec.add(0, pcm16_bytes, at=time.time())      # caller chunk as it arrives
    rec.add(1, pcm16_bytes, at=start_ts)         # agent audio at the moment playback starts
    path = rec.save(events.call_file().with_suffix(".wav"))

Placement rule: a chunk goes to sample offset (at - t0) * 8000, but never before the end of
the previous chunk on that channel - contiguous streams stay contiguous, gaps become silence.
Memory: 16 KB/s per channel (~10 MB for a 5-minute call), written once at the end.
"""
import threading
import time
import wave
from pathlib import Path

import numpy as np

RATE = 8000


class CallRecorder:
    def __init__(self, t0: float | None = None):
        self.t0 = t0 or time.time()
        self._lock = threading.Lock()
        self._chunks: list[list[tuple[int, bytes]]] = [[], []]     # per channel: (offset_samples, pcm)
        self._end: list[int] = [0, 0]                              # per channel: end offset in samples

    def add(self, channel: int, pcm: bytes, at: float | None = None) -> None:
        if not pcm:
            return
        at = at or time.time()
        with self._lock:
            off = max(int((at - self.t0) * RATE), self._end[channel])
            self._chunks[channel].append((off, pcm))
            self._end[channel] = off + len(pcm) // 2

    def seconds(self) -> float:
        return max(self._end) / RATE

    def save(self, path: Path) -> Path | None:
        with self._lock:
            n = max(self._end)
            if n == 0:
                return None
            out = np.zeros((n, 2), dtype=np.int16)
            for ch in (0, 1):
                for off, pcm in self._chunks[ch]:
                    a = np.frombuffer(pcm, dtype=np.int16)
                    a = a[: max(0, n - off)]
                    out[off:off + len(a), ch] = a
        path.parent.mkdir(exist_ok=True)
        with wave.open(str(path), "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(out.tobytes())
        return path
