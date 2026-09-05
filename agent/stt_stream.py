r"""Streaming speech-to-text for SP-B/SP-C: 20 ms audio slices in, partial and final transcripts out.

This is the shape the PoV has: the session controller (SP-B) pushes every RTP frame into the
recogniser as it arrives; the recogniser (SP-C) answers with growing partial transcripts while
the caller is still speaking and decides itself when the utterance is over (endpointing).
Nothing waits for "the whole sentence" any more.

Two backends behind one interface - the STT is a swap point:

  whisper   faster-whisper (large-v3-turbo on the GPU) driven as a stream: a decode worker
            re-transcribes the audio since the last committed word every `interval` seconds;
            words that two consecutive hypotheses agree on are committed (LocalAgreement-2,
            Machacek et al. 2023). Multilingual (en/de/da), same quality as the old batch path.
  sherpa    sherpa-onnx streaming Zipformer transducer (real streaming model, CPU, ~300 ms
            word latency, English only). The recogniser also has its own endpoint rules; we
            use the shared Silero VAD for consistency.

Endpointing (both backends): Silero VAD (sherpa-onnx, stateful) on the 16 kHz stream.
    speech  -> `vad.start`
    pause   -> VAD reports non-speech for PAUSE_MS: the partial is decoded once more and
               flagged `stable` (this is what the speculative LLM start reacts to)
    endpoint-> pause lasts ENDPOINT_MS: `final` - for whisper the stable hypothesis IS the
               final (no second decode), so the transcript is ready ~0 ms after the endpoint.

    stt = make_stt(settings, language="en")      # or None to auto-detect (whisper only)
    stt.reset()
    for chunk in rtp_frames_20ms:                # 320 bytes s16le 8 kHz
        for r in stt.feed(chunk):                # 0..n Result objects
            ...
    Result(kind="speech_start" | "partial" | "endpoint" | "final",
           text, stable, language, probability, audio_ms, decode_ms)

Everything here is thread-safe for one feeder thread; the whisper worker runs in its own
thread and never blocks feed().
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
IN_RATE = 8000
RATE = 16000
PREROLL_S = 0.3            # audio kept before the VAD's speech start
PAUSE_MS = 250             # VAD hangover: "the caller paused" -> stable partial
ENDPOINT_MS = 600          # pause length that ends the utterance
MAX_UTTERANCE_S = 30.0
_WORD_NORM = re.compile(r"[^\w]+", re.UNICODE)


@dataclass
class Result:
    kind: str                      # speech_start | partial | endpoint | final
    text: str = ""
    stable: bool = False
    language: str | None = None
    probability: float | None = None
    audio_ms: int = 0
    decode_ms: float | None = None
    silence_ms: int = 0
    backend: str = ""
    audio: bytes | None = None     # final only: the utterance as s16le 8 kHz (for the recording/tests)


class _Upsampler:
    """8 kHz -> 16 kHz, stateful linear interpolation (no chunk-edge artefacts)."""

    def __init__(self):
        self.last = 0.0

    def __call__(self, pcm8k: bytes) -> np.ndarray:
        x = np.frombuffer(pcm8k, dtype=np.int16).astype(np.float32) / 32768.0
        if not len(x):
            return x
        prev = np.concatenate(([self.last], x[:-1]))
        out = np.empty(2 * len(x), dtype=np.float32)
        out[0::2] = (prev + x) / 2
        out[1::2] = x
        self.last = float(x[-1])
        return out


class _Vad:
    """Silero VAD via sherpa-onnx, streaming/stateful. is_speech() follows the model with a
    hangover of PAUSE_MS (min_silence_duration)."""

    def __init__(self, model_path: Path, threshold: float = 0.5):
        import sherpa_onnx
        cfg = sherpa_onnx.VadModelConfig()
        cfg.silero_vad.model = str(model_path)
        cfg.silero_vad.threshold = threshold
        cfg.silero_vad.min_silence_duration = PAUSE_MS / 1000
        cfg.silero_vad.min_speech_duration = 0.15
        cfg.silero_vad.max_speech_duration = MAX_UTTERANCE_S
        cfg.sample_rate = RATE
        self._vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=MAX_UTTERANCE_S + 5)
        self._buf = np.zeros(0, dtype=np.float32)

    def feed(self, x: np.ndarray) -> bool:
        self._buf = np.concatenate((self._buf, x))
        while len(self._buf) >= 512:
            self._vad.accept_waveform(self._buf[:512])
            self._buf = self._buf[512:]
        while not self._vad.empty():
            self._vad.pop()
        return self._vad.is_speech_detected()

    def reset(self):
        self._vad.reset()
        self._buf = np.zeros(0, dtype=np.float32)


class StreamingSTT:
    """Common state machine: VAD, utterance buffer, pause/endpoint timing. Backends implement
    _on_audio(x16k), _decode(stable: bool) -> (text, language, prob), _reset_backend()."""
    name = "base"

    def __init__(self, vad_model: Path, language: str | None):
        self.language = language
        self._up = _Upsampler()
        self._vad = _Vad(vad_model)
        self.reset()

    # ------------------------------------------------------------------ public
    def reset(self):
        self._vad.reset()
        self._up = _Upsampler()
        self._pre: list[bytes] = []
        self._utt = bytearray()          # s16le 8 kHz of the current utterance
        self._in_speech = False
        self._pause_since: float | None = None
        self._stable_sent = False
        self._t_speech_start = None
        self._clock = 0.0                # audio time in seconds (samples fed / 8000), not wall-clock
        self._reset_backend()

    def feed(self, pcm8k: bytes) -> list[Result]:
        out: list[Result] = []
        self._clock += len(pcm8k) / 2 / IN_RATE
        now = self._clock
        x = self._up(pcm8k)
        speech = self._vad.feed(x)
        if not self._in_speech:
            self._pre.append(pcm8k)
            if len(self._pre) > int(PREROLL_S * 50):
                self._pre.pop(0)
            if speech:
                self._in_speech = True
                self._t_speech_start = now
                self._utt = bytearray(b"".join(self._pre))
                self._on_audio(self._up16(self._utt))          # backend gets the pre-roll too
                out.append(Result("speech_start", backend=self.name))
            return out
        # in an utterance
        self._utt.extend(pcm8k)
        self._on_audio(x)
        if speech:
            self._pause_since = None
            self._stable_sent = False
            r = self._partial(stable=False)
            if r:
                out.append(r)
        else:
            if self._pause_since is None:
                self._pause_since = now - PAUSE_MS / 1000      # VAD hangover already elapsed
            silence_ms = int((now - self._pause_since) * 1000)
            if not self._stable_sent:
                self._stable_sent = True
                r = self._partial(stable=True, force=True)
                if r:
                    out.append(r)
            if silence_ms >= ENDPOINT_MS or len(self._utt) / 2 / IN_RATE >= MAX_UTTERANCE_S:
                out.append(Result("endpoint", silence_ms=silence_ms, backend=self.name,
                                  audio_ms=int(len(self._utt) / 16)))
                t0 = time.time()
                text, lang, prob = self._decode(final=True)
                out.append(Result("final", text=text, stable=True, language=lang, probability=prob,
                                  audio_ms=int(len(self._utt) / 16), decode_ms=(time.time() - t0) * 1000,
                                  silence_ms=silence_ms, backend=self.name, audio=bytes(self._utt)))
                self._in_speech = False
                self._pre = []
                self._utt = bytearray()
                self._pause_since = None
                self._stable_sent = False
                self._reset_backend()
        return out

    # ------------------------------------------------------------------ helpers
    def _up16(self, pcm: bytes) -> np.ndarray:
        return _Upsampler()(pcm)

    def _partial(self, stable: bool, force: bool = False) -> Result | None:
        raise NotImplementedError

    def _decode(self, final: bool):
        raise NotImplementedError

    def _on_audio(self, x16k: np.ndarray):
        raise NotImplementedError

    def _reset_backend(self):
        raise NotImplementedError


# ====================================================================== whisper as a stream
class WhisperStream(StreamingSTT):
    name = "whisper"

    def __init__(self, model, vad_model: Path, language: str | None, beam: int = 5,
                 interval: float = 0.7):
        self.model = model
        self.hint = ""          # context for the decoder: domain vocabulary + what the agent just asked
        self.task = "transcribe"   # or "translate": whisper emits English whatever the caller speaks
        self.beam = beam
        self.interval = interval
        self._lock = threading.RLock()
        self._worker_lock = threading.Lock()
        super().__init__(vad_model, language)

    def _reset_backend(self):
        with self._lock:
            self._audio = np.zeros(0, dtype=np.float32)   # 16 kHz float of the utterance
            self._commit_off = 0                           # samples already committed
            self._committed: list[str] = []
            self._hyp: list[tuple[str, float]] = []        # (word, end_s relative to commit_off)
            self._last_decode_len = 0
            self._last_decode_t = 0.0
            self._detected: tuple[str | None, float | None] = (None, None)
            self._pending_stable = False
            self._version = 0
            self._result_version = 0

    def _on_audio(self, x16k):
        with self._lock:
            self._audio = np.concatenate((self._audio, x16k))

    def _words(self, audio: np.ndarray, beam: int) -> tuple[list[tuple[str, float]], str | None, float | None]:
        segments, info = self.model.transcribe(
            audio, language=self.language, task=self.task, beam_size=beam, word_timestamps=True,
            condition_on_previous_text=False, vad_filter=False,
            initial_prompt=(" ".join(filter(None, [self.hint, " ".join(self._committed[-30:])])) or None))
        words = []
        for s in segments:
            for w in (s.words or []):
                words.append((w.word.strip(), float(w.end)))
        lang = info.language if self.language is None else self.language
        prob = float(info.language_probability) if self.language is None else None
        return words, lang, prob

    @staticmethod
    def _norm(w: str) -> str:
        return _WORD_NORM.sub("", w).lower()

    def _decode_pass(self, beam: int, commit: bool) -> float:
        """One decode of the uncommitted audio; LocalAgreement commit. Returns decode ms."""
        with self._lock:
            audio = self._audio[self._commit_off:].copy()
            prev = list(self._hyp)
        if len(audio) < RATE // 4:
            return 0.0
        t0 = time.time()
        words, lang, prob = self._words(audio, beam)
        ms = (time.time() - t0) * 1000
        with self._lock:
            if lang:
                self._detected = (lang, prob)
            if commit and prev:
                n = 0
                while n < len(prev) and n < len(words) and self._norm(prev[n][0]) == self._norm(words[n][0]):
                    n += 1
                if n:
                    self._committed.extend(w for w, _ in words[:n])
                    cut = int(words[n - 1][1] * RATE)
                    self._commit_off += min(cut, len(self._audio) - self._commit_off)
                    words = [(w, e - words[n - 1][1]) for w, e in words[n:]]
            self._hyp = words
            self._last_decode_len = len(self._audio)
            self._last_decode_t = time.time()
        return ms

    def _partial(self, stable: bool, force: bool = False) -> Result | None:
        with self._lock:
            due = (len(self._audio) - self._last_decode_len) >= int(self.interval * RATE) and \
                  (time.time() - self._last_decode_t) >= self.interval * 0.8
            busy = self._worker_lock.locked()
        if force:
            # the caller paused: decode what we have now, this hypothesis may become the final
            with self._worker_lock:
                ms = self._decode_pass(self.beam, commit=False)
            return Result("partial", text=self.text(), stable=True, decode_ms=ms, backend=self.name,
                          language=self._detected[0], probability=self._detected[1],
                          audio_ms=int(len(self._utt) / 16))
        if due and not busy:
            threading.Thread(target=self._worker, daemon=True).start()
            return None
        # report a new hypothesis produced by the worker since the last report
        with self._lock:
            if self._result_version != self._version:
                self._result_version = self._version
                return Result("partial", text=self.text(), stable=False, backend=self.name,
                              language=self._detected[0], probability=self._detected[1],
                              audio_ms=int(len(self._utt) / 16), decode_ms=self._last_ms)
        return None

    _last_ms = 0.0

    def _worker(self):
        if not self._worker_lock.acquire(blocking=False):
            return
        try:
            self._last_ms = self._decode_pass(1, commit=True)
            with self._lock:
                self._version += 1
        finally:
            self._worker_lock.release()

    def text(self) -> str:
        with self._lock:
            return " ".join(self._committed + [w for w, _ in self._hyp]).strip()

    def _decode(self, final: bool):
        with self._worker_lock:
            with self._lock:
                fresh = (len(self._audio) - self._last_decode_len) < int(0.8 * RATE)   # only silence since the stable decode
            if not fresh:
                self._decode_pass(self.beam, commit=False)
            return self.text(), self._detected[0], self._detected[1]


# ====================================================================== sherpa-onnx streaming
class SherpaStream(StreamingSTT):
    name = "sherpa"

    def __init__(self, model_dir: Path, vad_model: Path, language: str | None = "en", threads: int = 4):
        import sherpa_onnx
        d = Path(model_dir)
        enc = next(d.glob("encoder*chunk*.int8.onnx"), None) or next(d.glob("encoder*.onnx"))
        dec = next(d.glob("decoder*.onnx"))
        joi = next(d.glob("joiner*chunk*.int8.onnx"), None) or next(d.glob("joiner*.onnx"))
        self.rec = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(d / "tokens.txt"), encoder=str(enc), decoder=str(dec), joiner=str(joi),
            num_threads=threads, sample_rate=RATE, feature_dim=80, decoding_method="greedy_search",
            enable_endpoint_detection=False)
        self._stream = None
        super().__init__(vad_model, "en")

    def _reset_backend(self):
        self._stream = self.rec.create_stream()
        self._text = ""
        self._last_text = ""

    def _on_audio(self, x16k):
        self._stream.accept_waveform(RATE, x16k)
        while self.rec.is_ready(self._stream):
            self.rec.decode_stream(self._stream)
        self._text = self.rec.get_result(self._stream).strip()

    def _partial(self, stable: bool, force: bool = False) -> Result | None:
        if force or self._text != self._last_text:
            self._last_text = self._text
            return Result("partial", text=self._text, stable=stable, backend=self.name, language="en",
                          audio_ms=int(len(self._utt) / 16))
        return None

    def _decode(self, final: bool):
        # flush the encoder's look-ahead so the last word is complete
        self._stream.accept_waveform(RATE, np.zeros(int(0.5 * RATE), dtype=np.float32))
        self._stream.input_finished()
        while self.rec.is_ready(self._stream):
            self.rec.decode_stream(self._stream)
        return self.rec.get_result(self._stream).strip(), "en", None


# ====================================================================== factory
def make_stt(settings, whisper_model=None, language: str | None = None) -> StreamingSTT:
    vad = PROJECT / settings.VAD_MODEL
    if settings.STT_BACKEND == "sherpa":
        if language not in (None, "en"):
            raise ValueError("sherpa backend: only the English streaming model is installed")
        return SherpaStream(PROJECT / settings.SHERPA_MODEL, vad, "en")
    return WhisperStream(whisper_model, vad, language, beam=settings.STT_BEAM, interval=settings.STT_INTERVAL)
