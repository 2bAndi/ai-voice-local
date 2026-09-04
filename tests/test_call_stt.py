r"""Step 5c: call -> Whisper, now with 16-bit audio (voip16 patch).
- reads non-blocking on a 20 ms clock (silence is ordinary audio)
- records everything to audio\call_last.wav
- calibrates the speech threshold against the line noise
- transcription runs in its own thread

Usage:
    python tests\test_call_stt.py
Cross-check after the call:
    python tests\test_whisper.py   (picks up audio\call_last.wav automatically)
"""
import configparser
import os
import queue
import socket
import sys
import threading
import time
import wave
import warnings
import audioop
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
from scipy.signal import resample_poly

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))


def add_cuda_dlls():
    nvidia = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    for sub in ("cublas", "cudnn"):
        p = nvidia / sub / "bin"
        if p.exists():
            os.add_dll_directory(str(p))
            os.environ["PATH"] = str(p) + os.pathsep + os.environ.get("PATH", "")


add_cuda_dlls()

import voip16  # noqa: F401  (patches pyVoIP to 16 bit, pins PCMU)
from pyVoIP.VoIP import VoIPPhone, CallState, InvalidStateError
from faster_whisper import WhisperModel

CONFIG_CANDIDATES = [
    Path(r"C:\Users\broic\Code\voiceagent-local\config.ini"),
    PROJECT / "config.ini",
]
GREETING = PROJECT / "audio" / "greeting_8k.wav"
RECORDING = PROJECT / "audio" / "call_last.wav"

LANGUAGE = "en"            # STT language of the caller (en, de, da, ...)
END_SILENCE_S = 0.8
MIN_SPEECH_S = 0.15        # keep short confirmations ("yes, exactly")
PREROLL_S = 0.3            # pre-roll so soft word onsets are not clipped
CHUNK_BYTES = 320          # 20 ms at 8 kHz, 16 bit
CHUNK_S = 0.02


def load_config():
    for p in CONFIG_CANDIDATES:
        if p.exists():
            cfg = configparser.ConfigParser()
            cfg.read(p, encoding="utf-8")
            return cfg["sip"]
    sys.exit("No config.ini found.")


def local_ip(fritzbox: str) -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect((fritzbox, 5060))
    ip = s.getsockname()[0]
    s.close()
    return ip


def load_greeting_16() -> bytes:
    with wave.open(str(GREETING), "rb") as w:
        return w.readframes(w.getnframes())   # already 8 kHz mono s16


def to_whisper_input(pcm16: bytes) -> np.ndarray:
    audio = np.frombuffer(pcm16, dtype=np.int16)
    audio16k = resample_poly(audio, up=2, down=1)
    return (audio16k / 32768.0).astype(np.float32)


print("Loading Whisper model ...")
model = WhisperModel("large-v3-turbo", device="cuda", compute_type="int8_float16")
print("Model ready.")

GREETING_DATA = load_greeting_16()
work_q: "queue.Queue[bytes]" = queue.Queue()


def transcriber():
    while True:
        data = work_q.get()
        if data is None:
            break
        t0 = time.time()
        # vad_filter off: our energy VAD has already segmented the audio; Whisper's
        # own VAD would otherwise discard short or clipped utterances
        segments, _ = model.transcribe(to_whisper_input(data), language=LANGUAGE,
                                       vad_filter=False, beam_size=1)
        text = " ".join(s.text.strip() for s in segments).strip()
        dur = len(data) / 2 / 8000
        if text:
            print(f"CALLER [{dur:.1f}s audio, {time.time()-t0:.2f}s Whisper]: {text}")
        else:
            print(f"( {dur:.1f}s of audio produced no text )")


threading.Thread(target=transcriber, daemon=True).start()


def answer(call):
    print(">> Call answered, greeting is playing.")
    recorder = wave.open(str(RECORDING), "wb")
    recorder.setnchannels(1)
    recorder.setsampwidth(2)
    recorder.setframerate(8000)
    try:
        call.answer()
        call.write_audio(GREETING_DATA)
        time.sleep(len(GREETING_DATA) / 2 / 8000 + 0.2)

        # --- calibration: 0.5 s of line noise ---
        floor_vals = []
        for _ in range(25):
            chunk = call.read_audio(length=CHUNK_BYTES, blocking=False)
            recorder.writeframes(chunk)
            floor_vals.append(audioop.rms(chunk, 2))
            time.sleep(CHUNK_S)
        floor = sorted(floor_vals)[len(floor_vals) // 2]
        threshold = min(max(floor * 3, 250), 3000)
        print(f">> Noise floor RMS={floor}, speech threshold={threshold}. Start speaking.")

        speech = bytearray()
        preroll = bytearray()          # rolling pre-roll buffer
        preroll_max = int(PREROLL_S / CHUNK_S) * CHUNK_BYTES
        silence_chunks = 0
        sec_max = 0
        sec_t0 = time.time()
        next_read = time.time()
        while call.state == CallState.ANSWERED:
            next_read += CHUNK_S
            chunk = call.read_audio(length=CHUNK_BYTES, blocking=False)
            recorder.writeframes(chunk)
            rms = audioop.rms(chunk, 2)
            sec_max = max(sec_max, rms)
            if time.time() - sec_t0 >= 1.0:
                state = "SPEECH" if speech else "silent"
                print(f"   level max={sec_max:5d}  threshold={threshold}  [{state}]")
                sec_max = 0
                sec_t0 = time.time()

            if rms >= threshold:
                if not speech:
                    speech.extend(preroll)     # prepend the pre-roll
                speech.extend(chunk)
                silence_chunks = 0
            elif speech:
                speech.extend(chunk)
                silence_chunks += 1
                if silence_chunks * CHUNK_S >= END_SILENCE_S:
                    voiced_s = len(speech) / 2 / 8000 - END_SILENCE_S - PREROLL_S
                    if voiced_s >= MIN_SPEECH_S:
                        work_q.put(bytes(speech))
                    else:
                        print(f"( {voiced_s:.1f}s of speech discarded - too short )")
                    speech = bytearray()
                    silence_chunks = 0
            else:
                preroll.extend(chunk)          # silence: keep the pre-roll current
                if len(preroll) > preroll_max:
                    del preroll[:len(preroll) - preroll_max]

            time.sleep(max(0.0, next_read - time.time()))
        print(">> Caller hung up.")
    except InvalidStateError:
        pass
    finally:
        recorder.close()
        print(f">> Recording saved: {RECORDING}")


if __name__ == "__main__":
    sip = load_config()
    my_ip = local_ip(sip["server"])
    print(f"Registering {sip['user']}@{sip['server']} from {my_ip} (16-bit patch active) ...")

    phone = VoIPPhone(
        sip["server"], int(sip.get("port", 5060)),
        sip["user"], sip["password"],
        callCallback=answer, myIP=my_ip,
    )
    phone.start()
    print("Phone active. Call in and speak several sentences with pauses. Enter quits.")
    try:
        input()
    except (KeyboardInterrupt, EOFError):
        pass
    print("Shutting down ...")
    t = threading.Thread(target=phone.stop, daemon=True)
    t.start()
    t.join(timeout=5)
    os._exit(0)
