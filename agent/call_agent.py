r"""Local phone agent — use case: CREDIT CARD BLOCKING (English).

Pipeline identical to the appointment demo (V3): FRITZ!Box/pyVoIP(16-bit patch)
-> reader thread + energy VAD -> faster-whisper -> Qwen3 (streaming, tools)
-> Piper -> phone. Only the dialog layer and tools changed.

Dialog flow enforced by code-level guardrails in agent/bank_tools.py:
  reason -> identify (name + DOB) -> security question -> card selection
  -> explicit confirmation -> block -> reference number -> optional replacement.

Prerequisites: Ollama service (qwen3:8b), English Piper voice:
    python -m piper.download_voices en_US-lessac-high
Start:  python agent\call_agent.py
"""
import collections
import configparser
import datetime as dt
import json
import os
import queue
import random
import re
import socket
import subprocess
import sys
import tempfile
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

import voip16  # noqa: F401
from pyVoIP.VoIP import VoIPPhone, CallState, InvalidStateError
from faster_whisper import WhisperModel
from ollama import chat
from agent import bank_tools
from agent.bank_tools import (identify_customer, verify_identity, list_cards,
                              block_card, order_replacement_card)

MODEL = "qwen3:8b"

# ----- Language: "en" | "de" | "da" (prompt language + voice + STT) -----
# NOTE ON LANGUAGE: the project language is English — code, comments, prompts and
# logs are English throughout. The tables below (LANG_CFG, MONTHS) are the ONLY
# intentionally non-English literals in this repository: they are runtime voice
# output spoken to a caller in their own language, i.e. data, not documentation.
# Do not "translate" them — set LANGUAGE = "en" to run the agent in English.
LANGUAGE = "en"
LANG_CFG = {
    "en": {"voice": "en_US-lessac-high", "whisper": "en", "reply_lang": "English",
           "greeting": "Hello, you have reached the card security line. How can I help you?",
           "fallback": "I am sorry, I did not catch that. Could you say that again, please?",
           "fallback_start": "I am sorry, I did not catch that. Do you want to block your credit card?",
           "fillers": ["One moment, please.", "Just a second, please.", "Let me take care of that.", "Bear with me for a moment."]},
    "de": {"voice": "de_DE-thorsten-high", "whisper": "de", "reply_lang": "German",
           "greeting": "Guten Tag, hier ist die Kartensperr-Hotline. Wie kann ich Ihnen helfen?",
           "fallback": "Entschuldigung, das habe ich nicht verstanden. Koennen Sie das bitte wiederholen?",
           "fallback_start": "Entschuldigung, das habe ich nicht verstanden. Moechten Sie Ihre Kreditkarte sperren?",
           "fillers": ["Einen kleinen Moment bitte.", "Eine Sekunde, ich kuemmere mich darum.", "Augenblick bitte."]},
    "da": {"voice": "da_DK-talesyntese-medium", "whisper": "da", "reply_lang": "Danish",
           "greeting": "Goddag, De har ringet til kortspærringslinjen. Hvordan kan jeg hjælpe?",
           "fallback": "Undskyld, det fangede jeg ikke. Kan De gentage det?",
           "fallback_start": "Undskyld, det fangede jeg ikke. Oensker De at spaerre Deres kreditkort?",
           "fillers": ["Et øjeblik.", "Lige et øjeblik, tak."]},
}
CFG_L = LANG_CFG[LANGUAGE]
VOICE_NAME = CFG_L["voice"]
VOICE_ONNX = PROJECT / f"{VOICE_NAME}.onnx"
CONFIG_CANDIDATES = [
    Path(r"C:\Users\broic\Code\voiceagent-local\config.ini"),
    PROJECT / "config.ini",
]
END_SILENCE_S = 0.8
MIN_SPEECH_S = 0.15
PREROLL_S = 0.3
CHUNK_BYTES = 320
CHUNK_S = 0.02
VAD_THRESHOLD = 250

TOOLS = {
    "identify_customer": identify_customer,
    "verify_identity": verify_identity,
    "list_cards": list_cards,
    "block_card": block_card,
    "order_replacement_card": order_replacement_card,
}


def build_system_prompt() -> str:
    now = dt.datetime.now()
    return f"""You are the automated card security line of Demo Bank. Your only job is to help
callers BLOCK a credit card (and optionally order a replacement).
You MUST reply exclusively in {CFG_L['reply_lang']}. Today is {now.strftime('%A, %d %B %Y')}.

Follow this flow strictly, one step at a time:
1. Ask what happened (reason must be one of: lost, stolen, fraud, damaged).
   If the caller's first sentence already states what happened (e.g. "I lost my
   card", "my card was stolen"), take that as the reason and do NOT ask again -
   continue directly with identification.
2. Identify the caller: first ask ONLY for their full name. After they answer, ask
   for their date of birth in a separate question. Then call identify_customer.
   Convert the spoken date to YYYY-MM-DD internally for the tool call; when reading
   it back for confirmation, say it in words, never in a technical format.
3. Ask the security question returned by identify_customer, then call verify_identity
   with the caller's answer.
4. Call list_cards. Read the available cards to the caller (type and last four
   digits, digits spoken individually) and let them choose one.
5. Summarize card and reason and get an explicit yes.
6. Only then call block_card. Read the reference number back digit by digit.
7. Offer a replacement card; call order_replacement_card only if the caller wants one.

Hard rules:
- Reply in at most 1-2 short sentences; they are read out loud on the phone. Ask only
  ONE question at a time.
- NEVER ask for a full card number. Only the last four digits are ever used.
- Say all numbers, dates and digits in words when speaking.
- NEVER invent information the caller did not provide, and NEVER claim a card is
  blocked, verified, or a replacement ordered unless the tool result says SUCCESS.
  If a tool returns ERROR, follow the instruction inside the error message.
- Call tools directly WITHOUT any announcement text. A waiting message is played
  automatically while tools run. NEVER say things like "one moment, I will check" -
  every reply either asks the caller a question, states a tool result, or is a tool call.
- Ask for the date of birth in natural language. NEVER ask the caller to use a format
  like YYYY-MM-DD - you convert their spoken date yourself and read it back in words.
  Never state facts before the tool result.
- If verification fails three times, apologize and refer the caller to the human
  hotline; do not continue.
- If the caller asks for anything other than blocking or replacing a card, politely
  say this line only handles card blocking.
- Treat the situation with calm urgency; a caller reporting a stolen card may be
  stressed. Be reassuring but efficient."""


# ----------------------------- TTS: Piper in-process -----------------------------
_piper_voice = None


def _load_piper():
    global _piper_voice
    try:
        from piper import PiperVoice
        _piper_voice = PiperVoice.load(str(VOICE_ONNX))
        _ = _synth_inprocess("Test")
        print("Piper: loaded in-process.")
    except Exception as e:
        _piper_voice = None
        print(f"Piper: in-process API unavailable ({e}), using subprocess fallback.")


def _synth_inprocess(text: str) -> tuple[bytes, int]:
    chunks = list(_piper_voice.synthesize(text))
    rate = getattr(chunks[0], "sample_rate", None) or _piper_voice.config.sample_rate
    pcm = b"".join(
        getattr(c, "audio_int16_bytes", None) or getattr(c, "audio_int16_array").tobytes()
        for c in chunks
    )
    return pcm, rate


def _synth_subprocess(text: str) -> tuple[bytes, int]:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "out.wav"
        subprocess.run(
            [sys.executable, "-m", "piper", "-m", VOICE_NAME, "-f", str(out), "--", text],
            cwd=PROJECT, check=True, capture_output=True,
        )
        with wave.open(str(out), "rb") as w:
            return w.readframes(w.getnframes()), w.getframerate()


_md_cleanup = re.compile(r"[*_#`]")

# Month names per output language (spoken data, see the language note above).
MONTHS = {
    "en": ["January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December"],
    "de": ["Januar", "Februar", "Maerz", "April", "Mai", "Juni", "Juli",
           "August", "September", "Oktober", "November", "Dezember"],
    "da": ["januar", "februar", "marts", "april", "maj", "juni", "juli",
           "august", "september", "oktober", "november", "december"],
}


def _speakable(text: str) -> str:
    """Make technical formats readable aloud, whatever the LLM writes:
    - ISO date 1980-05-12   -> "May 12, 1980" (or "12. Mai 1980")
    - references CB-400654  -> "C B 4 0 0 6 5 4"
    - 4-digit groups (not years) -> single digits "4 8 2 1"
    """
    months = MONTHS[LANGUAGE]

    def date_repl(m):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if LANGUAGE == "en":
            return f"{months[mo-1]} {d}, {y}"
        return f"{d}. {months[mo-1]} {y}"

    text = re.sub(r"\b(\d{4})-(\d{2})-(\d{2})\b", date_repl, text)
    text = re.sub(r"\b([A-Z]{1,4})-(\d{3,10})\b",
                  lambda m: " ".join(m.group(1)) + " " + " ".join(m.group(2)), text)
    text = re.sub(r"\b\d{4}\b",
                  lambda m: m.group(0) if 1900 <= int(m.group(0)) <= 2099 else " ".join(m.group(0)),
                  text)
    return text


def tts_8k(text: str) -> bytes:
    text = _speakable(_md_cleanup.sub("", text).strip())
    if not text:
        return b""
    if _piper_voice is not None:
        pcm, rate = _synth_inprocess(text)
    else:
        pcm, rate = _synth_subprocess(text)
    audio = np.frombuffer(pcm, dtype=np.int16)
    return resample_poly(audio, up=8000, down=rate).astype(np.int16).tobytes()


# ----------------------------- STT -----------------------------
def to_whisper_input(pcm16: bytes) -> np.ndarray:
    audio = np.frombuffer(pcm16, dtype=np.int16)
    return (resample_poly(audio, up=2, down=1) / 32768.0).astype(np.float32)


def capture_utterance(call, audio_in: collections.deque) -> bytes | None:
    speech = bytearray()
    preroll = bytearray()
    preroll_max = int(PREROLL_S / CHUNK_S) * CHUNK_BYTES
    silence_chunks = 0
    while call.state == CallState.ANSWERED:
        if not audio_in:
            time.sleep(0.005)
            continue
        chunk = audio_in.popleft()
        rms = audioop.rms(chunk, 2)
        if rms >= VAD_THRESHOLD:
            if not speech:
                speech.extend(preroll)
            speech.extend(chunk)
            silence_chunks = 0
        elif speech:
            speech.extend(chunk)
            silence_chunks += 1
            if silence_chunks * CHUNK_S >= END_SILENCE_S:
                voiced = len(speech) / 2 / 8000 - END_SILENCE_S - PREROLL_S
                if voiced >= MIN_SPEECH_S:
                    return bytes(speech)
                speech = bytearray()
                silence_chunks = 0
        else:
            preroll.extend(chunk)
            if len(preroll) > preroll_max:
                del preroll[:len(preroll) - preroll_max]
    return None


# ----------------------------- LLM turn with sentence streaming -----------------------------
SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


def make_typing_clip(duration_s: float = 1.4) -> bytes:
    """Synthetic keyboard clatter (quiet, low-pass shaped), 8 kHz s16 mono."""
    rate = 8000
    n = int(duration_s * rate)
    out = np.zeros(n, dtype=np.float64)
    kernel = np.exp(-np.linspace(0.0, 3.0, 20))          # soft low-pass
    t = 0.05
    while t < duration_s - 0.08:
        length = int(rate * random.uniform(0.006, 0.016))
        i = int(t * rate)
        noise = np.convolve(np.random.randn(length), kernel)[:length]
        click = noise * np.exp(-np.linspace(0, 5.5, length))
        # quiet "thock" component (the key bottoming out)
        thump_len = min(length, int(rate * 0.008))
        thump = np.sin(2 * np.pi * random.uniform(130, 220) *
                       np.arange(thump_len) / rate) * np.exp(-np.linspace(0, 6, thump_len))
        click[:thump_len] += thump * 0.6
        out[i:i + length] += click * random.uniform(0.35, 0.9)
        t += random.uniform(0.06, 0.19)
        if random.random() < 0.12:      # short thinking pause while "typing"
            t += random.uniform(0.15, 0.35)
    peak = np.abs(out).max() or 1.0
    out = out / peak * 0.16             # quiet, background character
    return (out * 32767).astype(np.int16).tobytes()


def llm_speak_turn(messages: list, say, play_typing, fallback_text: str) -> str:
    """Run one dialog turn; returns the reply text that was spoken last."""
    spoken: set = set()
    nudges = 0

    def say_once(sentence: str):
        key = sentence.strip().lower()
        if key and key not in spoken:
            spoken.add(key)
            say(sentence.strip())

    while True:
        buf = ""
        full = ""
        tool_calls = []
        for part in chat(MODEL, messages=messages, tools=list(TOOLS.values()),
                         think=False, stream=True, keep_alive=-1):
            m = part.message
            if m.content:
                buf += m.content
                full += m.content
                pieces = SENTENCE_END.split(buf)
                for sentence in pieces[:-1]:
                    say_once(sentence)
                buf = pieces[-1]
            if m.tool_calls:
                tool_calls.extend(m.tool_calls)
        if buf.strip():
            say_once(buf)

        assistant_msg = {"role": "assistant", "content": full}
        if tool_calls:
            assistant_msg["tool_calls"] = [
                {"function": {"name": tc.function.name, "arguments": dict(tc.function.arguments)}}
                for tc in tool_calls
            ]
        messages.append(assistant_msg)

        if not tool_calls:
            text = full.strip()
            if text:
                print(f"AGENT: {text}")
            # Safety net: action announced but no tool called -> push the model
            if (nudges < 2 and text and len(text) < 90 and "?" not in text
                    and re.search(r"\b(moment|check|verify|look|hold on|checking)\b", text, re.I)):
                nudges += 1
                print("   [nudge: model announced an action but called no tool]")
                messages.append({"role": "system", "content":
                    "You announced an action but did not call any tool. "
                    "Call the appropriate tool NOW. Output no text."})
                continue
            # Anti dead air: turn ended without speech and without a tool -> speak the fallback
            if not text and not spoken:
                print("   [fallback: empty model reply]")
                say_once(fallback_text)
                return fallback_text
            return text

        if not spoken:
            say_once(random.choice(CFG_L["fillers"]))
        play_typing()
        for tc in tool_calls:
            fname = tc.function.name
            args = dict(tc.function.arguments)
            fn = TOOLS.get(fname)
            try:
                result = fn(**args) if fn else f"ERROR: unknown tool {fname}"
            except Exception as e:
                result = f"ERROR: {e}"
            print(f"   [{fname}({json.dumps(args, ensure_ascii=False)}) -> {result}]")
            messages.append({"role": "tool", "tool_name": fname,
                             "content": json.dumps(result, ensure_ascii=False)})


# ----------------------------- Startup -----------------------------
print("Loading Whisper ...")
whisper = WhisperModel("large-v3-turbo", device="cuda", compute_type="int8_float16")
_load_piper()
print("Warming up LLM (VRAM load) ...")
t0 = time.time()
chat(MODEL, messages=[{"role": "user", "content": "Reply with OK only."}], think=False, keep_alive=-1)
print(f"LLM ready ({time.time()-t0:.1f}s).")
GREETING_TEXT = CFG_L["greeting"]
GREETING_8K = tts_8k(GREETING_TEXT)
TYPING_CLIPS = [make_typing_clip(1.2), make_typing_clip(1.6), make_typing_clip(1.4)]
print("Ready.")


def handle_call(call):
    print("\n=== New call ===")
    bank_tools.reset_session()          # fresh identity state per call!
    messages = [{"role": "system", "content": build_system_prompt()},
                {"role": "assistant", "content": GREETING_TEXT}]

    audio_in: collections.deque = collections.deque()
    speak_q: "queue.Queue[bytes|None]" = queue.Queue()
    playing_until = [0.0]
    turn_t0 = [0.0]
    first_audio_reported = [True]

    def reader():
        next_t = time.time()
        while call.state == CallState.ANSWERED:
            next_t += CHUNK_S
            try:
                chunk = call.read_audio(length=CHUNK_BYTES, blocking=False)
            except Exception:
                break
            audio_in.append(chunk)
            time.sleep(max(0.0, next_t - time.time()))

    def speaker():
        while True:
            item = speak_q.get()
            if item is None:
                break
            if item and call.state == CallState.ANSWERED:
                call.write_audio(item)
                dur = len(item) / 2 / 8000
                playing_until[0] = max(playing_until[0], time.time()) + dur
            speak_q.task_done()

    def say(text: str):
        audio = tts_8k(text)
        if audio:
            if not first_audio_reported[0]:
                first_audio_reported[0] = True
                print(f"   [first audio after {time.time()-turn_t0[0]:.1f}s]")
            speak_q.put(audio)

    def play_typing():
        speak_q.put(random.choice(TYPING_CLIPS))

    try:
        call.answer()
        print(f">> answered, call.state = {call.state}")
        threading.Thread(target=reader, daemon=True).start()
        threading.Thread(target=speaker, daemon=True).start()
        time.sleep(0.3)
        print(f">> before greeting, call.state = {call.state}")
        speak_q.put(GREETING_8K)
        playing_until[0] = time.time() + len(GREETING_8K) / 2 / 8000
        user_turns = 0
        last_reply = ""

        while call.state == CallState.ANSWERED:
            speak_q.join()
            wait = playing_until[0] - time.time()
            if wait > 0:
                time.sleep(wait + 0.2)
            audio_in.clear()

            utt = capture_utterance(call, audio_in)
            if utt is None:
                break
            turn_t0[0] = time.time()
            first_audio_reported[0] = False
            segments, _ = whisper.transcribe(to_whisper_input(utt), language=CFG_L["whisper"],
                                             vad_filter=False, beam_size=1)
            text = " ".join(s.text.strip() for s in segments).strip()
            if not text:
                continue
            print(f"CALLER: {text}")
            user_turns += 1
            messages.append({"role": "user", "content": text})
            fb = CFG_L["fallback_start"] if user_turns <= 1 else CFG_L["fallback"]
            reply = llm_speak_turn(messages, say, play_typing, fb)
            # Repeat breaker: reply word-for-word identical to the last turn -> regenerate
            if reply and reply == last_reply:
                print("   [repeat-breaker: identical reply, regenerating the turn]")
                messages.append({"role": "system", "content":
                    "You just repeated your previous reply verbatim. Do not repeat it. "
                    "React to the caller's latest message: acknowledge the new information "
                    "and take the next step (ask for the next missing detail or call a tool)."})
                reply = llm_speak_turn(messages, say, play_typing, fb)
            last_reply = reply
        print(f"=== Call ended (state={call.state}) ===")
    except InvalidStateError as e:
        print(f"=== Call aborted (InvalidStateError: {e}) ===")
    except Exception:
        import traceback
        print("=== Call crashed with an exception: ===")
        traceback.print_exc()
    finally:
        speak_q.put(None)


if __name__ == "__main__":
    cfg = None
    for p in CONFIG_CANDIDATES:
        if p.exists():
            c = configparser.ConfigParser()
            c.read(p, encoding="utf-8")
            cfg = c["sip"]
            break
    if cfg is None:
        sys.exit("No config.ini found.")

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect((cfg["server"], 5060))
    my_ip = s.getsockname()[0]
    s.close()

    phone = VoIPPhone(cfg["server"], int(cfg.get("port", 5060)),
                      cfg["user"], cfg["password"],
                      callCallback=handle_call, myIP=my_ip)
    phone.start()
    print(f"\nAgent reachable as {cfg['user']}@{cfg['server']}. Enter quits.")
    try:
        input()
    except (KeyboardInterrupt, EOFError):
        pass
    print("Shutting down ...")
    t = threading.Thread(target=phone.stop, daemon=True)
    t.start()
    t.join(timeout=5)
    os._exit(0)
