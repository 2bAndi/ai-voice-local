r"""Local phone agent — use case: CREDIT CARD BLOCKING (English).

Pipeline identical to the appointment demo (V3): FRITZ!Box/pyVoIP(16-bit patch)
-> reader thread + energy VAD -> faster-whisper -> Qwen3 (streaming, tools)
-> Piper -> phone. Only the dialog layer and tools changed.

Dialog flow enforced by code-level guardrails in agent/bank_tools.py:
  reason -> identify (name + DOB) -> security question -> card selection
  -> explicit confirmation -> block -> reference number -> optional replacement.

Prerequisites: Ollama service with the configured LLM (default qwen3.8:27b) and
the Piper voice for the configured language (setup.ps1 downloads both).
Model / language selection lives in config.ini, see agent/settings.py.
Start:  python agent\call_agent.py
"""
import collections
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

from agent import settings  # noqa: E402  (registers the CUDA DLL dirs first)
from agent import events  # noqa: E402  (Glass Box event emitter)
import voip16  # noqa: F401,E402
from pyVoIP.VoIP import VoIPPhone, CallState, InvalidStateError  # noqa: E402
from ollama import chat  # noqa: E402
from agent import gateway  # noqa: E402  (tool gateway: confirm gate + WORM + bank mock)
from agent import recorder  # noqa: E402  (per-call stereo WAV for the Glass Box player)
from agent import stt_stream  # noqa: E402  (streaming STT: 20 ms slices in, partials out)

MODEL = settings.LLM_MODEL

# ----- Language: "en" | "de" | "da" (prompt language + voice + STT) -----
# NOTE ON LANGUAGE: the project language is English — code, comments, prompts and
# logs are English throughout. The tables below (LANG_CFG, MONTHS) are the ONLY
# intentionally non-English literals in this repository: they are runtime voice
# output spoken to a caller in their own language, i.e. data, not documentation.
# Do not "translate" them — set language = en|de|da in config.ini [agent].
LANGUAGE = settings.LANGUAGE
LANG_CFG = {
    "en": {"voice": "en_US-lessac-high", "whisper": "en", "reply_lang": "English",
           "greeting": "Hello, you have reached the card security line. You are speaking with an automated assistant; a human colleague is available at any time if you ask. How can I help you?",
           "fallback": "I am sorry, I did not catch that. Could you say that again, please?",
           "fallback_start": "I am sorry, I did not catch that. Do you want to block your credit card?",
           "fillers": ["One moment, please.", "Just a second, please.", "Let me take care of that.", "Bear with me for a moment."]},
    "de": {"voice": "de_DE-thorsten-high", "whisper": "de", "reply_lang": "German",
           "greeting": "Guten Tag, hier ist die Kartensperr-Hotline. Sie sprechen mit einem automatischen Assistenten; auf Wunsch verbinde ich Sie jederzeit mit einem Mitarbeiter. Wie kann ich Ihnen helfen?",
           "fallback": "Entschuldigung, das habe ich nicht verstanden. Koennen Sie das bitte wiederholen?",
           "fallback_start": "Entschuldigung, das habe ich nicht verstanden. Moechten Sie Ihre Kreditkarte sperren?",
           "fillers": ["Einen kleinen Moment bitte.", "Eine Sekunde, ich kuemmere mich darum.", "Augenblick bitte."]},
    "da": {"voice": "da_DK-talesyntese-medium", "whisper": "da", "reply_lang": "Danish",
           "greeting": "Goddag, De har ringet til kortspærringslinjen. De taler med en automatisk assistent; De kan altid bede om en medarbejder. Hvordan kan jeg hjælpe?",
           "fallback": "Undskyld, det fangede jeg ikke. Kan De gentage det?",
           "fallback_start": "Undskyld, det fangede jeg ikke. Oensker De at spaerre Deres kreditkort?",
           "fillers": ["Et øjeblik.", "Lige et øjeblik, tak."]},
}
CFG_L = LANG_CFG[LANGUAGE]
VOICE_NAME = CFG_L["voice"]
VOICE_ONNX = PROJECT / "voices" / f"{VOICE_NAME}.onnx"
LANG_DETECT_MIN_PROB = 0.6      # whisper language probability needed to switch
# Vocabulary the streaming decoder is primed with (whisper initial_prompt), plus the agent's
# last sentence - short telephone answers ("Visa", "Smith") are otherwise easy to mishear.
STT_VOCAB = {"en": "Demo Bank card security line. Visa, Mastercard, credit card, lost, stolen, fraud, damaged, blocked, yes, no, correct.",
             "de": "Demo Bank Kartensperr-Hotline. Visa, Mastercard, Kreditkarte, verloren, gestohlen, Betrug, beschaedigt, gesperrt, ja, nein, richtig.",
             "da": "Demo Bank kortspaerring. Visa, Mastercard, kreditkort, mistet, stjaalet, svindel, beskadiget, spaerret, ja, nej, korrekt."}
LANG_DETECT_TURNS = 2           # try on the first N caller utterances, stop after a switch


def voice_available(lang: str) -> bool:
    return (PROJECT / "voices" / f"{LANG_CFG[lang]['voice']}.onnx").exists()


def set_language(lang: str) -> None:
    """Rebind the per-call language: prompt language, Piper voice, STT language, spoken formats.
    Every function reads these module globals at call time, so a rebind takes effect at once."""
    global LANGUAGE, CFG_L, VOICE_NAME, VOICE_ONNX, _piper_voice
    LANGUAGE = lang
    CFG_L = LANG_CFG[lang]
    VOICE_NAME = CFG_L["voice"]
    VOICE_ONNX = PROJECT / "voices" / f"{VOICE_NAME}.onnx"
    _piper_voice = _piper_cache.get(VOICE_NAME)
END_SILENCE_S = 0.8
MIN_SPEECH_S = 0.15
PREROLL_S = 0.3
CHUNK_BYTES = 320
CHUNK_S = 0.02
VAD_THRESHOLD = 160          # RMS of 20 ms s16 frames; quiet callers sat below the old 250
LEVEL_REPORT_S = 2.0         # while listening, report the peak level this often (Glass Box)

TOOLS = gateway.build_tools()


def build_system_prompt(translated_input: bool = False) -> str:
    now = dt.datetime.now()
    note = (" The caller speaks another language; their words reach you machine-translated to English."
            if translated_input else "")
    return f"""You are the automated card security line of Demo Bank. Your only job is to help
callers BLOCK a credit card (and optionally order a replacement).
You MUST reply exclusively in {CFG_L['reply_lang']}.{note} Today is {now.strftime('%A, %d %B %Y')}.

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
5. When card AND reason are known, call block_card(card_last4, reason) ONCE without
   caller_reply. It does not block yet: it returns what to read back. Read card and
   reason back in ONE sentence and ask "Is that correct?".
6. Call block_card AGAIN with the same card_last4 and reason plus caller_reply = the
   caller's EXACT words. The gate decides. If the result says NOT CONFIRMED, do what it
   says (ask for a clear yes or no, or accept that nothing is changed). Never claim the
   card is blocked unless the result starts with SUCCESS.
7. Read the reference number back exactly as the tool result spells it. Replacement
   cards are NOT ordered on this line: tell the caller a colleague or the mobile bank can
   arrange one, then ask whether there is anything else.
8. If the caller asks for a human, if verification fails three times, if a tool reports
   a failure you cannot resolve, or if the request is outside card blocking: call
   escalate(reason) and tell the caller you are connecting them to a colleague.

Hard rules:
- Reply in at most 1-2 short sentences; they are read out loud on the phone. Ask only
  ONE question at a time.
- NEVER ask for a full card number. Only the last four digits are ever used.
- Say all numbers, dates and digits in words when speaking.
- NEVER invent information the caller did not provide, and NEVER claim a card is
  blocked, verified, or a replacement ordered unless the tool result says SUCCESS.
  If a tool returns ERROR, follow the instruction inside the error message.
- NEVER say an answer "did not match", "is verified", "is blocked" or "is confirmed"
  unless you called the corresponding tool in THIS turn and are quoting its result. If
  the caller's reply is unclear (for example a noise or a fragment), ask them to repeat
  or spell it - do not judge it yourself.
- Confirmation is decided by the gate inside block_card, not by you. Do not treat "hmm",
  silence, a question or a new request as a yes - pass the words on and let it decide.
- Call tools directly WITHOUT any announcement text. A waiting message is played
  automatically while tools run. NEVER say things like "one moment, I will check" -
  every reply either asks the caller a question, states a tool result, or is a tool call.
- Ask for the date of birth in natural language. NEVER ask the caller to use a format
  like YYYY-MM-DD - you convert their spoken date yourself and read it back in words.
  Never state facts before the tool result.
- If verification fails three times, apologize and call escalate; do not continue.
- If the caller asks for anything other than blocking a card, politely say this line
  only handles card blocking and offer to connect them to a colleague.
- Treat the situation with calm urgency; a caller reporting a stolen card may be
  stressed. Be reassuring but efficient."""


# ----------------------------- TTS: Piper in-process -----------------------------
_piper_voice = None
_piper_cache: dict = {}      # voice name -> loaded PiperVoice (every installed language, for switching)


def _load_piper():
    global _piper_voice
    if not voice_available(LANGUAGE):
        sys.exit(f"Piper voice for language '{LANGUAGE}' is missing: {VOICE_ONNX}\n"
                 f"Download it once (venv active, project folder):\n"
                 f"    python -m piper.download_voices --download-dir voices {VOICE_NAME}\n"
                 f"or set  language = en | de  in config.ini.")
    try:
        from piper import PiperVoice
        for lang, cfg in LANG_CFG.items():
            if voice_available(lang):
                _piper_cache[cfg["voice"]] = PiperVoice.load(str(PROJECT / "voices" / f"{cfg['voice']}.onnx"))
        _piper_voice = _piper_cache[VOICE_NAME]
        _ = _synth_inprocess("Test")
        print(f"Piper: loaded in-process: {', '.join(_piper_cache)}")
    except Exception as e:
        _piper_voice = None
        _piper_cache.clear()
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
            [sys.executable, "-m", "piper", "-m", VOICE_NAME, "--data-dir", str(VOICE_ONNX.parent),
             "-f", str(out), "--", text],
            cwd=PROJECT, check=True, capture_output=True,
        )
        with wave.open(str(out), "rb") as w:
            return w.readframes(w.getnframes()), w.getframerate()


_md_cleanup = re.compile(r"[*_#`]|</?[a-zA-Z]+>")     # markdown marks and any HTML/XML-like tag

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
    t0 = time.time()
    if _piper_voice is not None:
        pcm, rate = _synth_inprocess(text)
    else:
        pcm, rate = _synth_subprocess(text)
    audio = np.frombuffer(pcm, dtype=np.int16)
    out = resample_poly(audio, up=8000, down=rate).astype(np.int16).tobytes()
    events.emit("SP-C→SP-B", "tts.synth", {"chars": len(text), "src_rate": rate, "out_rate": 8000,
                                          "audio_ms": round(len(out) / 16), "voice": VOICE_NAME,
                                          "language": LANGUAGE,
                                          "engine": "piper in-process" if _piper_voice is not None else "piper subprocess",
                                          "text": text[:120]},
                ms=(time.time() - t0) * 1000)
    return out


# ----------------------------- STT -----------------------------
def to_whisper_input(pcm16: bytes) -> np.ndarray:
    audio = np.frombuffer(pcm16, dtype=np.int16)
    return (resample_poly(audio, up=2, down=1) / 32768.0).astype(np.float32)


def capture_utterance(call, audio_in: collections.deque) -> bytes | None:
    speech = bytearray()
    preroll = bytearray()
    preroll_max = int(PREROLL_S / CHUNK_S) * CHUNK_BYTES
    silence_chunks = 0
    peak, last_report = 0, time.time()
    while call.state == CallState.ANSWERED:
        if not audio_in:
            time.sleep(0.005)
            continue
        chunk = audio_in.popleft()
        rms = audioop.rms(chunk, 2)
        peak = max(peak, rms)
        if time.time() - last_report >= LEVEL_REPORT_S:
            events.emit("SP-A→SP-B", "vad.level", {"peak_rms": peak, "threshold": VAD_THRESHOLD,
                                                   "speech": bool(speech)})
            peak, last_report = 0, time.time()
        if rms >= VAD_THRESHOLD:
            if not speech:
                speech.extend(preroll)
                events.emit("SP-A→SP-B", "vad.start", {"rms": rms, "threshold": VAD_THRESHOLD})
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
# Qwen occasionally emits a stray <think>/</think> even with think=False - never speak or keep it
THINK_TAG = re.compile(r"</?think>")
# Phrases that assert a tool result (verification / block / confirmation) - only legitimate
# in a turn that actually called a tool.
CLAIM_WITHOUT_TOOL = re.compile(
    r"(did(n't| not) match|is (now )?(verified|blocked|confirmed)|has been (verified|blocked)|"
    r"(verification|answer) (failed|was (in)?correct)|(nicht|stimmt) (überein|korrekt)|"
    r"ist (jetzt )?(gesperrt|verifiziert)|er (nu )?(spærret|verificeret|bekræftet)|matchede ikke)", re.I)


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


class Cancelled(Exception):
    """A speculative turn was discarded because the final transcript differed."""


class Speculation:
    """A speculative LLM turn started on a stable partial transcript while the caller pauses.
    Nothing it produces reaches the caller until confirm(): sentences are held, tool calls wait.
    If the final transcript differs, cancel() aborts the turn and everything held is dropped."""

    def __init__(self, text: str, messages: list):
        self.text = text
        self.messages = messages            # private copy of the dialog, adopted on confirm
        self.confirmed = threading.Event()
        self.cancelled = threading.Event()
        self.result = None
        self._pending: list[str] = []
        self._lock = threading.Lock()
        self.t0 = time.time()

    @staticmethod
    def norm(t: str) -> str:
        return re.sub(r"[^\w]+", " ", t).strip().lower()

    def matches(self, final: str) -> bool:
        return self.norm(final) == self.norm(self.text)

    def say(self, sentence: str, say):
        with self._lock:
            if self.confirmed.is_set():
                say(sentence)
            elif not self.cancelled.is_set():
                self._pending.append(sentence)

    def confirm(self, say):
        with self._lock:
            self.confirmed.set()
            for sentence in self._pending:
                say(sentence)
            self._pending.clear()

    def cancel(self):
        self.cancelled.set()

    def wait(self) -> bool:
        """Block until confirmed (True) or cancelled (False)."""
        while not (self.confirmed.is_set() or self.cancelled.is_set()):
            self.confirmed.wait(0.05)
            if self.cancelled.is_set():
                return False
        return self.confirmed.is_set()


def llm_speak_turn(messages: list, say, play_typing, fallback_text: str, spec: "Speculation | None" = None) -> str:
    """Run one dialog turn; returns the reply text that was spoken last.
    With `spec` the turn is speculative: speech is held and tools wait until spec.confirm()."""
    spoken: set = set()
    nudges = 0
    tools_called_this_turn = False
    held: list = []          # claimed tool results held back until a tool has actually run

    def speak(sentence: str):
        if spec is not None:
            spec.say(sentence, say)
        else:
            say(sentence)

    def say_once(sentence: str):
        key = sentence.strip().lower()
        if not key or key in spoken:
            return
        if held or (not tools_called_this_turn and CLAIM_WITHOUT_TOOL.search(sentence)):
            held.append(sentence.strip())       # do not speak a result nobody produced
            return                              # (and nothing that follows it in this pass)
        spoken.add(key)
        speak(sentence.strip())

    iteration = 0
    while True:
        buf = ""
        full = ""
        tool_calls = []
        iteration += 1
        t_llm = time.time()
        first_token = None
        n_tokens = 0
        events.emit("SP-D→SP-C", "llm.request", {"iteration": iteration, "messages": len(messages),
                                                "model": MODEL, "tools": len(TOOLS), "think": False,
                                                "temperature": settings.OLLAMA_OPTIONS.get("temperature"),
                                                "num_ctx": settings.OLLAMA_OPTIONS.get("num_ctx"),
                                                "last_role": messages[-1]["role"],
                                                "last_message": str(messages[-1].get("content", ""))[:200],
                                                "speculative": bool(spec and not spec.confirmed.is_set())})
        think_stripped = False
        for part in chat(MODEL, messages=messages, tools=list(TOOLS.values()),
                         think=False, stream=True, keep_alive=-1,
                         options=settings.OLLAMA_OPTIONS):
            m = part.message
            n_tokens += 1
            if spec is not None and spec.cancelled.is_set():
                raise Cancelled()
            if first_token is None and (m.content or m.tool_calls):
                first_token = time.time()
                events.emit("SP-C→SP-D", "llm.token.first", {"iteration": iteration},
                            ms=(first_token - t_llm) * 1000)
            content = THINK_TAG.sub("", m.content) if m.content else ""
            if m.content and content != m.content and not think_stripped:
                think_stripped = True
                events.emit("AGENT", "guardrail.think_stripped", {"iteration": iteration})
            if content:
                buf += content
                full += content
                pieces = SENTENCE_END.split(buf)
                for sentence in pieces[:-1]:
                    events.emit("SP-C→SP-D", "llm.sentence", {"text": sentence.strip()})
                    say_once(sentence)
                buf = pieces[-1]
            if m.tool_calls:
                tool_calls.extend(m.tool_calls)
        if buf.strip():
            events.emit("SP-C→SP-D", "llm.sentence", {"text": buf.strip()})
            say_once(buf)
        events.emit("SP-C→SP-D", "llm.done", {"iteration": iteration, "chunks": n_tokens,
                                             "tool_calls": [tc.function.name for tc in tool_calls],
                                             "text": full.strip()[:300]},
                    ms=(time.time() - t_llm) * 1000)

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
                print(f"AGENT{' (speculative, held)' if spec is not None and not spec.confirmed.is_set() else ''}: {text}")
            # Safety net 1: action announced but no tool called -> push the model
            if (nudges < 2 and text and len(text) < 90 and "?" not in text
                    and re.search(r"\b(moment|check|verify|look|hold on|checking|augenblick|prüfe|überprüfe|schaue|øjeblik|tjekker|kontrollerer)\b", text, re.I)):
                nudges += 1
                print("   [nudge: model announced an action but called no tool]")
                instr = ("You announced an action but did not call any tool. "
                         "Call the appropriate tool NOW. Output no text.")
                events.emit("AGENT", "guardrail.nudge", {"text": text, "type": "announced_no_tool",
                                                         "instruction": instr})
                messages.append({"role": "system", "content": instr})
                continue
            # Safety net 2: a tool RESULT was claimed in a turn without any tool call
            # (FR-COM-004: no fabricated facts). Observed: "that answer didn't match" without
            # verify_identity. The claim is logged and the model has to redo the turn properly.
            if (nudges < 2 and text and not tools_called_this_turn
                    and CLAIM_WITHOUT_TOOL.search(text)):
                nudges += 1
                held.clear()
                print("   [nudge: model claimed a tool result without calling a tool]")
                instr = ("You stated a verification, blocking or confirmation result, but you did not "
                         "call any tool in this turn. You may not judge that yourself. Either call the "
                         "appropriate tool with the caller's words now, or ask the caller to repeat or "
                         "spell their answer. Output no claims.")
                events.emit("AGENT", "guardrail.nudge", {"text": text, "type": "claimed_without_tool",
                                                         "instruction": instr})
                messages.append({"role": "system", "content": instr})
                continue
            if held:
                # nudges exhausted and the claim is still there: never speak it, ask again instead
                print(f"   [suppressed unverified claim: {held}]")
                events.emit("AGENT", "guardrail.suppressed_claim", {"held": held})
                held.clear()
                say_once(fallback_text)
                return fallback_text
            # Anti dead air: turn ended without speech and without a tool -> speak the fallback
            if not text and not spoken:
                print("   [fallback: empty model reply]")
                events.emit("AGENT", "guardrail.fallback", {"text": fallback_text})
                say_once(fallback_text)
                return fallback_text
            events.emit("AGENT", "guardrail.pass", {
                "checks": ["think_tag_strip", "announced_no_tool", "claimed_without_tool",
                           "held_claims", "dead_air"],
                "tools_called_this_turn": tools_called_this_turn, "nudges": nudges,
                "iterations": iteration})
            return text

        if spec is not None and not spec.confirmed.is_set():
            # tools change the world: a speculative turn may not call them before the transcript is final
            events.emit("AGENT", "llm.speculative.wait", {"reason": "tool call pending", "tools": [tc.function.name for tc in tool_calls]})
            if not spec.wait():
                raise Cancelled()
        if not spoken:
            say_once(random.choice(CFG_L["fillers"]))
        play_typing()
        tools_called_this_turn = True
        held.clear()                 # the model will restate after the real tool result
        for tc in tool_calls:
            fname = tc.function.name
            args = dict(tc.function.arguments)
            fn = TOOLS.get(fname)
            events.emit("SP-D→GW", "tool.call", {"tool": fname, "args": args})
            t_tool = time.time()
            try:
                result = fn(**args) if fn else f"ERROR: unknown tool {fname}"
            except Exception as e:
                result = f"ERROR: {e}"
            ok = not str(result).startswith("ERROR")
            events.emit("GW→SP-D", "tool.result", {"tool": fname, "ok": ok, "result": str(result)[:300]},
                        ms=(time.time() - t_tool) * 1000)
            print(f"   [{fname}({json.dumps(args, ensure_ascii=False)}) -> {result}]")
            messages.append({"role": "tool", "tool_name": fname,
                             "content": json.dumps(result, ensure_ascii=False)})


# ----------------------------- Startup -----------------------------
print(settings.describe())
print(f"Loading Whisper {settings.STT_MODEL} ({settings.STT_COMPUTE}) ...")
whisper = settings.load_whisper()
try:
    import sherpa_onnx  # noqa: F401 - Silero VAD for both STT backends (and the sherpa recogniser)
except ImportError:
    sys.exit("The streaming STT needs the sherpa-onnx package (Silero VAD). In the venv run:\n"
             "    pip install sherpa-onnx\n(or re-run setup.ps1), then start the agent again.")
if not (PROJECT / settings.VAD_MODEL).exists():
    sys.exit(f"VAD model missing: {PROJECT / settings.VAD_MODEL} - run setup.ps1 (downloads models\\silero_vad.onnx).")
_stt_engines: dict = {}


def get_stt_engine(name: str):
    """STT backends are created once and kept (the STT swap point); raises if a backend is unavailable."""
    if name not in _stt_engines:
        if name == "sherpa":
            _stt_engines[name] = stt_stream.SherpaStream(PROJECT / settings.SHERPA_MODEL, PROJECT / settings.VAD_MODEL)
        elif name == "whisper":
            _stt_engines[name] = stt_stream.WhisperStream(whisper, PROJECT / settings.VAD_MODEL, LANG_CFG[LANGUAGE]["whisper"],
                                                          beam=settings.STT_BEAM, interval=settings.STT_INTERVAL)
        else:
            raise ValueError(f"unknown STT backend '{name}'")
    return _stt_engines[name]


try:
    stt_engine = get_stt_engine(settings.STT_BACKEND)
except Exception as e:  # noqa: BLE001 - e.g. sherpa model missing
    print(f"Streaming STT backend '{settings.STT_BACKEND}' unavailable ({e}) - using whisper stream.")
    stt_engine = get_stt_engine("whisper")

# ----- Runtime controls: what the next call uses. Changed from the Glass Box page (POST /api/control)
# without restarting the agent; config.ini only provides the start values.
RUNTIME = {"language": settings.LANGUAGE, "stt_backend": stt_engine.name, "speculative": settings.SPECULATIVE,
           "endpoint_ms": settings.ENDPOINT_MS, "language_detect": settings.LANGUAGE_DETECT,
           "canonical_english": settings.CANONICAL_ENGLISH}
_runtime_lock = threading.Lock()


def control_options() -> dict:
    return {"language": [l for l in LANG_CFG if voice_available(l)], "stt_backend": ["whisper", "sherpa"],
            "endpoint_ms": [400, 600, 800, 1000, 1200], "speculative": [True, False],
            "language_detect": [True, False], "canonical_english": [True, False],
            "voices_missing": [l for l in LANG_CFG if not voice_available(l)]}


def get_control() -> dict:
    with _runtime_lock:
        return {"runtime": dict(RUNTIME), "options": control_options(), "applies": "next call"}


def apply_control(changes: dict) -> dict:
    """Validate and apply control changes; returns {ok, error?, runtime}."""
    errors = []
    new = {}
    for k, v in (changes or {}).items():
        if k == "language":
            if v not in LANG_CFG:
                errors.append(f"unknown language '{v}'")
            elif not voice_available(v):
                errors.append(f"no Piper voice for '{v}' - python -m piper.download_voices --download-dir voices {LANG_CFG[v]['voice']}")
            else:
                new[k] = v
        elif k == "stt_backend":
            try:
                get_stt_engine(v)
                new[k] = v
            except Exception as e:  # noqa: BLE001
                errors.append(f"STT backend '{v}' unavailable: {str(e)[:120]}")
        elif k == "endpoint_ms":
            try:
                ms = int(v)
                if not 200 <= ms <= 3000:
                    raise ValueError
                new[k] = ms
            except (TypeError, ValueError):
                errors.append("endpoint_ms must be 200..3000")
        elif k in ("speculative", "language_detect", "canonical_english"):
            new[k] = bool(v) if not isinstance(v, str) else v.lower() in ("1", "true", "yes", "on")
        else:
            errors.append(f"unknown control '{k}'")
    if errors:
        return {"ok": False, "error": "; ".join(errors), **get_control()}
    with _runtime_lock:
        RUNTIME.update(new)
    if new:
        print(f"Controls updated from the Glass Box: {new}  (next call)")
        events.emit("AGENT", "control.changed", {**new, "applies": "next call"})
    return {"ok": True, **get_control()}


stt_stream.ENDPOINT_MS = settings.ENDPOINT_MS
print(f"Streaming STT: {stt_engine.name} (endpoint after {settings.ENDPOINT_MS} ms pause, "
      f"speculative LLM start {'on' if settings.SPECULATIVE else 'off'}, "
      f"canonical English {'on' if settings.CANONICAL_ENGLISH else 'off'})")
_load_piper()
print(f"Warming up LLM {MODEL} (VRAM load + prompt cache for system prompt and tool schemas) ...")
t0 = time.time()
chat(MODEL, messages=[{"role": "system", "content": build_system_prompt()},
                      {"role": "user", "content": "Reply with OK only."}],
     tools=list(TOOLS.values()), think=False, keep_alive=-1, options=settings.OLLAMA_OPTIONS)
print(f"LLM ready ({time.time()-t0:.1f}s).")
events.emit("AGENT", "agent.models", {"llm": MODEL, "stt": settings.STT_MODEL, "stt_backend": stt_engine.name,
                                      "stt_compute": settings.STT_COMPUTE, "stt_beam": settings.STT_BEAM,
                                      "tts_voice": VOICE_NAME, "language": LANGUAGE,
                                      "llm_warmup_s": round(time.time() - t0, 1)})
_greeting_cache: dict = {}


def greeting_audio(lang: str) -> bytes:
    """Greeting audio per language, synthesised once (the language can change between calls)."""
    if lang not in _greeting_cache:
        _greeting_cache[lang] = tts_8k(LANG_CFG[lang]["greeting"])
    return _greeting_cache[lang]


greeting_audio(LANGUAGE)      # warm the configured language now, others on first use
TYPING_WAV = PROJECT / "audio" / "typing_8k.wav"


def load_typing_clips(n: int = 6) -> list[bytes]:
    """Random 1.2-1.6 s slices from a real typing recording (tests\record_typing.py),
    with a short fade in/out so cuts do not click. Falls back to the synthetic clip."""
    if not TYPING_WAV.exists():
        print("Typing sound: synthetic (record a real one with tests\\record_typing.py).")
        return [make_typing_clip(1.2), make_typing_clip(1.6), make_typing_clip(1.4)]
    with wave.open(str(TYPING_WAV), "rb") as w:
        assert w.getframerate() == 8000 and w.getnchannels() == 1 and w.getsampwidth() == 2
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
    total = len(pcm)
    clips = []
    fade = int(0.03 * 8000)
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    for _ in range(n):
        length = int(random.uniform(1.2, 1.6) * 8000)
        if length >= total:
            seg = pcm.copy()
        else:
            start = random.randint(0, total - length)
            seg = pcm[start:start + length].copy()
        seg[:fade] *= ramp
        seg[-fade:] *= ramp[::-1]
        clips.append(seg.astype(np.int16).tobytes())
    print(f"Typing sound: {TYPING_WAV.name} ({total/8000:.1f}s recording, {n} slices).")
    return clips


TYPING_CLIPS = load_typing_clips()

# Glass Box page (Phase 2): served from this process so it sees the live events.
if settings.GLASSBOX_PORT:
    try:
        from glassbox import server as glassbox_server
        glassbox_server.set_control_handlers(get_control, apply_control)
        glassbox_server.start_in_background(settings.GLASSBOX_PORT)
        print(f"Glass Box: http://{socket.gethostbyname(socket.gethostname())}:{settings.GLASSBOX_PORT}  "
              f"(any browser in the LAN)")
    except Exception as e:  # noqa: BLE001
        print(f"Glass Box not started ({e}) - pip install fastapi uvicorn websockets")
print("Ready.")


def handle_call(call):
    print("\n=== New call ===")
    hdr = getattr(getattr(call, "request", None), "headers", {}) or {}
    corr = events.start_call(sip_call_id=getattr(call, "call_id", None),
                             sip_from=str(hdr.get("From", {}).get("raw", hdr.get("From", "")))[:120],
                             sip_to=str(hdr.get("To", {}).get("raw", hdr.get("To", "")))[:120],
                             language=LANGUAGE)
    print(f"correlation_id={corr}")
    gateway.reset_call()                # fresh identity + gate state per call!
    with _runtime_lock:
        rt = dict(RUNTIME)               # controls as set for this call (Glass Box or config.ini)
    set_language(rt["language"])        # a previous call may have switched the language
    stt_engine = get_stt_engine(rt["stt_backend"])
    stt_stream.ENDPOINT_MS = rt["endpoint_ms"]
    speculative = rt["speculative"]
    canonical = rt["canonical_english"]
    detect_langs = [l for l in LANG_CFG if voice_available(l)] if rt["language_detect"] else []
    switched = False
    greeting_text = CFG_L["greeting"]
    greeting_8k = greeting_audio(LANGUAGE)
    system_prompt = build_system_prompt(translated_input=canonical and LANGUAGE != "en")
    messages = [{"role": "system", "content": system_prompt},
                {"role": "assistant", "content": greeting_text}]
    # What the LLM is told (SP-D hands the prompt to SP-C) - the sequence shows the rules, the
    # detail pane the full text.
    events.emit("SP-D→SP-C", "llm.prompt", {
        "model": MODEL, "chars": len(system_prompt), "reply_language": CFG_L["reply_lang"],
        "tools": sorted(TOOLS), "think": False,
        "rules": ["AI disclosure in the greeting", "one step at a time: reason → name → DOB → security answer",
                  "identity via tools only, no facts before a tool result",
                  "block_card two-call pattern: read back, explicit yes, then execute",
                  "reference number = WORM id, read digit by digit", "escalate on doubt / second action"],
        "guardrails": ["think-tag strip", "announced action without tool → nudge",
                       "claimed tool result without tool → nudge, then suppressed",
                       "dead-air fallback", "repeat breaker"],
        "text": system_prompt})
    # How speech is understood and produced in SP-B/SP-C - language is fixed by config, not detected
    events.emit("SP-B→SP-C", "speech.config", {
        "stt": {"backend": stt_engine.name, "streaming": "20 ms slices, partials while speaking",
                "model": settings.STT_MODEL if stt_engine.name == "whisper" else Path(settings.SHERPA_MODEL).name,
                "compute": settings.STT_COMPUTE, "beam": settings.STT_BEAM,
                "endpoint_ms": rt["endpoint_ms"], "pause_ms": stt_stream.PAUSE_MS,
                "speculative_llm": speculative,
                "canonical_english": canonical,
                "language": CFG_L["whisper"],
                "mode": (f"auto-detect on the first {LANG_DETECT_TURNS} utterances, switchable to {detect_langs}"
                         if detect_langs else "forced by config (no auto-detect)"),
                "vad_threshold": VAD_THRESHOLD},
        "tts": {"voice": VOICE_NAME, "language": LANGUAGE,
                "engine": "piper in-process" if _piper_voice is not None else "piper subprocess",
                "out_rate": 8000},
        "reply_language": CFG_L["reply_lang"]})

    audio_in: collections.deque = collections.deque()
    speak_q: "queue.Queue[bytes|None]" = queue.Queue()
    playing_until = [0.0]
    turn_t0 = [0.0]
    first_audio_reported = [True]
    rec = recorder.CallRecorder(events.call_t0())     # stereo WAV: left caller, right agent
    rec_file = events.call_file()

    def reader():
        next_t = time.time()
        while call.state == CallState.ANSWERED:
            next_t += CHUNK_S
            try:
                chunk = call.read_audio(length=CHUNK_BYTES, blocking=False)
            except Exception:
                break
            audio_in.append(chunk)
            rec.add(0, chunk, time.time())
            time.sleep(max(0.0, next_t - time.time()))

    def speaker():
        while True:
            item = speak_q.get()
            if item is None:
                break
            if item and call.state == CallState.ANSWERED:
                call.write_audio(item)
                dur = len(item) / 2 / 8000
                starts_at = max(playing_until[0], time.time())     # queued behind what is still playing
                rec.add(1, item, starts_at)
                playing_until[0] = starts_at + dur
                events.emit("SP-B→SP-A", "rtp.write", {"audio_ms": round(dur * 1000)})
            speak_q.task_done()

    def say(text: str):
        audio = tts_8k(text)
        if audio:
            if not first_audio_reported[0]:
                first_audio_reported[0] = True
                print(f"   [first audio after {time.time()-turn_t0[0]:.1f}s]")
                events.emit("AGENT", "turn.first_audio", {}, ms=(time.time() - turn_t0[0]) * 1000)
            speak_q.put(audio)

    def play_typing():
        clip = random.choice(TYPING_CLIPS)
        events.emit("SP-B→SP-A", "filler.typing", {"audio_ms": round(len(clip) / 16)})
        speak_q.put(clip)

    try:
        call.answer()
        events.emit("AGENT", "call.answered", {"state": str(call.state)})
        threading.Thread(target=reader, daemon=True).start()
        threading.Thread(target=speaker, daemon=True).start()
        time.sleep(0.3)
        print(f"AGENT: {greeting_text}")
        events.emit("SP-C→SP-B", "tts.greeting", {"text": greeting_text, "audio_ms": round(len(greeting_8k) / 16)})
        speak_q.put(greeting_8k)      # the speaker thread sets playing_until when it starts playback
        user_turns = 0
        last_reply = ""

        while call.state == CallState.ANSWERED:
            speak_q.join()
            wait = playing_until[0] - time.time()
            if wait > 0:
                time.sleep(wait + 0.2)
            audio_in.clear()

            # ---- streaming STT: every 20 ms slice goes straight into the recogniser (SP-B -> SP-C)
            detecting = bool(detect_langs) and not switched and user_turns < LANG_DETECT_TURNS \
                and stt_engine.name == "whisper"
            stt_engine.reset()
            stt_engine.language = None if detecting else CFG_L["whisper"]
            translating = canonical and stt_engine.name == "whisper" and (detecting or LANGUAGE != "en")
            if hasattr(stt_engine, "task"):
                stt_engine.task = "translate" if translating else "transcribe"
            if hasattr(stt_engine, "hint"):
                # the decoder's context is in its OUTPUT language: English when translating
                hint_lang = "en" if translating else LANGUAGE
                stt_engine.hint = (STT_VOCAB.get(hint_lang, "") + " " + (last_reply or greeting_text)[-160:]).strip()
            fb = CFG_L["fallback_start"] if user_turns < 1 else CFG_L["fallback"]
            spec: Speculation | None = None
            spec_thread = None
            final = None
            n_partials = 0
            while call.state == CallState.ANSWERED and final is None:
                if not audio_in:
                    time.sleep(0.005)
                    continue
                chunk = audio_in.popleft()
                for r in stt_engine.feed(chunk):
                    if r.kind == "speech_start":
                        events.emit("SP-A→SP-B", "vad.start", {"backend": r.backend, "vad": "silero"})
                    elif r.kind == "partial":
                        n_partials += 1
                        events.emit("SP-B→SP-C", "stt.partial", {"text": r.text, "stable": r.stable,
                                                                 "backend": r.backend, "n": n_partials,
                                                                 "audio_ms": r.audio_ms},
                                    ms=r.decode_ms)
                        if r.stable and r.text and spec is None and speculative:
                            spec = Speculation(r.text, messages + [{"role": "user", "content": r.text}])
                            events.emit("AGENT", "llm.speculative.start", {"text": r.text, "on": "stable partial (caller paused)"})

                            def _run_spec(sp=spec, fb=fb):
                                try:
                                    sp.result = llm_speak_turn(sp.messages, say, play_typing, fb, sp)
                                except Cancelled:
                                    sp.result = None
                            spec_thread = threading.Thread(target=_run_spec, daemon=True)
                            spec_thread.start()
                    elif r.kind == "endpoint":
                        turn_t0[0] = time.time()
                        first_audio_reported[0] = False
                        events.emit("SP-A→SP-B", "vad.utterance", {"audio_ms": r.audio_ms, "turn": user_turns + 1,
                                                                   "endpoint": "silero vad", "silence_ms": r.silence_ms})
                        events.emit("SP-B→SP-C", "stt.endpoint", {"silence_ms": r.silence_ms, "backend": r.backend})
                    elif r.kind == "final":
                        final = r
                        break
            if final is None:
                if spec:
                    spec.cancel()
                break
            text = final.text.strip()
            switched_this_turn = False
            lang_used = final.language if detecting else CFG_L["whisper"]
            events.emit("SP-B→SP-C", "stt.done", {"text": text, "language": lang_used,
                                                  "language_mode": "auto" if detecting else "forced",
                                                  "translated_to_en": bool(translating and lang_used != "en"),
                                                  "backend": final.backend, "partials": n_partials,
                                                  "model": settings.STT_MODEL if final.backend == "whisper" else "zipformer",
                                                  "beam": settings.STT_BEAM, "audio_ms": final.audio_ms},
                        ms=final.decode_ms)
            if detecting and final.language:
                # SP-B decides how the caller is understood; SP-C follows with voice + reply language
                det, prob = final.language, round(float(final.probability or 0), 2)
                if det == LANGUAGE:
                    decision, reason = "keep", "matches the current language"
                elif det not in LANG_CFG:
                    decision, reason = "keep", f"'{det}' is not a supported language ({sorted(LANG_CFG)})"
                elif det not in detect_langs:
                    decision, reason = "keep", f"no Piper voice installed for '{det}'"
                elif prob < LANG_DETECT_MIN_PROB:
                    decision, reason = "keep", f"probability {prob} below {LANG_DETECT_MIN_PROB}, retry next turn"
                else:
                    decision, reason = f"switch to {det}", f"probability {prob}"
                events.emit("SP-B→SP-C", "stt.language", {"detected": det, "probability": prob,
                                                          "current": LANGUAGE, "decision": decision,
                                                          "reason": reason, "turn": user_turns + 1})
                if decision.startswith("switch"):
                    old_voice, old_lang = VOICE_NAME, LANGUAGE
                    set_language(det)
                    switched = True
                    switched_this_turn = True
                    print(f"   [language switch {old_lang} -> {det} (p={prob})]")
                    events.emit("SP-C→SP-B", "speech.switch", {"from": old_lang, "to": det,
                                                               "voice_from": old_voice, "voice_to": VOICE_NAME,
                                                               "reply_language": CFG_L["reply_lang"],
                                                               "engine": "piper in-process" if _piper_voice is not None else "piper subprocess"})
                    system_prompt = build_system_prompt(translated_input=canonical and LANGUAGE != "en")
                    messages[0] = {"role": "system", "content": system_prompt}
                    events.emit("SP-D→SP-C", "llm.prompt", {
                        "model": MODEL, "chars": len(system_prompt), "reply_language": CFG_L["reply_lang"],
                        "update": f"language switch {old_lang} -> {det}: reply language and formats",
                        "text": system_prompt})
                elif not switched and user_turns + 1 >= LANG_DETECT_TURNS:
                    events.emit("SP-B→SP-C", "stt.language", {"detected": det, "probability": prob,
                                                              "current": LANGUAGE, "decision": "keep, final",
                                                              "reason": f"no confident switch within {LANG_DETECT_TURNS} utterances; "
                                                                        f"language now forced to '{LANGUAGE}'",
                                                              "turn": user_turns + 1})
            if not text:
                if spec:
                    spec.cancel()
                continue
            print(f"CALLER: {text}")
            user_turns += 1
            if spec is not None and spec.matches(text) and not switched_this_turn:
                # the caller's pause was the end of the sentence: the speculative turn is the real one
                events.emit("AGENT", "llm.speculative.confirmed", {"text": text,
                                                                   "head_start_ms": round((time.time() - spec.t0) * 1000)})
                spec.confirm(say)
                spec_thread.join()
                messages[:] = spec.messages
                reply = spec.result if spec.result is not None else llm_speak_turn(messages, say, play_typing, fb)
            else:
                if spec is not None:
                    events.emit("AGENT", "llm.speculative.discarded", {"speculative": spec.text, "final": text,
                                                                       "reason": "language switch" if switched_this_turn else "final differs"})
                    spec.cancel()
                    spec_thread.join(timeout=10)
                messages.append({"role": "user", "content": text})
                reply = llm_speak_turn(messages, say, play_typing, fb)
            # Repeat breaker: reply word-for-word identical to the last turn -> regenerate
            if reply and reply == last_reply:
                print("   [repeat-breaker: identical reply, regenerating the turn]")
                events.emit("AGENT", "guardrail.repeat_breaker", {"text": reply})
                messages.append({"role": "system", "content":
                    "You just repeated your previous reply verbatim. Do not repeat it. "
                    "React to the caller's latest message: acknowledge the new information "
                    "and take the next step (ask for the next missing detail or call a tool)."})
                reply = llm_speak_turn(messages, say, play_typing, fb)
            last_reply = reply
        print(f"=== Call ended (state={call.state}) ===")
        events.end_call("call ended", state=str(call.state), turns=user_turns)
    except InvalidStateError as e:
        print(f"=== Call aborted (InvalidStateError: {e}) ===")
        events.end_call("aborted", error=str(e))
    except Exception:
        import traceback
        print("=== Call crashed with an exception: ===")
        traceback.print_exc()
        events.end_call("crashed", error=traceback.format_exc()[-500:])
    finally:
        speak_q.put(None)
        if rec_file is not None:
            try:
                wav = rec.save(rec_file.with_suffix(".wav"))
                if wav:
                    print(f"Recording: {wav.name} ({rec.seconds():.1f}s, stereo: caller | agent)")
            except Exception as e:  # noqa: BLE001
                print(f"Recording not saved: {e}")


if __name__ == "__main__":
    cfg = settings.sip_config()

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
