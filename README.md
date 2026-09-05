# Local Voice Agent

Speech-to-speech telephone agent, 100 % local (FRITZ!Box → pyVoIP → faster-whisper →
Qwen via Ollama → Piper). No cloud, no external AI APIs, one conversation at a time.

**Current use case: credit-card blocking** (`agent/bank_tools.py`, mock banking backend).
The original appointment-booking demo (`agent/tools.py`, Radicale/CalDAV with Thunderbird
as the human frontend) is kept as legacy reference.

Project language is **English**: code, comments, prompts, logs and docs. The only
intentional exception are the language packs `LANG_CFG` / `MONTHS` in
`agent/call_agent.py` — those are spoken output for non-English callers, i.e. runtime
data, not documentation. Set `language = en | de | da` in `config.ini` to switch.
With `language_detect = true` (default) the greeting is spoken in that language, whisper
auto-detects the caller's language on the first two utterances and the agent switches Piper
voice, STT language and reply language for the rest of the call (needs that language's voice
in `voices\`; the Glass Box shows the decision as `stt.language` / `speech.switch`).

**Streaming STT.** Audio is not collected until the sentence is over: every 20 ms RTP slice
goes straight into the recogniser (`agent/stt_stream.py`), which emits partial transcripts
while the caller speaks (`stt.partial`), flags a stable partial when the caller pauses, ends
the utterance itself with the Silero VAD (`stt.endpoint`, default 600 ms) and delivers the
final (`stt.done`) with ~0 ms extra latency. Two backends behind the same interface — the STT
is a swap point: `stt_backend = whisper` (faster-whisper large-v3-turbo as a stream, multilingual,
LocalAgreement commits) or `sherpa` (sherpa-onnx streaming Zipformer, English, CPU). With
`speculative = true` the LLM turn already starts on the stable partial; its speech and tool
calls are held until the final confirms the text (`llm.speculative.*` events), otherwise the
turn is discarded and re-run. Offline check with a recorded call:
`python tests\test_stt_stream.py calls\<file>.wav [whisper|sherpa] [--realtime]`.

**Event store.** Every event also lands in a query layer under the data flow (`agent/store.py`):
a DynamoDB single-table model (PK `CALL#<corr>`, SK `EV#<seq>` / `META`, GSI1 by kind + time, GSI2
by day) with two backends behind one interface — `sqlite` (`calls\glassbox.db`, no install, FTS5
full-text search; default) and `dynamodb` (boto3 against AWS or DynamoDB Local, same items and
keys). The JSONL file per call stays the raw stream. The page has a "Search · event store" panel
whose hits open the call in replay at that event; API: `/api/db/calls`, `/api/db/search`,
`/api/db/stats`. Import older call files: `python -m agent.store --import`.

**Runtime controls.** The Glass Box page has an "Agent controls" panel (language, STT backend,
endpoint, speculative LLM, auto-detect, canonical English) that changes the agent for the next
call without a restart (`GET/POST /api/control`); `config.ini` only sets the start values. With
`canonical_english = true` whisper translates non-English callers to English while they speak
(`task=translate`), so orchestration, confirm gate and guardrails see one language; replies are
still spoken in the caller's language.

Every call is also recorded as `calls\<stamp>_<corr8>.wav` (stereo 8 kHz: left caller, right
agent, same clock as the events). In the Glass Box a replay with a recording shows a player in
the header: scrubbing the audio moves map, sequence and events to that moment; clicking a row
seeks the audio.

## Target hardware & model choice

Tuned for one **RTX 3090 (24 GB)** — everything sits in VRAM at once:

| Component | Model | VRAM | Notes |
|---|---|---|---|
| LLM | `qwen3.8:27b` (Q4_K_M) via Ollama | ~18 GB + KV cache | dense 27B, tool calling, `think=False`; ~30 tok/s |
| STT | faster-whisper `large-v3-turbo`, `float16`, beam 5 | ~1.8 GB | 2-3× more accurate on digits/names than int8 + beam 1 |
| TTS | Piper (`en_US-lessac-high`, `de_DE-thorsten-high`) | 0 (CPU) | faster than real time |

Ollama runs with `OLLAMA_FLASH_ATTENTION=1` and `OLLAMA_KV_CACHE_TYPE=q8_0` (set by
`setup.ps1`), which keeps the 27B model plus 8k context under ~20 GB. Budget: ~22 GB of
the ~23.5 GB Windows leaves usable. If `check_env.py` reports the LLM as *partially on
CPU*, switch to `llm = qwen3.5:9b` in `config.ini`.

All model choices live in `config.ini` `[models]` (see `config.example.ini`), read by
`agent/settings.py`; nothing is hard-coded in the scripts. Environment variables
`VOICEAGENT_LLM`, `VOICEAGENT_STT`, `VOICEAGENT_LANGUAGE` … override the file.

## Layout

```
C:\ai-voice-local\          (this folder — project code)
├── agent\call_agent.py    full phone agent (streaming TTS, reader thread, multi-language)
├── agent\bank_tools.py    card-blocking tools with code-level guardrails (current use case)
├── agent\settings.py      config.ini lookup, model defaults, CUDA DLL registration
├── agent\events.py        Glass Box event emitter (JSONL per call, correlation ID)
├── agent\gateway.py       tool gateway: schema check, confirm gate, WORM, bank mock (fixed core)
├── agent\confirm_gate.py  IDLE → PROPOSED → CONFIRMED → EXECUTED state machine
├── agent\worm.py          hash-chained append-only audit log (calls\worm.jsonl)
├── agent\tools.py         legacy appointment tools (CalDAV)
├── voip16.py              pyVoIP 16-bit / codec / silence monkeypatches + SIP/RTP/DTMF taps
├── setup.ps1              one-shot machine setup (venv, packages, Ollama, models, voices)
├── config.example.ini     template for SIP credentials + model selection
├── radicale.config        local CalDAV server config (legacy demo backend)
├── glassbox\              server.py (FastAPI/WebSocket) + index.html (the live page)
├── tests\                 check_env.py, timeline.py, test_gate.py + per-component tests (steps 2–6)
├── calls\                 event recordings, one JSONL per call (git-ignored)
├── voices\                Piper voices (git-ignored, downloaded by setup.ps1)
├── audio\                 test recordings and announcements
├── .venv\                 Python 3.12 venv (git-ignored, created by setup.ps1)
└── config.ini             SIP password + model selection (git-ignored, created by setup.ps1)
```
Keep the project folder out of OneDrive (venv sync churn). Legacy fallback for the
config: `%USERPROFILE%\Code\voiceagent-local\config.ini` (first machine).

## Setup & test order

**Step 0 — FRITZ!Box (once, no code):** create an IP phone "Voice-Agent"
(Telephony → Telephony Devices → LAN/Wi-Fi IP phone), assign a phone number exclusively
to it, cross-check with MicroSIP. Give the PC a static IPv4 in the FRITZ!Box.

**Step 1 — machine setup (idempotent, ~20 min incl. the 18 GB model download):**
```powershell
cd C:\ai-voice-local
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1                       # or: .\setup.ps1 -Llm qwen3.5:9b
```
Installs Python 3.12 (winget) + venv, pip packages incl. CUDA 12 wheels, Ollama (winget)
+ `qwen3.8:27b`, the Piper voices, creates `config.ini`, and finishes with
```powershell
python tests\check_env.py         # GPU, packages, whisper on CUDA, LLM tok/s + tool call, Piper, SIP
```
Then activate the venv for everything below: `.\.venv\Scripts\Activate.ps1`

**Step 2 — Whisper (STT):** put a short recording into `audio\`, then
```powershell
python tests\test_whisper.py          # auto-detects the language
python tests\test_whisper.py de       # or force a language code
```

**Step 3 — Piper (TTS):** voices are already in `voices\`. To regenerate the greeting:
```powershell
python -m piper -m en_US-lessac-high --data-dir voices -f audio\greeting.wav -- "Hello, you have reached the card security line. How can I help you?"
python tests\resample_to_8k.py audio\greeting.wav audio\greeting_8k.wav
```

**Step 4 — telephony (pyVoIP):** put the IP-phone password into `config.ini`
(project folder), quit MicroSIP, then
```powershell
python tests\test_call.py
```
Success: a call to the agent's number is answered and the Piper voice speaks the greeting.

**Step 5 — call + STT:**
```powershell
python tests\test_call_stt.py
```

**Step 6 — dialog core without audio (legacy appointment demo):** needs Ollama and
Radicale running (`.\start_radicale.ps1`), then
```powershell
python tests\test_chat.py
```

**Full agent:**
```powershell
python agent\call_agent.py
```

**Glass Box (Phase 0 — event recording):** every call is recorded hop by hop as
`calls\<stamp>_<corr>.jsonl` by `agent/events.py` (SIP datagrams, RTP first packets, VAD,
STT, LLM loop, tool calls, guardrail transitions, TTS, first-audio latency). Inspect with
```powershell
python tests\timeline.py            # latest call: SIP ladder, hop timeline, per-turn latency
python tests\timeline.py --raw      # plus the raw SIP messages
```
Set `VOICEAGENT_EVENT_ECHO=1` to also print every event to the console while the agent runs.

**Glass Box page (Phase 2):** the agent serves a live page at `http://<agent-ip>:8080`
(port in `config.ini [glassbox]`, 0 disables) — architecture map with pulsing hops, growing
sequence diagram, event ticker, confirm-gate chip, WORM chain, per-turn latency, SIP inspector,
and a replay of any recorded call with speed control. Replay without the agent:
```powershell
python -m glassbox.server           # replay-only server on :8080
```

## Notes

- Python 3.12 (pyVoIP uses `audioop`, removed in 3.13); pyVoIP stable below 2.0
- CUDA comes from the pip wheels `nvidia-cublas-cu12` / `nvidia-cudnn-cu12`; on Windows
  their DLL folders must be registered before importing faster_whisper —
  `agent/settings.py` does that, so import it first in every script
- scipy instead of torchaudio for resampling (no PyTorch needed, faster-whisper runs on CTranslate2)
- Phone audio: G.711, 8 kHz mono — upsample inbound to 16 kHz for Whisper,
  downsample the Piper output (22 kHz) to 8 kHz
- Windows firewall: allow the venv `python.exe` on private networks (SIP 5060/UDP + RTP);
  `setup.ps1` creates the rule when run elevated, otherwise prints the command
- NVIDIA Control Panel → *CUDA – Sysmem Fallback Policy* = *Prefer No Sysmem Fallback*:
  otherwise a VRAM overflow silently spills to system RAM and everything gets ~10× slower
- Radicale 3.7.x needs Windows Developer Mode (symlink privilege, WinError 1314)
- The legacy CalDAV calendar was named `termine`; `agent/tools.py` now uses `appointments`
  and creates that calendar on first use
