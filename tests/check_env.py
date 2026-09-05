r"""Step 1: environment check — run after setup.ps1 (setup.ps1 runs it automatically).

Verifies, in order, everything the agent needs on this machine:
  GPU / driver, Python version, packages, faster-whisper on CUDA (real transcription
  of audio\test_german_1.m4a), Ollama + configured LLM (tokens/s + tool-call smoke test),
  Piper voice for the configured language, SIP config.ini.

Usage (venv active, project folder):
    python tests\check_env.py
"""
import importlib
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

OK, WARN, FAIL = "  [ OK ]", "  [WARN]", "  [FAIL]"
failures = 0


def report(status, msg):
    global failures
    if status == FAIL:
        failures += 1
    print(f"{status} {msg}")


# ---------------------------------------------------------------- GPU
print("\n== GPU")
if shutil.which("nvidia-smi"):
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version",
         "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    report(OK if out else FAIL, out or "nvidia-smi returned nothing")
else:
    report(FAIL, "nvidia-smi not found - NVIDIA driver missing?")

# ---------------------------------------------------------------- Python + packages
print("\n== Python")
v = sys.version_info
report(OK if (v.major, v.minor) == (3, 12) else FAIL,
       f"Python {sys.version.split()[0]} at {sys.executable} (3.12 required: pyVoIP needs audioop)")

for mod, pkg in [("faster_whisper", "faster-whisper"), ("ctranslate2", "ctranslate2"),
                 ("piper", "piper-tts"), ("pyVoIP", "pyVoIP<2"), ("scipy", "scipy"),
                 ("numpy", "numpy"), ("ollama", "ollama"), ("audioop", "stdlib (3.12)")]:
    try:
        m = importlib.import_module(mod)
        report(OK, f"{mod} {getattr(m, '__version__', '')}")
    except Exception as e:  # noqa: BLE001
        report(FAIL, f"{mod} missing ({pkg}): {e}")

# ---------------------------------------------------------------- settings
print("\n== Settings")
from agent import settings  # noqa: E402  (registers CUDA DLL dirs)
print("      " + settings.describe())
report(OK if settings.CONFIG_PATH else WARN,
       f"config.ini: {settings.CONFIG_PATH or 'not found (needed only for the phone tests)'}")

# ---------------------------------------------------------------- faster-whisper on CUDA
print(f"\n== faster-whisper {settings.STT_MODEL} / {settings.STT_COMPUTE} on CUDA")
try:
    import ctranslate2
    n = ctranslate2.get_cuda_device_count()
    report(OK if n > 0 else FAIL, f"CTranslate2 sees {n} CUDA device(s); "
           f"compute types: {sorted(ctranslate2.get_supported_compute_types('cuda')) if n else '-'}")
    t0 = time.time()
    model = settings.load_whisper()
    report(OK, f"model loaded in {time.time()-t0:.1f}s (first run downloads ~1.6 GB from Hugging Face)")
    sample = next((p for p in sorted((PROJECT / "audio").iterdir())
                   if p.suffix.lower() in (".m4a", ".wav", ".mp3") and not p.name.startswith("greeting")), None)
    if sample:
        t0 = time.time()
        segs, info = model.transcribe(str(sample), beam_size=settings.STT_BEAM, vad_filter=True)
        text = " ".join(s.text.strip() for s in segs)
        dt = time.time() - t0
        report(OK if text else WARN,
               f"{sample.name}: {info.duration:.1f}s audio in {dt:.2f}s ({info.duration/dt:.0f}x realtime, {info.language})")
        print(f"      \"{text[:120]}{'...' if len(text) > 120 else ''}\"")
    # keep the whisper model loaded on purpose: the LLM check below must fit NEXT to it
except Exception as e:  # noqa: BLE001
    report(FAIL, f"whisper on CUDA failed: {e}")

# ---------------------------------------------------------------- Ollama + LLM
print(f"\n== Ollama / {settings.LLM_MODEL}")
try:
    import ollama
    names = [m.model for m in ollama.list().models]
    have = any(n == settings.LLM_MODEL or n.startswith(settings.LLM_MODEL + ":") for n in names)
    report(OK if have else FAIL,
           f"model present: {have}  (installed: {', '.join(names) or 'none'})")
    if have:
        def block_card(card_ref: str, reason: str) -> str:
            """Block a credit card. card_ref: e.g. visa-4821; reason: lost|stolen|fraud|damaged."""
            return "SUCCESS"

        t0 = time.time()
        r = ollama.chat(settings.LLM_MODEL, think=False, keep_alive=-1, options=settings.OLLAMA_OPTIONS,
                        messages=[{"role": "user", "content": "Write two short sentences about phones."}])
        load = time.time() - t0
        tps = r.eval_count / (r.eval_duration / 1e9) if r.eval_duration else 0
        report(OK if tps > 10 else WARN,
               f"first reply after {load:.1f}s (incl. VRAM load), generation {tps:.0f} tok/s")
        r = ollama.chat(settings.LLM_MODEL, think=False, keep_alive=-1, tools=[block_card],
                        options=settings.OLLAMA_OPTIONS,
                        messages=[{"role": "system", "content": "Call tools directly, no text."},
                                  {"role": "user", "content": "Block card visa-4821, it was stolen."}])
        tc = r.message.tool_calls or []
        report(OK if tc and tc[0].function.name == "block_card" else FAIL,
               f"tool calling: {[(t.function.name, dict(t.function.arguments)) for t in tc] or r.message.content[:80]}")
        ps = ollama.ps().models
        for m in ps:
            gb = m.size_vram / 2**30
            report(OK if m.size_vram >= m.size * 0.99 else WARN,
                   f"{m.model}: {gb:.1f} GB in VRAM" + ("" if m.size_vram >= m.size * 0.99
                   else f" of {m.size/2**30:.1f} GB total -> partially on CPU, too slow for voice! Use a smaller LLM or STT."))
except Exception as e:  # noqa: BLE001
    report(FAIL, f"Ollama not reachable / failed: {e}  (is the Ollama app running? 'ollama serve')")

# ---------------------------------------------------------------- Piper
print("\n== Piper TTS")
# (call_agent.py is not imported here: its module-level startup loads whisper + LLM)
VOICES = {"en": "en_US-lessac-high", "de": "de_DE-thorsten-high", "da": "da_DK-talesyntese-medium"}
voice = VOICES.get(settings.LANGUAGE, "?")
onnx = PROJECT / "voices" / f"{voice}.onnx"
if onnx.exists():
    try:
        from piper import PiperVoice
        pv = PiperVoice.load(str(onnx))
        t0 = time.time()
        chunks = list(pv.synthesize("This is a test of the local voice agent."))
        report(OK, f"{voice}: synthesized {len(chunks)} chunk(s) in {time.time()-t0:.2f}s "
                   f"@ {pv.config.sample_rate} Hz (CPU)")
    except Exception as e:  # noqa: BLE001
        report(FAIL, f"{voice} present but synthesis failed: {e}")
else:
    report(FAIL, f"{onnx} missing -> python -m piper.download_voices --download-dir voices {voice}")

# ---------------------------------------------------------------- SIP
print("\n== SIP (FRITZ!Box IP phone)")
if settings.CONFIG_PATH:
    import configparser
    c = configparser.ConfigParser()
    c.read(settings.CONFIG_PATH, encoding="utf-8")
    if c.has_section("sip"):
        pw = c["sip"].get("password", "")
        placeholder = not pw or "PUT_THE" in pw
        report(WARN if placeholder else OK,
               f"{c['sip'].get('user')}@{c['sip'].get('server')}:{c['sip'].get('port', '5060')} "
               + ("- password still the placeholder" if placeholder else "- credentials set"))
    else:
        report(FAIL, "config.ini has no [sip] section")
else:
    report(WARN, "no config.ini - phone tests will not run until it exists")

print()
if failures:
    print(f"{failures} check(s) FAILED - fix those before running the agent.")
    sys.exit(1)
print("All checks passed. Next: python tests\\test_call.py (needs the SIP password), then agent\\call_agent.py")
