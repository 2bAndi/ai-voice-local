r"""GPU performance diagnosis — run when check_env.py reports slow whisper / low tok/s.

Collects, in one go:
  1. nvidia-smi: PCIe link (gen/width), clocks, power state, throttle reasons
  2. faster-whisper: cold vs. warm transcription time of the same file (JIT/autotune vs. real)
  3. Ollama: two consecutive generations (prompt eval + generation tok/s), 'ollama ps',
     and the offload lines from the Ollama server log
  4. nvidia-smi snapshot DURING generation (utilization, clocks, PCIe throughput)

Usage (venv python, project folder):
    python tests\gpu_diag.py
"""
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from agent import settings  # noqa: E402


def smi(query, extra=()):
    try:
        return subprocess.run(["nvidia-smi", *extra, f"--query-gpu={query}", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"nvidia-smi failed: {e}"


print("== 1. GPU state (idle)")
print("   ", smi("name,driver_version,pstate,pcie.link.gen.current,pcie.link.gen.max,"
                "pcie.link.width.current,pcie.link.width.max,clocks.sm,clocks.max.sm,clocks.mem,"
                "power.draw,power.limit,temperature.gpu,memory.used,memory.total"))
perf = subprocess.run(["nvidia-smi", "-q", "-d", "PERFORMANCE"], capture_output=True, text=True).stdout
active = [ln.strip() for ln in perf.splitlines() if "Active" in ln and "Not Active" not in ln]
print("    throttle reasons active:", active or "none")

print("\n== 2. faster-whisper cold vs. warm")
sample = next(p for p in sorted((PROJECT / "audio").iterdir())
              if p.suffix.lower() in (".m4a", ".wav", ".mp3") and not p.name.startswith("greeting"))
t0 = time.time()
model = settings.load_whisper()
print(f"    load: {time.time()-t0:.1f}s")
for label in ("cold", "warm", "warm"):
    t0 = time.time()
    segs, info = model.transcribe(str(sample), beam_size=settings.STT_BEAM, vad_filter=True)
    text = " ".join(s.text for s in segs)
    dt = time.time() - t0
    print(f"    {label}: {info.duration:.1f}s audio -> {dt:.2f}s  ({info.duration/dt:.1f}x realtime)")
print("    whisper resident:", smi("memory.used"))

print(f"\n== 3. Ollama {settings.LLM_MODEL}")
import ollama  # noqa: E402

snap = []


def sampler(stop):
    while not stop.is_set():
        snap.append(smi("utilization.gpu,clocks.sm,power.draw,memory.used,pcie.link.width.current"))
        time.sleep(1.0)


for i in range(2):
    stop = threading.Event()
    th = threading.Thread(target=sampler, args=(stop,), daemon=True)
    th.start()
    t0 = time.time()
    r = ollama.chat(settings.LLM_MODEL, think=False, keep_alive=-1, options=settings.OLLAMA_OPTIONS,
                    messages=[{"role": "user", "content": "Write five short sentences about telephones."}])
    stop.set()
    th.join()
    wall = time.time() - t0
    load = (r.load_duration or 0) / 1e9
    pe = r.prompt_eval_count / (r.prompt_eval_duration / 1e9) if r.prompt_eval_duration else 0
    ge = r.eval_count / (r.eval_duration / 1e9) if r.eval_duration else 0
    print(f"    run {i+1}: wall {wall:.1f}s | load {load:.1f}s | prompt {pe:.0f} tok/s | "
          f"generation {ge:.1f} tok/s ({r.eval_count} tokens)")
    if snap:
        print("      during generation (util, sm clock, power, mem, pcie width):")
        for s in snap[-4:]:
            print("        ", s)
    snap.clear()

print("\n    ollama ps:")
for m in ollama.ps().models:
    print(f"      {m.model}: size {m.size/2**30:.1f} GB, in VRAM {m.size_vram/2**30:.1f} GB, "
          f"ctx {getattr(m, 'context_length', '?')}")

log = Path(os.environ.get("LOCALAPPDATA", "")) / "Ollama" / "server.log"
if log.exists():
    lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    keys = re.compile(r"offload|flash|kv_cache|cache type|vram|inference compute|library=|"
                      r"compute capability|memory.*available|CUDA|runner started", re.I)
    hits = [ln for ln in lines if keys.search(ln)]
    print(f"\n    {log} (last relevant lines):")
    for ln in hits[-25:]:
        print("      ", ln[-220:])
else:
    print(f"\n    no server log at {log}")

print("\n== 4. env vars seen by this process (Ollama reads the user-level ones at ITS start)")
for k in ("OLLAMA_FLASH_ATTENTION", "OLLAMA_KV_CACHE_TYPE", "OLLAMA_KEEP_ALIVE", "OLLAMA_MAX_LOADED_MODELS",
          "CUDA_VISIBLE_DEVICES", "OLLAMA_NUM_GPU"):
    print(f"    {k} = {os.environ.get(k, '(unset)')}")
