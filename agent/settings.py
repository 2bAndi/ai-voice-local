r"""Shared runtime settings for the voice agent and the test scripts.

Everything that differs between machines lives here, resolved at import time:

  * CUDA DLL directories of the pip-installed NVIDIA wheels (Windows only) —
    must be registered before faster_whisper is imported.
  * Location of config.ini (SIP credentials + optional [models] / [agent] sections).
  * Model choices with defaults tuned for a single RTX 3090 (24 GB):
        LLM  qwen3.8:27b   (Q4_K_M, ~18 GB)  via Ollama
        STT  large-v3-turbo, float16, beam 5  (~1.8 GB) via faster-whisper
    Override any of them in config.ini without touching code:

        [models]
        llm = qwen3.5:9b
        stt = large-v3
        stt_compute = float16
        stt_beam = 5
        num_ctx = 8192

        [agent]
        language = de
        language_detect = true

        [glassbox]
        port = 8080          ; 0 = do not start the Glass Box page

config.ini search order (first hit wins):
  1. <project>\config.ini                             (git-ignored; default on AI-2)
  2. %USERPROFILE%\Code\voiceagent-local\config.ini   (legacy location on the first machine)
"""
import configparser
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
CONFIG_CANDIDATES = [
    PROJECT / "config.ini",
    Path.home() / "Code" / "voiceagent-local" / "config.ini",
]

DEFAULTS = {
    "llm": "qwen3.8:27b",
    "stt": "large-v3-turbo",
    "stt_compute": "float16",
    "stt_beam": "5",
    "num_ctx": "8192",
    "language": "en",
    "language_detect": "true",
    "port": "8080",
}


def add_cuda_dlls() -> None:
    """Make the pip-installed NVIDIA DLLs (cuBLAS/cuDNN) discoverable on Windows.
    Must run BEFORE importing faster_whisper, else 'cublas64_12.dll not found'."""
    if os.name != "nt":
        return
    nvidia = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    for sub in ("cublas", "cudnn"):
        p = nvidia / sub / "bin"
        if p.exists():
            os.add_dll_directory(str(p))
            os.environ["PATH"] = str(p) + os.pathsep + os.environ.get("PATH", "")


add_cuda_dlls()
# Hugging Face cache without symlinks on Windows (no Developer Mode needed) - silence the warning
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

_cfg = configparser.ConfigParser()
CONFIG_PATH: Path | None = None
for _p in CONFIG_CANDIDATES:
    if _p.exists():
        _cfg.read(_p, encoding="utf-8")
        CONFIG_PATH = _p
        break


def _get(section: str, key: str) -> str:
    env = os.environ.get(f"VOICEAGENT_{key.upper()}")
    if env:
        return env
    if _cfg.has_option(section, key):
        return _cfg.get(section, key).strip()
    return DEFAULTS[key]


LLM_MODEL = _get("models", "llm")
STT_MODEL = _get("models", "stt")
STT_COMPUTE = _get("models", "stt_compute")
STT_BEAM = int(_get("models", "stt_beam"))
NUM_CTX = int(_get("models", "num_ctx"))
LANGUAGE = _get("agent", "language")
# Detect the caller's language on the first utterances (whisper) and switch voice + reply
# language for the rest of the call. The greeting is always in LANGUAGE.
LANGUAGE_DETECT = _get("agent", "language_detect").strip().lower() in ("1", "true", "yes", "on")
GLASSBOX_PORT = int(_get("glassbox", "port"))      # 0 disables the page

# Ollama request options shared by every chat() call
OLLAMA_OPTIONS = {"num_ctx": NUM_CTX, "temperature": 0.3}


def sip_config() -> configparser.SectionProxy:
    """Return the [sip] section or exit with a helpful message."""
    if CONFIG_PATH is None or not _cfg.has_section("sip"):
        sys.exit(
            "No config.ini with a [sip] section found. Copy config.example.ini to\n"
            f"  {CONFIG_CANDIDATES[0]}\n"
            "and fill in the FRITZ!Box IP-phone credentials."
        )
    return _cfg["sip"]


def load_whisper():
    """Instantiate the configured faster-whisper model on the GPU."""
    from faster_whisper import WhisperModel
    return WhisperModel(STT_MODEL, device="cuda", compute_type=STT_COMPUTE)


def describe() -> str:
    return (f"config: {CONFIG_PATH or '(none)'} | llm: {LLM_MODEL} | "
            f"stt: {STT_MODEL}/{STT_COMPUTE}, beam {STT_BEAM} | "
            f"num_ctx: {NUM_CTX} | language: {LANGUAGE} (detect: {LANGUAGE_DETECT})")
