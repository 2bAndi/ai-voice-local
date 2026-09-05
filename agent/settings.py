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

        [store]
        backend = sqlite     ; or dynamodb

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
    "stt_backend": "whisper",
    "stt_interval": "0.7",
    "endpoint_ms": "600",
    "vad_model": "models/silero_vad.onnx",
    "sherpa_model": "models/sherpa-onnx-streaming-zipformer-en-2023-06-26",
    "num_ctx": "8192",
    "language": "en",
    "language_detect": "true",
    "speculative": "true",
    "canonical_english": "true",
    "backend": "sqlite",
    "sqlite_path": "calls/glassbox.db",
    "dynamodb_table": "glassbox-events",
    "dynamodb_region": "eu-central-1",
    "dynamodb_endpoint": "",
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
# Streaming STT (agent/stt_stream.py): "whisper" (faster-whisper as a stream, multilingual, GPU)
# or "sherpa" (sherpa-onnx streaming Zipformer, English, CPU). Both use the Silero VAD for endpointing.
STT_BACKEND = _get("models", "stt_backend").strip().lower()
STT_INTERVAL = float(_get("models", "stt_interval"))      # whisper: seconds of new audio per partial decode
ENDPOINT_MS = int(_get("models", "endpoint_ms"))          # pause that ends an utterance
VAD_MODEL = _get("models", "vad_model")
SHERPA_MODEL = _get("models", "sherpa_model")
NUM_CTX = int(_get("models", "num_ctx"))
LANGUAGE = _get("agent", "language")
# Detect the caller's language on the first utterances (whisper) and switch voice + reply
# language for the rest of the call. The greeting is always in LANGUAGE.
LANGUAGE_DETECT = _get("agent", "language_detect").strip().lower() in ("1", "true", "yes", "on")
# Start the LLM turn on a stable partial transcript (caller paused); discarded if the final differs.
SPECULATIVE = _get("agent", "speculative").strip().lower() in ("1", "true", "yes", "on")
# Whisper translates non-English callers to English on the fly (task=translate): the orchestration,
# the confirm gate and the guardrails then only ever see one language. The reply is still spoken
# in the caller's language.
CANONICAL_ENGLISH = _get("agent", "canonical_english").strip().lower() in ("1", "true", "yes", "on")
GLASSBOX_PORT = int(_get("glassbox", "port"))      # 0 disables the page
# Event store under the data flow (agent/store.py): "sqlite" (local file, full-text search) or
# "dynamodb" (boto3 - AWS or DynamoDB Local via dynamodb_endpoint). Same single-table data model.
STORE_BACKEND = _get("store", "backend").strip().lower()
SQLITE_PATH = _get("store", "sqlite_path")
DYNAMODB_TABLE = _get("store", "dynamodb_table")
DYNAMODB_REGION = _get("store", "dynamodb_region")
DYNAMODB_ENDPOINT = _get("store", "dynamodb_endpoint").strip()

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
            f"stt: {STT_BACKEND} {STT_MODEL}/{STT_COMPUTE}, beam {STT_BEAM}, endpoint {ENDPOINT_MS} ms | "
            f"num_ctx: {NUM_CTX} | language: {LANGUAGE} (detect: {LANGUAGE_DETECT})")
