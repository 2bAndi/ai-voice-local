r"""Glass Box event emitter — Phase 0 of the architecture visualisation.

Every transition of a call through the reference architecture (Avaya-ingress PoV, v1.10)
is recorded as one JSON event. Events go to a JSON-lines file per call under calls\ and,
optionally, to in-process subscribers (the Phase-2 WebSocket server hooks in there).

Hop vocabulary (names are the diagram's own):
    CALLER→AVAYA      caller / PSTN side (we only see what the FRITZ!Box forwards)
    AVAYA→SP-A        SIP signalling from the FRITZ!Box (playing Avaya SBC + "media bridge")
    SP-A→SP-B         media: RTP in / out, DTMF
    SP-B→SP-C         PCM → STT text
    SP-C→SP-D         text → LLM turn (tokens, sentences, loop iterations)
    SP-D→GW           tool call towards the tool gateway
    GW→GATE           confirm-gate check          (Phase 1)
    GW→WORM           pre-/post-commit record     (Phase 1)
    GW→BANK           banking API mock
    SP-C→SP-B         TTS PCM → RTP out
    SP-D→SP-E         escalation / handover        (Phase 1 mock, Phase 3 real)
    AGENT             agent-internal (call start / end, guardrail nudges)

Event shape:
    {"ts": 1788550312.481, "t": 3.214, "corr": "6f1c...", "hop": "SP-B→SP-C",
     "kind": "stt.done", "ms": 412, "payload": {...}}
    ts   = wall-clock epoch seconds · t = seconds since call start · ms = duration if measured

Usage:
    from agent import events
    events.start_call(sip_call_id="...", remote="0301234567")      # mints the correlation ID
    events.emit("SP-B→SP-C", "stt.done", {"text": "..."}, ms=412)
    events.end_call("caller hung up")

Before start_call() (registration, startup) events are written to calls\_agent.jsonl.
"""
import json
import os
import threading
import time
import uuid
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
CALLS_DIR = PROJECT / "calls"
CALLS_DIR.mkdir(exist_ok=True)

_lock = threading.Lock()
_state = {"corr": None, "t0": None, "file": None, "seq": 0}
# Events emitted while no call is active (INVITE, 100/180, REGISTER refreshes ...). The
# INVITE always precedes start_call(), so start_call() replays everything since the last
# INVITE into the new call file with negative t.
import collections
_pending: collections.deque = collections.deque(maxlen=64)
_subscribers: list = []          # callables receiving each event dict (Phase 2 WebSocket)
ECHO = os.environ.get("VOICEAGENT_EVENT_ECHO", "0") == "1"   # print every event to stdout


def subscribe(fn) -> None:
    """Register an in-process subscriber; fn(event: dict) is called synchronously."""
    _subscribers.append(fn)


def correlation_id() -> str | None:
    return _state["corr"]


def call_t0() -> float | None:
    """Wall-clock start of the current call - the zero of every event's `t` (and of the recording)."""
    return _state["t0"]


def call_file():
    """Path of the current per-call event file, or None outside a call."""
    f = _state["file"]
    return Path(f.name) if f else None


def start_call(**payload) -> str:
    """Mint the bank-side correlation ID (UUIDv4) and open the per-call event file.
    Called at the INVITE — 'mint first, enrich later' (fact base §6)."""
    corr = str(uuid.uuid4())
    with _lock:
        _close_file()
        _state["corr"] = corr
        _state["t0"] = time.time()
        _state["seq"] = 0
        stamp = time.strftime("%Y%m%d-%H%M%S")
        _state["file"] = open(CALLS_DIR / f"{stamp}_{corr[:8]}.jsonl", "a", encoding="utf-8")
        # replay the signalling that led to this call (from the last INVITE on)
        pend = list(_pending)
        idx = max((i for i, e in enumerate(pend) if e["kind"] == "sip.rx.invite"), default=None)
        if idx is not None:
            for e in pend[idx:]:
                e = dict(e)
                e["corr"] = corr
                e["t"] = round(e["ts"] - _state["t0"], 3)
                _state["seq"] += 1
                e["seq"] = _state["seq"]
                _state["file"].write(json.dumps(e, ensure_ascii=False) + "\n")
            _state["file"].flush()
        _pending.clear()
    emit("AGENT", "call.start", {"correlation_id": corr, **payload})
    return corr


def end_call(reason: str = "", **payload) -> None:
    emit("AGENT", "call.end", {"reason": reason, **payload})
    with _lock:
        _close_file()
        _state["corr"] = None
        _state["t0"] = None


def emit(hop: str, kind: str, payload: dict | None = None, ms: float | None = None) -> dict:
    """Record one event. Safe from any thread; never raises into the caller."""
    now = time.time()
    ev = {
        "ts": round(now, 3),
        "t": round(now - _state["t0"], 3) if _state["t0"] else None,
        "corr": _state["corr"],
        "hop": hop,
        "kind": kind,
    }
    if ms is not None:
        ev["ms"] = round(ms, 1)
    if payload:
        ev["payload"] = payload
    try:
        with _lock:
            _state["seq"] += 1
            ev["seq"] = _state["seq"]
            f = _state["file"] or _agent_file()
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
            f.flush()
            if _state["file"] is None:
                _pending.append(ev)
        if ECHO:
            print(f"   [ev {ev['t'] if ev['t'] is not None else '-':>7} {hop:<12} {kind}]")
        for fn in list(_subscribers):
            try:
                fn(ev)
            except Exception:       # a broken subscriber must never break the call
                pass
    except Exception:
        pass
    return ev


class Timer:
    """Measure a span and emit it on exit:  with events.Timer("SP-B→SP-C", "stt.done") as t: ...
    Extra payload can be attached via t.payload[...] = ... inside the block."""

    def __init__(self, hop: str, kind: str, payload: dict | None = None):
        self.hop, self.kind = hop, kind
        self.payload = dict(payload or {})

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        emit(self.hop, self.kind, self.payload, ms=(time.time() - self.t0) * 1000)
        return False


# ----------------------------------------------------------------------------- internals
_agent_fh = None


def _agent_file():
    global _agent_fh
    if _agent_fh is None:
        _agent_fh = open(CALLS_DIR / "_agent.jsonl", "a", encoding="utf-8")
    return _agent_fh


def _close_file():
    f = _state["file"]
    if f:
        try:
            f.close()
        except Exception:
            pass
    _state["file"] = None
