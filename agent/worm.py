r"""Pre-commit WORM audit log (Glass Box Phase 1) - the local stand-in for S3 Object Lock.

Append-only JSON lines with a hash chain: every record carries the SHA-256 of the previous
record, so any edit or deletion in the middle breaks verification. The write returns an
acknowledgement {record_id, hash}; the agent must not execute an action until it holds
that ack (FR-COM-010), and the record_id is what the caller hears as the reference number.

File: calls\worm.jsonl (one chain for the whole installation, not per call).

    from agent import worm
    ack = worm.append({"corr": ..., "phase": "pre-commit", "action": "block", ...})
    ack -> {"record_id": "WORM-000042", "hash": "9f3a...", "prev": "b71c...", "ts": ...}
    worm.verify() -> (ok: bool, records: int, first_bad: str | None)

Failure injection for the N4 negative test: set VOICEAGENT_WORM_FAIL=1 in the environment
(or worm.FAIL_NEXT = True) and the next append raises WormWriteError.
"""
import hashlib
import json
import os
import threading
import time
from pathlib import Path

from agent import events

PROJECT = Path(__file__).resolve().parent.parent
WORM_FILE = PROJECT / "calls" / "worm.jsonl"
WORM_FILE.parent.mkdir(exist_ok=True)

_lock = threading.Lock()
FAIL_NEXT = False
GENESIS = "0" * 64


class WormWriteError(RuntimeError):
    pass


def _last() -> tuple[int, str]:
    """(count, hash of last record) - reads the tail of the file."""
    if not WORM_FILE.exists():
        return 0, GENESIS
    last, n = None, 0
    with open(WORM_FILE, encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                last, n = ln, n + 1
    if last is None:
        return 0, GENESIS
    return n, json.loads(last)["hash"]


def _digest(record: dict) -> str:
    body = json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def append(data: dict) -> dict:
    """Append one immutable record; returns the ack. Raises WormWriteError on failure."""
    global FAIL_NEXT
    t0 = time.time()
    with _lock:
        if FAIL_NEXT or os.environ.get("VOICEAGENT_WORM_FAIL") == "1":
            FAIL_NEXT = False
            events.emit("GATE→WORM", "worm.write_failed", {"phase": data.get("phase"), "action": data.get("action")})
            raise WormWriteError("WORM write failed (injected)")
        n, prev = _last()
        record = {"record_id": f"WORM-{n + 1:06d}", "ts": round(time.time(), 3), "prev": prev, **data}
        record["hash"] = _digest({k: v for k, v in record.items() if k != "hash"})
        with open(WORM_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
    ack = {"record_id": record["record_id"], "hash": record["hash"], "prev": prev, "ts": record["ts"]}
    events.emit("GATE→WORM", "worm.ack", {**ack, "phase": data.get("phase"), "action": data.get("action"),
                                        "corr": data.get("corr")}, ms=(time.time() - t0) * 1000)
    return ack


def verify() -> tuple[bool, int, str | None]:
    """Walk the chain; returns (ok, records, first_bad_record_id)."""
    if not WORM_FILE.exists():
        return True, 0, None
    prev, n = GENESIS, 0
    with open(WORM_FILE, encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            rec = json.loads(ln)
            n += 1
            h = rec.pop("hash")
            if rec.get("prev") != prev or _digest(rec) != h:
                return False, n, rec.get("record_id")
            prev = h
    return True, n, None


def spoken(record_id: str) -> str:
    """'WORM-000042' -> 'W O R M 0 0 0 0 4 2' - read digit by digit on the phone."""
    return " ".join(ch for ch in record_id if ch.isalnum())
