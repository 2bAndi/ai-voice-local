r"""Glass Box server (Phase 2) - serves the page and streams call events to it.

Runs INSIDE the agent process (started by call_agent.py in a background thread) so that it
can subscribe to agent/events.py directly; or standalone for replay only:

    python -m glassbox.server            # http://0.0.0.0:8080 - replay of recorded calls
    python -m glassbox.server --port 9090

Endpoints
    GET  /                    the page (glassbox/index.html)
    GET  /api/status          {"live": bool, "corr": current correlation id, "agent": {...}}
    GET  /api/calls           recorded calls, newest first
    GET  /api/calls/{name}    all events of one recorded call
    GET  /api/calls/{name}/audio   stereo WAV of that call (left caller, right agent), if recorded
    GET  /api/worm            WORM chain + verification result
    WS   /ws                  live stream: {"type":"hello", ...} then {"type":"event", "event": {...}}

Nothing leaves the LAN; the server binds to all interfaces so a laptop or phone on the same
network can open http://<agent-ip>:8080 while the call is in progress.
"""
import asyncio
import json
import re
import sys
import threading
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from agent import events  # noqa: E402

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
    import uvicorn
except ImportError as e:  # pragma: no cover
    raise SystemExit("Glass Box needs: pip install fastapi uvicorn websockets   (" + str(e) + ")")

HERE = Path(__file__).resolve().parent
CALLS = PROJECT / "calls"
CALL_RE = re.compile(r"^(\d{8})-(\d{6})_([0-9a-f]{8})\.jsonl$")

app = FastAPI(title="Voice Agent Glass Box")
_loop: asyncio.AbstractEventLoop | None = None
_clients: set = set()
_current: list = []            # events of the call in progress (or the last one), for late joiners
_agent_info: dict = {}
_live = False


# ----------------------------------------------------------------------------- event intake
def _on_event(ev: dict) -> None:
    """events.subscribe callback - runs on the agent's thread; hop to the asyncio loop."""
    global _current
    if ev["kind"] == "call.start":
        _current = []
    if ev["kind"] == "agent.models":
        _agent_info.update(ev.get("payload") or {})
    if ev.get("corr") or ev["kind"].startswith("sip."):
        _current.append(ev)
        if len(_current) > 5000:
            del _current[:1000]
    if _loop is not None:
        asyncio.run_coroutine_threadsafe(_broadcast({"type": "event", "event": ev}), _loop)


async def _broadcast(msg: dict) -> None:
    dead = []
    data = json.dumps(msg, ensure_ascii=False)
    for ws in list(_clients):
        try:
            await ws.send_text(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _clients.discard(ws)


# ----------------------------------------------------------------------------- HTTP
@app.get("/", response_class=HTMLResponse)
async def index():
    return (HERE / "index.html").read_text(encoding="utf-8")


@app.get("/api/status")
async def status():
    return {"live": _live, "corr": events.correlation_id(), "agent": _agent_info,
            "buffered": len(_current)}


@app.get("/api/calls")
async def calls():
    out = []
    for p in sorted(CALLS.glob("*.jsonl"), reverse=True):
        m = CALL_RE.match(p.name)
        if not m:
            continue
        d, t, corr = m.groups()
        n = 0
        first = None
        with open(p, encoding="utf-8") as f:
            for ln in f:
                if ln.strip():
                    n += 1
                    if first is None:
                        first = json.loads(ln)
        out.append({"name": p.name, "date": f"{d[:4]}-{d[4:6]}-{d[6:]}", "time": f"{t[:2]}:{t[2:4]}:{t[4:]}",
                    "corr": (first or {}).get("corr") or corr, "events": n,
                    "audio": p.with_suffix(".wav").exists()})
    return out


@app.get("/api/calls/{name}")
async def call(name: str):
    if not CALL_RE.match(name):
        return JSONResponse({"error": "bad name"}, status_code=400)
    p = CALLS / name
    if not p.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    evs = [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return evs


@app.get("/api/calls/{name}/audio")
async def call_audio(name: str):
    if not CALL_RE.match(name):
        return JSONResponse({"error": "bad name"}, status_code=400)
    p = (CALLS / name).with_suffix(".wav")
    if not p.exists():
        return JSONResponse({"error": "no recording for this call"}, status_code=404)
    return FileResponse(str(p), media_type="audio/wav", filename=p.name)


@app.get("/api/worm")
async def worm_chain():
    from agent import worm
    ok, n, bad = worm.verify()
    recs = []
    if worm.WORM_FILE.exists():
        recs = [json.loads(ln) for ln in worm.WORM_FILE.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return {"ok": ok, "records": n, "first_bad": bad, "chain": recs[-50:]}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    _clients.add(ws)
    try:
        await ws.send_text(json.dumps({"type": "hello", "live": _live, "corr": events.correlation_id(),
                                       "agent": _agent_info, "current": _current[-2000:]},
                                      ensure_ascii=False))
        while True:
            await ws.receive_text()      # keep-alive / ignore client messages
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        _clients.discard(ws)


# ----------------------------------------------------------------------------- lifecycle
def _serve(host: str, port: int, live: bool) -> None:
    global _loop, _live
    _live = live
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)
    config = uvicorn.Config(app, host=host, port=port, log_level="warning", loop="asyncio")
    server = uvicorn.Server(config)
    _loop.run_until_complete(server.serve())


def start_in_background(port: int = 8080, host: str = "0.0.0.0") -> threading.Thread:
    """Call from the agent process: subscribes to events and serves the page."""
    events.subscribe(_on_event)
    t = threading.Thread(target=_serve, args=(host, port, True), name="glassbox", daemon=True)
    t.start()
    return t


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="0.0.0.0")
    a = ap.parse_args()
    print(f"Glass Box (replay only) on http://{a.host}:{a.port}  - Ctrl+C to stop")
    _serve(a.host, a.port, live=False)
