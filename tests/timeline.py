r"""Glass Box Phase 0 - print a recorded call as a timeline.

Reads calls\<stamp>_<corr>.jsonl (written by agent/events.py) and prints:
  1. the SIP ladder (every datagram in/out with method or status, CSeq, Call-ID tail)
  2. the hop-by-hop timeline with durations and a one-line payload summary
  3. per-turn latency: utterance -> STT -> first LLM token -> first TTS -> first audio

Usage (venv active, project folder):
    python tests\timeline.py                  # latest call
    python tests\timeline.py calls\20260904-213012_6f1c9a2b.jsonl
    python tests\timeline.py --sip            # SIP ladder only
    python tests\timeline.py --raw            # also dump raw SIP messages
"""
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
CALLS = PROJECT / "calls"

args = [a for a in sys.argv[1:] if not a.startswith("--")]
flags = {a for a in sys.argv[1:] if a.startswith("--")}
if args:
    path = Path(args[0])
else:
    import re
    files = sorted(p for p in CALLS.glob("*.jsonl") if re.match(r"\d{8}-\d{6}_", p.name))
    if not files:
        sys.exit(f"No call files in {CALLS} - make a call with agent\\call_agent.py first.")
    path = files[-1]

events = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
print(f"{path.name}: {len(events)} events, correlation {events[0].get('corr')}\n")


def short(ev) -> str:
    p = ev.get("payload") or {}
    k = ev["kind"]
    if k.startswith("sip."):
        return f"{p.get('line','')}  CSeq {p.get('cseq','')}  Call-ID …{(p.get('call_id') or '')[-8:]}"
    if k == "stt.done":
        return f"\"{p.get('text','')[:70]}\"  lang={p.get('language')} ({p.get('language_mode','')}) beam {p.get('beam','')}"
    if k == "llm.sentence":
        return f"\"{p.get('text','')[:90]}\""
    if k == "llm.prompt":
        return f"system prompt {p.get('chars')} chars · reply in {p.get('reply_language')} · tools {p.get('tools')} · {len(p.get('rules') or [])} rules · {len(p.get('guardrails') or [])} guardrails"
    if k == "stt.language":
        return f"detected {p.get('detected')} (p={p.get('probability')}) · current {p.get('current')} -> {p.get('decision')} · {p.get('reason','')}"
    if k == "speech.switch":
        return f"{p.get('from')} -> {p.get('to')} · voice {p.get('voice_from')} -> {p.get('voice_to')} · reply in {p.get('reply_language')}"
    if k == "llm.request":
        return f"iter {p.get('iteration')} · {p.get('messages')} msgs · {p.get('model')} · temp {p.get('temperature')} · think off · after {p.get('last_role')}"
    if k == "speech.config":
        s, t = p.get("stt") or {}, p.get("tts") or {}
        return f"STT {s.get('model')} {s.get('compute')} beam {s.get('beam')} lang={s.get('language')} ({s.get('mode')}) · TTS {t.get('voice')} {t.get('engine')} · reply {p.get('reply_language')}"
    if k == "guardrail.pass":
        return f"passed {len(p.get('checks') or [])} checks · tools this turn={p.get('tools_called_this_turn')} · nudges {p.get('nudges')}"
    if k == "guardrail.nudge":
        return f"{p.get('type')} · LLM told: \"{(p.get('instruction') or '')[:70]}\""
    if k == "tool.call":
        return f"{p['tool']}({json.dumps(p.get('args'), ensure_ascii=False)})"
    if k == "tool.result":
        return f"{p['tool']} -> {'OK ' if p.get('ok') else 'ERROR '}{p.get('result','')[:80]}"
    if k == "llm.done":
        return f"iter {p.get('iteration')} · {p.get('chunks')} chunks · tools {p.get('tool_calls')}"
    if k in ("tts.synth", "rtp.write", "filler.typing", "vad.utterance", "tts.greeting"):
        return f"{p.get('audio_ms','?')} ms audio" + (f"  {p['voice']}" if p.get("voice") else "") + (f"  \"{p['text'][:60]}\"" if p.get("text") else "")
    if k == "gate.state":
        return f"{p.get('from')} -> {p.get('to')}  {p.get('action') or ''} {p.get('card') or ''}  ({p.get('reason')})"
    if k == "worm.ack":
        return f"{p.get('phase')} {p.get('record_id')}  hash {str(p.get('hash'))[:12]}…"
    if k.startswith("guardrail.") or k.startswith("bank.") or k.startswith("gate.") or k.startswith("action.") or k.startswith("bpm."):
        return ", ".join(f"{a}={b}" for a, b in p.items())
    if k == "read.result":
        return f"{p.get('tool')} -> {'OK ' if p.get('ok') else 'ERROR '}{p.get('result','')[:80]}"
    if k == "read.call":
        return f"{p.get('tool')} via {p.get('path')}"
    if k == "gate.call":
        return f"{p.get('action')} {p.get('card')} · " + ("with caller reply" if p.get("with_reply") else "call 1 (no reply)")
    if k == "vad.level":
        return f"peak {p.get('peak_rms')} / thr {p.get('threshold')}" + (" · in speech" if p.get("speech") else "")
    if k == "vad.start":
        return f"speech onset rms {p.get('rms')}"
    if k == "rtp.rx.first":
        return f"PT {p.get('payload_type')} {p.get('codec')} SSRC {p.get('ssrc')} from {p.get('from')}"
    if k == "rtp.tx.start":
        return f"{p.get('codec')} to {p.get('to')} SSRC {p.get('ssrc')}"
    if k == "call.start":
        return f"from {p.get('sip_from','')}"
    if k == "call.end":
        return f"{p.get('reason')} {p.get('state','')} turns={p.get('turns','')}"
    return json.dumps(p, ensure_ascii=False)[:100] if p else ""


# ---------------------------------------------------------------- 1. SIP ladder
sip = [e for e in events if e["kind"].startswith("sip.") and e["kind"] != "sip.socket"]
if sip:
    print("== SIP ladder  (tx = we send to the FRITZ!Box, rx = we receive)")
    for e in sip:
        p = e.get("payload") or {}
        arrow = "  --> " if ".tx." in e["kind"] else "  <-- "
        t = f"{e['t']:7.3f}" if e.get("t") is not None else "   pre "
        sdp = p.get("sdp") or {}
        extra = f"   SDP {sdp.get('m','')} {sdp.get('dir','')}" if sdp else ""
        hx = {k: v for k, v in (p.get("headers") or {}).items()
              if k.upper().startswith("X-") or k in ("User-to-User", "Refer-To", "Replaces", "Reason")}
        extra += f"   {hx}" if hx else ""
        print(f"{t}{arrow}{p.get('line','')}    CSeq {p.get('cseq','')}{extra}")
        if "--raw" in flags:
            print("        " + (p.get("raw") or "").replace("\r\n", "\n        ").rstrip())
    print()
if "--sip" in flags:
    sys.exit(0)

# ---------------------------------------------------------------- 2. timeline
print("== Timeline")
print(f"{'t':>8}  {'hop':<12} {'kind':<24} {'ms':>7}  detail")
for e in events:
    if e["kind"].startswith("sip.") and "--all" not in flags:
        continue
    t = f"{e['t']:8.3f}" if e.get("t") is not None else "     pre"
    ms = f"{e['ms']:7.0f}" if e.get("ms") is not None else "       "
    print(f"{t}  {e['hop']:<12} {e['kind']:<24} {ms}  {short(e)}")

# ---------------------------------------------------------------- 3. per-turn latency
print("\n== Turn latency (ms from end of caller utterance)")
print(f"{'turn':>4} {'stt':>6} {'llm 1st tok':>12} {'tts 1st':>8} {'1st audio':>10}   caller said")
turn = None
for e in events:
    k = e["kind"]
    if k == "vad.utterance":
        if turn:
            print(f"{turn['n']:>4} {turn.get('stt','-'):>6} {turn.get('tok','-'):>12} {turn.get('tts','-'):>8} {turn.get('audio','-'):>10}   {turn.get('text','')[:60]}")
        turn = {"n": e["payload"].get("turn"), "t0": e["ts"]}
    elif turn:
        rel = round((e["ts"] - turn["t0"]) * 1000)
        if k == "stt.done":
            turn["stt"], turn["text"] = rel, e["payload"].get("text", "")
        elif k == "llm.token.first" and "tok" not in turn:
            turn["tok"] = rel
        elif k == "tts.synth" and "tts" not in turn:
            turn["tts"] = rel
        elif k == "turn.first_audio" and "audio" not in turn:
            turn["audio"] = rel
if turn:
    print(f"{turn['n']:>4} {turn.get('stt','-'):>6} {turn.get('tok','-'):>12} {turn.get('tts','-'):>8} {turn.get('audio','-'):>10}   {turn.get('text','')[:60]}")

hops = sorted({e["hop"] for e in events})
print(f"\nHops seen: {', '.join(hops)}")
