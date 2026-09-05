r"""Tool gateway (Glass Box Phase 1) - the one door between the model and the bank.

Mirrors the fixed core of the reference architecture in one small module:

  * READ tools pass through after argument validation (identify, verify, list).
  * ACTION tools (block_card) use the TWO-CALL PATTERN of the reference architecture
    (v1.20, "Confirm gate: state machine, two-call pattern"). The confirm gate is touched
    only when the final action arrives with ALL its parameters:
        call 1  block_card(card, reason)                -> gate PROPOSED; read back, ask for a yes
        call 2  block_card(card, reason, caller_reply)  -> gate classifies the reply; if CONFIRMED:
                WORM pre-commit + ack -> BPM async action -> banking API -> core banking
                -> WORM post-commit -> gate EXECUTED
    Reads (identify / verify / list) never touch the gate or the WORM: they go
    gateway -> read tools / BPM sync path. The reference number spoken to the caller is
    the WORM record id, never something the bank mock or the model made up.
  * The correlation ID and an idempotency key are stamped by the gateway; the model never
    sees or invents them (FR-COM-011, FR-BLK-004). The model has no gate tool - the gate
    lives inside the gateway on the action path.

Usage in the agent:
    from agent import gateway
    TOOLS = gateway.build_tools()          # name -> callable (docstrings become the schema)
    gateway.reset_call()                   # at call start, after events.start_call()
"""
import hashlib
import json

from agent import bank_tools, confirm_gate, events, worm

READ_TOOLS = {"identify_customer", "verify_identity", "list_cards"}
ACTION_TOOLS = {"block_card": "block"}
OPTIONAL = {"caller_reply"}

_SCHEMA = {
    "identify_customer": {"full_name": str, "date_of_birth": str},
    "verify_identity": {"security_answer": str},
    "list_cards": {},
    "block_card": {"card_last4": str, "reason": str, "caller_reply": str},
}


def reset_call() -> None:
    bank_tools.reset_session()
    confirm_gate.reset()


def _validate(tool: str, args: dict) -> dict:
    """Schema check + normalisation. Rejects unknown or missing arguments loudly."""
    spec = _SCHEMA[tool]
    clean = {}
    problems = []
    for name, typ in spec.items():
        if name not in args or args[name] in (None, ""):
            if name in OPTIONAL:
                clean[name] = ""
            else:
                problems.append(f"missing {name}")
            continue
        v = args[name]
        if typ is str and not isinstance(v, str):
            v = str(v)
        if name == "card_last4":
            v = "".join(ch for ch in v if ch.isdigit())[-4:]
            if len(v) != 4:
                problems.append("card_last4 must be exactly four digits")
        clean[name] = v
    extra = set(args) - set(spec)
    if extra:
        problems.append(f"unknown arguments {sorted(extra)}")
    events.emit("SP-D→GW", "gw.validate", {"tool": tool, "ok": not problems, "problems": problems})
    if problems:
        raise ValueError("ERROR: invalid arguments - " + "; ".join(problems))
    return clean


def _idempotency_key(card: str, action: str) -> str:
    corr = events.correlation_id() or "no-call"
    return hashlib.sha256(f"{corr}:{card}:{action}".encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------- model-facing tools
def _read(tool: str, fn, **args) -> str:
    """Read path: gateway -> read tools / BPM sync. No gate, no WORM."""
    events.emit("GW→READ", "read.call", {"tool": tool, "path": "BPM sync / read tools"})
    result = fn(**args)
    # the data comes back through the gateway to the orchestration (SP-D), which uses it to ask
    # the caller for the next specific item (security answer, card, ...)
    events.emit("READ→GW", "read.result", {"tool": tool, "ok": not result.startswith("ERROR"), "result": result[:160]})
    return result


def identify_customer(full_name: str, date_of_birth: str) -> str:
    """Step 2: Look up the caller by full name and date of birth (YYYY-MM-DD).
    Returns the security question to ask, or an error."""
    a = _validate("identify_customer", locals())
    return _read("identify_customer", bank_tools.identify_customer, **a)


def verify_identity(security_answer: str) -> str:
    """Step 3: Check the caller's answer to the security question. Returns pass or fail."""
    a = _validate("verify_identity", locals())
    return _read("verify_identity", bank_tools.verify_identity, **a)


def list_cards() -> str:
    """Step 4: List the verified caller's cards (type and last four digits)."""
    _validate("list_cards", {})
    return _read("list_cards", bank_tools.list_cards)


def block_card(card_last4: str, reason: str, caller_reply: str = "") -> str:
    """Step 5: Block a card - TWO calls.
    First call (card_last4 + reason, no caller_reply): the gate registers the request and
    tells you what to read back. Read it back in one sentence and ask "Is that correct?".
    Second call (same card_last4 + reason, plus caller_reply = the caller's EXACT words):
    the gate decides. Only an explicit yes executes the block. reason: lost | stolen |
    fraud | damaged. On success the result contains the reference number to read out."""
    a = _validate("block_card", locals())
    card, action = a["card_last4"], ACTION_TOOLS["block_card"]
    corr = events.correlation_id()
    events.emit("GW→GATE", "gate.call", {"action": action, "card": card,
                                         "with_reply": bool(a["caller_reply"])})
    st = confirm_gate.state()
    # ---- call 1: all parameters present, no reply yet -> propose
    if not a["caller_reply"]:
        try:
            confirm_gate.propose(card, action)
        except confirm_gate.GateError as e:
            return str(e)
        return (f"PROPOSED (not executed): block card ending {card}, reason {a['reason']}. "
                f"Read exactly this back to the caller in one sentence, then ask \"Is that correct?\". "
                f"Then call block_card again with the caller's exact reply in caller_reply.")
    # ---- call 2: classify the reply unless already confirmed for this card/action
    if not (st["state"] == "CONFIRMED" and st["card"] == card and st["action"] == action):
        if st["state"] != "PROPOSED" or st["card"] != card or st["action"] != action:
            # the model skipped call 1 (or changed card/reason): propose now, do not execute
            try:
                confirm_gate.propose(card, action)
            except confirm_gate.GateError as e:
                return str(e)
            return (f"NOT EXECUTED: the read-back for card ending {card} had not been registered. "
                    f"Read it back now and ask \"Is that correct?\", then call block_card again with the reply.")
        try:
            verdict = confirm_gate.confirm(a["caller_reply"])
        except confirm_gate.GateError as e:
            return str(e)
        if not verdict.startswith("CONFIRMED"):
            return verdict + " The card has NOT been blocked."
    try:
        confirm_gate.check_execute(card, action)
    except confirm_gate.GateError as e:
        return str(e)
    idem = _idempotency_key(card, action)
    # 2 - WORM pre-commit (evidence before execution)
    try:
        pre = worm.append({"corr": corr, "phase": "pre-commit", "action": action, "card_last4": card,
                           "reason": a["reason"], "verification": "passed",
                           "confirmation_utterance": confirm_gate.state()["utterance"],
                           "idempotency_key": idem})
    except worm.WormWriteError as e:
        events.emit("GATE→WORM", "action.refused", {"why": "worm write failed", "action": action, "card": card})
        return (f"ERROR: The audit record could not be written ({e}). The card has NOT been blocked. "
                "Tell the caller and connect them to a human agent.")
    # 2b - BPM async path (mock): it may start ONLY now, after the WORM ack. The pre-commit record
    #      is the evidence that this action is still owed if anything downstream fails. The BPM action
    #      is what calls the banking API, which in turn changes the core banking system.
    events.emit("WORM→BPM", "bpm.async.start", {"process": "card-block", "evidence": pre["record_id"],
                                                "idempotency_key": idem, "mock": True})
    # 3 - banking API, called from the BPM action
    events.emit("BPM→BANK", "bank.call", {"api": "block", "card_last4": card, "idempotency_key": idem, "corr": corr})
    result = bank_tools.block_card(card, a["reason"])
    ok = result.startswith("SUCCESS")
    # 4 - WORM post-commit
    try:
        worm.append({"corr": corr, "phase": "post-commit", "action": action, "card_last4": card,
                     "pre_record": pre["record_id"], "result": "blocked" if ok else "failed",
                     "detail": result[:200]})
    except worm.WormWriteError:
        pass   # the pre-commit record already proves the intent; execution result is logged in events
    if not ok:
        events.emit("BPM→BANK", "action.failed", {"action": action, "card": card, "detail": result[:200]})
        return ("ERROR: The bank could not block the card. Tell the caller the card has NOT been "
                "blocked and connect them to a human agent. Detail: " + result)
    confirm_gate.mark_executed()
    ref = worm.spoken(pre["record_id"])
    events.emit("GW→SP-D", "action.done", {"action": action, "card": card, "reference": pre["record_id"]})
    return (f"SUCCESS: card ending {card} is now blocked (reason: {a['reason']}). "
            f"Tell the caller the card is blocked and state the reference number EXACTLY like this, "
            f"without changing it: {ref}. Do NOT ask the caller to repeat it.")


def escalate(reason: str) -> str:
    """Connect the caller to a human agent (any time on request, or when verification fails,
    a tool errors, or the request is outside card blocking). Freezes all actions."""
    confirm_gate.freeze(reason)
    payload = {"corr": events.correlation_id(), "reason": reason, "gate": confirm_gate.state()["state"]}
    events.emit("SP-D→SP-E", "escalate", payload)
    return ("Escalation registered (mock). Tell the caller you are connecting them to a colleague. "
            "Do not execute any further action.")


def _guarded(fn):
    """Validation errors become tool results the model can read, never exceptions."""
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ValueError as e:
            return str(e)
    return wrapper


def build_tools() -> dict:
    return {name: _guarded(fn) for name, fn in {
        "identify_customer": identify_customer,
        "verify_identity": verify_identity,
        "list_cards": list_cards,
        "block_card": block_card,
        "escalate": escalate,
    }.items()}


def describe() -> str:
    return json.dumps({"read": sorted(READ_TOOLS), "action": sorted(ACTION_TOOLS), "escalation": ["escalate"]})
