r"""Phase 1 unit test: gateway + confirm gate + WORM, scripted, no LLM and no phone.

Two-call pattern (architecture v1.20): block_card(card, reason) -> PROPOSED; block_card(card,
reason, caller_reply) -> gate decides -> WORM -> bank. Reads never touch gate or WORM.

    HP   identify -> verify -> list -> block(1) PROPOSED -> block(2,"yes") -> SUCCESS, WORM ref
    N1   block(2) with a reply but without call 1     -> NOT EXECUTED, no WORM record
    N1b  "hmm" / question as reply                    -> NOT CONFIRMED, nothing changed
    N2   second action in the same call               -> rejected, escalation required
    N4   WORM write fails                             -> action refused, card stays active
    R    reads do not emit gate/WORM events
    S    schema validation
    chain verification at the end

Usage (venv active, project folder):
    python tests\test_gate.py
"""
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from agent import events, gateway, worm, confirm_gate, bank_tools  # noqa: E402

failures = 0
seen = []
events.subscribe(lambda ev: seen.append(ev))


def check(label, cond, detail=""):
    global failures
    print(f"  [{'OK ' if cond else 'FAIL'}] {label}" + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        failures += 1


def new_call(name):
    events.start_call(test=name)
    gateway.reset_call()
    for c in bank_tools._CUSTOMERS["cust-001"]["cards"]:
        c["status"] = "active"
    T = gateway.build_tools()
    seen.clear()
    assert "Customer found" in T["identify_customer"]("John Miller", "1980-05-12")
    assert "verified" in T["verify_identity"]("Smith")
    assert "Visa" in T["list_cards"]()
    return T


def card_status(last4):
    return next(c["status"] for c in bank_tools._CUSTOMERS["cust-001"]["cards"] if c["last4"] == last4)


print("== R reads stay on the read path")
T = new_call("reads")
hops = {e["hop"] for e in seen}
check("reads emit GW→READ", "GW→READ" in hops, str(hops))
check("reads never touch gate or WORM", not any(h in hops for h in ("GATE→WORM", "WORM→BANK")), str(hops))
check("gate still IDLE after reads", confirm_gate.state()["state"] == "IDLE")
events.end_call("test")

print("== HP happy path (two calls)")
T = new_call("happy")
n_before = worm.verify()[1]
r = T["block_card"]("9034", "stolen")
check("call 1 -> PROPOSED, not executed", r.startswith("PROPOSED") and confirm_gate.state()["state"] == "PROPOSED", r)
check("no WORM record after call 1", worm.verify()[1] == n_before)
check("card still active after call 1", card_status("9034") == "active")
r = T["block_card"]("9034", "stolen", "Yes, that is correct.")
check("call 2 with explicit yes -> SUCCESS", r.startswith("SUCCESS"), r)
check("reference is the WORM id", "W O R M" in r, r)
check("two WORM records (pre + post)", worm.verify()[1] == n_before + 2)
check("gate EXECUTED", confirm_gate.state()["state"] == "EXECUTED")
check("card blocked", card_status("9034") == "blocked")
order = [e["hop"] for e in seen if e["hop"] in ("GW→GATE", "GATE→WORM", "WORM→BANK")]
check("path order gate -> worm -> bank", order[:3] == ["GW→GATE", "GW→GATE", "GATE→WORM"] or order[0] == "GW→GATE", str(order))
events.end_call("test")

print("== N1 call 2 without call 1")
T = new_call("n1")
n_before = worm.verify()[1]
r = T["block_card"]("4821", "lost", "yes")
check("not executed, re-proposed", r.startswith("NOT EXECUTED") and confirm_gate.state()["state"] == "PROPOSED", r)
check("no WORM record", worm.verify()[1] == n_before)
check("card still active", card_status("4821") == "active")
print("== N1b hmm / question")
r = T["block_card"]("4821", "lost", "hmm")
check("'hmm' -> NOT CONFIRMED, stays PROPOSED", r.startswith("NOT CONFIRMED") and confirm_gate.state()["state"] == "PROPOSED", r)
r = T["block_card"]("4821", "lost", "Wait, which card was that?")
check("question -> NOT CONFIRMED, back to IDLE", r.startswith("NOT CONFIRMED") and confirm_gate.state()["state"] == "IDLE", r)
check("card still active", card_status("4821") == "active")
events.end_call("test")

print("== N2 second action in the same call")
T = new_call("n2")
T["block_card"]("4821", "lost")
check("first block ok", T["block_card"]("4821", "lost", "yes").startswith("SUCCESS"))
r = T["block_card"]("9034", "lost")
check("second proposal rejected", r.startswith("ERROR") and "escalate" in r, r)
r = T["block_card"]("9034", "lost", "yes")
check("second execution rejected", r.startswith("ERROR"), r)
check("second card untouched", card_status("9034") == "active")
T["escalate"]("second action requested")
check("escalate freezes gate", confirm_gate.state()["state"] == "FROZEN")
events.end_call("test")

print("== N4 WORM write failure")
T = new_call("n4")
T["block_card"]("4821", "stolen")
worm.FAIL_NEXT = True
r = T["block_card"]("4821", "stolen", "yes please")
check("action refused", r.startswith("ERROR") and "NOT been blocked" in r, r)
check("card still active", card_status("4821") == "active")
check("gate CONFIRMED but not EXECUTED", confirm_gate.state()["state"] == "CONFIRMED")
events.end_call("test")

print("== S schema validation")
T = new_call("schema")
r = T["block_card"]("48", "lost")
check("bad card_last4 rejected", r.startswith("ERROR") and "four digits" in r, r)
events.end_call("test")

ok, n, bad = worm.verify()
print(f"\n== WORM chain: {'intact' if ok else 'BROKEN at ' + str(bad)}, {n} records")
check("chain verifies", ok)
print(f"\n{'ALL PASSED' if not failures else str(failures) + ' FAILED'}")
sys.exit(1 if failures else 0)
