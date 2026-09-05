r"""Confirm gate (Glass Box Phase 1) - the bank-owned state machine outside the model.

FR-COM-007/008/009 as code:

    IDLE --propose(card, action)--> PROPOSED --confirm(utterance)--> CONFIRMED
         --execute(card, action)--> EXECUTED  (gate closed for the rest of the call)
    PROPOSED --withdraw()--> IDLE            (FR-BLK-011: nothing changed, no record)
    any --freeze()--> FROZEN                 (FR-HIT-003: after escalation, no action ever)

Rules:
  * An action may only execute when the gate is CONFIRMED for EXACTLY that card and action.
  * Confirmation must be an explicit affirmative. "hmm", silence, a question, or a new
    request do not confirm (the utterance is classified here, not by the model).
  * At most ONE irreversible action per call. A second proposal after EXECUTED is rejected;
    the agent must escalate.
  * The gate is per call: call reset() at call start.

Every transition emits a GW→GATE event, so the page can show the state chip.
"""
import re
import time

from agent import events

IDLE, PROPOSED, CONFIRMED, EXECUTED, FROZEN = "IDLE", "PROPOSED", "CONFIRMED", "EXECUTED", "FROZEN"

_state = {"state": IDLE, "card": None, "action": None, "utterance": None, "proposed_at": None}

# Explicit affirmatives (en / de / da), matched as whole words on the normalised utterance.
_YES = re.compile(
    r"^(yes|yes please|yes that('s| is) (correct|right)|correct|that('s| is) (correct|right)|"
    r"confirmed?|i confirm|go ahead|please do|do it|absolutely|yep|yeah|"
    r"ja|ja bitte|ja genau|genau|richtig|das ist richtig|korrekt|ja das ist richtig|bestätigt|"
    r"ja tak|det er rigtigt|korrekt|bekræftet)[.!]*$")
_NO = re.compile(r"\b(no|nope|not|don'?t|wait|stop|cancel|nein|nicht|stopp|abbrechen|nej|ikke)\b")
_MAX_AGE_S = 90   # a proposal older than this is stale and must be re-read


class GateError(Exception):
    """Raised towards the tool caller; the message is what the model gets to read."""


def reset() -> None:
    _set(IDLE, card=None, action=None, utterance=None, reason="call start")


def state() -> dict:
    return dict(_state)


def propose(card: str, action: str) -> str:
    """The agent read the card + action back and is about to ask for confirmation."""
    if _state["state"] == FROZEN:
        raise GateError("ERROR: Gate is frozen after escalation. No further actions in this call.")
    if _state["state"] == EXECUTED:
        raise GateError("ERROR: One action has already been executed in this call. "
                        "A second action requires a human agent - escalate.")
    _set(PROPOSED, card=card, action=action, utterance=None, reason="read-back")
    return f"Proposal registered: {action} on {card}. Ask for an explicit yes now."


def confirm(utterance: str) -> str:
    """Classify the caller's reply. Only an explicit affirmative moves the gate."""
    if _state["state"] != PROPOSED:
        raise GateError(f"ERROR: Nothing is proposed (gate is {_state['state']}). "
                        "Read the card and action back first (propose_action).")
    if time.time() - _state["proposed_at"] > _MAX_AGE_S:
        _set(IDLE, card=None, action=None, utterance=None, reason="proposal expired")
        raise GateError("ERROR: The proposal expired. Read card and action back again.")
    norm = re.sub(r"[^a-zæøåäöüß' ]", " ", (utterance or "").lower())
    norm = re.sub(r"\s+", " ", norm).strip()
    if _NO.search(norm) or "?" in (utterance or ""):
        verdict = "negative_or_question"
    elif _YES.match(norm):
        verdict = "affirmative"
    else:
        verdict = "unclear"
    events.emit("GW→GATE", "gate.classify", {"utterance": utterance, "verdict": verdict})
    if verdict == "affirmative":
        _set(CONFIRMED, card=_state["card"], action=_state["action"], utterance=utterance, reason="explicit yes")
        return f"CONFIRMED: {_state['action']} on {_state['card']}. You may execute now."
    if verdict == "negative_or_question":
        _set(IDLE, card=None, action=None, utterance=None, reason="caller declined or asked")
        return "NOT CONFIRMED: the caller declined or asked something. Nothing has been changed."
    return "NOT CONFIRMED: the reply was not an explicit yes. Ask again for a clear yes or no."


def withdraw() -> str:
    _set(IDLE, card=None, action=None, utterance=None, reason="caller withdrew")
    return "Withdrawn. Nothing has been changed."


def check_execute(card: str, action: str) -> None:
    """Called by the gateway right before an action tool runs. Raises unless CONFIRMED for
    exactly this card and action."""
    s = _state
    if s["state"] != CONFIRMED or s["card"] != card or s["action"] != action:
        events.emit("GW→GATE", "gate.rejected", {"state": s["state"], "wanted": {"card": card, "action": action},
                                                 "have": {"card": s["card"], "action": s["action"]}})
        raise GateError(f"ERROR: Gate is {s['state']} for {s['action']} on {s['card']}; "
                        f"{action} on {card} is not confirmed. Read back and get an explicit yes first.")


def mark_executed() -> None:
    _set(EXECUTED, card=_state["card"], action=_state["action"], utterance=_state["utterance"], reason="action done")


def freeze(reason: str = "escalated") -> None:
    _set(FROZEN, card=_state["card"], action=_state["action"], utterance=_state["utterance"], reason=reason)


def _set(new: str, *, card, action, utterance, reason: str) -> None:
    old = _state["state"]
    _state.update(state=new, card=card, action=action, utterance=utterance,
                  proposed_at=time.time() if new == PROPOSED else _state["proposed_at"])
    events.emit("GW→GATE", "gate.state", {"from": old, "to": new, "card": card, "action": action, "reason": reason})
