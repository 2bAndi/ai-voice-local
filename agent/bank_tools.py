r"""Mock banking tools for the credit-card blocking use case.

Everything is simulated in-memory (no real banking backend). The guardrail
philosophy from the appointment demo applies, only stricter:

  - NO information about cards and NO state change without completed identity
    verification (enforced in code, not by prompt).
  - block_card only works after verify_identity succeeded in THIS call.
  - Success messages come exclusively from tool results.

call_agent.py must call reset_session() at the start of every call.

Test customers (for phone testing):
  1) John Miller,  born 1980-05-12
     security question: mother's maiden name -> "Smith"
     cards: Visa ending 4821, Mastercard ending 9034
  2) Emma Johnson, born 1992-11-03
     security question: name of first pet -> "Bella"
     cards: Visa ending 7755
"""
import random
import re
import string

from agent import events


def _state(kind: str, **payload) -> None:
    """Guardrail state transitions of the mock bank. Reads live on the read path (GW→READ),
    the block itself is reached only through gate + WORM (WORM→BANK)."""
    hop = "BANK→SOR" if kind.startswith("bank.") else "GW→READ"   # bank.* = core banking changed
    events.emit(hop, kind, payload)

_CUSTOMERS = {
    "cust-001": {
        "name": "John Miller",
        "dob": "1980-05-12",
        "question": "What is your mother's maiden name?",
        "answer": "smith",
        "cards": [
            {"ref": "visa-4821", "type": "Visa", "last4": "4821", "status": "active"},
            {"ref": "mc-9034", "type": "Mastercard", "last4": "9034", "status": "active"},
        ],
    },
    "cust-002": {
        "name": "Emma Johnson",
        "dob": "1992-11-03",
        "question": "What is the name of your first pet?",
        "answer": "bella",
        "cards": [
            {"ref": "visa-7755", "type": "Visa", "last4": "7755", "status": "active"},
        ],
    },
}

VALID_REASONS = {"lost", "stolen", "fraud", "damaged"}

_session = {"identified": None, "verified": None, "failed_verifications": 0}


def reset_session() -> None:
    """Reset per-call state. Called by the agent at the start of every call."""
    _state("guardrail.reset")
    _session["identified"] = None
    _session["verified"] = None
    _session["failed_verifications"] = 0


def identify_customer(full_name: str, date_of_birth: str) -> str:
    """Step 1: Identify the customer by full name and date of birth.

    Only call this AFTER the caller has actually stated both their full name and
    their date of birth in this conversation.

    Args:
      full_name: The caller's full name exactly as stated on the phone
      date_of_birth: Date of birth in format YYYY-MM-DD, e.g. 1980-05-12

    Returns:
      On success: the security question that must be asked next (verify_identity).
      On failure: an error message with the next step.
    """
    name = (full_name or "").strip()
    lower = name.lower()
    dob = (date_of_birth or "").strip()
    placeholders = ("caller", "full name", "your name", "unknown", "customer", "<", "{", "n.n")
    if len(lower) < 5 or " " not in name or any(p in lower for p in placeholders):
        return ("ERROR: You do not have the caller's real name yet. Ask the caller: "
                "'May I have your full name, please?' and wait for the answer before "
                "calling this tool again.")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", dob):
        return ("ERROR: date_of_birth is missing or not in YYYY-MM-DD form. If the caller "
                "has not stated their date of birth yet, ask for it now; otherwise convert "
                "their spoken date to YYYY-MM-DD and call this tool again.")
    for cid, c in _CUSTOMERS.items():
        if lower == c["name"].lower() and dob == c["dob"]:
            _session["identified"] = cid
            _session["verified"] = None
            _state("guardrail.identified", customer=cid)
            return (f"Customer found. Now ask this security question and then call "
                    f"verify_identity: \"{c['question']}\"")
    return (f"ERROR: No customer named '{name}' with date of birth {dob} found. "
            f"Confirm the spelling of the name and the date of birth with the caller, "
            f"then call identify_customer again with the corrected values.")


def verify_identity(security_answer: str) -> str:
    """Step 2: Verify the identified customer with their security answer.

    Only call after identify_customer succeeded and the caller answered the
    security question.

    Args:
      security_answer: The caller's answer to the security question

    Returns:
      Verification result. After 3 failed attempts the call must be referred
      to a human hotline.
    """
    cid = _session["identified"]
    if cid is None:
        return "ERROR: Call identify_customer first (full name + date of birth)."
    if _session["failed_verifications"] >= 3:
        return ("ERROR: Verification locked after 3 failed attempts. Tell the caller "
                "to contact the human hotline. Do not proceed.")
    answer = (security_answer or "").strip().lower()
    if _CUSTOMERS[cid]["answer"] in answer:
        _session["verified"] = cid
        _state("guardrail.verified", customer=cid, attempts_failed=_session["failed_verifications"])
        return "Identity verified. You may now list the customer's cards."
    _session["failed_verifications"] += 1
    left = 3 - _session["failed_verifications"]
    _state("guardrail.verify_failed", customer=cid, attempts_left=left)
    return f"ERROR: Wrong answer. Attempts left: {left}."


def list_cards() -> str:
    """Step 3: List the verified customer's cards (type + last four digits only).

    Returns:
      Card list, or an error if identity is not verified yet.
    """
    cid = _session["verified"]
    if cid is None:
        return "ERROR: Identity not verified. Complete identify_customer and verify_identity first."
    cards = _CUSTOMERS[cid]["cards"]
    lines = [f"{c['type']} ending {c['last4']} ({c['status']})" for c in cards]
    return "Cards on file: " + "; ".join(lines)


def block_card(card_last4: str, reason: str) -> str:
    """Step 4: Block a card permanently. Irreversible in this demo.

    Only call after the caller explicitly confirmed card AND reason.

    Args:
      card_last4: Last four digits of the card to block, e.g. "4821"
      reason: One of: lost, stolen, fraud, damaged

    Returns:
      Success message with a reference number, or an error.
    """
    cid = _session["verified"]
    if cid is None:
        return "ERROR: Identity not verified. Never block a card without verification."
    r = (reason or "").strip().lower()
    if r not in VALID_REASONS:
        return f"ERROR: Invalid reason '{reason}'. Valid reasons: lost, stolen, fraud, damaged."
    last4 = "".join(ch for ch in (card_last4 or "") if ch.isdigit())
    for c in _CUSTOMERS[cid]["cards"]:
        if c["last4"] == last4:
            if c["status"] == "blocked":
                return f"ERROR: The {c['type']} ending {c['last4']} is already blocked."
            c["status"] = "blocked"
            digits = "".join(random.choices(string.digits, k=6))
            ref_spoken = "C B " + " ".join(digits)
            _state("bank.card_blocked", card=c["ref"], reason=r, reference=f"CB-{digits}")
            return (f"SUCCESS: {c['type']} ending {c['last4']} is now blocked "
                    f"(reason: {r}). Tell the caller the card is blocked and state the "
                    f"reference number EXACTLY like this, without changing it: {ref_spoken}. "
                    f"Do NOT ask the caller to repeat it.")
    return f"ERROR: No card ending {card_last4} on file. Use list_cards and let the caller choose."


def order_replacement_card(card_last4: str) -> str:
    """Optional step 5: Order a replacement for a blocked card.

    Args:
      card_last4: Last four digits of the blocked card

    Returns:
      Confirmation with delivery estimate, or an error.
    """
    cid = _session["verified"]
    if cid is None:
        return "ERROR: Identity not verified."
    last4 = "".join(ch for ch in (card_last4 or "") if ch.isdigit())
    for c in _CUSTOMERS[cid]["cards"]:
        if c["last4"] == last4:
            if c["status"] != "blocked":
                return "ERROR: Replacement is only possible for a blocked card."
            return (f"SUCCESS: Replacement {c['type']} ordered. "
                    f"It will arrive by mail within 5 business days.")
    return f"ERROR: No card ending {card_last4} on file."
