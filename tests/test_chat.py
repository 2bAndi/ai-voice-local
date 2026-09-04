r"""Step 6: dialog core as a text chat (no audio).
Qwen3 via Ollama runs the appointment dialog and uses the real calendar tools
against Radicale. A hundred times faster to debug than over the phone.

Legacy note: this exercises the original appointment demo (agent/tools.py).
The current target use case (credit-card blocking) lives in agent/bank_tools.py
and is driven by agent\call_agent.py.

Prerequisites:
  - Ollama running (service), model pulled:  ollama pull qwen3:8b
  - Radicale running (own window):          .\start_radicale.ps1
  - pip install caldav ollama

Usage:
    python tests\test_chat.py
Quit with  exit
"""
import datetime as dt
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from ollama import chat  # noqa: E402
from agent.tools import check_availability, book_appointment, send_confirmation  # noqa: E402

MODEL = "qwen3:8b"
TOOLS = {
    "check_availability": check_availability,
    "book_appointment": book_appointment,
    "send_confirmation": send_confirmation,
}

now = dt.datetime.now()

SYSTEM = f"""You are the friendly telephone assistant for booking appointments.
Today is {now.strftime('%A, %d %B %Y')}, the time is {now.strftime('%H:%M')}.

Your job: take down an appointment with the mandatory details name, reason, date and time.
You ask for missing mandatory details one at a time - including the reason ("What is it about?").

Rules:
- Reply in at most 1-2 short sentences. No lists, no enumerations - your replies are read out loud.
- Ask only ONE question at a time.
- NEVER invent details the caller did not provide - e-mail addresses above all. If a confirmation
  is wanted, ask for the address and read it back to check it.
- Resolve relative dates ("next Wednesday") into a concrete date yourself and state it for
  confirmation ("That would be Wednesday the twelfth of August"). If the statement is ambiguous
  (e.g. "next Tuesday" when tomorrow is a Tuesday), briefly ask which day is meant.
- Before booking, summarize name, reason, date and time and get an explicit yes. Only then call
  book_appointment. Never book while a mandatory detail is missing.
- Appointments are available Monday to Friday between 9 am and 5 pm, 30 minutes each.
- If the requested slot is taken, offer concrete free alternatives on the same day
  (from check_availability).
- After booking, ask whether an e-mail confirmation is wanted.
- Stay on the topic of appointment booking."""


def run_turn(messages: list) -> None:
    while True:
        resp = chat(MODEL, messages=messages, tools=list(TOOLS.values()), think=False)
        msg = resp.message
        messages.append(msg)
        if msg.tool_calls:
            for tc in msg.tool_calls:
                fname = tc.function.name
                args = dict(tc.function.arguments)
                fn = TOOLS.get(fname)
                try:
                    result = fn(**args) if fn else f"ERROR: unknown tool {fname}"
                except Exception as e:
                    result = f"ERROR: {e}"
                print(f"   [{fname}({json.dumps(args, ensure_ascii=False)}) -> {result}]")
                messages.append({
                    "role": "tool",
                    "tool_name": fname,
                    "content": json.dumps(result, ensure_ascii=False),
                })
            continue
        print(f"AGENT: {msg.content.strip()}\n")
        return


if __name__ == "__main__":
    print(f"Appointment dialog test ({MODEL}). 'exit' quits.\n")
    messages = [{"role": "system", "content": SYSTEM}]
    messages.append({"role": "user", "content": "(The caller is connected now. Greet them briefly.)"})
    run_turn(messages)
    while True:
        try:
            user = input("YOU:   ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if user.lower() in ("exit", "quit"):
            break
        if not user:
            continue
        messages.append({"role": "user", "content": user})
        run_turn(messages)
