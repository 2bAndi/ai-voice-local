r"""Appointment tools of the agent. They talk CalDAV to Radicale (127.0.0.1:5232).

These functions are handed to the LLM as tools; the docstrings and type
annotations serve as the schema description for function calling.

Legacy note: this is the tool set of the original appointment-booking demo.
The target use case (credit-card blocking) uses agent/bank_tools.py instead.
"""
import datetime as dt
import uuid

import caldav

CALDAV_URL = "http://127.0.0.1:5232"
CAL_NAME = "appointments"   # renamed from "termine"; a new calendar is created on first use
WORK_START = 9      # slots from 09:00 ...
WORK_END = 17       # ... until 17:00
SLOT_MIN = 30

_client = None
_calendar = None
_last_checked = {"date": None}

WD = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _get_calendar():
    global _client, _calendar
    if _calendar is None:
        _client = caldav.DAVClient(url=CALDAV_URL, username="agent", password="agent")
        principal = _client.principal()
        cals = [c for c in principal.calendars() if c.name == CAL_NAME]
        _calendar = cals[0] if cals else principal.make_calendar(name=CAL_NAME)
    return _calendar


def check_availability(date: str) -> dict:
    """Check free appointment slots on a given day. ALWAYS call this before booking.

    Args:
      date: Date in format YYYY-MM-DD, e.g. 2026-08-11

    Returns:
      Dict with date, weekday and free_slots (start times such as "09:00").
      On weekends free_slots is empty.
    """
    day = dt.date.fromisoformat(date)
    _last_checked["date"] = date
    if day.weekday() >= 5:
        return {"date": date, "weekday": WD[day.weekday()],
                "free_slots": [], "note": "Weekend - no appointments (Mon-Fri, 9 am to 5 pm)"}
    cal = _get_calendar()
    start = dt.datetime.combine(day, dt.time(0, 0))
    end = start + dt.timedelta(days=1)
    busy = []
    for ev in cal.search(start=start, end=end, event=True, expand=True):
        comp = ev.icalendar_component
        b0 = comp["DTSTART"].dt
        b1 = comp["DTEND"].dt if "DTEND" in comp else b0 + dt.timedelta(minutes=SLOT_MIN)
        if isinstance(b0, dt.datetime):
            busy.append((b0.replace(tzinfo=None), b1.replace(tzinfo=None)))
    free = []
    slot = dt.datetime.combine(day, dt.time(WORK_START, 0))
    day_end = dt.datetime.combine(day, dt.time(WORK_END, 0))
    while slot + dt.timedelta(minutes=SLOT_MIN) <= day_end:
        slot_end = slot + dt.timedelta(minutes=SLOT_MIN)
        if not any(b0 < slot_end and b1 > slot for b0, b1 in busy):
            free.append(slot.strftime("%H:%M"))
        slot = slot_end
    return {"date": date, "weekday": WD[day.weekday()], "free_slots": free}


def book_appointment(name: str, topic: str, start_iso: str, duration_min: int = 30) -> str:
    """Book an appointment bindingly in the calendar.

    Only call this after the caller has explicitly confirmed date and time.

    Args:
      name: Full name of the caller
      topic: Reason/subject of the appointment
      start_iso: Start in format YYYY-MM-DDTHH:MM, e.g. 2026-08-11T14:00
      duration_min: Duration in minutes (default 30)

    Returns:
      Confirmation text with appointment ID, or an error message if the slot is taken.
    """
    start = dt.datetime.fromisoformat(start_iso)
    wd = WD[start.weekday()]
    clean = (name or "").strip()
    if (len(clean) < 2 or "name of" in clean.lower()
            or clean.lower() in {"caller", "unknown", "customer", "n.n.", "nn", "name"}):
        return "ERROR: You do not have a real name yet. Ask the caller for their name first."
    if _last_checked["date"] != start.date().isoformat():
        return (f"ERROR: Call check_availability for {start.date().isoformat()} first. "
                f"Note: that date is a {wd}.")
    if start.weekday() >= 5:
        return f"ERROR: {start.strftime('%d %B %Y')} is a {wd} - there are no appointments on weekends."
    if start.strftime("%H:%M") not in check_availability(start.date().isoformat())["free_slots"]:
        return "ERROR: That slot is not free. Please pick another time."
    end = start + dt.timedelta(minutes=duration_min)
    uid = uuid.uuid4().hex[:8]
    cal = _get_calendar()
    cal.save_event(
        dtstart=start,
        dtend=end,
        summary=f"Appointment: {name} - {topic}",
        uid=f"agent-{uid}",
    )
    return (f"Booked: {wd}, {start.strftime('%d %B %Y at %H:%M')}, {name} ({topic}). "
            f"Appointment ID {uid}. Tell the caller this weekday.")


def send_confirmation(email: str, name: str, start_iso: str) -> str:
    """Send an appointment confirmation by e-mail (currently simulated only).

    ONLY call this if the caller stated their e-mail address verbatim in this
    conversation. Never invent or guess an address.

    Args:
      email: E-mail address stated by the caller themselves
      name: Name of the caller
      start_iso: Appointment start in format YYYY-MM-DDTHH:MM

    Returns:
      Status message.
    """
    e = (email or "").strip().lower()
    if "@" not in e or "example." in e:
        return ("ERROR: This e-mail address was not stated by the caller, or it is invalid. "
                "Ask the caller for their address and read it back for confirmation.")
    # TODO next stage: real delivery via smtplib + IMAP append
    return f"(simulated) Confirmation to {email} for {start_iso} queued."
