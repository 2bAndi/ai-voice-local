r"""Step 4b: diagnostic call test — codec pinned, two sample formats.
Plays the announcement TWICE: first variant A (unsigned 8 bit), pause, then
variant B (signed 8 bit). Note which one sounds clean.

Usage:
    python tests\test_call.py
"""
import os
import socket
import sys
import threading
import time
import wave
import warnings
import audioop
from pathlib import Path

warnings.filterwarnings("ignore")  # silence audioop deprecation + RTP payload warnings

import pyVoIP
from pyVoIP.RTP import PayloadType
from pyVoIP.VoIP import VoIPPhone, InvalidStateError

# Pin the codec to PCMU (u-law) -> no more ambiguous negotiation
pyVoIP.RTPCompatibleCodecs = [PayloadType.PCMU, PayloadType.EVENT]

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from agent import settings  # noqa: E402
GREETING = PROJECT / "audio" / "greeting_8k.wav"


def local_ip(fritzbox: str) -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect((fritzbox, 5060))
    ip = s.getsockname()[0]
    s.close()
    return ip


def load_greeting_variants():
    with wave.open(str(GREETING), "rb") as w:
        pcm16 = w.readframes(w.getnframes())
    pcm8_signed = audioop.lin2lin(pcm16, 2, 1)
    pcm8_unsigned = audioop.bias(pcm8_signed, 1, 128)
    return pcm8_unsigned, pcm8_signed


VARIANT_A = b""  # unsigned
VARIANT_B = b""  # signed


def answer(call):
    print(">> Call answered. Playing variant A (unsigned) ...")
    try:
        call.answer()
        call.write_audio(VARIANT_A)
        time.sleep(len(VARIANT_A) / 8000 + 1.5)
        print(">> Now variant B (signed) ...")
        call.write_audio(VARIANT_B)
        time.sleep(len(VARIANT_B) / 8000 + 1)
        call.hangup()
        print(">> Done, hung up. Which variant was clean - A (the first) or B (the second)?")
    except InvalidStateError:
        pass


if __name__ == "__main__":
    sip = settings.sip_config()
    print(f"Configuration: {settings.CONFIG_PATH}")
    VARIANT_A, VARIANT_B = load_greeting_variants()
    my_ip = local_ip(sip["server"])
    print(f"Registering {sip['user']}@{sip['server']} from {my_ip} (codec: PCMU only) ...")

    phone = VoIPPhone(
        sip["server"], int(sip.get("port", 5060)),
        sip["user"], sip["password"],
        callCallback=answer, myIP=my_ip,
    )
    phone.start()
    print("Phone active. Call in and listen to both announcements. Enter (or Ctrl+C) quits.")
    try:
        input()
    except (KeyboardInterrupt, EOFError):
        pass
    print("Shutting down ...")
    t = threading.Thread(target=phone.stop, daemon=True)
    t.start()
    t.join(timeout=5)          # pyVoIP occasionally hangs on stop
    os._exit(0)                # then exit hard - fine for a test script
