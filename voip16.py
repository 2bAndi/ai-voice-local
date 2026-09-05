r"""voip16 - monkeypatches for pyVoIP 1.6.x (just import it, the rest happens automatically).

Fixes three problems in the library:
1. Audio was routed through an 8-bit stage internally (ulaw2lin/lin2ulaw with
   width=1), which destroys the G.711 dynamic range -> now 16-bit signed PCM
   end to end.
2. Copy-paste bug: with PCMA negotiated, the PCMU encoder was called -> fixed.
3. Buffer padding was 0x80 (8-bit silence); interpreted as 16-bit data that
   would be loud garbage -> padding is now 0x00 (16-bit silence).

Resulting API convention:
  call.read_audio(length=320, blocking=False)  -> 320 bytes = 20 ms s16le, 8 kHz
  call.write_audio(pcm16_bytes)                -> s16le, 8 kHz, mono
Recommended: read non-blocking and run your own 20 ms loop; silence is then
ordinary audio (zeros) and the VAD logic stays on the clock.

Since Phase 0 of the Glass Box this module also installs observation taps (raw SIP
datagrams, first RTP packets, DTMF) that feed agent/events.py. They change nothing.
"""
import audioop
import time

import pyVoIP
from pyVoIP.RTP import RTPClient, RTPPacketManager, RTPParseError, PayloadType

# Pin the codec to PCMU (unambiguous negotiation with the FRITZ!Box)
pyVoIP.RTPCompatibleCodecs = [PayloadType.PCMU, PayloadType.EVENT]


# --- 16-bit decode (inbound); timestamp counts samples, buffer offset counts bytes ---
def _parse_pcmu(self, packet):
    data = audioop.ulaw2lin(packet.payload, 2)
    self.pmin.write(packet.timestamp * 2, data)


def _parse_pcma(self, packet):
    data = audioop.alaw2lin(packet.payload, 2)
    self.pmin.write(packet.timestamp * 2, data)


# --- 16-bit encode (outbound), including the fix for the PCMA bug ---
def _encode_packet(self, payload):
    if self.preference == PayloadType.PCMU:
        return audioop.lin2ulaw(payload, 2)
    elif self.preference == PayloadType.PCMA:
        return audioop.lin2alaw(payload, 2)
    raise RTPParseError("Unsupported codec (encode): " + str(self.preference))


# --- Sender: now reads 320 bytes (20 ms in 16 bit) per packet ---
def _trans(self):
    while self.NSD:
        last_sent = time.monotonic_ns()
        payload = self.pmout.read(320)
        payload = self.encode_packet(payload)
        packet = b"\x80"
        packet += chr(int(self.preference)).encode("utf8")
        try:
            packet += self.outSequence.to_bytes(2, byteorder="big")
        except OverflowError:
            self.outSequence = 0
        try:
            packet += self.outTimestamp.to_bytes(4, byteorder="big")
        except OverflowError:
            self.outTimestamp = 0
        packet += self.outSSRC.to_bytes(4, byteorder="big")
        packet += payload
        try:
            self.sout.sendto(packet, (self.outIP, self.outPort))
        except OSError:
            pass
        _count_tx(self, payload)
        self.outSequence += 1
        self.outTimestamp += len(payload)  # samples == bytes in the encoded payload
        delay = (1 / self.preference.rate) * 160
        sleep_time = max(0, delay - ((time.monotonic_ns() - last_sent) / 1000000000))
        time.sleep(sleep_time / self.trans_delay_reduction)


# --- Buffer padding: 0x00 instead of 0x80 (= silence in 16 bit) ---
def _pm_read(self, length=160):
    while self.rebuilding:
        time.sleep(0.01)
    with self.bufferLock:
        packet = self.buffer.read(length)
        if len(packet) < length:
            packet = packet + (b"\x00" * (length - len(packet)))
    return packet


# --- read: silence detection adapted to the new padding ---
def _client_read(self, length=160, blocking=True):
    if not blocking:
        return self.pmin.read(length)
    packet = self.pmin.read(length)
    while packet == (b"\x00" * length) and self.NSD:
        time.sleep(0.01)
        packet = self.pmin.read(length)
    return packet


# =============================================================================
# Glass Box taps (Phase 0): raw SIP messages, first RTP packets, DTMF events.
# Everything below only OBSERVES - no behaviour changes. Events go to agent/events.py.
# =============================================================================
import re
import socket
# Order matters: pyVoIP.SIP imports pyVoIP.VoIP at module level and VoIP needs SIP.SIPMessage,
# so the package must be entered through pyVoIP.VoIP (as pyVoIP's own users do).
from pyVoIP.VoIP import VoIPCall
from pyVoIP.SIP import SIPClient

try:
    from agent import events as _ev
except Exception:  # running outside the project (e.g. a bare test) - taps become no-ops
    _ev = None

_INTERESTING = ("Via", "From", "To", "Call-ID", "CSeq", "Contact", "User-Agent", "Server",
                "Allow", "Supported", "Content-Type", "Content-Length", "WWW-Authenticate",
                "Authorization", "Expires", "User-to-User", "P-Asserted-Identity", "Refer-To",
                "Referred-By", "Replaces", "Reason")
_RAW_CAP = 2048


def _sip_summary(raw: bytes, direction: str, peer) -> dict:
    """Reduce a raw SIP datagram to what the Glass Box shows: request/status line, the
    headers that matter for correlation and routing, X-headers, and the SDP media line."""
    text = raw.decode("utf-8", errors="replace")
    head, _, body = text.partition("\r\n\r\n")
    lines = head.split("\r\n")
    first = lines[0] if lines else ""
    headers = {}
    for ln in lines[1:]:
        name, sep, val = ln.partition(":")
        if not sep:
            continue
        name = name.strip()
        if name in _INTERESTING or name.upper().startswith("X-"):
            headers.setdefault(name, val.strip())
    sdp = {}
    if body.startswith("v=0"):
        for ln in body.split("\r\n"):
            if ln.startswith("m="):
                sdp["m"] = ln[2:]
            elif ln.startswith("c="):
                sdp["c"] = ln[2:]
            elif ln.startswith("a=rtpmap:"):
                sdp.setdefault("rtpmap", []).append(ln[9:])
            elif ln.startswith("a=") and ln[2:] in ("sendrecv", "sendonly", "recvonly", "inactive"):
                sdp["dir"] = ln[2:]
    m = re.match(r"([A-Z]+) (\S+) SIP/2\.0", first)
    if m:
        kind, line = f"sip.{direction}.{m.group(1).lower()}", first
    else:
        m2 = re.match(r"SIP/2\.0 (\d{3}) ?(.*)", first)
        kind = f"sip.{direction}.{m2.group(1)}" if m2 else f"sip.{direction}"
        line = first
    cseq = headers.get("CSeq", "")
    return kind, {"line": line, "cseq": cseq, "call_id": headers.get("Call-ID"),
                  "peer": f"{peer[0]}:{peer[1]}" if peer else None,
                  "headers": headers, "sdp": sdp or None,
                  "bytes": len(raw), "raw": text[:_RAW_CAP]}


class _SipTap:
    """Wraps the SIPClient UDP socket: logs every datagram in and out, forwards everything."""

    def __init__(self, sock):
        self._s = sock

    def sendto(self, data, addr):
        if _ev is not None:
            try:
                kind, payload = _sip_summary(data, "tx", addr)
                _ev.emit("AVAYA→SP-A", kind, payload)
            except Exception:
                pass
        return self._s.sendto(data, addr)

    def recv(self, bufsize, *a):
        data = self._s.recv(bufsize, *a)
        self._tap_rx(data, None)
        return data

    def recvfrom(self, bufsize, *a):
        data, addr = self._s.recvfrom(bufsize, *a)
        self._tap_rx(data, addr)
        return data, addr

    def _tap_rx(self, data, addr):
        if _ev is not None and data:
            try:
                kind, payload = _sip_summary(data, "rx", addr)
                _ev.emit("AVAYA→SP-A", kind, payload)
            except Exception:
                pass

    def __getattr__(self, name):
        return getattr(self._s, name)


def _sip_start(self):
    """Re-implementation of SIPClient.start (pyVoIP 1.6.8) that installs the tap socket."""
    from threading import Timer
    if self.NSD:
        raise RuntimeError("Attempted to start already started SIPClient")
    self.NSD = True
    raw_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    raw_sock.bind((self.myIP, self.myPort))
    self.s = _SipTap(raw_sock)
    self.out = self.s
    if _ev is not None:
        _ev.emit("AVAYA→SP-A", "sip.socket", {"bind": f"{self.myIP}:{self.myPort}", "transport": "UDP"})
    self.register()
    t = Timer(1, self.recv_loop)
    t.name = "SIP Recieve"
    t.start()


# --- RTP: first packet in / out per RTPClient, DTMF digits (masked) ---
_orig_parse_packet = RTPClient.parse_packet
_orig_trans = _trans


SLICE_BATCH = 10        # Glass Box: one event per 10 RTP frames (= 200 ms) per direction


def _count_tx(self, payload):
    """SP-A -> Avaya: every packet we send, reported in batches of SLICE_BATCH."""
    if _ev is None:
        return
    b = self.__dict__.setdefault("_gb_tx", {"n": 0, "first": None, "active": False})
    if b["n"] == 0:
        b["first"] = self.outSequence
    b["n"] += 1
    if any(payload):                     # 0x00 padding = nothing queued (silence)
        b["active"] = True
    if b["n"] >= SLICE_BATCH:
        _ev.emit("SP-A→AVAYA", "rtp.tx", {"slices": b["n"], "frame_ms": 20, "seq_first": b["first"],
                                           "seq_last": self.outSequence, "bytes": b["n"] * (12 + len(payload)),
                                           "active": b["active"], "to": f"{self.outIP}:{self.outPort}"})
        b.update(n=0, first=None, active=False)


def _count_rx(self, packet):
    """Avaya -> SP-A: every RTP frame that arrives, reported in batches of SLICE_BATCH."""
    if _ev is None or len(packet) < 12:
        return
    b = self.__dict__.setdefault("_gb_rx", {"n": 0, "first": None, "peak": 0})
    seq = int.from_bytes(packet[2:4], "big")
    if b["n"] == 0:
        b["first"] = seq
    b["n"] += 1
    try:
        pt = packet[1] & 0x7F
        lin = audioop.ulaw2lin(packet[12:], 2) if pt == 0 else audioop.alaw2lin(packet[12:], 2)
        b["peak"] = max(b["peak"], audioop.rms(lin, 2))
    except Exception:
        pass
    if b["n"] >= SLICE_BATCH:
        _ev.emit("AVAYA→SP-A", "rtp.rx", {"slices": b["n"], "frame_ms": 20, "seq_first": b["first"], "seq_last": seq,
                                           "bytes": b["n"] * len(packet), "peak_rms": b["peak"],
                                           "active": b["peak"] > 150, "from": f"{self.outIP}:{self.outPort}"})
        b.update(n=0, first=None, peak=0)


def _tap_parse_packet(self, packet):
    _count_rx(self, packet)
    if not getattr(self, "_gb_rx_seen", False):
        self._gb_rx_seen = True
        if _ev is not None:
            pt = packet[1] & 0x7F
            ssrc = int.from_bytes(packet[8:12], "big")
            _ev.emit("AVAYA→SP-A", "rtp.rx.first",
                     {"payload_type": pt, "codec": str(self.assoc.get(pt, "?")), "ssrc": ssrc,
                      "from": f"{self.outIP}:{self.outPort}", "to_port": self.inPort,
                      "bytes": len(packet), "frame_ms": 20})
    return _orig_parse_packet(self, packet)


def _tap_trans(self):
    if _ev is not None:
        _ev.emit("SP-A→AVAYA", "rtp.tx.start",
                 {"codec": str(self.preference), "to": f"{self.outIP}:{self.outPort}",
                  "ssrc": self.outSSRC, "frame_ms": 20, "bytes_per_frame": 160})
    return _orig_trans(self)


_orig_dtmf_cb = VoIPCall.dtmf_callback


def _tap_dtmf(self, code):
    # FR-COM-014: keypad digits are intercepted in the session controller and never reach
    # the model or the transcript. The event carries a mask, not the digit.
    if _ev is not None:
        _ev.emit("SP-A→SP-B", "rtp.dtmf", {"digit": "*" if code in "0123456789" else code,
                                              "masked": code in "0123456789", "rfc": 4733})
    return _orig_dtmf_cb(self, code)


def apply():
    RTPClient.parse_pcmu = _parse_pcmu
    RTPClient.parse_pcma = _parse_pcma
    RTPClient.encode_packet = _encode_packet
    RTPClient.trans = _tap_trans
    RTPClient.read = _client_read
    RTPPacketManager.read = _pm_read
    RTPClient.parse_packet = _tap_parse_packet
    SIPClient.start = _sip_start
    VoIPCall.dtmf_callback = _tap_dtmf


apply()
