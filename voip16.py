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


def apply():
    RTPClient.parse_pcmu = _parse_pcmu
    RTPClient.parse_pcma = _parse_pcma
    RTPClient.encode_packet = _encode_packet
    RTPClient.trans = _trans
    RTPClient.read = _client_read
    RTPPacketManager.read = _pm_read


apply()
