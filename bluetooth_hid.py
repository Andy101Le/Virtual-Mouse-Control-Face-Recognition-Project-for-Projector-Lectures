"""
bluetooth_hid.py
─────────────────
Makes the Raspberry Pi act as a Bluetooth HID *peripheral* — i.e. the
paired PC sees it as a real Bluetooth touchpad/mouse at the OS driver
level, indistinguishable from physical hardware.

This is the piece that actually moves the cursor. The web UI never
touches Bluetooth; it only tells this module to become pairable. Once a
PC is paired, the cursor is driven straight from the Pi over HID and the
website is out of the loop entirely.

HOW IT WORKS
  - Registers a HID service record with BlueZ via
    org.bluez.ProfileManager1.RegisterProfile (the modern D-Bus way;
    replaces the old `sdptool add` approach, which no longer works on
    current BlueZ).
  - Listens on two L2CAP sockets: PSM 17 (0x11) = control channel,
    PSM 19 (0x13) = interrupt channel. Input reports go out over the
    interrupt channel.
  - Sends ABSOLUTE-position pointer reports (a digitizer-style
    descriptor), not relative deltas.

WHY ABSOLUTE, NOT RELATIVE
  The whole control-zone design in this project maps a fixed camera
  region onto the full screen: touching a corner of the control zone
  must put the cursor in the corresponding screen corner. That only
  survives if coordinates are absolute. Relative deltas would accumulate
  drift and lose the corner-to-corner correspondence entirely.

  Caveat worth knowing: absolute HID pointers are handled somewhat
  inconsistently across host OSes. Linux and Windows generally accept
  this descriptor fine. macOS is fussier about absolute pointers that
  declare themselves as Mouse rather than Digitizer. If the cursor
  doesn't track on a given host, that descriptor choice is the first
  thing to suspect.

SETUP REQUIRED ON THE PI (see SETUP_BLUETOOTH.md)
  BlueZ's own `input` plugin claims PSM 17/19, so bluetoothd must be
  started with --noplugin=input or binding these sockets will fail with
  EADDRINUSE. This module needs to run as root (raw L2CAP + D-Bus system
  bus).
"""

import logging
import socket
import struct
import threading

log = logging.getLogger(__name__)

try:
    import dbus
    import dbus.service
    import dbus.mainloop.glib
    _HAS_DBUS = True
except ImportError:  # allows import on a dev laptop without BlueZ
    _HAS_DBUS = False

# L2CAP PSMs defined by the Bluetooth HID profile spec
PSM_CONTROL   = 17
PSM_INTERRUPT = 19

HID_UUID = "00001124-0000-1000-8000-00805f9b34fb"   # HumanInterfaceDeviceService
PROFILE_DBUS_PATH = "/org/bluez/hid_touchpad_profile"

# Absolute coordinates are reported in this range on both axes. The
# gesture pipeline works in normalized 0.0–1.0, so we just scale into
# this and the host maps it across the full screen.
ABS_MAX = 32767

# Button bitmask positions in the report
BTN_LEFT   = 0x01
BTN_RIGHT  = 0x02
BTN_MIDDLE = 0x04

REPORT_ID = 0x02
# HID transport header byte for the interrupt channel: 0xA1 = (DATA << 4) | INPUT
HID_INPUT_HEADER = 0xA1

# ── HID report descriptor: absolute pointer + 3 buttons + relative wheel ─────
# Report layout (7 bytes after the transport header):
#   [0] report id (0x02)
#   [1] buttons bitmask
#   [2:4] X, uint16 little-endian, 0..ABS_MAX  (ABSOLUTE)
#   [4:6] Y, uint16 little-endian, 0..ABS_MAX  (ABSOLUTE)
#   [6] wheel, int8 (RELATIVE — wheels are inherently relative)
HID_REPORT_DESCRIPTOR = bytes([
    0x05, 0x01,        # Usage Page (Generic Desktop)
    0x09, 0x02,        # Usage (Mouse)
    0xA1, 0x01,        # Collection (Application)
    0x85, REPORT_ID,   #   Report ID (2)
    0x09, 0x01,        #   Usage (Pointer)
    0xA1, 0x00,        #   Collection (Physical)

    0x05, 0x09,        #     Usage Page (Button)
    0x19, 0x01,        #     Usage Minimum (Button 1)
    0x29, 0x03,        #     Usage Maximum (Button 3)
    0x15, 0x00,        #     Logical Minimum (0)
    0x25, 0x01,        #     Logical Maximum (1)
    0x75, 0x01,        #     Report Size (1)
    0x95, 0x03,        #     Report Count (3)
    0x81, 0x02,        #     Input (Data, Variable, Absolute)
    0x75, 0x05,        #     Report Size (5)
    0x95, 0x01,        #     Report Count (1)
    0x81, 0x03,        #     Input (Constant)  — padding to a full byte

    0x05, 0x01,        #     Usage Page (Generic Desktop)
    0x09, 0x30,        #     Usage (X)
    0x09, 0x31,        #     Usage (Y)
    0x15, 0x00,        #     Logical Minimum (0)
    0x26, 0xFF, 0x7F,  #     Logical Maximum (32767)
    0x75, 0x10,        #     Report Size (16 bits)
    0x95, 0x02,        #     Report Count (2)
    0x81, 0x02,        #     Input (Data, Variable, ABSOLUTE)

    0x09, 0x38,        #     Usage (Wheel)
    0x15, 0x81,        #     Logical Minimum (-127)
    0x25, 0x7F,        #     Logical Maximum (127)
    0x75, 0x08,        #     Report Size (8)
    0x95, 0x01,        #     Report Count (1)
    0x81, 0x06,        #     Input (Data, Variable, Relative)

    0xC0,              #   End Collection
    0xC0,              # End Collection
])


def _sdp_record_xml():
    """
    BlueZ ProfileManager1 wants the SDP record as an XML string. The
    HIDDescriptorList must carry the exact report descriptor bytes above,
    or the host will parse our reports against the wrong layout and the
    cursor will jump around nonsensically.
    """
    desc_hex = "".join(f"{b:02x}" for b in HID_REPORT_DESCRIPTOR)
    return f"""<?xml version="1.0" encoding="UTF-8" ?>
<record>
  <attribute id="0x0001">
    <sequence><uuid value="0x1124" /></sequence>
  </attribute>
  <attribute id="0x0004">
    <sequence>
      <sequence>
        <uuid value="0x0100" />
        <uint16 value="0x0011" />
      </sequence>
      <sequence><uuid value="0x0011" /></sequence>
    </sequence>
  </attribute>
  <attribute id="0x0005">
    <sequence><uuid value="0x1002" /></sequence>
  </attribute>
  <attribute id="0x0006">
    <sequence>
      <uint16 value="0x656e" />
      <uint16 value="0x006a" />
      <uint16 value="0x0100" />
    </sequence>
  </attribute>
  <attribute id="0x0009">
    <sequence>
      <sequence>
        <uuid value="0x1124" />
        <uint16 value="0x0100" />
      </sequence>
    </sequence>
  </attribute>
  <attribute id="0x000d">
    <sequence>
      <sequence>
        <sequence>
          <uuid value="0x0100" />
          <uint16 value="0x0013" />
        </sequence>
        <sequence><uuid value="0x0011" /></sequence>
      </sequence>
    </sequence>
  </attribute>
  <attribute id="0x0100">
    <text value="Gesture Touchpad" />
  </attribute>
  <attribute id="0x0101">
    <text value="Lecture gesture control virtual touchpad" />
  </attribute>
  <attribute id="0x0102">
    <text value="Virtual Mouse Project" />
  </attribute>
  <attribute id="0x0200"><uint16 value="0x0100" /></attribute>
  <attribute id="0x0201"><uint16 value="0x0111" /></attribute>
  <attribute id="0x0202"><uint8 value="0x40" /></attribute>
  <attribute id="0x0203"><uint8 value="0x00" /></attribute>
  <attribute id="0x0204"><boolean value="false" /></attribute>
  <attribute id="0x0205"><boolean value="true" /></attribute>
  <attribute id="0x0206">
    <sequence>
      <sequence>
        <uint8 value="0x22" />
        <text encoding="hex" value="{desc_hex}" />
      </sequence>
    </sequence>
  </attribute>
  <attribute id="0x0207">
    <sequence>
      <sequence>
        <uint16 value="0x0409" />
        <uint16 value="0x0100" />
      </sequence>
    </sequence>
  </attribute>
  <attribute id="0x020b"><boolean value="true" /></attribute>
  <attribute id="0x020c"><uint16 value="0x0c80" /></attribute>
  <attribute id="0x020d"><boolean value="false" /></attribute>
  <attribute id="0x020e"><boolean value="false" /></attribute>
</record>
"""


class BluetoothHIDDevice:
    """
    Owns the two L2CAP listening sockets and the outbound report stream.

    Lifecycle:
        hid = BluetoothHIDDevice()
        hid.register_profile()     # tell BlueZ we speak HID
        hid.start()                # background accept loop
        hid.move_absolute(0.5, 0.5)
        hid.click('left')
        hid.stop()
    """

    def __init__(self):
        self._ctrl_sock = None
        self._intr_sock = None
        self._ctrl_conn = None
        self._intr_conn = None
        self._peer_addr = None

        self._running = False
        self._accept_thread = None
        self._lock = threading.Lock()

        # Cached pointer state — every report must carry the full state,
        # since HID reports are absolute snapshots, not diffs.
        self._x = 0
        self._y = 0
        self._buttons = 0

    # ── BlueZ profile registration ──────────────────────────────────────────
    def register_profile(self):
        """
        Registers the HID SDP record so remote hosts discover us as a
        HID device. Without this, a PC can pair with the Pi but will not
        recognise it as a pointing device.
        """
        if not _HAS_DBUS:
            raise RuntimeError(
                "python3-dbus is not installed — Bluetooth HID needs it. "
                "Install with: sudo apt install python3-dbus"
            )

        dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
        bus = dbus.SystemBus()
        manager = dbus.Interface(
            bus.get_object("org.bluez", "/org/bluez"),
            "org.bluez.ProfileManager1",
        )

        # RegisterProfile's third argument is a{sv} — a dict of string→VARIANT.
        # A plain Python dict makes python-dbus guess the value type from the
        # first entry (ServiceRecord, a str), infer a{ss}, and then fail on the
        # first dbus.Boolean with "Expected a string or unicode object". Wrapping
        # it in dbus.Dictionary(..., signature="sv") forces each value to be
        # marshalled as a variant, which is what BlueZ actually wants.
        opts = dbus.Dictionary(
            {
                "ServiceRecord":          dbus.String(_sdp_record_xml()),
                "Role":                   dbus.String("server"),
                "RequireAuthentication":  dbus.Boolean(True),
                "RequireAuthorization":   dbus.Boolean(False),
                "AutoConnect":            dbus.Boolean(True),
            },
            signature="sv",
        )

        try:
            # RegisterProfile's first argument must marshal as D-Bus signature
            # 'o' (object path), not 's' (string). A plain Python str for
            # PROFILE_DBUS_PATH serialises as 's', so BlueZ sees a call with
            # signature "ssa{sv}" and reports RegisterProfile as nonexistent
            # (no overload matches) even though the method itself is fine.
            manager.RegisterProfile(dbus.ObjectPath(PROFILE_DBUS_PATH), HID_UUID, opts)
            log.info("HID profile registered with BlueZ")
        except dbus.exceptions.DBusException as e:
            # Already-registered is benign on a restart; anything else isn't.
            if "AlreadyExists" in str(e):
                log.info("HID profile already registered — continuing")
            else:
                raise

    # ── Socket lifecycle ────────────────────────────────────────────────────
    def start(self):
        """Bind both L2CAP PSMs and begin accepting a host connection."""
        self._ctrl_sock = socket.socket(
            socket.AF_BLUETOOTH, socket.SOCK_SEQPACKET, socket.BTPROTO_L2CAP)
        self._intr_sock = socket.socket(
            socket.AF_BLUETOOTH, socket.SOCK_SEQPACKET, socket.BTPROTO_L2CAP)

        for sock, psm in ((self._ctrl_sock, PSM_CONTROL),
                          (self._intr_sock, PSM_INTERRUPT)):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("00:00:00:00:00:00", psm))
            except OSError as e:
                raise RuntimeError(
                    f"Could not bind L2CAP PSM {psm}: {e}. "
                    "This almost always means BlueZ's own input plugin holds "
                    "the HID PSMs — start bluetoothd with --noplugin=input "
                    "(see SETUP_BLUETOOTH.md). Root is also required."
                ) from e
            sock.listen(1)

        self._running = True
        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()
        log.info("HID L2CAP sockets listening on PSM %d/%d", PSM_CONTROL, PSM_INTERRUPT)

    def _accept_loop(self):
        """
        Waits for the paired host to open the HID channels. The host is
        the one that initiates this connection (typically right after
        pairing, and again automatically on every reconnect), which is
        why this is a loop rather than a one-shot accept.
        """
        while self._running:
            try:
                ctrl_conn, ctrl_info = self._ctrl_sock.accept()
                intr_conn, intr_info = self._intr_sock.accept()
            except OSError:
                if self._running:
                    log.exception("HID accept failed")
                return

            with self._lock:
                self._ctrl_conn = ctrl_conn
                self._intr_conn = intr_conn
                self._peer_addr = ctrl_info[0]
            log.info("HID host connected: %s", self._peer_addr)

    @property
    def connected(self):
        with self._lock:
            return self._intr_conn is not None

    @property
    def peer_address(self):
        with self._lock:
            return self._peer_addr

    # ── Report sending ──────────────────────────────────────────────────────
    def _send(self):
        """Push the current pointer state as one HID input report."""
        with self._lock:
            conn = self._intr_conn
            if conn is None:
                return False
            report = struct.pack(
                "<BBBHHb",
                HID_INPUT_HEADER,
                REPORT_ID,
                self._buttons,
                self._x,
                self._y,
                0,          # wheel; set via scroll()
            )
            try:
                conn.send(report)
                return True
            except OSError:
                # Host went away (laptop slept, moved out of range, etc).
                # Drop the connection so _accept_loop can take a fresh one.
                log.warning("HID host disconnected")
                self._intr_conn = None
                self._ctrl_conn = None
                self._peer_addr = None
                return False

    def move_absolute(self, nx, ny):
        """
        nx, ny: normalized 0.0–1.0 position. (0,0) is top-left of the
        host's screen, (1,1) bottom-right — which is exactly what the
        control zone's corners are defined to map onto.
        """
        nx = min(max(float(nx), 0.0), 1.0)
        ny = min(max(float(ny), 0.0), 1.0)
        with self._lock:
            self._x = int(nx * ABS_MAX)
            self._y = int(ny * ABS_MAX)
        return self._send()

    def set_buttons(self, left=False, right=False, middle=False):
        with self._lock:
            self._buttons = ((BTN_LEFT if left else 0) |
                             (BTN_RIGHT if right else 0) |
                             (BTN_MIDDLE if middle else 0))
        return self._send()

    def click(self, button="left", hold_s=0.02):
        """Press and release. hold_s gives the host a frame to latch the
        press — sending press and release back-to-back with no gap gets
        coalesced and dropped by some hosts."""
        import time
        kwargs = {"left": button == "left",
                  "right": button == "right",
                  "middle": button == "middle"}
        if not self.set_buttons(**kwargs):
            return False
        time.sleep(hold_s)
        return self.set_buttons()

    def scroll(self, clicks):
        """clicks: positive scrolls up, negative down. Wheel is relative."""
        with self._lock:
            conn = self._intr_conn
            if conn is None:
                return False
            report = struct.pack(
                "<BBBHHb",
                HID_INPUT_HEADER, REPORT_ID, self._buttons,
                self._x, self._y,
                max(-127, min(127, int(clicks))),
            )
            try:
                conn.send(report)
                return True
            except OSError:
                self._intr_conn = None
                self._ctrl_conn = None
                self._peer_addr = None
                return False

    # ── Teardown ────────────────────────────────────────────────────────────
    def stop(self):
        self._running = False
        with self._lock:
            for s in (self._intr_conn, self._ctrl_conn,
                      self._intr_sock, self._ctrl_sock):
                if s is not None:
                    try:
                        s.close()
                    except OSError:
                        pass
            self._ctrl_conn = self._intr_conn = None
            self._ctrl_sock = self._intr_sock = None
            self._peer_addr = None
        log.info("HID device stopped")
