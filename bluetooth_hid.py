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
  - ALSO dials out: the SDP record advertises HIDReconnectInitiate, so
    paired hosts expect the Pi to initiate reconnection. A background
    loop connects to known paired hosts whenever the channels are down
    (hosts only ever connect to us on their own right after pairing).
  - Sends ABSOLUTE-position pointer reports (a digitizer-style
    descriptor), not relative deltas.
  - Also exposes a minimal keyboard (its own report ID) so gestures can
    send modifier+wheel chords — Ctrl+scroll is what hosts bind to zoom.

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

REPORT_ID     = 0x02   # pointer reports
KBD_REPORT_ID = 0x01   # keyboard reports (conventional combo layout: kbd=1, mouse=2)
# HID transport header byte for the interrupt channel: 0xA1 = (DATA << 4) | INPUT
HID_INPUT_HEADER = 0xA1

# Keyboard modifier bits (byte 0 of the keyboard report)
MOD_LCTRL = 0x01

# ── HID report descriptor: composite pointer + keyboard ─────────────────────
# Two top-level collections, distinguished by report ID:
#
# Pointer report (ID 0x02, 7 bytes after the transport header):
#   [0] report id (0x02)
#   [1] buttons bitmask
#   [2:4] X, uint16 little-endian, 0..ABS_MAX  (ABSOLUTE)
#   [4:6] Y, uint16 little-endian, 0..ABS_MAX  (ABSOLUTE)
#   [6] wheel, int8 (RELATIVE — wheels are inherently relative)
#
# Keyboard report (ID 0x01, 8 bytes after the transport header):
#   [0] report id (0x01)
#   [1] modifier bitmask (LCtrl=0x01, LShift=0x02, ... RGui=0x80)
#   [2] reserved (always 0)
#   [3:9] up to 6 concurrently-held key usage codes
#
# The keyboard exists so gestures can send modifier+wheel chords —
# Ctrl+scroll is what actually zooms on host apps; a bare wheel only
# scrolls (see BTCursorController.handle_action).
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

    0x05, 0x01,          # Usage Page (Generic Desktop)
    0x09, 0x06,          # Usage (Keyboard)
    0xA1, 0x01,          # Collection (Application)
    0x85, KBD_REPORT_ID, #   Report ID (1)

    0x05, 0x07,          #   Usage Page (Keyboard/Keypad)
    0x19, 0xE0,          #   Usage Minimum (Left Control)
    0x29, 0xE7,          #   Usage Maximum (Right GUI)
    0x15, 0x00,          #   Logical Minimum (0)
    0x25, 0x01,          #   Logical Maximum (1)
    0x75, 0x01,          #   Report Size (1)
    0x95, 0x08,          #   Report Count (8)
    0x81, 0x02,          #   Input (Data, Variable, Absolute) — modifier bits

    0x75, 0x08,          #   Report Size (8)
    0x95, 0x01,          #   Report Count (1)
    0x81, 0x03,          #   Input (Constant) — reserved byte

    0x05, 0x07,          #   Usage Page (Keyboard/Keypad)
    0x19, 0x00,          #   Usage Minimum (0)
    0x29, 0x65,          #   Usage Maximum (101)
    0x15, 0x00,          #   Logical Minimum (0)
    0x25, 0x65,          #   Logical Maximum (101)
    0x75, 0x08,          #   Report Size (8)
    0x95, 0x06,          #   Report Count (6)
    0x81, 0x00,          #   Input (Data, Array) — 6 key slots

    0xC0,                # End Collection
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
    <text value="Galaxy Mouse" />
  </attribute>
  <attribute id="0x0101">
    <text value="Lecture gesture control virtual touchpad" />
  </attribute>
  <attribute id="0x0102">
    <text value="Galaxy Mouse Project" />
  </attribute>
  <attribute id="0x0200"><uint16 value="0x0100" /></attribute>
  <attribute id="0x0201"><uint16 value="0x0111" /></attribute>
  <attribute id="0x0202"><uint8 value="0xc0" /></attribute>  <!-- subclass: keyboard + pointing combo -->
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

    # How often to retry dialing paired hosts while disconnected.
    RECONNECT_INTERVAL_S = 8.0

    def __init__(self):
        self._ctrl_sock = None
        self._intr_sock = None
        self._ctrl_conn = None
        self._intr_conn = None
        self._peer_addr = None

        self._running = False
        self._accept_thread = None
        self._reconnect_targets = None
        self._lock = threading.Lock()

        # Which host is allowed to hold the HID link. None = the original
        # free-for-all: accept whoever dials in, and dial every paired host
        # until one answers. Once the dashboard hands control to a specific
        # machine this is that machine's MAC, and every other host is
        # refused — otherwise the accept loop would happily let an idle
        # second PC steal the cursor back the moment it woke up.
        self._preferred_mac = None

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
        # RequireAuthentication=True would demand an authenticated
        # (MITM-protected) link, which Just Works pairing can never provide
        # by definition — bluetooth_manager.py's agent is registered
        # NoInputNoOutput specifically to get Just Works (no passkey), so
        # this must be False or BlueZ ends up unable to satisfy both
        # constraints at once and pairing stalls without completing.
        opts = dbus.Dictionary(
            {
                "ServiceRecord":          dbus.String(_sdp_record_xml()),
                "Role":                   dbus.String("server"),
                "RequireAuthentication":  dbus.Boolean(False),
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
            if "AlreadyExists" in str(e):
                # A previous run's registration is still live in this
                # bluetoothd session. Its ServiceRecord may carry an OLD
                # report descriptor, so re-register rather than keep it —
                # otherwise descriptor changes silently never reach hosts.
                log.info("HID profile already registered — re-registering "
                         "so the current report descriptor is the one served")
                manager.UnregisterProfile(dbus.ObjectPath(PROFILE_DBUS_PATH))
                manager.RegisterProfile(dbus.ObjectPath(PROFILE_DBUS_PATH), HID_UUID, opts)
            else:
                raise

    # ── Socket lifecycle ────────────────────────────────────────────────────
    def start(self, reconnect_targets=None):
        """
        Bind both L2CAP PSMs and begin accepting a host connection.

        reconnect_targets: optional zero-arg callable returning the MACs of
        already-paired hosts. When given, a background loop actively dials
        them whenever the HID channels are down — see _reconnect_loop for
        why the Pi (not the host) must initiate that connection.
        """
        self._reconnect_targets = reconnect_targets
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
        if self._reconnect_targets is not None:
            threading.Thread(target=self._reconnect_loop, daemon=True).start()
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

            peer = ctrl_info[0]
            with self._lock:
                pref = self._preferred_mac
                refuse = not self._peer_allowed(peer)
                if not refuse:
                    self._ctrl_conn = ctrl_conn
                    self._intr_conn = intr_conn
                    self._peer_addr = peer

            if refuse:
                # Not the machine that holds control. Hang up rather than
                # serve it — a sleeping PC that reconnects on wake must not
                # silently take the cursor away from the active presenter.
                log.info("HID connection from %s refused — %s holds control",
                         peer, pref)
                for s in (ctrl_conn, intr_conn):
                    try:
                        s.close()
                    except OSError:
                        pass
                continue

            log.info("HID host connected: %s", peer)

    def _reconnect_loop(self):
        """
        Actively re-open the HID channels to an already-paired host.

        Our SDP record declares HIDReconnectInitiate=true (attribute
        0x0205), which tells hosts that the DEVICE initiates reconnection
        — so after a Pi reboot or server restart, the paired PC just sits
        waiting for us to dial. The accept loop alone only ever produced a
        connection right after fresh pairing (the one time hosts connect
        on their own), which is why cursor control used to require a
        forget + re-pair on every restart. This loop is what real BT mice
        do: while disconnected, periodically dial each paired host until
        one answers.
        """
        import time
        while self._running:
            if not self.connected:
                for mac in self._dial_targets():
                    if not self._running or self.connected:
                        break
                    if self._connect_to_host(mac):
                        break
            time.sleep(self.RECONNECT_INTERVAL_S)

    def _peer_allowed(self, mac):
        """May this host hold the HID link? Everything is allowed until the
        dashboard pins control to one machine. Caller holds the lock."""
        return (self._preferred_mac is None
                or (mac or "").upper() == self._preferred_mac)

    def _dial_targets(self):
        """Which hosts the reconnect loop should try, in order."""
        with self._lock:
            pref = self._preferred_mac
        if pref is not None:
            # Someone holds control: dial only their machine. Dialing the
            # whole paired list here would reconnect the wrong PC and undo
            # the handoff a few seconds after it happened.
            return [pref]
        if self._reconnect_targets is None:
            return []
        try:
            return list(self._reconnect_targets() or [])
        except Exception:
            log.exception("reconnect target lookup failed")
            return []

    def _connect_to_host(self, mac):
        """Dial one host: control channel first, then interrupt, per the
        HID profile spec. Returns True on a fully-open channel pair."""
        ctrl = intr = None
        try:
            ctrl = socket.socket(
                socket.AF_BLUETOOTH, socket.SOCK_SEQPACKET, socket.BTPROTO_L2CAP)
            ctrl.settimeout(5.0)
            ctrl.connect((mac, PSM_CONTROL))
            intr = socket.socket(
                socket.AF_BLUETOOTH, socket.SOCK_SEQPACKET, socket.BTPROTO_L2CAP)
            intr.settimeout(5.0)
            intr.connect((mac, PSM_INTERRUPT))
        except OSError:
            # Host offline / out of range / BT off — normal, retry later.
            for s in (ctrl, intr):
                if s is not None:
                    try:
                        s.close()
                    except OSError:
                        pass
            return False

        ctrl.settimeout(None)
        intr.settimeout(None)
        with self._lock:
            if self._intr_conn is not None or not self._peer_allowed(mac):
                # Either the host beat us to it via the accept loop while we
                # were dialing — keep that connection, drop ours — or control
                # moved to a different machine mid-dial, which makes this
                # connection stale before it was ever used.
                for s in (ctrl, intr):
                    try:
                        s.close()
                    except OSError:
                        pass
                return False
            self._ctrl_conn = ctrl
            self._intr_conn = intr
            self._peer_addr = mac
        log.info("HID reconnected to paired host %s", mac)
        return True

    @property
    def connected(self):
        with self._lock:
            return self._intr_conn is not None

    @property
    def peer_address(self):
        with self._lock:
            return self._peer_addr

    @property
    def preferred_peer(self):
        with self._lock:
            return self._preferred_mac

    def set_preferred_peer(self, mac):
        """
        Point the HID link at one specific paired host, dropping whatever
        host holds it now. Pass None to release the pin and go back to
        accepting/dialing any paired machine.

        Returns immediately: the actual dial runs on a background thread
        because _connect_to_host blocks for up to two 5 s socket timeouts,
        and this is called from a web request handler.
        """
        mac = mac.upper() if mac else None
        with self._lock:
            if mac == self._preferred_mac:
                return
            self._preferred_mac = mac
            stale = (self._peer_addr is not None
                     and mac is not None
                     and self._peer_addr.upper() != mac)
            if stale:
                for s in (self._intr_conn, self._ctrl_conn):
                    if s is not None:
                        try:
                            s.close()
                        except OSError:
                            pass
                self._ctrl_conn = self._intr_conn = None
                self._peer_addr = None

        if stale:
            log.info("HID link released — control moved to %s", mac)
        if mac is not None and self._running:
            threading.Thread(target=self._connect_to_host, args=(mac,),
                             daemon=True).start()

    # ── Report sending ──────────────────────────────────────────────────────
    def _send_report(self, report):
        """Send one raw input report on the interrupt channel."""
        with self._lock:
            conn = self._intr_conn
            if conn is None:
                return False
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

    def _send(self):
        """Push the current pointer state as one HID input report."""
        with self._lock:
            report = struct.pack(
                "<BBBHHb",
                HID_INPUT_HEADER,
                REPORT_ID,
                self._buttons,
                self._x,
                self._y,
                0,          # wheel; set via scroll()
            )
        return self._send_report(report)

    def _send_keyboard(self, modifiers, keys=()):
        """Push one keyboard report: a modifier bitmask plus up to six
        concurrently-held key usage codes (0-padded)."""
        keys = tuple(keys)[:6]
        keys += (0,) * (6 - len(keys))
        report = struct.pack(
            "<BBBB6B",
            HID_INPUT_HEADER, KBD_REPORT_ID,
            modifiers,
            0,              # reserved byte, per the boot-keyboard layout
            *keys,
        )
        return self._send_report(report)

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
            report = struct.pack(
                "<BBBHHb",
                HID_INPUT_HEADER, REPORT_ID, self._buttons,
                self._x, self._y,
                max(-127, min(127, int(clicks))),
            )
        return self._send_report(report)

    def ctrl_scroll(self, clicks, latch_s=0.03):
        """
        Wheel movement with Ctrl held — the near-universal zoom chord
        (browsers, PDFs, Office, image viewers all bind Ctrl+wheel to
        zoom, while a bare wheel just scrolls).

        The sleeps give the host time to latch the modifier before the
        wheel event arrives and to process it before release — chording
        across two report types back-to-back gets misordered by some
        hosts otherwise. Ctrl is ALWAYS released, even if the wheel
        report fails, so a dropped connection can't leave the host with
        a stuck modifier when it reconnects.
        """
        import time
        if not self._send_keyboard(MOD_LCTRL):
            return False
        time.sleep(latch_s)
        ok = self.scroll(clicks)
        time.sleep(latch_s)
        self._send_keyboard(0)
        return ok

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
