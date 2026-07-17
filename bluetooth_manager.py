"""
bluetooth_manager.py
─────────────────────
Everything about *pairing* — as opposed to bluetooth_hid.py, which
handles the HID data stream once a host is paired.

This is what the website's "Pair a computer" button drives. The browser
does none of this; it just POSTs to the Pi, and the Pi makes itself
discoverable and accepts the pairing request. The user then pairs from
their laptop's normal Bluetooth settings, where the Pi shows up as a
touchpad.

Uses BlueZ over D-Bus:
  - org.bluez.Adapter1  → Powered / Discoverable / Pairable properties
  - org.bluez.AgentManager1 → register a pairing agent
  - org.bluez.Device1   → enumerate, trust, and remove paired devices

The agent runs in DisplayYesNo mode: BlueZ hands us a 6-digit passkey,
we surface it to the web UI so the user can confirm it matches what
their laptop shows, then we confirm. That's a real security boundary —
auto-accepting any pairing request would let a passerby pair with the
Pi and inherit cursor control.
"""

import logging
import threading
import time

log = logging.getLogger(__name__)

try:
    import dbus
    import dbus.service
    import dbus.mainloop.glib
    from gi.repository import GLib
    _HAS_DBUS = True
except ImportError:  # dev laptop without BlueZ
    _HAS_DBUS = False
    GLib = None

BLUEZ_SERVICE   = "org.bluez"
ADAPTER_IFACE   = "org.bluez.Adapter1"
DEVICE_IFACE    = "org.bluez.Device1"
AGENT_IFACE     = "org.bluez.Agent1"
AGENT_PATH      = "/org/bluez/gesture_agent"
PROPS_IFACE     = "org.freedesktop.DBus.Properties"
OBJMGR_IFACE    = "org.freedesktop.DBus.ObjectManager"


class PairingState:
    """Snapshot of what the web UI needs to render the pairing panel."""
    IDLE       = "idle"
    PAIRABLE   = "pairable"     # discoverable, waiting for a host
    CONFIRMING = "confirming"   # passkey shown, waiting for user confirm
    PAIRED     = "paired"
    FAILED     = "failed"


if _HAS_DBUS:

    class _Agent(dbus.service.Object):
        """
        BlueZ calls into this when a host tries to pair. We don't
        auto-accept: RequestConfirmation parks the request and surfaces
        the passkey, and the pairing only completes once the logged-in
        user confirms it in the web UI.
        """

        def __init__(self, bus, path, manager):
            super().__init__(bus, path)
            self._manager = manager

        @dbus.service.method(AGENT_IFACE, in_signature="", out_signature="")
        def Release(self):
            log.info("Pairing agent released")

        @dbus.service.method(AGENT_IFACE, in_signature="ou", out_signature="")
        def RequestConfirmation(self, device, passkey):
            log.info("Pairing confirmation requested for %s (passkey %06d)",
                     device, passkey)
            confirmed = self._manager._await_user_confirmation(
                str(device), f"{passkey:06d}")
            if not confirmed:
                raise dbus.DBusException(
                    "org.bluez.Error.Rejected",
                    "User rejected pairing in the web UI")

        @dbus.service.method(AGENT_IFACE, in_signature="o", out_signature="")
        def RequestAuthorization(self, device):
            confirmed = self._manager._await_user_confirmation(str(device), None)
            if not confirmed:
                raise dbus.DBusException(
                    "org.bluez.Error.Rejected", "User rejected pairing")

        @dbus.service.method(AGENT_IFACE, in_signature="os", out_signature="")
        def AuthorizeService(self, device, uuid):
            # The host is asking to use our HID service. If it's already
            # paired and trusted, this is the expected reconnect path.
            log.info("Service authorized for %s (%s)", device, uuid)

        @dbus.service.method(AGENT_IFACE, in_signature="", out_signature="")
        def Cancel(self):
            log.info("Pairing cancelled by remote")
            self._manager._cancel_confirmation()


class BluetoothManager:
    def __init__(self, adapter_name="hci0", device_alias="Gesture Touchpad"):
        self.adapter_name = adapter_name
        self.device_alias = device_alias

        self._bus = None
        self._adapter = None
        self._adapter_props = None
        self._agent = None
        self._glib_loop = None

        self._lock = threading.Lock()
        self._state = PairingState.IDLE
        self._pending = None          # {'device','passkey','event','approved'}
        self._last_paired_mac = None
        self._error = None

    @property
    def available(self):
        return _HAS_DBUS

    # ── Setup ───────────────────────────────────────────────────────────────
    def start(self):
        """Connect to BlueZ and register the pairing agent."""
        if not _HAS_DBUS:
            raise RuntimeError(
                "python3-dbus / PyGObject not installed — Bluetooth pairing "
                "needs them. Install: sudo apt install python3-dbus python3-gi"
            )

        dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
        self._bus = dbus.SystemBus()

        path = f"/org/bluez/{self.adapter_name}"
        obj = self._bus.get_object(BLUEZ_SERVICE, path)
        self._adapter = dbus.Interface(obj, ADAPTER_IFACE)
        self._adapter_props = dbus.Interface(obj, PROPS_IFACE)

        self._adapter_props.Set(ADAPTER_IFACE, "Powered", dbus.Boolean(True))
        self._adapter_props.Set(ADAPTER_IFACE, "Alias", dbus.String(self.device_alias))

        self._agent = _Agent(self._bus, AGENT_PATH, self)
        agent_mgr = dbus.Interface(
            self._bus.get_object(BLUEZ_SERVICE, "/org/bluez"),
            "org.bluez.AgentManager1")
        agent_mgr.RegisterAgent(AGENT_PATH, "DisplayYesNo")
        agent_mgr.RequestDefaultAgent(AGENT_PATH)

        # BlueZ delivers agent callbacks on a GLib main loop, so it needs
        # its own thread — the Flask/SocketIO server owns the main one.
        self._glib_loop = GLib.MainLoop()
        threading.Thread(target=self._glib_loop.run, daemon=True).start()

        log.info("Bluetooth manager started on %s as '%s'",
                 self.adapter_name, self.device_alias)

    # ── Pairing flow ────────────────────────────────────────────────────────
    def begin_pairing(self, timeout_s=180):
        """
        Make the Pi discoverable so the user's laptop can find it. Called
        when a logged-in user clicks "Pair a computer".
        """
        with self._lock:
            self._state = PairingState.PAIRABLE
            self._error = None
            self._last_paired_mac = None

        self._adapter_props.Set(ADAPTER_IFACE, "DiscoverableTimeout",
                                dbus.UInt32(timeout_s))
        self._adapter_props.Set(ADAPTER_IFACE, "PairableTimeout",
                                dbus.UInt32(timeout_s))
        self._adapter_props.Set(ADAPTER_IFACE, "Discoverable", dbus.Boolean(True))
        self._adapter_props.Set(ADAPTER_IFACE, "Pairable", dbus.Boolean(True))
        log.info("Discoverable for %ds", timeout_s)

    def cancel_pairing(self):
        with self._lock:
            self._state = PairingState.IDLE
            self._cancel_confirmation_locked()
        try:
            self._adapter_props.Set(ADAPTER_IFACE, "Discoverable", dbus.Boolean(False))
            self._adapter_props.Set(ADAPTER_IFACE, "Pairable", dbus.Boolean(False))
        except Exception:
            log.exception("Failed to clear discoverable")

    def _await_user_confirmation(self, device_path, passkey, timeout_s=60):
        """
        Blocks the BlueZ agent callback until the web user approves or
        rejects. Runs on the GLib thread, so blocking here is fine — it
        does not stall Flask.
        """
        event = threading.Event()
        with self._lock:
            self._pending = {
                "device":   device_path,
                "passkey":  passkey,
                "event":    event,
                "approved": False,
            }
            self._state = PairingState.CONFIRMING

        if not event.wait(timeout_s):
            with self._lock:
                self._state = PairingState.FAILED
                self._error = "Pairing confirmation timed out"
                self._pending = None
            return False

        with self._lock:
            approved = bool(self._pending and self._pending["approved"])
            mac = self._mac_from_path(device_path)
            if approved:
                self._state = PairingState.PAIRED
                self._last_paired_mac = mac
            else:
                self._state = PairingState.FAILED
                self._error = "Pairing rejected"
            self._pending = None

        if approved:
            # Trust the device so it can silently reconnect its HID
            # channels later without a fresh authorization prompt —
            # this is what makes "pair once, never touch the site again"
            # actually hold.
            self._trust_device(device_path)
        return approved

    def confirm_pending(self, approve=True):
        """Called from the web UI's Confirm / Reject buttons."""
        with self._lock:
            if not self._pending:
                return False
            self._pending["approved"] = bool(approve)
            self._pending["event"].set()
            return True

    def _cancel_confirmation(self):
        with self._lock:
            self._cancel_confirmation_locked()

    def _cancel_confirmation_locked(self):
        if self._pending:
            self._pending["approved"] = False
            self._pending["event"].set()
            self._pending = None

    def _trust_device(self, device_path):
        try:
            props = dbus.Interface(
                self._bus.get_object(BLUEZ_SERVICE, device_path), PROPS_IFACE)
            props.Set(DEVICE_IFACE, "Trusted", dbus.Boolean(True))
            log.info("Device trusted: %s", device_path)
        except Exception:
            log.exception("Could not trust device %s", device_path)

    # ── Device queries ──────────────────────────────────────────────────────
    @staticmethod
    def _mac_from_path(device_path):
        # /org/bluez/hci0/dev_AA_BB_CC_DD_EE_FF -> AA:BB:CC:DD:EE:FF
        tail = device_path.rsplit("/", 1)[-1]
        if tail.startswith("dev_"):
            return tail[4:].replace("_", ":").upper()
        return None

    def list_paired_devices(self):
        """Returns [{'mac','name','connected','trusted'}] straight from BlueZ."""
        if not _HAS_DBUS or self._bus is None:
            return []
        out = []
        mgr = dbus.Interface(
            self._bus.get_object(BLUEZ_SERVICE, "/"), OBJMGR_IFACE)
        for path, ifaces in mgr.GetManagedObjects().items():
            dev = ifaces.get(DEVICE_IFACE)
            if not dev or not bool(dev.get("Paired", False)):
                continue
            out.append({
                "mac":       str(dev.get("Address", "")),
                "name":      str(dev.get("Name", dev.get("Alias", "Unknown"))),
                "connected": bool(dev.get("Connected", False)),
                "trusted":   bool(dev.get("Trusted", False)),
            })
        return out

    def remove_device(self, mac):
        """Unpair at the BlueZ level. The DB mapping is removed separately."""
        if not _HAS_DBUS:
            return False
        dev_path = f"/org/bluez/{self.adapter_name}/dev_{mac.replace(':', '_').upper()}"
        try:
            self._adapter.RemoveDevice(dev_path)
            log.info("Removed paired device %s", mac)
            return True
        except Exception:
            log.exception("Failed to remove device %s", mac)
            return False

    def status(self):
        with self._lock:
            pending = None
            if self._pending:
                pending = {
                    "passkey": self._pending["passkey"],
                    "mac":     self._mac_from_path(self._pending["device"]),
                }
            return {
                "state":       self._state,
                "pending":     pending,
                "last_paired": self._last_paired_mac,
                "error":       self._error,
            }

    def stop(self):
        self.cancel_pairing()
        if self._glib_loop:
            self._glib_loop.quit()
