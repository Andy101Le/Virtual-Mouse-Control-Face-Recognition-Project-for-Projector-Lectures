"""
Who holds the cursor.

There is one camera and one Bluetooth radio, so exactly one machine can be
driven at a time — but the same account is routinely signed in on a laptop
and a desktop at once (that is the normal case for a presenter: the lecture
PC plus their own laptop). This module owns that arbitration and nothing
else, so it can be tested without a camera, a radio or a Flask app.

Two rules do the real work:

  * **Ownership is per device, not per account.** Signing out on the laptop
    must not disarm the desktop. The old code kept a single process-wide
    "active user" that /logout cleared unconditionally, which left every
    other session on that account connected and authenticated but silently
    dead — the recogniser had been rebuilt to match nobody.

  * **Unclaimed falls back to the newest viewer.** A user with one machine
    should never have to press anything; the handoff UI only matters under
    contention.

The effects (retarget the recogniser, move the HID link) are injected so the
caller decides what "apply" means — see web_server.py for the wiring.
"""
import logging
import threading

log = logging.getLogger(__name__)


class ControlRegistry:
    def __init__(self, on_active_user, on_preferred_peer, on_broadcast=None):
        """
        on_active_user(username|None)  — who the recogniser should follow.
        on_preferred_peer(mac|None)    — which host the HID link should hold;
                                         None releases the pin.
        on_broadcast()                 — optional, called after any change a
                                         connected dashboard should see.
        """
        self._on_active_user   = on_active_user
        self._on_preferred_peer = on_preferred_peer
        self._on_broadcast     = on_broadcast
        # RLock: apply() is called from inside methods that already hold it.
        self._lock    = threading.RLock()
        self._viewers = {}     # sid -> {"user", "device_id"}
        self._holder  = None   # {"sid","user","device_id","mac","name"}

    # ── State ───────────────────────────────────────────────────────────────
    @property
    def holder(self):
        with self._lock:
            return dict(self._holder) if self._holder else None

    def any_viewers(self):
        with self._lock:
            return bool(self._viewers)

    def has_device(self, device_id):
        with self._lock:
            return any(v["device_id"] == device_id
                       for v in self._viewers.values())

    def viewer_count(self):
        with self._lock:
            return len(self._viewers)

    # ── Effects ─────────────────────────────────────────────────────────────
    def apply(self):
        """Push the current decision down to the camera and the radio."""
        with self._lock:
            if self._holder is not None:
                user, mac = self._holder["user"], self._holder["mac"]
            elif self._viewers:
                # Newest viewer wins the implicit claim — dicts keep
                # insertion order, so this is "whoever connected last".
                user, mac = next(reversed(self._viewers.values()))["user"], None
            else:
                user, mac = None, None
        self._on_active_user(user)
        self._on_preferred_peer(mac)

    def _broadcast(self):
        if self._on_broadcast is not None:
            self._on_broadcast()

    # ── Viewers ─────────────────────────────────────────────────────────────
    def add_viewer(self, sid, user, device_id):
        with self._lock:
            self._viewers[sid] = {"user": user, "device_id": device_id}
        self.apply()

    def remove_viewer(self, sid):
        """Returns the device_id that socket belonged to, or None.

        Deliberately does NOT release control: a page refresh disconnects and
        reconnects, and yanking the cursor away mid-refresh would make the
        dashboard unusable. Control survives until forget_device() decides
        the device is really gone.
        """
        with self._lock:
            gone = self._viewers.pop(sid, None)
        return gone["device_id"] if gone else None

    # ── Ownership ───────────────────────────────────────────────────────────
    def take(self, sid, user, device_id, mac, name=None):
        with self._lock:
            self._holder = {"sid": sid, "user": user, "device_id": device_id,
                            "mac": mac.upper(), "name": name}
        log.info("Control taken by '%s' -> %s", user, mac)
        self.apply()
        self._broadcast()
        return self.holder

    def release(self, reason="released"):
        """Drop the holder unconditionally. Returns True if anything changed."""
        with self._lock:
            changed = self._release_locked(reason)
        if changed:
            self.apply()
            self._broadcast()
        return changed

    def _release_locked(self, reason):
        if self._holder is None:
            return False
        log.info("Control released by '%s' (%s) — %s",
                 self._holder["user"], self._holder["mac"], reason)
        self._holder = None
        return True

    def forget_device(self, device_id):
        """
        One browser has stopped being a live viewer (signed out, or its last
        tab closed past the grace period). Drop its sockets and release the
        cursor if it held it — scoped to this device_id alone, so the same
        account on another machine keeps working.
        """
        with self._lock:
            for sid in [s for s, v in self._viewers.items()
                        if v["device_id"] == device_id]:
                self._viewers.pop(sid, None)
            changed = (self._holder is not None
                       and self._holder["device_id"] == device_id
                       and self._release_locked("its device signed out"))
        self.apply()
        if changed:
            self._broadcast()
        return changed

    def drop_mac(self, mac):
        """Unpairing the machine that holds the cursor must not leave the
        radio pinned to a MAC it can no longer reach."""
        with self._lock:
            changed = (self._holder is not None
                       and self._holder["mac"] == mac.upper()
                       and self._release_locked("its computer was unpaired"))
        if changed:
            self.apply()
            self._broadcast()
        return changed
