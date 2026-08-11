"""
web_server.py
──────────────
Flask + Socket.IO control panel for the gesture system. Replaces the
Tkinter login app entirely.

WHAT THIS SERVER IS (AND ISN'T)
  It is: a control surface. Log in, pair your laptop to the Pi, watch
  the camera, nudge the PTZ, register your face, manage users.

  It is NOT in the cursor path. Once your laptop is paired, the Pi talks
  Bluetooth HID straight to it at the OS driver level. You can close this
  tab, shut the browser, walk away — the touchpad keeps working. That's
  the entire point of the design, and it's why the browser never touches
  Bluetooth (which it couldn't do anyway: Web Bluetooth is central-role
  only, and Firefox doesn't implement it at all).

ROUTES
  /            → redirect to dashboard or login
  /login       → sign in
  /signup      → create account (first account becomes admin)
  /dashboard   → camera preview, PTZ/zoom controls, pairing, face registration
  /admin       → user + device management (admins only)

Run:  sudo .venv/bin/python web_server.py
(root is required for raw L2CAP sockets — see SETUP_BLUETOOTH.md)
"""

import functools
import logging
import os
import secrets
import uuid

from flask import (Flask, render_template, request, redirect, url_for,
                   session, jsonify, flash)
from flask_socketio import SocketIO

from user_database import UserDatabase
from bluetooth_hid import BluetoothHIDDevice
from bluetooth_manager import BluetoothManager, PairingState
from ptz_controller import PTZController
from ptz_manual import ManualPTZ, MODE_AUTO, MODE_MANUAL
from gesture_session import GestureSession
from control_registry import ControlRegistry

logging.basicConfig(level=logging.INFO,


                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATABASE_NAME = os.path.join(_SCRIPT_DIR, "login_system.db")

app = Flask(__name__)
# Persist the key so sessions survive a restart; regenerate if absent.
app.config["SECRET_KEY"] = os.environ.get("GESTURE_SECRET_KEY") or secrets.token_hex(32)
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")

db      = UserDatabase(DATABASE_NAME)
hid     = BluetoothHIDDevice()
btmgr   = BluetoothManager()

# TILT_GPIO_PIN=18 routes tilt-servo pulses out that GPIO pin instead of
# the Arducam board, whose tilt channel is dead — see gpio_tilt.py for
# the diagnosis and wiring. Unset = drive the board as normal.
#
# PTZ_ZOOM=1 enables the OPTICAL auto-zoom (the zoom motor). It defaults
# OFF because this board's zoom channel has never been confirmed to move
# the lens — it is the same board whose tilt channel ACKs every write
# while driving nothing, and a register readback proves only that the
# write was accepted. Run test_zoom_sweep.py first: if the field of view
# visibly narrows (not just sharpens — that is the focus motor), set
# PTZ_ZOOM=1, and if the LOW end of the range is the telephoto one, also
# set PTZController.ZOOM_TELE_AT_MAX = False. Digital auto-zoom in
# zoom_webcam.py is unaffected by this and stays on either way.
_tilt_pin = os.environ.get("TILT_GPIO_PIN")
ptz     = PTZController(active_user=None,
                        tilt_gpio_pin=int(_tilt_pin) if _tilt_pin else None,
                        enable_zoom=os.environ.get("PTZ_ZOOM", "0") == "1")
manual  = ManualPTZ(ptz)
gesture = GestureSession(db, hid, ptz, socketio)


# ── Auth helpers ────────────────────────────────────────────────────────────
# Sessions live in a signed cookie, so the server can't delete them when the
# user simply closes the site. Instead each login gets a device_id and a
# sign-out epoch: login stamps the current epoch into the cookie, and closing
# the last dashboard tab on THAT device bumps it, which retroactively
# invalidates the cookie sitting in that browser. In-memory on purpose —
# after a server restart the epoch resets to 0 and old cookies (epoch >= 0)
# stay valid, preserving the existing sessions-survive-a-restart behaviour.
#
# Keyed by device_id, not username: the same account is routinely open on a
# laptop and a desktop at once, and closing the laptop tab must not sign the
# desktop out from under the presenter. That means one entry per login rather
# than one per account, so this grows with use — a few dozen bytes each, and
# a server restart clears it, which is well within scale for one classroom Pi.
_signout_epoch = {}


def _session_valid():
    user = session.get("user")
    if user is None:
        return False
    # Cookies issued before device_ids existed have no key here and read 0,
    # so they stay valid until their next login upgrades them.
    if session.get("epoch", 0) < _signout_epoch.get(session.get("device_id"), 0):
        session.clear()   # cookie predates a close-tab sign-out — kill it
        return False
    return True


def login_required(fn):
    @functools.wraps(fn)
    def wrapper(*a, **kw):
        if not _session_valid():
            return redirect(url_for("login", next=request.path))
        return fn(*a, **kw)
    return wrapper


def admin_required(fn):
    @functools.wraps(fn)
    def wrapper(*a, **kw):
        if not _session_valid():
            return redirect(url_for("login", next=request.path))
        if not session.get("is_admin"):
            return render_template("error.html",
                                   code=403,
                                   message="This page is for admins only."), 403
        return fn(*a, **kw)
    return wrapper


def current_user():
    return session.get("user")


# ── Pages ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    if _session_valid():
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    # Bootstrap: with no accounts at all, send the first visitor to signup
    # so they can create the admin account.
    if not db.admin_exists():
        return redirect(url_for("signup"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        row = db.check_login(username, password)
        if row:
            session["user"]      = row["username"]
            session["is_admin"]  = bool(row["is_admin"])
            # A fresh device_id per login: this browser's sign-out state is
            # its own, independent of the same account on another machine.
            session["device_id"] = uuid.uuid4().hex
            session["epoch"]     = 0
            return redirect(request.args.get("next") or url_for("dashboard"))
        flash("That username and password don't match an account.", "error")

    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    first_account = not db.admin_exists()

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm  = request.form.get("confirm", "")

        if not username or not password:
            flash("Enter both a username and a password.", "error")
        elif password != confirm:
            flash("Those passwords don't match.", "error")
        elif db.add_user(username, password, is_admin=first_account):
            flash("Account created. Sign in to pair a computer.", "ok")
            return redirect(url_for("login"))
        else:
            flash("That username is already taken.", "error")

    return render_template("signup.html", first_account=first_account)


@app.route("/logout")
def logout():
    # Scoped to THIS browser: drop its sockets and release the cursor if it
    # held it, then re-derive who the camera follows. Clearing the tracked
    # user outright (the old behaviour) stranded every other device on the
    # same account — they stayed connected and authenticated, but the
    # recogniser had been rebuilt to match nobody, so gestures silently died
    # with no way to recover short of a page refresh.
    _forget_device(session.get("device_id"))
    session.clear()
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    user = current_user()
    # A session restored from a pre-device_id cookie (or from before a server
    # restart) needs an identity before it can own control or sign itself out.
    if not session.get("device_id"):
        session["device_id"] = uuid.uuid4().hex
        session["epoch"] = 0   # brand-new id, so nothing can have revoked it
    # Tracking is (re-)armed by the socket connect this page is about to make,
    # which is also what decides who holds the cursor.
    return render_template(
        "dashboard.html",
        user=user,
        is_admin=session.get("is_admin"),
        has_face=db.has_face(user),
        devices=db.get_bt_devices(user),
        limits=manual.limits(),
    )


@app.route("/admin")
@admin_required
def admin():
    return render_template(
        "admin.html",
        user=current_user(),
        users=db.get_all_users(),
        devices=db.get_bt_devices(),
    )


# ── Bluetooth pairing API ───────────────────────────────────────────────────
@app.post("/api/bt/pair/start")
@login_required
def bt_pair_start():
    if not btmgr.available:
        return jsonify(error="Bluetooth isn't available on this host. "
                             "Install python3-dbus and python3-gi."), 503
    try:
        btmgr.begin_pairing()
    except Exception as e:
        log.exception("begin_pairing failed")
        return jsonify(error=str(e)), 500
    return jsonify(ok=True, status=btmgr.status())


@app.post("/api/bt/pair/start_permanent")
@login_required
def bt_pair_start_permanent():
    if not btmgr.available:
        return jsonify(error="Bluetooth isn't available on this host. "
                             "Install python3-dbus and python3-gi."), 503
    try:
        btmgr.begin_permanent_pairing()
    except Exception as e:
        log.exception("begin_permanent_pairing failed")
        return jsonify(error=str(e)), 500
    return jsonify(ok=True, status=btmgr.status())


def _claim_last_paired(reassign):
    """
    Attach the most recently paired MAC to the logged-in account (the line
    that makes the device belong to the account) and return the pairing
    status snapshot.

    Claims on any of three conditions:
      - reassign=True: an explicit approval (the confirm endpoint) always
        takes ownership, so re-confirming a MAC moves it between accounts.
      - needs_claim: bluetooth_manager.py sets this once a device auto-pairs
        under Just Works (whether that came through the agent or the
        PropertiesChanged fallback — see its module docstring). There is no
        confirm click in that flow, so this is what actually claims most
        devices now. mark_claimed() below clears the flag on the same call
        that consumes it, so it can't latch True forever and can't double-fire
        even though the dashboard polls every second.
      - the MAC has no owner yet: covers the older confirm-based flow, where
        the confirm response itself can be lost to the pairing-time
        Wi-Fi/Bluetooth radio stall, leaving a paired-but-unowned device for
        the next status poll to adopt.
    """
    st = btmgr.status()
    if st["state"] != PairingState.PAIRED or not st["last_paired"]:
        return st
    mac = st["last_paired"]
    if reassign or st["needs_claim"] or db.get_bt_device_owner(mac.upper()) is None:
        name = next((d["name"] for d in btmgr.list_paired_devices()
                     if d["mac"].upper() == mac.upper()), "Paired computer")
        db.add_bt_device(mac, name, current_user())
        btmgr.mark_claimed()
        log.info("Device %s claimed by %s", mac, current_user())
    return st


@app.post("/api/bt/pair/confirm")
@login_required
def bt_pair_confirm():
    approve = bool(request.json.get("approve", True))
    if not btmgr.confirm_pending(approve):
        return jsonify(error="No pairing request is waiting."), 409

    # Give BlueZ a moment to finish the bond before we read back the MAC.
    socketio.sleep(1.0)
    st = _claim_last_paired(reassign=True) if approve else btmgr.status()

    return jsonify(ok=True, status=st)


@app.post("/api/bt/pair/cancel")
@login_required
def bt_pair_cancel():
    btmgr.cancel_pairing()
    return jsonify(ok=True, status=btmgr.status())


@app.get("/api/bt/status")
@login_required
def bt_status():
    if btmgr.available:
        # Claims a freshly auto-paired device under Just Works pairing
        # (needs_claim, set by bluetooth_manager.py) and, for the older
        # confirm-based flow, adopts an unowned device left behind by a
        # confirm response lost to the pairing-time radio stall.
        _claim_last_paired(reassign=False)
    pairing = btmgr.status()
    paired = btmgr.list_paired_devices() if btmgr.available else []
    owners = {d["mac"].upper(): d["username"] for d in db.get_bt_devices()}
    for d in paired:
        d["owner"] = owners.get(d["mac"].upper())
    return jsonify(
        pairing=pairing,
        devices=paired,
        hid_connected=hid.connected,
        hid_peer=hid.peer_address,
    )


@app.post("/api/bt/forget")
@login_required
def bt_forget():
    mac = (request.json.get("mac") or "").upper()
    owner = db.get_bt_device_owner(mac)
    # A non-admin may only unpair their own device.
    if owner != current_user() and not session.get("is_admin"):
        return jsonify(error="That device belongs to another account."), 403
    btmgr.remove_device(mac)
    db.remove_bt_device(mac)
    control.drop_mac(mac)
    return jsonify(ok=True)


# ── Cursor control ownership ────────────────────────────────────────────────
@app.get("/api/control")
@login_required
def control_status():
    return jsonify(_control_payload())


@app.post("/api/control/take")
@login_required
def control_take():
    """
    Claim the cursor for one specific paired machine. The browser can't
    discover its own Bluetooth MAC, so the machine is named explicitly from
    the caller's paired list rather than inferred from the connection.
    """
    mac = (request.json.get("mac") or "").upper()
    if not mac:
        return jsonify(error="Pick which computer you're sitting at."), 400

    owner = db.get_bt_device_owner(mac)
    if owner is None:
        return jsonify(error="That computer isn't paired."), 404
    if owner != current_user() and not session.get("is_admin"):
        return jsonify(error="That computer belongs to another account."), 403

    name = next((d["name"] for d in db.get_bt_devices() if d["mac"] == mac), None)
    control.take(request.json.get("sid"), current_user(),
                 session.get("device_id"), mac, name)
    return jsonify(_control_payload())


@app.post("/api/control/release")
@login_required
def control_release():
    holder = control.holder
    if holder is not None and holder["user"] != current_user() \
            and not session.get("is_admin"):
        return jsonify(error="Someone else holds control."), 403
    control.release("released from the dashboard")
    return jsonify(_control_payload())


# ── PTZ + zoom API ──────────────────────────────────────────────────────────
@app.post("/api/ptz/mode")
@login_required
def ptz_mode():
    mode = request.json.get("mode")
    if mode not in (MODE_AUTO, MODE_MANUAL):
        return jsonify(error="Mode must be 'auto' or 'manual'."), 400
    manual.set_mode(mode)
    return jsonify(ok=True, status=manual.status())


@app.post("/api/ptz/nudge")
@login_required
def ptz_nudge():
    if manual.mode != MODE_MANUAL:
        return jsonify(error="Switch to manual control first."), 409
    manual.nudge(d_pan=int(request.json.get("pan", 0)),
                 d_tilt=int(request.json.get("tilt", 0)))
    return jsonify(ok=True, status=manual.status())


@app.post("/api/ptz/set")
@login_required
def ptz_set():
    if manual.mode != MODE_MANUAL:
        return jsonify(error="Switch to manual control first."), 409
    body = request.json or {}
    if "pan" in body:
        manual.set_pan(int(body["pan"]))
    if "tilt" in body:
        manual.set_tilt(int(body["tilt"]))
    if "hw_zoom" in body:
        manual.set_hw_zoom(int(body["hw_zoom"]))
    return jsonify(ok=True, status=manual.status())


@app.post("/api/ptz/center")
@login_required
def ptz_center():
    if manual.mode != MODE_MANUAL:
        return jsonify(error="Switch to manual control first."), 409
    manual.center()
    return jsonify(ok=True, status=manual.status())


@app.post("/api/zoom")
@login_required
def zoom_set():
    body = request.json or {}
    if "enabled" in body:
        gesture.set_zoom_enabled(bool(body["enabled"]))
    if "max_zoom" in body:
        gesture.set_zoom_cap(float(body["max_zoom"]))
    return jsonify(ok=True, telemetry=gesture.telemetry())


@app.post("/api/controls")
@login_required
def controls_set():
    """
    Manual override for the open-palm master switch. Exists so the user is
    never stranded: if the gesture stops registering — poor light, hand out
    of frame, controls switched off from across the room — the browser can
    always put control back.
    """
    body = request.json or {}
    if "enabled" not in body:
        return jsonify(error="Missing 'enabled'."), 400
    state = gesture.control_toggle.set_enabled(bool(body["enabled"]))
    socketio.emit("controls_toggled", gesture.control_toggle.status())
    return jsonify(ok=True, controls_enabled=state,
                   telemetry=gesture.telemetry())


@app.get("/api/ptz/status")
@login_required
def ptz_status():
    return jsonify(ptz=manual.status(), telemetry=gesture.telemetry())


# ── Autofocus API ────────────────────────────────────────────────────────────
@app.post("/api/focus")
@login_required
def focus_set():
    body = request.json or {}
    if "auto" in body:
        gesture.set_focus_auto(bool(body["auto"]))
    if "value" in body:
        gesture.set_focus_manual(int(body["value"]))
    if body.get("refocus"):
        gesture.trigger_refocus()
    return jsonify(ok=True, focus=gesture.focus_status(),
                   telemetry=gesture.telemetry())


@app.get("/api/focus/status")
@login_required
def focus_status():
    return jsonify(focus=gesture.focus_status(), telemetry=gesture.telemetry())


# ── Face registration API ───────────────────────────────────────────────────
@app.post("/api/face/start")
@login_required
def face_start():
    gesture.begin_face_capture(current_user())
    return jsonify(ok=True)


@app.post("/api/face/cancel")
@login_required
def face_cancel():
    gesture.cancel_face_capture()
    return jsonify(ok=True)


@app.post("/api/face/clear")
@login_required
def face_clear():
    db.clear_face_embedding(current_user())
    if gesture.auth is not None:
        gesture.auth.face_rec.reload_db()
    return jsonify(ok=True)


# ── Admin API ───────────────────────────────────────────────────────────────
@app.post("/api/admin/user/add")
@admin_required
def admin_user_add():
    b = request.json or {}
    username = (b.get("username") or "").strip()
    password = b.get("password") or ""
    if not username or not password:
        return jsonify(error="Enter both a username and a password."), 400
    if not db.add_user(username, password, bool(b.get("is_admin"))):
        return jsonify(error="That username is already taken."), 409
    return jsonify(ok=True)


@app.post("/api/admin/user/delete")
@admin_required
def admin_user_delete():
    username = (request.json or {}).get("username")
    if username == current_user():
        return jsonify(error="You can't delete the account you're signed in to."), 400
    if db.get_is_admin(username) and db.count_admins() <= 1:
        return jsonify(error="This is the last admin account — promote another "
                             "admin before deleting it."), 400
    db.delete_user(username)
    return jsonify(ok=True)


@app.post("/api/admin/user/rename")
@admin_required
def admin_user_rename():
    b = request.json or {}
    old, new = b.get("username"), (b.get("new_username") or "").strip()
    if not new:
        return jsonify(error="Enter a new username."), 400
    if not db.change_username(old, new):
        return jsonify(error="That username is already taken."), 409
    if old == current_user():
        session["user"] = new
    return jsonify(ok=True)


@app.post("/api/admin/user/password")
@admin_required
def admin_user_password():
    b = request.json or {}
    password = b.get("password") or ""
    if not password:
        return jsonify(error="Enter a new password."), 400
    db.change_password(b.get("username"), password)
    return jsonify(ok=True)


@app.post("/api/admin/user/toggle_admin")
@admin_required
def admin_user_toggle():
    username = (request.json or {}).get("username")
    is_admin = db.get_is_admin(username)
    if is_admin is None:
        return jsonify(error="No such account."), 404
    if is_admin and db.count_admins() <= 1:
        return jsonify(error="This is the last admin account."), 400
    if is_admin and username == current_user():
        return jsonify(error="You can't remove your own admin access "
                             "while signed in."), 400
    db.set_admin(username, not is_admin)
    return jsonify(ok=True)


@app.post("/api/admin/user/clear_face")
@admin_required
def admin_clear_face():
    db.clear_face_embedding((request.json or {}).get("username"))
    if gesture.auth is not None:
        gesture.auth.face_rec.reload_db()
    return jsonify(ok=True)


@app.post("/api/admin/reset")
@admin_required
def admin_reset():
    db.reset_database()
    session.clear()
    return jsonify(ok=True)


# ── Socket.IO, presence and control ownership ───────────────────────────────
# Presence gating: the dashboard holds a Socket.IO connection whenever a
# logged-in tab is open, so "zero authenticated sockets" == "nobody is
# using the system" — and the detection pipeline suspends (no tracking,
# no gestures, no cursor) until someone comes back. The grace period
# keeps a page refresh or the pairing flow's location.reload() from
# bouncing the pipeline.
#
# Control ownership lives in control_registry.py — see there for why it is
# scoped per device rather than per account.
VIEWER_GRACE_S = 10.0

control = ControlRegistry(
    on_active_user=lambda u: gesture.set_active_user(u),
    on_preferred_peer=lambda m: hid.set_preferred_peer(m),
    on_broadcast=lambda: socketio.emit("control_changed", _control_payload()),
)


def _control_payload():
    return {"holder": control.holder, "hid_peer": hid.peer_address}


def _forget_device(device_id):
    """Sign one browser out: drop its viewers and its claim on the cursor,
    then kill the cookie still sitting in it."""
    control.forget_device(device_id)
    if device_id is not None:
        _signout_epoch[device_id] = _signout_epoch.get(device_id, 0) + 1


def _suspend_if_no_viewers():
    socketio.sleep(VIEWER_GRACE_S)
    if not control.any_viewers():
        gesture.set_suspended(True)


def _sign_out_device_if_gone(device_id):
    """
    A device's last tab closing IS a sign-out, not a pause — but only after
    the grace period, so a refresh or the pairing flow's location.reload()
    doesn't count.
    """
    socketio.sleep(VIEWER_GRACE_S)
    if control.has_device(device_id):
        return   # came back (refresh) — still signed in
    log.info("Last dashboard tab closed on device %s — signed out", device_id)
    _forget_device(device_id)


@socketio.on("connect")
def on_connect():
    if not _session_valid():
        return False   # reject unauthenticated / signed-out socket connections
    control.add_viewer(request.sid, session.get("user"),
                       session.get("device_id"))
    gesture.set_suspended(False)
    socketio.emit("telemetry", gesture.telemetry())
    socketio.emit("control_changed", _control_payload(), to=request.sid)


@socketio.on("disconnect")
def on_disconnect():
    device_id = control.remove_viewer(request.sid)
    if device_id is not None:
        socketio.start_background_task(_sign_out_device_if_gone, device_id)
    socketio.start_background_task(_suspend_if_no_viewers)


# ── Boot ────────────────────────────────────────────────────────────────────
def _boot_bluetooth():
    """
    Bluetooth failures are non-fatal: the preview, PTZ controls, and admin
    pages should still work so the user can see *why* it failed rather than
    getting a dead server.
    """
    try:
        btmgr.start()
        hid.register_profile()
        # The reconnect provider makes the Pi dial already-paired hosts on
        # startup (we advertise HIDReconnectInitiate, so hosts wait for us)
        # — without it, cursor control only worked right after a fresh
        # pairing and a restart needed a forget + re-pair.
        hid.start(reconnect_targets=lambda: [
            d["mac"] for d in btmgr.list_paired_devices()
        ] if btmgr.available else [])
        log.info("Bluetooth HID ready — pair a computer from the dashboard.")
    except Exception as e:
        log.error("Bluetooth unavailable: %s", e)
        log.error("The web UI will still run, but pairing is disabled. "
                  "See SETUP_BLUETOOTH.md.")


if __name__ == "__main__":
    _boot_bluetooth()
    gesture.start()
    # Boot idle: the camera/models build now (so first login is instant),
    # but nothing tracks until a logged-in dashboard actually connects.
    gesture.set_suspended(True)
    try:
        socketio.run(app, host="0.0.0.0", port=8080,
                     allow_unsafe_werkzeug=True)
    finally:
        gesture.stop()
        hid.stop()
        btmgr.stop()
        ptz.close()
