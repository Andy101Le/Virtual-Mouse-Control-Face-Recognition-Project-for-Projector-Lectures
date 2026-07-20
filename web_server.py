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

from flask import (Flask, render_template, request, redirect, url_for,
                   session, jsonify, flash)
from flask_socketio import SocketIO

from user_database import UserDatabase
from bluetooth_hid import BluetoothHIDDevice
from bluetooth_manager import BluetoothManager, PairingState
from ptz_controller import PTZController
from ptz_manual import ManualPTZ, MODE_AUTO, MODE_MANUAL
from gesture_session import GestureSession

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
ptz     = PTZController(active_user=None)
manual  = ManualPTZ(ptz)
gesture = GestureSession(db, hid, ptz, socketio)


# ── Auth helpers ────────────────────────────────────────────────────────────
def login_required(fn):
    @functools.wraps(fn)
    def wrapper(*a, **kw):
        if "user" not in session:
            return redirect(url_for("login", next=request.path))
        return fn(*a, **kw)
    return wrapper


def admin_required(fn):
    @functools.wraps(fn)
    def wrapper(*a, **kw):
        if "user" not in session:
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
    if "user" in session:
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
            session["user"]     = row["username"]
            session["is_admin"] = bool(row["is_admin"])
            gesture.set_active_user(row["username"])
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
    session.clear()
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    user = current_user()
    # Re-sync the tracker's target in case this is a session restored after a
    # server restart (login POST wouldn't have run). Idempotent when unchanged.
    gesture.set_active_user(user)
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


@app.post("/api/bt/pair/confirm")
@login_required
def bt_pair_confirm():
    approve = bool(request.json.get("approve", True))
    if not btmgr.confirm_pending(approve):
        return jsonify(error="No pairing request is waiting."), 409

    # Give BlueZ a moment to finish the bond before we read back the MAC.
    socketio.sleep(1.0)
    st = btmgr.status()

    if approve and st["state"] == PairingState.PAIRED and st["last_paired"]:
        mac = st["last_paired"]
        name = next((d["name"] for d in btmgr.list_paired_devices()
                     if d["mac"].upper() == mac.upper()), "Paired computer")
        # This is the line that makes the device belong to the account.
        db.add_bt_device(mac, name, current_user())
        log.info("Device %s claimed by %s", mac, current_user())

    return jsonify(ok=True, status=st)


@app.post("/api/bt/pair/cancel")
@login_required
def bt_pair_cancel():
    btmgr.cancel_pairing()
    return jsonify(ok=True, status=btmgr.status())


@app.get("/api/bt/status")
@login_required
def bt_status():
    paired = btmgr.list_paired_devices() if btmgr.available else []
    owners = {d["mac"].upper(): d["username"] for d in db.get_bt_devices()}
    for d in paired:
        d["owner"] = owners.get(d["mac"].upper())
    return jsonify(
        pairing=btmgr.status(),
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
    return jsonify(ok=True)


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
    return jsonify(ok=True)


@app.post("/api/admin/reset")
@admin_required
def admin_reset():
    db.reset_database()
    session.clear()
    return jsonify(ok=True)


# ── Socket.IO ───────────────────────────────────────────────────────────────
@socketio.on("connect")
def on_connect():
    if "user" not in session:
        return False   # reject unauthenticated socket connections
    socketio.emit("telemetry", gesture.telemetry())


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
        hid.start()
        log.info("Bluetooth HID ready — pair a computer from the dashboard.")
    except Exception as e:
        log.error("Bluetooth unavailable: %s", e)
        log.error("The web UI will still run, but pairing is disabled. "
                  "See SETUP_BLUETOOTH.md.")


if __name__ == "__main__":
    _boot_bluetooth()
    gesture.start()
    try:
        socketio.run(app, host="0.0.0.0", port=8080,
                     allow_unsafe_werkzeug=True)
    finally:
        gesture.stop()
        hid.stop()
        btmgr.stop()
        ptz.close()
