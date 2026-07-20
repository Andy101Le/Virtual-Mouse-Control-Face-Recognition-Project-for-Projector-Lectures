# Developer Setup (venv-based)

This is the manual, step-by-step setup for a developer working on this
project directly with a Python virtual environment. It applies whether
you're on a Raspberry Pi or a regular dev machine — camera, I2C
(PTZ/focus), and Bluetooth hardware are optional; the code falls back
or disables those features gracefully when the hardware or system
packages for them aren't present.

For a one-command Raspberry Pi deployment (systemd service, no manual
steps), use `../install.sh` instead — this document only covers the
venv workflow it's built on top of.

A second, venv-free setup path for pre-imaged SD cards is being
evaluated separately. This document may get a follow-up section once
that's decided; it isn't resolved here.

---

## 1. Clone the Repo

```bash
git clone git@github.com:Andy101Le/Virtual-Mouse-Control-Face-Recognition-Project-for-Projector-Lectures.git ~/virtMouse
cd ~/virtMouse
```

## 2. Fetch the MediaPipe Model Files

```bash
python3 download_models.py
```

This pulls `hand_landmarker.task`, `face_landmarker.task`, and
`pose_landmarker_lite.task` into the repo root if they aren't already
present. It only uses the standard library, so it can run before the
virtual environment exists.

## 3. System Packages (Raspberry Pi only)

If you're doing camera/PTZ/Bluetooth development on a Pi, install
these first — they're not available in a usable form from PyPI:

```bash
sudo apt update
sudo apt install -y python3.11-venv python3-picamera2 python3-smbus \
  python3-dbus python3-gi bluez
```

On a non-Pi dev machine you can skip this — the app falls back to a
USB webcam via OpenCV when `picamera2` isn't importable, and disables
the Bluetooth manager when `dbus`/`gi` aren't importable.

## 4. Create and Activate the Virtual Environment

```bash
python3.11 -m venv --system-site-packages .venv
source .venv/bin/activate
```

`--system-site-packages` matters on a Pi: it's what makes the
apt-installed `picamera2`, `smbus`, `dbus`, and `gi` packages visible
inside the venv. Without it, those imports fail even though the
packages are installed system-wide. On a non-Pi machine without those
system packages, this flag is harmless either way.

## 5. Install Dependencies

```bash
pip install --upgrade pip
pip install -r setup/requirements.txt
```

If you add a new import anywhere in the project, add the corresponding
package to `setup/requirements.txt` at the same time — it's meant to
be a complete, accurate list of what the app actually needs, not just
a starting point.

## 6. Run the App

```bash
sudo .venv/bin/python web_server.py
```

Root is required because `bluetooth_hid.py` binds raw L2CAP sockets
(PSM 17/19) for the Bluetooth HID touchpad — see
[`SETUP_BLUETOOTH.md`](../SETUP_BLUETOOTH.md) for why, and for the
one-time `bluetoothd` configuration pairing needs. If you're doing
pure UI/gesture-pipeline work without touching Bluetooth, running
without `sudo` is fine; the Bluetooth manager just logs that it's
unavailable and the rest of the app works normally.

The web UI serves on port 8080 (e.g. `http://<host>:8080`).

## 7. First-Run Setup (Manual Steps)

These aren't handled by any script — they're one-time steps through
the web UI itself:

1. **Create an account** at `/signup`. The first account created
   becomes an admin.
2. **Register your face** from the dashboard's face-registration
   control. This drives `gesture.begin_face_capture()` and stores the
   embedding in `login_system.db` — there's no standalone script for
   this anymore (an older one was removed when the Tkinter app was
   replaced by this web UI).
3. **Pair a Bluetooth host** from the dashboard's "Pair a computer"
   control, once `SETUP_BLUETOOTH.md`'s `bluetoothd` configuration is
   in place. BlueZ will show a 6-digit passkey in the web UI; confirm
   it matches what your laptop shows to complete pairing.
