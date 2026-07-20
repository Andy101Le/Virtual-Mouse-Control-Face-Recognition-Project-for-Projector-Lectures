#!/usr/bin/env bash
#
# install.sh
# ──────────
# One-shot setup for a fresh Raspberry Pi clone of this repo:
#   - checks for the required Python version and system packages
#   - creates/updates the .venv virtual environment
#   - fetches the MediaPipe model files
#   - installs and starts the app as a systemd service
#
# Safe to re-run: every step checks current state first and skips work
# that's already done.
#
# Run as your normal user, NOT with sudo — the script calls sudo itself
# for the specific steps that need it (apt installs, systemd, bluetoothd).
#
#   ./install.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

REQUIRED_PYTHON="python3.11"
VENV_DIR="$REPO_ROOT/.venv"
SERVICE_NAME="virtmouse.service"
SERVICE_PATH="/etc/systemd/system/$SERVICE_NAME"
BLUEZ_OVERRIDE_DIR="/etc/systemd/system/bluetooth.service.d"
BLUEZ_OVERRIDE_PATH="$BLUEZ_OVERRIDE_DIR/override.conf"

# The user the service will run as for hardware-permission purposes; the
# invoking user for everything else (venv, pip, model download).
RUN_USER="${SUDO_USER:-$(whoami)}"

if [ "$(whoami)" = "root" ] && [ -z "${SUDO_USER:-}" ]; then
    echo "ERROR: don't run this script directly as root — run it as your"
    echo "normal user. It calls sudo itself for the steps that need it."
    exit 1
fi

echo "== Repo root: $REPO_ROOT =="
echo "== Invoking user: $RUN_USER =="
echo

# ── 1. Python version ────────────────────────────────────────────────────────
echo "-- Checking for $REQUIRED_PYTHON --"
if ! command -v "$REQUIRED_PYTHON" >/dev/null 2>&1; then
    echo "ERROR: $REQUIRED_PYTHON not found."
    echo "  Install it with: sudo apt install python3.11 python3.11-venv"
    exit 1
fi
echo "OK: $($REQUIRED_PYTHON --version)"
echo

# ── 2. System (apt) packages pip can't provide ───────────────────────────────
# picamera2/smbus need the system libcamera/i2c stack; dbus/gi are needed for
# the BlueZ D-Bus pairing and HID profile registration.
REQUIRED_APT_PKGS=(
    python3.11-venv
    python3-picamera2
    python3-smbus
    python3-dbus
    python3-gi
    bluez
    git-lfs
)

echo "-- Checking required system packages --"
MISSING_APT=()
for pkg in "${REQUIRED_APT_PKGS[@]}"; do
    if ! dpkg -s "$pkg" >/dev/null 2>&1; then
        MISSING_APT+=("$pkg")
    fi
done

if [ ${#MISSING_APT[@]} -ne 0 ]; then
    echo "Installing missing packages: ${MISSING_APT[*]}"
    sudo apt-get update -qq
    sudo apt-get install -y "${MISSING_APT[@]}"
else
    echo "OK: all required system packages already installed"
fi
echo

# ── 3. Virtual environment ───────────────────────────────────────────────────
# --system-site-packages so picamera2/smbus (apt-installed, not on PyPI in a
# form that works here) are visible inside the venv.
echo "-- Setting up .venv --"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR"
    "$REQUIRED_PYTHON" -m venv --system-site-packages "$VENV_DIR"
else
    echo "OK: .venv already exists, skipping creation"
fi

VENV_PY="$VENV_DIR/bin/python3"
"$VENV_PY" -m pip install --upgrade pip --quiet
"$VENV_PY" -m pip install -r setup/requirements.txt --quiet
echo "OK: dependencies installed"
echo

# ── 4. MediaPipe model files ─────────────────────────────────────────────────
echo "-- Fetching MediaPipe model files --"
"$VENV_PY" download_models.py
echo

# ── 5. bluetoothd --noplugin=input override ──────────────────────────────────
# BlueZ's built-in "input" plugin claims the same HID PSMs (17/19) and HID
# UUID this app registers for itself; without this override, pairing fails
# with "UUID already registered" or EADDRINUSE. See SETUP_BLUETOOTH.md.
echo "-- Configuring bluetoothd (--noplugin=input) --"
NEW_OVERRIDE_CONTENT="[Service]
ExecStart=
ExecStart=/usr/libexec/bluetooth/bluetoothd --noplugin=input
"

if [ -f "$BLUEZ_OVERRIDE_PATH" ] && [ "$(cat "$BLUEZ_OVERRIDE_PATH")" = "$(printf '%s' "$NEW_OVERRIDE_CONTENT")" ]; then
    echo "OK: bluetoothd override already in place"
else
    sudo mkdir -p "$BLUEZ_OVERRIDE_DIR"
    printf '%s' "$NEW_OVERRIDE_CONTENT" | sudo tee "$BLUEZ_OVERRIDE_PATH" >/dev/null
    sudo systemctl daemon-reload
    sudo systemctl restart bluetooth.service
    echo "OK: bluetoothd override installed, service restarted"
fi
echo

# ── 6. systemd service for the app itself ────────────────────────────────────
# Root is required here: bluetooth_hid.py binds raw L2CAP sockets on PSM
# 17/19, which needs CAP_NET_BIND_SERVICE-equivalent privilege. Confirmed on
# real hardware — running as a regular user (even in the bluetooth group)
# fails with "Permission denied" on the bind call; running as root works.
# See bluetooth_hid.py's own module docstring for the same statement.
#
# If you want to try a less-privileged alternative later, AmbientCapabilities
# is the systemd mechanism for it (untested here — plain group membership in
# "bluetooth" was NOT sufficient in testing). It would replace User=root below
# with something like:
#   User=<your invoking user>
#   Group=<your invoking user's primary group>
#   AmbientCapabilities=CAP_NET_BIND_SERVICE CAP_NET_RAW
# Verify pairing still actually works before relying on it.
echo "-- Installing systemd service --"

NEW_SERVICE_CONTENT="[Unit]
Description=Virtual Mouse Control (gesture + face + Bluetooth HID touchpad)
After=network.target bluetooth.target
Wants=bluetooth.target

[Service]
Type=simple
User=root
WorkingDirectory=$REPO_ROOT
ExecStart=$VENV_PY $REPO_ROOT/web_server.py
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
"

NEEDS_RELOAD=0
if [ -f "$SERVICE_PATH" ] && [ "$(cat "$SERVICE_PATH")" = "$(printf '%s' "$NEW_SERVICE_CONTENT")" ]; then
    echo "OK: service file already up to date"
else
    printf '%s' "$NEW_SERVICE_CONTENT" | sudo tee "$SERVICE_PATH" >/dev/null
    NEEDS_RELOAD=1
    echo "OK: wrote $SERVICE_PATH"
fi

if [ "$NEEDS_RELOAD" = "1" ]; then
    sudo systemctl daemon-reload
fi

sudo systemctl enable "$SERVICE_NAME"

if systemctl is-active --quiet "$SERVICE_NAME"; then
    if [ "$NEEDS_RELOAD" = "1" ]; then
        echo "Restarting $SERVICE_NAME to pick up changes..."
        sudo systemctl restart "$SERVICE_NAME"
    else
        echo "OK: $SERVICE_NAME already running"
    fi
else
    echo "Starting $SERVICE_NAME..."
    sudo systemctl start "$SERVICE_NAME"
fi
echo

# ── 7. Summary ────────────────────────────────────────────────────────────────
echo "======================================================================"
echo "Install complete."
echo
sudo systemctl status "$SERVICE_NAME" --no-pager -l | head -10 || true
echo
echo "Web UI:       http://<this Pi's address>:8080"
echo "Live logs:    sudo journalctl -u $SERVICE_NAME -f"
echo "Restart:      sudo systemctl restart $SERVICE_NAME"
echo "Stop:         sudo systemctl stop $SERVICE_NAME"
echo "Disable:      sudo systemctl disable $SERVICE_NAME"
echo "======================================================================"
