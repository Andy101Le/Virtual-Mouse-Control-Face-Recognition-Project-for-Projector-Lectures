# Bluetooth HID Setup

This covers the system-level BlueZ configuration the Bluetooth touchpad
(`bluetooth_hid.py`, `bluetooth_manager.py`) needs to actually pair and
drive a host's cursor. `install.sh` does this automatically for a Pi
deployment; this document is for anyone running the app manually via
the venv workflow in `setup/virt-env-setup.md`, or debugging why
pairing isn't working.

## Why `bluetoothd` needs `--noplugin=input`

BlueZ ships a built-in `input` plugin that implements the standard HID
profile itself, claiming the same L2CAP PSMs (17 control, 19 interrupt)
and the same HID service UUID this app registers for its own custom
profile. With the plugin active, `bluetooth_hid.py`'s own profile
registration fails — either with `org.bluez.Error.NotPermitted: UUID
already registered` (the plugin got there first) or `EADDRINUSE`
binding the sockets, depending on timing.

Disabling the plugin is a `bluetoothd`-wide setting, not something the
Python app can do at runtime. Add a systemd drop-in:

```bash
sudo mkdir -p /etc/systemd/system/bluetooth.service.d
sudo tee /etc/systemd/system/bluetooth.service.d/override.conf <<'EOF'
[Service]
ExecStart=
ExecStart=/usr/libexec/bluetooth/bluetoothd --noplugin=input
EOF
sudo systemctl daemon-reload
sudo systemctl restart bluetooth.service
```

The empty `ExecStart=` line is required — it clears the unit's
original `ExecStart` before the second line sets the replacement;
without it, systemd tries to run both and fails.

## Why the app needs root

`bluetooth_hid.py` binds raw L2CAP sockets on PSM 17 and 19 directly
(not through BlueZ's socket-passing mechanism), which needs
`CAP_NET_BIND_SERVICE`-equivalent privilege for these low,
well-known-range PSM numbers. This was verified directly on hardware:

- Running as a regular user — even one already in the `bluetooth`
  group — fails at the bind call with `PermissionError: [Errno 13]
  Permission denied`. Group membership affects BlueZ's D-Bus/polkit
  policy, not this kernel-level socket permission check.
- Running as root works end-to-end: the HID profile registers and
  both PSM sockets bind successfully.

So the app is run with `sudo` (manually) or `User=root` (in
`install.sh`'s systemd unit). If you want to try a less-privileged
alternative, systemd's `AmbientCapabilities=CAP_NET_BIND_SERVICE
CAP_NET_RAW` is the mechanism for it — this hasn't been verified to
work, so treat it as an experiment and confirm pairing still functions
before relying on it.

## Pairing Flow

1. Make sure the `--noplugin=input` override above is in place and
   `bluetoothd` has been restarted since.
2. Start the app with root (see `setup/virt-env-setup.md` or
   `install.sh`). Startup logs should show:
   ```
   HID profile registered with BlueZ
   HID L2CAP sockets listening on PSM 17/19
   Bluetooth HID ready — pair a computer from the dashboard.
   ```
   If instead you see `Bluetooth unavailable: ...`, something above
   isn't in place yet — the web UI still runs, but pairing is
   disabled until it's fixed.
3. From the dashboard, click **"Pair a computer"**. The Pi becomes
   discoverable.
4. On the host you want to control, open its normal Bluetooth
   settings and pair with the device named "Gesture Touchpad" (or
   whatever alias `bluetooth_manager.py` was configured with).
5. BlueZ will surface a 6-digit passkey; the dashboard shows the same
   number. Confirm they match to complete pairing. This confirmation
   step is a real security boundary — the Pi doesn't auto-accept
   pairing requests, since a successful pairing hands over cursor
   control.
6. Once paired, the cursor is driven straight from the Pi over
   Bluetooth HID. The web UI is no longer in the loop — you can close
   the browser tab and the touchpad keeps working.

## Troubleshooting

**`UUID already registered`** — the `--noplugin=input` override isn't
active yet, or `bluetoothd` hasn't been restarted since it was added.
Check with `systemctl status bluetooth` — the `Drop-In` line should
list the override, and the process args should include
`--noplugin=input`.

**`Could not bind L2CAP PSM 17/19: Permission denied`** — the app
isn't running as root. See "Why the app needs root" above.

**Cursor doesn't move after a successful pairing** — this project
sends absolute-position HID reports (mapping a fixed camera control
zone onto the full screen). Linux and Windows generally accept this;
macOS is fussier about absolute pointers declared as `Mouse` rather
than `Digitizer`. If pairing succeeds but the cursor never tracks on
macOS, that descriptor choice in `bluetooth_hid.py` is the first thing
to check.
