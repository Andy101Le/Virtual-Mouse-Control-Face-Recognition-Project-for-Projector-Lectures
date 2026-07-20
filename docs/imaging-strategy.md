# SD Card Imaging Strategy (Research / Design)

Status: proposal for review. Nothing here has been implemented;
`install.sh` and the venv-based dev workflow are untouched. This
covers whether and how a pre-configured Raspberry Pi OS image would
replace per-device `install.sh` runs for deployment at scale.

## The premise, and where it needs a correction

The motivating idea was: "a `.venv` created fresh on every install is
the wrong tool once the environment is baked into an image and cloned,
not rebuilt per device." That's half right. Imaging does eliminate
*rebuilding* the environment per device — but a venv, once created,
becomes just another directory on the SD card. Cloning the card clones
the venv along with it. So imaging alone doesn't force a venv-free
approach; it only makes the *one-time* setup cost (creating the venv,
installing packages) something you pay once at image-build time
instead of per device, regardless of whether that setup uses a venv or
installs system-wide.

The real question isn't "does imaging require going venv-free" — it's
"given setup happens once either way, is the venv's isolation worth
keeping." That's investigated below with actual data from this
hardware, not just the general tradeoffs.

## Option comparison

### A. System-wide pip install (`--break-system-packages`)

Raspberry Pi OS (Debian 12/bookworm-based) marks its Python install as
externally-managed (PEP 668) — confirmed on this device:

```
$ /usr/bin/python3.11 -m pip install --dry-run <anything>
error: externally-managed-environment
```

`--break-system-packages` bypasses this. Packages land directly in
`/usr/lib/python3.11/dist-packages` alongside whatever apt manages
there.

- **Isolation:** none. Any future `apt upgrade` touching a Python
  package this app depends on can change what's imported system-wide,
  with no venv boundary to contain the blast radius.
- **Upgrade path:** simplest possible — `pip install --upgrade
  --break-system-packages -r requirements.txt`. But apt and pip are now
  both writing into the same directory tree; apt has no idea pip put
  something there and can overwrite or conflict with it on a system
  upgrade.
- **Reproducibility:** fragile. Nothing stops a future `apt upgrade`
  from silently changing a dependency version this app needs pinned
  exactly (see the numpy/protobuf data below for why that matters
  here specifically).
- **Conflicts with system packages:** real and checked, see next
  section.

### B. apt-packaged dependencies where available

Use `python3-numpy`, `python3-opencv`, etc. from Raspberry Pi OS's own
repos instead of pip, falling back to pip only for what apt doesn't
package (`mediapipe`, `tensorflow-aarch64` aren't in Raspberry Pi OS's
repos at all).

Checked apt's actual candidate versions against this project's pins:

| Package | apt candidate | requirements.txt pin |
|---|---|---|
| numpy | 1.24.2 | 1.26.4 |
| opencv (python3-opencv) | 4.6.0 | 4.11.0.86 (opencv-python) |
| Pillow (python3-pil) | 9.4.0 | 12.2.0 |
| scipy | 1.10.1 | 1.17.1 |
| h5py | 3.7.0 | 3.14.0 |
| protobuf | 3.21.12 | 4.25.9 |
| Flask | 2.2.2 | 3.1.3 |

Every one of these is meaningfully behind — protobuf is a major
version behind (3.x vs 4.x, a real breaking-change boundary that
TensorFlow/mediapipe are known to be sensitive to), and Flask 2.2 vs
3.1 spans breaking API changes too. `tensorflow-aarch64==2.16.1` and
`mediapipe==0.10.14` were almost certainly validated against the newer
numpy/protobuf this repo pins, not apt's. Swapping to apt's versions
risks import failures or, worse, numerical/behavioral differences that
don't crash but produce subtly wrong gesture classification —
harder to catch than a hard failure.

**Verdict: not viable for the ML-adjacent packages.** Could still make
sense for something genuinely version-insensitive that apt happens to
package, but there's little in this project's dependency list that
qualifies — most of what would be worth moving to apt (numpy, opencv,
Pillow, protobuf) is exactly what's most version-sensitive.

### C. Keep the venv, bake it into the image at build time

Same `--system-site-packages` venv `install.sh` already creates,
except it's created once on a "golden" device, not per deployment.
The image capture step (see workflow below) freezes it in place.

- **Isolation:** full — apt upgrades on a deployed device (if ever
  run) can't silently change what this app imports.
- **Upgrade path:** to update dependencies, rebuild the golden image
  and re-flash, or SSH into fleet devices and update the venv
  directly. Slightly more ceremony than option A, but the ceremony is
  exactly what protects reproducibility.
- **Reproducibility:** strongest of the three — the exact same pinned
  versions this project already tests against, unaffected by
  apt's package state.
- **Cost:** the "wasted indirection" of a venv is paid exactly once,
  at image-build time, not per device. On an appliance-style device
  that never runs another Python project, that overhead is close to
  free.

## Recommendation

**Keep the venv approach (Option C).** The apt-version-conflict data
above rules out Option B for anything but the least version-sensitive
packages, and Option A's lack of isolation is a real risk for a
project this tightly pinned, for a cost (the "downside" of using a
venv) that's already close to zero once you're building an image once
and cloning it. The premise that imaging obligates a venv-free
approach doesn't hold up: the venv isn't the thing fighting the
imaging model, it's already compatible with it.

If a genuinely venv-free path is still wanted for other reasons (e.g.
simplicity of the provisioning script, one less directory to reason
about), Option A is the one to pursue, not Option B — but do it with
eyes open about apt/pip write conflicts on system upgrades, and
consider pinning the *system* itself (e.g. disabling unattended
upgrades on deployed devices) to reduce that risk.

## systemd service: venv vs system Python

Structurally almost identical either way. The only line that changes
in the unit `install.sh` generates:

```ini
# venv (current):
ExecStart=/path/to/repo/.venv/bin/python3 /path/to/repo/web_server.py

# system-wide, if Option A were chosen instead:
ExecStart=/usr/bin/python3.11 /path/to/repo/web_server.py
```

Everything else — `User=root` (for the L2CAP bind), `WorkingDirectory`,
`After=`/`Wants=` on `bluetooth.target`, the `bluetoothd
--noplugin=input` prerequisite — is unrelated to venv vs system Python
and wouldn't change.

## Imaging workflow, end to end

1. **Flash a base image.** Raspberry Pi OS Lite (64-bit) onto a
   "golden" SD card, with SSH enabled for headless setup.
2. **Boot once, provision.** SSH in, clone the repo to a fixed,
   known path (unlike `install.sh`'s dynamic path resolution — for an
   image, the path is a convention decided up front, e.g.
   `/home/pi/virtMouse`), then run the equivalent of `install.sh`:
   apt packages, venv creation, `pip install -r requirements.txt`,
   `download_models.py`, the `bluetoothd` override, the systemd unit.
   The model files are the same for every device — safe to bake in at
   this stage rather than re-fetching per device.
3. **Enable the service**, but don't rely on anyone having logged in —
   `web_server.py` already redirects a fresh install with no accounts
   to `/signup`, so an untouched `login_system.db` is a valid
   "out of the box" state, not something that needs special handling.
4. **Clean device-specific state before capturing the image:**
   - Truncate `/etc/machine-id` to empty (`sudo truncate -s 0
     /etc/machine-id`) rather than removing it — systemd expects the
     file to exist and regenerates its contents on next boot only if
     it's present-but-empty. (`/var/lib/dbus/machine-id` is usually a
     symlink to the same file — check it resolves correctly rather
     than assuming.) Cloned devices with an identical machine-id can
     collide on the network (DHCP client id, some mDNS setups).
   - Remove SSH host keys (`sudo rm /etc/ssh/ssh_host_*`) so each
     clone gets unique ones on first boot, rather than every deployed
     device sharing the golden device's host identity.
   - Confirm no `login_system.db` or captured face embeddings exist
     from testing during provisioning — delete if present.
   - Leave hostname/Wi-Fi as generic/unset; handle per-device at flash
     time (next step), not baked into the golden image.
5. **Capture the image.** Shut the golden device down, read the card
   with `dd` from another machine (`sudo dd if=/dev/sdX of=golden.img
   bs=4M status=progress`) — this is the actual card-to-`.img` capture
   step. (`rpi-clone` is a different tool: live disk-to-disk cloning
   onto an attached USB target, not card-to-file capture — useful for
   backups, not this step.) Run the result through `PiShrink` to cut
   the image down to the actual data size — a raw `dd` capture is as
   large as the source card otherwise, which is wasteful to store and
   slow to flash.
6. **Flash to deployment cards.** Raspberry Pi Imager can flash a
   custom `.img` and still offers its own advanced-options dialog
   (hostname, Wi-Fi credentials, SSH key) even for non-official
   images — that's the natural place for the genuinely per-device
   settings, rather than building a first-boot wizard from scratch.
7. **First real boot of each deployed device:** machine-id and SSH
   host keys regenerate (per step 4), hostname/Wi-Fi apply from the
   imager's customization, and the systemd service starts. Everything
   after that — creating an account, registering a face, pairing a
   specific classroom's laptop over Bluetooth — is inherently
   per-device and manual; there's no baking that in ahead of time,
   nor should there be.

## What this doc doesn't resolve

- Whether to build/refresh the golden image manually or automate it
  (e.g. with `pi-gen` or a similar image-build pipeline) — worth
  revisiting once this approach is greenlit and it's clear how often
  the image needs rebuilding.
- Fleet update strategy for already-deployed devices (re-image each
  one vs. remote update over SSH) — out of scope until there's an
  actual fleet to manage.
- Exact `rpi-clone`/`PiShrink` invocation details — call out here as
  things to verify hands-on when this is actually built, not committed
  to sight-unseen.
