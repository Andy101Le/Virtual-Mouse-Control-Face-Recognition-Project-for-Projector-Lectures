# pyenv Setup on Raspberry Pi 5 (Raspberry Pi OS Trixie)

## Overview

pyenv lets you install and switch between multiple Python versions without touching the system Python. This guide installs Python 3.12 alongside the system Python 3.13.

---

## 1. Install Build Dependencies

pyenv compiles Python from source, so you need the full build toolchain first:

```bash
sudo apt update; sudo apt install -y \
  make build-essential libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev curl git \
  libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
```

---

## 2. Install pyenv

```bash
curl https://pyenv.run | bash
```

This installs pyenv and the following plugins automatically:
- `pyenv` — core version manager
- `pyenv-update` — lets you run `pyenv update`
- `pyenv-virtualenv` — venv management via pyenv

---

## 3. Configure Your Shell

Add the following to `~/.bashrc`:

```bash
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - bash)"
```

Apply immediately:

```bash
source ~/.bashrc
```

Verify pyenv is working:

```bash
pyenv --version
```

---

## 4. Install Python 3.12

```bash
pyenv install 3.12.10
```

> This compiles Python from source and takes **10–15 minutes** on a Pi 5. The `--enable-optimizations` flag is applied automatically by pyenv for better runtime performance.

Confirm it installed:

```bash
pyenv versions
```

You should see both `system` (3.13) and `3.12.10` listed.

---

## 5. Set the Python Version

You have three scope options:

| Command | Scope |
|---|---|
| `pyenv global 3.12.10` | Default for your user everywhere |
| `pyenv local 3.12.10` | This directory only (writes `.python-version`) |
| `pyenv shell 3.12.10` | Current shell session only |

For a project-specific setup (recommended):

```bash
git clone git@github.com:Andy101Le/Virtual-Mouse-Control-Face-Recognition-Project-for-Projector-Lectures.git ~/virtMouse && cd virtMouse
pyenv local 3.12.10
python --version  # should show 3.12.10
```

---

## 6. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
```

Your prompt will show `(.venv)` when active.

---

## 7. Install Your Packages

```bash
pip install "protobuf>=4.25.3,<5"

pip install \
  "mediapipe==0.10.14" \
  "tensorflow-aarch64==2.16.1" \
  opencv-python \
  numpy \
  pyautogui
```

> **TensorFlow note:** The install is large (~500 MB). If you only need inference, consider `pip install tflite-runtime` instead — it's much smaller and loads faster on Pi hardware.

---

## 8. Verify All Imports

```bash
python - <<'EOF'
import cv2
import time
import threading
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
import numpy as np
from tensorflow.keras.models import load_model
import pyautogui

print("All imports OK")
print(f"Python:  {__import__('sys').version}")
print(f"OpenCV:  {cv2.__version__}")
print(f"NumPy:   {np.__version__}")
EOF
```

---

## Quick Reference

| Task | Command |
|---|---|
| List installed versions | `pyenv versions` |
| List installable versions | `pyenv install --list \| grep "3\.12"` |
| Activate venv | `source .venv/bin/activate` |
| Deactivate venv | `deactivate` |
| Update pyenv itself | `pyenv update` |
| Uninstall a version | `pyenv uninstall 3.12.10` |
| Check active version | `pyenv version` |

---

## Troubleshooting

**`pyenv: command not found` after install**
You haven't sourced your shell config. Run `source ~/.bashrc` or open a new terminal.

**Build fails with `lzma` or `ssl` warning**
The build dependency install in Step 1 covers this. If you skipped it, run it now and reinstall: `pyenv uninstall 3.12.10 && pyenv install 3.12.10`.

**`python` still shows 3.13**
Check that `pyenv local` or `pyenv global` has been set and that the pyenv shims are in your `PATH` before `/usr/bin`. Run `which python` — it should point to somewhere inside `~/.pyenv/shims/`.

**pyautogui fails at runtime (no display)**
pyautogui requires an X display. If running headless, prefix your command with `DISPLAY=:0` or switch to the desktop image of Raspberry Pi OS.
