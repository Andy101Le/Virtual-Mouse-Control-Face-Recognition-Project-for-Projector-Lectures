"""
login_system.py
───────────────
Entry point for the Hand Gesture Control project's account system.

Wires together:
  - UserDatabase    (user_database.py)    — SQLite account + face storage
  - FaceCapture     (face_capture.py)     — MediaPipe face-sample capture
  - GestureLauncher (gesture_launcher.py) — launches main.py as a subprocess
  - LoginApp        (login_app.py)        — the Tkinter GUI itself

Run this FIRST. main.py is launched from inside the main menu.
"""

import os

from user_database import UserDatabase
from face_capture import FaceCapture
from gesture_launcher import GestureLauncher
from login_app import LoginApp

# Resolve paths relative to this script's folder, not the cwd PyCharm uses
_SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
DATABASE_NAME  = os.path.join(_SCRIPT_DIR, "login_system.db")
FACE_TASK_PATH = os.path.join(_SCRIPT_DIR, "face_landmarker.task")

if __name__ == "__main__":
    db               = UserDatabase(DATABASE_NAME)
    face_capture     = FaceCapture(FACE_TASK_PATH)
    gesture_launcher = GestureLauncher(_SCRIPT_DIR)
    app              = LoginApp(db, face_capture, gesture_launcher)
    app.run()