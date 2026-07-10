"""
gesture_launcher.py
────────────────────
Launches main.py (the gesture-control loop) as a subprocess for the
current logged-in user, blocking until it exits so control returns
cleanly to the Tkinter login app.
"""

import os
import subprocess
import sys


class GestureLauncher:
    def __init__(self, script_dir, main_filename="main.py"):
        self.script_dir = script_dir
        self.main_path  = os.path.join(script_dir, main_filename)

    def launch(self, username):
        """
        Runs main.py --user <username> and blocks until it exits.
        Returns True if it ran, False if main.py wasn't found.
        """
        try:
            subprocess.run([sys.executable, self.main_path, "--user", username],
                           cwd=self.script_dir)
            return True
        except FileNotFoundError:
            return False
