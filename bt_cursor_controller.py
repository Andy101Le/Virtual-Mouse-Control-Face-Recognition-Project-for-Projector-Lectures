"""
bt_cursor_controller.py
────────────────────────
Drop-in replacement for
that drives
the paired PC over Bluetooth HID instead of the local machine via
pyautogui.

Deliberately keeps the SAME public interface (CAM_MARGIN,
absolute_to_screen, handle_action) so nothing else in the pipeline has
to change — main.py / gesture_session.py just construct this one
instead.

KEY DIFFERENCE FROM THE PYAUTOGUI VERSION
  pyautogui needed to convert normalized camera coords into *pixel*
  coordinates, so it had to know the screen resolution. Bluetooth HID
  absolute reporting doesn't: we hand the host a 0..32767 fraction on
  each axis and the host maps it across whatever screen it has. So
  there's no screen_w/screen_h here, and no resolution to configure —
  the same Pi drives a 1080p laptop and a 4K monitor identically.

  This is also why "corner of the control zone = corner of the screen"
  holds for free: the control-zone margin math produces a clean 0.0–1.0,
  and 0.0/1.0 are by definition the host's screen edges.
"""

import time
import numpy as np


class BTCursorController:
    # Must stay in sync with the control zone drawn by hud_renderer and
    # the zoom cap in zoom_webcam — same constant, same meaning as the
    # pyautogui version.
    CAM_MARGIN     = 0.15
    CURSOR_SMOOTH  = 0.35
    CLICK_COOLDOWN = 2.0
    SCROLL_COOLDOWN = 1.2
    SCROLL_CLICKS   = 3     # wheel notches per ZOOM IN / ZOOM OUT gesture

    def __init__(self, hid_device):
        """hid_device: a started BluetoothHIDDevice."""
        self.hid = hid_device
        self._cam_scale = 1.0 / (1.0 - 2 * self.CAM_MARGIN)

        # Cursor position is tracked in normalized 0–1 space now, not px.
        self.cursor_nx = 0.5
        self.cursor_ny = 0.5

        self._last_click_time  = {}
        self._last_scroll_time = {}

    @property
    def connected(self):
        return self.hid.connected

    def absolute_to_screen(self, nx, ny):
        """
        Maps a raw camera coordinate into normalized screen space by
        stripping the control-zone margin. Returns 0.0–1.0 (not pixels —
        see the module docstring).
        """
        sx = float(np.clip((nx - self.CAM_MARGIN) * self._cam_scale, 0.0, 1.0))
        sy = float(np.clip((ny - self.CAM_MARGIN) * self._cam_scale, 0.0, 1.0))
        return sx, sy

    def handle_action(self, confirmed, hand_id, tip_xy):
        """
        confirmed: debounced gesture label ('MOVE', 'LEFT CLICK', ...)
        tip_xy:    normalized (x, y) fingertip position, used for MOVE.

        No-ops safely when no host is connected, so the gesture pipeline
        can keep running (and previewing) before anything is paired.
        """
        if not self.hid.connected:
            return

        now = time.perf_counter()
        self._last_click_time.setdefault(hand_id, 0.0)
        self._last_scroll_time.setdefault(hand_id, 0.0)

        if confirmed == 'MOVE':
            tgt_x, tgt_y = self.absolute_to_screen(float(tip_xy[0]), float(tip_xy[1]))
            self.cursor_nx += self.CURSOR_SMOOTH * (tgt_x - self.cursor_nx)
            self.cursor_ny += self.CURSOR_SMOOTH * (tgt_y - self.cursor_ny)
            self.hid.move_absolute(self.cursor_nx, self.cursor_ny)

        elif confirmed == 'LEFT CLICK':
            if (now - self._last_click_time[hand_id]) > self.CLICK_COOLDOWN:
                self.hid.click('left')
                self._last_click_time[hand_id] = now

        elif confirmed == 'RIGHT CLICK':
            if (now - self._last_click_time[hand_id]) > self.CLICK_COOLDOWN:
                self.hid.click('right')
                self._last_click_time[hand_id] = now

        elif confirmed == 'ZOOM IN':
            # The pyautogui version sent ctrl+'+'. We can't send keystrokes
            # from a pointer-only HID descriptor, so this maps to scroll
            # wheel instead — which is the more universal "zoom" gesture and
            # works in slides, PDFs, and browsers without a modifier key.
            if (now - self._last_scroll_time[hand_id]) > self.SCROLL_COOLDOWN:
                self.hid.scroll(self.SCROLL_CLICKS)
                self._last_scroll_time[hand_id] = now

        elif confirmed == 'ZOOM OUT':
            if (now - self._last_scroll_time[hand_id]) > self.SCROLL_COOLDOWN:
                self.hid.scroll(-self.SCROLL_CLICKS)
                self._last_scroll_time[hand_id] = now
