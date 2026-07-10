"""
cursor_controller.py
─────────────────────
Translates a confirmed gesture + fingertip position into actual mouse
control via pyautogui: cursor movement (with a normalized-camera-space
"control zone" margin and EMA smoothing), and cooldown-gated left/right
clicks and zoom-in/zoom-out hotkeys.
"""

import time
import numpy as np
import pyautogui

pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0


class CursorController:
    CAM_MARGIN     = 0.15
    CURSOR_SMOOTH  = 0.35
    CLICK_COOLDOWN = 2.0
    ZOOM_COOLDOWN  = 1.2

    def __init__(self):
        self.screen_w, self.screen_h = pyautogui.size()
        self._cam_scale = 1.0 / (1.0 - 2 * self.CAM_MARGIN)

        self.cursor_x = self.screen_w / 2.0
        self.cursor_y = self.screen_h / 2.0

        self._last_click_time = {}
        self._last_zoom_time  = {}

    def absolute_to_screen(self, nx, ny):
        sx = float(np.clip((nx - self.CAM_MARGIN) * self._cam_scale, 0.0, 1.0)) * self.screen_w
        sy = float(np.clip((ny - self.CAM_MARGIN) * self._cam_scale, 0.0, 1.0)) * self.screen_h
        return sx, sy

    def handle_action(self, confirmed, hand_id, tip_xy):
        """
        confirmed: debounced gesture label ('MOVE', 'LEFT CLICK', ...)
        tip_xy:    normalized (x, y) fingertip position, only used for MOVE.
        """
        now = time.perf_counter()
        self._last_click_time.setdefault(hand_id, 0.0)
        self._last_zoom_time.setdefault(hand_id, 0.0)

        if confirmed == 'MOVE':
            tgt_x, tgt_y = self.absolute_to_screen(float(tip_xy[0]), float(tip_xy[1]))
            self.cursor_x += self.CURSOR_SMOOTH * (tgt_x - self.cursor_x)
            self.cursor_y += self.CURSOR_SMOOTH * (tgt_y - self.cursor_y)
            pyautogui.moveTo(int(self.cursor_x), int(self.cursor_y))

        elif confirmed == 'LEFT CLICK':
            if (now - self._last_click_time[hand_id]) > self.CLICK_COOLDOWN:
                pyautogui.click(button='left')
                self._last_click_time[hand_id] = now

        elif confirmed == 'RIGHT CLICK':
            if (now - self._last_click_time[hand_id]) > self.CLICK_COOLDOWN:
                pyautogui.click(button='right')
                self._last_click_time[hand_id] = now

        elif confirmed == 'ZOOM IN':
            if (now - self._last_zoom_time[hand_id]) > self.ZOOM_COOLDOWN:
                pyautogui.hotkey('ctrl', '+')
                self._last_zoom_time[hand_id] = now

        elif confirmed == 'ZOOM OUT':
            if (now - self._last_zoom_time[hand_id]) > self.ZOOM_COOLDOWN:
                pyautogui.hotkey('ctrl', '-')
                self._last_zoom_time[hand_id] = now
