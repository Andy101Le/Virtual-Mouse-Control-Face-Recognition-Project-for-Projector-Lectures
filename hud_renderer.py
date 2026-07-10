"""
hud_renderer.py
────────────────
All on-screen drawing for main.py: face/pose overlays, the hand
skeleton + gesture labels, the "control zone" box, and the status HUD
(FPS, per-hand action strings, authentication banner, legend).

Kept separate from the detection/control logic so main.py's loop reads
as "compute state, then draw it" rather than interleaving cv2 calls
with the actual pipeline.

Every method that draws something tied to a real-world tracked
position (face box, pose skeleton, control zone, crosshair, hand
skeleton) takes an optional `zoom` parameter — a ZoomWebcamController
instance. When zoom is active, each raw-frame pixel coordinate is
remapped through zoom.transform_point()/transform_array() before
drawing, and line/font sizes are scaled via zoom.get_scale(). This is
what makes the control zone box, skeletons, and labels track correctly
and stay crisp under auto-zoom, instead of the old approach of drawing
everything on the raw frame and stretching the whole finished image
(which is geometrically fine but blurs text/thin lines).

Call these methods on the frame ZoomWebcamController.apply() returns
(i.e. AFTER cropping/resizing), passing the same zoom instance, so the
remapping and the actual crop agree pixel-for-pixel. Status HUD/auth
banner are fixed-position UI chrome (not tied to any tracked
coordinate) and don't need a zoom parameter — draw them on the same
post-crop frame regardless.
"""

import cv2
import numpy as np

POSE_CONNECTIONS = np.array([
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
], dtype=np.int32)

POSE_VIS_THRESH = 0.4

# Clamp how far HUD scale can drift from 1.0 so text/lines never become
# illegibly tiny or absurdly oversized at extreme zoom levels.
_SCALE_MIN = 0.7
_SCALE_MAX = 2.5


def _clamped_scale(zoom):
    if zoom is None:
        return 1.0
    return max(_SCALE_MIN, min(_SCALE_MAX, zoom.get_scale()))


class HUDRenderer:
    def __init__(self, num_registered):
        self.num_registered = num_registered
        self._chain_bufs = None  # lazily sized from gesture_engine.FINGER_CHAINS

    # ── Face / pose overlays ─────────────────────────────────────────────
    @staticmethod
    def draw_face_boxes(frame, face_rec, cached_face_lms, recognised_user, unknown_label, w, h, zoom=None):
        for fi, face_lms in enumerate(cached_face_lms):
            xs = [lm.x for lm in face_lms]
            ys = [lm.y for lm in face_lms]
            pts_px = np.array([[x * w, y * h] for x, y in zip(xs, ys)], dtype=np.float32)
            nose_x, nose_y = face_rec.get_nose_tip(face_lms)
            nose_px = np.array([nose_x * w, nose_y * h], dtype=np.float32)

            if zoom is not None:
                pts_px = zoom.transform_array(pts_px)
                nose_px = np.array(zoom.transform_point(*nose_px), dtype=np.float32)

            xs_px, ys_px = pts_px[:, 0], pts_px[:, 1]
            box_col = (0, 220, 0) if (fi == 0 and recognised_user != unknown_label) else (0, 0, 200)
            cv2.rectangle(frame,
                          (int(max(xs_px.min() - 10, 0)), int(max(ys_px.min() - 10, 0))),
                          (int(min(xs_px.max() + 10, w)), int(min(ys_px.max() + 10, h))),
                          box_col, 2)
            cv2.circle(frame, (int(nose_px[0]), int(nose_px[1])), 5, (0, 255, 255), -1)

    @staticmethod
    def draw_pose_skeleton(frame, pose_lms, color, w, h, label=None, scores=None, zoom=None):
        lm_arr = np.array([[lm.x, lm.y] for lm in pose_lms], dtype=np.float32)
        vis    = np.array([lm.visibility for lm in pose_lms], dtype=np.float32)
        pts    = (lm_arr * (w, h)).astype(np.float32)
        if zoom is not None:
            pts = zoom.transform_array(pts)
        pts = pts.astype(np.int32)

        s          = _clamped_scale(zoom)
        thickness  = max(1, int(round(2 * s)))
        radius     = max(2, int(round(4 * s)))
        font_big   = 0.45 * s
        font_small = 0.35 * s

        for a, b in POSE_CONNECTIONS:
            if vis[a] >= POSE_VIS_THRESH and vis[b] >= POSE_VIS_THRESH:
                cv2.line(frame, tuple(pts[a]), tuple(pts[b]), color, thickness)

        # Synthetic neck: midpoint of shoulders connected up to nose
        if vis[11] >= POSE_VIS_THRESH and vis[12] >= POSE_VIS_THRESH:
            neck = ((pts[11].astype(np.float32) + pts[12].astype(np.float32)) / 2).astype(np.int32)
            cv2.circle(frame, tuple(neck), radius, color, -1)
            if vis[0] >= POSE_VIS_THRESH:
                cv2.line(frame, tuple(neck), tuple(pts[0]), color, thickness)

        for pt in pts[vis >= POSE_VIS_THRESH]:
            cv2.circle(frame, tuple(pt), radius, color, -1)

        if label is not None and vis[0] >= POSE_VIS_THRESH:
            lx, ly = int(pts[0][0]) - 20, max(int(pts[0][1]) - 18, 10)
            cv2.putText(frame, label, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, font_big, color, 1, cv2.LINE_AA)
            if scores:
                cv2.putText(frame, scores, (lx, ly + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, font_small, color, 1, cv2.LINE_AA)

    def draw_pose_skeletons(self, frame, cached_pose_lms, user_active, face_nose_pos,
                             recog_score, face_size, w, h, zoom=None):
        for pi, pose_lms in enumerate(cached_pose_lms):
            # Identity matching happens in raw normalized-coordinate space —
            # unaffected by display zoom, so no transform needed here.
            p_nose = np.array([pose_lms[0].x, pose_lms[0].y], dtype=np.float32)
            is_reg = (user_active and face_nose_pos is not None and
                      float(np.linalg.norm(p_nose - face_nose_pos)) < 0.18)
            skel_col   = (0, 220, 0) if is_reg else (0, 0, 200)
            above_txt  = "YOU" if is_reg else f"P{pi}:UNK"
            scores_txt = f"pca={recog_score:.3f}  geo={face_size:.3f}" if (is_reg and face_size) else None
            self.draw_pose_skeleton(frame, pose_lms, skel_col, w, h, above_txt, scores_txt, zoom=zoom)

    # ── Control zone / crosshair ─────────────────────────────────────────
    @staticmethod
    def draw_control_zone(frame, limb_mode, cam_margin, w, h, zoom=None):
        """
        Draws the exact rectangle CursorController.absolute_to_screen()
        uses for its cam_margin/scale math — (cam_margin, cam_margin) to
        (1-cam_margin, 1-cam_margin) in raw-frame space. Since this is
        the SAME region driving the real cursor mapping (which always
        operates on raw, un-cropped hand coordinates regardless of
        display zoom), remapping these two corners through zoom keeps
        the drawn box, the tracked face, and the actual reachable
        cursor range in exact visual agreement at any zoom level: the
        box always represents the same physical distance from the
        face, and its corners always correspond to the screen's
        corners.
        """
        x1, y1 = cam_margin * w, cam_margin * h
        x2, y2 = w - cam_margin * w, h - cam_margin * h
        if zoom is not None:
            x1, y1 = zoom.transform_point(x1, y1)
            x2, y2 = zoom.transform_point(x2, y2)

        s = _clamped_scale(zoom)
        thickness = max(1, int(round(1 * s)))
        zone_col = (0, 200, 80) if limb_mode else (60, 60, 60)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), zone_col, thickness)
        cv2.putText(frame, "control zone", (int(x1) + 4, int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33 * s, zone_col, 1)

    @staticmethod
    def draw_crosshair(frame, limb_mode, face_nose_pos, w, h, zoom=None):
        if limb_mode and face_nose_pos is not None:
            nx_px, ny_px = face_nose_pos[0] * w, face_nose_pos[1] * h
            if zoom is not None:
                nx_px, ny_px = zoom.transform_point(nx_px, ny_px)
            nx_px, ny_px = int(nx_px), int(ny_px)
            arm = max(4, int(round(15 * _clamped_scale(zoom))))
            cv2.line(frame, (nx_px - arm, ny_px), (nx_px + arm, ny_px), (0, 255, 255), 1)
            cv2.line(frame, (nx_px, ny_px - arm), (nx_px, ny_px + arm), (0, 255, 255), 1)

    # ── Hand drawing ──────────────────────────────────────────────────────
    def draw_hand(self, frame, finger_chains, sp, line_col, dot_col, zoom=None):
        sp_draw = zoom.transform_array(sp) if zoom is not None else np.asarray(sp, dtype=np.float32)
        sp_draw = sp_draw.astype(np.int32)

        if self._chain_bufs is None:
            self._chain_bufs = [np.zeros((len(c), 1, 2), dtype=np.int32) for c in finger_chains]
        radius = max(2, int(round(4 * _clamped_scale(zoom))))
        for ci, chain in enumerate(finger_chains):
            self._chain_bufs[ci][:, 0, :] = sp_draw[chain]
            cv2.polylines(frame, [self._chain_bufs[ci]], False, line_col, 2)
        for pt in sp_draw:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), radius, dot_col, -1)

    @staticmethod
    def draw_blocked_hand(frame, tip_px, zoom=None):
        x, y = tip_px
        if zoom is not None:
            x, y = zoom.transform_point(x, y)
        x, y = int(x), int(y)
        cv2.circle(frame, (x, y), 10, (0, 0, 200), 2)
        cv2.putText(frame, "blocked", (x + 6, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 200), 1)

    @staticmethod
    def draw_move_indicator(frame, tip_px, limb_mode, face_nose_pos, w, h, zoom=None):
        x, y = tip_px
        if zoom is not None:
            x, y = zoom.transform_point(x, y)
        x, y = int(x), int(y)
        cv2.circle(frame, (x, y), 10, (0, 255, 255), 2)
        if limb_mode and face_nose_pos is not None:
            nx, ny = face_nose_pos[0] * w, face_nose_pos[1] * h
            if zoom is not None:
                nx, ny = zoom.transform_point(nx, ny)
            cv2.line(frame, (int(nx), int(ny)), (x, y), (0, 255, 255), 1)

    # ── Text HUD (fixed-position chrome — no tracked coordinate, no zoom needed) ──
    @staticmethod
    def draw_status_hud(frame, fps, num_people, num_faces, hand_action_strs, w):
        cv2.putText(frame, f"FPS:{fps:.0f}  People:{num_people}  Faces:{num_faces}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 100), 2, cv2.LINE_AA)

        for i, (hstr, hcol) in enumerate(hand_action_strs):
            cv2.putText(frame, hstr, (10, 50 + i * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, hcol, 2, cv2.LINE_AA)

        cv2.putText(frame, "Q=quit Z=zoom", (w - 110, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

        legend = [
            "GREEN = registered user  |  RED = bystander (blocked)",
            "LEFT=fist  RIGHT=thumbs-down  ZOOM IN=rock  ZOOM OUT=peace",
        ]
        for i, line in enumerate(legend):
            cv2.putText(frame, line, (6, frame.shape[0] - 8 - (len(legend) - 1 - i) * 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (120, 120, 120), 1)

    def draw_auth_banner(self, frame, auth_temp, user_active, grace_remaining,
                          temp_activate, w, h):
        temp_pct = int(auth_temp * 100)
        if user_active:
            banner_txt = (f"REGISTERED USER ACTIVE  temp={temp_pct}%"
                          if auth_temp >= temp_activate
                          else f"REGISTERED USER ACTIVE  temp={temp_pct}%  grace {grace_remaining:.1f}s")
            banner_col = (0, 220, 0)
        else:
            banner_txt = ("NO USERS REGISTERED — run face_register.py"
                          if self.num_registered == 0
                          else f"Users: {self.num_registered} registered  |  temp={temp_pct}%  (show face)")
            banner_col = (0, 140, 200)

        cv2.rectangle(frame, (0, h - 48), (w, h - 20), (20, 20, 20), -1)
        cv2.putText(frame, banner_txt, (10, h - 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, banner_col, 2, cv2.LINE_AA)