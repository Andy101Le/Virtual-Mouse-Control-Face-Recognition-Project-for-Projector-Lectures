import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'pose_landmarker_lite.task')

latest_result = None

def result_callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=result_callback
)
landmarker = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

zoom = 1.0
crop_cx, crop_cy = None, None

TARGET_RATIO = 0.80
FAR_THRESH = 0.45
FAR_BOOST = 0.18
HEAD_PAD = 0.20
HIP_PAD = 0.10
ZOOM_IN_SMOOTH = 0.14
ZOOM_OUT_SMOOTH = 0.07
PAN_SMOOTH = 0.12
MAX_ZOOM = 5.0
VIS_THRESH = 0.5


FACE_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8]
SHOULDER_IDS = [11, 12]
HIP_IDS = [23, 24]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    if crop_cx is None:
        crop_cx, crop_cy = w / 2, h / 2

    timestamp_ms = int(time.time() * 1000)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    landmarker.detect_async(mp_image, timestamp_ms)

    status = "No person detected"
    frame_box = None

    if latest_result and latest_result.pose_landmarks:
        lms = latest_result.pose_landmarks[0]

        def pts(ids):
            return [(lms[i].x * w, lms[i].y * h) for i in ids
                    if lms[i].visibility > VIS_THRESH]

        face = pts(FACE_IDS)
        shoulders = pts(SHOULDER_IDS)
        hips = pts(HIP_IDS)

        if face and hips:
            head_y = min(p[1] for p in face)
            hip_y = sum(p[1] for p in hips) / len(hips)
            torso_h = max(hip_y - head_y, 1)

            top = head_y - HEAD_PAD * torso_h
            bottom = hip_y + HIP_PAD * torso_h
            region_h = bottom - top

            all_x = [p[0] for p in face + shoulders + hips]
            rx1, rx2 = min(all_x), max(all_x)
            region_w = max(rx2 - rx1, 1) * 1.30   # arm room

            frame_box = (rx1, top, rx2, bottom)

            torso_ratio = torso_h / h
            far_factor = max(0.0, min(1.0, (FAR_THRESH - torso_ratio) / FAR_THRESH))
            eff_target = min(TARGET_RATIO + FAR_BOOST * far_factor, 0.95)

            zoom_for_height = (eff_target * h) / region_h
            fit_zoom_h = h / region_h
            fit_zoom_w = w / region_w

            target_zoom = min(zoom_for_height, fit_zoom_h, fit_zoom_w, MAX_ZOOM)
            target_zoom = max(target_zoom, 1.0)
            smooth = ZOOM_IN_SMOOTH if target_zoom > zoom else ZOOM_OUT_SMOOTH
            zoom += (target_zoom - zoom) * smooth

            person_cx = (rx1 + rx2) / 2
            person_cy = (top + bottom) / 2
            crop_cx += (person_cx - crop_cx) * PAN_SMOOTH
            crop_cy += (person_cy - crop_cy) * PAN_SMOOTH

            status = f"torso={torso_ratio:.2f}  tgt={eff_target:.2f}  zoom->{target_zoom:.2f}"

    crop_w = int(w / zoom)
    crop_h = int(h / zoom)
    x1 = int(crop_cx - crop_w / 2)
    y1 = int(crop_cy - crop_h / 2)
    x1 = max(0, min(x1, w - crop_w))
    y1 = max(0, min(y1, h - crop_h))
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    cropped = frame[y1:y2, x1:x2]
    display = cv2.resize(cropped, (w, h))

    if frame_box:
        scale_x = w / crop_w
        scale_y = h / crop_h
        dbx1 = int((frame_box[0] - x1) * scale_x)
        dby1 = int((frame_box[1] - y1) * scale_y)
        dbx2 = int((frame_box[2] - x1) * scale_x)
        dby2 = int((frame_box[3] - y1) * scale_y)
        cv2.rectangle(display, (dbx1, dby1), (dbx2, dby2), (0, 255, 0), 2)
        cv2.putText(display, "Waist-up", (dbx1, dby1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.putText(display, f"Zoom: {zoom:.2f}x", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    cv2.putText(display, status, (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    cv2.imshow("Auto Zoom", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
