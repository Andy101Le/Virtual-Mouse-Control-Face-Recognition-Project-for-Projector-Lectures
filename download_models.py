"""
download_models.py
──────────────────
Downloads the MediaPipe model files needed for this project.
Run this once before anything else.

    python download_models.py
"""
import urllib.request, os, sys

MODELS = {
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    ),
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
    ),
}

for filename, url in MODELS.items():
    if os.path.exists(filename):
        size_mb = os.path.getsize(filename) / 1e6
        print(f"  {filename} already exists ({size_mb:.1f} MB) — skipping")
        continue
    print(f"Downloading {filename} ...", end=" ", flush=True)
    try:
        urllib.request.urlretrieve(url, filename)
        size_mb = os.path.getsize(filename) / 1e6
        print(f"done ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"FAILED: {e}")
        print(f"  Manual download URL:\n  {url}")

print("\nAll done. Run face_register.py next.")
