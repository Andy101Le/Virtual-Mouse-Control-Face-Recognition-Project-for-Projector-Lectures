#!/usr/bin/env bash
# Fetch the model binaries this project needs but does not track in git
# (see .gitignore — they are large and re-downloadable).
#
# The face recognition model is the one you cannot skip: without it nobody
# can be recognised, so nobody can control the cursor.
set -euo pipefail

cd "$(dirname "$0")/.."

fetch() {
    local name="$1" url="$2" expect_bytes="$3"
    if [ -f "$name" ]; then
        echo "  $name already present — skipping"
        return
    fi
    echo "  downloading $name ..."
    curl -fsSL -o "$name.part" "$url"
    local got
    got=$(stat -c %s "$name.part")
    # A truncated or HTML-error-page download would otherwise surface much
    # later as an unreadable model at startup.
    if [ "$got" -lt "$expect_bytes" ]; then
        rm -f "$name.part"
        echo "  ERROR: $name came back only $got bytes — download failed." >&2
        exit 1
    fi
    mv "$name.part" "$name"
    echo "  $name ok ($got bytes)"
}

echo "Fetching models into $(pwd)"

# SFace face recognition (OpenCV Zoo). Runs via cv2.FaceRecognizerSF, which
# ships in opencv-contrib-python — no extra Python dependency.
fetch face_recognition_sface.onnx \
  "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx" \
  30000000

echo
echo "Done. MediaPipe .task models (hand/face/pose landmarkers) are not"
echo "fetched here — they ship with the project setup; see setup/virt-env-setup.md."
echo
echo "NOTE: anyone enrolled before the switch to SFace must re-register their"
echo "face. Old embeddings use an incompatible format and are ignored."
