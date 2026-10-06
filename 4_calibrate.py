"""
STEP 4: Hand Calibration

Run this ONCE per user/session before live prediction. Asks the user to
hold both hands open (all 10 fingers spread) in front of the camera for a
few seconds, records their specific hand-span, and saves it to
data/calibration.json. The main app then uses this instead of a generic
per-frame scale, which makes recognition noticeably more accurate for
that particular person's hand size.

Run:  python 4_calibrate.py
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import json
import time
import os

LANDMARKER_MODEL = "hand_landmarker.task"
CALIBRATION_FILE = "data/calibration.json"
CALIBRATION_SECONDS = 3

os.makedirs("data", exist_ok=True)

base_options = mp_python.BaseOptions(model_asset_path=LANDMARKER_MODEL)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    running_mode=vision.RunningMode.VIDEO,
)
landmarker = vision.HandLandmarker.create_from_options(options)


def hand_span(hand_landmarks):
    """Max distance from wrist to any fingertip -- this is our scale reference."""
    base_x, base_y, base_z = hand_landmarks[0].x, hand_landmarks[0].y, hand_landmarks[0].z
    dists = []
    for lm in hand_landmarks:
        dx, dy, dz = lm.x - base_x, lm.y - base_y, lm.z - base_z
        dists.append((dx**2 + dy**2 + dz**2) ** 0.5)
    return max(dists)


def main():
    cap = cv2.VideoCapture(0)
    spans = []
    start_time = None
    frame_ts = 0

    print("Hold BOTH hands open, all fingers spread, facing the camera...")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        frame_ts += 33
        result = landmarker.detect_for_video(mp_image, frame_ts)

        hands_visible = len(result.hand_landmarks) if result.hand_landmarks else 0

        if hands_visible >= 1:
            if start_time is None:
                start_time = time.time()
            elapsed = time.time() - start_time
            for hl in result.hand_landmarks:
                spans.append(hand_span(hl))

            remaining = max(0, CALIBRATION_SECONDS - elapsed)
            cv2.putText(frame, f"Hold steady... {remaining:.1f}s",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            if elapsed >= CALIBRATION_SECONDS:
                avg_span = sum(spans) / len(spans)
                with open(CALIBRATION_FILE, "w") as f:
                    json.dump({"hand_span": avg_span}, f)
                print(f"Calibration done. Hand span = {avg_span:.4f}. Saved to {CALIBRATION_FILE}")
                break
        else:
            start_time = None
            spans = []
            cv2.putText(frame, "Show your hand(s) with fingers spread",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Calibration", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

