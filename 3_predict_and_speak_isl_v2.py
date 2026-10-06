"""
STEP 3 (ISL version, NEW Tasks API): Real-time ISL Sign -> Text -> Voice

Run this LOCALLY on your own laptop (not on Kaggle/Colab) since it needs
your webcam.

Before running, make sure hand_landmarker.task is in the same folder
(download it if you don't already have it locally):

    wget -O hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

Also make sure data/sign_model.pkl (downloaded from Kaggle) is in place.

Controls: SPACE = space | b = backspace | s = speak | c = clear | q = quit

Run:  python 3_predict_and_speak_isl_v2.py
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import joblib
import time
import pyttsx3

MODEL_FILE = "data/sign_model.pkl"
LANDMARKER_MODEL = "hand_landmarker.task"
CONFIDENCE_THRESHOLD = 0.3
HOLD_TIME = 0.5

base_options = mp_python.BaseOptions(model_asset_path=LANDMARKER_MODEL)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    running_mode=vision.RunningMode.VIDEO,
)
landmarker = vision.HandLandmarker.create_from_options(options)

model = joblib.load(MODEL_FILE)

tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 160)

def landmarks_to_list(hand_landmarks):
    base_x = hand_landmarks[0].x
    base_y = hand_landmarks[0].y
    base_z = hand_landmarks[0].z
    coords = [(lm.x - base_x, lm.y - base_y, lm.z - base_z) for lm in hand_landmarks]
    max_dist = max((x**2 + y**2 + z**2) ** 0.5 for x, y, z in coords) or 1e-6
    row = []
    for x, y, z in coords:
        row.extend([x / max_dist, y / max_dist, z / max_dist])
    return row

def build_two_hand_row(result):
    left = [0.0] * 63
    right = [0.0] * 63
    if result.hand_landmarks:
        for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            label = handedness[0].category_name
            values = landmarks_to_list(hand_landmarks)
            if label == "Left":
                left = values
            else:
                right = values
    return left + right


def draw_landmarks(frame, result):
    h, w, _ = frame.shape
    for hand_landmarks in result.hand_landmarks:
        for lm in hand_landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)


def speak(text):
    if text.strip():
        tts_engine.say(text)
        tts_engine.runAndWait()


def main():
    cap = cv2.VideoCapture(0)
    sentence = ""
    last_label = None
    hold_start = None
    last_added_time = 0
    frame_timestamp_ms = 0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        frame_timestamp_ms += 33  # approx for ~30fps
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        predicted_label = None
        confidence = 0.0

        if result.hand_landmarks:
            draw_landmarks(frame, result)
            features = build_two_hand_row(result)
            probs = model.predict_proba([features])[0]
            best_idx = probs.argmax()
            predicted_label = model.classes_[best_idx]
            confidence = probs[best_idx]

        now = time.time()
        if predicted_label is not None and confidence >= CONFIDENCE_THRESHOLD:
            if predicted_label == last_label:
                if hold_start and (now - hold_start) >= HOLD_TIME and (now - last_added_time) > HOLD_TIME:
                    sentence += predicted_label
                    last_added_time = now
            else:
                last_label = predicted_label
                hold_start = now
        else:
            last_label = None
            hold_start = None

        cv2.putText(frame, f"Sign: {predicted_label or '-'} ({confidence:.2f})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Text: {sentence}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, "SPACE=space  b=backspace  s=speak  c=clear  q=quit",
                    (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("ISL to Text/Voice", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == 32:
            sentence += " "
        elif key == ord("b"):
            sentence = sentence[:-1]
        elif key == ord("c"):
            sentence = ""
        elif key == ord("s"):
            speak(sentence)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
