
"""
STEP 5: The Advanced App

Combines:
  - Calibration-aware landmark normalization (uses data/calibration.json if present)
  - Prediction smoothing (last N frames must agree before committing a letter -- stops flicker)
  - A confidence bar drawn on screen
  - A correction workflow: press 'x' to log a wrong prediction with the right
    answer into a local SQLite database (data/corrections.db), which you can
    later fold back into training data to improve the model over time
  - Optional LLM step: press 'l' to send the raw recognized glosses (e.g.
    "H E L L O") to Claude and get back a natural sentence, which is then spoken

Requires an ANTHROPIC_API_KEY environment variable for the LLM step. If it's
not set, the LLM step is skipped gracefully and the raw text is spoken instead.

Controls:
  SPACE = add space | b = backspace | c = clear sentence
  s = speak sentence as-is | l = polish with LLM then speak
  x = log a correction for the current prediction | q = quit

Run:  python 5_advanced_app.py
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import joblib
import json
import os
import time
import sqlite3
import pyttsx3
from collections import deque, Counter

MODEL_FILE = "data/sign_model.pkl"
LANDMARKER_MODEL = "hand_landmarker.task"
CALIBRATION_FILE = "data/calibration.json"
DB_FILE = "data/corrections.db"

CONFIDENCE_THRESHOLD = 0.4          # lower = more permissive; tune based on your model
SMOOTHING_WINDOW = 8                # how many recent frames must agree
HOLD_TIME = 0.6

# ---------- Setup ----------
os.makedirs("data", exist_ok=True)

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

# Load calibration if available
calibrated_span = None
if os.path.exists(CALIBRATION_FILE):
    with open(CALIBRATION_FILE) as f:
        calibrated_span = json.load(f)["hand_span"]
    print(f"Loaded calibration: hand_span={calibrated_span:.4f}")
else:
    print("No calibration found -- run 4_calibrate.py first for best accuracy. "
          "Falling back to per-frame scaling.")

# Correction DB setup
conn = sqlite3.connect(DB_FILE)
conn.execute("""
    CREATE TABLE IF NOT EXISTS corrections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        predicted_label TEXT,
        correct_label TEXT,
        features TEXT,
        timestamp REAL
    )
""")
conn.commit()

# Optional LLM client
llm_client = None
try:
    import anthropic
    if os.environ.get("ANTHROPIC_API_KEY"):
        llm_client = anthropic.Anthropic()
except ImportError:
    pass


def polish_with_llm(glosses):
    """Turn a rough list of recognized signs into a natural sentence."""
    if not llm_client or not glosses:
        return " ".join(glosses)
    try:
        prompt = (
            "You are helping a deaf/mute person communicate. Below is a sequence "
            "of recognized Indian Sign Language glosses (rough word-by-word output, "
            "no grammar). Turn it into a short, natural, grammatically correct "
            "sentence in the same apparent language. Output ONLY the sentence, "
            "nothing else.\n\nGlosses: " + " ".join(glosses)
        )
        response = llm_client.messages.create(
            model="claude-sonnet-5",
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except Exception as e:
        print(f"LLM call failed ({e}), using raw text instead.")
        return " ".join(glosses)


def speak(text):
    if text.strip():
        tts_engine.say(text)
        tts_engine.runAndWait()


# ---------- Landmark extraction (calibration-aware) ----------
def landmarks_to_list(hand_landmarks, scale_override=None):
    base_x = hand_landmarks[0].x
    base_y = hand_landmarks[0].y
    base_z = hand_landmarks[0].z
    coords = [(lm.x - base_x, lm.y - base_y, lm.z - base_z) for lm in hand_landmarks]
    if scale_override:
        max_dist = scale_override
    else:
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
            values = landmarks_to_list(hand_landmarks, scale_override=calibrated_span)
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


def draw_confidence_bar(frame, confidence):
    bar_x, bar_y, bar_w, bar_h = 10, 100, 200, 20
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
    fill_w = int(bar_w * confidence)
    color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 100, 255)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)
    cv2.putText(frame, f"{confidence:.2f}", (bar_x + bar_w + 10, bar_y + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


def main():
    cap = cv2.VideoCapture(0)
    sentence = ""
    glosses = []  # list of committed signs, for LLM polishing
    recent_predictions = deque(maxlen=SMOOTHING_WINDOW)
    last_committed = None
    last_commit_time = 0
    frame_ts = 0

    last_features = None
    last_predicted_label = None

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        frame_ts += 33
        result = landmarker.detect_for_video(mp_image, frame_ts)

        predicted_label = None
        confidence = 0.0

        if result.hand_landmarks:
            draw_landmarks(frame, result)
            features = build_two_hand_row(result)
            probs = model.predict_proba([features])[0]
            best_idx = probs.argmax()
            predicted_label = model.classes_[best_idx]
            confidence = probs[best_idx]
            last_features = features
            last_predicted_label = predicted_label

        # --- smoothing: only count a frame's vote if confident enough ---
        if predicted_label is not None and confidence >= CONFIDENCE_THRESHOLD:
            recent_predictions.append(predicted_label)
        else:
            recent_predictions.append(None)

        # Commit a letter only if the smoothing window mostly agrees
        now = time.time()
        if len(recent_predictions) == SMOOTHING_WINDOW:
            counts = Counter(recent_predictions)
            top_label, top_count = counts.most_common(1)[0]
            if top_label is not None and top_count >= SMOOTHING_WINDOW * 0.7:
                if top_label != last_committed or (now - last_commit_time) > HOLD_TIME:
                    sentence += top_label
                    glosses.append(top_label)
                    last_committed = top_label
                    last_commit_time = now

        # --- UI ---
        cv2.putText(frame, f"Sign: {predicted_label or '-'}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        draw_confidence_bar(frame, confidence)
        cv2.putText(frame, f"Text: {sentence}",
                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, "SPACE=space b=back c=clear s=speak l=LLM-polish x=correct q=quit",
                    (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.imshow("ISL Advanced App", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == 32:
            sentence += " "
        elif key == ord("b"):
            sentence = sentence[:-1]
        elif key == ord("c"):
            sentence = ""
            glosses = []
        elif key == ord("s"):
            speak(sentence)
        elif key == ord("l"):
            polished = polish_with_llm(glosses)
            print(f"LLM polished sentence: {polished}")
            speak(polished)
        elif key == ord("x"):
            if last_features is not None:
                print(f"Current prediction was: {last_predicted_label}")
                correct = input("Type the CORRECT label for what you just signed: ").strip().upper()
                if correct:
                    conn.execute(
                        "INSERT INTO corrections (predicted_label, correct_label, features, timestamp) VALUES (?, ?, ?, ?)",
                        (last_predicted_label, correct, json.dumps(last_features), time.time()),
                    )
                    conn.commit()
                    print(f"Logged correction: predicted={last_predicted_label} -> correct={correct}")

    cap.release()
    cv2.destroyAllWindows()
    conn.close()


if __name__ == "__main__":
    main()
