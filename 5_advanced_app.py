
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
from deep_translator import GoogleTranslator
from gtts import gTTS
from playsound import playsound
from elevenlabs.client import ElevenLabs
from elevenlabs import play as elevenlabs_play
from PIL import Image, ImageDraw, ImageFont
import numpy as np

MODEL_FILE = "data/sign_model.pkl"
LANDMARKER_MODEL = "hand_landmarker.task"
CALIBRATION_FILE = "data/calibration.json"
DB_FILE = "data/corrections.db"

CONFIDENCE_THRESHOLD = 0.4          # lower = more permissive; tune based on your model
SMOOTHING_WINDOW = 8                # how many recent frames must agree
HOLD_TIME = 0.6
LANGUAGES = {
    "1": ("English", "en", "en-GB"),
    "2": ("Hindi", "hi", "hi-IN"),
    "3": ("Telugu", "te", "te-IN"),
    "4": ("Bengali", "bn", "bn-IN"),
    "5": ("Odia", "or", "or-IN"),
}

current_lang_key = "1"  # default English

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
translation_cache = {}
last_speak_time = 0
SPEAK_COOLDOWN = 2.0
displayed_translation = ""

def translate_text(text, lang_code):
    """Try Google Translate first, fall back to MyMemory if rate-limited."""
    cache_key = (text, lang_code)
    if cache_key in translation_cache:
        return translation_cache[cache_key]

    try:
        from deep_translator import GoogleTranslator
        result = GoogleTranslator(source="en", target=lang_code).translate(text)
        translation_cache[cache_key] = result
        return result
    except Exception as e1:
        print(f"Google Translate failed ({e1}), trying backup translator...")
        try:
            from deep_translator import MyMemoryTranslator
            result = MyMemoryTranslator(source="en-GB", target=lang_code).translate(text)
            translation_cache[cache_key] = result
            return result
        except Exception as e2:
            print(f"Backup translator also failed ({e2}).")
            raise
def put_unicode_text(img, text, position, font_size=24, color=(255, 150, 0)):

    img_pil = Image.fromarray(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    )

    draw = ImageDraw.Draw(img_pil)

    # Try fonts that support Indian scripts
    font_paths = [
        r"C:\Windows\Fonts\NirmalaUI.ttf",
        r"C:\Windows\Fonts\Nirmala.ttf",
        r"C:\Windows\Fonts\NotoSansDevanagari-Regular.ttf",
        r"C:\Windows\Fonts\NotoSansBengali-Regular.ttf",
        r"C:\Windows\Fonts\NotoSansTelugu-Regular.ttf",
        r"C:\Windows\Fonts\NotoSansOriya-Regular.ttf",
        r"C:\Windows\Fonts\gautami.ttf",
        r"C:\Windows\Fonts\FreeSans.ttf"
    ]

    font = None

    for path in font_paths:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, font_size)
                print("Using font:", path)
                break
            except Exception as e:
                print("Could not load:", path, e)

    if font is None:
        print(" NO UNICODE FONT FOUND!")
        return img

    draw.text(
        position,
        text,
        font=font,
        fill=color
    )

    return cv2.cvtColor(
        np.array(img_pil),
        cv2.COLOR_RGB2BGR
    )
def put_unicode_text(img, text, position, font_size=24, color=(255, 150, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # Local project ki NIRMALA.TTC file ka index=0 use karenge
    font_paths = [
        "NIRMALA.TTC",
        "Nirmala.ttc",
        r"C:\Windows\Fonts\NIRMALA.TTC"
    ]

    font = None
    for path in font_paths:
        if os.path.exists(path):
            try:
                # TTC file ke liye index=0 batana padta hai
                font = ImageFont.truetype(path, font_size, index=0)
                break
            except Exception as e:
                print(f"Font loading error: {e}")
                continue

    if font is None:
        font = ImageFont.load_default()

    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
# ============================================================
# Replace your old speak(), translate_text(), and the top-level
# translation_cache / last_speak_time / displayed_translation
# with everything below.
# ============================================================

import os
from elevenlabs.client import ElevenLabs
from elevenlabs import play as elevenlabs_play

VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # <-- replace with your chosen voice ID if different

elevenlabs_client = None
if os.environ.get("ELEVENLABS_API_KEY"):
    elevenlabs_client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])

translation_cache = {}
last_speak_time = 0
SPEAK_COOLDOWN = 2.0
displayed_translation = ""

COMMON_WORDS = {
    "hi": {"HELLO": "नमस्ते", "THANK YOU": "धन्यवाद", "YES": "हाँ", "NO": "नहीं",
           "PLEASE": "कृपया", "SORRY": "माफ़ करें", "HELP": "मदद", "NAME": "नाम"},
    "te": {"HELLO": "నమస్కారం", "THANK YOU": "ధన్యవాదాలు", "YES": "అవును", "NO": "లేదు"},
    "bn": {"HELLO": "নমস্কার", "THANK YOU": "ধন্যবাদ", "YES": "হ্যাঁ", "NO": "না"},
    "or": {"HELLO": "ନମସ୍କାର", "THANK YOU": "ଧନ୍ୟବାଦ", "YES": "ହଁ", "NO": "ନାଁ"},
}


def translate_text(text, lang_code, mymemory_code):
    text_upper = text.strip().upper()
    if lang_code in COMMON_WORDS and text_upper in COMMON_WORDS[lang_code]:
        return COMMON_WORDS[lang_code][text_upper]

    cache_key = (text, lang_code)
    if cache_key in translation_cache:
        return translation_cache[cache_key]
    try:
        from deep_translator import GoogleTranslator
        result = GoogleTranslator(source="en", target=lang_code).translate(text)
        translation_cache[cache_key] = result
        return result
    except Exception:
        try:
            from deep_translator import MyMemoryTranslator
            result = MyMemoryTranslator(source="en-GB", target=mymemory_code).translate(text)
            translation_cache[cache_key] = result
            return result
        except Exception as e2:
            print(f"Translation failed completely ({e2}).")
            raise

def speak(text):
    global last_speak_time, displayed_translation
    if not text.strip():
        return

    now = time.time()
    if now - last_speak_time < SPEAK_COOLDOWN:
        print("Please wait a moment before speaking again.")
        return
    last_speak_time = now

    lang_name, lang_code, mymemory_code = LANGUAGES[current_lang_key]

    try:
        if lang_code != "en":
            translated = translate_text(text, lang_code, mymemory_code)
        else:
            translated = text
        displayed_translation = translated
        print(f"Speaking ({lang_name}): {translated}")

        # Odia isn't confirmed-supported on ElevenLabs -- use gTTS for it
        if lang_code == "or" or elevenlabs_client is None:
            raise RuntimeError("Falling back to gTTS for this language / missing API key")

        audio = elevenlabs_client.text_to_speech.convert(
            voice_id=VOICE_ID,
            text=translated,
            model_id="eleven_v3",
        )
        elevenlabs_play(audio)

    except Exception as e:
        print(f"ElevenLabs failed or skipped ({e}), trying gTTS...")
        try:
            tts = gTTS(text=translated, lang=lang_code)
            tts.save("temp_speech.mp3")
            playsound("temp_speech.mp3")
        except Exception as e2:
            print(f"gTTS also failed ({e2}), using offline English voice.")
            displayed_translation = text
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

def build_two_hand_row(result, scale_override=None):
    slot1 = [0.0] * 63
    slot2 = [0.0] * 63
    if result.hand_landmarks:
        # Sort hands by their x-position on screen (left-to-right), NOT by
        # mediapipe's Left/Right label -- this stays consistent whether or
        # not the frame is mirrored.
        hands_sorted = sorted(result.hand_landmarks, key=lambda hl: hl[0].x)
        if len(hands_sorted) >= 1:
            slot1 = landmarks_to_list(hands_sorted[0], scale_override)
        if len(hands_sorted) >= 2:
            slot2 = landmarks_to_list(hands_sorted[1], scale_override)
    return slot1 + slot2

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
    global current_lang_key
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

        # Track how many recent frames had NO confident hand -- this means
        # the person moved their hand away / reset, so we're ready for a
        # fresh letter next.
        none_count = sum(1 for p in recent_predictions if p is None)
        if none_count >= SMOOTHING_WINDOW - 2:  # mostly empty -> reset
            last_committed = None

        # Commit a letter only if the smoothing window mostly agrees,
        # AND it's a genuinely new letter (hand was reset since the last commit)
        if len(recent_predictions) == SMOOTHING_WINDOW:
            counts = Counter(recent_predictions)
            top_label, top_count = counts.most_common(1)[0]
            if top_label is not None and top_count >= SMOOTHING_WINDOW * 0.7:
                if top_label != last_committed:
                    sentence += top_label
                    glosses.append(top_label)
                    last_committed = top_label
                    
        # --- UI ---
        cv2.putText(frame, f"Sign: {predicted_label or '-'}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        frame = put_unicode_text(frame, f"Translated: {displayed_translation}",
                         (10, 210), font_size=26, color=(255, 150, 0))
        draw_confidence_bar(frame, confidence)
        cv2.putText(frame, f"Lang: {LANGUAGES[current_lang_key][0]} (press 1-5 to change)",
                    (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
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
        elif chr(key) in LANGUAGES if key < 256 else False:
            current_lang_key = chr(key)
            print(f"Language switched to: {LANGUAGES[current_lang_key][0]}")

    cap.release()
    cv2.destroyAllWindows()
    conn.close()

if __name__ == "__main__":
    main()










