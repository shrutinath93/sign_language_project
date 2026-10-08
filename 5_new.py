"""
ISL Sign Language -> Text -> Voice   (UI-upgraded version of 5_advanced_app.py)

What's new vs. the old file:
  * Starts FULLSCREEN (press F to toggle), UI scales to any window size
  * Modern dark overlay UI: title bar, big text cards, detected-sign card with
    confidence bar, language selector pills, key-hint bar
  * Hindi / Telugu / Bengali / Odia text is drawn crisply (Nirmala UI font)
  * Hand skeleton drawn with connected lines
  * Correction ('X') is typed INSIDE the window (no hidden terminal in fullscreen)
  * 'Speaking...' status shows while audio is being generated/played
  * Recognition / translation / voice logic is the same as before

Keys:
  SPACE add space | B backspace | C clear | S speak | L AI-polish + speak
  X log a correction | 1-5 language | F fullscreen | Q quit

Run:  python 5_advanced_app_ui.py
"""

import os
import json
import time
import sqlite3
import warnings
from collections import deque, Counter

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PIL import features as pil_features
try:                      # correct Hindi/Telugu/Bengali/Odia text shaping
    import uharfbuzz as hb
    import freetype
    HAVE_SHAPER = True
except Exception:
    HAVE_SHAPER = False
import joblib
import pyttsx3
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from gtts import gTTS
from playsound import playsound
from elevenlabs.client import ElevenLabs
from elevenlabs import play as elevenlabs_play

# sklearn prints a warning on every frame otherwise
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# ============================================================
# SETTINGS
# ============================================================
MODEL_FILE = "data/sign_model.pkl"
LANDMARKER_MODEL = "hand_landmarker.task"
CALIBRATION_FILE = "data/calibration.json"
DB_FILE = "data/corrections.db"

CONFIDENCE_THRESHOLD = 0.4   # lower = more permissive
SMOOTHING_WINDOW = 8         # how many recent frames must agree
SPEAK_COOLDOWN = 2.0         # seconds between speak calls (avoids rate limits)

CAM_W, CAM_H = 1280, 720     # if the app feels laggy, try 960 x 540
WINDOW_NAME = "ISL Sign Language to Voice"
START_FULLSCREEN = True
MAX_RENDER_W = 1600          # UI is drawn at most this wide, then scaled up (keeps FPS high)

# The model was trained with per-frame hand scaling, so calibration scaling stays OFF.
USE_CALIBRATION = False
# The model was tested on 4:3 webcam frames (640x480). For other shapes (e.g. 16:9)
# landmark x-values are corrected so the model sees what it saw during testing.
REF_ASPECT = 4 / 3

LANGUAGES = {
    "1": ("English", "en", "en-GB"),
    "2": ("Hindi", "hi", "hi-IN"),
    "3": ("Telugu", "te", "te-IN"),
    "4": ("Bengali", "bn", "bn-IN"),
    "5": ("Odia", "or", "or-IN"),
}
NATIVE_NAMES = {"1": "English", "2": "हिन्दी", "3": "తెలుగు", "4": "বাংলা", "5": "ଓଡ଼ିଆ"}

VOICE_ID = "21m00Tcm4TlvDq8ikWAM"   # ElevenLabs voice (change if you picked another)
ELEVEN_MODEL = "eleven_v3"

COMMON_WORDS = {
    "hi": {"HELLO": "नमस्ते", "THANK YOU": "धन्यवाद", "YES": "हाँ", "NO": "नहीं",
           "PLEASE": "कृपया", "SORRY": "माफ़ करें", "HELP": "मदद", "NAME": "नाम"},
    "te": {"HELLO": "నమస్కారం", "THANK YOU": "ధన్యవాదాలు", "YES": "అవును", "NO": "లేదు"},
    "bn": {"HELLO": "নমস্কার", "THANK YOU": "ধন্যবাদ", "YES": "হ্যাঁ", "NO": "না"},
    "or": {"HELLO": "ନମସ୍କାର", "THANK YOU": "ଧନ୍ୟବାଦ", "YES": "ହଁ", "NO": "ନାଁ"},
}

# ============================================================
# GLOBAL STATE (filled in by init_services)
# ============================================================
landmarker = None
model = None
tts_engine = None
llm_client = None
elevenlabs_client = None
db_conn = None
calibrated_span = None

current_lang_key = "1"
translation_cache = {}
last_speak_time = 0.0
displayed_translation = ""


def init_services():
    global landmarker, model, tts_engine, llm_client, elevenlabs_client
    global db_conn, calibrated_span

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

    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE) as f:
            calibrated_span = json.load(f)["hand_span"]
        print(f"Loaded calibration: hand_span={calibrated_span:.4f}")
    else:
        print("No calibration found -- run 4_calibrate.py first for best accuracy.")

    db_conn = sqlite3.connect(DB_FILE)
    db_conn.execute("""
        CREATE TABLE IF NOT EXISTS corrections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            predicted_label TEXT,
            correct_label TEXT,
            features TEXT,
            timestamp REAL
        )
    """)
    db_conn.commit()

    try:
        import anthropic
        if os.environ.get("ANTHROPIC_API_KEY"):
            llm_client = anthropic.Anthropic()
    except ImportError:
        pass

    if os.environ.get("ELEVENLABS_API_KEY"):
        elevenlabs_client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])


# ============================================================
# TRANSLATION + VOICE
# ============================================================
def polish_with_llm(glosses, fallback_text):
    """Turn rough recognized signs into a natural sentence (needs ANTHROPIC_API_KEY)."""
    if not llm_client or not glosses:
        return fallback_text
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
        return fallback_text


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
    translated = text

    try:
        if lang_code != "en":
            translated = translate_text(text, lang_code, mymemory_code)
        displayed_translation = translated
        print(f"Speaking ({lang_name}): {translated}")

        # Odia isn't supported on ElevenLabs -> gTTS fallback
        if lang_code == "or" or elevenlabs_client is None:
            raise RuntimeError("Using gTTS for this language / missing ElevenLabs key")

        audio = elevenlabs_client.text_to_speech.convert(
            voice_id=VOICE_ID,
            text=translated,
            model_id=ELEVEN_MODEL,
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


# ============================================================
# LANDMARKS
# ============================================================
def landmarks_to_list(hand_landmarks, scale_override=None, ax=1.0):
    """ax = aspect correction for x (and z), 1.0 for a 4:3 camera frame."""
    base_x = hand_landmarks[0].x
    base_y = hand_landmarks[0].y
    base_z = hand_landmarks[0].z
    coords = [((lm.x - base_x) * ax, lm.y - base_y, (lm.z - base_z) * ax)
              for lm in hand_landmarks]
    if scale_override:
        max_dist = scale_override
    else:
        max_dist = max((x**2 + y**2 + z**2) ** 0.5 for x, y, z in coords) or 1e-6
    row = []
    for x, y, z in coords:
        row.extend([x / max_dist, y / max_dist, z / max_dist])
    return row


def build_two_hand_row(result, scale_override=None, ax=1.0):
    slot1 = [0.0] * 63
    slot2 = [0.0] * 63
    if result.hand_landmarks:
        # sort by on-screen x-position (left->right), not by MediaPipe's Left/Right label
        hands_sorted = sorted(result.hand_landmarks, key=lambda hl: hl[0].x)
        if len(hands_sorted) >= 1:
            slot1 = landmarks_to_list(hands_sorted[0], scale_override, ax)
        if len(hands_sorted) >= 2:
            slot2 = landmarks_to_list(hands_sorted[1], scale_override, ax)
    return slot1 + slot2


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]
FINGERTIPS = {4, 8, 12, 16, 20}


def draw_hand_skeleton(frame, result):
    h, w = frame.shape[:2]
    for hl in result.hand_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hl]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (191, 212, 45), 3, cv2.LINE_AA)   # teal (BGR)
        for i, p in enumerate(pts):
            if i in FINGERTIPS:
                cv2.circle(frame, p, 8, (36, 191, 251), -1, cv2.LINE_AA)       # amber
            else:
                cv2.circle(frame, p, 5, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, p, 5, (191, 212, 45), 1, cv2.LINE_AA)


# ============================================================
# UI RENDERING
# ============================================================
THEME = {
    "panel": (15, 23, 42),
    "panel_alpha": 205,
    "accent": (45, 212, 191),
    "accent_dark": (13, 148, 136),
    "amber": (251, 191, 36),
    "text": (241, 245, 249),
    "muted": (148, 163, 184),
    "good": (74, 222, 128),
    "warn": (251, 146, 60),
    "track": (51, 65, 85),
}
BG_BGR = (42, 23, 15)


def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


FONT_PATHS = {
    "bold": _first_existing([
        r"C:\Windows\Fonts\segoeuib.ttf", r"C:\Windows\Fonts\arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]),
    "regular": _first_existing([
        r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]),
    # Nirmala UI covers Devanagari, Telugu, Bengali and Odia
    "indic": _first_existing([
        "NIRMALA.TTC", "Nirmala.ttc", r"C:\Windows\Fonts\Nirmala.ttc",
        r"C:\Windows\Fonts\NIRMALA.TTC",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]),
}
_font_cache = {}
_font_bytes = {}
RAQM_OK = pil_features.check("raqm")
INDIC_OK = HAVE_SHAPER or RAQM_OK


class ShapedFont:
    """Draws Hindi/Telugu/Bengali/Odia correctly (HarfBuzz shaping + FreeType).
    Pillow alone cannot join conjuncts / place matras on Windows without Raqm+FriBiDi."""

    def __init__(self, path, size, index=0):
        self.size = int(size)
        if path not in _font_bytes:
            with open(path, "rb") as f:
                _font_bytes[path] = hb.Blob(f.read())
        self.hb_font = hb.Font(hb.Face(_font_bytes[path], index))
        self.hb_font.scale = (self.size * 64, self.size * 64)
        self.ft = freetype.Face(path, index)
        self.ft.set_pixel_sizes(0, self.size)
        self.ascent = int(self.ft.size.ascender / 64 + 0.99)
        self.descent = int(-self.ft.size.descender / 64 + 0.99)
        self._cache = {}

    def _shape(self, text):
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        hb.shape(self.hb_font, buf)
        return buf.glyph_infos, buf.glyph_positions

    def getlength(self, text):
        if not text:
            return 0.0
        _, poss = self._shape(text)
        return sum(p.x_advance for p in poss) / 64.0

    def _mask(self, text):
        if text in self._cache:
            return self._cache[text]
        infos, poss = self._shape(text)
        total = sum(p.x_advance for p in poss) / 64.0
        pad = self.size // 4 + 2
        W = int(total) + 2 * pad
        H = self.ascent + self.descent + pad
        baseline = self.ascent + pad // 2
        arr = np.zeros((H, W), np.uint8)
        pen = 0.0
        for info, pos in zip(infos, poss):
            self.ft.load_glyph(info.codepoint, freetype.FT_LOAD_RENDER | freetype.FT_LOAD_NO_HINTING)
            glyph = self.ft.glyph
            bmp = glyph.bitmap
            if bmp.rows and bmp.width:
                g = np.array(bmp.buffer, dtype=np.uint8).reshape(bmp.rows, bmp.pitch)[:, :bmp.width]
                gx = pad + int(round(pen + pos.x_offset / 64.0)) + glyph.bitmap_left
                gy = baseline - glyph.bitmap_top - int(round(pos.y_offset / 64.0))
                x0, y0 = max(gx, 0), max(gy, 0)
                x1, y1 = min(gx + g.shape[1], W), min(gy + g.shape[0], H)
                if x1 > x0 and y1 > y0:
                    sub = g[y0 - gy:y1 - gy, x0 - gx:x1 - gx]
                    arr[y0:y1, x0:x1] = np.maximum(arr[y0:y1, x0:x1], sub)
            pen += pos.x_advance / 64.0
        out = (Image.fromarray(arr, "L"), total, baseline, pad)
        if len(self._cache) > 64:
            self._cache.clear()
        self._cache[text] = out
        return out

    def draw(self, img, xy, text, fill, anchor="lm"):
        if not text:
            return
        mask, total, baseline, pad = self._mask(text)
        x, y = xy
        if anchor[0] == "m":
            x -= total / 2
        elif anchor[0] == "r":
            x -= total
        left = int(round(x)) - pad
        top = int(round(y + (self.ascent - self.descent) / 2.0)) - baseline
        color = tuple(fill) + (255,) if len(fill) == 3 else tuple(fill)
        img.paste(color, (left, top, left + mask.width, top + mask.height), mask)


def get_font(kind, size):
    size = max(8, int(size))
    key = (kind, size)
    if key in _font_cache:
        return _font_cache[key]
    font = None
    if kind == "indic" and HAVE_SHAPER and FONT_PATHS.get("indic"):
        try:
            font = ShapedFont(FONT_PATHS["indic"], size, 0)
        except Exception as e:
            print(f"ShapedFont unavailable ({e}); using plain Pillow text.")
    if font is None:
        path = (FONT_PATHS.get(kind) or FONT_PATHS["indic"]
                or FONT_PATHS["regular"] or FONT_PATHS["bold"])
        try:
            font = ImageFont.truetype(path, size, index=0) if path else ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()
    _font_cache[key] = font
    return font


def text_w(font, text):
    try:
        return font.getlength(text)
    except AttributeError:
        return font.getsize(text)[0]


def put_text(img, xy, text, font, fill, anchor="lm"):
    """Draw text on a PIL image, using HarfBuzz shaping for Indic fonts."""
    if isinstance(font, ShapedFont):
        font.draw(img, xy, text, fill, anchor)
    else:
        ImageDraw.Draw(img).text(xy, text, font=font, fill=fill, anchor=anchor)


def fit_tail(text, font, max_w):
    """If text is too wide, keep the END of it (latest part) with '...' in front."""
    if text_w(font, text) <= max_w:
        return text
    if isinstance(font, ShapedFont):          # trim whole words so scripts never break
        words = text.split(" ")
        while len(words) > 1 and text_w(font, "... " + " ".join(words)) > max_w:
            words = words[1:]
        return "... " + " ".join(words)
    while text and text_w(font, "..." + text) > max_w:
        text = text[1:]
    return "..." + text


def letterbox(frame, win_w, win_h):
    fh, fw = frame.shape[:2]
    if fw == win_w and fh == win_h:
        return frame
    scale = min(win_w / fw, win_h / fh)
    nw, nh = max(1, int(fw * scale)), max(1, int(fh * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((win_h, win_w, 3), BG_BGR, dtype=np.uint8)
    x0, y0 = (win_w - nw) // 2, (win_h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


class _Geo:
    """All layout numbers for a given window size (designed at 720p, scaled)."""
    def __init__(self, win_w, win_h):
        s = win_h / 720.0
        S = self.S = lambda v: int(round(v * s))
        self.margin = S(24)
        self.top_h = S(56)
        self.card_y0 = S(68)
        self.card_y1 = self.card_y0 + S(172)
        self.text_x0 = self.margin
        self.text_x1 = win_w - self.margin
        self.side_y0 = self.card_y1 + S(14)
        self.sign_w, self.sign_h = S(190), S(172)
        self.pill_w, self.pill_h, self.pill_gap = S(240), S(44), S(10)
        self.pill_x0 = win_w - self.margin - self.pill_w
        self.ctrl_h = S(46)


HINT_TEXT = "Show your hand(s) to the camera"
_layer_cache = {}


def build_static_layer(win_w, win_h, lang_key, show_hint, toast):
    """Everything that does not change every frame: panels, labels, pills, key hints.
    Built once and cached, so the live loop stays fast."""
    g = _Geo(win_w, win_h)
    S = g.S
    P, PA = THEME["panel"], THEME["panel_alpha"]
    layer = Image.new("RGBA", (win_w, win_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)

    f_title = get_font("bold", S(24))
    f_label = get_font("bold", S(13))
    f_pill = get_font("indic", S(20))
    f_pill_s = get_font("regular", S(14))
    f_key = get_font("bold", S(14))
    f_chip = get_font("regular", S(15))
    f_toast = get_font("bold", S(20))
    f_hint = get_font("bold", S(24))
    f_num = get_font("bold", S(16))

    # --- top bar
    d.rectangle([0, 0, win_w, g.top_h], fill=P + (230,))
    cy = g.top_h // 2
    d.ellipse([g.margin, cy - S(9), g.margin + S(18), cy + S(9)], fill=THEME["accent"])
    d.text((g.margin + S(30), cy), "ISL  Sign Language to Voice", font=f_title,
           fill=THEME["text"], anchor="lm")

    # --- text card (labels only; the text itself is dynamic)
    d.rounded_rectangle([g.text_x0, g.card_y0, g.text_x1, g.card_y1], radius=S(20), fill=P + (PA,))
    inner_x = g.text_x0 + S(22)
    d.text((inner_x, g.card_y0 + S(16)), "RECOGNIZED TEXT", font=f_label,
           fill=THEME["muted"], anchor="lm")
    lang_name = LANGUAGES[lang_key][0]
    d.text((inner_x, g.card_y0 + S(100)), f"TRANSLATED  -  {lang_name.upper()}",
           font=f_label, fill=THEME["muted"], anchor="lm")

    # --- detected-sign card frame
    sx, sy = g.margin, g.side_y0
    d.rounded_rectangle([sx, sy, sx + g.sign_w, sy + g.sign_h], radius=S(20), fill=P + (PA,))
    d.text((sx + S(16), sy + S(18)), "DETECTED", font=f_label, fill=THEME["muted"], anchor="lm")
    bx, bw = sx + S(16), g.sign_w - S(32)
    by, bh = sy + g.sign_h - S(28), S(10)
    d.rounded_rectangle([bx, by, bx + bw, by + bh], radius=S(5), fill=THEME["track"])
    tick = bx + int(bw * CONFIDENCE_THRESHOLD)
    d.rectangle([tick - 1, by - S(4), tick + 1, by + bh + S(4)], fill=THEME["text"])

    # --- language pills
    for i, (k, (name, _c, _m)) in enumerate(LANGUAGES.items()):
        y = g.side_y0 + i * (g.pill_h + g.pill_gap)
        active = (k == lang_key)
        d.rounded_rectangle([g.pill_x0, y, g.pill_x0 + g.pill_w, y + g.pill_h],
                            radius=g.pill_h // 2,
                            fill=(THEME["accent"] + (255,)) if active else (P + (PA,)))
        cyp = y + g.pill_h // 2
        r = S(14)
        bx0 = g.pill_x0 + S(10)
        d.ellipse([bx0, cyp - r, bx0 + 2 * r, cyp + r], fill=P if active else THEME["track"])
        d.text((bx0 + r, cyp), k, font=f_num,
               fill=THEME["accent"] if active else THEME["text"], anchor="mm")
        label = NATIVE_NAMES[k] if INDIC_OK else name
        put_text(layer, (bx0 + 2 * r + S(12), cyp), label, f_pill,
                 P if active else THEME["text"], "lm")
        if INDIC_OK and k != "1":
            d.text((g.pill_x0 + g.pill_w - S(18), cyp), name, font=f_pill_s,
                   fill=P if active else THEME["muted"], anchor="rm")

    # --- bottom key hints
    d.rectangle([0, win_h - g.ctrl_h, win_w, win_h], fill=P + (235,))
    chips = [("SPACE", "Space"), ("B", "Back"), ("C", "Clear"), ("S", "Speak"),
             ("L", "AI polish"), ("X", "Correct"), ("F", "Fullscreen"), ("Q", "Quit")]
    x = g.margin
    cyc = win_h - g.ctrl_h // 2
    for key, label in chips:
        kw = int(max(S(26), text_w(f_key, key) + S(16)))
        d.rounded_rectangle([x, cyc - S(13), x + kw, cyc + S(13)], radius=S(6), fill=THEME["track"])
        d.text((x + kw // 2, cyc), key, font=f_key, fill=THEME["text"], anchor="mm")
        x += kw + S(8)
        d.text((x, cyc), label, font=f_chip, fill=THEME["muted"], anchor="lm")
        x += int(text_w(f_chip, label)) + S(22)

    # --- "show your hand" hint
    if show_hint:
        tw = text_w(f_hint, HINT_TEXT)
        hx0 = int((win_w - tw) // 2 - S(24))
        hx1 = int(hx0 + tw + S(48))
        hy1 = win_h - g.ctrl_h - S(24)
        hy0 = hy1 - S(52)
        d.rounded_rectangle([hx0, hy0, hx1, hy1], radius=S(26), fill=P + (PA,))
        d.text(((hx0 + hx1) // 2, (hy0 + hy1) // 2), HINT_TEXT, font=f_hint,
               fill=THEME["text"], anchor="mm")

    # --- status toast
    if toast:
        tw2 = text_w(f_toast, toast)
        tx0 = int((win_w - tw2) // 2 - S(22))
        tx1 = int(tx0 + tw2 + S(44))
        ty0 = g.side_y0 + S(10)
        ty1 = ty0 + S(44)
        d.rounded_rectangle([tx0, ty0, tx1, ty1], radius=S(22),
                            fill=THEME["accent_dark"] + (240,))
        d.text(((tx0 + tx1) // 2, (ty0 + ty1) // 2), toast, font=f_toast,
               fill=(255, 255, 255), anchor="mm")
    return layer


def draw_dynamic(img, st, win_w, win_h):
    """Parts that change every frame: fps, typed text, translation, detected sign."""
    d = ImageDraw.Draw(img)
    g = _Geo(win_w, win_h)
    S = g.S

    f_small = get_font("regular", S(14))
    f_label = get_font("bold", S(13))
    f_sent = get_font("bold", S(42))
    f_trans = get_font("indic", S(38))
    f_big = get_font("bold", S(88))

    # top bar (right side)
    cy = g.top_h // 2
    right = win_w - g.margin
    fps_txt = f"{st['fps']:.0f} FPS"
    d.text((right, cy), fps_txt, font=f_small, fill=THEME["muted"], anchor="rm")
    right -= int(text_w(f_small, fps_txt)) + S(26)
    cal_txt = "Calibrated" if st["calibrated"] else "Auto scaling"
    d.text((right, cy), cal_txt, font=f_small, fill=THEME["text"], anchor="rm")
    right -= int(text_w(f_small, cal_txt)) + S(10)
    d.ellipse([right - S(12), cy - S(6), right, cy + S(6)], fill=THEME["good"])

    # text card
    inner_x = g.text_x0 + S(22)
    max_w = (g.text_x1 - g.text_x0) - S(44)
    y_s = g.card_y0 + S(52)
    sent = fit_tail(st["sentence"], f_sent, max_w - S(20))
    if sent:
        d.text((inner_x, y_s), sent, font=f_sent, fill=THEME["text"], anchor="lm")
    else:
        d.text((inner_x + S(14), y_s), "Start signing...", font=f_sent, fill=THEME["track"], anchor="lm")
    if st["blink"] and st["mode"] != "correct":
        cx = inner_x + (int(text_w(f_sent, sent)) + S(6) if sent else 0)
        d.rectangle([cx, y_s - S(22), cx + S(3), y_s + S(22)], fill=THEME["accent"])

    y_t = g.card_y0 + S(138)
    trans = fit_tail(st["translated"], f_trans, max_w)
    if trans:
        put_text(img, (inner_x, y_t), trans, f_trans, THEME["amber"], "lm")
    else:
        put_text(img, (inner_x, y_t), "Press S to speak", f_trans, THEME["track"], "lm")

    # detected sign card
    sx, sy = g.margin, g.side_y0
    conf = max(0.0, min(1.0, float(st["confidence"])))
    d.text((sx + g.sign_w - S(16), sy + S(18)), f"{int(conf * 100)}%", font=f_label,
           fill=THEME["muted"], anchor="rm")
    sign = st["sign"]
    d.text((sx + g.sign_w // 2, sy + S(80)), str(sign) if sign else "?", font=f_big,
           fill=THEME["accent"] if sign else THEME["track"], anchor="mm")
    bx, bw = sx + S(16), g.sign_w - S(32)
    by, bh = sy + g.sign_h - S(28), S(10)
    fw = int(bw * conf)
    if fw > S(8):
        col = THEME["good"] if conf >= CONFIDENCE_THRESHOLD else THEME["warn"]
        d.rounded_rectangle([bx, by, bx + fw, by + bh], radius=S(5), fill=col)


def draw_correction_popup(img, st, win_w, win_h):
    g = _Geo(win_w, win_h)
    S = g.S
    P = THEME["panel"]
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    o = ImageDraw.Draw(ov)
    o.rectangle([0, 0, win_w, win_h], fill=(0, 0, 0, 150))
    mw, mh = S(580), S(270)
    mx0, my0 = (win_w - mw) // 2, (win_h - mh) // 2
    o.rounded_rectangle([mx0, my0, mx0 + mw, my0 + mh], radius=S(24), fill=P + (255,))
    img = Image.alpha_composite(img, ov)
    d = ImageDraw.Draw(img)

    f_small = get_font("regular", S(14))
    d.text((mx0 + S(30), my0 + S(38)), "Log a correction", font=get_font("bold", S(28)),
           fill=THEME["text"], anchor="lm")
    d.text((mx0 + S(30), my0 + S(78)),
           f"Model predicted  {st['pred'] or '-'}  -  type the correct sign (A-Z / 0-9)",
           font=f_small, fill=THEME["muted"], anchor="lm")
    ix0, iy0 = mx0 + S(30), my0 + S(104)
    ix1, iy1 = mx0 + mw - S(30), iy0 + S(80)
    d.rounded_rectangle([ix0, iy0, ix1, iy1], radius=S(14), fill=THEME["track"],
                        outline=THEME["accent"], width=max(2, S(2)))
    typed = st["input"]
    d.text(((ix0 + ix1) // 2, (iy0 + iy1) // 2), typed if typed else "_",
           font=get_font("bold", S(52)),
           fill=THEME["accent"] if typed else THEME["muted"], anchor="mm")
    d.text((mx0 + S(30), my0 + mh - S(44)),
           "ENTER  save        ESC  cancel        BACKSPACE  delete",
           font=f_small, fill=THEME["muted"], anchor="lm")
    return img


def render(cam_frame, st, win_w, win_h):
    """Draw the whole interface on top of the camera frame."""
    canvas = letterbox(cam_frame, win_w, win_h)
    img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGBA))

    show_hint = (not st["hands"]) and st["mode"] != "correct"
    toast = st.get("status") or ""
    key = (win_w, win_h, st["lang_key"], show_hint, toast)
    layer = _layer_cache.get(key)
    if layer is None:
        if len(_layer_cache) > 24:
            _layer_cache.clear()
        layer = build_static_layer(win_w, win_h, st["lang_key"], show_hint, toast)
        _layer_cache[key] = layer

    img = Image.alpha_composite(img, layer)
    draw_dynamic(img, st, win_w, win_h)
    if st["mode"] == "correct":
        img = draw_correction_popup(img, st, win_w, win_h)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGBA2BGR)


# ============================================================
# MAIN LOOP
# ============================================================
def get_window_size():
    try:
        _, _, w, h = cv2.getWindowImageRect(WINDOW_NAME)
    except Exception:
        w, h = CAM_W, CAM_H
    if w < 320 or h < 240:
        w, h = CAM_W, CAM_H
    return int(w), int(h)


def main():
    global current_lang_key, displayed_translation
    init_services()
    if not INDIC_OK:
        print("NOTE: Hindi/Telugu/Bengali/Odia text on screen may look broken.\n"
              "      Fix:  pip install uharfbuzz freetype-py")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    if not cap.isOpened():
        print("Could not open the webcam. Close other apps using it and try again.")
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, CAM_W, CAM_H)
    fullscreen = START_FULLSCREEN
    if fullscreen:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    sentence = ""
    glosses = []
    recent_predictions = deque(maxlen=SMOOTHING_WINDOW)
    last_committed = None
    frame_ts = 0
    last_features = None
    last_predicted_label = None
    fps = 0.0
    prev_t = time.time()

    ui = {"status": "", "until": 0.0, "mode": None, "input": "", "snap": None}

    # current-frame values (updated every loop, used by show())
    frame = None
    predicted_label = None
    confidence = 0.0
    hands_present = False

    def set_status(msg, seconds=2.5):
        ui["status"] = msg
        ui["until"] = time.time() + seconds

    def show():
        st = {
            "sign": predicted_label,
            "confidence": confidence,
            "sentence": sentence,
            "translated": displayed_translation,
            "lang_key": current_lang_key,
            "hands": hands_present,
            "calibrated": USE_CALIBRATION and calibrated_span is not None,
            "status": ui["status"] if time.time() < ui["until"] else "",
            "fps": fps,
            "mode": ui["mode"],
            "input": ui["input"],
            "pred": ui["snap"][1] if ui["snap"] else None,
            "blink": int(time.time() * 2) % 2 == 0,
        }
        ww, wh = get_window_size()
        if ww > MAX_RENDER_W:
            k = MAX_RENDER_W / ww
            out = render(frame, st, MAX_RENDER_W, int(wh * k))
            out = cv2.resize(out, (ww, wh), interpolation=cv2.INTER_LINEAR)
        else:
            out = render(frame, st, ww, wh)
        cv2.imshow(WINDOW_NAME, out)

    def run_blocking(msg, fn, *args):
        """Show a status banner, then run something slow (speech etc.)."""
        set_status(msg, 120)
        show()
        cv2.waitKey(1)
        try:
            fn(*args)
        finally:
            ui["until"] = 0.0

    def polish_and_speak(full_sentence):
        polished = polish_with_llm(list(glosses), full_sentence)
        print(f"Sentence to speak: {polished}")
        speak(polished)

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        now = time.time()
        dt = now - prev_t
        prev_t = now
        if dt > 0:
            fps = (0.9 * fps + 0.1 / dt) if fps else 1.0 / dt

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        frame_ts += 33
        result = landmarker.detect_for_video(mp_image, frame_ts)

        predicted_label = None
        confidence = 0.0
        hands_present = bool(result.hand_landmarks)

        if hands_present:
            draw_hand_skeleton(frame, result)
            ax = (frame.shape[1] / frame.shape[0]) / REF_ASPECT
            features = build_two_hand_row(
                result, calibrated_span if USE_CALIBRATION else None, ax)
            probs = model.predict_proba([features])[0]
            best_idx = probs.argmax()
            predicted_label = model.classes_[best_idx]
            confidence = float(probs[best_idx])
            last_features = features
            last_predicted_label = predicted_label

        # ---- smoothing + commit (same logic as before) ----
        if predicted_label is not None and confidence >= CONFIDENCE_THRESHOLD:
            recent_predictions.append(predicted_label)
        else:
            recent_predictions.append(None)

        none_count = sum(1 for p in recent_predictions if p is None)
        if none_count >= SMOOTHING_WINDOW - 2:
            last_committed = None   # hand moved away -> ready for a new letter

        if len(recent_predictions) == SMOOTHING_WINDOW and ui["mode"] is None:
            top_label, top_count = Counter(recent_predictions).most_common(1)[0]
            if top_label is not None and top_count >= SMOOTHING_WINDOW * 0.7:
                if top_label != last_committed:
                    sentence += str(top_label)
                    glosses.append(str(top_label))
                    last_committed = top_label

        show()
        key = cv2.waitKey(1) & 0xFF
        if key == 255:
            continue

        # ---- correction popup captures all keys ----
        if ui["mode"] == "correct":
            if key in (13, 10):
                label = ui["input"].strip().upper()
                feats, pred = ui["snap"]
                if label:
                    db_conn.execute(
                        "INSERT INTO corrections (predicted_label, correct_label, features, timestamp) "
                        "VALUES (?, ?, ?, ?)",
                        (str(pred), label, json.dumps(feats), time.time()),
                    )
                    db_conn.commit()
                    set_status(f"Saved correction: {pred} -> {label}")
                    print(f"Logged correction: predicted={pred} -> correct={label}")
                ui["mode"] = None
            elif key == 27:
                ui["mode"] = None
                set_status("Cancelled")
            elif key in (8, 127):
                ui["input"] = ui["input"][:-1]
            elif 32 < key < 127 and chr(key).isalnum() and len(ui["input"]) < 3:
                ui["input"] += chr(key).upper()
            continue

        ch = chr(key).lower() if 32 <= key < 127 else ""

        if ch == "q":
            break
        elif ch == "f":
            fullscreen = not fullscreen
            cv2.setWindowProperty(
                WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
        elif ch == " ":
            sentence += " "
        elif ch == "b":
            sentence = sentence[:-1]
        elif ch == "c":
            sentence = ""
            glosses = []
            displayed_translation = ""
            set_status("Cleared")
        elif ch == "s":
            if sentence.strip():
                run_blocking("Speaking...", speak, sentence)
            else:
                set_status("Nothing to speak yet")
        elif ch == "l":
            if sentence.strip():
                run_blocking("AI is polishing your sentence...", polish_and_speak, sentence)
            else:
                set_status("Nothing to speak yet")
        elif ch == "x":
            if last_features is None:
                set_status("Show a sign first, then press X")
            else:
                ui["mode"] = "correct"
                ui["input"] = ""
                ui["snap"] = (last_features, last_predicted_label)
        elif ch in LANGUAGES:
            current_lang_key = ch
            set_status(f"Language: {LANGUAGES[ch][0]}")

    cap.release()
    cv2.destroyAllWindows()
    if db_conn:
        db_conn.close()


if __name__ == "__main__":
    main()
