"""
ISL Sign Language -> Text -> Voice   (Dashboard UI version)

What's new:
  * Dashboard layout like the mockup: camera panel on the left (hand boxes,
    Left/Right hand confidence), 4-step pipeline bar, and on the right:
    recognized sign + sentence, translation + Speak button, language tiles
    with flags, and a "Recent translations" list
  * BACKSPACE key now works (deletes only the last letter). 'B' also works.
  * Starts fullscreen (F toggles), scales to any window size
  * Hindi / Telugu / Bengali / Odia text drawn correctly (HarfBuzz shaping)

Keys:
  SPACE add space | BACKSPACE (or B) delete last letter | C clear all
  S speak | L AI-polish + speak | X log a correction | 1-5 language
  F fullscreen | Q quit

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

warnings.filterwarnings("ignore", message="X does not have valid feature names")

# ============================================================
# SETTINGS
# ============================================================
MODEL_FILE = "data/sign_model.pkl"
LANDMARKER_MODEL = "hand_landmarker.task"
CALIBRATION_FILE = "data/calibration.json"
DB_FILE = "data/corrections.db"

CONFIDENCE_THRESHOLD = 0.4
SMOOTHING_WINDOW = 8
SPEAK_COOLDOWN = 2.0

CAM_W, CAM_H = 1280, 720
WINDOW_NAME = "ISL Sign Language to Voice"
START_FULLSCREEN = True
MAX_RENDER_W = 1600          # UI drawn at most this wide, then scaled up (keeps FPS high)

USE_CALIBRATION = False
REF_ASPECT = 4 / 3

LANGUAGES = {
    "1": ("English", "en", "en-GB"),
    "2": ("Hindi", "hi", "hi-IN"),
    "3": ("Telugu", "te", "te-IN"),
    "4": ("Bengali", "bn", "bn-IN"),
    "5": ("Odia", "or", "or-IN"),
}
NATIVE_NAMES = {"1": "English", "2": "हिन्दी", "3": "తెలుగు", "4": "বাংলা", "5": "ଓଡ଼ିଆ"}

VOICE_ID = "21m00Tcm4TlvDq8ikWAM"
ELEVEN_MODEL = "eleven_v3"

COMMON_WORDS = {
    "hi": {"HELLO": "नमस्ते", "THANK YOU": "धन्यवाद", "YES": "हाँ", "NO": "नहीं",
           "PLEASE": "कृपया", "SORRY": "माफ़ करें", "HELP": "मदद", "NAME": "नाम"},
    "te": {"HELLO": "నమస్కారం", "THANK YOU": "ధన్యవాదాలు", "YES": "అవును", "NO": "లేదు"},
    "bn": {"HELLO": "নমস্কার", "THANK YOU": "ধন্যবাদ", "YES": "হ্যাঁ", "NO": "না"},
    "or": {"HELLO": "ନମସ୍କାର", "THANK YOU": "ଧନ୍ୟବାଦ", "YES": "ହଁ", "NO": "ନାଁ"},
}

# ============================================================
# GLOBAL STATE
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
history = []                 # [(english_text, translated_text, timestamp), ...]


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
            model="claude-sonnet-5-5",
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
        history.append((text, translated, time.time()))
        del history[:-20]
        print(f"Speaking ({lang_name}): {translated}")

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


def delete_last(sentence, glosses):
    """Backspace: remove the last character, keep glosses in sync."""
    if not sentence:
        return sentence, glosses
    removed = sentence[-1]
    sentence = sentence[:-1]
    if removed != " " and glosses:
        trimmed = glosses[-1][:-1]
        if trimmed:
            glosses[-1] = trimmed
        else:
            glosses.pop()
    return sentence, glosses


# ============================================================
# LANDMARKS
# ============================================================
def landmarks_to_list(hand_landmarks, scale_override=None, ax=1.0):
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
TEAL_BGR = (191, 212, 45)


def draw_hands(frame, result):
    """Skeleton + bounding box + 'Right Hand 96%' label. Returns [(name, score), ...]."""
    h, w = frame.shape[:2]
    info = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, hl in enumerate(result.hand_landmarks):
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hl]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], TEAL_BGR, 3, cv2.LINE_AA)
        for k, p in enumerate(pts):
            if k in FINGERTIPS:
                cv2.circle(frame, p, 8, (36, 191, 251), -1, cv2.LINE_AA)
            else:
                cv2.circle(frame, p, 5, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, p, 5, TEAL_BGR, 1, cv2.LINE_AA)

        name, score = "Hand", 0.0
        try:
            cat = result.handedness[i][0]
            name, score = cat.category_name, float(cat.score)
        except Exception:
            pass
        info.append((name, score))

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x0, x1 = max(min(xs) - 16, 2), min(max(xs) + 16, w - 2)
        label = f"{name} Hand {int(score * 100)}%"
        (tw, th), _ = cv2.getTextSize(label, font, 0.7, 2)
        y0 = max(min(ys) - 16, th + 18)
        y1 = min(max(ys) + 16, h - 2)
        cv2.rectangle(frame, (x0, y0), (x1, y1), TEAL_BGR, 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x0, y0 - th - 14), (x0 + tw + 16, y0), TEAL_BGR, -1)
        cv2.putText(frame, label, (x0 + 8, y0 - 8), font, 0.7, (42, 23, 15), 2, cv2.LINE_AA)
    return info


# ============================================================
# FONTS
# ============================================================
THEME = {
    "panel": (15, 23, 42),
    "panel_alpha": 215,
    "accent": (45, 212, 191),
    "accent_dark": (13, 148, 136),
    "amber": (251, 191, 36),
    "text": (241, 245, 249),
    "muted": (148, 163, 184),
    "good": (74, 222, 128),
    "warn": (251, 146, 60),
    "track": (51, 65, 85),
    "edge": (51, 65, 85),
    "blue": (37, 99, 235),
    "blue_dark": (29, 78, 216),
    "chip": (30, 41, 59),
}
BG_RGB = (7, 12, 26)


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
    """Draws Hindi/Telugu/Bengali/Odia correctly (HarfBuzz shaping + FreeType)."""

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
    if isinstance(font, ShapedFont):
        font.draw(img, xy, text, fill, anchor)
    else:
        ImageDraw.Draw(img).text(xy, text, font=font, fill=fill, anchor=anchor)


def fit_tail(text, font, max_w):
    """If text is too wide, keep the END of it (latest part) with '...' in front."""
    if text_w(font, text) <= max_w:
        return text
    if isinstance(font, ShapedFont):
        words = text.split(" ")
        while len(words) > 1 and text_w(font, "... " + " ".join(words)) > max_w:
            words = words[1:]
        return "... " + " ".join(words)
    while text and text_w(font, "..." + text) > max_w:
        text = text[1:]
    return "..." + text


def ago(t):
    s = int(time.time() - t)
    if s < 10:
        return "Just now"
    if s < 60:
        return f"{s}s ago"
    if s < 3600:
        return f"{s // 60} min ago"
    return f"{s // 3600} h ago"


# ============================================================
# UI GEOMETRY + HELPERS
# ============================================================
class _Geo:
    """All layout numbers for a given window size (designed at 720p, scaled by height)."""
    def __init__(self, win_w, win_h):
        s = win_h / 720.0
        S = self.S = lambda v: int(round(v * s))
        self.m = S(20)
        self.gap = gap = S(12)
        self.top_h = S(60)
        self.ctrl_h = S(58)
        self.y0 = self.top_h + gap
        self.y1 = win_h - self.ctrl_h - gap

        self.rw = max(S(400), int(win_w * 0.32))
        self.rx0 = win_w - self.m - self.rw
        self.rx1 = win_w - self.m
        self.lx0 = self.m
        self.lx1 = self.rx0 - gap

        pipe_h = S(104)
        self.cam = (self.lx0, self.y0, self.lx1, self.y1 - pipe_h - gap)
        self.pipe = (self.lx0, self.y1 - pipe_h, self.lx1, self.y1)

        y = self.y0
        self.sign = (self.rx0, y, self.rx1, y + S(160)); y = self.sign[3] + gap
        self.trans = (self.rx0, y, self.rx1, y + S(100)); y = self.trans[3] + gap
        self.lang = (self.rx0, y, self.rx1, y + S(116)); y = self.lang[3] + gap
        self.recent = (self.rx0, y, self.rx1, self.y1)

        self.rows_y0 = self.recent[1] + S(40)
        self.row_h = max(S(18), int((self.recent[3] - S(8) - self.rows_y0) / 5))

        self.speak_btn = (self.trans[2] - S(18) - S(128), self.trans[1] + S(40),
                          self.trans[2] - S(18), self.trans[1] + S(40) + S(46))

    def step_cx(self, i):
        return int(self.pipe[0] + (i + 0.5) * (self.pipe[2] - self.pipe[0]) / 4)

    @property
    def step_cy(self):
        return self.pipe[1] + self.S(36)


_mask_cache = {}
_base_cache = {}
_layer_cache = {}
HINT_TEXT = "Show your hand(s) to the camera"


def rounded_mask(w, h, r):
    key = (w, h, r)
    m = _mask_cache.get(key)
    if m is None:
        m = Image.new("L", (w, h), 0)
        ImageDraw.Draw(m).rounded_rectangle([0, 0, w - 1, h - 1], radius=r, fill=255)
        _mask_cache[key] = m
    return m


def glass(img, box, radius, alpha=205):
    """Semi-transparent dark rounded box on an RGBA image."""
    x0, y0, x1, y1 = [int(v) for v in box]
    w, h = x1 - x0, y1 - y0
    if w < 2 or h < 2:
        return
    region = img.crop((x0, y0, x1, y1))
    region = Image.alpha_composite(region, Image.new("RGBA", (w, h), THEME["panel"] + (alpha,)))
    img.paste(region, (x0, y0), rounded_mask(w, h, radius))


def cover_crop(frame, w, h):
    fh, fw = frame.shape[:2]
    scale = max(w / fw, h / fh)
    nw, nh = max(w, int(round(fw * scale))), max(h, int(round(fh * scale)))
    r = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x0, y0 = (nw - w) // 2, (nh - h) // 2
    return r[y0:y0 + h, x0:x0 + w]


def draw_flag(layer, cx, cy, r, kind):
    d = 2 * r
    fl = Image.new("RGBA", (d, d), (0, 0, 0, 0))
    fd = ImageDraw.Draw(fl)
    if kind == "uk":
        white, red = (255, 255, 255), (220, 38, 38)
        fd.rectangle([0, 0, d, d], fill=(30, 58, 138))
        fd.line([0, 0, d, d], fill=white, width=max(2, r // 4))
        fd.line([0, d, d, 0], fill=white, width=max(2, r // 4))
        fd.line([0, 0, d, d], fill=red, width=max(1, r // 8))
        fd.line([0, d, d, 0], fill=red, width=max(1, r // 8))
        fd.rectangle([r - r // 4, 0, r + r // 4, d], fill=white)
        fd.rectangle([0, r - r // 4, d, r + r // 4], fill=white)
        fd.rectangle([r - r // 8, 0, r + r // 8, d], fill=red)
        fd.rectangle([0, r - r // 8, d, r + r // 8], fill=red)
    else:
        third = d / 3
        fd.rectangle([0, 0, d, third], fill=(255, 153, 51))
        fd.rectangle([0, third, d, 2 * third], fill=(255, 255, 255))
        fd.rectangle([0, 2 * third, d, d], fill=(19, 136, 8))
        fd.ellipse([r - r // 4, r - r // 4, r + r // 4, r + r // 4],
                   outline=(0, 0, 128), width=max(1, r // 10))
    mask = Image.new("L", (d, d), 0)
    ImageDraw.Draw(mask).ellipse([0, 0, d - 1, d - 1], fill=255)
    layer.paste(fl, (cx - r, cy - r), mask)


# ============================================================
# STATIC LAYER (cached): panels, labels, tiles, key hints
# ============================================================
def build_static_layer(win_w, win_h, lang_key, show_hint, toast):
    g = _Geo(win_w, win_h)
    S = g.S
    P, PA = THEME["panel"], THEME["panel_alpha"]
    T = THEME
    layer = Image.new("RGBA", (win_w, win_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)

    f_title = get_font("bold", S(24))
    f_title2 = get_font("regular", S(24))
    f_label = get_font("bold", S(13))
    f_badge = get_font("bold", S(14))
    f_tile = get_font("indic", S(14))
    f_tile_s = get_font("regular", S(11))
    f_num = get_font("bold", S(10))
    f_key = get_font("bold", S(14))
    f_chip = get_font("regular", S(15))
    f_toast = get_font("bold", S(20))
    f_hint = get_font("bold", S(22))
    f_step = get_font("bold", S(13))
    f_btn = get_font("bold", S(18))

    def card(box):
        d.rounded_rectangle(list(box), radius=S(18), fill=P + (PA,), outline=T["edge"], width=1)

    # ---- top bar
    d.rectangle([0, 0, win_w, g.top_h], fill=(11, 18, 36, 255))
    d.line([0, g.top_h, win_w, g.top_h], fill=T["edge"], width=1)
    cy = g.top_h // 2
    d.rounded_rectangle([g.m, cy - S(16), g.m + S(32), cy + S(16)], radius=S(9), fill=T["accent_dark"])
    d.text((g.m + S(16), cy), "ISL", font=f_num, fill=(255, 255, 255), anchor="mm")
    tx = g.m + S(46)
    d.text((tx, cy), "ISL", font=f_title, fill=T["text"], anchor="lm")
    tx += int(text_w(f_title, "ISL ")) + S(4)
    d.text((tx, cy), "Sign Language to Voice", font=f_title2, fill=T["text"], anchor="lm")

    # ---- camera panel frame + badges
    cx0, cy0, cx1, cy1 = g.cam
    d.rounded_rectangle([cx0, cy0, cx1, cy1], radius=S(22), outline=T["accent_dark"], width=2)
    bw = int(text_w(f_badge, "Camera Active")) + S(46)
    d.rounded_rectangle([cx0 + S(14), cy0 + S(14), cx0 + S(14) + bw, cy0 + S(46)],
                        radius=S(16), fill=P + (215,))
    d.ellipse([cx0 + S(26), cy0 + S(24), cx0 + S(38), cy0 + S(36)], fill=T["good"])
    d.text((cx0 + S(46), cy0 + S(30)), "Camera Active", font=f_badge, fill=T["text"], anchor="lm")

    if show_hint:
        tw = text_w(f_hint, HINT_TEXT)
        hx0 = int((cx0 + cx1) / 2 - tw / 2 - S(24))
        hx1 = int(hx0 + tw + S(48))
        hy1 = cy1 - S(18)
        hy0 = hy1 - S(48)
        d.rounded_rectangle([hx0, hy0, hx1, hy1], radius=S(24), fill=P + (225,))
        d.text(((hx0 + hx1) // 2, (hy0 + hy1) // 2), HINT_TEXT, font=f_hint,
               fill=T["text"], anchor="mm")

    if toast:
        tw2 = text_w(f_toast, toast)
        tx0 = int((cx0 + cx1) / 2 - tw2 / 2 - S(22))
        tx1 = int(tx0 + tw2 + S(44))
        ty0 = cy0 + S(14)
        ty1 = ty0 + S(44)
        d.rounded_rectangle([tx0, ty0, tx1, ty1], radius=S(22), fill=T["accent_dark"] + (245,))
        d.text(((tx0 + tx1) // 2, (ty0 + ty1) // 2), toast, font=f_toast,
               fill=(255, 255, 255), anchor="mm")

    # ---- pipeline bar
    card(g.pipe)
    lang_name = LANGUAGES[lang_key][0]
    steps = [("1. Detecting", "Hands"), ("2. Recognizing", "Sign"),
             ("3. Translating", f"({lang_name})"), ("4. Speaking", "")]
    for i, (l1, l2) in enumerate(steps):
        cx = g.step_cx(i)
        r = S(24)
        d.ellipse([cx - r, g.step_cy - r, cx + r, g.step_cy + r], outline=T["track"], width=2)
        d.text((cx, g.step_cy), str(i + 1), font=f_btn, fill=T["muted"], anchor="mm")
        d.text((cx, g.pipe[1] + S(74)), l1, font=f_step, fill=T["text"], anchor="mm")
        if l2:
            d.text((cx, g.pipe[1] + S(91)), l2, font=f_tile_s, fill=T["muted"], anchor="mm")
        if i < 3:
            ax = int((g.step_cx(i) + g.step_cx(i + 1)) / 2)
            d.line([ax - S(16), g.step_cy, ax + S(16), g.step_cy], fill=T["muted"], width=2)
            d.line([ax + S(8), g.step_cy - S(7), ax + S(16), g.step_cy], fill=T["muted"], width=2)
            d.line([ax + S(8), g.step_cy + S(7), ax + S(16), g.step_cy], fill=T["muted"], width=2)

    # ---- recognized sign card
    x0, y0, x1, y1 = g.sign
    card(g.sign)
    ix = x0 + S(22)
    d.ellipse([ix, y0 + S(14), ix + S(12), y0 + S(26)], fill=T["accent"])
    d.text((ix + S(20), y0 + S(20)), "RECOGNIZED SIGN", font=f_label, fill=T["accent"], anchor="lm")
    bx0, bx1, by = x0 + S(120), x1 - S(22), y0 + S(64)
    d.rounded_rectangle([bx0, by - S(5), bx1, by + S(5)], radius=S(5), fill=T["track"])
    tick = bx0 + int((bx1 - bx0) * CONFIDENCE_THRESHOLD)
    d.rectangle([tick - 1, by - S(10), tick + 1, by + S(10)], fill=T["text"])
    d.line([ix, y0 + S(100), x1 - S(22), y0 + S(100)], fill=T["edge"], width=1)
    d.text((ix, y0 + S(113)), "SENTENCE", font=f_label, fill=T["muted"], anchor="lm")

    # ---- translation card
    x0, y0, x1, y1 = g.trans
    card(g.trans)
    d.text((ix, y0 + S(20)), f"{lang_name.upper()} TRANSLATION", font=f_label,
           fill=T["accent"], anchor="lm")
    sb = g.speak_btn
    d.rounded_rectangle(list(sb), radius=(sb[3] - sb[1]) // 2, fill=T["blue"])
    scy = (sb[1] + sb[3]) // 2
    d.text((sb[0] + S(24), scy), "Speak", font=f_btn, fill=(255, 255, 255), anchor="lm")
    d.ellipse([sb[2] - S(36), scy - S(12), sb[2] - S(12), scy + S(12)], fill=T["blue_dark"])
    d.text((sb[2] - S(24), scy), "S", font=f_num, fill=(255, 255, 255), anchor="mm")

    # ---- language tiles
    x0, y0, x1, y1 = g.lang
    card(g.lang)
    d.text((ix, y0 + S(20)), "OUTPUT LANGUAGE", font=f_label, fill=T["accent"], anchor="lm")
    area_x0, area_x1 = x0 + S(14), x1 - S(14)
    tgap = S(8)
    tw_ = int((area_x1 - area_x0 - 4 * tgap) / 5)
    ty0, ty1 = y0 + S(38), y1 - S(8)
    for i, (k, (name, _c, _m)) in enumerate(LANGUAGES.items()):
        tx = area_x0 + i * (tw_ + tgap)
        active = (k == lang_key)
        d.rounded_rectangle([tx, ty0, tx + tw_, ty1], radius=S(12),
                            fill=(17, 60, 66, 255) if active else T["chip"] + (255,),
                            outline=T["accent"] if active else T["edge"],
                            width=2 if active else 1)
        mid = tx + tw_ // 2
        draw_flag(layer, mid, ty0 + S(22), S(11), "uk" if k == "1" else "in")
        d.ellipse([tx + S(4), ty0 + S(4), tx + S(18), ty0 + S(18)],
                  fill=T["accent"] if active else T["track"])
        d.text((tx + S(11), ty0 + S(11)), k, font=f_num,
               fill=P if active else T["text"], anchor="mm")
        label = NATIVE_NAMES[k] if INDIC_OK else name
        put_text(layer, (mid, ty0 + S(46)), label, f_tile, T["text"], "mm")
        if k != "1" and INDIC_OK:
            d.text((mid, ty0 + S(61)), name, font=f_tile_s,
                   fill=T["accent"] if active else T["muted"], anchor="mm")

    # ---- recent translations (row backgrounds)
    x0, y0, x1, y1 = g.recent
    card(g.recent)
    d.ellipse([ix, y0 + S(14), ix + S(12), y0 + S(26)], fill=T["accent"])
    d.text((ix + S(20), y0 + S(20)), "RECENT TRANSLATIONS", font=f_label, fill=T["accent"], anchor="lm")
    for i in range(5):
        ry = g.rows_y0 + i * g.row_h
        d.rounded_rectangle([x0 + S(14), ry, x1 - S(14), ry + g.row_h - S(4)],
                            radius=S(8), fill=T["chip"] + (255,))

    # ---- bottom key hints
    d.rectangle([0, win_h - g.ctrl_h, win_w, win_h], fill=(11, 18, 36, 255))
    d.line([0, win_h - g.ctrl_h, win_w, win_h - g.ctrl_h], fill=T["edge"], width=1)
    chips = [("SPACE", "Space"), ("BKSP", "Back"), ("C", "Clear"), ("S", "Speak"),
             ("L", "AI Polish"), ("X", "Correct"), ("F", "Fullscreen"), ("Q", "Quit")]
    x = g.m
    cyc = win_h - g.ctrl_h // 2
    for key, label in chips:
        kw = int(max(S(30), text_w(f_key, key) + S(18)))
        hot = key == "SPACE"
        d.rounded_rectangle([x, cyc - S(15), x + kw, cyc + S(15)], radius=S(8),
                            fill=T["blue"] if hot else T["track"])
        d.text((x + kw // 2, cyc), key, font=f_key, fill=(255, 255, 255), anchor="mm")
        x += kw + S(8)
        d.text((x, cyc), label, font=f_chip, fill=T["muted"], anchor="lm")
        x += int(text_w(f_chip, label)) + S(24)
    return layer


# ============================================================
# DYNAMIC DRAWING (every frame)
# ============================================================
def draw_dynamic(img, st, win_w, win_h):
    d = ImageDraw.Draw(img)
    g = _Geo(win_w, win_h)
    S = g.S
    T = THEME

    f_small = get_font("regular", S(14))
    f_live = get_font("bold", S(14))
    f_label = get_font("bold", S(13))
    f_big = get_font("bold", S(56))
    f_mid = get_font("bold", S(28))
    f_sent = get_font("bold", S(30))
    f_trans = get_font("indic", S(32))
    f_trans_s = get_font("indic", S(20))
    f_row = get_font("regular", S(14))
    f_row_i = get_font("indic", S(14))
    f_tiny = get_font("regular", S(12))
    f_step = get_font("bold", S(18))

    # ---- top bar (right)
    cy = g.top_h // 2
    right = win_w - g.m
    fps_txt = f"{st['fps']:.0f} FPS"
    d.text((right, cy), fps_txt, font=f_small, fill=T["muted"], anchor="rm")
    right -= int(text_w(f_small, fps_txt)) + S(16)
    d.line([right, cy - S(11), right, cy + S(11)], fill=T["edge"], width=1)
    right -= S(16)
    d.text((right, cy), "LIVE", font=f_live, fill=T["good"], anchor="rm")
    right -= int(text_w(f_live, "LIVE")) + S(10)
    d.ellipse([right - S(12), cy - S(6), right, cy + S(6)], fill=T["good"])

    # ---- hands-detected box (camera, top right)
    cx0, cy0, cx1, cy1 = g.cam
    hi = sorted(st["hands_info"])
    if hi:
        n = len(hi)
        bw = S(200)
        title_h, row_h = S(36), S(26)
        bh = title_h + (n + 1) * row_h + S(6)
        bx1, by0 = cx1 - S(14), cy0 + S(14)
        bx0 = bx1 - bw
        glass(img, (bx0, by0, bx1, by0 + bh), S(14))
        d.text((bx0 + S(14), by0 + title_h // 2), f"{n} Hand{'s' if n > 1 else ''} Detected",
               font=f_live, fill=T["text"], anchor="lm")
        y = by0 + title_h + row_h // 2
        for name, score in hi:
            d.text((bx0 + S(14), y), f"{name} Hand:", font=f_small, fill=T["text"], anchor="lm")
            d.text((bx1 - S(14), y), f"{int(score * 100)}%", font=f_live, fill=T["good"], anchor="rm")
            y += row_h
        d.text((bx0 + S(14), y), "Tracking:", font=f_small, fill=T["text"], anchor="lm")
        d.text((bx1 - S(14), y), "Stable", font=f_live, fill=T["good"], anchor="rm")

    # ---- pipeline circles (active steps)
    active = [st["hands"], st["sign"] is not None and st["confidence"] >= CONFIDENCE_THRESHOLD,
              bool(st["translated"]), st["busy"]]
    for i, on in enumerate(active):
        if on:
            cx = g.step_cx(i)
            r = S(24)
            d.ellipse([cx - r, g.step_cy - r, cx + r, g.step_cy + r],
                      fill=T["accent_dark"], outline=T["accent"], width=2)
            d.text((cx, g.step_cy), str(i + 1), font=f_step, fill=(255, 255, 255), anchor="mm")

    # ---- recognized sign card
    x0, y0, x1, y1 = g.sign
    ix = x0 + S(22)
    conf = max(0.0, min(1.0, float(st["confidence"])))
    ok = conf >= CONFIDENCE_THRESHOLD
    d.text((x1 - S(22), y0 + S(20)), f"{int(conf * 100)}%", font=f_live,
           fill=T["good"] if ok else T["warn"], anchor="rm")
    sign = str(st["sign"]) if st["sign"] is not None else "?"
    sf = f_big if text_w(f_big, sign) <= S(92) else f_mid
    d.text((ix, y0 + S(64)), sign, font=sf,
           fill=T["accent"] if st["sign"] is not None else T["track"], anchor="lm")
    bx0, bx1, by = x0 + S(120), x1 - S(22), y0 + S(64)
    fw = int((bx1 - bx0) * conf)
    if fw > S(8):
        d.rounded_rectangle([bx0, by - S(5), bx0 + fw, by + S(5)], radius=S(5),
                            fill=T["good"] if ok else T["warn"])

    max_w = (x1 - x0) - S(44)
    y_s = y0 + S(138)
    sent = fit_tail(st["sentence"], f_sent, max_w - S(14))
    if sent:
        d.text((ix, y_s), sent, font=f_sent, fill=T["text"], anchor="lm")
    else:
        d.text((ix, y_s), "Start signing...", font=f_sent, fill=T["track"], anchor="lm")
    if st["blink"] and st["mode"] != "correct":
        cxx = ix + (int(text_w(f_sent, sent)) + S(5) if sent else 0)
        d.rectangle([cxx, y_s - S(15), cxx + S(3), y_s + S(15)], fill=T["accent"])

    # ---- translation text
    x0, y0, x1, y1 = g.trans
    avail = g.speak_btn[0] - ix - S(14)
    trans = fit_tail(st["translated"], f_trans, avail)
    ty = y0 + S(68)
    if trans:
        put_text(img, (ix, ty), trans, f_trans, T["amber"], "lm")
    else:
        put_text(img, (ix, ty), "Press S to speak", f_trans_s, T["track"], "lm")

    # ---- recent rows
    x0, y0, x1, y1 = g.recent
    rx0, rx1 = x0 + S(14), x1 - S(14)
    w = rx1 - rx0
    hist = st["history"][-5:][::-1]
    if not hist:
        ry = g.rows_y0 + (g.row_h - S(4)) // 2
        d.text((rx0 + S(10), ry), "Spoken sentences will appear here", font=f_row,
               fill=T["muted"], anchor="lm")
    for i, (en, tr, t) in enumerate(hist):
        ry = g.rows_y0 + i * g.row_h + (g.row_h - S(4)) // 2
        d.text((rx0 + S(10), ry), fit_tail(en, f_row, int(w * 0.28)), font=f_row,
               fill=T["text"], anchor="lm")
        d.text((rx0 + int(w * 0.33), ry), "->", font=f_row, fill=T["muted"], anchor="lm")
        put_text(img, (rx0 + int(w * 0.42), ry), fit_tail(tr, f_row_i, int(w * 0.33)),
                 f_row_i, T["accent"], "lm")
        d.text((rx1 - S(10), ry), ago(t), font=f_tiny, fill=T["muted"], anchor="rm")


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
    g = _Geo(win_w, win_h)
    cx0, cy0, cx1, cy1 = g.cam
    cw, ch = cx1 - cx0, cy1 - cy0

    base = _base_cache.get((win_w, win_h))
    if base is None:
        base = Image.new("RGB", (win_w, win_h), BG_RGB)
        _base_cache.clear()
        _base_cache[(win_w, win_h)] = base
    img = base.copy()

    cam = Image.fromarray(cv2.cvtColor(cover_crop(cam_frame, cw, ch), cv2.COLOR_BGR2RGB))
    img.paste(cam, (cx0, cy0), rounded_mask(cw, ch, g.S(22)))
    img = img.convert("RGBA")

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

    ui = {"status": "", "until": 0.0, "mode": None, "input": "", "snap": None, "busy": False}

    frame = None
    predicted_label = None
    confidence = 0.0
    hands_present = False
    hands_info = []

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
            "hands_info": hands_info,
            "history": history,
            "busy": ui["busy"],
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
        ui["busy"] = True
        show()
        cv2.waitKey(1)
        try:
            fn(*args)
        finally:
            ui["until"] = 0.0
            ui["busy"] = False

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
        hands_info = []

        if hands_present:
            hands_info = draw_hands(frame, result)
            ax = (frame.shape[1] / frame.shape[0]) / REF_ASPECT
            features = build_two_hand_row(
                result, calibrated_span if USE_CALIBRATION else None, ax)
            probs = model.predict_proba([features])[0]
            best_idx = probs.argmax()
            predicted_label = model.classes_[best_idx]
            confidence = float(probs[best_idx])
            last_features = features
            last_predicted_label = predicted_label

        # ---- smoothing + commit ----
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

        # ---- BACKSPACE (key code 8) / DEL (127): delete last letter ----
        if key in (8, 127):
            sentence, glosses = delete_last(sentence, glosses)
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
            sentence, glosses = delete_last(sentence, glosses)
        elif ch == "c":
            sentence = ""
            glosses = []
            displayed_translation = ""
            history.clear()
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


























