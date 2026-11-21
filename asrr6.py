"""
SHENAVA AI | Persian Speech Intelligence
=========================================


قابلیت‌ها
---------
* حالت زنده : میکروفون -> فیلتر بالاگذر -> Silero VAD -> Shenava (NeMo CTC)
              + شناسایی آنلاین گوینده و بازتفکیک دقیق پس از توقف
* حالت فایل : فایل صوتی/تصویری -> (تفکیک گوینده) -> Whisper یا Shenava
* تحلیل متن : ارسال متن نهایی به Qwen از طریق Ollama (کاملاً محلی)
* ترجمه     : TranslateGemma (از طریق Ollama، دقیق‌ترین مدل ترجمه‌ی باز)
              و در صورت نبود آن، NLLB-200 3.3B به‌عنوان جایگزین


وابستگی‌ها
----------
numpy, sounddevice, sherpa-onnx, requests
اختیاری : scipy (resample و فیلتر)، faster-whisper، pyannote.audio، torch،
          transformers + sentencepiece (فقط برای جایگزین NLLB)

نصب مدل ترجمه (یک‌بار):
    ollama pull translategemma:4b       # پیش‌فرض سبک: ~3.3GB
    ollama pull translategemma:12b      # دقیق‌تر: ~8GB
    ollama pull translategemma:27b      # دقیق‌ترین؛ حدود ۱۷ گیگ VRAM
"""

import os
import re
import json
import time
import queue
import tarfile
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
import sounddevice as sd
import sherpa_onnx
import requests

# scipy اختیاری است؛ در نبود آن resample با درون‌یابی خطی انجام می‌شود
# و فیلتر بالاگذر و خوشه‌بندی سلسله‌مراتبی غیرفعال می‌مانند.
try:
    from scipy.signal import resample_poly
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


# ============================================================
# مسیر مدل‌ها (Shenava / VAD / تشخیص گوینده)
# ============================================================

SHENAVA_DIR = os.path.abspath(
    os.environ.get("SHENAVA_DIR", os.path.join(os.path.dirname(__file__), "Shenava"))
)

SHENAVA_MODEL = os.path.join(SHENAVA_DIR, "model.onnx")
SHENAVA_TOKENS = os.path.join(SHENAVA_DIR, "tokens.txt")
VAD_MODEL = os.path.join(SHENAVA_DIR, "silero_vad.onnx")

SPEAKER_MODELS_DIR = os.path.join(SHENAVA_DIR, "speaker_models")
SPEAKER_SEG_DEFAULT = os.path.join(SPEAKER_MODELS_DIR, "pyannote_segmentation.onnx")
SPEAKER_EMBED_DEFAULT = os.path.join(SPEAKER_MODELS_DIR, "speaker_embedding.onnx")

SPEAKER_SEG_MODEL = SPEAKER_SEG_DEFAULT
SPEAKER_EMBED_MODEL = SPEAKER_EMBED_DEFAULT

SEG_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
)
EMBED_BASE_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/"
)
EMBED_MODEL_CHOICES = [
    "wespeaker_en_voxceleb_resnet293_LM.onnx",
    "wespeaker_en_voxceleb_resnet221_LM.onnx",
    "3dspeaker_speech_eres2net_large_sv_zh-cn_3dspeaker_16k.onnx",
    "wespeaker_en_voxceleb_CAM++_LM.onnx",
]

_EMBED_KEYWORDS = (
    "speaker_embedding", "wespeaker", "3dspeaker", "eres2net", "campplus",
    "titanet", "ecapa", "resnet", "voxceleb", "cnceleb", "speaker", "embed"
)
_EMBED_QUALITY_ORDER = (
    "resnet293", "resnet221", "resnet152", "eres2net_large",
    "cam++", "campplus", "titanet", "ecapa"
)


def _find_onnx_files(root, max_depth=3):
    """همه‌ی فایل‌های ``.onnx`` زیر ``root`` را (تا عمق ``max_depth``) برمی‌گرداند."""
    found = []
    if not os.path.isdir(root):
        return found

    root = os.path.normpath(root)
    base_depth = root.count(os.sep)

    for d, dirs, files in os.walk(root):
        if d.count(os.sep) - base_depth >= max_depth:
            dirs[:] = []
        for f in files:
            if f.lower().endswith(".onnx"):
                found.append(os.path.join(d, f))
    return found


def _embed_rank(path):
    """رتبه‌ی کیفیت مدل embedding؛ عدد کمتر = مدل بهتر (۹۹ یعنی ناشناخته)."""
    low = os.path.basename(path).lower()
    for i, k in enumerate(_EMBED_QUALITY_ORDER):
        if k in low:
            return i
    return 99


def resolve_speaker_models():
    """
    مسیر مدل‌های تشخیص گوینده را نهایی می‌کند. اگر فایل‌های پیش‌فرض نباشند،
    SHENAVA_DIR جست‌وجو و بهترین مدل segmentation/embedding انتخاب می‌شود.
    """
    global SPEAKER_SEG_MODEL, SPEAKER_EMBED_MODEL

    if os.path.isfile(SPEAKER_SEG_DEFAULT):
        SPEAKER_SEG_MODEL = SPEAKER_SEG_DEFAULT
    if os.path.isfile(SPEAKER_EMBED_DEFAULT):
        SPEAKER_EMBED_MODEL = SPEAKER_EMBED_DEFAULT

    skip = {
        os.path.normpath(SHENAVA_MODEL).lower(),
        os.path.normpath(VAD_MODEL).lower(),
    }

    candidates = [
        p for p in _find_onnx_files(SHENAVA_DIR)
        if os.path.normpath(p).lower() not in skip
        and "silero" not in os.path.basename(p).lower()
    ]
    candidates.sort(
        key=lambda p: ("int8" in p.lower(), _embed_rank(p), len(p))
    )

    if not os.path.isfile(SPEAKER_SEG_MODEL):
        for p in candidates:
            low = p.lower()
            if "segmentation" in low or "pyannote" in low:
                SPEAKER_SEG_MODEL = p
                break

    if not os.path.isfile(SPEAKER_EMBED_MODEL):
        for p in candidates:
            low = p.lower()
            if "segmentation" in low or "pyannote" in low:
                continue
            if any(k in os.path.basename(low) for k in _EMBED_KEYWORDS):
                SPEAKER_EMBED_MODEL = p
                break


resolve_speaker_models()


# ============================================================
# دانلود مدل‌های تشخیص گوینده
# ============================================================

def _download(url, dest, status_cb=None, label=""):
    """
    فایل را به‌صورت stream دانلود می‌کند (ابتدا در ``.part`` و سپس ``os.replace``
    تا فایل ناقص هرگز جای مدل سالم را نگیرد).
    """
    tmp = dest + ".part"
    last_report = 0.0

    with requests.get(url, stream=True, timeout=(15, 120)) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=256 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)

                now = time.monotonic()
                if status_cb and now - last_report > 0.3:
                    last_report = now
                    if total:
                        status_cb(
                            f"دانلود {label}: {int(100 * done / total)}٪ "
                            f"({done / 1e6:.0f}/{total / 1e6:.0f} MB)"
                        )
                    else:
                        status_cb(f"دانلود {label}: {done / 1e6:.0f} MB")

    os.replace(tmp, dest)


def download_speaker_models(status_cb=None):
    """
    مدل segmentation (pyannote 3.0) و بهترین مدل embedding در دسترس را دانلود می‌کند.

    Raises:
        RuntimeError: اگر هیچ‌یک از مدل‌های embedding دانلود نشود یا آرشیو
            segmentation فایل onnx نداشته باشد.
    """
    os.makedirs(SPEAKER_MODELS_DIR, exist_ok=True)

    if not os.path.isfile(SPEAKER_SEG_DEFAULT):
        archive = os.path.join(SPEAKER_MODELS_DIR, "seg.tar.bz2")
        _download(SEG_MODEL_URL, archive, status_cb, "مدل segmentation")
        try:
            with tarfile.open(archive, "r:bz2") as tar:
                member = None
                for m in tar.getmembers():
                    if m.name.lower().endswith("/model.onnx"):
                        member = m
                        break
                if member is None:
                    for m in tar.getmembers():
                        if m.name.lower().endswith(".onnx") and "int8" not in m.name.lower():
                            member = m
                            break
                if member is None:
                    raise RuntimeError("فایل onnx داخل آرشیو segmentation پیدا نشد.")

                with tar.extractfile(member) as src, open(SPEAKER_SEG_DEFAULT, "wb") as dst:
                    dst.write(src.read())
        finally:
            try:
                os.remove(archive)
            except OSError:
                pass

    if not os.path.isfile(SPEAKER_EMBED_DEFAULT):
        errors = []
        ok = False
        for name in EMBED_MODEL_CHOICES:
            try:
                _download(
                    EMBED_BASE_URL + name, SPEAKER_EMBED_DEFAULT,
                    status_cb, f"مدل گوینده ({name})"
                )
                ok = True
                break
            except Exception as exc:
                errors.append(f"{name}: {exc}")

        if not ok:
            raise RuntimeError(
                "دانلود مدل embedding ناموفق بود:\n" + "\n".join(errors) +
                "\n\nاگر GitHub در شبکه‌ات فیلتر است از VPN استفاده کن یا فایل را "
                f"دستی با نام speaker_embedding.onnx در پوشه زیر بگذار:\n{SPEAKER_MODELS_DIR}"
            )

    resolve_speaker_models()


# ============================================================
# تنظیمات Ollama / Qwen (تحلیل متن)
# ============================================================

OLLAMA_URL = "http://127.0.0.1:11434"
QWEN_MODEL_NAME = "qwen3:8b-q8"

QWEN_MAX_TOKENS = 900
QWEN_TEMPERATURE = 0.2
OLLAMA_TIMEOUT = (10, 600)      # (اتصال، خواندن) به ثانیه


# ============================================================
# تنظیمات Whisper (حالت ضبط‌شده)
# ============================================================

WHISPER_MODELS = ["large-v3", "large-v3-turbo", "medium", "small", "base"]
DEFAULT_WHISPER_MODEL = "large-v3"
WHISPER_LANGUAGE = "fa"


# ============================================================
# تنظیمات ترجمه
# ============================================================
# موتور اصلی: TranslateGemma (مدل اختصاصی ترجمه‌ی گوگل، ۵۵ زبان، روی Gemma 3).
# نسخه‌ی 12B از Gemma 3 27B عادی هم دقیق‌تر است و 27B دقیق‌ترین است.
# موتور جایگزین: NLLB-200 3.3B (بزرگ‌ترین نسخه‌ی NLLB) در صورت نبود مدل Ollama.

# به ترتیب اولویت؛ اولین مدل نصب‌شده انتخاب می‌شود.
TRANSLATION_OLLAMA_PREFERRED = [
    "translategemma:4b",     # سبک (~3.3GB)؛ با «ترجمه‌ی دقیق‌تر» (پایین) دقتش بالا می‌رود
    "translategemma:12b",
    "translategemma:27b",
]
# اگر هیچ TranslateGemma نصب نبود، از این مدل‌های عمومی (به ترتیب) استفاده می‌شود.
TRANSLATION_OLLAMA_GENERAL = [
    "qwen3:32b", "qwen3:30b", "qwen3:14b", "gemma3:27b", "gemma3:12b",
]
# مدلی که دکمه‌ی «دانلود مدل ترجمه» دریافت می‌کند (برای ۱۶ گیگ VRAM مناسب‌تر است؛
# اگر GPU قوی داری 27b را بگذار).
TRANSLATION_PULL_MODEL = "translategemma:4b"

# حالت «ترجمه‌ی دقیق‌تر» (برای مدل سبک 4b): هر ترجمه با ترجمه‌ی برگشتی به فارسی
# سنجیده می‌شود؛ اگر با متن اصلی نخواند، مسیر دوم (از طریق انگلیسی) امتحان و
# بهترین نتیجه انتخاب می‌شود. کمی کندتر است ولی خطاهای معنایی را کم می‌کند.
TRANSLATION_HIGH_ACCURACY = True
TRANSLATION_RT_ACCEPT = 0.75     # شباهت برگشتی که بدون امتحان مسیر دوم پذیرفته می‌شود
TRANSLATION_RT_MARGIN = 0.03     # مسیر دوم فقط وقتی برنده است که دست‌کم این‌قدر بهتر باشد

TRANSLATION_OLLAMA_TIMEOUT = (10, 180)
TRANSLATION_NUM_PREDICT = 700
TRANSLATION_NUM_CTX = 4096
TRANSLATION_KEEP_ALIVE = "30m"

# جایگزین NLLB (برای صرفه‌جویی در حافظه می‌توان "facebook/nllb-200-distilled-1.3B" گذاشت)
NLLB_MODEL_NAME = "facebook/nllb-200-distilled-1.3B"
NLLB_SOURCE_LANG = "pes_Arab"
NLLB_MAX_NEW_TOKENS = 400
NLLB_MIN_FREE_VRAM_GB = 3.5

TRANSLATION_SOURCE_NAME = "Persian"
TRANSLATION_SOURCE_ISO = "fa"

# نام فارسی -> (کد ISO، کد NLLB، نام انگلیسی)
TRANSLATION_LANGUAGES = {
    "انگلیسی": ("en", "eng_Latn", "English"),
    "عربی": ("ar", "arb_Arab", "Arabic"),
    "فرانسوی": ("fr", "fra_Latn", "French"),
    "آلمانی": ("de", "deu_Latn", "German"),
    "اسپانیایی": ("es", "spa_Latn", "Spanish"),
    "روسی": ("ru", "rus_Cyrl", "Russian"),
    "ترکی": ("tr", "tur_Latn", "Turkish"),
    "چینی ساده‌شده": ("zh", "zho_Hans", "Chinese (Simplified)"),
    "ژاپنی": ("ja", "jpn_Jpan", "Japanese"),
    "کره‌ای": ("ko", "kor_Hang", "Korean"),
    "ایتالیایی": ("it", "ita_Latn", "Italian"),
    "پرتغالی": ("pt", "por_Latn", "Portuguese"),
    "هندی": ("hi", "hin_Deva", "Hindi"),
    "اردو": ("ur", "urd_Arab", "Urdu"),
    "هلندی": ("nl", "nld_Latn", "Dutch"),
    "سوئدی": ("sv", "swe_Latn", "Swedish"),
    "لهستانی": ("pl", "pol_Latn", "Polish"),
    "اوکراینی": ("uk", "ukr_Cyrl", "Ukrainian"),
    "یونانی": ("el", "ell_Grek", "Greek"),
    "عبری": ("he", "heb_Hebr", "Hebrew"),
    "اندونزیایی": ("id", "ind_Latn", "Indonesian"),
    "ویتنامی": ("vi", "vie_Latn", "Vietnamese"),
    "تایلندی": ("th", "tha_Thai", "Thai"),
    "بنگالی": ("bn", "ben_Beng", "Bengali"),
    "فارسی": ("fa", "pes_Arab", "Persian"),
}


def is_rtl_language(name):
    """آیا زبان مقصد راست‌به‌چپ است؟ (برای جهت‌دهی نمایش ترجمه)"""
    entry = TRANSLATION_LANGUAGES.get(name)
    return bool(entry) and entry[1].endswith(("_Arab", "_Hebr"))


# ============================================================
# تنظیمات صدا
# ============================================================

DEVICE_INDEX = 1                # شماره‌ی دستگاه ورودی در sounddevice (sd.query_devices())

MIC_SAMPLE_RATE = 44100
ASR_SAMPLE_RATE = 16000
CHANNELS = 1

VAD_THRESHOLD = 0.45
MIN_SPEECH_DURATION = 0.25
MIN_SILENCE_DURATION = 0.60
MAX_SPEECH_DURATION = 18.0
VAD_WINDOW = 512
VAD_BUFFER_SECONDS = 60

# صدای اضافه قبل و بعد از هر قطعه تا حروف اول/آخر کلمه بریده نشوند
VAD_PAD_SEC = 0.25

HIGHPASS_HZ = 80

WHISPER_VAD_PARAMS = dict(
    threshold=0.45,
    min_speech_duration_ms=250,
    min_silence_duration_ms=600,
    speech_pad_ms=300,
)


# ============================================================
# تنظیمات تشخیص گوینده
# ============================================================

# --- حالت زنده ---
SPEAKER_THRESHOLD = 0.62
MAX_LIVE_SPEAKERS = 8
SPEAKER_MIN_EMBED_SEC = 0.6
SPEAKER_MIN_NEW_SEC = 0.8
SPEAKER_MIN_UPDATE_SEC = 1.0

# --- حالت فایل ---
DIARIZATION_CLUSTER_THRESHOLD = 0.35

# Pyannote.audio (community-1) دقیق‌ترین روش موجود است و پیش‌فرض می‌شود؛
# دو روش دیگر فقط جایگزین‌اند.
DIAR_METHODS = ["Pyannote.audio", "Embedding + Clustering", "Sherpa Pyannote"]

try:
    import importlib.util as _ilu
    _HAS_PYANNOTE = _ilu.find_spec("pyannote.audio") is not None
except Exception:
    _HAS_PYANNOTE = False

DEFAULT_DIAR_METHOD = DIAR_METHODS[0] if _HAS_PYANNOTE else DIAR_METHODS[1]

# community-1 (pyannote 4) دقیق‌تر از 3.1 است؛ اگر نصب نبود 3.1 امتحان می‌شود.
PYANNOTE_PIPELINES = [
    "pyannote/speaker-diarization-community-1",
    "pyannote/speaker-diarization-3.1",
]
HF_TOKEN_FILE = os.path.join(SHENAVA_DIR, "hf_token.txt")

DIAR_WINDOW_SEC = 1.5
DIAR_HOP_SEC = 0.75
DIAR_WEAK_SEC = 0.9
DIAR_MAX_SPEAKERS = 8
DIAR_MIN_CLUSTER_FRAC = 0.04
DIAR_CENTROID_MERGE_SIM = 0.72
DIAR_MIN_SILHOUETTE = 0.06
DIAR_SMOOTH = True

SPEAKER_COLORS = [
    "#38BDF8", "#34D399", "#FBBF24", "#FB7185",
    "#A78BFA", "#F97316", "#2DD4BF", "#E879F9",
    "#A3E635", "#60A5FA", "#F472B6", "#FACC15",
]


# ============================================================
# وضعیت سراسری برنامه
# ============================================================

audio_queue = queue.Queue(maxsize=100)

running = False
file_processing = False
recognizer = None
vad = None

whisper_model = None
whisper_model_name = None

# جایگزین NLLB (تنبل بارگذاری می‌شود)
nllb_tokenizer = None
nllb_model = None
nllb_device = "cpu"
nllb_lock = threading.Lock()

# مدل Ollama انتخاب‌شده برای ترجمه (کش)
_translation_model_cache = None
_translation_model_checked = False
translation_model_lock = threading.Lock()

qwen = None
qwen_analyzing = False

last_rms = 0.0
last_peak = 0.0

# هر عضو: {"speaker", "text", "start", "end", "translation"?, "translation_language"?, "translation_engine"?}
transcript_segments = []

state_lock = threading.Lock()


# ============================================================
# توابع کمکی
# ============================================================

def fmt_time(sec):
    """ثانیه را به قالب ``mm:ss`` تبدیل می‌کند (None -> رشته‌ی خالی)."""
    if sec is None:
        return ""
    sec = int(max(0, sec))
    return f"{sec // 60:02d}:{sec % 60:02d}"


def speaker_label(n):
    """برچسب فارسی گوینده، مثلاً «گوینده ۱»؛ برای None رشته‌ی خالی."""
    return f"گوینده {n}" if n is not None else ""


def format_transcript_for_llm(with_time=True):
    """متن کامل را خط‌به‌خط (``[mm:ss] گوینده N: متن``) برای مدل زبانی یا ذخیره آماده می‌کند."""
    lines = []
    for seg in transcript_segments:
        parts = []
        if with_time and seg.get("start") is not None:
            parts.append(f"[{fmt_time(seg['start'])}]")
        if seg.get("speaker") is not None:
            parts.append(f"{speaker_label(seg['speaker'])}:")
        parts.append(seg["text"])
        lines.append(" ".join(parts))
        if seg.get("translation"):
            lines.append(
                f"    ترجمه ({seg.get('translation_language', '')}): "
                f"{seg['translation']}"
            )
    return "\n".join(lines)


def has_multiple_speakers():
    """آیا حداقل دو گوینده‌ی متفاوت در متن وجود دارد؟"""
    return len({
        s["speaker"] for s in transcript_segments
        if s.get("speaker") is not None
    }) > 1


# ============================================================
# پردازش سیگنال صوتی
# ============================================================

def resample_audio(audio, source_rate, target_rate):
    """نرخ نمونه‌برداری را تغییر می‌دهد (resample_poly با scipy، وگرنه درون‌یابی خطی)."""
    if audio is None or len(audio) == 0:
        return np.zeros(0, dtype=np.float32)

    audio = np.asarray(audio, dtype=np.float32)

    if source_rate == target_rate:
        return audio

    if HAS_SCIPY:
        from math import gcd
        g = gcd(int(source_rate), int(target_rate))
        up = int(target_rate) // g
        down = int(source_rate) // g
        return resample_poly(audio, up, down).astype(np.float32)

    target_length = int(round(len(audio) * target_rate / source_rate))
    if target_length <= 0:
        return np.zeros(0, dtype=np.float32)

    old_x = np.linspace(0, len(audio) - 1, len(audio))
    new_x = np.linspace(0, len(audio) - 1, target_length)
    return np.interp(new_x, old_x, audio).astype(np.float32)


class HighPass:
    """فیلتر بالاگذر Butterworth مرتبه ۲ برای پردازش پیوسته (وضعیت ``zi`` بین بلوک‌ها حفظ می‌شود)."""

    def __init__(self):
        self.sos = None
        self.zi = None
        if HAS_SCIPY and HIGHPASS_HZ > 0:
            try:
                from scipy.signal import butter
                self.sos = butter(
                    2, HIGHPASS_HZ, btype="highpass",
                    fs=ASR_SAMPLE_RATE, output="sos"
                )
            except Exception:
                self.sos = None

    def __call__(self, x):
        if self.sos is None or len(x) == 0:
            return x
        from scipy.signal import sosfilt, sosfilt_zi

        if self.zi is None:
            self.zi = sosfilt_zi(self.sos) * float(x[0])
        y, self.zi = sosfilt(self.sos, x, zi=self.zi)
        return y.astype(np.float32)


class AudioHistory:
    """
    بافر حلقوی از صدای اخیر (۱۶kHz) برای برداشتن padding اطراف قطعه‌های VAD.
    موقعیت‌ها «شماره‌ی نمونه‌ی مطلق از شروع جلسه» هستند.
    """

    def __init__(self, seconds=45):
        self.max = int(seconds * ASR_SAMPLE_RATE)
        self.buf = np.zeros(0, dtype=np.float32)
        self.offset = 0

    def add(self, x):
        self.buf = np.concatenate((self.buf, x))
        extra = len(self.buf) - self.max
        if extra > 0:
            self.buf = self.buf[extra:]
            self.offset += extra

    def total(self):
        """شماره‌ی مطلقِ انتهای صدای ذخیره‌شده."""
        return self.offset + len(self.buf)

    def cut(self, a, b):
        """بازه‌ی ``[a, b)`` (شماره‌ی مطلق نمونه) یا None در صورت خالی بودن."""
        a = max(int(a), self.offset)
        b = min(int(b), self.total())
        if b <= a:
            return None
        return self.buf[a - self.offset:b - self.offset]


# ============================================================
# Callback میکروفون
# ============================================================

def audio_callback(indata, frames, time_info, status):
    """Callback سبک sounddevice: سطح صدا را به‌روز و بلوک را به صف می‌دهد."""
    global last_rms, last_peak

    try:
        if status:
            print("Audio status:", status)

        if indata is None or len(indata) == 0:
            return

        audio = indata[:, 0].copy().astype(np.float32)

        last_rms = float(np.sqrt(np.mean(audio ** 2) + 1e-12))
        last_peak = float(np.max(np.abs(audio)))

        try:
            audio_queue.put_nowait(audio)
        except queue.Full:
            try:
                audio_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                audio_queue.put_nowait(audio)
            except queue.Full:
                pass

    except Exception as exc:
        print("Audio callback error:", exc)


# ============================================================
# بارگذاری Shenava و VAD
# ============================================================

def make_vad():
    """یک VoiceActivityDetector تازه (Silero) با تنظیمات سراسری می‌سازد."""
    vad_config = sherpa_onnx.VadModelConfig()
    vad_config.silero_vad.model = VAD_MODEL
    vad_config.silero_vad.threshold = VAD_THRESHOLD
    vad_config.silero_vad.min_speech_duration = MIN_SPEECH_DURATION
    vad_config.silero_vad.min_silence_duration = MIN_SILENCE_DURATION
    vad_config.silero_vad.max_speech_duration = MAX_SPEECH_DURATION
    vad_config.silero_vad.window_size = VAD_WINDOW
    vad_config.sample_rate = ASR_SAMPLE_RATE
    vad_config.num_threads = 1

    return sherpa_onnx.VoiceActivityDetector(
        config=vad_config,
        buffer_size_in_seconds=VAD_BUFFER_SECONDS
    )


def load_shenava():
    """
    مدل Shenava (NeMo CTC) و VAD را بارگذاری می‌کند.

    Raises:
        FileNotFoundError: اگر یکی از فایل‌های لازم موجود نباشد.
    """
    global recognizer, vad

    required_files = [
        (SHENAVA_MODEL, "model.onnx"),
        (SHENAVA_TOKENS, "tokens.txt"),
        (VAD_MODEL, "silero_vad.onnx"),
    ]

    for path, filename in required_files:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"فایل {filename} پیدا نشد:\n{path}"
            )

    if recognizer is None:
        recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
            model=SHENAVA_MODEL,
            tokens=SHENAVA_TOKENS,
            num_threads=4
        )

    vad = make_vad()


def offline_vad_segments(audio, pad_sec=VAD_PAD_SEC, merge=True):
    """
    یک فایل کامل را به بخش‌های گفتار تقسیم می‌کند.

    Returns:
        فهرست ``[(start_sec, end_sec), ...]``
    """
    v = make_vad()
    n = len(audio)
    raw = []

    def drain():
        while not v.empty():
            s = v.front
            start = int(s.start)
            length = len(s.samples)
            v.pop()
            raw.append((start, start + length))

    audio = np.asarray(audio, dtype=np.float32)

    for i in range(0, n - VAD_WINDOW + 1, VAD_WINDOW):
        v.accept_waveform(audio[i:i + VAD_WINDOW])
        drain()

    try:
        v.flush()
        drain()
    except Exception:
        pass

    pad = int(pad_sec * ASR_SAMPLE_RATE)
    max_merge = int(24 * ASR_SAMPLE_RATE)
    out = []
    for a, b in raw:
        a = max(0, a - pad)
        b = min(n, b + pad)
        if merge and out and a <= out[-1][1] and (b - out[-1][0]) <= max_merge:
            out[-1] = (out[-1][0], max(b, out[-1][1]))
        else:
            out.append((a, b))

    return [(a / ASR_SAMPLE_RATE, b / ASR_SAMPLE_RATE) for a, b in out]


# ============================================================
# Whisper (faster-whisper)
# ============================================================


def load_whisper(name=None):
    """
    بارگذاری و کش مدل Whisper از مسیر محلی.
    اولویت: GPU با float16؛ سپس CPU با int8.
    """

    global whisper_model, whisper_model_name

    WHISPER_MODEL_PATH = os.environ.get(
        "WHISPER_MODEL_PATH",
        os.path.join(SHENAVA_DIR, "models--deepdml--faster-whisper-large-v3-turbo-ct2"),
    )

    model_path = WHISPER_MODEL_PATH

    if whisper_model is not None and whisper_model_name == model_path:
        return whisper_model

    import os

    if not os.path.isdir(model_path):
        raise RuntimeError(
            f"پوشه مدل Whisper پیدا نشد:\n{model_path}"
        )

    if not os.path.isfile(os.path.join(model_path, "model.bin")):
        raise RuntimeError(
            "فایل model.bin در پوشه مدل پیدا نشد."
        )

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise RuntimeError(
            "کتابخانه faster-whisper نصب نیست.\n"
            "اجرا کن: python -m pip install faster-whisper"
        )

    try:
        import ctranslate2
        cuda = ctranslate2.get_cuda_device_count() > 0
    except Exception:
        cuda = False

    try:
        if cuda:
            model = WhisperModel(
                model_path,
                device="cuda",
                compute_type="float16",
                local_files_only=True
            )
        else:
            model = WhisperModel(
                model_path,
                device="cpu",
                compute_type="int8",
                local_files_only=True
            )

    except Exception:
        if not cuda:
            raise

        model = WhisperModel(
            model_path,
            device="cpu",
            compute_type="int8",
            local_files_only=True
        )

    whisper_model = model
    whisper_model_name = model_path

    return model


def whisper_transcribe(model, audio, vad_filter=False):
    """رونویسی با Whisper؛ خروجی ``[(start, end, text), ...]`` نسبت به ابتدای ``audio``."""
    segments, _info = model.transcribe(
        audio,
        language=WHISPER_LANGUAGE,
        beam_size=5,
        vad_filter=vad_filter,
        vad_parameters=WHISPER_VAD_PARAMS if vad_filter else None,
        condition_on_previous_text=False
    )
    out = []
    for s in segments:
        t = s.text.strip()
        if t:
            out.append((float(s.start), float(s.end), t))
    return out


def shenava_transcribe(audio):
    """رونویسی با Shenava؛ قطعه‌های بلند به تکه‌های ۲۵ ثانیه‌ای شکسته می‌شوند."""
    chunk = 25 * ASR_SAMPLE_RATE
    texts = []
    for i in range(0, len(audio), chunk):
        part = audio[i:i + chunk]
        if len(part) < int(0.2 * ASR_SAMPLE_RATE):
            continue
        s = recognizer.create_stream()
        s.accept_waveform(ASR_SAMPLE_RATE, np.asarray(part, dtype=np.float32))
        recognizer.decode_stream(s)
        t = s.result.text.strip()
        if t:
            texts.append(t)
    return " ".join(texts)


# ============================================================
# شناسایی آنلاین گوینده (حالت زنده)
# ============================================================

class LiveSpeakerTracker:
    """
    شناسایی گوینده به‌صورت افزایشی (online) با مقایسه‌ی embedding هر قطعه
    با مرکز (centroid) گویندگان شناخته‌شده.

    * شباهت >= آستانه  -> همان گوینده (و به‌روزرسانی مرکز در صورت کافی بودن طول)
    * در غیر این صورت  -> گوینده‌ی جدید (به شرط طول کافی و نرسیدن به سقف)
    * قطعه‌ی بسیار کوتاه -> گوینده‌ی قبلی

    اگر ``fixed_k > 0`` باشد تعداد گویندگان همان مقدار سقف می‌شود.
    """

    def __init__(self, extractor, threshold, max_speakers, fixed_k=0):
        self.extractor = extractor
        self.threshold = threshold
        self.fixed_k = max(0, int(fixed_k))
        self.max_speakers = self.fixed_k if self.fixed_k > 0 else max_speakers
        self.sums = []
        self.counts = []
        self.last = None

    def reset(self):
        """پاک‌سازی همه‌ی گویندگان (شروع جلسه‌ی جدید)."""
        self.sums = []
        self.counts = []
        self.last = None

    def _embed(self, audio):
        """embedding نرمال‌شده (طول ۱) یا None."""
        s = self.extractor.create_stream()
        s.accept_waveform(ASR_SAMPLE_RATE, np.asarray(audio, dtype=np.float32))
        s.input_finished()
        if not self.extractor.is_ready(s):
            return None
        e = np.array(self.extractor.compute(s), dtype=np.float32)
        n = np.linalg.norm(e)
        if n <= 0:
            return None
        return e / n

    def identify(self, audio):
        """شماره‌ی گوینده (از ۱) را برای قطعه‌ی داده‌شده برمی‌گرداند."""
        dur = len(audio) / ASR_SAMPLE_RATE

        emb = self._embed(audio) if dur >= SPEAKER_MIN_EMBED_SEC else None

        if emb is None:
            return self.last if self.last is not None else (
                self._new_speaker_from(None)
            )

        if not self.sums:
            return self._new_speaker_from(emb)

        sims = []
        for s in self.sums:
            c = s / (np.linalg.norm(s) + 1e-12)
            sims.append(float(np.dot(emb, c)))

        best = int(np.argmax(sims))

        if sims[best] >= self.threshold:
            if dur >= SPEAKER_MIN_UPDATE_SEC:
                self.sums[best] = self.sums[best] + emb
                self.counts[best] += 1
            self.last = best + 1
            return self.last

        min_new = SPEAKER_MIN_EMBED_SEC if self.fixed_k > 0 else SPEAKER_MIN_NEW_SEC
        if dur >= min_new and len(self.sums) < self.max_speakers:
            return self._new_speaker_from(emb)

        self.last = best + 1
        return self.last

    def _new_speaker_from(self, emb):
        """گوینده‌ی جدید با embedding داده‌شده می‌سازد و شماره‌ی او را برمی‌گرداند."""
        if emb is None:
            if not self.sums:
                self.last = 1
                return 1
            return self.last or 1
        self.sums.append(emb.copy())
        self.counts.append(1)
        self.last = len(self.sums)
        return self.last


def create_speaker_extractor(num_threads=2):
    """استخراج‌کننده‌ی embedding گوینده را می‌سازد (FileNotFoundError در نبود مدل)."""
    resolve_speaker_models()
    if not os.path.isfile(SPEAKER_EMBED_MODEL):
        raise FileNotFoundError(
            "مدل embedding گوینده پیدا نشد.\n"
            "با دکمه «⬇ دانلود مدل گوینده» آن را دانلود کن یا فایل .onnx را در پوشه زیر بگذار:\n"
            f"{SPEAKER_MODELS_DIR}"
        )
    cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=SPEAKER_EMBED_MODEL,
        num_threads=num_threads
    )
    return sherpa_onnx.SpeakerEmbeddingExtractor(cfg)


# ============================================================
# تفکیک گوینده‌ی آفلاین (حالت فایل)
# ============================================================

def run_diarization(audio, num_speakers, progress_cb=None):
    """
    تفکیک گوینده با پایپ‌لاین داخلی sherpa-onnx (pyannote segmentation + embedding).

    Returns:
        فهرست ``[(start, end, speaker), ...]`` با شماره‌ی گوینده از ۱.
    """
    resolve_speaker_models()
    for path in (SPEAKER_SEG_MODEL, SPEAKER_EMBED_MODEL):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"مدل تشخیص گوینده پیدا نشد:\n{path}")

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=SPEAKER_SEG_MODEL
            ),
            num_threads=4,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=SPEAKER_EMBED_MODEL,
            num_threads=4,
        ),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_speakers if num_speakers > 0 else -1,
            threshold=DIARIZATION_CLUSTER_THRESHOLD,
        ),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )

    if not config.validate():
        raise RuntimeError("تنظیمات تشخیص گوینده نامعتبر است.")

    diarizer = sherpa_onnx.OfflineSpeakerDiarization(config)

    def cb(num_processed, num_total):
        if progress_cb:
            progress_cb(num_processed, num_total)
        return 0

    result = diarizer.process(audio, callback=cb).sort_by_start_time()

    mapping = {}
    segs = []
    for r in result:
        if r.speaker not in mapping:
            mapping[r.speaker] = len(mapping) + 1
        segs.append((float(r.start), float(r.end), mapping[r.speaker]))

    return segs


def _embed_audio(extractor, x):
    """embedding نرمال‌شده‌ی یک قطعه‌ی صوتی یا None."""
    s = extractor.create_stream()
    s.accept_waveform(ASR_SAMPLE_RATE, np.asarray(x, dtype=np.float32))
    s.input_finished()
    if not extractor.is_ready(s):
        return None
    e = np.array(extractor.compute(s), dtype=np.float32)
    n = np.linalg.norm(e)
    return e / n if n > 0 else None


def _silhouette(D, labels):
    """میانگین silhouette بر اساس ماتریس فاصله‌ی ``D`` (کمتر از دو خوشه: ۱-)."""
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return -1.0
    n = len(labels)
    s = np.zeros(n)
    for i in range(n):
        same = labels == labels[i]
        cnt = int(same.sum())
        if cnt <= 1:
            continue
        a = D[i, same].sum() / (cnt - 1)
        b = min(D[i, labels == c].mean() for c in uniq if c != labels[i])
        s[i] = (b - a) / max(a, b, 1e-9)
    return float(s.mean())


def cluster_embeddings(E, num_speakers=0):
    """
    خوشه‌بندی سلسله‌مراتبی embeddingها (average linkage، فاصله‌ی کسینوسی).

    در حالت خودکار بهترین k با silhouette انتخاب می‌شود، به شرط: نبودن خوشه‌ی
    نویزی، تمایز مراکز خوشه‌ها (DIAR_CENTROID_MERGE_SIM) و حداقل silhouette.

    Returns:
        برچسب‌های ``0..k-1`` به طول n.
    """
    n = len(E)
    if n <= 1 or num_speakers == 1:
        return np.zeros(n, dtype=int)

    try:
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
    except Exception:
        raise RuntimeError("برای تفکیک گوینده scipy لازم است: pip install scipy")

    D = np.clip(1.0 - E @ E.T, 0.0, 2.0)
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="average")

    if num_speakers and num_speakers > 1:
        labels = fcluster(Z, min(num_speakers, n), criterion="maxclust") - 1
        return np.unique(labels, return_inverse=True)[1]

    best_labels = np.zeros(n, dtype=int)
    best_sil = -1.0
    min_size = max(2, int(DIAR_MIN_CLUSTER_FRAC * n))

    for k in range(2, min(DIAR_MAX_SPEAKERS, n - 1) + 1):
        labels = fcluster(Z, k, criterion="maxclust") - 1
        uniq = np.unique(labels)
        if len(uniq) < 2:
            continue

        if min(int((labels == c).sum()) for c in uniq) < min_size:
            continue

        cents = np.stack([E[labels == c].mean(axis=0) for c in uniq])
        cents /= (np.linalg.norm(cents, axis=1, keepdims=True) + 1e-12)
        sims = cents @ cents.T
        np.fill_diagonal(sims, -1.0)
        if sims.max() >= DIAR_CENTROID_MERGE_SIM:
            continue

        sil = _silhouette(D, labels)
        if sil >= DIAR_MIN_SILHOUETTE and sil > best_sil:
            best_sil = sil
            best_labels = labels

    return np.unique(best_labels, return_inverse=True)[1]


def diarize_embeddings(audio, num_speakers=0, progress_cb=None):
    """
    تفکیک گوینده به روش Embedding + Clustering.

    مراحل: VAD -> پنجره‌های هم‌پوشان + embedding -> خوشه‌بندی پنجره‌های قوی ->
    برچسب‌دهی پنجره‌های ضعیف -> هموارسازی A-B-A -> ساخت قطعات پیوسته.
    """
    resolve_speaker_models()
    if not os.path.isfile(SPEAKER_EMBED_MODEL):
        raise FileNotFoundError(f"مدل تشخیص گوینده پیدا نشد:\n{SPEAKER_EMBED_MODEL}")

    sr = ASR_SAMPLE_RATE
    regions = offline_vad_segments(audio, pad_sec=0.0, merge=False)
    if not regions:
        return []

    W = int(DIAR_WINDOW_SEC * sr)
    H = int(DIAR_HOP_SEC * sr)
    items = []
    for ri, (rs, re_) in enumerate(regions):
        a = int(rs * sr)
        b = min(len(audio), int(re_ * sr))
        if b - a < int(0.2 * sr):
            continue
        if b - a <= W:
            items.append([ri, a, b])
        else:
            starts = list(range(a, b - W + 1, H)) or [a]
            if starts[-1] + W < b - H // 2:
                starts.append(b - W)
            for st in starts:
                items.append([ri, st, st + W])

    if not items:
        return []

    extractor = create_speaker_extractor(num_threads=4)
    total = len(items)
    embs = []
    for i, (_ri, a, b) in enumerate(items):
        embs.append(_embed_audio(extractor, audio[a:b]))
        if progress_cb and (i % 5 == 0 or i == total - 1):
            progress_cb(i + 1, total)

    weak_len = int(DIAR_WEAK_SEC * sr)
    strong = [
        i for i in range(total)
        if embs[i] is not None and (items[i][2] - items[i][1]) >= weak_len
    ]
    if not strong:
        strong = [i for i in range(total) if embs[i] is not None]
    if not strong:
        raise RuntimeError("embedding گوینده محاسبه نشد.")

    E = np.stack([embs[i] for i in strong])
    lab_s = cluster_embeddings(E, num_speakers)
    k = int(lab_s.max()) + 1
    cents = np.stack([E[lab_s == c].mean(axis=0) for c in range(k)])
    cents /= (np.linalg.norm(cents, axis=1, keepdims=True) + 1e-12)

    labels = [None] * total
    for j, i in enumerate(strong):
        labels[i] = int(lab_s[j])
    for i in range(total):
        if labels[i] is None and embs[i] is not None:
            labels[i] = int(np.argmax(cents @ embs[i]))
    last = 0
    for i in range(total):
        if labels[i] is None:
            labels[i] = last
        else:
            last = labels[i]

    by_region = {}
    for idx, it in enumerate(items):
        by_region.setdefault(it[0], []).append(idx)

    def center(idx):
        return (items[idx][1] + items[idx][2]) / 2.0 / sr

    raw = []
    for ri, idxs in by_region.items():
        if DIAR_SMOOTH and len(idxs) >= 3:
            for p in range(1, len(idxs) - 1):
                l0, l1, l2 = labels[idxs[p - 1]], labels[idxs[p]], labels[idxs[p + 1]]
                if l0 == l2 and l1 != l0:
                    labels[idxs[p]] = l0

        rs, re_ = regions[ri]
        cur = None
        for p, idx in enumerate(idxs):
            lo = rs if p == 0 else (center(idxs[p - 1]) + center(idx)) / 2.0
            hi = re_ if p == len(idxs) - 1 else (center(idx) + center(idxs[p + 1])) / 2.0
            if cur is not None and cur[2] == labels[idx]:
                cur[1] = hi
            else:
                if cur is not None:
                    raw.append(tuple(cur))
                cur = [lo, hi, labels[idx]]
        if cur is not None:
            raw.append(tuple(cur))

    raw.sort(key=lambda x: x[0])

    mapping = {}
    segs = []
    for s, e, lab in raw:
        if lab not in mapping:
            mapping[lab] = len(mapping) + 1
        segs.append((float(s), float(e), mapping[lab]))
    return segs


# کش پایپ‌لاین pyannote (بارگذاری آن زمان‌بر است)
_pyannote_pipeline = None
_pyannote_pipeline_name = None


def _hf_token():
    """توکن HuggingFace از متغیر محیطی یا فایل HF_TOKEN_FILE؛ در نبود، None."""
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not tok and os.path.isfile(HF_TOKEN_FILE):
        try:
            with open(HF_TOKEN_FILE, encoding="utf-8") as f:
                tok = f.read().strip()
        except Exception:
            tok = None
    return tok or None


def load_pyannote_pipeline():
    """
    پایپ‌لاین pyannote.audio را بارگذاری و کش می‌کند (روی GPU در صورت وجود).

    Raises:
        RuntimeError: اگر کتابخانه نصب نباشد یا دسترسی HuggingFace کامل نباشد.
    """
    global _pyannote_pipeline, _pyannote_pipeline_name

    if _pyannote_pipeline is not None:
        return _pyannote_pipeline

    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError:
        raise RuntimeError(
            "کتابخانه pyannote.audio نصب نیست.\n"
            "اجرا کن: pip install pyannote.audio\n"
            "(برای GPU باید نسخه CUDA از torch نصب باشد.)"
        )

    token = _hf_token()
    errors = []
    pipe = None
    used = None

    for name in PYANNOTE_PIPELINES:
        for kw in ("token", "use_auth_token"):
            try:
                pipe = Pipeline.from_pretrained(name, **{kw: token})
            except TypeError:
                continue
            except Exception as exc:
                errors.append(f"{name}: {exc}")
                pipe = None
            break
        if pipe is not None:
            used = name
            break

    if pipe is None:
        raise RuntimeError(
            "بارگذاری پایپ‌لاین pyannote ناموفق بود.\n\n"
            "مراحل لازم (یک‌بار):\n"
            "1) در huggingface.co ثبت‌نام کن و یک Access Token (read) بساز.\n"
            "2) در صفحه این مدل‌ها دکمه Accept (موافقت با شرایط) را بزن:\n"
            "   pyannote/speaker-diarization-community-1\n"
            "   pyannote/speaker-diarization-3.1\n"
            "   pyannote/segmentation-3.0\n"
            f"3) توکن را در فایل {HF_TOKEN_FILE} بنویس (یا متغیر محیطی HF_TOKEN بگذار).\n\n"
            + "\n".join(errors[-2:])
        )

    try:
        if torch.cuda.is_available():
            pipe.to(torch.device("cuda"))
    except Exception:
        pass

    _pyannote_pipeline = pipe
    _pyannote_pipeline_name = used
    return pipe


def run_pyannote_audio(audio, num_speakers=0, progress_cb=None):
    """
    تفکیک گوینده با پایپ‌لاین کامل pyannote.audio.

    Returns:
        فهرست ``[(start, end, speaker), ...]`` با شماره‌ی گوینده از ۱.
    """
    import torch

    pipe = load_pyannote_pipeline()

    wav = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
    inp = {"waveform": wav, "sample_rate": ASR_SAMPLE_RATE}

    kwargs = {}
    if num_speakers and num_speakers > 0:
        kwargs["num_speakers"] = int(num_speakers)

    def hook(step_name, step_artifact, file=None, total=None, completed=None):
        if progress_cb and total:
            progress_cb(completed or 0, total)

    try:
        out = pipe(inp, hook=hook, **kwargs)
    except TypeError:
        out = pipe(inp, **kwargs)

    # نسخه ۴: خروجی «بدون هم‌پوشانی» برای رونویسی مناسب‌تر است.
    ann = getattr(out, "exclusive_speaker_diarization", None)
    if ann is None:
        ann = getattr(out, "speaker_diarization", None)
    if ann is None:
        ann = out

    raw = [
        (float(turn.start), float(turn.end), spk)
        for turn, _, spk in ann.itertracks(yield_label=True)
    ]
    raw.sort(key=lambda x: x[0])

    mapping = {}
    segs = []
    for s, e, spk in raw:
        if spk not in mapping:
            mapping[spk] = len(mapping) + 1
        segs.append((s, e, mapping[spk]))
    return segs


def diarize(audio, num_speakers, method, progress_cb=None):
    """نقطه‌ی ورود یکپارچه‌ی تفکیک گوینده بر اساس نام روش."""
    if method and method.startswith("Pyannote.audio"):
        return run_pyannote_audio(audio, num_speakers, progress_cb)
    if method == "Sherpa Pyannote":
        return run_diarization(audio, num_speakers, progress_cb)
    return diarize_embeddings(audio, num_speakers, progress_cb)


def assign_speakers_by_overlap(segs, diar):
    """
    به هر قطعه‌ی متن، گوینده‌ای با بیشترین هم‌پوشانی زمانی را نسبت می‌دهد
    (در نبود هم‌پوشانی: نزدیک‌ترین قطعه). ``segs`` درجا تغییر می‌کند.
    """
    for seg in segs:
        s, e = seg.get("start"), seg.get("end")
        if s is None or e is None or not diar:
            continue
        score = {}
        for ds, de, spk in diar:
            ov = min(e, de) - max(s, ds)
            if ov > 0:
                score[spk] = score.get(spk, 0.0) + ov
        if score:
            seg["speaker"] = max(score, key=score.get)
        else:
            mid = (s + e) / 2.0
            seg["speaker"] = min(
                diar, key=lambda d: abs((d[0] + d[1]) / 2.0 - mid)
            )[2]


def merge_diarization_segments(segs, max_gap=1.0, max_len=28.0):
    """قطعات پشت‌سرهم یک گوینده را ادغام می‌کند تا ASR زمینه‌ی بیشتری ببیند."""
    merged = []
    for start, end, spk in segs:
        if (
            merged
            and merged[-1][2] == spk
            and start - merged[-1][1] <= max_gap
            and end - merged[-1][0] <= max_len
        ):
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]), spk)
        else:
            merged.append((start, end, spk))
    return merged


# ============================================================
# خواندن فایل صوتی / تصویری
# ============================================================

def _decode_with_av(path):
    """خواندن صدا مستقیم با PyAV (آرایه‌ی float32 تک‌کاناله، نرخ ASR_SAMPLE_RATE)."""
    import av

    container = av.open(path)
    try:
        stream = next(
            (s for s in container.streams if s.type == "audio"), None
        )
        if stream is None:
            raise RuntimeError("این فایل مسیر صوتی ندارد.")

        resampler = av.AudioResampler(
            format="s16", layout="mono", rate=ASR_SAMPLE_RATE
        )

        chunks = []

        def collect(out):
            if out is None:
                return
            if not isinstance(out, (list, tuple)):
                out = [out]
            for f in out:
                arr = f.to_ndarray().reshape(-1)
                chunks.append(arr.astype(np.float32) / 32768.0)

        for frame in container.decode(stream):
            frame.pts = None
            collect(resampler.resample(frame))

        try:
            collect(resampler.resample(None))
        except Exception:
            pass
    finally:
        container.close()

    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32)


def load_audio_file(path):
    """فایل صوتی/تصویری را به آرایه‌ی float32 تک‌کاناله با نرخ ۱۶kHz تبدیل می‌کند."""
    try:
        from faster_whisper.audio import decode_audio
    except ImportError:
        raise RuntimeError(
            "برای خواندن فایل صوتی faster-whisper لازم است.\n"
            "اجرا کن: pip install faster-whisper"
        )
    try:
        audio = decode_audio(path, sampling_rate=ASR_SAMPLE_RATE)
    except TypeError:
        audio = _decode_with_av(path)
    return np.asarray(audio, dtype=np.float32)


# ============================================================
# ارتباط پایه با Ollama
# ============================================================

def ollama_installed_models():
    """نام همه‌ی مدل‌های نصب‌شده در Ollama (set)."""
    response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
    response.raise_for_status()
    return {
        m.get("name", "") for m in response.json().get("models", [])
        if m.get("name")
    }


def ollama_pull(model, status_cb=None):
    """
    مدلی را از کتابخانه‌ی Ollama دانلود می‌کند (stream) و پیشرفت را گزارش می‌دهد.

    Raises:
        RuntimeError: اگر Ollama خطا برگرداند.
    """
    last = 0.0
    with requests.post(
        f"{OLLAMA_URL}/api/pull",
        json={"model": model, "stream": True},
        stream=True, timeout=(10, 600)
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line.decode("utf-8"))
            except Exception:
                continue
            if data.get("error"):
                raise RuntimeError(str(data["error"]))
            now = time.monotonic()
            if status_cb and now - last > 0.3:
                last = now
                total = data.get("total") or 0
                done = data.get("completed") or 0
                st = data.get("status", "")
                if total:
                    status_cb(
                        f"دانلود {model}: {int(100 * done / total)}٪ "
                        f"({done / 1e9:.1f}/{total / 1e9:.1f} GB)"
                    )
                else:
                    status_cb(f"دانلود {model}: {st}")


def ollama_chat(model, messages, options=None, think=None, timeout=None):
    """
    فراخوانی /api/chat و برگرداندن متن پاسخ.

    اگر ``think`` داده شود و مدل از آن پشتیبانی نکند (خطای ۴۰۰)، یک‌بار بدون
    آن تکرار می‌شود.
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "keep_alive": TRANSLATION_KEEP_ALIVE,
        "options": options or {},
    }
    if think is not None:
        payload["think"] = think

    response = requests.post(
        f"{OLLAMA_URL}/api/chat", json=payload,
        timeout=timeout or OLLAMA_TIMEOUT
    )
    if response.status_code == 400 and think is not None:
        payload.pop("think", None)
        response = requests.post(
            f"{OLLAMA_URL}/api/chat", json=payload,
            timeout=timeout or OLLAMA_TIMEOUT
        )
    response.raise_for_status()
    return response.json().get("message", {}).get("content", "")


# ============================================================
# ترجمه‌ی دقیق: TranslateGemma (Ollama) + جایگزین NLLB 3.3B
# ============================================================

def pick_translation_model(force_refresh=False):
    """
    بهترین مدل ترجمه‌ی نصب‌شده در Ollama را انتخاب می‌کند (یا None).

    ترتیب: TranslateGemma (27b > 12b > 4b > هر تگ دیگر) -> مدل‌های عمومی
    TRANSLATION_OLLAMA_GENERAL -> QWEN_MODEL_NAME. نتیجه کش می‌شود.
    """
    global _translation_model_cache, _translation_model_checked

    with translation_model_lock:
        if _translation_model_checked and not force_refresh:
            return _translation_model_cache

        try:
            names = ollama_installed_models()
        except Exception:
            # Ollama در دسترس نیست؛ کش نمی‌کنیم تا بعداً دوباره بررسی شود
            return None

        chosen = None
        for m in TRANSLATION_OLLAMA_PREFERRED:
            if m in names:
                chosen = m
                break
        if chosen is None:
            tg = sorted(n for n in names if n.lower().startswith("translategemma"))
            if tg:
                chosen = tg[-1]
        if chosen is None:
            for m in TRANSLATION_OLLAMA_GENERAL + [QWEN_MODEL_NAME]:
                if m in names:
                    chosen = m
                    break

        _translation_model_cache = chosen
        _translation_model_checked = True
        return chosen


def reset_translation_model_cache():
    """پس از دانلود مدل جدید، انتخاب دوباره انجام می‌شود."""
    global _translation_model_checked, _translation_model_cache
    with translation_model_lock:
        _translation_model_checked = False
        _translation_model_cache = None


def _is_translategemma(model):
    return bool(model) and model.lower().startswith("translategemma")


def _build_translation_messages(model, text, lang_entry, prev_text=None, src=None):
    """
    پیام‌های ارسالی به مدل را می‌سازد.

    * TranslateGemma: پرامپت رسمی خودِ مدل (بدون system و بدون توضیح اضافه).
    * مدل‌های عمومی: پرامپت عمومی با هشدار درباره‌ی خطای ASR (بدون هیچ حالت خاص برای موضوع یا عبارت).
    """
    tgt_iso, _nllb, tgt_name = lang_entry
    src_iso, src_name = src or (TRANSLATION_SOURCE_ISO, TRANSLATION_SOURCE_NAME)

    if _is_translategemma(model):
        prompt = (
            f"You are a professional {src_name} ({src_iso}) to {tgt_name} ({tgt_iso}) "
            f"translator. Your goal is to accurately convey the meaning and nuances "
            f"of the original {src_name} text while adhering to {tgt_name} grammar, "
            f"vocabulary, and cultural sensitivities.\n"
            f"Produce only the {tgt_name} translation, without any additional "
            f"explanations or commentary. Please translate the following {src_name} "
            f"text into {tgt_name}:\n\n\n{text}"
        )
        return [{"role": "user", "content": prompt}]

    system = (
        f"You are an expert {src_name}-to-{tgt_name} translator. The input is a "
        f"transcript from automatic speech recognition, so it may lack punctuation "
        f"and contain small recognition errors. Translate the full meaning faithfully "
        f"and naturally, keeping the register and tone of the speaker. Translate "
        f"idioms and fixed expressions by their meaning, not word by word, and keep "
        f"proper names as they are. Never add, omit or explain anything. "
        f"Output only the {tgt_name} translation."
    )
    user = f"Translate this {src_name} text into {tgt_name}:\n\n{text}"
    if prev_text:
        user = (
            f"(Previous sentence, for context only. Do NOT translate it: {prev_text})\n\n"
            + user
        )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def translate_with_ollama(text, lang_entry, model, prev_text=None, src=None):
    """ترجمه با مدل Ollama؛ دما صفر برای خروجی پایدار و بدون خلاقیت."""
    messages = _build_translation_messages(model, text, lang_entry, prev_text, src)
    raw = ollama_chat(
        model, messages,
        options={
            "temperature": 0.0,
            "top_p": 1.0,
            "num_predict": TRANSLATION_NUM_PREDICT,
            "num_ctx": TRANSLATION_NUM_CTX,
        },
        # TranslateGemma تفکر ندارد؛ برای مدل‌های عمومی (Qwen3) تفکر خاموش می‌شود
        think=None if _is_translategemma(model) else False,
        timeout=TRANSLATION_OLLAMA_TIMEOUT,
    )
    out = clean_qwen_output(raw)
    if not out:
        raise RuntimeError("مدل ترجمه خروجی خالی برگرداند.")
    return out


def load_nllb_model():
    """مدل جایگزین NLLB-200 را یک‌بار بارگذاری می‌کند (دانلود اولیه از HuggingFace)."""
    global nllb_tokenizer, nllb_model, nllb_device
    with nllb_lock:
        if nllb_model is not None and nllb_tokenizer is not None:
            return nllb_tokenizer, nllb_model
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        except ImportError as exc:
            raise RuntimeError(
                "برای ترجمه‌ی جایگزین باید transformers و sentencepiece نصب باشند.\n"
                "python -m pip install transformers sentencepiece"
            ) from exc

        nllb_device = "cpu"
        if torch.cuda.is_available():
            try:
                free_vram, _ = torch.cuda.mem_get_info()
                if free_vram >= int(NLLB_MIN_FREE_VRAM_GB * 1024 ** 3):
                    nllb_device = "cuda"
            except Exception:
                nllb_device = "cpu"
        dtype = torch.float16 if nllb_device == "cuda" else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            NLLB_MODEL_NAME, torch_dtype=dtype
        )
        model.to(nllb_device)
        model.eval()
        nllb_tokenizer = tokenizer
        nllb_model = model
        return tokenizer, model


def translate_with_nllb(text, lang_entry):
    """ترجمه با NLLB-200 3.3B (beam=5) به‌عنوان جایگزین."""
    import torch
    target_code = lang_entry[1]
    tokenizer, model = load_nllb_model()
    tokenizer.src_lang = NLLB_SOURCE_LANG
    encoded = tokenizer(
        text.strip(), return_tensors="pt", truncation=True, max_length=512
    )
    encoded = {k: v.to(nllb_device) for k, v in encoded.items()}
    target_id = tokenizer.convert_tokens_to_ids(target_code)
    if target_id is None or target_id == tokenizer.unk_token_id:
        raise RuntimeError(f"کد زبان مقصد پشتیبانی نشد: {target_code}")
    with torch.inference_mode():
        generated = model.generate(
            **encoded, forced_bos_token_id=target_id,
            max_new_tokens=NLLB_MAX_NEW_TOKENS,
            num_beams=5, do_sample=False, no_repeat_ngram_size=4
        )
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()


# ============================================================
# نرمال‌سازی اعداد فارسی (حروف -> رقم) پیش از ترجمه
# ============================================================
# مدل‌های سبک اعداد نوشته‌شده با حروف را (مثل «صد و ده») بد ترجمه می‌کنند.
# تبدیل به رقم (110) معنا را بدون تغییر نگه می‌دارد و خطا را بسیار کم می‌کند.

_NUM_SMALL = {
    "یک": 1, "دو": 2, "سه": 3, "چهار": 4, "پنج": 5, "شش": 6, "شیش": 6,
    "هفت": 7, "هشت": 8, "نه": 9, "ده": 10, "یازده": 11, "دوازده": 12,
    "سیزده": 13, "چهارده": 14, "پانزده": 15, "پونزده": 15, "شانزده": 16,
    "شونزده": 16, "هفده": 17, "هجده": 18, "هیجده": 18, "نوزده": 19,
    "بیست": 20, "سی": 30, "چهل": 40, "پنجاه": 50, "شصت": 60, "هفتاد": 70,
    "هشتاد": 80, "نود": 90, "صد": 100, "یکصد": 100, "دویست": 200,
    "سیصد": 300, "چهارصد": 400, "پانصد": 500, "ششصد": 600, "هفتصد": 700,
    "هشتصد": 800, "نهصد": 900,
}
_NUM_MULT = {"هزار": 1000, "میلیون": 10 ** 6, "میلیارد": 10 ** 9}
_NUM_EDGE = "«»\"'()[]{}،,.;:!?؟؛…"


def _parse_number_run(words):
    """ارزش عددی دنباله‌ای از کلمه‌های عدد (بدون «و») را برمی‌گرداند."""
    total, cur = 0, 0
    for w in words:
        if w in _NUM_SMALL:
            cur += _NUM_SMALL[w]
        else:
            total += (cur or 1) * _NUM_MULT[w]
            cur = 0
    return total + cur


def normalize_persian_numbers(text):
    """
    اعداد نوشته‌شده با حروف را به رقم تبدیل می‌کند: «صد و ده فروند» -> «110 فروند».

    برای جلوگیری از تغییرهای نابه‌جا، تک‌کلمه‌های کوچک (مثل «یک روز»، «ده») و
    ارقام زیر ۱۰ دست‌نخورده می‌مانند؛ فقط عددهای دست‌کم دوکلمه‌ای یا ≥۲۰ تبدیل می‌شوند.
    """
    tokens = text.split()
    out, i, n = [], 0, len(tokens)

    def core(tok):
        return tok.strip(_NUM_EDGE)

    while i < n:
        c = core(tokens[i])
        if c in _NUM_SMALL or c in _NUM_MULT:
            words, j = [c], i + 1
            while j < n:
                cj = core(tokens[j])
                if cj in _NUM_SMALL or cj in _NUM_MULT:
                    words.append(cj)
                    j += 1
                elif cj == "و" and j + 1 < n and (
                    core(tokens[j + 1]) in _NUM_SMALL
                    or core(tokens[j + 1]) in _NUM_MULT
                ):
                    j += 1
                else:
                    break
            value = _parse_number_run(words)
            if len(words) > 1 or value >= 20:
                tail = tokens[j - 1][len(tokens[j - 1].rstrip(_NUM_EDGE)):]
                head = tokens[i][:len(tokens[i]) - len(tokens[i].lstrip(_NUM_EDGE))]
                out.append(f"{head}{value}{tail}")
                i = j
                continue
        out.append(tokens[i])
        i += 1
    return " ".join(out)


_DIGIT_MAP = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


def _norm_fa(t):
    """نرمال‌سازی فارسی برای مقایسه: یکسان‌سازی ی/ک، ارقام، نیم‌فاصله و حذف علائم."""
    t = (t or "").replace("ي", "ی").replace("ك", "ک").replace("\u200c", " ")
    t = t.translate(_DIGIT_MAP)
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip().lower()


def _similarity(a, b):
    """شباهت دو متن فارسی (میانگین شباهت نویسه‌ای و واژه‌ای) بین ۰ و ۱."""
    import difflib
    a, b = _norm_fa(a), _norm_fa(b)
    if not a or not b:
        return 0.0
    ch = difflib.SequenceMatcher(None, a, b).ratio()
    wd = difflib.SequenceMatcher(None, a.split(), b.split()).ratio()
    return 0.5 * ch + 0.5 * wd


def translate_accurate(text, entry, model, prev_text=None):
    """
    ترجمه‌ی دقیق‌تر با مدل سبک: ترجمه‌ی مستقیم + سنجش با ترجمه‌ی برگشتی.

    1. ترجمه‌ی مستقیم فارسی -> مقصد، سپس برگرداندن آن به فارسی و مقایسه با متن اصلی.
    2. اگر شباهت کم بود، مسیر دوم (فارسی -> انگلیسی -> مقصد) هم سنجیده می‌شود.
    3. نتیجه‌ای که برگشتش به متن اصلی نزدیک‌تر است انتخاب می‌شود.

    برای مقصد انگلیسی فقط ترجمه‌ی مستقیم (و سنجش آن) انجام می‌شود.
    """
    fa_entry = TRANSLATION_LANGUAGES["فارسی"]
    en_entry = TRANSLATION_LANGUAGES["انگلیسی"]
    fa_src = (TRANSLATION_SOURCE_ISO, TRANSLATION_SOURCE_NAME)

    direct = translate_with_ollama(text, entry, model, prev_text)

    if entry[0] == "en":
        return direct, model

    def score(candidate):
        back = translate_with_ollama(
            candidate, fa_entry, model, src=(entry[0], entry[2])
        )
        return _similarity(back, text)

    try:
        s_direct = score(direct)
        if s_direct >= TRANSLATION_RT_ACCEPT:
            return direct, model

        english = translate_with_ollama(text, en_entry, model, prev_text)
        pivot = translate_with_ollama(
            english, entry, model, src=("en", "English")
        )
        s_pivot = score(pivot)
        if s_pivot > s_direct + TRANSLATION_RT_MARGIN:
            return pivot, f"{model} · pivot"
    except Exception:
        pass    # اگر سنجش شکست خورد، ترجمه‌ی مستقیم معتبر است
    return direct, model


def translate_persian_text(text, lang_name, prev_text=None):
    """
    ترجمه‌ی متن فارسی به ``lang_name`` (نام فارسی زبان).

    ابتدا بهترین مدل Ollama (TranslateGemma)، و در صورت نبود یا خطا NLLB 3.3B.

    Returns:
        ``(ترجمه, نام_موتور)``

    Raises:
        RuntimeError: اگر هر دو موتور ناموفق باشند.
    """
    if not text or not text.strip():
        return "", "-"
    entry = TRANSLATION_LANGUAGES.get(lang_name)
    if entry is None:
        raise RuntimeError(f"زبان مقصد پشتیبانی نمی‌شود: {lang_name}")
    if entry[0] == TRANSLATION_SOURCE_ISO:
        return text.strip(), "-"

    # اعداد حروفی -> رقم (فقط برای ورودی مدل؛ متن نمایش‌داده‌شده تغییر نمی‌کند)
    text = normalize_persian_numbers(text)

    errors = []
    model = pick_translation_model()
    if model:
        try:
            if TRANSLATION_HIGH_ACCURACY:
                return translate_accurate(text, entry, model, prev_text)
            return translate_with_ollama(text, entry, model, prev_text), model
        except Exception as exc:
            errors.append(f"Ollama ({model}): {exc}")
    else:
        errors.append(
            "هیچ مدل ترجمه‌ای در Ollama نصب نیست "
            f"(دستور: ollama pull {TRANSLATION_PULL_MODEL})"
        )

    try:
        return translate_with_nllb(text, entry), "NLLB-1.3B"
    except Exception as exc:
        errors.append(f"NLLB: {exc}")

    raise RuntimeError("\n".join(errors))


# ============================================================
# اتصال به Ollama (تحلیل)
# ============================================================

def load_qwen():
    """
    اتصال به Ollama و وجود مدل تحلیل را بررسی می‌کند؛ سپس ``qwen`` را True می‌کند.

    Raises:
        RuntimeError: اگر مدل در Ollama نصب نباشد.
        requests.RequestException: اگر سرور Ollama در دسترس نباشد.
    """
    global qwen

    available_names = {n for n in ollama_installed_models() if n}

    if QWEN_MODEL_NAME not in available_names:
        available_short_names = {
            name.split(":")[0] for name in available_names
        }

        if QWEN_MODEL_NAME.split(":")[0] not in available_short_names:
            installed = "\n".join(sorted(available_names))
            raise RuntimeError(
                f"مدل {QWEN_MODEL_NAME} در Ollama پیدا نشد.\n\n"
                f"مدل‌های نصب‌شده:\n{installed or 'هیچ مدلی پیدا نشد'}\n\n"
                f"برای نصب مدل اجرا کن:\n"
                f"ollama pull {QWEN_MODEL_NAME}"
            )

    qwen = True


def clean_qwen_output(text):
    """خروجی مدل را پاک‌سازی می‌کند: حذف بلوک‌های ``<think>`` و خطوط خالی."""
    if not text:
        return ""

    text = str(text)

    patterns = [
        r"<think>.*?</think>",
        r"<think>.*",
        r"<\|think\|>.*?<\|end[_-]?think\|>",
        r"<\|think\|>.*",
        r"<\|thinking\|>.*?</\|thinking\|>",
    ]
    for p in patterns:
        text = re.sub(p, "", text, flags=re.DOTALL | re.IGNORECASE)

    return "\n".join(
        line.rstrip() for line in text.splitlines() if line.strip()
    ).strip()


# ============================================================
# پرامپت و تحلیل متن با Qwen
# ============================================================

def build_qwen_prompt(text, speakers_present):
    """پرامپت تحلیل را می‌سازد (با بخش «گوینده‌ها» در صورت وجود چند گوینده)."""
    speaker_part = ""
    if speakers_present:
        speaker_part = """
گوینده‌ها:
برای هر گوینده در یک تا دو جمله بنویس چه گفته یا چه موضعی داشته است.
(برچسب گوینده‌ها خودکار تشخیص داده شده و ممکن است گاهی اشتباه باشد؛
اگر مطمئن نیستی این را ذکر کن.)
"""

    return f"""
تو یک تحلیل‌گر حرفه‌ای زبان فارسی هستی.

متن زیر توسط سیستم تشخیص گفتار فارسی استخراج شده است
و ممکن است شامل اشتباهات تشخیص گفتار باشد.
فقط بر اساس اطلاعات موجود در متن تحلیل کن.
اطلاعات، اشخاص، مکان‌ها یا رویدادهای جدید اختراع نکن.

متن:
--------------------------------------------------
{text}
--------------------------------------------------

تحلیل را با ساختار زیر ارائه کن:

موضوع اصلی:
یک جمله کوتاه درباره موضوع اصلی متن.

خلاصه:
خلاصه دقیق و کوتاه از آنچه از متن می‌فهمی.
{speaker_part}"""


def analyze_with_qwen(text, speakers_present=False):
    """
    متن را برای تحلیل به Qwen (از طریق Ollama) می‌فرستد.

    Returns:
        ``(نتیجه, زمان_اجرا_به_ثانیه)``
    """
    if qwen is None:
        raise RuntimeError("ابتدا از طریق دکمه بارگذاری به Ollama متصل شو.")

    if not text or not text.strip():
        return "متنی برای تحلیل وجود ندارد.", 0.0

    prompt = build_qwen_prompt(text, speakers_present)

    start = time.perf_counter()

    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": QWEN_MODEL_NAME,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "تو تحلیل‌گر حرفه‌ای متن فارسی هستی. "
                        "فقط پاسخ نهایی را به زبان فارسی ارائه کن."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "think": False,
            "options": {
                "temperature": QWEN_TEMPERATURE,
                "top_p": 0.9,
                "num_predict": QWEN_MAX_TOKENS
            }
        },
        timeout=OLLAMA_TIMEOUT
    )

    response.raise_for_status()
    data = response.json()

    elapsed = time.perf_counter() - start

    result = clean_qwen_output(data.get("message", {}).get("content", ""))

    if not result:
        result = "مدل پاسخ متنی قابل نمایشی تولید نکرد."

    return result, elapsed


# ============================================================
# رابط کاربری (GUI)
# ============================================================

class ShenavaQwenGUI:
    """
    پنجره‌ی اصلی برنامه.

    ترد‌ها: ترد اصلی (Tk) | audio_worker | segment_worker | file_thread |
    refine_thread | translation_worker | qwen_*_thread.

    قاعده: هر ترد پس‌زمینه برای دست‌کاری ویجت‌ها باید از ``root.after(0, ...)``
    استفاده کند، چون tkinter ایمن در برابر چندترد نیست.
    """

    def __init__(self, root):
        self.root = root

        self.root.title("SHENAVA AI | Persian Speech Intelligence")
        self.root.geometry("1360x960")
        self.root.minsize(1050, 760)
        self.root.configure(bg="#0B1120")

        self.colors = {
            "bg": "#0B1120",
            "panel": "#111B2E",
            "panel2": "#162238",
            "input": "#0D1728",
            "border": "#26364F",
            "text": "#E8EEF8",
            "muted": "#91A2BC",
            "accent": "#38BDF8",
            "green": "#34D399",
            "red": "#FB7185",
            "purple": "#A78BFA",
            "yellow": "#FBBF24",
        }

        self.segment_queue = None
        self.session_start = 0.0
        self.tracker = None
        self.speaker_extractor = None

        self.live_diar_var = tk.BooleanVar(
            value=os.path.isfile(SPEAKER_EMBED_MODEL)
        )
        self.file_path_var = tk.StringVar()
        self.engine_var = tk.StringVar(value="Whisper")
        self.whisper_var = tk.StringVar(value=DEFAULT_WHISPER_MODEL)
        self.file_diar_var = tk.BooleanVar(
            value=(
                os.path.isfile(SPEAKER_EMBED_MODEL)
                and os.path.isfile(SPEAKER_SEG_MODEL)
            )
        )
        self.num_speakers_var = tk.IntVar(value=0)
        self.diar_method_var = tk.StringVar(value=DEFAULT_DIAR_METHOD)
        self.refine_var = tk.BooleanVar(value=True)
        self.translation_enabled_var = tk.BooleanVar(value=False)
        self.translation_language_var = tk.StringVar(value="انگلیسی")
        self.translation_accurate_var = tk.BooleanVar(value=TRANSLATION_HIGH_ACCURACY)
        # مقادیر ساده برای خواندن امن از تردهای پس‌زمینه (بدون دسترسی به Tkinter)
        self.translation_enabled = False
        self.translation_target_name = "انگلیسی"
        self.translation_queue = queue.Queue()

        self.rec_chunks = []

        self._setup_style()
        self.create_gui()
        threading.Thread(target=self.translation_worker, daemon=True).start()

        if not os.path.isfile(SPEAKER_EMBED_MODEL):
            self.status.config(
                text=(
                    "مدل تشخیص گوینده پیدا نشد؛ "
                    "با دکمه «دانلود مدل گوینده» آن را دانلود کن."
                ),
                fg=self.colors["yellow"]
            )

        if not (
            os.path.isfile(SPEAKER_EMBED_MODEL)
            and os.path.isfile(SPEAKER_SEG_MODEL)
        ):
            self.root.after(800, self.offer_download_models)

        self.root.after(100, self.update_audio_level)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    # --------------------------------------------------------
    def _setup_style(self):
        """تعریف استایل‌های ttk (دکمه‌های عادی، Accent، Stop و نوار پیشرفت)."""
        c = self.colors
        style = ttk.Style(self.root)

        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(
            "TButton",
            font=("Tahoma", 9, "bold"),
            padding=(13, 10),
            background=c["panel2"],
            foreground=c["text"],
            borderwidth=0
        )
        style.map(
            "TButton",
            background=[
                ("disabled", "#1B2639"),
                ("pressed", "#263D5B"),
                ("active", "#263D5B")
            ]
        )

        style.configure(
            "Accent.TButton",
            background="#087EA4",
            foreground="#FFFFFF",
            padding=(16, 11),
            font=("Tahoma", 10, "bold")
        )
        style.map(
            "Accent.TButton",
            background=[
                ("disabled", "#26364F"),
                ("pressed", "#075B79"),
                ("active", "#0891B2")
            ]
        )

        style.configure(
            "Stop.TButton",
            background="#7F2941",
            foreground="#FFFFFF",
            padding=(16, 11),
            font=("Tahoma", 10, "bold")
        )
        style.map(
            "Stop.TButton",
            background=[
                ("disabled", "#26364F"),
                ("pressed", "#9F294A"),
                ("active", "#BE3458")
            ]
        )

        style.configure(
            "TProgressbar",
            troughcolor=c["input"],
            background=c["accent"],
            bordercolor=c["input"],
            thickness=10
        )

    # --------------------------------------------------------
    def _card(self, parent, **grid):
        """یک کارت (Frame با حاشیه) می‌سازد و با پارامترهای grid قرار می‌دهد."""
        c = self.colors
        f = tk.Frame(
            parent,
            bg=c["panel"],
            highlightbackground=c["border"],
            highlightthickness=1
        )
        f.grid(**grid)
        return f

    def create_gui(self):
        """چیدمان کامل پنجره (ردیف‌های ``outer``: هدر، جداکننده، Ollama، فایل، مانیتور، عملیات، فوتر)."""
        c = self.colors

        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        outer = tk.Frame(self.root, bg=c["bg"])
        outer.grid(row=0, column=0, sticky="nsew")
        outer.grid_columnconfigure(0, weight=1)
        outer.grid_rowconfigure(4, weight=1)

        # ---------------- هدر ----------------
        header = tk.Frame(outer, bg=c["bg"])
        header.grid(row=0, column=0, sticky="ew", padx=24, pady=(16, 10))
        header.grid_columnconfigure(0, weight=1)

        brand = tk.Frame(header, bg=c["bg"])
        brand.grid(row=0, column=0, sticky="w")

        logo = tk.Canvas(
            brand, width=54, height=54, bg=c["bg"], highlightthickness=0
        )
        logo.pack(side="left", padx=(0, 14))
        logo.create_oval(
            3, 3, 51, 51, fill="#12334B", outline="#236A8B", width=2
        )
        logo.create_text(
            27, 27, text="S", fill=c["accent"],
            font=("Segoe UI", 24, "bold")
        )

        titles = tk.Frame(brand, bg=c["bg"])
        titles.pack(side="left")

        tk.Label(
            titles, text="SHENAVA AI", bg=c["bg"], fg=c["text"],
            font=("Segoe UI", 23, "bold")
        ).pack(anchor="w")

        tk.Label(
            titles,
            text="PERSIAN SPEECH INTELLIGENCE  •  SPEAKERS  •  WHISPER  •  LOCAL AI",
            bg=c["bg"], fg=c["muted"], font=("Segoe UI", 9)
        ).pack(anchor="w", pady=(2, 0))

        self.header_status = tk.Label(
            header, text="● SYSTEM READY", bg=c["panel"], fg=c["green"],
            font=("Segoe UI", 9, "bold"), padx=14, pady=10
        )
        self.header_status.grid(row=0, column=1, sticky="e")

        ttk.Separator(outer, orient="horizontal").grid(
            row=1, column=0, sticky="ew", padx=24, pady=(0, 10)
        )

        # ---------------- کارت Ollama ----------------
        model_card = self._card(
            outer, row=2, column=0, sticky="ew", padx=24, pady=(0, 10)
        )
        model_card.grid_columnconfigure(0, weight=1)

        model_heading = tk.Frame(model_card, bg=c["panel"])
        model_heading.grid(
            row=0, column=0, columnspan=2, sticky="ew", padx=18, pady=(10, 5)
        )
        tk.Label(
            model_heading, text="مدل تحلیل متن", bg=c["panel"],
            fg=c["text"], font=("Tahoma", 11, "bold")
        ).pack(side="right")
        tk.Label(
            model_heading, text="OLLAMA / QWEN", bg=c["panel"],
            fg=c["purple"], font=("Segoe UI", 9, "bold")
        ).pack(side="left")

        self.model_label = tk.Label(
            model_card,
            text=f"مدل: {QWEN_MODEL_NAME}   |   آدرس: {OLLAMA_URL}",
            bg=c["input"], fg=c["text"], font=("Consolas", 9),
            anchor="w", justify="left", padx=12, pady=8
        )
        self.model_label.grid(
            row=1, column=0, sticky="ew", padx=18, pady=(0, 8)
        )

        model_buttons = tk.Frame(model_card, bg=c["panel"])
        model_buttons.grid(row=1, column=1, padx=(0, 18), pady=(0, 8))

        self.select_model_button = ttk.Button(
            model_buttons, text="⚙ تنظیم مدل",
            command=self.select_qwen_model
        )
        self.select_model_button.pack(side="right", padx=4)

        self.load_model_button = ttk.Button(
            model_buttons, text="⚡ اتصال به Ollama",
            command=self.load_qwen_from_gui, style="Accent.TButton"
        )
        self.load_model_button.pack(side="right", padx=4)

        self.model_status = tk.Label(
            model_card, text="● Ollama هنوز بررسی نشده است",
            bg=c["panel"], fg=c["yellow"], font=("Tahoma", 9, "bold"),
            anchor="e"
        )
        self.model_status.grid(
            row=2, column=0, columnspan=2, sticky="ew", padx=18, pady=(0, 9)
        )

        # ---------------- کارت حالت فایل ----------------
        file_card = self._card(
            outer, row=3, column=0, sticky="ew", padx=24, pady=(0, 10)
        )
        file_card.grid_columnconfigure(1, weight=1)

        tk.Label(
            file_card, text="حالت ضبط‌شده (فایل صوتی / تصویری)",
            bg=c["panel"], fg=c["text"], font=("Tahoma", 11, "bold")
        ).grid(row=0, column=0, columnspan=4, sticky="e", padx=18, pady=(10, 6))

        self.browse_button = ttk.Button(
            file_card, text="📁 انتخاب فایل", command=self.browse_file
        )
        self.browse_button.grid(row=1, column=0, padx=(18, 6), pady=4)

        self.file_entry = tk.Entry(
            file_card, textvariable=self.file_path_var,
            bg=c["input"], fg=c["text"], insertbackground=c["accent"],
            relief="flat", font=("Consolas", 10)
        )
        self.file_entry.grid(
            row=1, column=1, columnspan=3, sticky="ew",
            padx=(6, 18), pady=4, ipady=7
        )

        opts = tk.Frame(file_card, bg=c["panel"])
        opts.grid(
            row=2, column=0, columnspan=4, sticky="ew", padx=18, pady=(4, 10)
        )

        self.process_file_button = ttk.Button(
            opts, text="▶ پردازش فایل", command=self.process_file,
            style="Accent.TButton"
        )
        self.process_file_button.pack(side="left", padx=(0, 12))

        def lbl(text):
            tk.Label(
                opts, text=text, bg=c["panel"], fg=c["muted"],
                font=("Tahoma", 9)
            ).pack(side="left", padx=(8, 3))

        lbl("موتور:")
        ttk.Combobox(
            opts, textvariable=self.engine_var, values=["Whisper", "Shenava"],
            state="readonly", width=9
        ).pack(side="left")

        lbl("مدل Whisper (یا مسیر محلی):")
        ttk.Combobox(
            opts, textvariable=self.whisper_var, values=WHISPER_MODELS,
            width=18
        ).pack(side="left")

        tk.Checkbutton(
            opts, text="تشخیص گوینده", variable=self.file_diar_var,
            bg=c["panel"], fg=c["text"], selectcolor=c["input"],
            activebackground=c["panel"], activeforeground=c["text"],
            font=("Tahoma", 9)
        ).pack(side="left", padx=(14, 4))

        lbl("روش تفکیک:")
        ttk.Combobox(
            opts, textvariable=self.diar_method_var, values=DIAR_METHODS,
            state="readonly", width=24
        ).pack(side="left")

        lbl("تعداد گوینده (۰ = خودکار):")
        tk.Spinbox(
            opts, from_=0, to=20, width=4, textvariable=self.num_speakers_var,
            bg=c["input"], fg=c["text"], relief="flat",
            buttonbackground=c["panel2"]
        ).pack(side="left")

        # ---------------- ناحیه‌ی مانیتور ----------------
        monitor_row = tk.Frame(outer, bg=c["bg"])
        monitor_row.grid(row=4, column=0, sticky="nsew", padx=24)
        monitor_row.grid_columnconfigure(0, weight=1)
        monitor_row.grid_columnconfigure(1, weight=3)
        monitor_row.grid_rowconfigure(1, weight=1)

        audio_card = self._card(
            monitor_row, row=0, column=0, sticky="nsew",
            padx=(0, 8), pady=(0, 10)
        )
        tk.Label(
            audio_card, text="سطح صدای میکروفون", bg=c["panel"],
            fg=c["text"], font=("Tahoma", 10, "bold")
        ).pack(anchor="e", padx=15, pady=(10, 4))

        self.audio_bar = ttk.Progressbar(audio_card, maximum=100)
        self.audio_bar.pack(fill="x", padx=15, pady=6)

        self.audio_info = tk.Label(
            audio_card, text="RMS: 0.00000\nPeak: 0.000",
            bg=c["panel"], fg=c["muted"], font=("Consolas", 9),
            justify="left", anchor="w"
        )
        self.audio_info.pack(fill="x", padx=15, pady=(0, 10))

        status_card = self._card(
            monitor_row, row=0, column=1, sticky="nsew",
            padx=(8, 0), pady=(0, 10)
        )
        tk.Label(
            status_card, text="وضعیت پردازش", bg=c["panel"],
            fg=c["muted"], font=("Tahoma", 9)
        ).pack(anchor="e", padx=16, pady=(10, 3))

        self.status = tk.Label(
            status_card, text="آماده شروع", bg=c["panel"], fg=c["green"],
            font=("Tahoma", 13, "bold"), anchor="e", justify="right",
            wraplength=650
        )
        self.status.pack(fill="x", padx=16, pady=(0, 6))

        self.metrics = tk.Label(
            status_card, text="Duration: -- | ASR: -- | RTF: --",
            bg=c["panel"], fg=c["muted"], font=("Consolas", 9), anchor="w"
        )
        self.metrics.pack(fill="x", padx=16, pady=(0, 10))

        content = ttk.PanedWindow(monitor_row, orient=tk.HORIZONTAL)
        content.grid(
            row=1, column=0, columnspan=2, sticky="nsew", pady=(0, 10)
        )

        # --- پنل متن استخراج‌شده ---
        transcript_card = tk.Frame(
            content, bg=c["panel"],
            highlightbackground=c["border"], highlightthickness=1
        )
        content.add(transcript_card, weight=1)

        th = tk.Frame(transcript_card, bg=c["panel"])
        th.pack(fill="x", padx=15, pady=(11, 6))
        tk.Label(
            th, text="متن استخراج‌شده", bg=c["panel"], fg=c["text"],
            font=("Tahoma", 11, "bold")
        ).pack(side="right")
        tk.Label(
            th, text="TRANSCRIPT + SPEAKERS", bg=c["panel"],
            fg=c["accent"], font=("Segoe UI", 8, "bold")
        ).pack(side="left")

        # کنترل‌های ترجمه
        tr_controls = tk.Frame(transcript_card, bg=c["panel"])
        tr_controls.pack(fill="x", padx=12, pady=(0, 8))
        tk.Checkbutton(
            tr_controls, text="ترجمه همزمان", variable=self.translation_enabled_var,
            bg=c["panel"], fg=c["text"], selectcolor=c["input"],
            activebackground=c["panel"], activeforeground=c["text"],
            font=("Tahoma", 9), command=self.translation_toggle_changed
        ).pack(side="right", padx=(4, 8))
        tk.Checkbutton(
            tr_controls, text="ترجمه‌ی دقیق‌تر (کندتر)",
            variable=self.translation_accurate_var,
            bg=c["panel"], fg=c["text"], selectcolor=c["input"],
            activebackground=c["panel"], activeforeground=c["text"],
            font=("Tahoma", 9), command=self.translation_accuracy_changed
        ).pack(side="right", padx=(4, 8))
        tk.Label(
            tr_controls, text="زبان مقصد:", bg=c["panel"], fg=c["muted"],
            font=("Tahoma", 9)
        ).pack(side="right")
        self.translation_language_combo = ttk.Combobox(
            tr_controls, textvariable=self.translation_language_var,
            values=list(TRANSLATION_LANGUAGES.keys()), state="readonly", width=16
        )
        self.translation_language_combo.pack(side="right", padx=6)
        self.translation_language_combo.bind(
            "<<ComboboxSelected>>", self.translation_language_changed
        )
        ttk.Button(
            tr_controls, text="بررسی مدل ترجمه",
            command=self.load_translation_model_gui
        ).pack(side="right", padx=5)
        self.translation_status = tk.Label(
            tr_controls, text="ترجمه غیرفعال", bg=c["panel"],
            fg=c["muted"], font=("Tahoma", 8)
        )
        self.translation_status.pack(side="left", padx=6)

        tb = tk.Frame(transcript_card, bg=c["input"])
        tb.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        ts = ttk.Scrollbar(tb, orient="vertical")
        ts.pack(side="right", fill="y")

        self.transcript = tk.Text(
            tb, font=("Tahoma", 12), wrap="word", padx=14, pady=14,
            bg=c["input"], fg=c["text"], insertbackground=c["accent"],
            selectbackground="#245078", selectforeground="#FFFFFF",
            relief="flat", borderwidth=0, undo=True,
            yscrollcommand=ts.set
        )
        self.transcript.pack(side="left", fill="both", expand=True)
        ts.config(command=self.transcript.yview)

        self.transcript.tag_configure(
            "rtl", justify="right", spacing1=5, spacing3=7
        )
        # ترجمه در خط جداگانه: برچسب راست‌چین (فارسی) و متن بسته به جهت زبان مقصد
        self.transcript.tag_configure(
            "translation_label", justify="right", foreground=c["muted"],
            font=("Tahoma", 9), spacing1=2, spacing3=1
        )
        self.transcript.tag_configure(
            "translation_ltr", justify="left", foreground=c["green"],
            font=("Segoe UI", 11), lmargin1=18, lmargin2=18,
            spacing1=1, spacing3=8
        )
        self.transcript.tag_configure(
            "translation_rtl", justify="right", foreground=c["green"],
            font=("Tahoma", 11), rmargin=18, spacing1=1, spacing3=8
        )
        for i, col in enumerate(SPEAKER_COLORS, start=1):
            self.transcript.tag_configure(
                f"spk{i}", foreground=col, font=("Tahoma", 11, "bold")
            )

        # --- پنل تحلیل ---
        analysis_card = tk.Frame(
            content, bg=c["panel"],
            highlightbackground=c["border"], highlightthickness=1
        )
        content.add(analysis_card, weight=1)

        ah = tk.Frame(analysis_card, bg=c["panel"])
        ah.pack(fill="x", padx=15, pady=(11, 6))
        tk.Label(
            ah, text="تحلیل هوشمند", bg=c["panel"], fg=c["text"],
            font=("Tahoma", 11, "bold")
        ).pack(side="right")
        tk.Label(
            ah, text="QWEN ANALYSIS", bg=c["panel"], fg=c["purple"],
            font=("Segoe UI", 8, "bold")
        ).pack(side="left")

        ab = tk.Frame(analysis_card, bg=c["input"])
        ab.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        asb = ttk.Scrollbar(ab, orient="vertical")
        asb.pack(side="right", fill="y")

        self.analysis = tk.Text(
            ab, font=("Tahoma", 12), wrap="word", padx=14, pady=14,
            bg=c["input"], fg="#D7D4FE", insertbackground=c["accent"],
            selectbackground="#493B77", selectforeground="#FFFFFF",
            relief="flat", borderwidth=0, yscrollcommand=asb.set
        )
        self.analysis.pack(side="left", fill="both", expand=True)
        asb.config(command=self.analysis.yview)
        self.analysis.tag_configure(
            "rtl", justify="right", spacing1=5, spacing3=7
        )

        # ---------------- نوار عملیات ----------------
        action_bar = self._card(
            outer, row=5, column=0, sticky="ew", padx=24, pady=(0, 10)
        )

        actions_left = tk.Frame(action_bar, bg=c["panel"])
        actions_left.pack(side="left", padx=12, pady=10)

        actions_right = tk.Frame(action_bar, bg=c["panel"])
        actions_right.pack(side="right", padx=12, pady=10)

        self.start_button = ttk.Button(
            actions_right, text="🎙 شروع تشخیص زنده",
            command=self.start, style="Accent.TButton"
        )
        self.start_button.pack(side="right", padx=4)

        self.stop_button = ttk.Button(
            actions_right, text="■ توقف", command=self.stop,
            state="disabled", style="Stop.TButton"
        )
        self.stop_button.pack(side="right", padx=4)

        self.analyze_button = ttk.Button(
            actions_right, text="✦ تحلیل با Qwen",
            command=self.analyze_transcript, state="disabled"
        )
        self.analyze_button.pack(side="right", padx=4)

        tk.Checkbutton(
            actions_right, text="تشخیص گوینده (زنده)",
            variable=self.live_diar_var, bg=c["panel"], fg=c["text"],
            selectcolor=c["input"], activebackground=c["panel"],
            activeforeground=c["text"], font=("Tahoma", 9)
        ).pack(side="right", padx=10)

        tk.Checkbutton(
            actions_right, text="بازتفکیک دقیق بعد از توقف",
            variable=self.refine_var, bg=c["panel"], fg=c["text"],
            selectcolor=c["input"], activebackground=c["panel"],
            activeforeground=c["text"], font=("Tahoma", 9)
        ).pack(side="right", padx=10)

        self.clear_button = ttk.Button(
            actions_left, text="پاک کردن متن", command=self.clear
        )
        self.clear_button.pack(side="left", padx=4)

        self.save_button = ttk.Button(
            actions_left, text="💾 ذخیره متن", command=self.save_transcript
        )
        self.save_button.pack(side="left", padx=4)

        self.download_models_button = ttk.Button(
            actions_left, text="⬇ دانلود مدل گوینده",
            command=self.download_speaker_models_gui
        )
        self.download_models_button.pack(side="left", padx=4)

        self.download_translation_button = ttk.Button(
            actions_left, text="⬇ دانلود مدل ترجمه",
            command=self.download_translation_model_gui
        )
        self.download_translation_button.pack(side="left", padx=4)

        # ---------------- فوتر ----------------
        footer = tk.Frame(outer, bg=c["bg"])
        footer.grid(row=6, column=0, sticky="ew", padx=26, pady=(0, 10))

        tk.Label(
            footer, text="SHENAVA AI • LOCAL PROCESSING", bg=c["bg"],
            fg="#64748B", font=("Segoe UI", 8)
        ).pack(side="left")
        tk.Label(
            footer, text="Persian Speech Recognition", bg=c["bg"],
            fg="#64748B", font=("Segoe UI", 8)
        ).pack(side="right")

    # ========================================================
    # دانلود مدل‌های تشخیص گوینده
    # ========================================================

    def offer_download_models(self):
        """پس از شروع برنامه، در صورت نبود مدل‌ها پیشنهاد دانلود می‌دهد."""
        if running or file_processing:
            return
        if messagebox.askyesno(
            "مدل تشخیص گوینده",
            "مدل‌های تشخیص گوینده (pyannote 3.0 و WeSpeaker ResNet293) "
            "هنوز دانلود نشده‌اند.\n\n"
            "حجم کل حدود ۲۵۰ مگابایت است. همین الان دانلود شود؟"
        ):
            self.download_speaker_models_gui()

    def download_speaker_models_gui(self):
        """دانلود مدل‌ها را در یک ترد پس‌زمینه اجرا می‌کند."""
        if running or file_processing:
            return

        self.download_models_button.config(state="disabled")
        self.header_status.config(
            text="● DOWNLOADING MODELS", fg=self.colors["purple"]
        )
        self.status.config(
            text="در حال دانلود مدل‌های تشخیص گوینده...",
            fg=self.colors["purple"]
        )

        def ui(*a):
            self.root.after(0, *a)

        def work():
            try:
                download_speaker_models(lambda msg: ui(self.set_status, msg))
                ui(self._models_downloaded, None)
            except Exception as exc:
                ui(self._models_downloaded, str(exc))

        threading.Thread(target=work, daemon=True).start()

    def _models_downloaded(self, error):
        """نتیجه‌ی دانلود را در رابط کاربری اعمال می‌کند (ترد اصلی)."""
        self.download_models_button.config(state="normal")

        if error:
            self.header_status.config(text="● ERROR", fg=self.colors["red"])
            self.status.config(
                text="دانلود مدل گوینده ناموفق بود.", fg=self.colors["red"]
            )
            messagebox.showerror("دانلود مدل", error)
            return

        resolve_speaker_models()
        self.speaker_extractor = None
        self.live_diar_var.set(True)
        self.file_diar_var.set(True)
        self.header_status.config(text="● SYSTEM READY", fg=self.colors["green"])
        self.status.config(
            text="مدل‌های تشخیص گوینده آماده شدند.", fg=self.colors["green"]
        )

    # ========================================================
    # وضعیت دکمه‌ها
    # ========================================================

    def refresh_buttons(self):
        """فعال/غیرفعال بودن دکمه‌ها را بر اساس وضعیت فعلی برنامه هماهنگ می‌کند."""
        busy = running or file_processing

        self.start_button.config(
            state="disabled" if busy else "normal"
        )
        self.stop_button.config(
            state="normal" if running else "disabled"
        )
        self.process_file_button.config(
            state="disabled" if busy else "normal"
        )
        self.browse_button.config(
            state="disabled" if busy else "normal"
        )

        can_analyze = (
            bool(transcript_segments)
            and qwen is not None
            and not busy
            and not qwen_analyzing
        )
        self.analyze_button.config(
            state="normal" if can_analyze else "disabled"
        )

    # ========================================================
    # انتخاب مدل Ollama
    # ========================================================

    def select_qwen_model(self):
        """فهرست مدل‌های نصب‌شده در Ollama را نشان می‌دهد تا کاربر یکی را انتخاب کند."""
        global QWEN_MODEL_NAME

        if qwen is not None:
            messagebox.showinfo(
                "Ollama",
                "مدل قبلاً بررسی شده است. برای تغییر مدل، "
                "نام آن را در تنظیمات کد عوض کن و برنامه را دوباره اجرا کن."
            )
            return

        try:
            names = sorted(ollama_installed_models())

            if not names:
                messagebox.showwarning("Ollama", "هیچ مدلی در Ollama پیدا نشد.")
                return

            dialog = tk.Toplevel(self.root)
            dialog.title("انتخاب مدل Ollama")
            dialog.configure(bg=self.colors["panel"])
            dialog.geometry("480x220")
            dialog.transient(self.root)
            dialog.grab_set()

            tk.Label(
                dialog, text="مدل موردنظر را انتخاب کن",
                bg=self.colors["panel"], fg=self.colors["text"],
                font=("Tahoma", 11, "bold")
            ).pack(padx=20, pady=(20, 10))

            choice = tk.StringVar(value=(
                QWEN_MODEL_NAME if QWEN_MODEL_NAME in names else names[0]
            ))

            ttk.Combobox(
                dialog, textvariable=choice, values=names,
                state="readonly", font=("Consolas", 10)
            ).pack(fill="x", padx=20, pady=8)

            def apply_model():
                global QWEN_MODEL_NAME
                selected = choice.get().strip()
                if selected:
                    QWEN_MODEL_NAME = selected
                    self.model_label.config(
                        text=f"مدل: {QWEN_MODEL_NAME}   |   آدرس: {OLLAMA_URL}"
                    )
                    self.model_status.config(
                        text="مدل انتخاب شده؛ آماده اتصال",
                        fg=self.colors["yellow"]
                    )
                dialog.destroy()

            ttk.Button(
                dialog, text="تأیید انتخاب", command=apply_model,
                style="Accent.TButton"
            ).pack(pady=12)

        except Exception as exc:
            messagebox.showerror("خطا در دریافت مدل‌ها", str(exc))

    # ========================================================
    # اتصال به Ollama
    # ========================================================

    def load_qwen_from_gui(self):
        """بررسی اتصال به Ollama را در ترد پس‌زمینه شروع می‌کند."""
        if qwen is not None:
            messagebox.showinfo("Ollama", "اتصال به Ollama برقرار است.")
            return

        self.model_status.config(
            text="در حال بررسی Ollama...", fg=self.colors["yellow"]
        )
        self.load_model_button.config(state="disabled")
        self.select_model_button.config(state="disabled")

        threading.Thread(target=self.qwen_loader_thread, daemon=True).start()

    def qwen_loader_thread(self):
        """ترد بررسی اتصال؛ نتیجه را به ترد اصلی برمی‌گرداند."""
        try:
            load_qwen()
            self.root.after(0, self.qwen_loaded)
        except Exception as exc:
            self.root.after(0, self.show_qwen_load_error, str(exc))

    def qwen_loaded(self):
        self.model_status.config(
            text=f"● Ollama متصل است: {QWEN_MODEL_NAME}",
            fg=self.colors["green"]
        )
        self.load_model_button.config(state="normal")
        self.select_model_button.config(state="normal")

        self.header_status.config(text="● OLLAMA READY", fg=self.colors["green"])
        self.status.config(
            text="Ollama آماده است؛ می‌توانی متن را تحلیل کنی.",
            fg=self.colors["green"]
        )
        self.refresh_buttons()

    def show_qwen_load_error(self, error):
        self.model_status.config(
            text="● خطا در اتصال به Ollama", fg=self.colors["red"]
        )
        self.load_model_button.config(state="normal")
        self.select_model_button.config(state="normal")
        self.status.config(
            text="اتصال به Ollama ناموفق بود.", fg=self.colors["red"]
        )
        messagebox.showerror("خطا در اتصال به Ollama", error)

    # ========================================================
    # ترجمه (TranslateGemma / NLLB)
    # ========================================================

    def _set_translation_status(self, text, color_key="muted"):
        """به‌روزرسانی امن برچسب وضعیت ترجمه (فقط از ترد اصلی)."""
        try:
            self.translation_status.config(text=text, fg=self.colors[color_key])
        except tk.TclError:
            pass

    def translation_toggle_changed(self):
        self.translation_enabled = bool(self.translation_enabled_var.get())
        if self.translation_enabled:
            self._set_translation_status(
                "ترجمه فعال؛ در حال بررسی مدل...", "yellow"
            )
            self.load_translation_model_gui()
        else:
            self._set_translation_status("ترجمه غیرفعال", "muted")

    def translation_accuracy_changed(self):
        global TRANSLATION_HIGH_ACCURACY
        TRANSLATION_HIGH_ACCURACY = bool(self.translation_accurate_var.get())

    def translation_language_changed(self, event=None):
        self.translation_target_name = self.translation_language_var.get()

    def load_translation_model_gui(self):
        """مدل ترجمه‌ی موجود در Ollama را پیدا و آن را گرم (warm-up) می‌کند."""
        self.translation_enabled_var.set(True)
        self.translation_enabled = True
        self.translation_target_name = self.translation_language_var.get()
        self._set_translation_status("در حال بررسی مدل ترجمه...", "yellow")
        threading.Thread(target=self._load_translation_model_thread, daemon=True).start()

    def _load_translation_model_thread(self):
        ui = lambda *a: self.root.after(0, *a)
        try:
            model = pick_translation_model(force_refresh=True)
            if model is None:
                ui(self._set_translation_status,
                   "مدل ترجمه نصب نیست؛ دکمه «دانلود مدل ترجمه» را بزن", "yellow")
                return
            ui(self._set_translation_status, f"بارگذاری {model}...", "yellow")
            # ترجمه‌ی کوتاه برای بارگذاری مدل در حافظه‌ی GPU
            translate_persian_text("سلام", self.translation_target_name)
            note = "" if _is_translategemma(model) else " (مدل عمومی؛ TranslateGemma پیشنهاد می‌شود)"
            ui(self._set_translation_status,
               f"مدل ترجمه آماده: {model}{note}", "green")
        except Exception as exc:
            ui(self._set_translation_status, "خطا در بارگذاری مدل ترجمه", "red")
            ui(messagebox.showerror, "مدل ترجمه", str(exc))

    def download_translation_model_gui(self):
        """مدل TRANSLATION_PULL_MODEL را با Ollama دانلود می‌کند."""
        self.download_translation_button.config(state="disabled")
        self._set_translation_status(
            f"دانلود {TRANSLATION_PULL_MODEL} شروع شد...", "purple"
        )

        ui = lambda *a: self.root.after(0, *a)

        def work():
            try:
                ollama_pull(
                    TRANSLATION_PULL_MODEL,
                    lambda msg: ui(self._set_translation_status, msg, "purple")
                )
                reset_translation_model_cache()
                ui(self._translation_model_downloaded, None)
            except Exception as exc:
                ui(self._translation_model_downloaded, str(exc))

        threading.Thread(target=work, daemon=True).start()

    def _translation_model_downloaded(self, error):
        self.download_translation_button.config(state="normal")
        if error:
            self._set_translation_status("دانلود مدل ترجمه ناموفق بود", "red")
            messagebox.showerror(
                "دانلود مدل ترجمه",
                f"{error}\n\nمطمئن شو Ollama در حال اجراست و در ترمینال بزن:\n"
                f"ollama pull {TRANSLATION_PULL_MODEL}"
            )
        else:
            self._set_translation_status(
                f"{TRANSLATION_PULL_MODEL} دانلود شد", "green"
            )
            if self.translation_enabled:
                self.load_translation_model_gui()

    def enqueue_translation(self, index, seg):
        # از تردهای ASR فراخوانی می‌شود؛ فقط از متغیرهای معمولی استفاده کن.
        if not self.translation_enabled:
            return
        target_name = self.translation_target_name
        if target_name not in TRANSLATION_LANGUAGES:
            return
        prev = None
        if index >= 2 and index - 2 < len(transcript_segments):
            prev = transcript_segments[index - 2].get("text")
        self.translation_queue.put((index, seg, target_name, prev))

    def translation_worker(self):
        """ترد ترجمه: صف را ترتیبی پردازش می‌کند (یک ترجمه در هر لحظه)."""
        ui = lambda *a: self.root.after(0, *a)
        while True:
            item = self.translation_queue.get()
            if item is None:
                break
            index, seg, target_name, prev = item
            try:
                ui(self._set_translation_status,
                   f"در حال ترجمه به {target_name}...", "yellow")
                translated, engine = translate_persian_text(
                    seg.get("text", ""), target_name, prev
                )
                seg["translation"] = translated
                seg["translation_language"] = target_name
                seg["translation_engine"] = engine
                ui(self.add_translation_to_view, index, seg)
                ui(self._set_translation_status,
                   f"ترجمه آماده: {target_name} ({engine})", "green")
            except Exception as exc:
                ui(self._set_translation_status, "خطا در ترجمه", "red")
                ui(self.set_status, f"خطای ترجمه: {exc}")

    def add_translation_to_view(self, index, seg):
        """متن را بازترسیم می‌کند تا ترجمه دقیقاً زیر قطعه‌ی مربوط بیاید."""
        self.rerender_transcript()

    # ========================================================
    # حالت زنده: شروع
    # ========================================================

    def start(self):
        """
        جلسه‌ی تشخیص زنده را راه‌اندازی می‌کند: Shenava -> ردیاب گوینده ->
        VAD تازه -> خالی کردن صف -> اجرای segment_worker و audio_worker.
        """
        global running, vad

        if running or file_processing:
            return

        try:
            if recognizer is None or vad is None:
                self.status.config(
                    text="در حال آماده‌سازی Shenava...",
                    fg=self.colors["yellow"]
                )
                self.root.update_idletasks()
                load_shenava()
        except Exception as exc:
            self.show_error(str(exc))
            return

        self.tracker = None
        if self.live_diar_var.get():
            try:
                if self.speaker_extractor is None:
                    self.status.config(
                        text="در حال بارگذاری مدل تشخیص گوینده...",
                        fg=self.colors["yellow"]
                    )
                    self.root.update_idletasks()
                    self.speaker_extractor = create_speaker_extractor()

                try:
                    fixed_k = int(self.num_speakers_var.get())
                except Exception:
                    fixed_k = 0

                self.tracker = LiveSpeakerTracker(
                    self.speaker_extractor,
                    SPEAKER_THRESHOLD,
                    MAX_LIVE_SPEAKERS,
                    fixed_k
                )
            except Exception as exc:
                messagebox.showwarning(
                    "تشخیص گوینده غیرفعال شد",
                    f"{exc}\n\nتشخیص گفتار بدون تشخیص گوینده ادامه می‌یابد."
                )
                self.tracker = None

        try:
            vad = make_vad()
        except Exception as exc:
            self.show_error(str(exc))
            return

        while True:
            try:
                audio_queue.get_nowait()
            except queue.Empty:
                break

        self.segment_queue = queue.Queue()
        self.session_start = time.monotonic()

        running = True
        self.refresh_buttons()

        self.header_status.config(text="● LISTENING", fg=self.colors["green"])
        self.status.config(
            text="در حال گوش دادن و استخراج متن...", fg=self.colors["green"]
        )

        threading.Thread(
            target=self.segment_worker,
            args=(self.segment_queue,),
            daemon=True
        ).start()

        threading.Thread(target=self.audio_worker, daemon=True).start()

    # ========================================================
    # حالت زنده: توقف
    # ========================================================

    def stop(self):
        """ضبط را متوقف می‌کند؛ segment_worker صف را تخلیه و on_live_finished را اجرا می‌کند."""
        global running
        running = False

        self.header_status.config(text="● STOPPED", fg=self.colors["yellow"])

        if transcript_segments:
            self.status.config(
                text="استخراج متوقف شد؛ می‌توانی متن را تحلیل کنی.",
                fg=self.colors["yellow"]
            )
        else:
            self.status.config(
                text="استخراج متوقف شد؛ متنی ثبت نشده است.",
                fg=self.colors["muted"]
            )
        self.refresh_buttons()

    # ========================================================
    # حالت زنده: ترد صدا
    # ========================================================

    def _drain_vad(self, q, hist):
        """
        قطعه‌های آماده‌ی VAD را تخلیه و در صف رونویسی می‌گذارد.
        هر آیتم: ``(audio, t_start, t_end, core)``؛ audio با padding و core بدون آن.
        """
        pad = int(VAD_PAD_SEC * ASR_SAMPLE_RATE)

        while not vad.empty():
            segment = vad.front
            samples = np.asarray(segment.samples, dtype=np.float32)
            start = int(getattr(segment, "start", -1))
            vad.pop()

            if len(samples) < int(MIN_SPEECH_DURATION * ASR_SAMPLE_RATE):
                continue

            audio = samples
            t_start = None

            if hist is not None and start >= 0:
                cut = hist.cut(start - pad, start + len(samples) + pad)
                if cut is not None and len(cut) >= len(samples):
                    audio = cut
                    t_start = start / ASR_SAMPLE_RATE

            if t_start is None:
                end_t = time.monotonic() - self.session_start
                t_start = max(0.0, end_t - len(samples) / ASR_SAMPLE_RATE)
            end_t = t_start + len(samples) / ASR_SAMPLE_RATE

            q.put((audio, t_start, end_t, samples))

    def audio_worker(self):
        """ترد ضبط: میکروفون -> resample -> بالاگذر -> VAD -> صف قطعه‌ها."""
        q = self.segment_queue
        resample_buffer = np.zeros(0, dtype=np.float32)
        hp = HighPass()
        hist = AudioHistory(45)
        rec = []
        self.rec_chunks = rec

        try:
            if vad is None:
                raise RuntimeError("VAD آماده نیست.")

            with sd.InputStream(
                samplerate=MIC_SAMPLE_RATE,
                device=DEVICE_INDEX,
                channels=CHANNELS,
                dtype="float32",
                blocksize=int(MIC_SAMPLE_RATE * 0.1),
                callback=audio_callback
            ):
                while running:
                    try:
                        audio = audio_queue.get(timeout=0.2)
                    except queue.Empty:
                        continue

                    audio16 = resample_audio(
                        audio, MIC_SAMPLE_RATE, ASR_SAMPLE_RATE
                    )
                    if len(audio16) == 0:
                        continue

                    audio16 = hp(audio16)
                    hist.add(audio16)
                    rec.append(
                        (np.clip(audio16, -1.0, 1.0) * 32767).astype(np.int16)
                    )

                    resample_buffer = np.concatenate(
                        (resample_buffer, audio16)
                    )

                    while len(resample_buffer) >= VAD_WINDOW and running:
                        frame = resample_buffer[:VAD_WINDOW]
                        resample_buffer = resample_buffer[VAD_WINDOW:]
                        vad.accept_waveform(frame)
                        self._drain_vad(q, hist)

            try:
                vad.flush()
                self._drain_vad(q, hist)
            except Exception:
                pass

        except Exception as exc:
            if running:
                self.root.after(0, self.show_error, str(exc))
        finally:
            q.put(None)

    # ========================================================
    # حالت زنده: ترد رونویسی
    # ========================================================

    def segment_worker(self, q):
        """قطعه‌ها را ترتیبی رونویسی می‌کند (هیچ قطعه‌ای حذف نمی‌شود)."""
        while True:
            item = q.get()
            if item is None:
                break
            audio, t_start, t_end, core = item
            try:
                self.process_live_segment(audio, t_start, t_end, core)
            except Exception as exc:
                self.root.after(0, self.set_status, f"خطا: {exc}")

        self.root.after(0, self.on_live_finished)

    def on_live_finished(self):
        """در صورت فعال بودن گزینه، تفکیک دقیق گوینده را روی کل صدای ضبط‌شده اجرا می‌کند."""
        global file_processing

        chunks = self.rec_chunks
        self.rec_chunks = []

        if (
            running
            or file_processing
            or not self.refine_var.get()
            or self.tracker is None
            or not transcript_segments
            or not chunks
        ):
            self.refresh_buttons()
            return

        audio = np.concatenate(chunks).astype(np.float32) / 32767.0
        if len(audio) < 2 * ASR_SAMPLE_RATE:
            self.refresh_buttons()
            return

        try:
            num_spk = int(self.num_speakers_var.get())
        except Exception:
            num_spk = 0
        method = self.diar_method_var.get()

        file_processing = True
        self.header_status.config(
            text="● REFINING SPEAKERS", fg=self.colors["purple"]
        )
        self.refresh_buttons()

        threading.Thread(
            target=self.refine_thread, args=(audio, num_spk, method),
            daemon=True
        ).start()

    def refine_thread(self, audio, num_spk, method):
        """تفکیک دقیق گوینده در پس‌زمینه؛ نتیجه با هم‌پوشانی زمانی به قطعه‌ها نسبت داده می‌شود."""
        def ui(*a):
            self.root.after(0, *a)

        try:
            ui(self.set_status, "بازتفکیک دقیق گویندگان روی صدای ضبط‌شده...")

            def prog(done, total):
                if total:
                    ui(self.set_status,
                       f"بازتفکیک گویندگان: {int(100 * done / total)}٪")

            diar = diarize(audio, num_spk, method, prog)
            if not diar:
                raise RuntimeError("گفتاری برای تفکیک پیدا نشد.")
            assign_speakers_by_overlap(transcript_segments, diar)
            ui(self.refine_done, None)
        except Exception as exc:
            ui(self.refine_done, str(exc))

    def refine_done(self, error):
        """پایان بازتفکیک؛ در خطا برچسب‌های زنده حفظ می‌شوند."""
        global file_processing
        file_processing = False

        if error:
            self.header_status.config(text="● STOPPED", fg=self.colors["yellow"])
            self.status.config(
                text=f"بازتفکیک انجام نشد (برچسب‌های زنده حفظ شد): {error}",
                fg=self.colors["yellow"]
            )
        else:
            self.rerender_transcript()
            self.header_status.config(text="● SPEAKERS REFINED", fg=self.colors["green"])
            self.status.config(
                text="گویندگان با تفکیک دقیق به‌روز شدند؛ می‌توانی متن را تحلیل کنی.",
                fg=self.colors["green"]
            )
        self.refresh_buttons()

    def rerender_transcript(self):
        """
        کل متن را از ``transcript_segments`` دوباره می‌نویسد، بدون پراندن اسکرول:
        اگر کاربر انتهای متن بود به انتها می‌رود، وگرنه موقعیت قبلی حفظ می‌شود.
        """
        top, bottom = self.transcript.yview()
        at_end = bottom >= 0.999

        self.transcript.delete("1.0", "end")
        for i, seg in enumerate(list(transcript_segments), start=1):
            self.add_transcript(i, seg, autoscroll=False, refresh=False)

        if at_end:
            self.transcript.see("end")
        else:
            self.transcript.yview_moveto(top)
        self.refresh_buttons()

    def process_live_segment(self, audio, t_start, t_end, core=None):
        """یک قطعه‌ی زنده را رونویسی و (در صورت فعال بودن) گوینده‌اش را تشخیص می‌دهد."""
        self.root.after(
            0, self.set_status, "Shenava در حال تبدیل صوت به متن..."
        )

        start = time.perf_counter()

        stream = recognizer.create_stream()
        stream.accept_waveform(
            ASR_SAMPLE_RATE, np.asarray(audio, dtype=np.float32)
        )
        recognizer.decode_stream(stream)

        text = stream.result.text.strip()

        asr_time = time.perf_counter() - start
        duration = len(audio) / ASR_SAMPLE_RATE

        if not text:
            return

        speaker = None
        if self.tracker is not None:
            try:
                speaker = self.tracker.identify(
                    core if core is not None else audio
                )
            except Exception as exc:
                print("Speaker error:", exc)

        seg = {
            "speaker": speaker,
            "text": text,
            "start": t_start,
            "end": t_end
        }
        transcript_segments.append(seg)
        idx = len(transcript_segments)

        self.root.after(0, self.add_transcript, idx, seg)
        self.enqueue_translation(idx, seg)

        rtf = asr_time / max(duration, 0.001)
        self.root.after(0, self.set_metrics, duration, asr_time, rtf)
        self.root.after(0, self.refresh_buttons)

        if running:
            self.root.after(
                0, self.set_status, "در حال گوش دادن و استخراج متن..."
            )

    # ========================================================
    # حالت فایل
    # ========================================================

    def browse_file(self):
        """پنجره‌ی انتخاب فایل صوتی/تصویری را باز می‌کند."""
        path = filedialog.askopenfilename(
            title="انتخاب فایل صوتی یا تصویری",
            filetypes=[
                ("Audio / Video",
                 "*.wav *.mp3 *.m4a *.flac *.ogg *.opus *.aac *.wma "
                 "*.mp4 *.mkv *.mov *.webm *.avi"),
                ("All files", "*.*"),
            ]
        )
        if path:
            self.file_path_var.set(path)

    def process_file(self):
        """تنظیمات رابط را می‌خواند و پردازش فایل را در یک ترد جدا آغاز می‌کند."""
        global file_processing

        if running or file_processing:
            return

        path = self.file_path_var.get().strip().strip('"')

        if not path or not os.path.isfile(path):
            messagebox.showwarning("فایل", "یک فایل معتبر انتخاب کن.")
            return

        engine = self.engine_var.get()
        diar_method = self.diar_method_var.get()
        whisper_name = self.whisper_var.get().strip() or DEFAULT_WHISPER_MODEL
        use_diar = bool(self.file_diar_var.get())

        try:
            num_spk = int(self.num_speakers_var.get())
        except Exception:
            num_spk = 0

        if transcript_segments:
            if not messagebox.askyesno(
                "متن موجود",
                "متن قبلی پاک شود و فایل جدید پردازش شود؟"
            ):
                return
            self._clear_all()

        file_processing = True
        self.header_status.config(text="● PROCESSING FILE", fg=self.colors["purple"])
        self.refresh_buttons()

        threading.Thread(
            target=self.file_thread,
            args=(path, engine, whisper_name, use_diar, num_spk, diar_method),
            daemon=True
        ).start()

    def file_thread(self, path, engine, whisper_name, use_diar, num_spk,
                    diar_method=DEFAULT_DIAR_METHOD):
        """
        پردازش کامل فایل.

        مسیر ۱ (با گوینده): تفکیک -> ادغام قطعات -> رونویسی هر قطعه.
        مسیر ۲ (بدون گوینده): Whisper با VAD داخلی، یا VAD + Shenava.
        """
        t0 = time.perf_counter()

        try:
            ui = lambda *a: self.root.after(0, *a)

            ui(self.set_status, "در حال خواندن فایل...")
            audio = load_audio_file(path)
            total_dur = len(audio) / ASR_SAMPLE_RATE

            if total_dur < 0.5:
                raise RuntimeError("فایل صوتی خالی یا بسیار کوتاه است.")

            whisper = None
            if engine == "Whisper":
                ui(self.set_status,
                   f"در حال بارگذاری Whisper ({whisper_name})...")
                whisper = load_whisper(whisper_name)
            else:
                ui(self.set_status, "در حال بارگذاری Shenava...")
                load_shenava()

            def add(seg):
                transcript_segments.append(seg)
                idx = len(transcript_segments)
                ui(self.add_transcript, idx, seg)
                self.enqueue_translation(idx, seg)

            diar_segments = None
            diar_failed = None

            if use_diar:
                try:
                    ui(self.set_status, "در حال تشخیص گویندگان...")

                    def prog(done, total):
                        if total:
                            ui(self.set_status,
                               f"تشخیص گویندگان: {int(100 * done / total)}٪")

                    diar_segments = diarize(audio, num_spk, diar_method, prog)
                    diar_segments = merge_diarization_segments(
                        diar_segments,
                        max_len=28.0 if whisper is not None else 20.0
                    )
                except Exception as exc:
                    diar_failed = str(exc)
                    diar_segments = None

            if diar_segments:
                n = len(diar_segments)
                for i, (s, e, spk) in enumerate(diar_segments, start=1):
                    ui(self.set_status,
                       f"رونویسی بخش {i} از {n} "
                       f"({speaker_label(spk)})...")

                    pad = VAD_PAD_SEC
                    a = max(0, int((s - pad) * ASR_SAMPLE_RATE))
                    b = min(len(audio), int((e + pad) * ASR_SAMPLE_RATE))
                    chunk = audio[a:b]

                    if len(chunk) < int(0.3 * ASR_SAMPLE_RATE):
                        continue

                    if whisper is not None:
                        parts = whisper_transcribe(whisper, chunk)
                        text = " ".join(p[2] for p in parts).strip()
                    else:
                        text = shenava_transcribe(chunk)

                    if text:
                        add({
                            "speaker": spk, "text": text,
                            "start": s, "end": e
                        })
            else:
                if diar_failed:
                    ui(self.set_status,
                       "تشخیص گوینده انجام نشد؛ رونویسی بدون گوینده...")
                    ui(lambda: messagebox.showwarning(
                        "تشخیص گوینده", diar_failed))

                if whisper is not None:
                    ui(self.set_status, "Whisper در حال رونویسی...")
                    for s, e, t in whisper_transcribe(
                        whisper, audio, vad_filter=True
                    ):
                        add({"speaker": None, "text": t,
                             "start": s, "end": e})
                else:
                    ui(self.set_status, "تشخیص بخش‌های گفتار (VAD)...")
                    vsegs = offline_vad_segments(audio)
                    n = len(vsegs)
                    for i, (s, e) in enumerate(vsegs, start=1):
                        ui(self.set_status,
                           f"Shenava: رونویسی بخش {i} از {n}...")
                        a = int(s * ASR_SAMPLE_RATE)
                        b = int(e * ASR_SAMPLE_RATE)
                        text = shenava_transcribe(audio[a:b])
                        if text:
                            add({"speaker": None, "text": text,
                                 "start": s, "end": e})

            elapsed = time.perf_counter() - t0
            ui(self.set_metrics, total_dur, elapsed, elapsed / max(total_dur, 0.001))
            ui(self.file_done, None)

        except Exception as exc:
            self.root.after(0, self.file_done, str(exc))

    def file_done(self, error):
        """پایان پردازش فایل؛ وضعیت رابط را بر اساس نتیجه به‌روز می‌کند."""
        global file_processing
        file_processing = False

        if error:
            self.header_status.config(text="● ERROR", fg=self.colors["red"])
            self.status.config(text="پردازش فایل ناموفق بود.", fg=self.colors["red"])
            messagebox.showerror("خطا در پردازش فایل", error)
        else:
            self.header_status.config(text="● FILE DONE", fg=self.colors["green"])
            if transcript_segments:
                self.status.config(
                    text="پردازش فایل تمام شد؛ می‌توانی متن را تحلیل کنی.",
                    fg=self.colors["green"]
                )
            else:
                self.status.config(
                    text="پردازش تمام شد اما گفتاری پیدا نشد.",
                    fg=self.colors["yellow"]
                )
        self.refresh_buttons()

    # ========================================================
    # تحلیل متن
    # ========================================================

    def analyze_transcript(self):
        """متن فعلی را در یک ترد پس‌زمینه برای تحلیل به Qwen می‌فرستد."""
        global qwen_analyzing

        if running:
            messagebox.showwarning(
                "استخراج فعال است", "ابتدا استخراج صوت را متوقف کن."
            )
            return

        if file_processing or qwen_analyzing:
            return

        if qwen is None:
            messagebox.showwarning(
                "Ollama آماده نیست",
                "ابتدا روی «اتصال به Ollama» کلیک کن."
            )
            return

        if not transcript_segments:
            messagebox.showwarning("متن خالی", "متنی برای تحلیل وجود ندارد.")
            return

        speakers_present = has_multiple_speakers()
        full_text = format_transcript_for_llm(with_time=True)

        qwen_analyzing = True
        self.refresh_buttons()

        self.analysis.delete("1.0", "end")
        self.analysis.insert("1.0", "در حال تحلیل متن با Ollama...\n", "rtl")

        self.status.config(
            text="Qwen در حال تحلیل متن...", fg=self.colors["purple"]
        )

        threading.Thread(
            target=self.qwen_analysis_thread,
            args=(full_text, speakers_present),
            daemon=True
        ).start()

    def qwen_analysis_thread(self, text, speakers_present):
        """ترد تحلیل؛ نتیجه یا خطا را به ترد اصلی برمی‌گرداند."""
        try:
            analysis, elapsed = analyze_with_qwen(text, speakers_present)
            self.root.after(0, self.show_analysis, analysis, elapsed)
        except Exception as exc:
            self.root.after(0, self.show_analysis_error, str(exc))

    def show_analysis(self, analysis, elapsed):
        global qwen_analyzing
        qwen_analyzing = False

        self.analysis.delete("1.0", "end")
        self.analysis.insert("1.0", clean_qwen_output(analysis), "rtl")
        self.analysis.see("1.0")

        self.metrics.config(
            text=f"{self.metrics.cget('text')}  |  Qwen: {elapsed:.2f}s"
        )
        self.status.config(
            text="تحلیل Qwen انجام شد.", fg=self.colors["green"]
        )
        self.refresh_buttons()

    def show_analysis_error(self, error):
        global qwen_analyzing
        qwen_analyzing = False

        self.analysis.delete("1.0", "end")
        self.analysis.insert(
            "1.0", "خطا در تحلیل با Ollama:\n\n" + error, "rtl"
        )
        self.status.config(text="خطا در تحلیل Qwen", fg=self.colors["red"])
        self.refresh_buttons()

    # ========================================================
    # نمایش متن
    # ========================================================

    def add_transcript(self, index, seg, autoscroll=True, refresh=True):
        """
        یک قطعه را به انتهای متن اضافه می‌کند.

        قالب: ``[شماره] گوینده N (mm:ss): متن``. ترجمه (در صورت وجود) در دو
        خط جدا می‌آید: برچسب راست‌چین «ترجمه (زبان)»، سپس متن ترجمه با جهت
        مناسب زبان مقصد تا علائم و پرانتزها به‌هم نریزند.
        """
        spk = seg.get("speaker")
        head = f"[{index}] "

        if spk is not None:
            head += speaker_label(spk)
        if seg.get("start") is not None:
            head += f" ({fmt_time(seg['start'])})"
        head += ": "

        tags = ["rtl"]
        if spk is not None:
            tags.append(f"spk{((spk - 1) % len(SPEAKER_COLORS)) + 1}")

        self.transcript.insert("end", head, tuple(tags))
        self.transcript.insert("end", f"{seg['text']}\n", "rtl")

        if seg.get("translation"):
            lang = seg.get("translation_language", "")
            engine = seg.get("translation_engine", "")
            label = f"↲ ترجمه ({lang})"
            if engine and engine != "-":
                label += f" · {engine}"
            self.transcript.insert("end", label + "\n", "translation_label")
            self.transcript.insert(
                "end", seg["translation"] + "\n",
                "translation_rtl" if is_rtl_language(lang) else "translation_ltr"
            )
        self.transcript.insert("end", "\n")

        if autoscroll:
            self.transcript.see("end")
        if refresh:
            self.refresh_buttons()

    # ========================================================
    # شاخص‌ها / وضعیت / سطح صدا
    # ========================================================

    def set_metrics(self, duration, asr_time, rtf):
        """نمایش مدت صدا، زمان پردازش و RTF."""
        self.metrics.config(
            text=(
                f"Duration: {duration:.2f}s  |  "
                f"ASR: {asr_time:.3f}s  |  "
                f"RTF: {rtf:.3f}"
            )
        )

    def update_audio_level(self):
        """هر ۱۰۰ میلی‌ثانیه نوار و اعداد سطح صدای میکروفون را به‌روز می‌کند."""
        try:
            rms = float(last_rms)
            peak = float(last_peak)

            self.audio_bar["value"] = min(100, max(0, rms * 300))
            db = 20 * np.log10(rms + 1e-12)

            self.audio_info.config(
                text=f"RMS: {rms:.5f}\nPeak: {peak:.3f} | {db:.1f} dB"
            )
        except Exception:
            pass

        try:
            self.root.after(100, self.update_audio_level)
        except tk.TclError:
            pass

    def set_status(self, text):
        """متن وضعیت را تغییر می‌دهد (در صورت بسته شدن پنجره، خطا نادیده گرفته می‌شود)."""
        try:
            self.status.config(text=text)
        except tk.TclError:
            pass

    def show_error(self, error):
        """خطا را نمایش می‌دهد و ضبط زنده را متوقف می‌کند."""
        global running
        running = False

        try:
            self.header_status.config(text="● ERROR", fg=self.colors["red"])
            self.status.config(text="خطا رخ داد", fg=self.colors["red"])
            self.refresh_buttons()
            messagebox.showerror("خطا", error)
        except tk.TclError:
            pass

    # ========================================================
    # پاک‌سازی / ذخیره
    # ========================================================

    def _clear_all(self):
        """متن، تحلیل، گویندگان شناخته‌شده، صف ترجمه و شاخص‌ها را پاک می‌کند."""
        transcript_segments.clear()
        if self.tracker is not None:
            self.tracker.reset()

        # ترجمه‌های معوقِ متن قبلی نباید روی متن جدید بنشینند
        try:
            while True:
                self.translation_queue.get_nowait()
        except queue.Empty:
            pass

        self.transcript.delete("1.0", "end")
        self.analysis.delete("1.0", "end")

        self.metrics.config(text="Duration: -- | ASR: -- | RTF: --")
        self.refresh_buttons()

    def clear(self):
        """پاک کردن متن به درخواست کاربر (در حین پردازش مجاز نیست)."""
        if running or file_processing:
            messagebox.showwarning(
                "پردازش فعال است", "ابتدا پردازش را متوقف کن یا صبر کن تمام شود."
            )
            return

        self._clear_all()
        self.status.config(
            text="متن و تحلیل پاک شدند.", fg=self.colors["muted"]
        )

    def save_transcript(self):
        """متن (و در صورت وجود، تحلیل) را در یک فایل UTF-8 ذخیره می‌کند."""
        if not transcript_segments:
            messagebox.showinfo("ذخیره", "متنی برای ذخیره وجود ندارد.")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text", "*.txt")],
            title="ذخیره متن"
        )
        if not path:
            return

        try:
            content = format_transcript_for_llm(with_time=True)
            analysis = self.analysis.get("1.0", "end").strip()
            if analysis and not analysis.startswith("در حال تحلیل"):
                content += "\n\n" + "=" * 40 + "\nتحلیل:\n" + analysis
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            self.status.config(text="متن ذخیره شد.", fg=self.colors["green"])
        except Exception as exc:
            messagebox.showerror("خطا در ذخیره", str(exc))

    # ========================================================
    # بستن برنامه
    # ========================================================

    def close(self):
        """بستن امن پنجره؛ ابتدا ترد‌های ضبط با false کردن ``running`` متوقف می‌شوند."""
        global running
        running = False
        try:
            self.translation_queue.put_nowait(None)
        except Exception:
            pass

        try:
            sd.stop()
        except Exception:
            pass

        self.root.destroy()


# ============================================================
# نقطه‌ی ورود
# ============================================================

def main():
    root = tk.Tk()
    ShenavaQwenGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
