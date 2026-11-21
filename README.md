# Shenava AI — Persian speech desktop app

برنامهٔ رومیزی برای تبدیل گفتار فارسی به متن، تفکیک گوینده، تحلیل متن و ترجمه. رابط کاربری با Tkinter نوشته شده است. قابلیت‌های اصلی از `sherpa-onnx` و `sounddevice` استفاده می‌کنند؛ Whisper، pyannote و Ollama اختیاری‌اند.

## پیش‌نیازها

- Python 3.12 یا 3.13 روی Windows، macOS یا Linux؛ روی Linux بستهٔ سیستم `python3-tk` و کتابخانهٔ PortAudio نیز لازم است.
- میکروفون برای حالت زنده، و محیط گرافیکی برای اجرای رابط کاربری.
- حداقل مدل‌های Shenava و Silero VAD، که به دلیل حجم بالا در Git نگهداری نمی‌شوند.

## نصب و اجرا

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python asrr6.py
```

در Windows می‌توان به جای `python` از `py -3.13` استفاده کرد. اگر قابلیت‌های اختیاری را می‌خواهید، `requirements-optional.txt` را نصب کنید و برای pyannote، PyTorch یا Ollama دستور نصب متناسب با سیستم خود را دنبال کنید.

## فایل‌های مدل

فایل‌های زیر را در پوشهٔ `Shenava/` کنار کد قرار دهید:

| فایل | کاربرد | منبع |
| --- | --- | --- |
| `model.onnx` | تشخیص گفتار Shenava | [بستهٔ رسمی Shenava Koochik](https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0-sherpa-onnx) |
| `tokens.txt` | واژگان مدل | همراه کد در این مخزن |
| `silero_vad.onnx` | تشخیص بخش‌های گفتار | [مدل Silero در sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx/releases) |

مدل‌های تفکیک گوینده را می‌توان از داخل برنامه دانلود کرد؛ مسیرشان `Shenava/speaker_models/` است. برای Whisper، پوشهٔ مدل باید حاوی `model.bin` باشد. مسیر پیش‌فرض آن `Shenava/models--deepdml--faster-whisper-large-v3-turbo-ct2/` است و با `WHISPER_MODEL_PATH` قابل تغییر است. مسیر اصلی مدل‌ها با `SHENAVA_DIR` قابل تنظیم است. توکن Hugging Face برای pyannote را از طریق `HF_TOKEN` یا فایل محلی `Shenava/hf_token.txt` بدهید؛ آن فایل در Git نادیده گرفته می‌شود.

## آزمون و CI

```bash
python -m unittest discover -s tests -v
```

GitHub Actions این بررسی‌ها را روی Windows، macOS و Linux با Python 3.12 و 3.13 اجرا می‌کند، وابستگی‌های اصلی را نصب می‌کند و import برنامه را بررسی می‌کند. این مراحل بدون دانلود مدل انجام می‌شوند. این پروژه برنامهٔ رومیزی است و مرحلهٔ استقرار سرور ندارد؛ هر push یک نسخهٔ قابل بررسی از کد و مستندات را در GitHub نگه می‌دارد. آزمون عملی صدا، میکروفون و رابط گرافیکی نیازمند محیط محلی با مدل‌ها و وابستگی‌های نصب‌شده است.

## سیاست مخزن

فقط کد، مستندات، فایل واژگان و تنظیمات CI در Git قرار می‌گیرند. وزن مدل‌ها، فایل‌های صوتی، کش، توکن و خروجی‌های محلی در `.gitignore` هستند. مجوز فایل‌های مدل و کدهای همراهشان را در منبع اصلی بررسی کنید؛ مجوز بستهٔ Shenava در [Shenava/README.md](Shenava/README.md) توضیح داده شده است.
