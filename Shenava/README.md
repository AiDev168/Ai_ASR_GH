---
license: apache-2.0
language: fa
library_name: sherpa-onnx
pipeline_tag: automatic-speech-recognition
base_model: Reza2kn/Shenava-Koochik-v1.0
base_model_relation: quantized
tags: [persian, farsi, asr, sherpa-onnx, onnx, ctc, fastconformer, on-device, offline, shenava]
---

# 🧰🎙️ Shenava Koochik v1.0 — sherpa-onnx

Offline Persian ASR bundle for [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx). Use this repository when you want sherpa-onnx’s common API across Python, C++, Android, iOS, WebAssembly, C#, Go, Swift, Kotlin, or Rust.

## ✨ At a glance | معرفی سریع

| | English | فارسی |
|---|---|---|
| 🎯 Role | Cross-platform offline ASR | تشخیص گفتار آفلاین چندسکویی |
| 📦 Bundle | ONNX model, tokens, Persian ITN | مدل ONNX، توکن‌ها و ITN فارسی |
| 🧩 Targets | Desktop, mobile, WebAssembly | دسکتاپ، موبایل و WebAssembly |
| 🎧 Audio | 16 kHz mono | صوت تک‌کانالهٔ ۱۶ کیلوهرتز |
| ⚠️ Key rule | Disable per-feature normalization | نرمال‌سازی per-feature را خاموش کنید |

- Canonical repository: [`Reza2kn/Shenava-Koochik-v1.0-sherpa-onnx`](https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0-sherpa-onnx)
- PersianML mirror: [`PersianML/Shenava-Koochik-v1.0-sherpa-onnx`](https://huggingface.co/PersianML/Shenava-Koochik-v1.0-sherpa-onnx)
- Parent model: [`Reza2kn/Shenava-Koochik-v1.0`](https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0)

## 📦 Files

- `model.onnx`: offline FastConformer CTC graph.
- `tokens.txt`: 1,025-token SentencePiece/CTC vocabulary.
- `persian_itn.py`: optional spoken-number to Persian-digit postprocessor.

## 🚀 Python example

```python
import numpy as np
import sherpa_onnx
import soundfile as sf
from persian_itn import itn

recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
    model="model.onnx",
    tokens="tokens.txt",
    num_threads=4,
)

audio, sample_rate = sf.read("speech.wav")
stream = recognizer.create_stream()
stream.accept_waveform(sample_rate, np.asarray(audio, dtype=np.float32))
recognizer.decode_stream(stream)
print(itn(stream.result.text))
```

## ⚠️ Important preprocessing rule

Use 16 kHz mono audio. The graph expects unnormalized NeMo log-mel features; its ONNX metadata deliberately leaves `normalize_type` empty. Enabling sherpa-onnx’s default per-feature normalization produces invalid output for this export.

This is an offline recognizer, not the cache-aware streaming graph. Numbers are emitted in spoken form unless `persian_itn.py` is applied.

## 🇮🇷 خلاصهٔ فارسی

این مخزن بستهٔ آفلاین sherpa-onnx برای تشخیص گفتار فارسی است. نرمال‌سازی per-feature را فعال نکنید؛ متادیتای مدل عمداً `normalize_type` را خالی گذاشته است. برای تبدیل عددهای گفتاری به رقم از `persian_itn.py` استفاده کنید.

## 🌌 Explore Shenava-1

[🧠 Koochik source](https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0) · [🌐 Browser ONNX](https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0-ONNX-fp16) · **🧰 sherpa-onnx** · [🍎 Core ML](https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0-CoreML-fp16) · [🦀 tract offline](https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0-tract-offline)

Apache-2.0.
