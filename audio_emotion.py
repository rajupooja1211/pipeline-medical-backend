"""
audio_emotion.py
────────────────
Audio Emotion Analysis using HuggingFace transformers pipeline.
No SpeechBrain version dependency issues.

Model: ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition
Output:
  - emotion classes     → sad, angry, happy, neutral, fearful, disgusted, surprised
  - arousal estimate    → derived from class + signal energy (0.0–1.0)
  - valence estimate    → derived from class + signal features (0.0–1.0)
  - dominance estimate  → derived from class + speaking rate (0.0–1.0)
  - stress_level        → composite medical stress score (0.0–1.0)

CPU latency optimizations:
  1. Model loaded ONCE at startup — never per request
  2. Audio trimmed to first 3s — emotion established in opening utterance
  3. Runs in asyncio thread pool — does not block FastAPI event loop
"""

import io
import time
import asyncio
import torch
import torchaudio
import numpy as np
from typing import Dict, Tuple
from transformers import pipeline as hf_pipeline

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
TARGET_SR     = 16000
TRIM_SECONDS  = 3
DEVICE        = 0 if torch.cuda.is_available() else -1   # 0=GPU, -1=CPU for transformers
DEVICE_LABEL  = "GPU" if DEVICE == 0 else "CPU"
MODEL_SOURCE  = "superb/wav2vec2-base-superb-er"

print(f"ℹ️  Audio emotion will run on: {DEVICE_LABEL}")

# ─────────────────────────────────────────────────────────
# MODEL SINGLETON
# ─────────────────────────────────────────────────────────
_classifier = None


def load_model():
    global _classifier
    if _classifier is not None:
        return

    print("\n📦 Loading emotion model (one-time)...")
    t = time.time()

    _classifier = hf_pipeline(
        "audio-classification",
        model=MODEL_SOURCE,
        device=DEVICE,
    )

    print(f"✅ Emotion model ready in {time.time() - t:.1f}s on {DEVICE_LABEL}\n")


# ─────────────────────────────────────────────────────────
# AUDIO PREPROCESSING
# ─────────────────────────────────────────────────────────
def _preprocess(audio_bytes: bytes) -> Tuple[torch.Tensor, float]:
    """
    Load raw audio bytes → 16kHz mono tensor trimmed to TRIM_SECONDS.
    Handles webm, wav, ogg, mp4 — whatever the browser sends.
    """
    waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))

    # Stereo → mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 16kHz if needed
    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)

    full_duration = waveform.shape[1] / TARGET_SR

    # Trim to first TRIM_SECONDS — ~50% latency saving on long clips
    max_samples = int(TARGET_SR * TRIM_SECONDS)
    waveform    = waveform[:, :max_samples]

    return waveform.squeeze(0), full_duration   # shape: [samples]


# ─────────────────────────────────────────────────────────
# SIGNAL FEATURES (no model — pure math, ~1ms)
# ─────────────────────────────────────────────────────────
def _extract_signal_features(waveform: torch.Tensor) -> Dict[str, float]:
    y = waveform.numpy()

    rms           = float(np.sqrt(np.mean(y ** 2)))
    zcr           = float(np.mean(np.abs(np.diff(np.sign(y)))) / 2)
    peak          = float(np.max(np.abs(y)))
    silence_ratio = float(np.mean(np.abs(y) < 0.01))

    return {
        "rms":           round(rms,           5),
        "zcr":           round(zcr,           4),
        "peak":          round(peak,          4),
        "silence_ratio": round(silence_ratio, 3),
    }


# ─────────────────────────────────────────────────────────
# AROUSAL / VALENCE / DOMINANCE
# Based on the circumplex model of affect.
# These map each emotion class to a 3D position in affect space.
# ─────────────────────────────────────────────────────────
_AVD_MAP = {
    #              arousal  valence  dominance
    "angry":      (0.85,    0.15,    0.80),
    "fearful":    (0.80,    0.10,    0.15),
    "disgusted":  (0.60,    0.10,    0.50),
    "sad":        (0.20,    0.20,    0.25),
    "surprised":  (0.70,    0.60,    0.40),
    "happy":      (0.75,    0.85,    0.70),
    "neutral":    (0.45,    0.55,    0.50),
    # fallback keys in case model uses different label names
    "anger":      (0.85,    0.15,    0.80),
    "fear":       (0.80,    0.10,    0.15),
    "disgust":    (0.60,    0.10,    0.50),
    "happiness":  (0.75,    0.85,    0.70),
    "surprise":   (0.70,    0.60,    0.40),
    # superb/wav2vec2-base-superb-er abbreviated labels
    "neu":        (0.45,    0.55,    0.50),
    "hap":        (0.75,    0.85,    0.70),
    "ang":        (0.85,    0.15,    0.80),
    "exc":        (0.90,    0.70,    0.75),   # excited
}


def _derive_avd(class_probs: Dict[str, float], signal: Dict[str, float]) -> Dict[str, float]:
    arousal = valence = dominance = 0.0

    for cls, prob in class_probs.items():
        a, v, d = _AVD_MAP.get(cls, (0.45, 0.55, 0.50))
        arousal   += prob * a
        valence   += prob * v
        dominance += prob * d

    # Signal energy correction
    rms = signal["rms"]
    if rms > 0.08:
        arousal = min(arousal + 0.15, 1.0)
    elif rms < 0.01:
        arousal = max(arousal - 0.15, 0.0)

    # High silence → lower dominance (hesitant/tired speaker)
    if signal["silence_ratio"] > 0.4:
        dominance = max(dominance - 0.1, 0.0)

    return {
        "arousal":   round(arousal,   3),
        "valence":   round(valence,   3),
        "dominance": round(dominance, 3),
    }


# ─────────────────────────────────────────────────────────
# STRESS LEVEL — medical composite score
# High stress = high arousal + low valence + low dominance
# ─────────────────────────────────────────────────────────
def _compute_stress(avd: Dict[str, float], class_probs: Dict[str, float]) -> float:
    stress = (
        (avd["arousal"]           * 0.4) +
        ((1 - avd["valence"])     * 0.4) +
        ((1 - avd["dominance"])   * 0.2)
    )

    # Boost for high-distress classes
    distress_classes = ["angry", "anger", "fearful", "fear", "sad"]
    if any(class_probs.get(c, 0) > 0.5 for c in distress_classes):
        stress = min(stress + 0.1, 1.0)

    return round(stress, 3)


# ─────────────────────────────────────────────────────────
# MEDICAL EMOTION MAPPING
# Maps model output labels → your existing pipeline vocabulary
# ─────────────────────────────────────────────────────────
_MEDICAL_MAP = {
    "angry":     "frustration",
    "anger":     "frustration",
    "sad":       "sadness",
    "fearful":   "anxiety",
    "fear":      "anxiety",
    "disgusted": "frustration",
    "disgust":   "frustration",
    "happy":     "relief",
    "happiness": "relief",
    "surprised": "confusion",
    "surprise":  "confusion",
    "neutral":   "neutral",
    # superb/wav2vec2-base-superb-er abbreviated labels
    "neu":       "neutral",
    "hap":       "relief",
    "ang":       "frustration",
    "exc":       "relief",    # excited → positive/relief
}


def _to_medical_emotions(
    class_probs: Dict[str, float],
    avd:         Dict[str, float],
    stress:      float,
) -> Dict[str, float]:
    medical = {}

    for cls, prob in class_probs.items():
        med_key = _MEDICAL_MAP.get(cls, cls)
        # If two classes map to same medical key, take the higher score
        medical[med_key] = max(medical.get(med_key, 0), round(prob, 3))

    # Derive fatigue from low arousal + high silence
    if avd["arousal"] < 0.3:
        medical["fatigue"] = round((0.3 - avd["arousal"]) * 2, 3)

    # Inject distress if composite stress is high
    if stress > 0.6:
        medical["distress"] = round(stress, 3)

    return medical


# ─────────────────────────────────────────────────────────
# MAIN ANALYSIS FUNCTION (sync — runs in thread pool)
# ─────────────────────────────────────────────────────────
def analyse_audio_sync(audio_bytes: bytes) -> Dict:
    """
    Full audio emotion pipeline.
    Returns:
      class_probabilities  — raw model output
      top_emotion          — highest confidence class
      top_confidence       — 0.0–1.0
      arousal/valence/dominance — continuous affect scores
      stress_level         — medical composite 0.0–1.0
      medical_emotions     — mapped to your pipeline vocabulary
      signal_features      — rms, zcr, peak, silence_ratio
      duration_sec         — original clip length
      latency_ms           — wall-clock inference time
    """
    if _classifier is None:
        # Model still loading in background — return neutral fallback so the
        # rest of the pipeline (Whisper + text sentiment + TTS) still works.
        return {
            "class_probabilities": {"neutral": 1.0},
            "top_emotion":         "neutral",
            "top_confidence":      0.0,
            "arousal":             0.45,
            "valence":             0.55,
            "dominance":           0.50,
            "stress_level":        0.0,
            "medical_emotions":    {"neutral": 1.0},
            "signal_features":     {},
            "duration_sec":        0.0,
            "latency_ms":          0.0,
            "model_loading":       True,
        }

    t_start = time.time()

    # 1. Preprocess audio
    waveform, duration = _preprocess(audio_bytes)

    # 2. Fast signal features
    signal = _extract_signal_features(waveform)

    # 3. Model inference
    audio_array = waveform.numpy()
    results     = _classifier({"array": audio_array, "sampling_rate": TARGET_SR})
    # results = [{"label": "sad", "score": 0.72}, {"label": "angry", "score": 0.18}, ...]

    class_probs = {
        r["label"].lower(): round(float(r["score"]), 4)
        for r in results
    }

    top_emotion    = max(class_probs, key=class_probs.get)
    top_confidence = class_probs[top_emotion]

    # 4. Derive continuous affect scores
    avd = _derive_avd(class_probs, signal)

    # 5. Medical stress composite
    stress = _compute_stress(avd, class_probs)

    # 6. Map to medical vocabulary
    medical_emotions = _to_medical_emotions(class_probs, avd, stress)

    return {
        "class_probabilities": class_probs,
        "top_emotion":         top_emotion,
        "top_confidence":      round(top_confidence, 3),
        "arousal":             avd["arousal"],
        "valence":             avd["valence"],
        "dominance":           avd["dominance"],
        "stress_level":        stress,
        "medical_emotions":    medical_emotions,
        "signal_features":     signal,
        "duration_sec":        round(duration, 2),
        "latency_ms":          round((time.time() - t_start) * 1000, 1),
    }


# ─────────────────────────────────────────────────────────
# ASYNC WRAPPER — use this in FastAPI endpoints
# Runs in thread pool so it never blocks the event loop
# while Whisper runs in parallel via asyncio.gather()
# ─────────────────────────────────────────────────────────
async def analyse_audio(audio_bytes: bytes) -> Dict:
    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, analyse_audio_sync, audio_bytes)
    return result