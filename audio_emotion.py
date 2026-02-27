"""
audio_emotion.py  —  librosa signal analysis (no torch, no ML model)
─────────────────────────────────────────────────────────────────────
Lightweight audio emotion analysis using signal processing only.
No torch / transformers / speechbrain — fits in Render free tier (512MB).

Features extracted:
  rms            → loudness / emotional intensity
  mean_pitch_hz  → anxiety indicator (high pitch = anxious)
  pitch_var      → expressiveness / emotional variability
  zcr            → speech rate / agitation proxy
  silence_ratio  → hesitancy / fatigue indicator
  spec_centroid  → voice brightness (higher = anxiety, lower = sadness)

Emotion output:
  anxious / sad / frustrated / fatigued / neutral
  arousal / valence / dominance (circumplex model)
  stress_level   → medical composite 0.0–1.0
  medical_emotions → mapped to pipeline vocabulary
"""

import io
import time
import asyncio
import numpy as np
from typing import Dict

try:
    import librosa
    _LIBROSA_OK = True
except ImportError:
    _LIBROSA_OK = False

TARGET_SR    = 16000
TRIM_SECONDS = 3

print("ℹ️  Audio emotion: librosa signal analysis (no ML model needed)")

# ── No model download needed — sentinel so rest of code stays the same ──
_classifier = True   # truthy = "ready"


def load_model():
    """No-op: librosa needs no model download."""
    print("✅ Audio emotion ready instantly (librosa — no model download)")


# ─────────────────────────────────────────────────────────
# MAIN ANALYSIS  (sync — runs in thread pool)
# ─────────────────────────────────────────────────────────
def analyse_audio_sync(audio_bytes: bytes) -> Dict:
    """
    Full audio emotion pipeline using signal features.
    Returns same schema as the previous ML version so the rest of
    the pipeline (fuse_emotions, empathy prefix, TTS) works unchanged.
    """
    if not _LIBROSA_OK:
        return _neutral("librosa not installed")

    t_start = time.time()

    try:
        # ── Load audio ────────────────────────────────────────
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=TARGET_SR, mono=True)
        full_duration = len(y) / TARGET_SR

        # Trim to first TRIM_SECONDS — ~50% latency saving
        y = y[: int(TARGET_SR * TRIM_SECONDS)]

        if len(y) < 500:
            return _neutral("audio too short", full_duration)

        # ── Signal features ───────────────────────────────────
        rms           = float(np.sqrt(np.mean(y ** 2)))
        zcr           = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        silence_ratio = float(np.mean(np.abs(y) < 0.01))
        spec_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

        # Pitch — YIN algorithm
        f0     = librosa.yin(y, fmin=60, fmax=400, sr=sr)
        voiced = f0[f0 > 60]
        mean_pitch = float(np.mean(voiced)) if len(voiced) > 5 else 150.0
        pitch_var  = float(np.std(voiced))  if len(voiced) > 5 else 0.0

        # ── Normalise to 0–1 ──────────────────────────────────
        energy_n  = min(rms / 0.15,              1.0)   # 0.15 RMS ≈ loud speech
        pitch_n   = max(min((mean_pitch - 80) / 220, 1.0), 0.0)  # 80–300 Hz
        zcr_n     = min(zcr / 0.15,              1.0)
        silence_n = silence_ratio                        # already 0–1

        # ── Emotion scores ────────────────────────────────────
        # Anxious  : high pitch + high energy + high zcr
        anxious     = round(pitch_n * 0.45 + energy_n * 0.30 + zcr_n * 0.25, 3)

        # Sad      : low energy + low pitch + high silence
        sad         = round((1 - energy_n) * 0.40 + (1 - pitch_n) * 0.30 + silence_n * 0.30, 3)

        # Frustrated: high energy + low-mid pitch + high zcr
        frustrated  = round(energy_n * 0.50 + zcr_n * 0.30 + (1 - pitch_n) * 0.20, 3)

        # Fatigued : low energy + low zcr + high silence
        fatigued    = round((1 - energy_n) * 0.40 + (1 - zcr_n) * 0.30 + silence_n * 0.30, 3)

        # Neutral  : complement of strongest signal
        neutral     = round(max(1.0 - max(anxious, sad, frustrated, fatigued), 0.0), 3)

        scores = {
            "anxious":    anxious,
            "sad":        sad,
            "frustrated": frustrated,
            "fatigued":   fatigued,
            "neutral":    neutral,
        }

        top_emotion    = max(scores, key=scores.get)
        top_confidence = scores[top_emotion]

        # ── Arousal / Valence / Dominance ─────────────────────
        arousal   = round(min(energy_n * 0.6 + pitch_n * 0.4,         1.0), 3)
        valence   = round(max(1.0 - sad * 0.7  - anxious * 0.3,       0.0), 3)
        dominance = round(max(1.0 - silence_n  - (1 - energy_n) * 0.5, 0.0), 3)

        # ── Medical stress composite ──────────────────────────
        stress = round(min(
            arousal * 0.4 + (1 - valence) * 0.4 + (1 - dominance) * 0.2,
            1.0
        ), 3)

        # ── Medical emotion mapping ───────────────────────────
        medical = {}
        if anxious    > 0.30: medical["anxiety"]     = anxious
        if sad        > 0.30: medical["sadness"]      = sad
        if frustrated > 0.30: medical["frustration"]  = frustrated
        if fatigued   > 0.35: medical["fatigue"]      = fatigued
        if stress     > 0.60: medical["distress"]     = stress
        if not medical:       medical["neutral"]      = 1.0

        return {
            "class_probabilities": scores,
            "top_emotion":         top_emotion,
            "top_confidence":      round(top_confidence, 3),
            "arousal":             arousal,
            "valence":             valence,
            "dominance":           dominance,
            "stress_level":        stress,
            "medical_emotions":    medical,
            "signal_features": {
                "rms":           round(rms,           5),
                "zcr":           round(zcr,           4),
                "mean_pitch_hz": round(mean_pitch,    1),
                "pitch_var":     round(pitch_var,     1),
                "silence_ratio": round(silence_ratio, 3),
                "spec_centroid": round(spec_centroid, 1),
            },
            "duration_sec": round(full_duration, 2),
            "latency_ms":   round((time.time() - t_start) * 1000, 1),
        }

    except Exception as e:
        return _neutral(str(e))


def _neutral(error: str = "", duration: float = 0.0) -> Dict:
    result = {
        "class_probabilities": {"neutral": 1.0},
        "top_emotion":         "neutral",
        "top_confidence":      0.0,
        "arousal":             0.45,
        "valence":             0.55,
        "dominance":           0.50,
        "stress_level":        0.0,
        "medical_emotions":    {"neutral": 1.0},
        "signal_features":     {},
        "duration_sec":        duration,
        "latency_ms":          0.0,
    }
    if error:
        result["error"] = error
    return result


# ── Async wrapper — use this in FastAPI endpoints ──────────
async def analyse_audio(audio_bytes: bytes) -> Dict:
    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, analyse_audio_sync, audio_bytes)
    return result
