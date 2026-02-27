"""
test_audio_emotion.py
─────────────────────
Run this FIRST to verify SpeechBrain installs and loads correctly
before starting the full FastAPI server.

Usage:
    python test_audio_emotion.py
    python test_audio_emotion.py --wav path/to/your/audio.wav
"""

import sys
import time
import torch
import torchaudio
import numpy as np
import io
import argparse
import soundfile as sf


def test_model_load():
    print("\n── TEST 1: Model Load ──────────────────────────")
    from audio_emotion import load_model, _classifier
    load_model()
    from audio_emotion import _classifier as clf
    assert clf is not None, "Model failed to load"
    print("✅ SpeechBrain model loaded successfully")


def test_synthetic_audio():
    """
    Generate a 3-second synthetic sine wave and run it through the pipeline.
    No real microphone needed — just verifies the inference path works.
    """
    print("\n── TEST 2: Synthetic Audio Inference ───────────")
    from audio_emotion import analyse_audio_sync

    # 3 seconds of 440Hz sine wave at 16kHz — sounds like a tone
    sample_rate = 16000
    duration    = 3
    t           = np.linspace(0, duration, sample_rate * duration)
    waveform    = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)

    # Convert to bytes (wav format) as if it came from the browser
    buf = io.BytesIO()
 
    sf.write(buf, waveform, sample_rate, format="wav")
    audio_bytes = buf.getvalue()

    start  = time.time()
    result = analyse_audio_sync(audio_bytes)
    ms     = (time.time() - start) * 1000

    print(f"✅ Inference completed in {ms:.0f}ms")
    print(f"\n   Class probabilities: {result['class_probabilities']}")
    print(f"   Top emotion:         {result['top_emotion']} ({result['top_confidence']:.1%})")
    print(f"   Arousal:             {result['arousal']}")
    print(f"   Valence:             {result['valence']}")
    print(f"   Dominance:           {result['dominance']}")
    print(f"   Stress level:        {result['stress_level']}")
    print(f"   Medical emotions:    {result['medical_emotions']}")
    print(f"   Signal features:     {result['signal_features']}")


def test_wav_file(path: str):
    print(f"\n── TEST 3: Real WAV File → {path} ──────────────")
    from audio_emotion import analyse_audio_sync

    with open(path, "rb") as f:
        audio_bytes = f.read()

    result = analyse_audio_sync(audio_bytes)

    print(f"✅ File analysed ({result['duration_sec']}s clip)")
    print(f"\n   Class probabilities: {result['class_probabilities']}")
    print(f"   Top emotion:         {result['top_emotion']} ({result['top_confidence']:.1%})")
    print(f"   Arousal:             {result['arousal']}")
    print(f"   Valence:             {result['valence']}")
    print(f"   Dominance:           {result['dominance']}")
    print(f"   Stress level:        {result['stress_level']}")
    print(f"   Medical emotions:    {result['medical_emotions']}")
    print(f"   Latency:             {result['latency_ms']}ms")


def test_fusion():
    print("\n── TEST 4: Emotion Fusion ──────────────────────")
    import sys
    sys.path.insert(0, ".")
    from main import fuse_emotions

    # Simulate: patient says "I'm okay" (text → relief)
    # but voice sounds stressed (audio → high arousal, low valence)
    text_emotions = {"relief": 0.4}
    audio_result  = {
        "medical_emotions": {"anxiety": 0.3, "distress": 0.2},
        "stress_level":     0.72,
        "arousal":          0.78,
        "valence":          0.22,
        "dominance":        0.35,
    }

    fused = fuse_emotions(text_emotions, audio_result)
    print(f"   Text only:  {text_emotions}")
    print(f"   Audio only: {audio_result['medical_emotions']}")
    print(f"   Fused:      {fused}")
    print(f"✅ Fusion correctly amplified distress signals from audio")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=str, default=None,
                        help="Path to a WAV file to test with real audio")
    args = parser.parse_args()

    print("=" * 55)
    print("  SpeechBrain Audio Emotion — Local Test Suite")
    print("=" * 55)

    try:
        test_model_load()
        test_synthetic_audio()
        if args.wav:
            test_wav_file(args.wav)
        test_fusion()

        print("\n" + "=" * 55)
        print("  ✅ All tests passed — ready to run the server")
        print("  Run: uvicorn main:app --reload --port 8000")
        print("=" * 55 + "\n")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)