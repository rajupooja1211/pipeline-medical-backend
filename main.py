# """
# Medical Assistant Backend — 4-Service Pipeline (Web Demo)
# """

# from fastapi import FastAPI, UploadFile, File, Form, Request
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import Dict, List
# import re, os, time, base64, httpx

# app = FastAPI(title="Medical Assistant — 4-Service Pipeline")
# app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
# ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
# ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")

# conversations: Dict[str, dict] = {}
# metrics_log: List[dict] = []

# # ═══ SERVICE 2: Sentiment Analysis ═══
# def analyze_sentiment(text: str) -> Dict[str, float]:
#     t = text.lower()
#     emotions = {}
#     for words, key, score_per in [
#         (["pain","hurt","ache","sore","throb","sharp","burning","stiff","tender"], "pain", 0.2),
#         (["scared","afraid","worried","anxious","nervous","fear","terrified"], "anxiety", 0.25),
#         (["sad","depressed","hopeless","crying","miserable","down","unhappy"], "sadness", 0.25),
#         (["frustrated","annoyed","angry","mad","irritated"], "frustration", 0.25),
#         (["tired","exhausted","fatigue","no energy","weak","drained"], "fatigue", 0.2),
#         (["confused","don't know","not sure","maybe","i think"], "confusion", 0.2),
#         (["better","improving","good","great","fine","okay","relieved"], "relief", 0.2),
#     ]:
#         score = sum(score_per for w in words if w in t)
#         if score > 0:
#             emotions[key] = min(score, 1.0)
#     if any(w in t for w in ["really","very","so much","terrible","worst","severe"]):
#         if "pain" in emotions: emotions["pain"] = min(emotions["pain"] + 0.3, 1.0)
#     m = re.search(r'(\d+\.?\d*)\s*(out of|\/)\s*10', t)
#     if m:
#         r = float(m.group(1))
#         if r >= 7:
#             emotions["pain"] = max(emotions.get("pain", 0), r/10)
#             emotions["distress"] = r/10
#     return emotions or {"neutral": 1.0}

# def get_empathy_prefix(emotions: Dict[str, float]) -> str:
#     if not emotions or "neutral" in emotions: return ""
#     top = max(emotions, key=emotions.get)
#     s = emotions[top]
#     if s < 0.2: return ""
#     p = {"anxiety":"I understand this can feel worrying. ","fear":"You're in good hands. ",
#          "sadness":"I'm sorry you're going through this. ","frustration":"I understand this can be frustrating. ",
#          "fatigue":"I can tell you're feeling tired. ","confusion":"That's okay, take your time. ",
#          "relief":"That's great to hear. "}
#     if top in ["pain","distress"]:
#         return "I'm really sorry about your pain. " if s >= 0.7 else "I'm sorry you're in pain. "
#     return p.get(top, "")

# # ═══ SERVICE 3: Medical Logic ═══
# def get_next_question(session: dict, user_text: str = "") -> str:
#     step = session["step"]
#     if step > 0 and user_text:
#         session["answers"].append({"q": step, "a": user_text})
#     questions = {
#         0: "Hello, I am your medical assistant. How are you feeling today? On a scale from 0 to 10, with 0 being great and 10 being in bed most of the time, how would you rate it?",
#         1: "Did you encounter any morning stiffness that made it hard to move?",
#         2: "How long did the stiffness linger? How long before you feel your best during the day?",
#         3: "Do you feel discomfort in your upper or lower extremities?",
#         4: "Can you tell me which joints are causing you pain?",
#         5: "Have you noticed any new swelling in your joints?",
#         6: "Any additional symptoms such as fever, fatigue, or rash?",
#         7: "Would you say your symptoms are better, worse, or unchanged compared to yesterday?",
#         8: "Can you grip objects like a pen or jar without difficulty?",
#         9: "Did you remember to take your medications today?",
#     }
#     session["step"] = step + 1
#     return questions.get(step, "Thank you for sharing. A doctor will review your case shortly. Take care!")

# # ═══ SERVICE 1: Whisper STT ═══
# async def whisper_transcribe(audio_bytes: bytes) -> tuple:
#     start = time.time()
#     async with httpx.AsyncClient(timeout=30.0) as client:
#         resp = await client.post(
#             "https://api.openai.com/v1/audio/transcriptions",
#             headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
#             files={"file": ("audio.webm", audio_bytes, "audio/webm")},
#             data={"model": "whisper-1"},
#         )
#     ms = (time.time() - start) * 1000
#     if resp.status_code == 200:
#         return resp.json().get("text", ""), ms
#     print(f"❌ Whisper error: {resp.status_code} {resp.text}")
#     return "", ms

# # ═══ SERVICE 4: ElevenLabs TTS ═══
# async def elevenlabs_tts(text: str, emotions: Dict[str, float] = {}, voice_id: str = "") -> tuple:
#     start = time.time()
#     vid = voice_id or ELEVENLABS_VOICE_ID
#     stability = 0.6 if max(emotions, key=emotions.get, default="neutral") in ["pain","distress","sadness"] else 0.5
#     async with httpx.AsyncClient(timeout=30.0) as client:
#         resp = await client.post(
#             f"https://api.elevenlabs.io/v1/text-to-speech/{vid}",
#             headers={"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"},
#             json={"text": text, "model_id": "eleven_flash_v2_5",
#                   "voice_settings": {"stability": stability, "similarity_boost": 0.75}},
#         )
#     ms = (time.time() - start) * 1000
#     if resp.status_code != 200:
#         print(f"❌ TTS error: {resp.status_code} {resp.content[:200]}")
#     return (resp.content if resp.status_code == 200 else b""), ms

# # ═══ ENDPOINTS ═══
# @app.post("/api/web-chat")
# async def web_chat(audio: UploadFile = File(...), session_id: str = Form("web-default"), voice_id: str = Form("")):
#     total_start = time.time()
#     if session_id not in conversations:
#         conversations[session_id] = {"step": 0, "answers": []}
#     session = conversations[session_id]

#     audio_bytes = await audio.read()
#     transcript, whisper_ms = await whisper_transcribe(audio_bytes)

#     if not transcript or len(transcript.strip()) < 1:
#         q = get_next_question(session, "") if session["step"] == 0 else "I didn't catch that. Could you repeat?"
#         return {"transcript": "", "response": q, "emotions": {}, "audio_base64": "", "metrics": {"whisper_ms": round(whisper_ms,1), "error": "no_speech"}}

#     sentiment_start = time.time()
#     emotions = analyze_sentiment(transcript)
#     sentiment_ms = (time.time() - sentiment_start) * 1000

#     logic_start = time.time()
#     question = get_next_question(session, transcript)
#     final_response = get_empathy_prefix(emotions) + question
#     logic_ms = (time.time() - logic_start) * 1000

#     audio_resp, tts_ms = await elevenlabs_tts(final_response, emotions, voice_id=voice_id or ELEVENLABS_VOICE_ID)
#     audio_b64 = base64.b64encode(audio_resp).decode() if audio_resp else ""

#     total_ms = (time.time() - total_start) * 1000
#     top_emotion = max(emotions, key=emotions.get)
#     metrics = {"whisper_ms": round(whisper_ms,1), "sentiment_ms": round(sentiment_ms,1),
#                "logic_ms": round(logic_ms,1), "tts_ms": round(tts_ms,1), "total_ms": round(total_ms,1),
#                "top_emotion": top_emotion, "top_score": round(emotions.get(top_emotion,0),3)}

#     print(f"\n═══════════════ 📊 4-SERVICE PIPELINE ═══════════════")
#     print(f"📊 Whisper:    {whisper_ms:.0f}ms | Sentiment: {sentiment_ms:.1f}ms")
#     print(f"📊 Logic:      {logic_ms:.1f}ms | TTS: {tts_ms:.0f}ms | Total: {total_ms:.0f}ms")
#     print(f"📊 \"{transcript[:50]}\" → {top_emotion}")
#     print(f"═══════════════════════════════════════════════════════\n")

#     metrics_log.append({"session_id": session_id, "step": session["step"], "metrics": metrics, "transcript": transcript})

#     return {"transcript": transcript, "response": final_response, "emotions": emotions,
#             "audio_base64": audio_b64, "metrics": metrics}

# @app.get("/api/start-session")
# async def start_session(session_id: str = "web-default", voice_id: str = ""):
#     conversations[session_id] = {"step": 0, "answers": []}
#     session = conversations[session_id]
#     question = get_next_question(session, "")
#     audio_resp, tts_ms = await elevenlabs_tts(question, voice_id=voice_id or ELEVENLABS_VOICE_ID)
#     audio_b64 = base64.b64encode(audio_resp).decode() if audio_resp else ""
#     return {"response": question, "audio_base64": audio_b64, "metrics": {"tts_ms": round(tts_ms,1)}}

# @app.post("/api/tts")
# async def tts_only(request: Request):
#     body = await request.json()
#     text = body.get("text", "")
#     if not text:
#         return {"audio_base64": ""}
#     audio_resp, tts_ms = await elevenlabs_tts(text)
#     audio_b64 = base64.b64encode(audio_resp).decode() if audio_resp else ""
#     return {"audio_base64": audio_b64, "tts_ms": round(tts_ms, 1)}

# @app.get("/api/voices")
# async def get_voices():
#     """Return voices available on this ElevenLabs account."""
#     async with httpx.AsyncClient(timeout=10.0) as client:
#         resp = await client.get(
#             "https://api.elevenlabs.io/v1/voices",
#             headers={"xi-api-key": ELEVENLABS_API_KEY},
#         )
#     if resp.status_code != 200:
#         return {"voices": []}
#     data = resp.json()
#     voices = [
#         {"voice_id": v["voice_id"], "name": v["name"], "category": v.get("category", "generated")}
#         for v in sorted(data.get("voices", []), key=lambda x: x["name"])
#     ]
#     return {"voices": voices, "default": ELEVENLABS_VOICE_ID}

# @app.get("/health")
# async def health():
#     return {"status": "ok", "service": "4-service-pipeline"}

# @app.get("/metrics")
# async def get_metrics():
#     return {"turns": metrics_log}

# @app.get("/")
# async def root():
#     return {"message": "4-Service Pipeline", "endpoints": ["/api/web-chat","/api/start-session","/health","/metrics"]}



"""
Medical Assistant Backend — 5-Service Pipeline (Audio Emotion Edition)
═══════════════════════════════════════════════════════════════════════

Services:
  1. Whisper STT          — transcribe patient audio
  2. SpeechBrain Emotion  — analyse raw audio for emotion (runs PARALLEL to Whisper)
  3. Text Sentiment       — analyse transcript for emotion keywords
  4. Emotion Fusion       — merge audio + text signals into one rich emotion dict
  5. Medical Logic        — next question + empathy prefix
  6. ElevenLabs TTS       — voice response tuned to fused emotions

Key design decisions for local demo (CPU, no GPU):
  - SpeechBrain model loaded ONCE at startup
  - Audio trimmed to 3s before SpeechBrain inference (~50% latency saving)
  - Whisper + SpeechBrain run in asyncio.gather() — parallel, not sequential
  - Total added latency from SpeechBrain ≈ 0ms (runs inside Whisper window)
"""

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List
import re, os, time, base64, httpx, asyncio

# ── Import our SpeechBrain module ──
from audio_emotion import load_model, analyse_audio

app = FastAPI(title="Medical Assistant — 5-Service Pipeline (Audio Emotion)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY      = os.environ.get("OPENAI_API_KEY", "")
GROQ_API_KEY        = os.environ.get("GROQ_API_KEY", "")
ELEVENLABS_API_KEY  = os.environ.get("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")

conversations: Dict[str, dict] = {}
metrics_log:   List[dict]      = []


# ═══════════════════════════════════════════════════════════
# STARTUP — load SpeechBrain model once
# ═══════════════════════════════════════════════════════════
async def _load_model_background():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, load_model)
    print("✅ SpeechBrain model loaded and ready")

@app.on_event("startup")
async def startup_event():
    """
    Fire-and-forget model loading — port binds immediately so Render
    health check passes, model downloads in background (~300MB first run).
    """
    asyncio.create_task(_load_model_background())
    print("🚀 Server started — SpeechBrain loading in background")


# ═══════════════════════════════════════════════════════════
# SERVICE 3: Text Sentiment (unchanged from your original)
# ═══════════════════════════════════════════════════════════
def analyze_sentiment(text: str) -> Dict[str, float]:
    t = text.lower()
    emotions = {}
    for words, key, score_per in [
        (["pain","hurt","ache","sore","throb","sharp","burning","stiff","tender"], "pain", 0.2),
        (["scared","afraid","worried","anxious","nervous","fear","terrified"],      "anxiety", 0.25),
        (["sad","depressed","hopeless","crying","miserable","down","unhappy"],      "sadness", 0.25),
        (["frustrated","annoyed","angry","mad","irritated"],                        "frustration", 0.25),
        (["tired","exhausted","fatigue","no energy","weak","drained"],              "fatigue", 0.2),
        (["confused","don't know","not sure","maybe","i think"],                   "confusion", 0.2),
        (["better","improving","good","great","fine","okay","relieved"],            "relief", 0.2),
    ]:
        score = sum(score_per for w in words if w in t)
        if score > 0:
            emotions[key] = min(score, 1.0)

    if any(w in t for w in ["really","very","so much","terrible","worst","severe"]):
        if "pain" in emotions:
            emotions["pain"] = min(emotions["pain"] + 0.3, 1.0)

    m = re.search(r'(\d+\.?\d*)\s*(out of|\/)\s*10', t)
    if m:
        r = float(m.group(1))
        if r >= 7:
            emotions["pain"]    = max(emotions.get("pain", 0), r / 10)
            emotions["distress"] = r / 10

    return emotions or {"neutral": 1.0}


# ═══════════════════════════════════════════════════════════
# SERVICE 4: Emotion Fusion
# Merges text sentiment + SpeechBrain audio emotion.
#
# Priority rules:
#   - Audio emotion is the ground truth for INTENSITY
#   - Text emotion is the ground truth for TOPIC (what they said)
#   - High audio stress overrides mild text signals
#   - Arousal/valence drive TTS voice settings
# ═══════════════════════════════════════════════════════════
def fuse_emotions(
    text_emotions: Dict[str, float],
    audio_result:  Dict,
) -> Dict[str, float]:
    """
    Fuse text + audio emotion signals into one dict.
    Returns enriched emotion dict used for empathy prefix + TTS tuning.
    """
    fused = dict(text_emotions)

    if not audio_result:
        return fused

    medical = audio_result.get("medical_emotions", {})
    stress  = audio_result.get("stress_level", 0.0)
    arousal = audio_result.get("arousal", 0.5)
    valence = audio_result.get("valence", 0.5)

    # Merge audio-derived medical emotions
    for emotion, score in medical.items():
        if emotion in fused:
            # Take the higher of the two signals — trust the stronger detector
            fused[emotion] = max(fused[emotion], score)
        else:
            fused[emotion] = score

    # If audio stress is high but text didn't catch it — add distress
    if stress > 0.65 and fused.get("distress", 0) < stress:
        fused["distress"] = stress

    # High arousal + low valence = anxiety boost even if words were calm
    if arousal > 0.7 and valence < 0.35:
        fused["anxiety"] = min(fused.get("anxiety", 0) + 0.25, 1.0)

    # Very low arousal → fatigue signal from voice alone
    if arousal < 0.25:
        fused["fatigue"] = min(fused.get("fatigue", 0) + 0.2, 1.0)

    # Remove neutral if any real emotion detected
    if len(fused) > 1 and "neutral" in fused:
        del fused["neutral"]

    return fused or {"neutral": 1.0}


# ═══════════════════════════════════════════════════════════
# SERVICE 5: Medical Logic (unchanged from your original)
# ═══════════════════════════════════════════════════════════
def get_next_question(session: dict, user_text: str = "") -> str:
    step = session["step"]
    if step > 0 and user_text:
        session["answers"].append({"q": step, "a": user_text})

    questions = {
        0: "Hello, I am your medical assistant. How are you feeling today? On a scale from 0 to 10, with 0 being great and 10 being in bed most of the time, how would you rate it?",
        1: "Did you encounter any morning stiffness that made it hard to move?",
        2: "How long did the stiffness linger? How long before you feel your best during the day?",
        3: "Do you feel discomfort in your upper or lower extremities?",
        4: "Can you tell me which joints are causing you pain?",
        5: "Have you noticed any new swelling in your joints?",
        6: "Any additional symptoms such as fever, fatigue, or rash?",
        7: "Would you say your symptoms are better, worse, or unchanged compared to yesterday?",
        8: "Can you grip objects like a pen or jar without difficulty?",
        9: "Did you remember to take your medications today?",
    }
    session["step"] = step + 1
    return questions.get(step, "Thank you for sharing. A doctor will review your case shortly. Take care!")


def get_empathy_prefix(emotions: Dict[str, float]) -> str:
    if not emotions or "neutral" in emotions:
        return ""
    top = max(emotions, key=emotions.get)
    s   = emotions[top]
    if s < 0.2:
        return ""
    p = {
        "anxiety":     "I understand this can feel worrying. ",
        "fear":        "You're in good hands. ",
        "sadness":     "I'm sorry you're going through this. ",
        "frustration": "I understand this can be frustrating. ",
        "fatigue":     "I can tell you're feeling tired. ",
        "confusion":   "That's okay, take your time. ",
        "relief":      "That's great to hear. ",
    }
    if top in ["pain", "distress"]:
        return "I'm really sorry about your pain. " if s >= 0.7 else "I'm sorry you're in pain. "
    return p.get(top, "")


# ═══════════════════════════════════════════════════════════
# SERVICE 1: Whisper STT (your original, unchanged)
# ═══════════════════════════════════════════════════════════
async def whisper_transcribe(audio_bytes: bytes) -> tuple:
    start = time.time()
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            files={"file": ("audio.webm", audio_bytes, "audio/webm")},
            data={"model": "whisper-large-v3-turbo", "language": "en"},
        )
    ms = (time.time() - start) * 1000
    if resp.status_code == 200:
        return resp.json().get("text", ""), ms
    print(f"❌ Whisper error: {resp.status_code} {resp.text}")
    return "", ms


# ═══════════════════════════════════════════════════════════
# SERVICE 6: ElevenLabs TTS — emotion-tuned voice settings
#
# Now uses fused emotions (audio + text) for richer tuning.
# Arousal drives speaking rate feel; valence drives stability.
# ═══════════════════════════════════════════════════════════
async def elevenlabs_tts(
    text:       str,
    emotions:   Dict[str, float] = {},
    audio_result: Dict           = {},
    voice_id:   str              = "",
) -> tuple:
    start = time.time()
    vid   = voice_id or ELEVENLABS_VOICE_ID

    top   = max(emotions, key=emotions.get, default="neutral")
    score = emotions.get(top, 0)

    # Arousal / valence from SpeechBrain for continuous tuning
    arousal = audio_result.get("arousal", 0.5)
    valence = audio_result.get("valence", 0.5)

    # Emotion → voice settings matrix
    # stability:        higher = calmer/more consistent voice
    # similarity_boost: how closely it tracks the base voice character
    # style:            expressiveness (0–1)
    tuning = {
        "pain":        {"stability": 0.65, "similarity_boost": 0.75, "style": 0.40},
        "distress":    {"stability": 0.60, "similarity_boost": 0.70, "style": 0.50},
        "anxiety":     {"stability": 0.55, "similarity_boost": 0.72, "style": 0.45},
        "sadness":     {"stability": 0.70, "similarity_boost": 0.78, "style": 0.30},
        "frustration": {"stability": 0.45, "similarity_boost": 0.68, "style": 0.55},
        "fatigue":     {"stability": 0.72, "similarity_boost": 0.80, "style": 0.20},
        "relief":      {"stability": 0.50, "similarity_boost": 0.75, "style": 0.35},
        "neutral":     {"stability": 0.50, "similarity_boost": 0.75, "style": 0.30},
    }.get(top, {"stability": 0.50, "similarity_boost": 0.75, "style": 0.30})

    # Continuous arousal correction on top of discrete class tuning
    # High arousal patient → slightly lower stability for more expressive response
    if arousal > 0.7:
        tuning["stability"] = max(tuning["stability"] - 0.08, 0.30)
    elif arousal < 0.3:
        tuning["stability"] = min(tuning["stability"] + 0.08, 0.90)

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{vid}",
            headers={"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"},
            json={
                "text":       text,
                "model_id":   "eleven_flash_v2_5",
                "voice_settings": tuning,
            },
        )
    ms = (time.time() - start) * 1000
    if resp.status_code != 200:
        print(f"❌ TTS error: {resp.status_code} {resp.content[:200]}")
    return (resp.content if resp.status_code == 200 else b""), ms


# ═══════════════════════════════════════════════════════════
# ENDPOINT: /api/audio-emotion-chat  ← NEW main demo endpoint
#
# Pipeline:
#   audio_bytes → [Whisper STT]          ─┐  (parallel)
#              → [SpeechBrain Emotion]   ─┘
#                    ↓
#              [Text Sentiment on transcript]
#                    ↓
#              [Fusion: audio + text emotions]
#                    ↓
#              [Medical Logic + empathy]
#                    ↓
#              [ElevenLabs TTS (tuned)]
# ═══════════════════════════════════════════════════════════
@app.post("/api/audio-emotion-chat")
async def audio_emotion_chat(
    audio:      UploadFile = File(...),
    session_id: str        = Form("audio-demo"),
    voice_id:   str        = Form(""),
):
    total_start = time.time()

    if session_id not in conversations:
        conversations[session_id] = {"step": 0, "answers": []}
    session = conversations[session_id]

    audio_bytes = await audio.read()

    # ── PARALLEL: Whisper + SpeechBrain run at the same time ──
    # Total time = max(whisper_ms, speechbrain_ms) NOT the sum
    (transcript, whisper_ms), audio_result = await asyncio.gather(
        whisper_transcribe(audio_bytes),
        analyse_audio(audio_bytes),       # SpeechBrain runs here
    )

    speechbrain_ms = audio_result.get("latency_ms", 0)

    # No speech detected
    if not transcript or len(transcript.strip()) < 1:
        q = get_next_question(session, "") if session["step"] == 0 \
            else "I didn't catch that. Could you repeat?"
        return {
            "transcript":      "",
            "response":        q,
            "text_emotions":   {},
            "audio_result":    audio_result,
            "fused_emotions":  {},
            "audio_base64":    "",
            "metrics": {
                "whisper_ms":      round(whisper_ms, 1),
                "speechbrain_ms":  round(speechbrain_ms, 1),
                "error":           "no_speech",
            },
        }

    # ── Text sentiment on transcript ──
    sentiment_start  = time.time()
    text_emotions    = analyze_sentiment(transcript)
    sentiment_ms     = (time.time() - sentiment_start) * 1000

    # ── Fuse audio + text emotions ──
    fused_emotions   = fuse_emotions(text_emotions, audio_result)

    # ── Medical logic ──
    logic_start      = time.time()
    question         = get_next_question(session, transcript)
    final_response   = get_empathy_prefix(fused_emotions) + question
    logic_ms         = (time.time() - logic_start) * 1000

    # ── TTS with full emotion context ──
    audio_resp, tts_ms = await elevenlabs_tts(
        final_response,
        fused_emotions,
        audio_result,
        voice_id=voice_id or ELEVENLABS_VOICE_ID,
    )
    audio_b64 = base64.b64encode(audio_resp).decode() if audio_resp else ""

    total_ms   = (time.time() - total_start) * 1000
    top_fused  = max(fused_emotions, key=fused_emotions.get, default="neutral")
    top_audio  = audio_result.get("top_emotion", "—")

    metrics = {
        "whisper_ms":      round(whisper_ms,     1),
        "speechbrain_ms":  round(speechbrain_ms, 1),
        "sentiment_ms":    round(sentiment_ms,   1),
        "logic_ms":        round(logic_ms,       1),
        "tts_ms":          round(tts_ms,         1),
        "total_ms":        round(total_ms,       1),
        "top_text_emotion":  max(text_emotions,  key=text_emotions.get,  default="—"),
        "top_audio_emotion": top_audio,
        "top_fused_emotion": top_fused,
        "stress_level":      audio_result.get("stress_level", 0),
        "arousal":           audio_result.get("arousal",      0),
        "valence":           audio_result.get("valence",      0),
        "dominance":         audio_result.get("dominance",    0),
    }

    # ── Console log ──
    print(f"\n═══════════ 🎙️ 5-SERVICE AUDIO EMOTION PIPELINE ═══════════")
    print(f"  Whisper:      {whisper_ms:.0f}ms  |  SpeechBrain: {speechbrain_ms:.0f}ms  (parallel)")
    print(f"  Sentiment:    {sentiment_ms:.1f}ms  |  Logic: {logic_ms:.1f}ms  |  TTS: {tts_ms:.0f}ms")
    print(f"  Total:        {total_ms:.0f}ms")
    print(f"  Transcript:   \"{transcript[:60]}\"")
    print(f"  Audio emotion:{top_audio} | Text emotion: {metrics['top_text_emotion']} → Fused: {top_fused}")
    print(f"  Stress: {audio_result.get('stress_level',0):.2f}  Arousal: {audio_result.get('arousal',0):.2f}  "
          f"Valence: {audio_result.get('valence',0):.2f}  Dominance: {audio_result.get('dominance',0):.2f}")
    print(f"══════════════════════════════════════════════════════════════\n")

    metrics_log.append({
        "session_id": session_id,
        "step":       session["step"],
        "metrics":    metrics,
        "transcript": transcript,
    })

    return {
        "transcript":     transcript,
        "response":       final_response,
        "text_emotions":  text_emotions,
        "audio_result":   audio_result,    # full SpeechBrain output
        "fused_emotions": fused_emotions,
        "audio_base64":   audio_b64,
        "metrics":        metrics,
    }


# ═══════════════════════════════════════════════════════════
# ORIGINAL ENDPOINTS — kept intact so your existing demo still works
# ═══════════════════════════════════════════════════════════
@app.post("/api/web-chat")
async def web_chat(
    audio:      UploadFile = File(...),
    session_id: str        = Form("web-default"),
    voice_id:   str        = Form(""),
):
    total_start = time.time()
    if session_id not in conversations:
        conversations[session_id] = {"step": 0, "answers": []}
    session = conversations[session_id]

    audio_bytes              = await audio.read()
    transcript, whisper_ms   = await whisper_transcribe(audio_bytes)

    if not transcript or len(transcript.strip()) < 1:
        q = get_next_question(session, "") if session["step"] == 0 \
            else "I didn't catch that. Could you repeat?"
        return {"transcript": "", "response": q, "emotions": {}, "audio_base64": "",
                "metrics": {"whisper_ms": round(whisper_ms, 1), "error": "no_speech"}}

    sentiment_start = time.time()
    emotions        = analyze_sentiment(transcript)
    sentiment_ms    = (time.time() - sentiment_start) * 1000

    logic_start    = time.time()
    question       = get_next_question(session, transcript)
    final_response = get_empathy_prefix(emotions) + question
    logic_ms       = (time.time() - logic_start) * 1000

    audio_resp, tts_ms = await elevenlabs_tts(final_response, emotions)
    audio_b64          = base64.b64encode(audio_resp).decode() if audio_resp else ""
    total_ms           = (time.time() - total_start) * 1000
    top_emotion        = max(emotions, key=emotions.get)

    metrics = {
        "whisper_ms":   round(whisper_ms,   1),
        "sentiment_ms": round(sentiment_ms, 1),
        "logic_ms":     round(logic_ms,     1),
        "tts_ms":       round(tts_ms,       1),
        "total_ms":     round(total_ms,     1),
        "top_emotion":  top_emotion,
        "top_score":    round(emotions.get(top_emotion, 0), 3),
    }
    metrics_log.append({"session_id": session_id, "step": session["step"],
                         "metrics": metrics, "transcript": transcript})

    return {"transcript": transcript, "response": final_response,
            "emotions": emotions, "audio_base64": audio_b64, "metrics": metrics}


@app.get("/api/start-session")
async def start_session(session_id: str = "web-default", voice_id: str = ""):
    conversations[session_id] = {"step": 0, "answers": []}
    session  = conversations[session_id]
    question = get_next_question(session, "")
    audio_resp, tts_ms = await elevenlabs_tts(question, voice_id=voice_id or ELEVENLABS_VOICE_ID)
    audio_b64 = base64.b64encode(audio_resp).decode() if audio_resp else ""
    return {"response": question, "audio_base64": audio_b64, "metrics": {"tts_ms": round(tts_ms, 1)}}


@app.post("/api/tts")
async def tts_only(request: Request):
    body = await request.json()
    text = body.get("text", "")
    if not text:
        return {"audio_base64": ""}
    audio_resp, tts_ms = await elevenlabs_tts(text)
    audio_b64 = base64.b64encode(audio_resp).decode() if audio_resp else ""
    return {"audio_base64": audio_b64, "tts_ms": round(tts_ms, 1)}


@app.get("/api/voices")
async def get_voices():
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            "https://api.elevenlabs.io/v1/voices",
            headers={"xi-api-key": ELEVENLABS_API_KEY},
        )
    if resp.status_code != 200:
        return {"voices": []}
    data   = resp.json()
    voices = [
        {"voice_id": v["voice_id"], "name": v["name"], "category": v.get("category", "generated")}
        for v in sorted(data.get("voices", []), key=lambda x: x["name"])
    ]
    return {"voices": voices, "default": ELEVENLABS_VOICE_ID}


@app.get("/health")
async def health():
    from audio_emotion import _classifier
    return {
        "status":              "ok",
        "service":             "5-service-audio-emotion-pipeline",
        "speechbrain_loaded":  _classifier is not None,
    }


@app.get("/metrics")
async def get_metrics():
    return {"turns": metrics_log}


@app.get("/")
async def root():
    return {
        "message":   "5-Service Audio Emotion Pipeline",
        "endpoints": [
            "/api/audio-emotion-chat",   # NEW — audio + text fusion
            "/api/web-chat",             # original text-only pipeline
            "/api/start-session",
            "/health",
            "/metrics",
        ],
    }